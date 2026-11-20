// freshcuda — CUDA kernels for SEGI FreshWatch. See include/freshcuda.h.
#include "freshcuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

namespace {

thread_local char g_err[512] = "";

int record(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        snprintf(g_err, sizeof(g_err), "%s: %s (%d)", what, cudaGetErrorString(e), (int)e);
        return (int)e;
    }
    return 0;
}

int invalid(const char* what) {
    snprintf(g_err, sizeof(g_err), "invalid argument: %s", what);
    return (int)cudaErrorInvalidValue;
}

#define FC_CHECK(call) do { int _r = record((call), #call); if (_r) return _r; } while (0)

// Kernel-argument structs are passed by value (constant bank), so concurrent
// launches from several camera threads never race on shared ROI state.
struct RoiBatch {
    FcRoi r[FC_MAX_ROIS];
};

struct NormParams {
    float mean[3];
    float inv_std[3];
};

__device__ __forceinline__ float load_bgr(const uint8_t* img, int pitch, int x, int y, int c) {
    return (float)img[y * pitch + x * 3 + c];
}

// One thread per output pixel; blockIdx.y selects the ROI. Bilinear sampling
// with half-pixel centres (matches cv2.resize INTER_LINEAR / torchvision).
__global__ void preprocess_kernel(const uint8_t* __restrict__ img, int width, int height,
                                  int pitch, RoiBatch rois, int out_size, int letterbox,
                                  NormParams np, float* __restrict__ out) {
    const int n = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = out_size * out_size;
    if (idx >= plane) return;

    const int ox = idx % out_size;
    const int oy = idx / out_size;
    const FcRoi roi = rois.r[n];

    float sx = (float)roi.w / out_size;
    float sy = (float)roi.h / out_size;
    float offx = 0.f, offy = 0.f;
    bool pad = false;
    const bool keep_bgr = letterbox & FC_PRE_KEEP_BGR;  // detectors such as YOLOX take BGR
    if (letterbox & FC_PRE_LETTERBOX) {
        const float s = fmaxf(sx, sy);
        sx = sy = s;
        offx = (out_size - roi.w / s) * 0.5f;
        offy = (out_size - roi.h / s) * 0.5f;
        pad = ox < offx || oy < offy || ox >= out_size - offx || oy >= out_size - offy;
    }

    float rgb[3];
    if (pad) {
        rgb[0] = rgb[1] = rgb[2] = 114.f;
    } else {
        float fx = (ox - offx + 0.5f) * sx - 0.5f + roi.x;
        float fy = (oy - offy + 0.5f) * sy - 0.5f + roi.y;
        fx = fminf(fmaxf(fx, (float)roi.x), (float)(roi.x + roi.w - 1));
        fy = fminf(fmaxf(fy, (float)roi.y), (float)(roi.y + roi.h - 1));
        const int x0 = (int)fx, y0 = (int)fy;
        const int x1 = min(x0 + 1, min(roi.x + roi.w - 1, width - 1));
        const int y1 = min(y0 + 1, min(roi.y + roi.h - 1, height - 1));
        const float ax = fx - x0, ay = fy - y0;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float top = load_bgr(img, pitch, x0, y0, c) * (1.f - ax) + load_bgr(img, pitch, x1, y0, c) * ax;
            const float bot = load_bgr(img, pitch, x0, y1, c) * (1.f - ax) + load_bgr(img, pitch, x1, y1, c) * ax;
            rgb[keep_bgr ? c : 2 - c] = top * (1.f - ay) + bot * ay;   // BGR -> RGB unless keep_bgr
        }
    }

    float* dst = out + (size_t)n * 3 * plane;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        dst[c * plane + idx] = (rgb[c] * (1.f / 255.f) - np.mean[c]) * np.inv_std[c];
    }
}

// RGB in [0,1] -> H in degrees [0,360), S,V in [0,1].
__device__ __forceinline__ void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v) {
    const float mx = fmaxf(r, fmaxf(g, b));
    const float mn = fminf(r, fminf(g, b));
    const float d = mx - mn;
    v = mx;
    s = mx > 1e-6f ? d / mx : 0.f;
    if (d < 1e-6f) { h = 0.f; return; }
    if (mx == r)      h = 60.f * fmodf((g - b) / d, 6.f);
    else if (mx == g) h = 60.f * ((b - r) / d + 2.f);
    else              h = 60.f * ((r - g) / d + 4.f);
    if (h < 0.f) h += 360.f;
}

constexpr int kStatBlock = 16;

// 2D tile per block, blockIdx.z selects the ROI. Per-block shared-memory
// counters are reduced first, then folded into global memory with one atomic
// per counter per block.
__global__ void color_stats_kernel(const uint8_t* __restrict__ img, int pitch, RoiBatch rois,
                                   int step, FcColorParams p, FcColorStats* __restrict__ stats) {
    __shared__ unsigned int s_cnt[5];
    __shared__ float s_sum[2];
    __shared__ unsigned int s_hist[FC_HUE_BINS];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < 5) s_cnt[tid] = 0;
    if (tid < 2) s_sum[tid] = 0.f;
    if (tid < FC_HUE_BINS) s_hist[tid] = 0;
    __syncthreads();

    const FcRoi roi = rois.r[blockIdx.z];
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int px = roi.x + gx * step;
    const int py = roi.y + gy * step;

    float sat = 0.f, val = 0.f;
    if (px < roi.x + roi.w && py < roi.y + roi.h) {
        const uint8_t* q = img + py * pitch + px * 3;
        float h, s, v;
        rgb_to_hsv(q[2] * (1.f / 255.f), q[1] * (1.f / 255.f), q[0] * (1.f / 255.f), h, s, v);
        atomicAdd(&s_cnt[0], 1u);
        if (v <= p.dark_val_max) atomicAdd(&s_cnt[3], 1u);
        if (s <= p.mould_sat_max && v >= p.mould_val_min) atomicAdd(&s_cnt[4], 1u);
        if (s >= p.fg_sat_min && v >= p.fg_val_min) {
            atomicAdd(&s_cnt[1], 1u);
            sat = s; val = v;
            atomicAdd(&s_hist[min((int)(h * (FC_HUE_BINS / 360.f)), FC_HUE_BINS - 1)], 1u);
            if (h >= p.brown_hue_lo && h <= p.brown_hue_hi && s >= p.brown_sat_min &&
                v >= p.brown_val_lo && v <= p.brown_val_hi)
                atomicAdd(&s_cnt[2], 1u);
        }
    }

    // Warp-level reduction of the float sums before touching shared memory.
    for (int off = 16; off > 0; off >>= 1) {
        sat += __shfl_down_sync(0xffffffffu, sat, off);
        val += __shfl_down_sync(0xffffffffu, val, off);
    }
    if ((tid & 31) == 0) {
        atomicAdd(&s_sum[0], sat);
        atomicAdd(&s_sum[1], val);
    }
    __syncthreads();

    FcColorStats* out = stats + blockIdx.z;
    if (tid == 0) {
        if (s_cnt[0]) atomicAdd(&out->total, s_cnt[0]);
        if (s_cnt[1]) atomicAdd(&out->foreground, s_cnt[1]);
        if (s_cnt[2]) atomicAdd(&out->brown, s_cnt[2]);
        if (s_cnt[3]) atomicAdd(&out->dark, s_cnt[3]);
        if (s_cnt[4]) atomicAdd(&out->mould, s_cnt[4]);
        atomicAdd(&out->sum_sat, s_sum[0]);
        atomicAdd(&out->sum_val, s_sum[1]);
    }
    if (tid < FC_HUE_BINS && s_hist[tid]) atomicAdd(&out->hue_hist[tid], s_hist[tid]);
}

int fill_rois(RoiBatch& b, const FcRoi* rois, int n, int width, int height, int* max_w, int* max_h) {
    if (!rois || n <= 0 || n > FC_MAX_ROIS) return invalid("n_rois must be in [1, FC_MAX_ROIS]");
    *max_w = *max_h = 0;
    for (int i = 0; i < n; ++i) {
        FcRoi r = rois[i];
        if (r.x < 0 || r.y < 0 || r.w <= 0 || r.h <= 0 || r.x + r.w > width || r.y + r.h > height)
            return invalid("roi outside frame");
        b.r[i] = r;
        if (r.w > *max_w) *max_w = r.w;
        if (r.h > *max_h) *max_h = r.h;
    }
    return 0;
}

}  // namespace

extern "C" {

const char* fc_last_error(void) { return g_err; }

int fc_init(int device) {
    FC_CHECK(cudaSetDevice(device));
    FC_CHECK(cudaFree(nullptr));  // force context creation
    return 0;
}

int fc_device_info(char* name, size_t name_len, int* sm_major, int* sm_minor, size_t* total_mem) {
    int dev = 0;
    FC_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    FC_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (name && name_len) { strncpy(name, prop.name, name_len - 1); name[name_len - 1] = 0; }
    if (sm_major) *sm_major = prop.major;
    if (sm_minor) *sm_minor = prop.minor;
    if (total_mem) *total_mem = prop.totalGlobalMem;
    return 0;
}

int fc_malloc(void** ptr, size_t bytes) { FC_CHECK(cudaMalloc(ptr, bytes)); return 0; }
int fc_free(void* ptr) { FC_CHECK(cudaFree(ptr)); return 0; }
int fc_host_alloc(void** ptr, size_t bytes) { FC_CHECK(cudaHostAlloc(ptr, bytes, cudaHostAllocDefault)); return 0; }
int fc_host_free(void* ptr) { FC_CHECK(cudaFreeHost(ptr)); return 0; }

int fc_memset_async(void* ptr, int value, size_t bytes, void* stream) {
    FC_CHECK(cudaMemsetAsync(ptr, value, bytes, (cudaStream_t)stream));
    return 0;
}
int fc_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    FC_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
    return 0;
}
int fc_memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream) {
    FC_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
    return 0;
}

int fc_stream_create(void** stream) {
    FC_CHECK(cudaStreamCreateWithFlags((cudaStream_t*)stream, cudaStreamNonBlocking));
    return 0;
}
int fc_stream_destroy(void* stream) { FC_CHECK(cudaStreamDestroy((cudaStream_t)stream)); return 0; }
int fc_stream_sync(void* stream) { FC_CHECK(cudaStreamSynchronize((cudaStream_t)stream)); return 0; }

int fc_preprocess_rois(const uint8_t* d_bgr, int width, int height, int pitch,
                       const FcRoi* rois, int n_rois, int out_size, int letterbox,
                       const float mean[3], const float stdv[3], float* d_out, void* stream) {
    if (!d_bgr || !d_out || out_size <= 0 || pitch < width * 3) return invalid("preprocess buffers");
    RoiBatch b;
    int mw, mh;
    if (int r = fill_rois(b, rois, n_rois, width, height, &mw, &mh)) return r;
    NormParams np;
    for (int c = 0; c < 3; ++c) {
        if (stdv[c] <= 0.f) return invalid("std must be > 0");
        np.mean[c] = mean[c];
        np.inv_std[c] = 1.f / stdv[c];
    }
    const int threads = 256;
    dim3 grid((out_size * out_size + threads - 1) / threads, n_rois);
    preprocess_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(d_bgr, width, height, pitch, b,
                                                                  out_size, letterbox, np, d_out);
    FC_CHECK(cudaGetLastError());
    return 0;
}

int fc_color_stats(const uint8_t* d_bgr, int width, int height, int pitch,
                   const FcRoi* rois, int n_rois, int sample_step,
                   const FcColorParams* params, FcColorStats* d_stats, void* stream) {
    if (!d_bgr || !d_stats || !params || sample_step < 1) return invalid("color_stats buffers");
    RoiBatch b;
    int mw, mh;
    if (int r = fill_rois(b, rois, n_rois, width, height, &mw, &mh)) return r;
    FC_CHECK(cudaMemsetAsync(d_stats, 0, sizeof(FcColorStats) * n_rois, (cudaStream_t)stream));
    const int sw = (mw + sample_step - 1) / sample_step;
    const int sh = (mh + sample_step - 1) / sample_step;
    dim3 block(kStatBlock, kStatBlock);
    dim3 grid((sw + kStatBlock - 1) / kStatBlock, (sh + kStatBlock - 1) / kStatBlock, n_rois);
    color_stats_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(d_bgr, pitch, b, sample_step,
                                                                 *params, d_stats);
    FC_CHECK(cudaGetLastError());
    return 0;
}

}  // extern "C"
