/*
 * freshcuda — CUDA kernels for SEGI FreshWatch.
 *
 * A small C ABI (loaded from Python via ctypes) that keeps the hot path on the
 * Jetson GPU:
 *   1. fc_preprocess_rois  — batched ROI crop + bilinear resize + BGR->RGB +
 *                            mean/std normalisation, written straight into the
 *                            TensorRT input binding (NCHW float32).
 *   2. fc_color_stats      — per-ROI HSV spoilage cues (browning, dark/rot
 *                            spots, saturation, hue histogram) using shared
 *                            memory reductions.
 *   3. memory / stream helpers so Python never needs pycuda/cupy.
 *
 * All functions return 0 on success, a non-zero cudaError_t otherwise; call
 * fc_last_error() for a message. Functions are thread-safe; use one stream per
 * camera worker.
 */
#ifndef FRESHCUDA_H
#define FRESHCUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FC_MAX_ROIS 32
#define FC_HUE_BINS 18

typedef struct {
    int32_t x, y, w, h;
} FcRoi;

/* Layout must match freshwatch/cuda_ops.py::ColorStats. */
typedef struct {
    uint32_t total;       /* sampled pixels                                   */
    uint32_t foreground;  /* pixels that look like produce (not bin/shadow)   */
    uint32_t brown;       /* oxidised / bruised tones                         */
    uint32_t dark;        /* very dark patches: rot, mould, deep bruising     */
    uint32_t mould;       /* low-saturation, bright-grey/white fuzz           */
    float    sum_sat;     /* over foreground                                  */
    float    sum_val;     /* over foreground                                  */
    uint32_t hue_hist[FC_HUE_BINS]; /* over foreground, 20 deg bins          */
} FcColorStats;

typedef struct {
    float brown_hue_lo, brown_hue_hi;   /* degrees                            */
    float brown_sat_min;                /* 0..1                               */
    float brown_val_lo, brown_val_hi;   /* 0..1                               */
    float dark_val_max;                 /* 0..1                               */
    float mould_sat_max, mould_val_min; /* 0..1                               */
    float fg_sat_min, fg_val_min;       /* foreground gate                    */
} FcColorParams;

const char* fc_last_error(void);
int  fc_init(int device);
int  fc_device_info(char* name, size_t name_len, int* sm_major, int* sm_minor,
                    size_t* total_mem);

/* memory */
int  fc_malloc(void** ptr, size_t bytes);
int  fc_free(void* ptr);
int  fc_host_alloc(void** ptr, size_t bytes);   /* pinned */
int  fc_host_free(void* ptr);
int  fc_memset_async(void* ptr, int value, size_t bytes, void* stream);
int  fc_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream);
int  fc_memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream);

/* streams / events */
int  fc_stream_create(void** stream);
int  fc_stream_destroy(void* stream);
int  fc_stream_sync(void* stream);

/*
 * Crop each ROI from a BGR8 interleaved frame and resize to out_size x out_size
 * (aspect-preserving letterbox if letterbox != 0), producing normalised RGB
 * planar float32 at d_out[n][3][out_size][out_size].
 */
/* fc_preprocess_rois `letterbox` flags (0 = plain resize; 1 keeps the old "letterbox" meaning). */
#define FC_PRE_LETTERBOX 1  /* keep aspect ratio, centre, pad with 114 */
#define FC_PRE_KEEP_BGR  2  /* write channels in BGR order instead of RGB */

int  fc_preprocess_rois(const uint8_t* d_bgr, int width, int height, int pitch,
                        const FcRoi* rois, int n_rois, int out_size,
                        int letterbox, const float mean[3], const float stdv[3],
                        float* d_out, void* stream);

/*
 * Accumulate FcColorStats per ROI into d_stats[n_rois] (zeroed by this call).
 * sample_step >= 1 subsamples pixels in both axes.
 */
int  fc_color_stats(const uint8_t* d_bgr, int width, int height, int pitch,
                    const FcRoi* rois, int n_rois, int sample_step,
                    const FcColorParams* params, FcColorStats* d_stats,
                    void* stream);

#ifdef __cplusplus
}
#endif
#endif /* FRESHCUDA_H */
