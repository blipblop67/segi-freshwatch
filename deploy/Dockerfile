# FreshWatch runtime image for JetPack 6.x (L4T r36). Build ON the Jetson:
#   docker build -f deploy/Dockerfile -t segi/freshwatch:1.2.0 .
# The TensorRT engine is device-specific: mount models/ built on the target
# (or run `make engine` inside the container once).
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv python3-opencv cmake \
        gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad python3-gi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/freshwatch
COPY requirements.txt .
RUN pip3 install "numpy<2" fastapi "uvicorn[standard]" "pydantic>=2" pyyaml prometheus-client

COPY cuda/ cuda/
RUN cmake -S cuda -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"$(nproc)"

COPY freshwatch/ freshwatch/
COPY config/ config/

RUN useradd --system --uid 990 freshwatch && install -d -o freshwatch /var/lib/freshwatch
USER freshwatch
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python3 -c \
  "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"
ENTRYPOINT ["python3", "-m", "freshwatch"]
CMD ["--config", "/etc/freshwatch/freshwatch.yaml"]
