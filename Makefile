PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
JP_INDEX = https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
export PATH := /usr/local/cuda/bin:$(PATH)


.DEFAULT_GOAL := help
.PHONY: help detector install-showcase install-service lint setup cuda dataset train export engine model demo-video demo showcase showcase-reset real-footage real-demo zones run test bench clean

help:             ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install-showcase: ## start the customer showcase on every boot (systemd, asks for the sudo password)
	sudo bash deploy/install.sh --showcase

install-service:  ## install the production service (edit /etc/freshwatch/freshwatch.yaml, then start it)
	sudo bash deploy/install.sh

lint:             ## ruff static checks
	$(PY) -m ruff check .

setup:            ## venv (sees system TensorRT/OpenCV) + Jetson CUDA PyTorch + deps
	python3 -m venv --system-site-packages .venv
	$(PIP) install --no-deps --index-url $(JP_INDEX) torch==2.8.0 torchvision==0.23.0
	$(PIP) install -r requirements.txt

cuda:             ## build libfreshcuda.so (sm_87)
	cmake -S cuda -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$$(nproc)

dataset:          ## download + prepare the CC-BY-4.0 AgML fresh/rotten dataset
	bash training/download_dataset.sh data/raw
	$(PY) training/prepare_dataset.py --raw data/raw --out data/processed

train:            ## fine-tune MobileNetV3 on the Jetson (CUDA + AMP)
	$(PY) training/train.py --data data/processed --out models

export:           ## PyTorch -> ONNX
	cd training && ../$(PY) export_onnx.py --ckpt ../models/freshness_best.pt --out ../models/freshness.onnx

engine:           ## TensorRT FP16 engine — must be built on the target Jetson
	bash training/build_engine.sh models/freshness.onnx models/freshness_fp16.engine

model: train export engine  ## train + export + engine

# Produce detector for cameras with `detect: true`: YOLOX-S (Apache-2.0, COCO), official release ONNX
YOLOX_URL := https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
detector:         ## download YOLOX-S (36 MB) and build its TensorRT FP16 engine on this Jetson
	mkdir -p models/detector
	test -f models/detector/yolox_s.onnx || curl -fL -o models/detector/yolox_s.onnx $(YOLOX_URL)
	/usr/src/tensorrt/bin/trtexec --onnx=models/detector/yolox_s.onnx \
	  --saveEngine=models/detector/yolox_s_fp16.engine --fp16 --memPoolSize=workspace:1024 --skipInference \
	  2>&1 | tail -n 3

demo-video:       ## build the simulated decay video + config/demo.yaml
	$(PY) tools/make_demo_video.py --data data/processed --out data/demo --config-out config/demo.yaml

demo: cuda        ## simulated decay scenario only
	$(PY) -m freshwatch --config config/demo.yaml

# Real market footage (free Pexels clips, see data/real/SOURCES.md)
PEXELS := https://videos.pexels.com/video-files
real-footage:     ## download the 4 free Pexels market clips (7.4 MB)
	mkdir -p data/real
	cd data/real && curl -fsSL -o tagaytay_stall.mp4 $(PEXELS)/27558366/12168193_1280_720_24fps.mp4 \
	  && curl -fsSL -o indoor_veg_market.mp4 $(PEXELS)/36017942/15273756_800_360_30fps.mp4 \
	  && curl -fsSL -o tropical_fruit_stand.mp4 $(PEXELS)/26221938/11940290_1280_720_25fps.mp4 \
	  && curl -fsSL -o veg_stall_closeup.mp4 $(PEXELS)/4366190/4366190-hd_1280_720_30fps.mp4

showcase: cuda    ## customer demo: decay simulation + real market cameras on one dashboard
	@test -f data/demo/shelf.mp4 || $(MAKE) demo-video
	@test -f data/real/tagaytay_stall.mp4 || $(MAKE) real-footage
	$(PY) -m freshwatch --config config/showcase.yaml

showcase-reset:   ## clear the showcase's alerts and history (keeps camera settings)
	rm -rf data/showcase/freshwatch.db data/showcase/freshwatch.db-* data/showcase/snapshots

real-demo: cuda   ## real market cameras only
	$(PY) -m freshwatch --config config/real.yaml

# Draw configured zones on a frame from each camera: make zones CONFIG=config/freshwatch.yaml
CONFIG ?= config/real.yaml
zones:            ## draw configured zones on a frame from each camera (CONFIG=...)
	$(PY) tools/zone_preview.py --config $(CONFIG) --out data/zones --grid

run:              ## production config (config/freshwatch.yaml)
	$(PY) -m freshwatch --config config/freshwatch.yaml

test:             ## unit + CUDA + end-to-end tests
	$(PY) -m pytest -q tests

bench:            ## accuracy + latency benchmark of the production path
	$(PY) tools/benchmark.py --engine models/freshness_fp16.engine --labels models/labels.json

clean:            ## remove build output and caches
	rm -rf build .pytest_cache
