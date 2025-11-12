setup-env:
	sh ops/setup_env.sh

install-all:
	make install-pytorch
	make install-dependencies

install-pytorch:
	uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

download-wayve101:
	mkdir -p data/wayve_scenes_101
	bash ops/download_wayve101.sh

install-dependencies:
	uv pip install -r requirements.txt