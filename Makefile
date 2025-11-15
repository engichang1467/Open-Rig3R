setup-env:
	sh ops/set_up_env.sh

download-wayve101:
	mkdir -p data/wayve_scenes_101
	bash ops/download_wayve101.sh

install:
	uv pip install --index-strategy unsafe-best-match -r requirements.txt