setup-env:
	sh ops/set_up_env.sh

setup-train:
	sh ops/get_pretrained_dust3r.sh

download-wayve101:
	mkdir -p data/wayve_scenes_101
	bash ops/download_wayve101.sh

install:
	uv pip install --index-strategy unsafe-best-match -r requirements.txt

train:
	python scripts/train.py --config configs/train.yaml

evaluate:
	python scripts/evaluate.py --config configs/evaluate.yaml
