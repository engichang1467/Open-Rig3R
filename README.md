# 🛰️ Rig3R: Rig-Aware Conditioning for Learned 3D Reconstruction

**Rig3R** is a transformer-based model for multiview 3D reconstruction and camera pose estimation that incorporates **rig-aware conditioning**.
Unlike prior models (e.g., DUSt3R, Fast3R), Rig3R learns to leverage **rig metadata** (camera ID, timestamp, and rig pose) when available, and can **infer rig structure** when not — enabling robust 3D reconstruction across unstructured and rig-based image sets.

This repository is a **reimplementation** of the original [Rig3R paper (Li et al., 2025)](https://arxiv.org/abs/2506.02265), built for clarity, reproducibility, and extensibility.

---

## 🚀 Key Features

* **Rig-aware transformer architecture** using ViT-Large encoder-decoder.
* **Joint prediction** of:
  * Pointmaps (dense 3D coordinates),
  * Pose raymaps (global frame),
  * Rig raymaps (rig-relative frame).
* **Rig discovery**: infers rig calibration from unordered image collections. (In progress)
* **Closed-form camera pose recovery** from raymaps. 
* **Multi-task learning** with dropout-based metadata conditioning.

---

## 🧠 Model Overview

![Rig3R Model Architecture](img/rig3r_architecture.png)

---

## ⚙️ Dependencies

### Environment Set Up

Set up environment with [`uv`](https://docs.astral.sh/uv/)

```bash
make setup-env
```

### Activate the Environment

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
make install
```

---

## 🧩 Training

To train Rig3R from scratch:

```bash
python scripts/train.py --config configs/train.yaml
```

Recommended configuration values (from paper):

* Batch size: 128
* Frames per sample: 24
* Image size: 512×512
* Optimizer: AdamW (lr=1e-4)
* Scheduler: Cosine annealing
* Dropout on metadata: 0.5

---

## 🧪 Evaluation

Evaluate a pretrained Rig3R model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/rig3r_calib.pt
```

Metrics:

* **RRA / RTA @5°/15°/30°**
* **mAA (mean angular accuracy)**
* **Chamfer distance**
* **Rig discovery accuracy (Hungarian assignment)**

---

## 🔍 Rig Discovery (Working in progress)

Run unsupervised rig calibration discovery:

```bash
python scripts/infer_rig_discovery.py --data data/sample_inputs/
```

Outputs:

* Clustered rig raymaps
* Reconstructed 3D pointclouds
* Estimated rig configurations (extrinsics)

---

## 🧰 Citation

If you use this reimplementation, please cite the original paper:

```bibtex
@article{li2025rig3r,
  title={Rig3R: Rig-Aware Conditioning for Learned 3D Reconstruction},
  author={Li, Samuel and Kachana, Pujith and Chidananda, Prajwal and Nair, Saurabh and Furukawa, Yasutaka and Brown, Matthew},
  journal={arXiv preprint arXiv:2506.02265},
  year={2025}
}
```
