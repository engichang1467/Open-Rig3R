# Rig3R: Training Scale and Multitask Loss

This document provides a detailed guide on the full-scale training configuration and the multitask loss function for the Rig3R model, based on the original published research.

## 1. Full-Scale Training Configuration

To achieve state-of-the-art results, the original Rig3R model was trained at a specific, large scale. The following parameters were used for the full training run.

| Training Parameter      | Value                     |
| :---------------------- | :------------------------ |
| **Total Training Steps**  | **250k steps**            |
| **Batch Size**          | 128                       |
| **Frames per Sample**   | 24                        |
| **Image Size**          | 512 x 512 (with padding)  |
| **Optimizer**           | AdamW                     |
| **Learning Rate**       | 0.0001 (1e-4)             |
| **Scheduler**           | Cosine annealing          |
| **Dropout on Metadata** | 0.5 (50% probability)     |
| **Hardware Used**       | 128 H100 GPUs             |
| **Training Duration**   | ~5 days                   |

During training, the process utilized 24-frame samples. Data augmentations included random per-frame color jitter, Gaussian blur, and centered aspect-ratio crops to simulate variations in focal length and image shape. The input sequences were also randomly shuffled to vary the reference frame and promote generalization.

## 2. Multitask Loss Function

Rig3R is trained using a multitask loss that combines objectives from its three prediction heads: the pointmap head, the pose raymap head, and the rig raymap head.

The total loss is defined as:
$$L_{total} = L_{pmap} + \lambda_p L_{p\_rmap} + \lambda_r L_{r\_rmap}$$

In this equation:
- $L_{pmap}$ is the **Pointmap loss**.
- $L_{p\_rmap}$ is the **Pose Raymap loss**.
- $L_{r\_rmap}$ is the **Rig Raymap loss**.
- $\lambda_p$ and $\lambda_r$ are weighting terms that balance the contribution of each loss component.

**Note:** The exact numerical values for the weighting terms $\lambda_p$ and $\lambda_r$ are not specified in the source material, but their implementation is critical for balancing the learning objectives.

### Pointmap Loss ($L_{pmap}$)

$L_{pmap}$ is a confidence-weighted regression objective, calculated with scale-normalized ground truth.

$$L_{pmap} = \sum_{i \in D_v} C_v^i \left\lVert X_v^i - \frac{1}{\bar{z}} \bar{X}_v^i \right\rVert - \alpha \log C_v^i$$

Where:
- $X_v^i$ is the predicted 3D point at pixel $i$.
- $\bar{X}_v^i$ is the ground truth 3D point.
- $C_v^i$ is the predicted confidence, used to weight the regression term.
- $\alpha$ is the weight of the regularization term penalizing confidence.
- $\bar{z}$ is the average scene depth used for scale normalization.

### Raymap Loss ($L_{rmap}$)

The raymap loss is used for both pose and rig raymaps. It includes terms for both ray directions and camera centers.

$$L_{rmap} = \sum_{h, w} \left\lVert r_{v,h,w} - \bar{r}_{v,h,w} \right\rVert + \beta \left\lVert c_v - \frac{1}{\bar{z}} \bar{c}_v \right\rVert$$

Where:
- $r_{v,h,w}$ is the predicted unit ray direction at pixel $(h, w)$.
- $\bar{r}_{v,h,w}$ is the ground-truth ray direction.
- $c_v$ is the predicted camera center for frame $v$.
- $\bar{c}_v$ is the ground-truth camera center.
- $\beta$ weights the center loss term.
- $\bar{z}$ is the average scene depth for scale normalization.

### Analogy for Multitask Loss

You can think of the Multitask Loss as a student preparing for three different subjects (Pointmaps, Pose Raymaps, and Rig Raymaps) on a single test. The overall final grade ($L_{total}$) depends on performance in all three. The weights ($\lambda_p$ and $\lambda_r$) are like **specific multipliers** applied to the grades of the raymap subjects. By setting these weights correctly, the researchers ensure the model dedicates the right amount of effort to learning geometry ($L_{pmap}$) versus learning the camera structure and position ($L_{p\_rmap}$ and $L_{r\_rmap}$).