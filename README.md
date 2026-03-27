<h1 align="center">RPG360: Robust 360 Depth Estimation with Perspective Foundation Models and Graph Optimization
</h1>

<!-- Arxiv Link, Project Link -->

<p align="center">
  <a href="https://arxiv.org/abs/2509.23991"><img src="https://img.shields.io/badge/arXiv-2502.20685-b31b1b.svg" alt="arXiv"></a>
  <a href="https://jdk9405.github.io/RPG360/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen" alt="Project Page"></a>
</p>

<p align="center">
  <video width="100%" autoplay loop muted playsinline>
    <source src="asset/mp3d.mp4" type="video/mp4">
  </video>
</p>

## ✨ Abstract
The increasing use of 360° images across various domains has emphasized the need for robust depth estimation techniques tailored for omnidirectional images. However, obtaining large-scale labeled datasets for 360° depth estimation remains a significant challenge. In this paper, we propose RPG360, a training-free 360° monocular depth estimation method that leverages perspective foundation models. Our approach converts 360° images into six-face cubemap representations, where a perspective foundation model is employed to estimate depth and surface normals. To address depth scale inconsistencies across different faces of the cubemap, we introduce a novel depth scale alignment technique using graph-based optimization, which parameterizes the predicted depth and normal maps while incorporating an additional per-face scale parameter. This optimization ensures depth scale consistency across the six-face cubemap while preserving 3D structural integrity. Furthermore, as foundation models exhibit inherent robustness in zero-shot settings, our method achieves superior performance across diverse datasets, including Matterport3D, Stanford2D3D, and 360Loc. We also demonstrate the versatility of our depth estimation approach by validating its benefits in downstream tasks such as feature matching 3.2 - 5.4% and Structure from Motion 0.2 - 9.7% in AUC@5.

## 🕹 Inference
First, you need to modify the config file.
#### Step1
```
python scripts/step1_initial.py
```

#### Step2
```
python scripts/step2_refine.py
```

## 📚 BibTex
```bibtex
@article{jung2025rpg360,
  title={RPG360: Robust 360 Depth Estimation with Perspective Foundation Models and Graph Optimization},
  author={Jung, Dongki and Choi, Jaehoon and Lee, Yonghan and Manocha, Dinesh},
  journal={arXiv preprint arXiv:2509.23991},
  year={2025}
}
```