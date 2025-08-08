# Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting

## Introduction
**Mesh-adsorbed Gaussian Splatting (MaGS)** is a novel framework for reconstructing and simulating **dynamic 3D objects**.  
It combines the rendering flexibility of **3D Gaussians** with the spatial coherence of **meshes**, enabling both high-quality rendering and realistic deformation in dynamic scenes.

📄 More details: [Project Page](https://wcwac.github.io/MaGS-page/) | [Paper (arXiv)](https://arxiv.org/abs/2406.01593/)

---

## Installation

> **Prerequisites**  
> Before installation, please ensure that:
> - **CUDA Toolkit** is installed and matches your GPU & driver version.  
> - **PyTorch** is installed with CUDA support.

```bash
# Clone the main repository
git clone https://github.com/wcwac/MaGS.git

# Clone required submodules
git clone --recursive https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization

# Install dependencies
pip install -r requirements.txt
pip install -e submodules/simple-knn
pip install -e submodules/diff-gaussian-rasterization
````

---

## Dataset Preparation

### 1. D-NeRF Dataset

1. **Download the dataset**

   * Original dataset: [D-NeRF (official)](https://github.com/albertpumarola/D-NeRF)
   * Fixed version: [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)

2. **Download the meshes archive**

   * Get **`D-NeRF_meshes.7z`** from the [release page](https://github.com/wcwac/MaGS/releases/tag/v0.0.1).

3. **Organize and extract**

   * Put both the image data and `D-NeRF_meshes.7z` in the **same** directory.
   * Extract both archives. You should get a structure like:

     ```
     bouncingballs/
     ├── train/
     │   ├── r_000.png
     │   └── r_001.png
     ├── train_meshes/
     │   ├── r_000.ply
     │   └── r_001.ply
     ├── test/
     │   ├── r_000.png
     │   └── r_001.png
     └── test_meshes/
         ├── r_000.ply
         └── r_001.ply
     ```

> The `.ply` files let you reproduce the paper’s results immediately.
> Scripts for generating meshes from scratch (and for additional scenes) will be released soon.

---

### 2. DG-Mesh Dataset

The steps are similar to D-NeRF, with just a few differences:

1. **Download the dataset**

   * Dataset: [DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh)

2. **Download the meshes archive**

   * Get **`DG-Mesh_meshes.7z`** from the [release page](https://github.com/wcwac/MaGS/releases/tag/v0.0.1).

3. **Organize and extract**

   * Same procedure as in the D-NeRF case — place both archives in the same directory and extract.

---

## Running the Code

```bash
# Example: D-NeRF Jumping Jacks
python main.py config/3dgs.yaml,config/dnerf/jumpingjacks.yaml
```

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{mags2024,
  title={MaGS: Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting},
  author={Shaojie Ma and Yawei Luo and Wei Yang and Yi Yang},
  year={2024},
  eprint={2406.01593},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2406.01593}
}
```
