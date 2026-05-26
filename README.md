# EG2026 Code Release

This repository is organized around the run configurations defined in `.vscode/launch.json`.

- `generate_synthetic_mar.py`: generate a MAR synthetic dataset
- `initialize_pcd.py`: create the initial Gaussian point cloud
- `train.py`: run training

Unless noted otherwise, all commands below are intended to be run from the `src` directory.

## 1. Local Setup

### Requirements

- NVIDIA GPU
- CUDA 11.6-compatible driver
- Conda or Miniconda
- Python 3.9

### Create the Environment

```bash
cd src
conda env create -f environment.yml
conda activate r2gs_bhc
```

`environment.yml` includes PyTorch, the CUDA toolkit, and editable installs for the CUDA extensions.

`simple-knn` is not vendored in this release. It is downloaded directly from its Git URL during environment creation.

`glm` is also not included in this release. When `xray_gaussian_rasterization_voxelization` is built, its `setup.py` automatically downloads the `GLM 1.0.2` archive. In network-restricted environments, you can provide an alternative archive URL through the `R2_GAUSSIAN_GLM_URL` environment variable.

### Rebuild the CUDA Extension

If you need to rebuild the extension, use the commands below.

If `third_party/glm` is empty during this process, the required version will be downloaded automatically again.

Windows:

```powershell
cd src
build_rasterizer.bat
```

Linux:

```bash
cd src
bash build_rasterizer.sh
```

### Additional Dependency for `initialize_pcd.py`

`initialize_pcd.py` uses `tigre`. If you want to run point cloud initialization, `tigre` must be installed separately.

If your dataset folder already contains `init_<dataset_name>.npy`, you can skip `initialize_pcd.py` and run `train.py` directly.

### Additional Dependencies for `generate_synthetic_mar.py`

`generate_synthetic_mar.py` uses both `tigre` and `xrayPhysics`. To generate synthetic datasets, both libraries must be installed separately.

## 2. Usage

### 2-1. Set the Dataset Path

Before training, update `source_path` in [default.yaml](/d:/codes/eg2026_code_release/src/config/default.yaml:1) so that it points to your dataset.

Example:

```yaml
source_path: "D:/data/R2_gaussian_dataset/CNU_dataset/walnut_bhc_fe"
```

`train.py` reads `--config config/default.yaml` and uses the `source_path` defined there.

### 2-2. Generate a Synthetic Dataset

This is equivalent to the `Generate synthetic dataset` VS Code launch configuration.

```bash
cd src
python generate_synthetic_mar.py --input D:/data/eg2026_data_release/sample_volume_for_mar_synthetic/pancreas_metal.mat
```

`cone_beam.yml` must be located in the same directory as the input `.mat` file.

If `--output` is omitted and the input file is `.../A.mat`, the output directory is automatically set to `.../A`.

Example:

```bash
python generate_synthetic_mar.py \
  --input D:/data/eg2026_data_release/sample_volume_for_mar_synthetic/pancreas_metal.mat \
  --output D:/data/eg2026_data_release/sample_volume_for_mar_synthetic/pancreas_metal_output
```

Main outputs include:

- `vol_gt.npy`
- `vol_mask.npy`
- `vol_fdk.npy`
- `meta_data.json`
- `proj_train/`
- `proj_test/`

You can then use the generated dataset directory as the `source_path` in [default.yaml](/d:/codes/eg2026_code_release/src/config/default.yaml:1) and start training directly.

### 2-3. Initialize the Point Cloud

This is equivalent to the `Initialize point cloud` VS Code launch configuration.

```bash
cd src
python initialize_pcd.py --data D:/data/R2_gaussian_dataset/CNU_dataset/walnut_bhc_fe
```

Example with optional arguments:

```bash
python initialize_pcd.py \
  --data D:/data/R2_gaussian_dataset/CNU_dataset/walnut_bhc_fe \
  --recon_method fdk \
  --n_points 50000
```

This generates `init_<dataset_name>.npy` in the dataset directory.

### 2-4. Run Training

This is equivalent to the `Train start!` VS Code launch configuration.

```bash
cd src
python train.py --config config/default.yaml
```

To match the launch configuration behavior in Windows PowerShell:

```powershell
cd src
$env:CUDA_LAUNCH_BLOCKING="1"
python train.py --config config/default.yaml
```

### 2-5. Output

If `model_path` is left empty, the output directory is created automatically in the following format:

```text
src/output/default_<timestamp>
```

The output directory contains:

- training configuration files (`cfg_args`, `cfg_args.yml`)
- checkpoints
- predicted `eta`
- saved volume results
- TensorBoard logs

## 3. Docker Setup and Usage

The Docker workflow is based on the root-level [Dockerfile](/d:/codes/eg2026_code_release/Dockerfile:1).

### Requirements

- Docker Desktop or Docker Engine
- NVIDIA Container Toolkit
- A system that can run GPU-enabled containers

To verify that the GPU is available inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

### Build the Image

Run this from the repository root:

```bash
docker build -t r2gs_bhc .
```

The Docker image uses `environment.docker.yml`, which is separated from the local Conda environment so that Docker-specific dependency constraints can be handled independently.

### Run the Container

Example: mount a host dataset directory to `/data` inside the container.

Linux/macOS:

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/src/output:/workspace/src/output \
  -v /path/to/dataset:/data \
  r2gs_bhc
```

Windows PowerShell:

```powershell
docker run --rm -it --gpus all `
  -v ${PWD}/src/output:/workspace/src/output `
  -v D:/data/R2_gaussian_dataset:/data `
  r2gs_bhc
```

Inside the container, the working directory is `/workspace/src`.

### Run Commands Inside Docker

1. Update `source_path` in `config/default.yaml` to a container-visible path.

Example:

```yaml
source_path: "/data/CNU_dataset/walnut_bhc_fe"
```

2. If initialization is required:

```bash
python initialize_pcd.py --data /data/CNU_dataset/walnut_bhc_fe
```

3. Run training:

```bash
python train.py --config config/default.yaml
```

### Notes for Docker Usage

- `initialize_pcd.py` requires `tigre`.
- The current `Dockerfile` installs both the training environment and the additional dependencies needed to run `generate_synthetic_mar.py`, including `tigre` and `xrayPhysics`.
- Dataset paths must use the container mount path, not the original host path.

## 4. Mapping to VS Code Launch Configurations

The commands in this README correspond to the following launch entries:

- `Generate synthetic dataset` -> `python generate_synthetic_mar.py --input ...`
- `Initialize point cloud` -> `python initialize_pcd.py --data ...`
- `Train start!` -> `python train.py --config config/default.yaml`

Even when launched from VS Code, the underlying behavior is the same.
