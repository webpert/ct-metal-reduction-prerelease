# Splat-based Metal Artifact Reduction in Cone-Beam CT via Polychromatic Modeling
Cone-beam computed tomography (CBCT) enables volumetric reconstruction from X-ray projections, but suffers from severe artifacts--especially beam hardening--when imaging materials with high attenuation such as metals. These artifacts arise from the polychromatic nature of X-rays and are not properly addressed by conventional monochromatic reconstruction algorithms. While recent neural representation-based methods offer improved reconstruction quality, they are computationally expensive and often impractical for deployment. We propose a novel physics-inspired, self-calibrating metal artifact reduction method that efficiently reconstructs 3D CBCT volumes while correcting beam hardening artifacts. Our method integrates a polychromatic X-ray projection model, material-dependent attenuation profiles, and system response modeling into a Gaussian Splatting framework. Unlike prior work, we eliminate the need for manual metal masks or strong prior assumptions, and we optimize both reconstruction parameters and X-ray spectral characteristics jointly during training. We further introduce a high-fidelity synthetic CBCT dataset generation pipeline validated on Monte-Carlo x-ray simulation toolbox and release new datasets with severe metal-induced artifacts to support the community. This is the first splat-based method for reducing beam hardening in CBCT. Extensive experiments on both synthetic and real-world datasets demonstrate that our method outperforms state-of-the-art approaches in artifact suppression and reconstruction accuracy.

### [Project page](https://vclab.kaist.ac.kr/eg2026/index.html) | [Paper](https://vclab.kaist.ac.kr/eg2026/paper1170_CRC.pdf) | [Supplemental](https://vclab.kaist.ac.kr/eg2026/paper1170_CRC_MM1.pdf)
[Kiseok Choi](https://sites.google.com/view/kiseokchoi), 
[Inchul Kim](https://inchul-kim.github.io/), 
[Jaemin Cho](http://vclab.kaist.ac.kr/jmcho/index.html), 
[Hyeongjun Cho](http://vclab.kaist.ac.kr/hjcho/index.html), 
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim/index.html)

This repository is organized around the execution pipeline defined in `.vscode/launch.json`.

- `generate_synthetic_mar.py`: Generates synthetic CBCT datasets using predefined configuration files (cone_beam.yml, xxx.mat)
- `initialize_pcd.py`: Initializes the Gaussian point cloud
- `train.py`: Runs CBCT reconstruction with metal artifact reduction

## Tested Environment
```
CPU: Intel Xeon 4214R @ 2.4Ghz
GPU: NVIDIA A6000 48GB
RAM: 256GB
OS: Ubuntu 22.04
Docker: 24.0.2
CUDA: 13.1 (Driver: 590.48.01)
```

## Setup
Ensure that your system supports CUDA and Docker. Then clone this repository and move into the project directory.
```
git clone https://github.com/webpert/ct-metal-reduction-prerelease.git
cd ct-metal-reduction-prerelease
```

Build the Docker image using Dockerfile. This process may take several minutes.
```
docker build -t r2gs_bhc:cuda118 .
```
Launch the Docker container. Modify the options below according to your environment.
```
docker run -it --gpus all \
  --name r2gs_bhc \
  -v /mnt/datassd/kschoi/data:/workspace/data \
  -p 20000:20000 \
  r2gs_bhc:cuda118
```

## Data Preparation
Download one of the datasets from [link](https://drive.google.com/drive/folders/1l4noH0qe3abyq17l8Ex3BiDFcygj9hLs?usp=drive_link) and extract it to your preferred directory.<br>
``sample_volume_for_synthetic_generation.zip``  is intended for synthetic data generation, while all other datasets are used for reconstruction experiments.

## Synthetic Data Generation
Download ``sample_volume_for_synthetic_generation.zip`` from [link](https://drive.google.com/drive/folders/1l4noH0qe3abyq17l8Ex3BiDFcygj9hLs?usp=drive_link) or prepare your own raw data following the sample format.<br>
Extract the archive and run:<br>
($RAW_DATA_PATH can be ``sample_volume_for_synthetic_generation/pancreas_metal.mat``)
```
python generate_synthetic_mar.py --input $RAW_DATA_PATH
```
If custom raw data is used, adjust the relevant hyperparameters in the source code when necessary.

## Gaussian Initialization
Run the following command to initialize the Gaussian representation.<br>
($SCENE_PATH can be ``real_walnut`` or something).
```
python initialize_pcd.py --data $SCENE_PATH
```
This step can be skipped if the dataset directory already contains initialized Gaussians (e.g., ``init_$SCENE_PATH.npy``).

## Reconstruction
Modify ``source_path`` in ``./config/default.yaml`` and start reconstruction using:
```
python train.py --config $CONFIG_FILE_PATH
```
After optimization completes (20,000 iterations; approximately 18 minutes on the tested hardware), the reconstructed volume will be saved to:
```
./output/$CONFIG_FILE_NAME_MM-DD-hh-mm-ss/point_cloud/iteration_20000/vol_center.npy
```

## Citation
```	
@Article{Choi:EG:2026,
  author  = {Choi, Kiseok and Kim, Inchul and Cho, Jaemin and Cho, Hyeongjun and Kim, Min H.},
  title   = {Splat-based Metal Artifact Reduction in Cone-Beam CT
             via Polychromatic Modeling},
  journal = {Computer Graphics Forum (Proc. EUROGRAPHICS 2026)},
  year    = {2026},
  volume  = {45},
  number  = {2},
  pages   = {}
}
```

## Acknowledgements
This project builds upon the implementation of [R2-Gaussian](https://github.com/ruyi-zha/r2_gaussian) as the primary reconstruction framework, with our proposed physical modeling components integrated into the original pipeline.
