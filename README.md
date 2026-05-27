# Splat-based Metal Artifact Reduction in Cone-Beam CT via Polychromatic Modeling
Cone-beam computed tomography (CBCT) enables volumetric reconstruction from X-ray projections, but suffers from severe artifacts--especially beam hardening--when imaging materials with high attenuation such as metals. These artifacts arise from the polychromatic nature of X-rays and are not properly addressed by conventional monochromatic reconstruction algorithms. While recent neural representation-based methods offer improved reconstruction quality, they are computationally expensive and often impractical for deployment. We propose a novel physics-inspired, self-calibrating metal artifact reduction method that efficiently reconstructs 3D CBCT volumes while correcting beam hardening artifacts. Our method integrates a polychromatic X-ray projection model, material-dependent attenuation profiles, and system response modeling into a Gaussian Splatting framework. Unlike prior work, we eliminate the need for manual metal masks or strong prior assumptions, and we optimize both reconstruction parameters and X-ray spectral characteristics jointly during training. We further introduce a high-fidelity synthetic CBCT dataset generation pipeline validated on Monte-Carlo x-ray simulation toolbox and release new datasets with severe metal-induced artifacts to support the community. This is the first splat-based method for reducing beam hardening in CBCT. Extensive experiments on both synthetic and real-world datasets demonstrate that our method outperforms state-of-the-art approaches in artifact suppression and reconstruction accuracy.

### [Project page](https://vclab.kaist.ac.kr/eg2026/index.html) | [Paper](https://vclab.kaist.ac.kr/eg2026/paper1170_CRC.pdf) | [Supplemental](https://vclab.kaist.ac.kr/eg2026/paper1170_CRC_MM1.pdf)
[Kiseok Choi](https://sites.google.com/view/kiseokchoi), 
[Inchul Kim](https://inchul-kim.github.io/), 
[Jaemin Cho](http://vclab.kaist.ac.kr/jmcho/index.html), 
[Hyeongjun Cho](http://vclab.kaist.ac.kr/hjcho/index.html), 
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim/index.html)

This repository is organized around the run configurations defined in `.vscode/launch.json`.

- `generate_synthetic_mar.py`: generate a synthetic dataset with predefined configuration files (cone_beam.yml, xxx.mat)
- `initialize_pcd.py`: create the initial Gaussian point cloud
- `train.py`: run reconstuction with metal artifact reduction

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
Make sure that your hardware supports CUDA and Docker and clone the following repository and enter into the created folder.
```
git clone https://github.com/webpert/ct-metal-reduction-prerelease.git
cd ct-metal-reduction-prerelease
```

The following command builds a docker image with ./Dockerfile. This may take some time.
```
docker build -t r2gs_bhc:cuda118 .
```
Start the docker container. You may modify some options for your own environment.
```
docker run -it --gpus all \
  --name r2gs_bhc \
  -v /mnt/datassd/kschoi/data:/workspace/data \
  -p 20000:20000 \
  r2gs_bhc:cuda118
```

## Data Preparation
Download one of the dataset from [link](https://drive.google.com/drive/folders/1l4noH0qe3abyq17l8Ex3BiDFcygj9hLs?usp=drive_link) and extract the zip file to a specific directory.<br>
Note that ``sample_volume_for_synthetic_generation.zip``  is an input for synthetic data generationn and the others are for reconstruction.

## Synthetic Data Generation
Download ``sample_volume_for_synthetic_generation.zip`` from [link](https://drive.google.com/drive/folders/1l4noH0qe3abyq17l8Ex3BiDFcygj9hLs?usp=drive_link).<br>
Extract the zip file to a specific directory and run the following command.<br>
($RAW_DATA_PATH can be ``sample_volume_for_synthetic_generation/pancreas_metal.mat``)
```
python generate_synthetic_mar.py --input $RAW_DATA_PATH
```

## Intialization of Gaussians
Run the following command ($SCENE_PATH can be ``real_walnut`` or something).<br>
This process can be ignored when $SCENE_PATH already has initial Gaussians such as ``init_$SCENE_PATH.npy``
```
python initialize_pcd.py --data $SCENE_PATH
```

## Reconstruction
Modify the data path in "./config/default.yaml" and run the reconstruction program using the following command.
```
python train.py --config $CONFIG_FILE_PATH
```
Once the optimization process is completed, with 20000 iterations (it took about 18 minutes on the tested hardware), <br>
the resulting volume will be saved in the directory "./output/$CONFIG_FILE_NAME_MM-DD-hh-mm-ss/point_cloud/iteration_20000" under the filename "vol_center.npy." 

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
We employed the [R2-Gaussian](https://github.com/ruyi-zha/r2_gaussian) code as the primary reconsruction algorithm and added our models into it.

