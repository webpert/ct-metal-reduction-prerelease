# projection generator for metal addition (3d version of polyner code)
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse
import os.path as osp
import json
from scipy.io import loadmat
from scipy.ndimage import binary_erosion
import yaml
import sys

import tigre
from tigre.utilities import CTnoise
import tigre.algorithms as algs

from xrayphysics import *

sys.path.append("./")

# ref_mode = 0: use the center energy as reference (default)
# ref_mode = 1: use the effective energy as reference
ref_mode = 0

# program input
# 1. kev: a scalar value of the x-ray energy used as a reference for volume slices
# 2. threshWater: high threshold value for water
# 3. threshBone: low threshold value for bone (normally threshBone > threshWater)
# 4. vol_gt : 3d slices of body CT, atten. coeff. @ keV
# 5. vol_mask: metal is 1, non-metal is 0, same dimension as vol_gt
# 6. cone_beam.yml: geometry information

# physical values (inherited from NIST database)
material_H2O = {'chemical_formula': 'H2O', 'mass_density': 1.0e-3}
material_BONE = {'chemical_formula': 'BONE', 'mass_density': 1.85e-3}
material_Ti = {'chemical_formula': 'Ti', 'mass_density': 4.5e-3}
material_Al = {'chemical_formula': 'Al', 'mass_density': 2.7e-3}
material_Va = {'chemical_formula': 'Va', 'mass_density': 6.099e-3}
material_Fe = {'chemical_formula': 'Fe', 'mass_density': 7.85e-3}
material_Cu = {'chemical_formula': 'Cu', 'mass_density': 8.96e-3}
material_Zn = {'chemical_formula': 'Zn', 'mass_density': 7.14e-3}
material_Au = {'chemical_formula': 'Au', 'mass_density': 19.3e-3}
material_Ag = {'chemical_formula': 'Ag', 'mass_density': 10.49e-2}

# scene dependent parameters
# ===========================
# pancreas
kev = 30.0      
threshWater = 0.04
threshBone = 0.05
volume_to_phys_scale = 0.1
enable_filter = True
filter_thickness = 1.0
material_FILTER = material_Cu
material_METAL = material_Ti
# ===========================

# output configuration
n_train = 720
n_test = 2

# paramters for debugging
enable_constant_spectrum = False
plot_reference_LAC = False
save_fdk_img = True

# x-ray spectrum parameters
sampling_min = 10.0 # should be smaller than kev
sampling_max = 90.0 # should be bigger than kev
sampling_step = 1.0


def recon_volume(projs, angles, geo, recon_method):
    """Reconstruct ct with traditional methods."""
    if recon_method == "fdk":
        vol = algs.fdk(projs[:, ::-1, :], geo, angles)
    elif recon_method == "cgls":
        vol, _ = algs.cgls(projs[:, ::-1, :], geo, angles, 60, computel2=True)
    else:
        raise ValueError("Unsupported reconstruction method")
    vol = np.transpose(vol, (2, 1, 0))
    return vol


def convert_to_builtin(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def get_geometry_tigre(cfg):
    """For TIGRE only."""
    if cfg["mode"] == "parallel":
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(cfg["nVoxel"][::-1]))
    elif cfg["mode"] == "cone":
        geo = tigre.geometry(mode="cone")
    else:
        raise NotImplementedError("Unsupported scanner mode!")

    geo.DSD = cfg["DSD"]  # Distance Source Detector
    geo.DSO = cfg["DSO"]  # Distance Source Origin
    # Detector parameters
    geo.nDetector = np.array(cfg["nDetector"])  # number of pixels
    geo.sDetector = np.array(cfg["sDetector"])  # size of each pixel
    geo.dDetector = geo.sDetector / geo.nDetector  # total size of the detector
    # Image parameters
    geo.nVoxel = np.array(cfg["nVoxel"][::-1])  # number of voxels
    geo.sVoxel = np.array(cfg["sVoxel"][::-1])  # size of each voxel
    geo.dVoxel = geo.sVoxel / geo.nVoxel  # total size of the image
    # Offsets
    geo.offOrigin = np.array(cfg["offOrigin"][::-1])  # Offset of image from origin
    geo.offDetector = np.array(
        [cfg["offDetector"][1], cfg["offDetector"][0], 0]
    )  # Offset of Detector
    # Auxiliary
    geo.accuracy = cfg["accuracy"]  # Accuracy of FWD proj
    # Mode
    geo.filter = cfg["filter"]
    return geo


def parse_arguments():
    parser = argparse.ArgumentParser(description="Load a 3D model file from a given path.")
    parser.add_argument("--input", type=str, help="Path to the mat file")
    parser.add_argument("--output", type=str, help="Output path of scene")
    return parser.parse_args()



def do_resampling(x, y, xmin, xmax, xstep):
    xnew = np.linspace(xmin, xmax, int((xmax-xmin)/xstep) + 1)
    if enable_constant_spectrum:
        ynew = np.ones(xnew.shape[0]) / xnew.shape[0]
    else:
        ynew = np.interp(xnew, x, y)

    return xnew, ynew



def generate_xray_source(physics):    
    physics.use_mm()

    kV = sampling_max
    takeOffAngle = 11.0
    Es, s = physics.simulateSpectra(kV, takeOffAngle)
    detResp = physics.detectorResponse('O2SGd2', 7.32e-3, 0.1, Es)
    if enable_filter:
        filtResp = physics.filterResponse(material_FILTER["chemical_formula"], material_FILTER["mass_density"], filter_thickness, Es)
    else:
        filtResp = np.ones_like(s)
    s_raw = s
    s = s_raw*filtResp*detResp
    Es, s = do_resampling(Es, s, xmin=sampling_min, xmax=sampling_max, xstep=sampling_step)

    return Es, s




if __name__ == '__main__':

    args = parse_arguments()
    if args.input is None:
        raise ValueError("--input is required.")

    args.input = osp.abspath(args.input)
    if args.output is None:
        args.output = osp.abspath(osp.splitext(args.input)[0])
    else:
        args.output = osp.abspath(args.output)

    # setup file names
    base_path = osp.dirname(args.input)
    total_fn = args.input
    vol_name = osp.basename(total_fn)[:-4]
    scanner_cfg_path = osp.join(base_path, "cone_beam.yml")      
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    print("##### Generate MAR synthetic data for Cone-Beam CT #####")
    # Load configuration
    with open(scanner_cfg_path, "r") as handle:
        scanner_cfg = yaml.safe_load(handle)
    case_name = f"{vol_name}_{scanner_cfg['mode']}"
    print(f"Generate data for case {case_name}")
    geo = get_geometry_tigre(scanner_cfg)        

    # generate a polychromatic x-ray source
    physics = xrayPhysics()    
    Es, s = generate_xray_source(physics)
    kev_idx = np.where(Es == kev)
    if len(kev_idx) != 1:
        print("reference keV is not in the valid range!")
        exit(0)
    else:
        kev_idx = kev_idx[0][0]

    # Mass Atten. Coeff.
    sigma_H2O = physics.sigma(material_H2O['chemical_formula'], Es)
    sigma_BONE = physics.sigma(material_BONE['chemical_formula'], Es)
    sigma_METAL = physics.sigma(material_METAL['chemical_formula'], Es)
    sigma_all = np.stack([sigma_H2O, sigma_BONE, sigma_METAL], axis=0)  # [M, E]

    if plot_reference_LAC:
        # for estimating threshold
        LAC_H2O = sigma_H2O * material_H2O['mass_density']
        LAC_BONE = sigma_BONE * material_BONE['mass_density']
        LAC_METAL = sigma_METAL * material_METAL['mass_density']
        plt.figure(figsize=(8, 5))
        plt.plot(Es, LAC_H2O, label='Water', marker='o')
        plt.plot(Es, LAC_BONE, label='Bone', marker='s')
        # plt.plot(Es, LAC_METAL, label='Metal', marker='^')

        plt.xlabel('Energy (keV)')
        plt.ylabel('Linear Attenuation Coefficient (1/cm)')
        plt.title('LAC vs Energy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # load the CT volume and meta data file
    mat_file = loadmat(total_fn)
    # print(mat_file.keys())
    vol_gt = mat_file['vol_gt'].astype(np.float32) * volume_to_phys_scale     # [slice, H, W]
    vol_mask = mat_file['vol_mask'].astype(np.float32) # [slice, H, W]
    # plt.figure()
    # plt.imshow(vol_gt[:, :, vol_gt.shape[-1]//2], cmap='gray')
    # plt.show(block=True)
    # metadata = mat_file['metadata']

    # Generate training/testing angles
    projs_train_angles = (
        np.linspace(0, scanner_cfg["totalAngle"] / 180 * np.pi, n_train + 1)[:-1]
        + scanner_cfg["startAngle"] / 180 * np.pi
    ).astype(np.float32)
    projs_test_angles = (
        np.sort(np.random.rand(n_test) * 360 / 180 * np.pi)  # Evaluate full circle
        + scanner_cfg["startAngle"] / 180 * np.pi
    ).astype(np.float32)
    # this part is just for dummy data (we are not interested in novel view synthesis)
    projs_test = tigre.Ax(np.transpose(vol_gt, (2, 1, 0)).copy(), geo, projs_test_angles)[
        :, ::-1, :
    ]    

    img = vol_gt
    maskWater = (img <= threshWater)
    maskBone = (img >= threshBone)
    maskBoth = ((img > threshWater) & (img < threshBone))

    imgWater = np.zeros_like(img)
    imgBone = np.zeros_like(img)
    imgWater[maskWater] = img[maskWater]
    imgBone[maskBone] = img[maskBone]
    imgBone[maskBoth] = (img[maskBoth] - threshWater) / (threshBone - threshWater) * img[maskBoth]
    imgWater[maskBoth] = img[maskBoth] - imgBone[maskBoth]

    imgMetal = vol_mask
    maskMetal = (imgMetal > 0)
    imgWater_local = imgWater
    imgBone_local = imgBone
    imgWater_local[maskMetal] = 0.0
    imgBone_local[maskMetal] = 0.0
    
    # synthesize non-metal poly CT
    # voxel_scale = scanner_cfg["sVoxel"][0] / scanner_cfg["nVoxel"][0]  # assumtion: cubic voxel, mm unit

    Pwater_kev = tigre.Ax(
        np.transpose(imgWater_local, (2, 1, 0)).copy(), geo, projs_train_angles
    )[:, ::-1, :]    
    # Pwater_kev = Pwater_kev * voxel_scale

    Pbone_kev = tigre.Ax(
        np.transpose(imgBone_local, (2, 1, 0)).copy(), geo, projs_train_angles
    )[:, ::-1, :]
    # Pbone_kev = Pbone_kev * voxel_scale

    Pmetal_kev = tigre.Ax(
        np.transpose(sigma_METAL[kev_idx] * material_METAL['mass_density'] * imgMetal, (2, 1, 0)).copy(), geo, projs_train_angles
    )[:, ::-1, :]    
    # Pmetal_kev = Pmetal_kev * voxel_scale

    # partial volume effect implementation (3-dimensional morphological operation)
    mask = Pmetal_kev > 0
    structure = np.ones((3, 3, 3), dtype=bool)
    eroded_mask = binary_erosion(mask, structure=structure)
    Pmetal_edge = np.logical_xor(mask, eroded_mask)
    Pmetal_kev[Pmetal_edge] = Pmetal_kev[Pmetal_edge] / 4

    proj_kev_all = np.stack([Pwater_kev, Pbone_kev, Pmetal_kev], axis=0)    # [M, B, H, W]

    # compute GT atten. coeff. volume
    poly_img = np.zeros((imgWater_local.shape[0], imgWater_local.shape[1], imgWater_local.shape[2], Es.shape[0]), dtype=np.float32)
    for idx in range(Es.shape[0]):
        poly_water = sigma_H2O[idx] / sigma_H2O[kev_idx] * imgWater_local
        poly_bone = sigma_BONE[idx] / sigma_BONE[kev_idx] * imgBone_local
        poly_metal = sigma_METAL[idx] * material_METAL['mass_density'] * imgMetal
        poly_img[..., idx] = poly_water.astype(np.float64) + poly_bone.astype(np.float64) + poly_metal.astype(np.float64)
    
    # export GT volume to be used for comparison (referece: kev)
    if ref_mode == 0:
        rep_idx = np.where(Es == (sampling_min + sampling_max) / 2) # center energy
    else:        
        E_eff = (Es * s).sum() / s.sum()
        rep_idx = np.where(Es == round(E_eff))
        print(f'Effective energy for simulation: {E_eff} keV, use {Es[rep_idx]} keV as reference.')
    vol_gt = poly_img[..., rep_idx].astype(np.float32)
    # vol_gt_all = poly_img.squeeze().transpose(3,0,1,2).astype(np.float32)

    # polychromatic integral using projection data
    proj_all = np.zeros_like(proj_kev_all).astype(np.float32)   # [M, B, H, W]
    proj_energy = np.zeros((proj_kev_all.shape[1], proj_kev_all.shape[2], proj_kev_all.shape[3])).astype(np.float32)   # [B, H, W]
    proj_kvp = np.zeros((proj_kev_all.shape[1], proj_kev_all.shape[2], proj_kev_all.shape[3])).astype(np.float32)   # [B, H, W]
    for ien in range(Es.shape[0]):
        for imat in range(proj_kev_all.shape[0]):
            # scale to the specified energy w.r.t. the reference energy
            proj_all[imat, ...] = sigma_all[imat, ien] / sigma_all[imat, kev_idx] * proj_kev_all[imat, ...]
        proj_sum = proj_all.sum(axis=0)
        ptmp = s[ien] * np.exp(-proj_sum)
        print(f'exponent max for {Es[ien]}kev : {proj_sum.max()}')
        proj_energy = proj_energy + ptmp
    proj_energy_blank_ratio = np.sum(s) * np.ones_like(proj_kvp)
    proj_kvp = -np.log(proj_energy / proj_energy_blank_ratio)   # final simulated projection

    # noise injection
    if scanner_cfg["noise"]:
        proj_kvp = CTnoise.add(
            proj_kvp.astype(np.float32),
            Poisson=scanner_cfg["possion_noise"],
            Gaussian=np.array(scanner_cfg["gaussian_noise"]),
        )  #
        proj_kvp[proj_kvp < 0.0] = 0.0

    # Save
    case_save_path = output_path
    os.makedirs(case_save_path, exist_ok=True)
    np.save(osp.join(case_save_path, "vol_gt.npy"), vol_gt.squeeze())
    # np.save(osp.join(case_save_path, "vol_gt_all.npy"), vol_gt_all)
    file_path_dict = {}
    for split, projs, angles in zip(
        ["proj_train", "proj_test"],
        [proj_kvp, projs_test],
        [projs_train_angles, projs_test_angles],
    ):
        os.makedirs(osp.join(case_save_path, split), exist_ok=True)
        file_path_dict[split] = []
        for i_proj in range(projs.shape[0]):
            proj = projs[i_proj]
            frame_save_name = osp.join(split, f"{split}_{i_proj:04d}.npy")
            np.save(osp.join(case_save_path, frame_save_name), proj)
            file_path_dict[split].append(
                {
                    "file_path": frame_save_name,
                    "angle": angles[i_proj],
                }
            )
    bbox_x = scanner_cfg["sVoxel"][0] / 2.0
    bbox_y = scanner_cfg["sVoxel"][1] / 2.0
    bbox_z = scanner_cfg["sVoxel"][2] / 2.0
    meta = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "bbox": [[-bbox_x, -bbox_y, -bbox_z], [bbox_x, bbox_y, bbox_z]],
        "proj_train": file_path_dict["proj_train"],
        "proj_test": file_path_dict["proj_test"],
    }
    with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, default=convert_to_builtin)
    print(f"Generate data for case {case_name} complete!")

    vol_fdk = recon_volume(proj_kvp, projs_train_angles, geo, "fdk")
    vol_fdk[vol_fdk<0.0] = 0.0
    f_recon_slice = vol_fdk[...,vol_fdk.shape[-1]//2]
    np.save(osp.join(case_save_path, "vol_fdk.npy"), vol_fdk)
 
    # save FDK reconstructed image (sample)
    if save_fdk_img:
        out_fn = osp.join(case_save_path, "sample_recon.png")
        imageio.imsave(out_fn, np.uint8(f_recon_slice/np.max(f_recon_slice)*255)) # sample    
    np.save(osp.join(case_save_path, 'vol_mask.npy'), vol_mask)
    
