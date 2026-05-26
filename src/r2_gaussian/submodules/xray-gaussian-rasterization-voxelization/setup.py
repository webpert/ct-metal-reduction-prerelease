#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import shutil
import tempfile
import urllib.request
import zipfile

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY_DIR = os.path.join(ROOT_DIR, "third_party")
GLM_DIR = os.path.join(THIRD_PARTY_DIR, "glm")
GLM_VERSION = "1.0.2"
GLM_URL = os.environ.get(
    "R2_GAUSSIAN_GLM_URL",
    f"https://github.com/g-truc/glm/archive/refs/tags/{GLM_VERSION}.zip",
)


def ensure_glm():
    glm_header = os.path.join(GLM_DIR, "glm", "detail", "setup.hpp")
    if os.path.exists(glm_header):
        return

    os.makedirs(THIRD_PARTY_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="glm-download-") as tmpdir:
        archive_path = os.path.join(tmpdir, "glm.zip")
        print(f"Downloading GLM {GLM_VERSION} from {GLM_URL}")
        try:
            urllib.request.urlretrieve(GLM_URL, archive_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download GLM. Set R2_GAUSSIAN_GLM_URL to a reachable archive URL."
            ) from exc

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmpdir)

        extracted_root = None
        for entry in os.listdir(tmpdir):
            candidate = os.path.join(tmpdir, entry)
            if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, "glm", "detail", "setup.hpp")
            ):
                extracted_root = candidate
                break

        if extracted_root is None:
            raise RuntimeError("Downloaded GLM archive has an unexpected layout.")

        if os.path.exists(GLM_DIR):
            shutil.rmtree(GLM_DIR)
        shutil.move(extracted_root, GLM_DIR)


ensure_glm()

setup(
    name="xray_gaussian_rasterization_voxelization",
    packages=["xray_gaussian_rasterization_voxelization"],
    ext_modules=[
        CUDAExtension(
            name="xray_gaussian_rasterization_voxelization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "cuda_voxelizer/voxelizer_impl.cu",
                "cuda_voxelizer/forward.cu",
                "cuda_voxelizer/backward.cu",
                "voxelize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I"
                    + os.path.join(ROOT_DIR, "third_party", "glm"),
                    "-D_USE_MATH_DEFINES"
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
