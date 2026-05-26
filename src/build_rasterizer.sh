rm -rf "./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/build"
pip uninstall -y xray_gaussian_rasterization_voxelization
pip install --no-build-isolation -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
