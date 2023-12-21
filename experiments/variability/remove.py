import shutil
import pathlib

path_root = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/Datasets/Cancer/htm_variability_models/htm_20_tpcs_20230927")
path_root2 = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/Datasets/S2CS-AI/htm_variability_models/htm_20_tpcs_20230929")

for path in [path_root, path_root2]:
    print("Removing submodels from", path.as_posix())
    [shutil.rmtree(folder) for folder in path_root.iterdir(
    ) if folder.is_dir() and folder.name.startswith(f"submodel_htm-ds")]
