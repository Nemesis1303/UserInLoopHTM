import argparse
import os
import shutil
import pathlib

def remove_heavy(path_models):
    path_models = pathlib.Path(path_models)
    for model in path_models.iterdir():
        if model.joinpath('modelFiles').is_dir():
            shutil.rmtree(model.joinpath('modelFiles'))
        if model.joinpath('corpus.txt').is_file():
            os.remove(model.joinpath('corpus.txt'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_models', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models",
                        help="Path to the training data.")

    args = parser.parse_args()
    remove_heavy(path_models=args.path_models)


if __name__ == "__main__":
    main()
