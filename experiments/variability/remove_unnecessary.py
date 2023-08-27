import argparse
import os
import shutil
import pathlib

def remove_heavy(path_models, root_topics):
    root_topics = root_topics.split(',')
    for root_topic in root_topics:
        path_models_tpc = pathlib.Path(path_models).joinpath(f"{root_topic}_tpc_root")
        
        print(f"-- -- Removing heavy for {path_models_tpc}")
        
        def remove(model_path):
            '''
            if model_path.joinpath('modelFiles').is_dir():
                shutil.rmtree(model_path.joinpath('modelFiles'))
            if model_path.joinpath('corpus.txt').is_file():
                os.remove(model_path.joinpath('corpus.txt'))
            '''
            if model_path.as_posix().endswith("old"):
                shutil.rmtree(model_path)
        
        for entry in path_models_tpc.iterdir():
            remove(entry)
            for entry_ in entry.iterdir():
                remove(entry_)      

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_models', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models",
                        help="Path to the training data.")
    parser.add_argument('--root_topics', type=str,
                    default="5,10,20",
                    help="Nr of topics for the root model.")

    args = parser.parse_args()
    remove_heavy(path_models=args.path_models,
                 root_topics=args.root_topics)


if __name__ == "__main__":
    main()
