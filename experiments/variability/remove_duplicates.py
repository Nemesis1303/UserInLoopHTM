import argparse
import os
import pathlib
import shutil

def remove_duplicates(path_models, root_topics):
    root_topics = root_topics.split(',')
    for root_topic in root_topics:
        path_models_tpc = pathlib.Path(path_models).joinpath(f"{root_topic}_tpc_root")
        
        print(f"-- -- Removing duplicates for {path_models_tpc}")
        
        def list_directories(directory_path):
            directories = []
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                if os.path.isdir(item_path) and item.startswith("submodel"):
                    directories.append(item)
            return directories
        
        # Iter over root models
        for entry in path_models_tpc.iterdir():
            directories_list = list_directories(entry)
            models_date = [int(x.split("_")[-1][-2:]) for x in directories_list ]
            models_name = ["_".join(x.split("_")[:-1]) for x in directories_list]
            
            for i, model_name in enumerate(models_name):
                for j, model_name_ in enumerate(models_name):
                    if i != j:
                        if model_name == model_name_:
                            if models_date[i] > models_date[j]:
                                try:
                                    shutil.rmtree(entry.joinpath(directories_list[j]))
                                    print("Removing", entry.joinpath(directories_list[j]))
                                    print("Keeping", entry.joinpath(directories_list[i]))
                                except:
                                    print("Could not be removed", entry.joinpath(directories_list[j]))
                                
                            else:
                                try:
                                    shutil.rmtree(entry.joinpath(directories_list[i]))
                                    print("Removing", entry.joinpath(directories_list[i]))
                                    print("Keeping", entry.joinpath(directories_list[j]))
                                except:
                                    print("Could not be removed", entry.joinpath(directories_list[i]))
                                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_models', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models",
                        help="Path to the training data.")
    parser.add_argument('--root_topics', type=str,
                    default="5,10,20",
                    help="Nr of topics for the root model.")

    args = parser.parse_args()
    remove_duplicates(path_models=args.path_models,
                 root_topics=args.root_topics)


if __name__ == "__main__":
    main()
