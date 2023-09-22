import argparse
import os
import pathlib
import shutil
import datetime
from subprocess import check_output

def compress_if_not_modified(directory, topics, compression_format=".zip"):
    # Get the current date
    current_date = datetime.datetime.now()

    # Iterate through the subdirectories in the given directory
    for root, dirs, files in os.walk(directory):
        print(root)
        print(dirs)
        for dir_name in dirs:
            if dir_name.startswith("root_model"):
                print(dir_name)
                dir_path = os.path.join(root, dir_name)

                # Get the modification time of the directory
                modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(dir_path))

                # Calculate the time difference in days
                time_difference = (current_date - modification_time).days

                # Compress the directory if it hasn't been modified in a day
                if time_difference >= 1:
                    print("Not modified in a day")
                    #compressed_filename = f"{dir_name}{compression_format}"
                    #compressed_path = os.path.join(root, compressed_filename)
                    #shutil.make_archive(compressed_path[:-len(compression_format)], compression_format[1:], dir_path)
                    #shutil.rmtree(dir_path)
                    #print(f"Compressed and removed: {dir_path}")
                
def compress_all(path_models, root_topics, compression_format=".tar.gz"):
    
    # Get the current date
    current_date = datetime.datetime.now()
    
    root_topics = root_topics.split(',')
    for root_topic in root_topics:
        path_models_tpc = pathlib.Path(path_models).joinpath(f"{root_topic}_tpc_root")
        print(f"-- -- Compressing {path_models_tpc}")
        # Iter over root models
        for entry in path_models_tpc.iterdir():
            
            # Get the modification time of the directory
            modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(entry))
            
            # Calculate the time difference in days
            time_difference = (current_date - modification_time).days

            # Compress the directory if it hasn't been modified in a day
            if time_difference >= 1 \
                and not entry.as_posix().startswith("root_model_0"):
                    
                print(f"Not modified in a day: {entry}. Compressing...")
                compressed_path = entry.with_suffix(compression_format)
                
                # tar -zcvf myfolder.tar.gz myfolder
                cmd = 'tar -zcvf %s %s' % (compressed_path, entry)

                try:
                    print(f'-- Running command {cmd}')
                    check_output(args=cmd, shell=True)
                except:
                    print('-- Failed to extract pipeline. Revise command')

                shutil.rmtree(entry)
                print(f"Compressed and removed: {entry}")
            else:
                print(f"Modified in a day: {entry}. Skipping...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_models', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models_ctm",
                        help="Path to the training data.")
    parser.add_argument('--root_topics', type=str,
                    default="5,10,20",
                    help="Nr of topics for the root model.")

    args = parser.parse_args()
    compress_all(path_models=args.path_models,
                 root_topics=args.root_topics)


if __name__ == "__main__":
    main()