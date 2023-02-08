import numpy as np
import pandas as pd
import pathlib
import tqdm

from src.topicmodeling.manageModels import TMmodel


path_models_ctm = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_htm_ctm")
path_models_mallet = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_htm")
path_models = path_models_mallet

dfs = []
for entry in path_models.iterdir():
    # check if it is a root model
    if "root" in entry.as_posix():
        
        # Thr and exp_tpc do not apply for the root model
        thr = -1
        exp_tpc = -1
        
        # Experiment iteration
        iter_ = int(entry.as_posix().split("model_")[1].split("_")[0])
        
        # Size of the topics
        alphas = np.load(entry.joinpath('TMmodel/alphas.npy')).tolist()
            
        # Create dataframe for the root model
        root_tpc_df = pd.DataFrame(
            {'iter': [iter_] * len(alphas),
             'path': [entry] * len(alphas),
             'thr': [thr] * len(alphas),
             'exp_tpc': [exp_tpc] * len(alphas),
            })
        
        # Append to the list of dataframes to concatenate them
        dfs.append(root_tpc_df)
df = pd.concat(dfs)
df = df.sort_values(by=['iter'])

# Iter over each root model (according to its corresponding iteration, iter)
concats = [df]
not_finished = []
for el in df.iter.unique():
    path_root = df[df.iter == el].iloc[0].path
    for entry in path_root.iterdir():
        if entry.joinpath('TMmodel/alphas.npy').is_file():
            
            # Get thr if htm-ds
            thr = 0 if "ws" in entry.as_posix() else float(entry.as_posix().split("thr_")[1].split("_")[0])

            # Get topic from which the submodel is generated
            exp_tpc = int(entry.as_posix().split("from_topic_")[1].split("_")[0]) 
            
            # Add entry of submodel to dataframe
            root_tpc_df = pd.DataFrame(
            {'iter': [el],
             'path': [entry],
             'thr': [thr],
             'exp_tpc': [exp_tpc],
            })
            concats.append(root_tpc_df)
        else:
            not_finished.append(entry)
df = pd.concat(concats)

for path_model in df.path.unique():
    print(path_model.as_posix())
    path_tm_model = path_model.joinpath('TMmodel')
    tm = TMmodel(path_tm_model)
    tm.recalculate_cohrs()
    
    