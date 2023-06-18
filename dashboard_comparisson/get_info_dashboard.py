import argparse
import configparser
import os
import pathlib
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.append('..')
from src.utils.misc import pickler, unpickler
from src.evaluateMatching.alignment import Alignment


def update_df_info(path_to_root_model: str,
                   corpus: str,
                   df: pd.DataFrame) -> pd.DataFrame:

    path_to_root_model = pathlib.Path(path_to_root_model)

    dfs = []

    # Append df to dfs
    if df is not None:
        dfs.append(df)

    # Append root model
    dfs.append(get_model_info(path_to_root_model, corpus, True))

    # Iterate over submodels
    for entry in path_to_root_model.iterdir():
         if entry.joinpath('TMmodel/alphas.npy').is_file():
             dfs.append(get_model_info(entry, corpus))

    # Concatenate all dfs
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by=['corpus'])
    return df


def update_sims_wmd(df, df_sims, corpus, root_model):

    dfs = []
    if df_sims is not None:
        dfs.append(df_sims)

    root_model = pathlib.Path(root_model)
    df_corpus = df[df.corpus == corpus]
    submodels = df_corpus[(df_corpus.father_model ==
                           root_model.stem)].model.unique()

    al = Alignment()
    for (sub1, sub2) in list(combinations(submodels, 2)):
        print(sub1, sub2)
        vs_sims, wmd1, wmd2 = al.do_one_to_one_matching(
            tmModel1=root_model.joinpath(sub1).as_posix(),
            tmModel2=root_model.joinpath(sub2).as_posix())

        df = pd.DataFrame(
            {
                'model_1': sub1,
                'model_2': sub2,
                'vs_sims': [vs_sims],
                'wmd_1': [wmd1],
                'wmd_2': [wmd2],
            }
        )
        dfs.append(df)
    # Concatenate all dfs
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by=['model_1'])

    return df


def get_model_info(path: pathlib.Path,
                   corpus: str,
                   root: bool = False) -> pd.DataFrame:
    """Get model info from a given model path.

    Parameters
    ----------
    path : pathlib.Pathr
        Path to the model.
    corpus : str
        Name of the corpus.
    root : bool, optional
        If True, the model is the root model, by default False.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the model info.
    """

    alphas = np.load(path.joinpath('TMmodel/alphas.npy')).tolist()
    alphas_len = len(alphas)
    alphas = ', '.join(str(element) for element in alphas)

    cohrs = np.load(path.joinpath('TMmodel/topic_coherence.npy')).tolist()
    if len(cohrs) > alphas_len:
        cohrs_cv = cohrs[0:alphas_len]
        cohrs_npmi = cohrs[alphas_len:]
    elif len(cohrs) == alphas_len:
        cohrs_cv = cohrs
        cohrs_npmi = [0] * alphas_len
    cohrs_cv = ', '.join(str(element) for element in cohrs_cv)
    cohrs_npmi = ', '.join(str(element) for element in cohrs_npmi)

    entropies = np.load(
        path.joinpath('TMmodel/topic_entropy.npy')).tolist()
    entropies = ', '.join(str(element) for element in entropies)

    with open(path.joinpath('TMmodel/tpc_descriptions.txt'), "r") as file:
        lines = file.readlines()
    keywords = [line.strip().split(", ") for line in lines]

    if root:
        model_type = "first"
        father_model = None
    else:
        if "ws" in path.as_posix():
            model_type = "WS"
        else:
            model_type = "DS"
        father_model = path.parent.stem

    return pd.DataFrame(
        {'model': path.as_posix().split('/')[-1],
         'model_type': model_type,
         'corpus': corpus,
         'father_model': father_model,
         'alphas': alphas,
         'cohrs_npmi': cohrs_npmi,
         'cohrs_cv': cohrs_cv,
         'entropies': entropies,
         'keywords': [keywords],
         }, index=[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_root_model", type=str, required=False,
                        default="/Volumes/usuarios_ml4ds/lbartolome/Datasets/CORDIS/mallet_models/root_model_1_20230613")
    parser.add_argument("--corpus", type=str, required=False,
                        default="Cordis")
    args = parser.parse_args()

    # Get config with path to data
    cf = configparser.ConfigParser()
    current_path = \
        os.path.dirname(os.path.dirname(os.getcwd()))
    cf.read(os.path.join(current_path, 'UserInLoopHTM',
            'dashboard_comparisson', 'config', 'config.cf'))

    # Update df_info
    df_info = unpickler(cf.get("paths", "path_df_info"))
    df_info = update_df_info(args.path_to_root_model, args.corpus, df_info)
    pickler(cf.get("paths", "path_df_info"), df_info)

    # Update df_sims
    df_sims_wmds = unpickler(cf.get("paths", "path_df_sims"))
    df_sims_wmds = update_sims_wmd(
        df_info, df_sims_wmds, args.corpus, args.path_to_root_model)
    pickler(cf.get("paths", "path_df_sims"), df_sims_wmds)


if __name__ == "__main__":
    main()
