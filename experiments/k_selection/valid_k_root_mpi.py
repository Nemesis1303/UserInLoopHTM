import argparse
import itertools
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedKFold, train_test_split
from mpi4py import MPI
import timeit
from datetime import datetime

# Add src to path and make imports
sys.path.append('../..')
from src.utils.misc import (
    corpus_df_to_mallet, mallet_corpus_to_df, read_config_experiments)
from src.tmWrapper.tm_wrapper import TMWrapper
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel

# Auxiliary functions
def save_values_to_file(filename, values):
    with open(filename, 'w') as file:
        for value in values:
            file.write(str(value) + '\n')


def train_hyper_comb(comb_idx: int,
                     grid_params_iter_lst: list,
                     rkf: RepeatedKFold,
                     corpus_train: pd.DataFrame,
                     training_params: dict,
                     corpusFile: str,
                     models_folder: str,
                     trainer: str):
    """_summary_

    Parameters
    ----------
    comb_idx : int
        Index of the hyperparameter combination to train.
    grid_params: list
        List of lists with hyperparameter values.
    """

    fold_scores = []

    print("*"*80)
    print(
        f"-- -- Training model with hyperparameter combination {comb_idx}")
    print("*"*80)

    # Set training parameters
    (ntopics, alpha, opt_int) = grid_params_iter_lst[comb_idx]
    training_params['ntopics'] = ntopics
    training_params['alpha'] = alpha
    training_params['optimize_interval'] = opt_int

    for j, (train_index, test_index) in enumerate(rkf.split(corpus_train)):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        print("#"*80)
        print(
            f"Fold {j} of {rkf.get_n_splits()} with hyperparameter combination {comb_idx}")
        print("#"*80)

        # Create training/test corpus
        corpus_train_copy = corpus_train.copy()
        X_train, X_test = corpus_train_copy.iloc[train_index], corpus_train_copy.iloc[test_index]
        corpus_train_file = \
            corpusFile.parent.joinpath(f'corpus_train_{comb_idx}_{j}.txt')
        corpus_df_to_mallet(X_train, corpus_train_file)
        X_test['text'] = X_test['text'].apply(lambda x: x.split())

        # Train model
        tm_wrapper = TMWrapper()
        name = f"ntopics_{ntopics}_alpha_{alpha}_optint_{opt_int}_fold_{j}"
        model_path = tm_wrapper.train_root_model(
            models_folder=models_folder,
            name=name,
            path_corpus=corpus_train_file,
            trainer=trainer,
            training_params=training_params,
        )

        # Calculate coherence score with test corpus
        tm = TMmodel(model_path.joinpath("TMmodel"))
        cohr = tm.calculate_topic_coherence(
            metrics=["c_npmi"],
            reference_text=X_test.text.values.tolist(),
            aggregated=True,
        )

        save_values_to_file(model_path.joinpath("TMmodel").joinpath(
            "fold_config.txt"), grid_params_iter_lst[comb_idx] + (cohr,))

        fold_scores.append(cohr)

        # Delete train file
        corpus_train_file.unlink()

    return (comb_idx,  grid_params_iter_lst[comb_idx], fold_scores)


def run_k_fold(models_folder: str,
               trainer: str,
               corpusFile: str,
               grid_params: dict,
               training_params: dict,
               val_size: int = 0.3) -> None:
    """Runs repeated k-fold cross-validation for a given corpus and hyperparameter grid.

    Parameters
    ----------
    models_folder : str
        Path to folder where models will be saved.
    trainer : str
        Trainer to use for training the model.
    corpusFile : str
        Path to corpus file.
    grid_params : dict
        Dictionary with hyperparameter grid.
    training_params: dict
        Dictionary with training parameters.
    val_size : int, optional
        Size of the validation set, by default 0.3.
    """

    # Read corpus as df
    corpusFile = pathlib.Path(corpusFile)
    corpus_df = mallet_corpus_to_df(corpusFile)

    # Create validation corpus and save it to folder with original corpus
    print("-- -- Creating validation corpus")
    corpus_train, corpus_val = train_test_split(
        corpus_df, test_size=val_size, random_state=42)
    corpus_train.reset_index(drop=True, inplace=True)
    outFile = corpusFile.parent.joinpath('corpus_val.txt')
    corpus_df_to_mallet(corpus_val, outFile)

    # Initialize the RepeatedKFold cross-validation object:
    rkf = RepeatedKFold(n_splits=10, n_repeats=6, random_state=42)

    # Iterate over the hyperparameters and perform cross-validation:
    print("-- -- Validation starts...")

    # Calculating list for distribution
    grid_params_iter_lst = list(itertools.product(*grid_params))
    inputs = range(0, len(grid_params_iter_lst))

    starTime = timeit.default_timer()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Only the root process executes these tasks
    if rank == 0:
        chunks = [[] for _ in range(size)]
        for i, chunk in enumerate(inputs):
            chunks[i % size].append(chunk)
            timeFlag = datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        chunks = None
        timeFlag = None

    # Scatter the data to all processes
    data = comm.scatter(chunks, root=0)

    print('Rank %s recibed %s out of %s' %
          (rank, len(data), len(grid_params_iter_lst)))
    
    results_all_hypers = [
        train_hyper_comb(comb_idx=d,
                     grid_params_iter_lst=grid_params_iter_lst,
                     rkf=rkf,
                     corpus_train=corpus_train,
                     training_params=training_params,
                     corpusFile=corpusFile,
                     models_folder=models_folder,
                     trainer=trainer) 
        for d in data
    ]
    
    results_all_hypers = comm.gather(results_all_hypers, root=0)
    print(results_all_hypers)
    
    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/S2CS-AI/models_preproc/iter_0/corpus.txt",
                        help="Path to the training data.")
    parser.add_argument('--models_folder', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/S2CS-AI/models_val_mallet",
                        help="Path where the models are going to be saved.")
    parser.add_argument('--trainer', type=str,
                        default="mallet",
                        help="Name of the underlying topic modeling algorithm to be used: mallet|ctm")
    args = parser.parse_args()

    # Read training_params
    config_file = os.path.dirname(os.path.dirname(os.getcwd()))
    if config_file.endswith("UserInLoopHTM"):
        config_file = os.path.join(
            config_file,
            'experiments',
            'config',
            'dft_params.cf',
        )
    else:
        config_file = os.path.join(
            config_file,
            'UserInLoopHTM',
            'experiments',
            'config',
            'dft_params.cf',
        )
    training_params = read_config_experiments(config_file)

    grid_params = [
        [5],
        [0.1, 0.5],
        [0]
    ]

    run_k_fold(
        models_folder=args.models_folder,
        trainer=args.trainer,
        corpusFile=args.path_corpus,
        grid_params=grid_params,
        training_params=training_params)


if __name__ == '__main__':
    main()
