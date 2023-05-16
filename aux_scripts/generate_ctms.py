import argparse
import datetime as DT
import itertools
import json
import logging
import multiprocessing as mp
import pathlib
import shutil
import sys
import time
from subprocess import check_output

import numpy as np
import os


################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################


def get_model_config(TMparam,
                     hierarchy_level=0,
                     htm_version=None,
                     expansion_tpc=None,
                     thr=None):
    """Select model configuration based on trainer"""

    params = {"trainer": "ctm",
              "TMparam": TMparam,
              "hierarchy-level": hierarchy_level,
              "htm-version": htm_version,
              "expansion_tpc": expansion_tpc,
              "thr": thr}

    return params


def train_automatic(path_corpus: str,
                    models_folder: str,
                    grid,
                    training_params: dict, 
                    path_tm: str):
    """Train a set of models with different dropout values.
    """

    # Create folder for saving model
    for drop_in, drop_out in grid:
        model_path = pathlib.Path(models_folder).joinpath(
            f"root_model_{str(drop_in)}_{str(drop_out)}_{DT.datetime.now().strftime('%Y%m%d')}")

        if model_path.exists():
            # Remove current backup folder, if it exists
            old_model_dir = pathlib.Path(str(model_path) + '_old/')
            if old_model_dir.exists():
                shutil.rmtree(old_model_dir)

            # Copy current model folder to the backup folder.
            shutil.move(model_path, old_model_dir)
            logger.info(
                f'-- -- Creating backup of existing model in {old_model_dir}')

        model_path.mkdir(parents=True, exist_ok=True)

        # Copy training corpus (already preprocessed) to folder
        corpusFile = pathlib.Path(path_corpus)
        if not corpusFile.is_dir() and not corpusFile.is_file:
            sys.exit(
                "The provided corpus file does not exist.")

        if corpusFile.is_dir():
            logger.info(f'-- -- Copying corpus.parquet.')
            dest = shutil.copytree(
                corpusFile, model_path.joinpath("corpus.parquet"))
        else:
            dest = shutil.copy(corpusFile, model_path)
        logger.info(f'-- -- Corpus file copied in {dest}')

        logger.info(
            f"-- Training model with dropout_in={drop_in} and dropout_out={drop_out} starts...")

        # Generate training config
        training_params["dropout_in"] = drop_in
        training_params["dropout_out"] = drop_out

        train_config = get_model_config(
            TMparam=training_params,
            hierarchy_level=0,
            htm_version=None,
            expansion_tpc=None,
            thr=None)

        configFile = model_path.joinpath("config.json")
        with configFile.open("w", encoding="utf-8") as fout:
            json.dump(train_config, fout, ensure_ascii=False,
                      indent=2, default=str)

        topicmodeling_path = os.path.join(path_tm, 'UserInLoopHTM', 'src', 'topicmodeling', 'topicmodeling.py')
        t_start = time.perf_counter()
        cmd = f'python {topicmodeling_path} --train --config {configFile.as_posix()}'
        logger.info(cmd)
        try:
            logger.info(f'-- -- Running command {cmd}')
            output = check_output(args=cmd, shell=True)
        except:
            logger.error('-- -- Command execution failed')
        t_end = time.perf_counter()

        t_total = t_end - t_start
        logger.info(f"Total training time root model --> {t_total}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus.txt",
                        help="Path to the training data.")
    parser.add_argument('--models_folder', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_ctm_eval",
                        help="Path where the models are going to be saved.")
    parser.add_argument('--ntopics', type=int,
                        default=5,
                        help="Iter number to start the naming of the root models.")
    args = parser.parse_args()

    ############### TRAINING PARAMS ############
    # Fixed params
    ntopics = args.ntopics
    numepochs = 100
    dropout_en = np.arange(0.0, 0.8, 0.1)
    dropout_dec = np.arange(0.0, 0.8, 0.1)
    grid_dropout = itertools.product(dropout_en, dropout_dec)
    training_params = {
        "ntopics": ntopics,
        "thetas_thr": 0.003,
        "labels": "",
        "model_type": "prodLDA",
        "ctm_model_type": "CombinedTM",
        "hidden_sizes": (50, 50),
        "activation": "softplus",
        "dropout_in": dropout_en,
        "dropout_out": dropout_dec,
        "learn_priors": True,
        "lr": 2e-3,
        "momentum": 0.99,
        "solver": "adam",
        "num_epochs": numepochs,
        "reduce_on_plateau": False,
        "batch_size": 64,
        "topic_prior_mean": 0.0,
        "topic_prior_variance": None,
        "num_samples": 20,
        "num_data_loader_workers": mp.cpu_count(),
    }
    
    path_topic_modeler = \
        os.path.dirname(os.path.dirname(os.getcwd()))

    train_automatic(path_corpus=args.path_corpus,
                    models_folder=args.models_folder,
                    grid=grid_dropout,
                    training_params=training_params,
                    path_tm=path_topic_modeler)
    
if __name__ == "__main__":
    main()