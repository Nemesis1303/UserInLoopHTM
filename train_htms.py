import argparse
import datetime as DT
import json
import logging
import multiprocessing as mp
import pathlib
import shutil
import sys
import time
from subprocess import check_output

import numpy as np

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

############### TRAINING PARAMS ############
# Fixed params
ntopics = 10
training_params = {
    "activation": "softplus",
    "alpha": 5.0,
    "batch_size": 64,
    "doc_topic_thr": 0.0,
    "dropout": 0.2,
    "hidden_sizes": (50, 50),
    "labels": "",
    "learn_priors": True,
    "lr": 2e-3,
    "momentum": 0.99,
    "num_data_loader_workers": mp.cpu_count(),
    "num_threads": 4,
    "optimize_interval": 10,
    "reduce_on_plateau": False,
    "sbert_model_to_load": "paraphrase-distilroberta-base-v1",
    "solver": "adam",
    "thetas_thr": 0.003,
    "topic_prior_mean": 0.0,
    "topic_prior_variance": None,
    "ctm_model_type": "CombinedTM",
    "model_type": "prodLDA",
    "n_components": ntopics,
    "ntopics": ntopics,
    "num_epochs": 100,
    "num_samples": 20,
    "doc_topic_thr": 0.0,
    "mallet_path": "/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/src/topicmodeling/mallet-2.0.8/bin/mallet",
    "thetas_thr": 0.003,
    "token_regexp": "[\\p{L}\\p{N}][\\p{L}\\p{N}\\p{P}]*\\p{L}",
    "alpha": 5.0,
    "num_threads": 4,
    "num_iterations": 1000,
}
############################################


def get_model_config(trainer,
                     TMparam,
                     hierarchy_level,
                     htm_version,
                     expansion_tpc,
                     thr):
    """Select model configuration based on trainer"""

    if trainer == 'mallet':
        fields = ["ntopics",
                  "labels",
                  "thetas_thr",
                  "mallet_path",
                  "alpha",
                  "optimize_interval",
                  "num_threads",
                  "num_iterations",
                  "doc_topic_thr",
                  "token_regexp"]
    elif trainer == 'ctm':

        fields = ["ntopics",
                  "thetas_thr",
                  "labels",
                  "model_type",
                  "ctm_model_type",
                  "hidden_sizes",
                  "activation",
                  "dropout",
                  "learn_priors",
                  "lr",
                  "momentum",
                  "solver",
                  "num_epochs",
                  "reduce_on_plateau",
                  "batch_size",
                  "topic_prior_mean",
                  "topic_prior_variance",
                  "num_samples",
                  "num_data_loader_workers"]

    params = {"trainer": trainer,
              "TMparam": {t: TMparam[t] for t in fields},
              "hierarchy-level": hierarchy_level,
              "htm-version": htm_version,
              "expansion_tpc": expansion_tpc,
              "thr": thr}

    return params


def train_automatic(path_corpus: str,
                    models_folder: str,
                    trainer: str,
                    iters: int,
                    start: int):

    for iter_ in range(iters):
        iter_ += start
        logger.info(f'-- -- Running iter {iter_}')

        # Create folder for saving HTM (root models and its descendents)
        model_path = pathlib.Path(models_folder).joinpath(
            f"root_model_{str(iter_)}_{DT.datetime.now().strftime('%Y%m%d')}")

        if model_path.exists():
            # Remove current backup folder, if it exists
            old_model_dir = pathlib.Path(str(model_path) + '_old/')
            if old_model_dir.exists():
                shutil.rmtree(old_model_dir)

            # Copy current model folder to the backup folder.
            shutil.move(model_path, old_model_dir)
            print(
                f'-- -- Creating backup of existing model in {old_model_dir}')

        model_path.mkdir(parents=True, exist_ok=True)

        # Copy training corpus (already preprocessed) to HTM folder (root)
        corpusFile = pathlib.Path(path_corpus)
        print(corpusFile)
        if not corpusFile.is_dir() and not corpusFile.is_file:
            sys.exit(
                "The provided corpus file does not exist.")

        if corpusFile.is_dir():
            print(f'-- -- Copying corpus.parquet.')
            dest = shutil.copytree(
                corpusFile, model_path.joinpath("corpus.parquet"))
        else:
            dest = shutil.copy(corpusFile, model_path)
        print(f'-- -- Corpus file copied in {dest}')

        # Generate root model
        print("#############################")
        print("Generating root model")

        # Train root model
        train_config = get_model_config(
            trainer=trainer,
            TMparam=training_params,
            hierarchy_level=0,
            htm_version=None,
            expansion_tpc=None,
            thr=None)

        configFile = model_path.joinpath("config.json")
        with configFile.open("w", encoding="utf-8") as fout:
            json.dump(train_config, fout, ensure_ascii=False,
                      indent=2, default=str)

        t_start = time.perf_counter()
        cmd = f'python src/topicmodeling/topicmodeling.py --train --config {configFile.as_posix()}'
        print(cmd)
        try:
            logger.info(f'-- -- Running command {cmd}')
            output = check_output(args=cmd, shell=True)
        except:
            logger.error('-- -- Command execution failed')
        t_end = time.perf_counter()

        t_total = t_end - t_start
        logger.info(f"Total training time root model --> {t_total}")

        # Generate submodels
        print("#############################")
        print("Generating submodels")

        # Save father's config file
        configFile_parent = configFile
        model_path_parent = model_path

        # Train submodels
        num_topics_sub = [6, 8, 10]
        for j in num_topics_sub:
            for i in range(ntopics):
                for version in ["htm-ws", "htm-ds"]:

                    if version == "htm-ws":
                        print("Generating submodel with HTM-WS")

                        # Create folder for saving node's outputs
                        model_path = pathlib.Path(model_path_parent).joinpath(
                            f"submodel_{version}_from_topic_{str(i)}_train_with_{str(j)}_{DT.datetime.now().strftime('%Y%m%d')}")

                        if model_path.exists():
                            # Remove current backup folder, if it exists
                            old_model_dir = pathlib.Path(
                                str(model_path) + '_old/')
                            if old_model_dir.exists():
                                shutil.rmtree(old_model_dir)

                            # Copy current model folder to the backup folder.
                            shutil.move(model_path, old_model_dir)
                            print(
                                f'-- -- Creating backup of existing model in {old_model_dir}')

                        model_path.mkdir(parents=True, exist_ok=True)

                        training_params["n_components"] = j
                        train_config = get_model_config(
                            trainer=trainer,
                            TMparam=training_params,
                            hierarchy_level=1,
                            htm_version=version,
                            expansion_tpc=i,
                            thr=None)

                        configFile = model_path.joinpath("config.json")
                        with configFile.open("w", encoding="utf-8") as fout:
                            json.dump(train_config, fout, ensure_ascii=False,
                                      indent=2, default=str)

                        t_start = time.perf_counter()

                        # Create submodel training corpus
                        cmd = f'python src/topicmodeling/topicmodeling.py --hierarchical --config {configFile_parent.as_posix()} --config_child {configFile.as_posix()}'
                        print(cmd)
                        try:
                            logger.info(f'-- -- Running command {cmd}')
                            output = check_output(args=cmd, shell=True)
                        except:
                            logger.error('-- -- Command execution failed')

                        # Train submodel
                        cmd = f'python src/topicmodeling/topicmodeling.py --train --config {configFile.as_posix()}'
                        print(cmd)
                        try:
                            logger.info(f'-- -- Running command {cmd}')
                            output = check_output(args=cmd, shell=True)
                        except:
                            logger.error('-- -- Command execution failed')

                        t_end = time.perf_counter()

                        t_total = t_end - t_start
                        logger.info(
                            f"Total training {model_path.as_posix()} --> {t_total}")

                    else:
                        print("Generating submodel with HTM-DS")
                        for thr in np.arange(0.1, 1, 0.1):
                            # Create folder for saving node's outputs
                            thr_f = "{:.1f}".format(thr)
                            model_path = pathlib.Path(model_path_parent).joinpath(
                                f"submodel_{version}_thr_{thr_f}_from_topic_{str(i)}_train_with_{str(j)}_{DT.datetime.now().strftime('%Y%m%d')}")

                            if model_path.exists():
                                # Remove current backup folder, if it exists
                                old_model_dir = pathlib.Path(
                                    str(model_path) + '_old/')
                                if old_model_dir.exists():
                                    shutil.rmtree(old_model_dir)

                                # Copy current model folder to the backup folder.
                                shutil.move(model_path, old_model_dir)
                                print(
                                    f'-- -- Creating backup of existing model in {old_model_dir}')

                            model_path.mkdir(parents=True, exist_ok=True)

                            training_params["n_components"] = j
                            train_config = get_model_config(
                                trainer=trainer,
                                TMparam=training_params,
                                hierarchy_level=1,
                                htm_version=version,
                                expansion_tpc=i,
                                thr=thr)

                            configFile = model_path.joinpath("config.json")
                            with configFile.open("w", encoding="utf-8") as fout:
                                json.dump(train_config, fout, ensure_ascii=False,
                                          indent=2, default=str)

                            t_start = time.perf_counter()

                            # Create submodel training corpus
                            cmd = f'python src/topicmodeling/topicmodeling.py --hierarchical --config {configFile_parent.as_posix()} --config_child {configFile.as_posix()}'
                            print(cmd)
                            try:
                                logger.info(f'-- -- Running command {cmd}')
                                output = check_output(args=cmd, shell=True)
                            except:
                                logger.error('-- -- Command execution failed')

                            # Train submodel
                            cmd = f'python src/topicmodeling/topicmodeling.py --train --config {configFile.as_posix()}'
                            print(cmd)
                            try:
                                logger.info(f'-- -- Running command {cmd}')
                                output = check_output(args=cmd, shell=True)
                            except:
                                logger.error('-- -- Command execution failed')

                            t_end = time.perf_counter()

                            t_total = t_end - t_start
                            logger.info(
                                f"Total training {model_path.as_posix()} --> {t_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus.txt",
                        help="Path to the training data.")
    parser.add_argument('--models_folder', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_htm",
                        help="Path where the models are going to be saved.")
    parser.add_argument('--trainer', type=str,
                        default="mallet",
                        help="Name of the underlying topic modeling algorithm to be used: mallet|ctm")
    parser.add_argument('--iters', type=int,
                        default=1,
                        help="Number of iteration to create htms from the same corpus")
    parser.add_argument('--start', type=int,
                        default=0,
                        help="Iter number to start the naming of the root models.")
    args = parser.parse_args()

    train_automatic(path_corpus=args.path_corpus,
                    models_folder=args.models_folder,
                    trainer=args.trainer,
                    iters=args.iters,
                    start=args.start)


if __name__ == "__main__":
    main()

# --path_corpus /export/usuarios_ml4ds/lbartolome/Datasets/S2CS/models_preproc_mallet/iter_0/corpus.txt --models_folder /export/usuarios_ml4ds/lbartolome/Datasets/S2CS/models_htm --iters 2 --start 0 --> hator05

# --path_corpus /export/usuarios_ml4ds/lbartolome/Datasets/S2CS/models_preproc_mallet/iter_0/corpus.txt --models_folder /export/usuarios_ml4ds/lbartolome/Datasets/S2CS/models_htm --iters 2 --start 2 --> hator00

# -----

# --path_corpus /export/usuarios_ml4ds/lbartolome/Datasets/S2CS/models_preproc_ctm/iter_0/corpus.parquet --models_folder /export/usuarios_ml4ds/lbartolome/Datasets/S2CS/models_htm_ctm --trainer ctm --iters 5 --start 0 --> kumo
