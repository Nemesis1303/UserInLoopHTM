import argparse
import datetime as DT
import json
import logging
import os
import pathlib
import shutil
import sys
import time
import multiprocessing as mp
import numpy as np
from code.topicmodeling.topicmodeling import CTMTrainer, HierarchicalTMManager, MalletTrainer

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
    "mallet_path": "/workspace/src/topicmodeling/models/mallet-2.0.8/bin/mallet",
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

        fields = ["n_components",
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


def train_model(train_config,
                corpusFile,
                modelFolder,
                embeddingsFile=None,
                train_config_child=None,
                train_config_father=None):
    """Train a model based on train_config, using corpusFile and embeddingsFile"""
    trainer = train_config["trainer"]
    TMparam = train_config["TMparam"]

    if trainer == 'ctm':
        trainer_obj = CTMTrainer(**TMparam)
    elif trainer == 'mallet':
        trainer_obj = MalletTrainer(**TMparam)

    if train_config['hierarchy-level'] == 1:

        tMmodel_path = train_config_father.parent.joinpath('TMmodel')
        if not os.path.isdir(tMmodel_path):
            sys.exit(
                'There must exist a valid TMmodel folder for the parent corpus')
        # Create hierarhicalTMManager object
        hierarchicalTMManager = HierarchicalTMManager()

        # Create corpus
        hierarchicalTMManager.create_submodel_tr_corpus(
            tMmodel_path, train_config_father.as_posix(), train_config_child.as_posix())

        if trainer == 'ctm':
            corpusFile = train_config_child.parent.joinpath('corpus.parquet')
            embeddingsFile = train_config_child.parent.joinpath(
                'embeddings.npy')
            trainer_obj.fit(corpusFile=corpusFile,
                            modelFolder=modelFolder,
                            embeddingsFile=embeddingsFile)
        elif trainer == 'mallet':
            corpusFile = train_config_child.parent.joinpath('corpus.txt')
            trainer_obj.fit(corpusFile=corpusFile,
                            modelFolder=modelFolder)

    else:
        trainer_obj.fit(corpusFile=corpusFile, modelFolder=modelFolder)


def train_automatic(path_corpus: str,
                    models_folder: str,
                    trainer: str):

    # Get training corpus (already preprocessed)
    corpusFile = pathlib.Path(path_corpus)
    print(corpusFile)
    if not corpusFile.is_dir() and not corpusFile.is_file:
        sys.exit(
            "The provided corpus file does not exist.")

    # Generate root model
    print("#############################")
    print("Generating root model")

    # Create folder for saving root model's outputs
    model_path = pathlib.Path(models_folder).joinpath(
        f"root_model_{DT.datetime.now().strftime('%Y%m%d')}")

    if model_path.exists():
        # Remove current backup folder, if it exists
        old_model_dir = pathlib.Path(str(model_path) + '_old/')
        if old_model_dir.exists():
            shutil.rmtree(old_model_dir)

        # Copy current model folder to the backup folder.
        shutil.move(model_path, old_model_dir)
        print(f'-- -- Creating backup of existing model in {old_model_dir}')

    model_path.mkdir(parents=True, exist_ok=True)

    # Train root model
    train_config = get_model_config(
        trainer=trainer,
        TMparam=training_params,
        hierarchy_level=0,
        htm_version=None,
        expansion_tpc=None,
        thr=None)

    config_file = model_path.joinpath("config.json")
    with config_file.open("w", encoding="utf-8") as fout:
        json.dump(train_config, fout, ensure_ascii=False,
                  indent=2, default=str)

    t_start = time.perf_counter()
    train_model(train_config, corpusFile, model_path)
    t_end = time.perf_counter()

    t_total = t_end - t_start
    logger.info(f"Total training time root model --> {t_total}")

    # Generate submodels
    print("#############################")
    print("Generating submodels")

    # Save father's config file
    config_file_parent = config_file

    # Train submodels
    num_topics_sub = [6, 8, 10]
    for j in num_topics_sub:
        for i in range(ntopics):
            for version in ["htm-ws", "htm-ds"]:

                if version == "htm-ws":
                    print("Generating submodel with HTM-WS")

                    # Create folder for saving node's outputs
                    model_path = pathlib.Path(models_folder).joinpath(
                        f"submodel_{version}_from_topic_{str(i)}_train_with_{str(j)}_{DT.datetime.now().strftime('%Y%m%d')}")

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

                    training_params["n_components"] = j
                    train_config = get_model_config(
                        trainer=trainer,
                        TMparam=training_params,
                        hierarchy_level=1,
                        htm_version=version,
                        expansion_tpc=i,
                        thr=None)

                    config_file = model_path.joinpath("config.json")
                    with config_file.open("w", encoding="utf-8") as fout:
                        json.dump(train_config, fout, ensure_ascii=False,
                                  indent=2, default=str)

                    t_start = time.perf_counter()
                    train_model(train_config=train_config,
                                corpusFile=corpusFile,
                                modelFolder=model_path,
                                train_config_child=config_file,
                                train_config_father=config_file_parent)
                    t_end = time.perf_counter()

                    t_total = t_end - t_start
                    logger.info(
                        f"Total training {model_path.as_posix()} --> {t_total}")

                else:
                    print("Generating submodel with HTM-DS")
                    for thr in np.arange(0.1, 0.8, 0.1):
                        # Create folder for saving node's outputs
                        thr_f = "{:.1f}".format(thr)
                        model_path = pathlib.Path(models_folder).joinpath(
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

                        config_file = model_path.joinpath("config.json")
                        with config_file.open("w", encoding="utf-8") as fout:
                            json.dump(train_config, fout, ensure_ascii=False,
                                      indent=2, default=str)

                        t_start = time.perf_counter()

                        train_model(train_config=train_config,
                                    corpusFile=corpusFile,
                                    modelFolder=model_path,
                                    train_config_child=config_file,
                                    train_config_father=config_file_parent)

                        t_end = time.perf_counter()

                        t_total = t_end - t_start
                        logger.info(
                            f"Total training {model_path.as_posix()} --> {t_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/workspace/data/training_data/corpus.parquet",
                        help="Path to the training data.")
    parser.add_argument('--models_folder', type=str,
                        default="/workspace/trial",
                        help="Path where the models are going to be saved.")
    parser.add_argument('--trainer', type=str,
                        default="ctm",
                        help="Name of the underlying topic modeling algorithm to be used: mallet|ctm")
    args = parser.parse_args()

    train_automatic(path_corpus=args.path_corpus,
                    models_folder=args.models_folder,
                    trainer=args.trainer)


if __name__ == "__main__":
    main()
