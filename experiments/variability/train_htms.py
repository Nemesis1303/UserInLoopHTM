import argparse
import datetime as DT
import logging
import multiprocessing as mp
import pathlib
import sys
from distutils.dir_util import copy_tree

import numpy as np

# Add src to path and make imports
sys.path.append('../..')
from src.tmWrapper.tm_wrapper import TMWrapper

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

# ======================================================
# Default training parameters
# ======================================================
training_params = {
    "activation": "softplus",
    "batch_size": 64,
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
    "ntopics": 10,
    "num_epochs": 100,
    "num_samples": 20,
    "doc_topic_thr": 0.0,
    "mallet_path": "/export/usuarios_ml4ds/lbartolome/mallet-2.0.8/bin/mallet",
    "thetas_thr": 0.003,
    "token_regexp": "[\\p{L}\\p{N}][\\p{L}\\p{N}\\p{P}]*\\p{L}",
    "alpha": 5.0,
    "num_threads": 4,
    "num_iterations": 1000,
}


def train_automatic(path_corpus: str,
                    models_folder: str,
                    trainer: str,
                    iters: int,
                    start: int,
                    model_path: str = None,
                    only_root: bool = False,
                    ntopics_root: int = 10):

    tm_wrapper = TMWrapper()

    for iter_ in range(iters):
        iter_ += start
        logger.info(f'-- -- Running iter {iter_}')

        if model_path is None:

            name = f"root_model_{str(iter_)}_{DT.datetime.now().strftime('%Y%m%d')}"
            model_path = tm_wrapper.train_root_model(
                models_folder=models_folder,
                name=name,
                path_corpus=path_corpus,
                trainer=trainer,
                training_params=training_params,
            )
        else:
            # If model_path exists we already have the root model
            old_model_path = pathlib.Path(model_path)

            # Create folder for saving HTM (root models and its descendents)
            model_path = pathlib.Path(models_folder).joinpath(
                f"root_model_{str(iter_)}_{DT.datetime.now().strftime('%Y%m%d')}")

            logger.info(
                '-- -- Copying existing root model into {model_path.as_posix()}')
            copy_tree(old_model_path.as_posix(), model_path.as_posix())

            configFile = model_path.joinpath("config.json")

        if not only_root:
            # Generate submodels
            print("#############################")
            print("Generating submodels")

            # Save father's model path
            model_path_parent = model_path

            # Train submodels
            num_topics_sub = [6, 8, 10]
            for j in num_topics_sub:
                training_params["n_components"] = j
                for i in range(ntopics_root):
                    for version in ["htm-ws", "htm-ds"]:

                        if version == "htm-ws":
                            print("Generating submodel with HTM-WS")

                            name = f"submodel_{version}_from_topic_{str(i)}_train_with_{str(j)}_{DT.datetime.now().strftime('%Y%m%d')}"

                            tm_wrapper.train_htm_submodel(
                                version=version,
                                father_model_path=model_path_parent,
                                name=name,
                                trainer=trainer,
                                training_params=training_params,
                                expansion_topic=i,
                                thr=None
                            )

                        else:
                            print("Generating submodel with HTM-DS")
                            for thr in np.arange(0.1, 1, 0.1):

                                thr_f = "{:.1f}".format(thr)
                                name = f"submodel_{version}_thr_{thr_f}_from_topic_{str(i)}_train_with_{str(j)}_{DT.datetime.now().strftime('%Y%m%d')}"

                                tm_wrapper.train_htm_submodel(
                                    version=version,
                                    father_model_path=model_path_parent,
                                    name=name,
                                    trainer=trainer,
                                    training_params=training_params,
                                    expansion_topic=i,
                                    thr=thr
                                )


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
    parser.add_argument('--model_path', type=str,
                        help="Path to the root model if it exists.")
    parser.add_argument('--only_root', default=False, required=False,
                        action='store_true', help="Flag to activate training of only one root model")
    args = parser.parse_args()

    train_automatic(path_corpus=args.path_corpus,
                    models_folder=args.models_folder,
                    trainer=args.trainer,
                    iters=args.iters,
                    start=args.start,
                    model_path=args.model_path,
                    only_root=args.only_root)


if __name__ == "__main__":
    main()
