import argparse
import datetime as DT
import logging
import os
import pathlib
import shutil
import sys
from distutils.dir_util import copy_tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path and make imports
sys.path.append('../..')
from src.utils.misc import (
    corpus_df_to_mallet, mallet_corpus_to_df, read_config_experiments)
from src.tmWrapper.tm_wrapper import TMWrapper

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

#######################
# Auxiliary functions #
#######################
def save_values_to_file(filename, values):
    with open(filename, 'w') as file:
        for value in values:
            file.write(str(value) + '\n')


def train_automatic(path_corpus: str,
                    path_ref_corpus: str,
                    models_folder: str,
                    trainer: str,
                    iters: int,
                    start: int,
                    training_params: dict,
                    num_topics_sub: str,
                    ntopics_root: int = 10,
                    model_path: str = None,
                    only_root: bool = False,
                    htm_ws: bool = False,
                    htm_ds: bool = False,
                    generate_val: bool = False,
                    use_same_model_path: bool = False,
                    val_size: int = 0.3):

    ############################
    ## Get train / val corpus ##
    ############################
    logger.info(f"{'-'*100}")
    logger.info(f"-- -- Getting train / val corpus...")
    logger.info(f"{'-'*100}")

    if not path_ref_corpus and not generate_val:
        # If no reference corpus is provided, we use the original corpus as the reference corpus (not the best option for coherence comparison, but it works)
        path_ref_corpus = path_corpus
    else:
        # Read whole corpus as df
        path_corpus = pathlib.Path(path_corpus)
        corpus_df_train = None

        if trainer == "mallet":
            corpus_df = mallet_corpus_to_df(path_corpus)
        elif trainer == "ctm":
            corpus_df = pd.read_parquet(path_corpus)
            corpus_df['id'] = corpus_df['id'].astype(str)
            corpus_df.rename(columns={'bow_text': 'text'}, inplace=True)

        if path_ref_corpus:
            # If a reference corpus is provided, we use it to create the train and validation corpus
            logger.info(
                "-- -- Validation corpus provided. Extracting train corpus from original corpus...")
            corpus_df_val = mallet_corpus_to_df(pathlib.Path(path_ref_corpus))
            merged_df = corpus_df.merge(
                corpus_df_val, on="id", how="outer", indicator=True)
            corpus_df_train = corpus_df[merged_df["_merge"] == "left_only"]
        else:
            path_ref_corpus = path_corpus.parent.joinpath('corpus_val.txt')
            if generate_val and not path_ref_corpus.is_file():
                # If no reference corpus is provided but flag generate_val is True, we create a validation corpus from the original corpus via a train_test_split
                logger.info(
                    "-- -- Validation corpus not provided. Generating validation corpus from original corpus...")
                corpus_df_train, corpus_df_val = train_test_split(
                    corpus_df, test_size=val_size, random_state=42)
                corpus_df_train.reset_index(drop=True, inplace=True)
                logger.info(
                    f"-- -- Saving validation corpus in {path_ref_corpus.as_posix()}")
                corpus_df_to_mallet(corpus_df_val, path_ref_corpus)
            else:
                logger.info(
                    "-- -- Not generating validation corpus because it already exists...")
        if corpus_df_train is not None:
            if trainer == "mallet":
                path_corpus = path_corpus.parent.joinpath('corpus_train.txt')
                logger.info(
                    f"-- -- Saving train corpus in {path_corpus.as_posix()}")
                corpus_df_to_mallet(corpus_df_train, path_corpus)
            elif trainer == "ctm":
                corpus_df_train.rename(columns={'text': 'bow_text'}, inplace=True)
                logger.info(
                    f"-- -- Saving train corpus in {path_corpus.as_posix()}")
                path_corpus = path_corpus.parent.joinpath('corpus_train.parquet')
                corpus_df_train.to_parquet(path_corpus)

    logger.info(f"{'-'*100}")
    logger.info(f"-- -- Train / val corpus obtained")
    logger.info(f"{'-'*100}")

    tm_wrapper = TMWrapper()

    if model_path is None:
        ######################
        ## Train root model ##
        ######################
        name = f"root_model_{ntopics_root}_tpcs_{DT.datetime.now().strftime('%Y%m%d')}"

        logger.info(f"{'-'*100}")
        logger.info(f"-- -- Training root model {name}...")
        logger.info(f"{'-'*100}")

        training_params['ntopics'] = ntopics_root
        model_path = tm_wrapper.train_root_model(
            models_folder=models_folder,
            name=name,
            path_corpus=path_corpus,
            trainer=trainer,
            training_params=training_params,
        )

        # Recalculate coherence vs ref corpus
        logger.info(f"-- -- Recalculating coherence vs ref corpus")
        for metric in ["c_v", "c_npmi"]:
            tm_wrapper.calculate_cohr_vs_ref(
                model_path=model_path,
                corpus_val=pathlib.Path(path_ref_corpus),
                type=metric)
        logger.info(f"-- -- Calculating RBO and TD")
        tm_wrapper.calculate_rbo(model_path)
        tm_wrapper.calculate_td(model_path)

    elif use_same_model_path:
        logger.info(
            f"-- -- Using existing root model {model_path}")
    else:
        logger.info(
            f"-- -- Using copy of existing root model {model_path}")
        k = int(pathlib.Path(model_path).name.split("_")[2])

        # If model_path exists we already have the root model
        old_model_path = pathlib.Path(model_path)

        # Create folder for saving HTM (root models and its descendents)
        model_path = pathlib.Path(models_folder).joinpath(
            f"htm_{k}_tpcs_{DT.datetime.now().strftime('%Y%m%d')}")

        logger.info(
            f"-- -- Copying existing root model into {model_path.as_posix()}")
        copy_tree(old_model_path.as_posix(), model_path.as_posix())

    if not only_root:
        #######################
        ## Generate submodels #
        #######################
        # Save father's model path
        model_path_parent = model_path

        # Train submodels
        num_topics_sub = num_topics_sub.split(",")
        num_topics_sub = [int(x) for x in num_topics_sub]

        if htm_ws:
            #######################
            # Train HTM-WS models
            #######################
            version = "htm-ws"
            logger.info(f"{'-'*100}")
            logger.info(f"-- -- Generating submodel with HTM-WS")
            logger.info(f"{'-'*100}")

            for iter_ in range(iters):
                iter_ += start
                for exp_tpc in range(ntopics_root):
                    for tr_tpc in num_topics_sub:

                        training_params["ntopics"] = tr_tpc

                        name = f"submodel_{version}_from_topic_{str(exp_tpc)}_train_with_{str(tr_tpc)}_iter_{iter_}_{DT.datetime.now().strftime('%Y%m%d')}"

                        submodel_path = tm_wrapper.train_htm_submodel(
                            version=version,
                            father_model_path=model_path_parent,
                            name=name,
                            trainer=trainer,
                            training_params=training_params,
                            expansion_topic=exp_tpc,
                            thr=None
                        )

                        # Recalculate coherence vs ref corpus
                        logger.info(
                            f"-- -- Recalculating coherence vs ref corpus")
                        for metric in ["c_v", "c_npmi"]:
                            tm_wrapper.calculate_cohr_vs_ref(
                                model_path=submodel_path,
                                corpus_val=pathlib.Path(path_ref_corpus),
                                type=metric)
                        logger.info(
                            f"-- -- Calculating RBO and TD")
                        tm_wrapper.calculate_rbo(submodel_path)
                        tm_wrapper.calculate_td(submodel_path)
            # ----> END HTM-WS

        if htm_ds:
            #######################
            # Train HTM-WS models
            #######################
            version = "htm-ds"
            logger.info(f"{'-'*100}")
            logger.info(f"-- -- Generating submodel with HTM-DS")
            logger.info(f"{'-'*100}")
            for iter_ in range(iters):
                iter_ += start
                for exp_tpc in range(ntopics_root):
                    for tr_tpc in num_topics_sub:
                        for thr in np.arange(0.1, 1, 0.1):

                            try:
                                thr_f = "{:.1f}".format(thr)

                                name = f"submodel_{version}_thr_{thr_f}_from_topic_{str(exp_tpc)}_train_with_{str(tr_tpc)}_iter_{iter_}_{DT.datetime.now().strftime('%Y%m%d')}"

                                submodel_path = tm_wrapper.train_htm_submodel(
                                    version=version,
                                    father_model_path=model_path_parent,
                                    name=name,
                                    trainer=trainer,
                                    training_params=training_params,
                                    expansion_topic=exp_tpc,
                                    thr=thr
                                )

                                # Recalculate coherence vs ref corpus
                                logger.info(
                                    f"-- -- Recalculating coherence vs ref corpus")
                                for metric in ["c_v", "c_npmi"]:
                                    tm_wrapper.calculate_cohr_vs_ref(
                                        model_path=submodel_path,
                                        corpus_val=pathlib.Path(
                                            path_ref_corpus),
                                        type=metric)
                                logger.info(
                                    f"Calculating RBO and TD")
                                tm_wrapper.calculate_rbo(submodel_path)
                                tm_wrapper.calculate_td(submodel_path)
                            except:
                                logger.error(f"Error with thr {thr}")
                                this_submodel_path = pathlib.Path(
                                    model_path_parent).joinpath(name)
                                if this_submodel_path.is_dir():
                                    shutil.rmtree(this_submodel_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus.txt",
                        help="Path to the training data.")
    parser.add_argument('--path_val_corpus', type=str, default=None,
                        help="Path to the validation training data.")
    parser.add_argument('--models_folder', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models",
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
    parser.add_argument('--htm_ws', default=False, required=False,
                        action='store_true', help="Flag to activate training of htm-ws submodels")
    parser.add_argument('--htm_ds', default=False, required=False,
                        action='store_true', help="Flag to activate training of htm-ds submodels")
    parser.add_argument('--generate_val', default=False, required=False,
                        action='store_true', help="Flag to activate generation of validation corpus")
    parser.add_argument('--use_same_model_path', default=False, required=False,
                        action='store_true', help="Flag to activate usage of model_path as root model without copying it into a new folder.")    
    parser.add_argument('--ntopics_root', type=int, default=5,
                        help="Number of topics in the root model.")
    parser.add_argument('--num_topics_sub', type=str,
                        default="5,10,20,30,40,50,60,70,80,90,100",
                        help="Number of topics to train the submodeles with.")
    parser.add_argument('--val_size', type=float, default=0.3,
                        help="Size of the validation corpus if no validation corpus is provided.")
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

    train_automatic(path_corpus=args.path_corpus,
                    path_ref_corpus=args.path_val_corpus,
                    models_folder=args.models_folder,
                    trainer=args.trainer,
                    iters=args.iters,
                    start=args.start,
                    training_params=training_params,
                    num_topics_sub=args.num_topics_sub,
                    ntopics_root=args.ntopics_root,
                    model_path=args.model_path,
                    only_root=args.only_root,
                    htm_ws=args.htm_ws,
                    htm_ds=args.htm_ds,
                    generate_val=args.generate_val,
                    use_same_model_path=args.use_same_model_path,
                    val_size=args.val_size
                    )


if __name__ == "__main__":
    main()
