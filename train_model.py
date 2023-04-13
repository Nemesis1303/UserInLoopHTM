import argparse
import json
import logging
import pathlib
import sys
import time
import multiprocessing as mp
from subprocess import check_output
#from src.topicmodeling.topicmodeling import CTMTrainer, HierarchicalTMManager, MalletTrainer

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

############### TRAINING PARAMS ############
# Fixed params
ntopics = 5
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
    "num_iterations": 500,
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
                    trainer: str):

    # Get training corpus (already preprocessed)
    corpusFile = pathlib.Path(path_corpus)
    print(corpusFile)
    if not corpusFile.is_dir() and not corpusFile.is_file:
        sys.exit(
            "The provided corpus file does not exist.")
        
    # Train root model
    train_config = get_model_config(
        trainer=trainer,
        TMparam=training_params,
        hierarchy_level=0,
        htm_version=None,
        expansion_tpc=None,
        thr=None)
    
    configFile = corpusFile.parent.joinpath("trainconfig.json")
    if configFile.is_file():
        with configFile.open('r', encoding='utf8') as fin:
            train_config_txt = json.load(fin)
            train_config_txt["TMparam"] = train_config["TMparam"]
        with configFile.open("w", encoding="utf-8") as fout:
            json.dump(train_config_txt, fout, ensure_ascii=False,
                  indent=2, default=str)

    t_start = time.perf_counter()
    #train_model(train_config, corpusFile, model_path)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus.txt",
                        help="Path to the training data.")
    parser.add_argument('--models_folder', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0",
                        help="Path where the models are going to be saved.")
    parser.add_argument('--trainer', type=str,
                        default="mallet",
                        help="Name of the underlying topic modeling algorithm to be used: mallet|ctm")
    args = parser.parse_args()
    import pdb; pdb.set_trace()
    train_automatic(path_corpus=args.path_corpus,
                    models_folder=args.models_folder,
                    trainer=args.trainer)


if __name__ == "__main__":
    main()


#--path_corpus /export/usuarios_ml4ds/lbartolome/Datasets/S2CS/models_preproc_ctm/iter_0/corpus.parquet --models_folder /export/usuarios_ml4ds/lbartolome/Datasets/S2CS/models_preproc_ctm --trainer ctm

#/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/cordis_612.parquet