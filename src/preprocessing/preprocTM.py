"""
Carries out specific preprocessing for TM.
"""

import argparse
import datetime as DT
import multiprocessing as mp
import logging
import json
# import shutil
import sys
import time
import warnings
from getpass import getuser
from pathlib import Path
from subprocess import check_output

sys.path.insert(0, Path(__file__).parent.parent.resolve().as_posix())
warnings.filterwarnings(action="ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

def main(nw=0, iter_=0):
    
    # Create folder structure
    models = Path("/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc")
    models.mkdir(parents=True, exist_ok=True)

    Preproc = {
        "min_lemas": 15,
        "no_below": 15,
        "no_above": 0.4,
        "keep_n": 100000,
        "stopwords": [
          "/export/usuarios_ml4ds/lbartolome/hierarchical-topic-models/data/wordlists/english_generic.json",
          "/export/usuarios_ml4ds/lbartolome/hierarchical-topic-models/data/wordlists/S2_stopwords.json",
          "/export/usuarios_ml4ds/lbartolome/hierarchical-topic-models/data/wordlists/S2CS_stopwords.json"
        ],
        "equivalences": [
          "/export/usuarios_ml4ds/lbartolome/hierarchical-topic-models/data/wordlists/S2_equivalences.json",
          "/export/usuarios_ml4ds/lbartolome/hierarchical-topic-models/data/wordlists/S2CS_equivalences.json"
        ]
    }
    
    # Create model folder
    model_path = models.joinpath("iter_" + str(iter_))
    model_path.mkdir(parents=True, exist_ok=True)
    model_stats = model_path.joinpath("stats")
    model_stats.mkdir(parents=True, exist_ok=True)

    # Save dataset json file
    DtsetConfig = model_path.joinpath(Dtset+'.json')
    parquetFile = Path("/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/preproc_cordis_embeddings.parquet")
    Dtset = "CORDIS"
    TrDtset = {
        "name": "Cordis",
        "Dtsets": [
        {
            "parquet": parquetFile,
            "source": "Cordis",
            "idfld": "id",
            "lemmasfld": [
            "lemmas_with_grams"
            ],
        "filter": ""
        }
        ]
    }
    with DtsetConfig.open('w', encoding='utf-8') as outfile:
        json.dump(TrDtset, outfile,
                    ensure_ascii=False, indent=2, default=str)

    # Save configuration file
    configFile = model_path.joinpath("trainconfig.json")
    train_config = {
        "name": Dtset,
        "description": "",
        "visibility": "Public",
        "trainer": "mallet",
        "TrDtSet": DtsetConfig.resolve().as_posix(),
        "Preproc": Preproc,
        "TMparam": {},
        "creation_date": DT.datetime.now(),
        "hierarchy-level": 0,
        "htm-version": None,
    }
    with configFile.open('w', encoding='utf-8') as outfile:
        json.dump(train_config, outfile,
                    ensure_ascii=False, indent=2, default=str)

    # Execute command
    cmd = f"python src/topicmodeling/topicmodeling.py --preproc --config {configFile.resolve().as_posix()} --nw {str(nw)}"
    logger.info(f"Running command '{cmd}'")
    
    t_start = time.perf_counter()
    check_output(args=cmd, shell=True)
    t_end = time.perf_counter()
    t_total = t_end - t_start
        
    logger.info(f"Total time --> {t_total}")
    print("\n")
    print("-" * 100)
    print("\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocessing for TM')
    parser.add_argument('--nw', type=int, required=False, default=0,
                        help="Number of workers when preprocessing data with Dask. Use 0 to use Dask default")
    parser.add_argument('--iter_', type=int, required=False, default=0,
                        help="Preprocessing number of this file.")
    args = parser.parse_args()
    main(args.nw, args.iter_)
