import argparse
import random
from typing import Union
import numpy as np
import pathlib
import pandas as pd
from tqdm import tqdm
import requests
from urllib import parse
from bs4 import BeautifulSoup
import html
import re
import time

import sys
sys.path.append('../../..')
from src.evaluateMatching.alignment import Alignment
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel
from src.utils.misc import mallet_corpus_to_df

def get_df_corpus_thetas(
    path_submodel: pathlib.Path,
    sub_type: str
) -> Union[pd.DataFrame, list]:
    """Get dataframe with the corpus and  thetas for a given submodel (ws or ds).

    Parameters
    ----------
    path_submodel: pathlib.Path
        Path to the submodel.
    sub_type: str
        Type of submodel (ws or ds).

    Returns
    -------
    pd.DataFrame: Dataframe with the corpus and thetas for the submodel.
    list: List of descriptions for the topics in the submodel.
    """
    df = mallet_corpus_to_df(path_submodel.joinpath("corpus.txt"))
    tm = TMmodel(path_submodel.joinpath("TMmodel"))
    tm._load_thetas()
    thetas = tm._thetas
    df["thetas_" + sub_type] = thetas.toarray().tolist()
    desc = [el[1] for el in tm.get_tpc_word_descriptions(15)]
    df["id"] = df["id"].apply(pd.to_numeric)
    return df, desc


def count_words_in_list(
    text: str,
    word_list: list
) -> int:
    """Count the number of words in a list that appear in a text.

    Parameters
    ----------
    text: str
        Text to check.
    word_list: list
        List of words to check.

    Returns
    -------
    int: Number of words in the list that appear in the text.
    """
    words = text.split()
    return sum(word in word_list for word in words)


def get_label(
    chem_desc: str
) -> str:

    print("-- -- Waiting for a minute to get label...")
    time.sleep(60)
    data = {'chemical_description': chem_desc}
    query_string = parse.urlencode(data)
    url_labeller = 'http://kumo01:8080/topiclabeller/getLabel/'
    url_ = '{}?{}'.format(url_labeller, query_string)
    try:
        print("Getting label...")
        resp = requests.get(
            url=url_,
            timeout=None,
            params={'Content-Type': 'application/json'}
        )
        print(f"Labels obtained: {resp.text}")
    except Exception as e:
        print(f"Exception when getting label: {e}")
        return ""
    return resp.text


def calculate_candidates(
    root_model: pathlib.Path,
    path_orig: pathlib.Path,
    iter_: int,
    tr_tpcs: int,
    thrs: list,
    high: float,
    small: float,
    suffix_save: str,
    column_merge: str = "projectID"
) -> pd.DataFrame:
    """
    Calculate the candidates for the second human evaluation task:
    - For each pair of submodels in the root model (iter=1, try different thresholds)
    -- Keep top-2 similar pairs
    -- Keep as candidates the documents where the difference between the thetas 
       for those topics is high
    ----- One in which ws is high and ds is low
    ----- One in which ws is low and ds is high

    Parameters
    ----------
    root_model: pathlib.Path
        Path to the root model.
    path_orig: pathlib.Path
        Path to the original corpus (path with the corpus with the raw text, before preprocessing)
    iter_: int
        Iteration of the submodel ws/ds.
    tr_tpcs: int
        Submodels ws/ds's training topics.
    thrs: list
        List of thresholds for the submodel ds.
    high: float
        Upper threshold in the thetas to look for candidates.
    small: float
        Lower threshold in the thetas to look for candidates.
    suffix_save: str
        Suffix to add to the name of the file to save the candidates.

    Returns
    -------
    pd.DataFrame: Dataframe with the candidates.    
    """

    nr_tpcs_root = int(root_model.name.split("_")[1])

    # For each topic in root model
    candidates = []
    id_pair = 0
    for topic_submodel_from in tqdm(range(nr_tpcs_root)):

        for thr in thrs:

            # get htm-ws submodel path
            sub_ws_path = [folder for folder in root_model.iterdir() if folder.is_dir() and folder.name.startswith(
                f"submodel_htm-ws_from_topic_{topic_submodel_from}_train_with_{tr_tpcs}_iter_{iter_}")][0]

            # get htm-ds submodel path
            sub_ds_path = [folder for folder in root_model.iterdir() if folder.is_dir() and folder.name.startswith(
                f"submodel_htm-ds_thr_{thr}_from_topic_{topic_submodel_from}_train_with_{tr_tpcs}_iter_{iter_}")][0]

            # get df for ws submodel
            df_ws, desc_ws = get_df_corpus_thetas(sub_ws_path, "ws")

            # get df for ds submodel
            df_ds, desc_ds = get_df_corpus_thetas(sub_ds_path, "ds")

            # get df for root model
            df_root = mallet_corpus_to_df(sub_ws_path.parent.joinpath("corpus.txt"))
            df_root = df_root.rename(columns={"id":column_merge})
            df_root[column_merge] = df_root[column_merge].apply(pd.to_numeric)
            df_root["id"] = list(range(len(df_root)))

            # get df for raw text
            df_root_orig = pd.read_parquet(path_orig)
            
            # Merge df_root and df_root_orig to get 'raw_text' column
            df_root = df_root.merge(df_root_orig, on=column_merge, how='inner').fillna("")

            # Merge df_ws and df_ds to get ids of documents that are in both submodels
            df_ws_ds = df_ws.merge(df_ds, on='id', how='inner').fillna("")

            # Merge df_ws_ds with df_root to get the raw corpus of the documents kept in both submodels
            df_root_sub = df_ws_ds.merge(df_root, on='id', how='inner').fillna("")
            df_root_sub['doc_length'] = df_root_sub['raw_text'].apply(lambda x: len(x.split()))
            df_root_sub = df_root_sub[df_root_sub['doc_length'] < 300]
                                    
            al = Alignment()
            vs_sims = al.do_one_to_one_matching(
                tmModel1=sub_ws_path.as_posix(),
                tmModel2=sub_ds_path.as_posix())

            # Get tuples of the index i,j and the value associated with them
            index_value_pairs = np.ndenumerate(vs_sims)
            
            # Sort the pairs based on the value in descending order
            sorted_pairs = sorted(index_value_pairs, key=lambda x: x[1], reverse=True)
            
            ws_topics_used = []
            ds_topics_used = []
            pairs_to_use = []
            for pair in sorted_pairs:
                if pair[0][0] not in ws_topics_used and pair[0][1] not in ds_topics_used:
                    ws_topics_used.append(pair[0][0])
                    ds_topics_used.append(pair[0][1])
                    pairs_to_use.append(pair)
                else:
                    continue
                            
            print(pairs_to_use)
          
            for sorted_pair in pairs_to_use:
                try:
                    def check_condition1(row): return row['thetas_ds'][sorted_pair[0][1]] > high and row[
                        'thetas_ws'][sorted_pair[0][0]] < small and row['thetas_ws'][sorted_pair[0][0]] > 0.0

                    def check_condition2(row): return row['thetas_ws'][sorted_pair[0][0]] > high and row[
                        'thetas_ds'][sorted_pair[0][1]] < small and row['thetas_ds'][sorted_pair[0][1]] > 0.0

                    print("-- -- Getting ds high and ws low...")
                    ids1 = df_root_sub[df_root_sub.apply(
                        check_condition1, axis=1)]

                    print("-- -- Getting ws high and ds low...")
                    ids2 = df_root_sub[df_root_sub.apply(
                        check_condition2, axis=1)]

                    ws_label = get_label(desc_ws[sorted_pair[0][0]])
                    ds_label = get_label(desc_ds[sorted_pair[0][1]])

                    # Sample position of document for ids1
                    pos = random.randint(0, len(ids1))
                    candidate = ids1[ids1.id == ids1.iloc[pos].id].copy()
                    candidate["corpus"] = suffix_save
                    candidate["id_pair"] = id_pair
                    candidate["opt_select"] = "high"
                    candidate["ws_tpc"] = sorted_pair[0][0]
                    candidate["ds_tpc"] = sorted_pair[0][1]
                    candidate["ws_desc"] = desc_ws[sorted_pair[0][0]]
                    candidate["ws_label"] = ws_label
                    candidate["ds_desc"] = desc_ds[sorted_pair[0][1]]
                    candidate["ds_label"] = ds_label
                    candidate["answer"] = "ds"
                    
                    candidates.append(candidate)

                    # Sample position of document for ids2
                    pos = random.randint(0, len(ids2))
                    candidate = ids2[ids2.id == ids2.iloc[pos].id].copy()
                    candidate["corpus"] = suffix_save
                    candidate["id_pair"] = id_pair
                    candidate["opt_select"] = "low"
                    candidate["ws_tpc"] = sorted_pair[0][0]
                    candidate["ds_tpc"] = sorted_pair[0][1]
                    candidate["ws_desc"] = desc_ws[sorted_pair[0][0]]
                    candidate["ws_label"] = ws_label
                    candidate["ds_desc"] = desc_ds[sorted_pair[0][1]]
                    candidate["ds_label"] = ds_label
                    candidate["answer"] = "ws"
                    
                    candidates.append(candidate)
                    
                    id_pair += 1

                except Exception as e:
                    print(
                        f"No candidate found with the spcified conditions: {e}")
                    
    df_candidates = pd.concat(candidates)
    
    def clean_text(text):
        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        
        # Unescape special characters
        text = html.unescape(text)
        
        # Remove non-alphanumeric characters except punctuation marks
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        
        return text
    
    df_candidates['clean_text'] = df_candidates['raw_text'].apply(clean_text)
            
    df_candidates.to_csv(f"output/candidates_{suffix_save}.csv", index=False)

    return df_candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_htm', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models/htm_6_tpcs_20230922",
                        help="Path to the HTM model (pointer to root model).")
    parser.add_argument('--path_orig', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/cordis_lemmas.parquet",
                        help="Path to the original corpus.")
    parser.add_argument('--iter_', type=int,
                        default=1, help="Iteration of the submodel ws/ds.")
    parser.add_argument('--thr', type=float,
                        default=0.6, help="Threshold for the submodel ds.")
    parser.add_argument('--tr_tpcs', type=int,
                        default=10, help="Submodels ws/ds's training topics.")
    parser.add_argument('--high', type=int,
                        default=0.9, help="Upper threshold in the thetas to look for candidates.")
    parser.add_argument('--low', type=int,
                        default=0.1, help="Lower threshold in the thetas to look for candidates.")
    parser.add_argument('--suffix_save', type=str,
                        default="cordis", help="Suffix to add to the name of the file to save the candidates.")

    args = parser.parse_args()

    df_candidates = calculate_candidates(
        root_model=pathlib.Path(args.path_htm),
        path_orig=pathlib.Path(args.path_orig),
        iter_=args.iter_,
        tr_tpcs=args.tr_tpcs,
        thrs=[args.thr],
        high=args.high,
        small=args.low,
        suffix_save=args.suffix_save,
        column_merge = "corpusid"
    )
