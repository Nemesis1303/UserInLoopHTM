import argparse
import logging
import random
import numpy as np
import pathlib
import tomotopy as tp
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../..')
from src.evaluateMatching.alignment import Alignment
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

p_intruder = 0.5  # Prob of introducing an intruder


def sample_bernoulli(n=1, p=0.5):
    return np.random.binomial(1, p, n)[0]


def load_models(paths, key):
    if key == 'hlda':
        return tp.HLDAModel.load(paths[key])
    else:
        return TMmodel(paths[key].joinpath('TMmodel'))


def get_wd_desc(tms, key, topn=100):
    if key == 'hlda':
        # TODO: Check that this is a list of list of str
        return [tms[key].get_topic_words(k, top_n=topn) for k in tms[key].k]
    else:
        return [el[1].split(', ') for el in tms[key].get_tpc_word_descriptions()]


def get_betas(tms, key):
    if key == 'hlda':
        return None
    else:
        return tms[key].to_dataframe()[0].betas[0]


def get_tms_topics_betas(paths):
    tms = {key: load_models(
        paths, key) if paths[key] is not None else None for key in paths.keys()}
    topics = {key: get_wd_desc(
        tms, key) if tms[key] is not None else None for key in tms.keys()}
    betas = {key: get_betas(
        tms, key) if tms[key] is not None else None for key in tms.keys()}
    return tms, topics, betas


def append_intruder_candidate(
        index_df,
        intruder_candidates: list,
        paths: dict,
        key: str,
        tp_id: int,
        tpc_with_intruder: list,
        intruder: str = None,
        pos: int = None,
        original_word: str = None):

    df_intruder = pd.DataFrame({
        'model_path': paths[key],
        'model_type': key,
        'topic_id': tp_id,
        'word_description': ", ".join(tpc_with_intruder),
        'intruder': intruder,
        'intruder_location': pos,
        'original_word': original_word,
        'coherent': None
    }, index=[index_df])
    intruder_candidates.append(df_intruder)
    index_df += 1

    return index_df, intruder_candidates


def calculate_candidate_intruders(
    path_htm: pathlib.Path,
    path_hlda: pathlib.Path,
    iter_: int,
    thr: float,
    tr_tpcs: int,
    top_words: int
) -> None:

    # Save dictionary with paths
    paths = {
        'parent': path_htm,
        'ws': None,
        'ds': None,
        #'hlda': path_hlda
    }

    # Create Alignment object
    al = Alignment()

    # Initialize dictionaries of topic models, topic descriptions and betas
    tms, topics, betas = get_tms_topics_betas(paths)

    nr_tpcs_root = int(paths['parent'].name.split("_")[1])
    # Get JS matrix root model betas vs betas
    betas_root_js_sim = al._sim_word_comp(
        betas1=betas['parent'],
        betas2=betas['parent'])

    # For each topic in root model
    # This is going to be a list of dataframes, where each dataframes corresponds to a topic intruder task
    intruder_candidates = []
    index_df = 0  # Index for the dataframe
    for topic_submodel_from in tqdm(range(nr_tpcs_root)):

        # get htm-ws submodel
        paths['ws'] = [folder for folder in paths['parent'].iterdir() if folder.is_dir() and folder.name.startswith(
            f"submodel_htm-ws_from_topic_{topic_submodel_from}_train_with_{tr_tpcs}_iter_{iter_}")][0]

        # get htm-ds submodel
        paths['ds'] = [folder for folder in paths['parent'].iterdir() if folder.is_dir() and folder.name.startswith(
            f"submodel_htm-ds_thr_{thr}_from_topic_{topic_submodel_from}_train_with_{tr_tpcs}_iter_{iter_}")][0]

        # Update dictionaries of topic models, topic descriptions and betas
        tms, topics, betas = get_tms_topics_betas(paths)

        logger.info(
            f"-- -- Chemical description of topic with ID {topic_submodel_from} in root model {paths['parent'].name}: \n {topics['parent'][topic_submodel_from]}")

        # Get topic in root model least similar to tp_id in submodel
        least_sim_tpc = np.argmin(betas_root_js_sim[topic_submodel_from, :])
        logger.info(
            f"-- -- The least similar topic to topic {topic_submodel_from} in root model is {least_sim_tpc} with description: \n {topics['parent'][least_sim_tpc]}")

        # For each topic in submodel ws/ds
        id_intruder_used = 0
        for tp_id in tqdm(range(tr_tpcs)):

            for key in ['ws', 'ds']:
                # Sample y from a Bernoulli to determine whether to introduce an intruder
                y = sample_bernoulli(p=p_intruder)

                if y == 1:
                    logger.info(f"-- -- Introducing intruder... ")

                    # Generate intruder as the most probable word of the least similar topic to tp_id in the root model according to JS similarity on the betas
                    try:
                        intruder = topics['parent'][least_sim_tpc][id_intruder_used]
                        logger.info(f"-- -- Generated intruder: {intruder}")
                        id_intruder_used += 1
                    except Exception as e:
                        logger.error(e)
                        logger.error(
                            f"-- -- Error generating intruder for topic {tp_id} in submodel {key}...")
                        sys.exit()

                    # Sample position of intruder
                    pos = random.randint(0, top_words-1)

                    # Introduce intruder in the topic description
                    tpc_with_intruder = topics[key][tp_id]
                    logger.info(
                        f"-- -- Topic before the intruder: {tpc_with_intruder}")
                    original_word = tpc_with_intruder[pos]
                    tpc_with_intruder[pos] = intruder
                    logger.info(
                        f"-- -- Topic after the intruder: {tpc_with_intruder}")

                    index_df, intruder_candidates = append_intruder_candidate(
                        index_df=index_df,
                        intruder_candidates=intruder_candidates,
                        paths=paths,
                        key=key,
                        tp_id=tp_id,
                        tpc_with_intruder=tpc_with_intruder,
                        intruder=intruder,
                        pos=pos,
                        original_word=original_word)
                else:
                    logger.info(f"-- -- No intruder is introduced...")
                    # Append topics without modifications
                    index_df, intruder_candidates = append_intruder_candidate(
                        index_df=index_df,
                        intruder_candidates=intruder_candidates,
                        paths=paths,
                        key=key,
                        tp_id=tp_id,
                        tpc_with_intruder=topics[key][tp_id],
                        intruder=None,
                        pos=None,
                        original_word=None)

    """
    # Iterate topics in HLDA model
    key = 'hlda'
    id_intruder_used = 0
    nr_candidates = len(intruder_candidates)
    nr_candidates_hlda = 0
    for hlda_candidate in tms['hlda'].k:
        if nr_candidates_hlda < nr_candidates:

            # Get WMD matrix root model vs HLDA
            wmd_root_hlda = al.wmd(
                topics1=topics['hlda'],
                topics2=topics['parent'],
                n_words=top_words)

            # Get HLDA topic most dissimilar to tp_id in root model
            # The closer the WMD to 1, the more dissimilar the topics are
            root_tp_id = np.argmax(wmd_root_hlda[hlda_candidate, :])
            print(
                f"-- -- Description of HLDA topic {hlda_candidate}: \n {topics['hlda'][hlda_candidate]}")
            print(
                f"-- -- The most dissimilar topic in root model is {root_tp_id} with #description: \n {topics['parent'][root_tp_id]}")

            paths['ws'] = [folder for folder in paths['parent'].iterdir() if folder.is_dir() and folder.name.startswith(
                f"submodel_htm-ws_from_topic_{root_tp_id}_train_with_{tr_tpcs}_iter_{iter_}")][0]

            # Update dictionaries of topic models, topic descriptions and betas
            tms, topics, betas = get_tms_topics_betas(paths)

            y = sample_bernoulli(p=p_intruder)

            if y == 1:
                logger.info(f"-- -- Introducing intruder... ")

                # Generate intruder as the most probable word of topic 0 of submodels generated from the least similar topic to hlda_candidate in the root model according to WMD on the topics descriptions
                intruder = topics['ws'][0][id_intruder_used]
                id_intruder_used += 1

                # Sample position of intruder
                pos = random.randint(0, top_words-1)

                # Introduce intruder in the topic description
                tpc_with_intruder = topics[key][hlda_candidate]
                original_word = tpc_with_intruder[pos]
                tpc_with_intruder[pos] = intruder
                
                index_df, intruder_candidates = append_intruder_candidate(
                    index_df=index_df,
                    intruder_candidates=intruder_candidates,
                    paths=paths,
                    key=key,
                    tp_id=hlda_candidate,
                    tpc_with_intruder=tpc_with_intruder,
                    intruder=intruder,
                    pos=pos,
                    original_word=original_word)        

            else:
                logger.info(f"-- -- No intruder is introduced...")
                # Append topics without modifications
                index_df, intruder_candidates = append_intruder_candidate(
                    index_df=index_df,
                    intruder_candidates=intruder_candidates,
                    paths=paths,
                    key=key,
                    tp_id=hlda_candidate,
                    tpc_with_intruder=topics[key][hlda_candidate],
                    intruder=None,
                    pos=None,
                    original_word=None)  

        nr_candidates_hlda += 1
    """

    df_intruder_candidates = pd.concat(intruder_candidates)
    df_intruder_candidates.to_csv("output/test.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_htm', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models/htm_6_tpcs_20230922",
                        help="Path to the HTM model (pointer to root model).")
    parser.add_argument('--path_hlda', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/experiments/hlda/output/hlda.bin",
                        help="Path to the HLDA model.")
    parser.add_argument('--iter_', type=int,
                        default=1, help="Iteration of the submodel ws/ds.")
    parser.add_argument('--thr', type=float,
                        default=0.6, help="Threshold for the submodel ds.")
    parser.add_argument('--tr_tpcs', type=int,
                        default=10, help="Submodels ws/ds's training topics.")
    parser.add_argument('--top_words', type=int,
                        default=15, help="Nr of words to consider in the topic description.")

    args = parser.parse_args()
    calculate_candidate_intruders(
        path_htm=pathlib.Path(args.path_htm),
        path_hlda=args.path_htm,  # pathlib.Path(args.path_hlda)
        iter_=args.iter_,
        thr=args.thr,
        tr_tpcs=args.tr_tpcs,
        top_words=args.top_words
    )


if __name__ == "__main__":
    main()
