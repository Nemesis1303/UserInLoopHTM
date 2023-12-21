import argparse
import logging
import random
import numpy as np
import pathlib
import tomotopy as tp
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../..')
from src.evaluateMatching.alignment import Alignment
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel
from src.utils.misc import mallet_corpus_to_df

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

p_intruder = 1  # Prob of introducing an intruder

def sample_bernoulli(n=1, p=0.75):
    return np.random.binomial(1, p, n)[0]

def load_models(paths, key):
    if key == 'hlda':
        return tp.HLDAModel.load(paths[key])
    else:
        return TMmodel(paths[key].joinpath('TMmodel'))
    
def calculate_hlda_coherence(hlda_path, suffix_save, topn=15):
    
    if hlda_path.parent.joinpath(f"{suffix_save}_c_npmi.npy").is_file():
        print("Coherence already calculated")
        return
        
    if suffix_save == "cordis":
        corpus_path = "CORDIS"
    elif suffix_save == "cancer":
        corpus_path = "Cancer"
    elif suffix_save == "ai":
        corpus_path = "S2CS-AI"

    corpus_val = pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Datasets/{corpus_path}/models_preproc/iter_0/corpus_val.txt")
    
    corpus_df = mallet_corpus_to_df(corpus_val)
    corpus_df['text'] = corpus_df['text'].apply(lambda x: x.split())
    reference_text=corpus_df.text.values.tolist()
    
    mdl = tp.HLDAModel.load(hlda_path.as_posix())
    tpc_descriptions = \
                [el[1] for el in get_hlda_descs(mdl, suffix_save, topn=topn)]
    tpc_descriptions = \
        [tpc.split(', ') for tpc in tpc_descriptions]
        
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    
    dictionary = Dictionary(reference_text)
    
    for metric in ["c_npmi", "c_v"]:
        cm = CoherenceModel(topics=tpc_descriptions,
                            texts=reference_text,
                            dictionary=dictionary, 
                            coherence=metric,
                            topn=topn)
        topic_coherence = cm.get_coherence_per_topic()
        
        print(topic_coherence)
        np.save(hlda_path.parent.joinpath(f"{suffix_save}_{metric}.npy"), topic_coherence)
    
    return
    
def get_hlda_descs(mdl,corpus,topn=15):
    tpc_descs = []
    
    # Get word-topic distribution (dims= n_topics x n_words)
    betas = np.array([mdl.get_topic_word_dist(el) for el in range(mdl.k)]) 
    betas_ds = np.copy(betas)
    if np.min(betas_ds) < 1e-12:
        betas_ds += 1e-12
    deno = np.reshape((sum(np.log(betas_ds)) / mdl.k), (len(mdl.vocabs), 1))
    deno = np.ones((mdl.k, 1)).dot(deno.T)
    betas_ds = betas_ds * (np.log(betas_ds) - deno)
    
    # Get topic descriptions
    tpc_descs = []
    for i in range(mdl.k):
        words = [mdl.vocabs[idx2] for idx2 in np.argsort(betas_ds[i])[::-1][0:topn]]
        tpc_descs.append((i, ', '.join(words)))
        
    if corpus == "cordis":
        tpc_descs = [el for el in tpc_descs if el[0] not in [0,2,3,4,5,6,7,85,89,92,93,94,95]]
    elif corpus == "cancer":
        pass
    elif corpus == "ai":
        tpc_descs = [el for el in tpc_descs if el[0] not in [0,2,3,4,5,6,7]]
    return tpc_descs

def get_coherence(paths, key, suffix_save, cohr):
    if key == 'hlda':
        return np.load(pathlib.Path(paths[key]).parent.joinpath(f"{suffix_save}_{cohr}.npy")).tolist()
    elif key == 'parent':
        return np.load(paths[key].joinpath(f"TMmodel/{cohr}_ref_coherence.npy")).tolist()
    else:
        return np.load(paths[key].joinpath(f"TMmodel/{cohr}_ref_coherence.npy")).tolist()

def get_wd_desc(tms, betas, key, corpus, topn=15):
    if key == 'hlda':
        return [el[1].split(', ') for el in get_hlda_descs(tms[key],corpus,topn=topn)]
    elif key == 'parent':
        return [el[1].split(', ') for el in tms[key].get_tpc_word_descriptions(n_words=betas[key].shape[1])]
    else:
        return [el[1].split(', ') for el in tms[key].get_tpc_word_descriptions(n_words=topn)]


def get_betas(tms, key):
    if key == 'hlda':
        return None
    else:
        return tms[key].to_dataframe()[0].betas[0]


def get_tms_topics_betas_cohrs(paths, suffix_save, topn):
    tms = {key: load_models(
        paths, key) if paths[key] is not None else None for key in paths.keys()}
    betas = {key: get_betas(
        tms, key) if tms[key] is not None else None for key in tms.keys()}
    topics = {key: get_wd_desc(
        tms, betas, key, suffix_save, topn) if tms[key] is not None else None for key in tms.keys()}
    cohrs_cv = {key: get_coherence(
        paths, key, suffix_save, 'c_v') if tms[key] is not None else None for key in tms.keys()}
    cohrs_npmi = {key: get_coherence(
        paths, key, suffix_save, 'c_npmi') if tms[key] is not None else None for key in tms.keys()}
    return tms, topics, betas, cohrs_cv, cohrs_npmi


def append_intruder_candidate(
        index_df,
        intruder_candidates: list,
        paths: dict,
        cohrs_cv: float,
        cohrs_npmi: float,
        key: str,
        candidate_id: int,
        tp_id: int,
        tpc_with_intruder: list,
        intruder: str = None,
        pos: int = None,
        original_word: str = None):

    df_intruder = pd.DataFrame({
        'model_path': paths[key],
        'model_type': key,
        'candidate_id': candidate_id,
        'topic_id': tp_id,
        'word_description': ", ".join(tpc_with_intruder),
        'intruder': intruder,
        'intruder_location': pos,
        'original_word': original_word,
        'cohrs_cv': cohrs_cv,
        'cohrs_npmi': cohrs_npmi
    }, index=[index_df])
    intruder_candidates.append(df_intruder)
    index_df += 1
    candidate_id += 1
    return index_df, intruder_candidates, candidate_id


def calculate_candidate_intruders(
    path_htm: pathlib.Path,
    path_hlda: pathlib.Path,
    iter_: int,
    thr: float,
    tr_tpcs: int,
    top_words: int,
    suffix_save: str
) -> None:

    # Save dictionary with paths
    paths = {
        'parent': path_htm,
        'ws': None,
        'ds': None,
        'hlda': path_hlda
    }

    # Create Alignment object
    al = Alignment()

    # Initialize dictionaries of topic models, topic descriptions and betas
    tms, topics, betas, cohrs_cv, cohrs_npmi = get_tms_topics_betas_cohrs(paths, suffix_save, topn=50)

    nr_tpcs_root = int(paths['parent'].name.split("_")[1])
    # Get JS matrix root model betas vs betas
    betas_root_js_sim = al._sim_word_comp(
        betas1=betas['parent'],
        betas2=betas['parent'])

    
    # This is going to be a list of dataframes, where each dataframes corresponds to a topic intruder task
    intruder_candidates = []
    index_df = 0  # Index for the dataframe
    candidate_id = 0  # Index for the candidate
    
    # For each topic in root model
    for topic_submodel_from in tqdm(range(nr_tpcs_root)):

        # get htm-ws submodel
        paths['ws'] = [folder for folder in paths['parent'].iterdir() if folder.is_dir() and folder.name.startswith(
            f"submodel_htm-ws_from_topic_{topic_submodel_from}_train_with_{tr_tpcs}_iter_{iter_}")][0]

        # get htm-ds submodel
        paths['ds'] = [folder for folder in paths['parent'].iterdir() if folder.is_dir() and folder.name.startswith(
            f"submodel_htm-ds_thr_{thr}_from_topic_{topic_submodel_from}_train_with_{tr_tpcs}_iter_{iter_}")][0]

        # Update dictionaries of topic models, topic descriptions and betas
        tms, topics, betas, cohrs_cv, cohrs_npmi = get_tms_topics_betas_cohrs(paths, suffix_save, topn=50)

        logger.info(
            f"-- -- Chemical description of topic with ID {topic_submodel_from} in root model {paths['parent'].name}: \n {topics['parent'][topic_submodel_from]}")

        # Get topic in root model least similar to tp_id in submodel
        least_sim_tpc = np.argmin(betas_root_js_sim[topic_submodel_from, :])
        logger.info(
            f"-- -- The least similar topic to topic {topic_submodel_from} in root model is {least_sim_tpc} with description: \n {topics['parent'][least_sim_tpc]}")

        # For each topic in submodel ws/ds
        intruders_used = []
        possible_intruders = topics['parent'][least_sim_tpc][:50]
        for tp_id in tqdm(range(tr_tpcs)):

            for key in ['ws', 'ds']:
                # Sample y from a Bernoulli to determine whether to introduce an intruder
                y = sample_bernoulli(p=p_intruder)

                if y == 1:
                    logger.info(f"-- -- Introducing intruder... ")

                    # Generate intruder as the most probable word of the least similar topic to tp_id in the root model according to JS similarity on the betas
                    # We need to assert that the intruder is not in the top 10 words of the original topic
                    
                    top_random_words = [word for word in possible_intruders[:50] if word not in topics[key][tp_id] and word not in intruders_used]
                    
                    try:
                        intruder = top_random_words[0]
                        intruders_used.append(intruder)
                        logger.info(f"-- -- Generated intruder: {intruder}")
                    except Exception as e:
                        logger.error(e)
                        logger.error(
                            f"-- -- Error generating intruder for topic {tp_id} in submodel {key}...")
                        sys.exit()
                        
                    # Sample position of intruder
                    pos = random.randint(0, top_words-1)

                    # Introduce intruder in the topic description
                    tpc_with_intruder = topics[key][tp_id][:top_words]
                    logger.info(
                        f"-- -- Topic before the intruder: {tpc_with_intruder}")
                    original_word = tpc_with_intruder[pos]
                    tpc_with_intruder[pos] = intruder
                    logger.info(
                        f"-- -- Topic after the intruder: {tpc_with_intruder}")

                    index_df, intruder_candidates, candidate_id = append_intruder_candidate(
                        index_df=index_df,
                        intruder_candidates=intruder_candidates,
                        paths=paths,
                        cohrs_cv=cohrs_cv[key][tp_id],
                        cohrs_npmi=cohrs_npmi[key][tp_id],
                        key=key,
                        candidate_id=candidate_id,
                        tp_id=tp_id,
                        tpc_with_intruder=tpc_with_intruder,
                        intruder=intruder,
                        pos=pos,
                        original_word=original_word)
                else:
                    logger.info(f"-- -- No intruder is introduced...")
                    # Append topics without modifications
                    index_df, intruder_candidates, candidate_id = append_intruder_candidate(
                        index_df=index_df,
                        intruder_candidates=intruder_candidates,
                        paths=paths,
                        cohrs_cv=cohrs_cv[key][tp_id],
                        cohrs_npmi=cohrs_npmi[key][tp_id],
                        key=key,
                        candidate_id=candidate_id,
                        tp_id=tp_id,
                        tpc_with_intruder=topics[key][tp_id],
                        intruder=None,
                        pos=None,
                        original_word=None)


    # Iterate topics in HLDA model
    key = 'hlda'
    intruders_used = []
    minimum = lambda a,b:a if a < b else b
    nr_candidates = int(minimum(len(intruder_candidates), len(topics['hlda'])))
    nr_candidates_hlda = 0
    
    for hlda_candidate in range(len(topics['hlda'])):
        if nr_candidates_hlda < nr_candidates:

            # Get WMD matrix root model vs HLDA
            wmd_root_hlda = al.wmd_two_tpcs(
                topics1=topics['hlda'],
                topics2=topics['parent'],
                n_words=15)#top_words

            # Get HLDA topic most dissimilar to tp_id in root model
            # The closer the WMD to 1, the more dissimilar the topics are
            print(hlda_candidate)
            print(wmd_root_hlda[hlda_candidate, :])
            root_tp_id = np.argmax(wmd_root_hlda[hlda_candidate, :])
            print(
                f"-- -- Description of HLDA topic {hlda_candidate}: \n {topics['hlda'][hlda_candidate]}")
            print(
                f"-- -- The most dissimilar topic in root model is {root_tp_id} with #description: \n {topics['parent'][root_tp_id]}")

            paths['ws'] = [folder for folder in paths['parent'].iterdir() if folder.is_dir() and folder.name.startswith(
                f"submodel_htm-ws_from_topic_{root_tp_id}_train_with_{tr_tpcs}_iter_{iter_}")][0]

            # Update dictionaries of topic models, topic descriptions and betas
            tms, topics, betas, cohrs_cv, cohrs_npmi = get_tms_topics_betas_cohrs(paths, suffix_save, topn=50)

            y = sample_bernoulli(p=p_intruder)

            if y == 1:
                logger.info(f"-- -- Introducing intruder... ")

                # Generate intruder as the most probable word of topic 0 of submodel WS generated from the least similar topic to hlda_candidate in the root model according to WMD on the topics descriptions
                # We need to assert that the intruder is not in the top 10 words of the original topic
                
                possible_intruders = topics['ws'][0][:50]
                                        
                top_random_words = [word for word in possible_intruders[:50] if word not in topics[key][hlda_candidate] and word not in intruders_used]
                    
                try:
                    intruder = top_random_words[0]
                    intruders_used.append(intruder)
                    logger.info(f"-- -- Generated intruder: {intruder}")
                except Exception as e:
                    logger.error(e)
                    logger.error(
                        f"-- -- Error generating intruder for topic {tp_id} in submodel {key}...")
                    sys.exit()
                
                # Sample position of intruder
                pos = random.randint(0, top_words-1)

                # Introduce intruder in the topic description
                tpc_with_intruder = topics[key][hlda_candidate][:top_words]
                original_word = tpc_with_intruder[pos]
                tpc_with_intruder[pos] = intruder
                
                
                index_df, intruder_candidates, candidate_id = append_intruder_candidate(
                    index_df=index_df,
                    intruder_candidates=intruder_candidates,
                    paths=paths,
                    cohrs_cv=cohrs_cv[key][hlda_candidate],
                    cohrs_npmi=cohrs_npmi[key][hlda_candidate],
                    key=key,
                    candidate_id=candidate_id,
                    tp_id=hlda_candidate,
                    tpc_with_intruder=tpc_with_intruder,
                    intruder=intruder,
                    pos=pos,
                    original_word=original_word)        

            else:
                logger.info(f"-- -- No intruder is introduced...")
                # Append topics without modifications
                index_df, intruder_candidates, candidate_id = append_intruder_candidate(
                    index_df=index_df,
                    intruder_candidates=intruder_candidates,
                    paths=paths,
                    cohrs_cv=cohrs_cv[key][hlda_candidate],
                    cohrs_npmi=cohrs_npmi[key][hlda_candidate],
                    key=key,
                    candidate_id=candidate_id,
                    tp_id=hlda_candidate,
                    tpc_with_intruder=topics[key][hlda_candidate],
                    intruder=None,
                    pos=None,
                    original_word=None)  

        nr_candidates_hlda += 1
        
    df_intruder_candidates = pd.concat(intruder_candidates)
    name_save = f"output/{suffix_save}.csv"
    df_intruder_candidates.to_csv(name_save, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_htm', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models/htm_6_tpcs_20230922",
                        help="Path to the HTM model (pointer to root model).")
    parser.add_argument('--path_hlda', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/experiments/hlda/output/hlda_cordis.bin",
                        help="Path to the HLDA model.")
    parser.add_argument('--iter_', type=int,
                        default=1, help="Iteration of the submodel ws/ds.")
    parser.add_argument('--thr', type=float,
                        default=0.6, help="Threshold for the submodel ds.")
    parser.add_argument('--tr_tpcs', type=int,
                        default=10, help="Submodels ws/ds's training topics.")
    parser.add_argument('--top_words', type=int,
                        default=6, help="Nr of words to consider in the topic description for the word intrusion task.")
    parser.add_argument('--suffix_save', type=str,
                        default="cordis", help="Suffix to save the coherence of the HLDA model.")

    args = parser.parse_args()
    
    print("entra")
    
    calculate_hlda_coherence(hlda_path=pathlib.Path(args.path_hlda),
                             suffix_save=args.suffix_save)
                             
    calculate_candidate_intruders(
        path_htm=pathlib.Path(args.path_htm),
        path_hlda=args.path_hlda,
        iter_=args.iter_,
        thr=args.thr,
        tr_tpcs=args.tr_tpcs,
        top_words=args.top_words,
        suffix_save=args.suffix_save
    )


if __name__ == "__main__":
    main()
