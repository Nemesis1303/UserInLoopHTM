import argparse
import logging
import numpy as np
import pathlib
import tomotopy as tp
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../..')
from src.utils.misc import mallet_corpus_to_df
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

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

    corpus_val = pathlib.Path(
        f"/export/usuarios_ml4ds/lbartolome/Datasets/{corpus_path}/models_preproc/iter_0/corpus_val.txt")

    corpus_df = mallet_corpus_to_df(corpus_val)
    corpus_df['text'] = corpus_df['text'].apply(lambda x: x.split())
    reference_text = corpus_df.text.values.tolist()

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
        np.save(hlda_path.parent.joinpath(
            f"{suffix_save}_{metric}.npy"), topic_coherence)

    return


def get_hlda_descs(mdl, corpus, topn=15):
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
        words = [mdl.vocabs[idx2]
                 for idx2 in np.argsort(betas_ds[i])[::-1][0:topn]]
        tpc_descs.append((i, ', '.join(words)))

    if corpus == "cordis":
        tpc_descs = [el for el in tpc_descs if el[0] not in [
            0, 2, 3, 4, 5, 6, 7, 85, 89, 92, 93, 94, 95]]
    elif corpus == "cancer":
        pass
    elif corpus == "ai":
        tpc_descs = [el for el in tpc_descs if el[0]
                     not in [0, 2, 3, 4, 5, 6, 7]]
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
        return [el[1].split(', ') for el in get_hlda_descs(tms[key], corpus, topn=topn)]
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


def append_candidate(
        index_df,
        candidates: list,
        paths: dict,
        cohrs_cv: float,
        cohrs_npmi: float,
        key: str,
        tp_id: int,
        word_description: list,
        candidate_id: int):

    df_candidates = pd.DataFrame({
        'candidate_id': candidate_id,
        'model_path': paths[key],
        'model_type': key,
        'topic_id': tp_id,
        'word_description': ", ".join(word_description),
        'cohrs_cv': cohrs_cv,
        'cohrs_npmi': cohrs_npmi
    }, index=[index_df])
    candidates.append(df_candidates)
    index_df += 1
    candidate_id += 1
    return index_df, candidates, candidate_id


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

    # Initialize dictionaries of topic models, topic descriptions and betas
    tms, topics, betas, cohrs_cv, cohrs_npmi = get_tms_topics_betas_cohrs(
        paths, suffix_save, topn=50)

    # Nr of topics in root model
    nr_tpcs_root = int(paths['parent'].name.split("_")[1])

    # This is going to be a list of dataframes, where each dataframes corresponds to a topic intruder task
    candidates = []
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
        _, topics, _, cohrs_cv, cohrs_npmi = get_tms_topics_betas_cohrs(
            paths, suffix_save, topn=top_words)

        # For each topic in submodel ws/ds
        for tp_id in tqdm(range(tr_tpcs)):
            for key in ['ws', 'ds']:
                index_df, candidates, candidate_id = append_candidate(
                    index_df=index_df,
                    candidates=candidates,
                    paths=paths,
                    cohrs_cv=cohrs_cv[key][tp_id],
                    cohrs_npmi=cohrs_npmi[key][tp_id],
                    key=key,
                    tp_id=tp_id,
                    word_description=topics[key][tp_id],
                    candidate_id=candidate_id
                )

    # Iterate topics in HLDA model
    key = 'hlda'
    def minimum(a, b): return a if a < b else b
    nr_candidates = int(minimum(len(candidates), len(topics['hlda'])))
    nr_candidates_hlda = 0

    for hlda_candidate in range(len(topics['hlda'])):
        if nr_candidates_hlda < nr_candidates:

            index_df, candidates, candidate_id = append_candidate(
                index_df=index_df,
                candidates=candidates,
                paths=paths,
                cohrs_cv=cohrs_cv[key][hlda_candidate],
                cohrs_npmi=cohrs_npmi[key][hlda_candidate],
                key=key,
                tp_id=hlda_candidate,
                word_description=topics[key][hlda_candidate],
                candidate_id=candidate_id
            )
        nr_candidates_hlda += 1

    df_candidates = pd.concat(candidates)
    name_save = f"output/{suffix_save}.csv"
    df_candidates.to_csv(name_save, index=False)


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
                        default=10, help="Nr of words to consider in the topic description for the word intrusion task.")
    parser.add_argument('--suffix_save', type=str,
                        default="cordis", help="Suffix to save the coherence of the HLDA model.")

    args = parser.parse_args()

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
