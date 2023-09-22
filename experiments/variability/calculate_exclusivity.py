import argparse
import os
import sys
import numpy as np
import pathlib

# Add src to path and make imports
sys.path.append('../..')
from src.utils.misc import mallet_corpus_to_df
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel

def recalculate_cohr(corpus_val, path_models, root_topics):
    
    def get_cohrs(model_path):
            if np.load(model_path.joinpath("TMmodel").joinpath(
                    'new_topic_coherence.npy'), allow_pickle=True).tolist() is None:
                tm = TMmodel(model_path.joinpath("TMmodel"))
                cohr = tm.calculate_topic_coherence(
                    metrics=["c_npmi"],
                    reference_text=corpus_df.text.values.tolist(),
                    aggregated=False,
                )
                if os.path.exists(model_path.joinpath("TMmodel").joinpath(
                        'new_topic_coherence.npy').as_posix()):
                    os.remove(model_path.joinpath("TMmodel").joinpath(
                        'new_topic_coherence.npy').as_posix())
                np.save(model_path.joinpath("TMmodel").joinpath(
                    'new_topic_coherence.npy'), cohr)

    def get_rbo(model_path):
        tm = TMmodel(model_path.joinpath("TMmodel"))
        rbo = tm.calculate_rbo()
        print(rbo)
        if os.path.exists(model_path.joinpath("TMmodel").joinpath(
                'rbo.npy').as_posix()):
            os.remove(model_path.joinpath("TMmodel").joinpath(
                'rbo.npy').as_posix())
        np.save(model_path.joinpath("TMmodel").joinpath(
                'rbo.npy'), rbo)
        return

    def get_td(model_path):
        tm = TMmodel(model_path.joinpath("TMmodel"))
        td = tm.calculate_topic_diversity()
        print(td)
        if os.path.exists(model_path.joinpath("TMmodel").joinpath(
                'td.npy').as_posix()):
            os.remove(model_path.joinpath("TMmodel").joinpath(
                'td.npy').as_posix())
        np.save(model_path.joinpath("TMmodel").joinpath(
                'td.npy'), td)
        return
        
    if root_topics:
        root_topics = root_topics.split(',')
        for root_topic in root_topics:
            path_models_tpc = pathlib.Path(
                path_models).joinpath(f"{root_topic}_tpc_root")

            print(f"-- -- Recalculating coherence for {path_models_tpc}")

            corpus_df = mallet_corpus_to_df(pathlib.Path(corpus_val))
            corpus_df['text'] = corpus_df['text'].apply(lambda x: x.split())

            
            for entry in path_models_tpc.iterdir():
                get_rbo(entry)
                get_td(entry)
                for entry_ in entry.iterdir():
                    if entry_.joinpath('TMmodel/alphas.npy').is_file() and not entry_.as_posix().endswith("old"):
                        get_rbo(entry_)
                        get_td(entry_)
    else:
        path_models_tpc = pathlib.Path(
                path_models)

        print(f"-- -- Recalculating coherence for {path_models_tpc}")

        corpus_df = mallet_corpus_to_df(pathlib.Path(corpus_val))
        corpus_df['text'] = corpus_df['text'].apply(lambda x: x.split())

        
        for entry in path_models_tpc.iterdir():
            get_rbo(entry)
            get_td(entry)
            for entry_ in entry.iterdir():
                if entry_.joinpath('TMmodel/alphas.npy').is_file() and not entry_.as_posix().endswith("old"):
                    get_rbo(entry_)
                    get_td(entry_)
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_models', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models",
                        help="Path to the training data.")
    parser.add_argument('--path_val_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus_val.txt",
                        help="Path to the validation training data.")
    parser.add_argument('--root_topics', type=str,
                        help="Nr of topics for the root model.")

    args = parser.parse_args()
    recalculate_cohr(corpus_val=args.path_val_corpus,
                     path_models=args.path_models,
                     root_topics=args.root_topics)


if __name__ == "__main__":
    main()
