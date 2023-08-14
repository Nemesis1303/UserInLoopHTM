import argparse
import os
import sys
from distutils.dir_util import copy_tree
import numpy as np

# Add src to path and make imports
sys.path.append('../..')
from src.utils.misc import mallet_corpus_to_df
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel


def recalculate_cohr(corpus_val, path_models, root_topics):
    root_topics = root_topics.split(',')
    for root_topic in root_topics:
        path_models = path_models.joinpath(f"{root_topic}_tpc_root")
        
    print(f"-- -- Recalculating coherence for {path_models}")
    
    corpus_df = mallet_corpus_to_df(corpus_val)
    corpus_df['text'] = corpus_df['text'].apply(lambda x: x.split())

    for model_path in path_models.iterdir():
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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_models', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/htm_variability_models",
                        help="Path to the training data.")
    parser.add_argument('--path_val_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus_val.txt",
                        help="Path to the validation training data.")
    parser.add_argument('--root_topics', type=str,
                    default="5,10,20",
                    help="Nr of topics for the root model.")
    
    args = parser.parse_args()

if __name__ == "__main__":
    main()