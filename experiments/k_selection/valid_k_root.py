import itertools
import multiprocessing as mp
import pathlib
import sys
import warnings
from matplotlib import pyplot as plt

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.model_selection import RepeatedKFold, train_test_split
from tqdm import tqdm


# Add src to path and make imports
sys.path.append('../..')
from src.utils.misc import corpus_df_to_mallet, mallet_corpus_to_df
from src.tmWrapper.tm_wrapper import TMWrapper
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel

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


def run_k_fold(models_folder, trainer, corpusFile, grid_params, val_size=0.3):

    # Read corpus as df
    corpusFile = pathlib.Path(corpusFile)
    corpus_df = mallet_corpus_to_df(corpusFile)

    # Create validation corpus and save it to folder with original corpus
    print("-- -- Creating validation corpus")
    corpus_train, corpus_val = train_test_split(
        corpus_df, test_size=val_size, random_state=42)
    corpus_train.reset_index(drop=True, inplace=True)
    outFile = corpusFile.parent.joinpath('corpus_val.txt')
    corpus_df_to_mallet(corpus_val, outFile)
    
    # Initialize the RepeatedKFold cross-validation object:
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
    
    # Iterate over the hyperparameters and perform cross-validation:
    print("-- -- Validation starts...")
    
    best_score = 0
    best_params = {}
    scores = []
    hyperparams = []
    
    for i, (ntopics, alpha, opt_int) in enumerate(itertools.product(*grid_params)):
        
        fold_scores = []
        
        print("*"*80)
        print(f"-- -- Training model with hyperparameter combination {i} of {len(list(itertools.product(*grid_params)))}")
        print("*"*80)
        
        # Set training parameters
        training_params['ntopics'] = ntopics
        training_params['alpha'] = alpha
        training_params['optimize_interval'] = opt_int
    
        for j, (train_index, test_index) in enumerate(rkf.split(corpus_train)):
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                print("#"*15)
                print(f"Fold {j} of {rkf.get_n_splits()}")
                print("#"*15)
            
                corpus_train_copy = corpus_train.copy() 
                X_train, X_test = corpus_train_copy.iloc[train_index], corpus_train_copy.iloc[test_index]
                corpus_train_file = \
                    corpusFile.parent.joinpath(f'corpus_train_{i}_{j}.txt')
                corpus_df_to_mallet(X_train, corpus_train_file)
                X_test['text'] = X_test['text'].apply(lambda x: x.split())
                
                # Train model
                tm_wrapper = TMWrapper()
                name = f"ntopics_{ntopics}_alpha_{alpha}_optint_{opt_int}_fold_{i}"
                model_path = tm_wrapper.train_root_model(
                    models_folder=models_folder,
                    name=name,
                    path_corpus=corpus_train_file,
                    trainer=trainer,
                    training_params=training_params,
                )
                
                # Calculate coherence score with test corpus
                tm = TMmodel(model_path.joinpath("TMmodel"))
                cohr = tm.calculate_topic_coherence(
                    metrics="c_npmi",
                    reference_text = X_test.text.values.tolist(),
                    aggregated=True,
                )
                
                fold_scores.append(cohr)
                
                # Delete train file
                corpus_train_file.unlink()
        
        # Fold average score
        avg_score = sum(fold_scores)/len(fold_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = {'ntopics': ntopics,
                           'alpha': alpha,
                           'optimize_interval': opt_int}
            
        scores.append(fold_scores)
        hyperparams.append({'ntopics': ntopics,
                            'alpha': alpha,
                            'optimize_interval': opt_int})
    
    
    plt.figure(figsize=(10, 6))

    for i, score in enumerate(scores):
        plt.plot(range(1, len(score)+1), score, label=f'Combination {i+1}')

    plt.xlabel('Fold')
    plt.ylabel('Accuracy Score')
    plt.title('Cross-Validation Scores for Hyperparameter Combinations')
    plt.legend()
    plt.savefig(models_folder.joinpath("plot.png"))  

    return


def _gen_measure_name(coherence_measure, window_size, top_n):
    """
    Make a unique measure name from the arguments
    """
    measure_name = f"{coherence_measure}_win{window_size}_top{top_n}"
    return measure_name


def coherence(
    topics,
    vocab,
    reference_text,
    coherence_measure,
    window_size,
    top_n,
):
    """
    Calculates coherence for a single model
    """
    data_dict = Dictionary([vocab])
    topics = [t[:top_n] for t in topics]

    cm = CoherenceModel(
        topics=topics,
        texts=tqdm(reference_text),
        dictionary=data_dict,
        coherence=coherence_measure,
        window_size=window_size,
    )

    confirmed_measures = cm.get_coherence_per_topic()
    mean = cm.aggregate_measures(confirmed_measures)

    measure_name = _gen_measure_name(coherence_measure, cm.window_size, top_n)
    return measure_name, float(mean), [float(i) for i in confirmed_measures]


def main():

    models_folder = "/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_val_mallet"
    corpusFile = '/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus.txt'
    #grid_params = [
    #    [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150], 
    #    [0.1, 0.5, 1, 5, 10, 20, 50], 
    #    [0, 10]
    #]
    
    grid_params = [
        [5, 10, 20]# 30, 40, 50, 60, 70, 80, 90, 100, 150], 
        [0.1, 0.5]# 1, 5, 10, 20, 50], 
        [0]# 10]
    ]
    run_k_fold(
        models_folder = models_folder, 
        trainer = "mallet", 
        corpusFile = corpusFile, 
        grid_params = grid_params)


if __name__ == '__main__':
    main()
