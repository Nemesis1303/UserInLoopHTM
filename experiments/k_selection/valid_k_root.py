import argparse
import itertools
import os
import pathlib
import sys
import warnings
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedKFold, train_test_split

# Add src to path and make imports
sys.path.append('../..')
from src.utils.misc import (
    corpus_df_to_mallet, mallet_corpus_to_df, read_config_experiments)
from src.tmWrapper.tm_wrapper import TMWrapper
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel

def run_k_fold(models_folder: str,
               trainer: str,
               corpusFile: str,
               grid_params: dict,
               training_params: dict,
               val_size: int = 0.3) -> None:
    """Runs repeated k-fold cross-validation for a given corpus and hyperparameter grid.

    Parameters
    ----------
    models_folder : str
        Path to folder where models will be saved.
    trainer : str
        Trainer to use for training the model.
    corpusFile : str
        Path to corpus file.
    grid_params : dict
        Dictionary with hyperparameter grid.
    training_params: dict
        Dictionary with training parameters.
    val_size : int, optional
        Size of the validation set, by default 0.3.
    """

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
    rkf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)#10,5

    # Iterate over the hyperparameters and perform cross-validation:
    print("-- -- Validation starts...")

    best_score = 0
    best_params = {}
    scores = []
    hyperparams = []

    for i, (ntopics, alpha, opt_int) in enumerate(itertools.product(*grid_params)):

        fold_scores = []

        print("*"*80)
        print(
            f"-- -- Training model with hyperparameter combination {i} of {len(list(itertools.product(*grid_params)))}")
        print("*"*80)

        # Set training parameters
        training_params['ntopics'] = ntopics
        training_params['alpha'] = alpha
        training_params['optimize_interval'] = opt_int

        for j, (train_index, test_index) in enumerate(rkf.split(corpus_train)):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                print("#"*80)
                print(
                    f"Fold {j} of {rkf.get_n_splits()} with hyperparameter combination {i} of {len(list(itertools.product(*grid_params)))}")
                print("#"*80)

                corpus_train_copy = corpus_train.copy()
                X_train, X_test = corpus_train_copy.iloc[train_index], corpus_train_copy.iloc[test_index]
                corpus_train_file = \
                    corpusFile.parent.joinpath(f'corpus_train_{i}_{j}.txt')
                corpus_df_to_mallet(X_train, corpus_train_file)
                X_test['text'] = X_test['text'].apply(lambda x: x.split())

                # Train model
                tm_wrapper = TMWrapper()
                name = f"ntopics_{ntopics}_alpha_{alpha}_optint_{opt_int}_fold_{j}"
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
                    metrics=["c_npmi"],
                    reference_text=X_test.text.values.tolist(),
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


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus.txt",
                        help="Path to the training data.")
    parser.add_argument('--models_folder', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_val_mallet",
                        help="Path where the models are going to be saved.")
    parser.add_argument('--trainer', type=str,
                        default="mallet",
                        help="Name of the underlying topic modeling algorithm to be used: mallet|ctm")
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
    
    # grid_params = [
    #    [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150],
    #    [0.1, 0.5, 1, 5, 10, 20, 50],
    #    [0, 10]
    # ]

    grid_params = [
        [5, 10, 20],
        [0.1, 0.5],
        [0]
    ]
    run_k_fold(
        models_folder=args.models_folder,
        trainer=args.trainer,
        corpusFile=args.path_corpus,
        grid_params=grid_params,
        training_params=training_params)


if __name__ == '__main__':
    main()
