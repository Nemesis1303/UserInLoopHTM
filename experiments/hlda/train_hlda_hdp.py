import logging
import tomotopy as tp
import pathlib
import pickle
from tqdm import tqdm
import sys
import argparse
from time import time

sys.path.append('../..')
from src.utils.misc import mallet_corpus_to_df

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

DEFAULT_CORPUS_PATH = f"/export/usuarios_ml4ds/lbartolome/Datasets/XXX/models_preproc/iter_0/corpus.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="Train HLDA or HDP models with a configurable corpus path.")
    parser.add_argument(
        '--corpus_name', type=str, required=True,
        help="Name of the corpus to train the model on, CORDIS, S2CS-AI or Cancer"
    )
    parser.add_argument(
        '--output_dir', type=str, default='/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models_v3',
        help="Directory to save trained models."
    )
    parser.add_argument(
        '--model_type', type=str, choices=['hlda', 'hdp', "hpam"], required=True,
        help="Type of model to train: 'hlda' or 'hdp'."
    )
    parser.add_argument(
        '--pickle_path', type=str, default='/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/tomo_corpus_objects',
        help="Path to save/load the processed Tomotopy corpus as a pickle file. "
    )
    parser.add_argument(
        '--runs', type=int, default=1,
        help="Number of training runs. Default is 3."
    )
    parser.add_argument("--n_first", type=int, required=False, help="Number of first-level topics for HPAM model.", default=6)
    
    parser.add_argument("--n_second", type=int, required=False, help="Number of second-level topics for HPAM model.", default=10)
    
    parser.add_argument("--iteration", type=int, required=False, help="Iteration number to start training from.")
    return parser.parse_args()

# Function to create or load a Tomotopy corpus
def load_or_create_corpus(corpus_name, pickle_path, output_dir):
    corpus_path = DEFAULT_CORPUS_PATH.replace("XXX", corpus_name)
    pickle_path = f"{pickle_path}/{corpus_name}.pkl"
    
    if pathlib.Path(pickle_path).exists():
        logger.info(f"Loading corpus from pickle file: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            corpus = pickle.load(f)
            
    else:
        logger.info(f"Processing corpus from file: {corpus_path}")
        df = mallet_corpus_to_df(corpus_path)
        corpus = [text[0].split() for text in df[["text"]].values.tolist()]
        with open(pickle_path, 'wb') as f:
            pickle.dump(corpus, f)
        logger.info(f"Corpus saved to pickle file: {pickle_path}")
    
    # reduce the size of the corpus for faster training
    if corpus_name.lower() == "cordis":
        # reduce by a factor of 10
        corpus = corpus[:len(corpus)//10]
    else:
        # reduce by a factor of 100
        corpus = corpus[:len(corpus)//100]
        
    # save corpus size in output directory
    with open(f"{output_dir}/{corpus_name.lower()}_corpus_size.txt", "w") as f:
        f.write(str(len(corpus)))
    return corpus

def train_model(model_type, run_id, corpus, corpus_name, output_dir, n_first=None, n_second=None):
    logger.info(f"-- Training {model_type}, Run #{run_id}...")
    start_time = time()
    
    if model_type == "hlda":
        model = tp.HLDAModel(
            tw=tp.TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, depth=2,
            alpha=1.0, eta=0.1, gamma=0.5
        )
    elif model_type == "hdp":
        initial_k = 6 if corpus_name == "CORDIS" else 20
        model = tp.HDPModel(
            tw=tp.TermWeight.ONE, min_cf=0, min_df=0, rm_top=0,
            alpha=0.1, initial_k=initial_k
        )
    elif model_type == "hpam":
        n_second = n_first * n_second
        model = tp.HPAModel(
            k1=n_first, k2=n_second
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Add documents to the model
    for doc in corpus:
        model.add_doc(doc)
    logger.info(f"-- -- Documents added to {model_type} model")

    # Initial training step
    model.train(0)
    logger.info(f"-- -- {model_type} model initialized.")
    
    # Iterative training
    if model_type == "hdp":
        for i in tqdm(range(0, 100, 20)): #range(0, 100, 20)
        #for i in tqdm(range(0, 1, 1)): #range(0, 100, 20)
            logger.info(f"Iteration: {i}, LL per word: {model.ll_per_word:.4}")
            model.train(20)
        logger.info(f"Final Iteration: 100, LL per word: {model.ll_per_word:.4}")
    else:
        #for i in tqdm(range(0, 1, 1)): #range(0, 100, 20)
        for i in tqdm(range(0, 100, 20)):  # Fewer iterations for faster training
            logger.info(f"Iteration: {i}, LL per word: {model.ll_per_word:.4}")
            model.train(10)
        logger.info(f"Final Iteration: 1000, LL per word: {model.ll_per_word:.4}")

    # Save the model
    corpus_save_key = corpus_name.lower() if corpus_name in ["CORDIS", "Cancer"] else "ai"
    output_path = f"{output_dir}/run{run_id}.bin"
    model.save(output_path)
    logger.info(f"-- {model_type} model saved at {output_path}")

    # Log summary and top words
    model.summary()
    for k in range(model.k):
        logger.info(f"Top 10 words of topic #{k}")
        logger.info(model.get_topic_words(k, top_n=15))

    end_time = time()
    logger.info(f"-- {model_type}, Run #{run_id} completed in {end_time - start_time:.2f} seconds.")
    # save time to file
    with open(f"{output_dir}/run{run_id}_time.txt", "w") as f:
        f.write(str(end_time - start_time))

if __name__ == "__main__":
    args = parse_args()
    pickle_path = pathlib.Path(args.pickle_path)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mycorpus = load_or_create_corpus(args.corpus_name, pickle_path, args.output_dir)

    for run_id in range(1, args.runs + 1):
        iteration_save = args.iteration if args.iteration is not None else run_id
        train_model(args.model_type, iteration_save, mycorpus, args.corpus_name,output_dir, args.n_first, args.n_second)