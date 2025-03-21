import os
import json
import pickle
import pathlib
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from topmost.preprocessing import Preprocessing
from topmost import evaluations, data, models, trainers
from tqdm import tqdm
import argparse


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train topic models with configurable parameters.")
    parser.add_argument(
        "--corpus_name", 
        type=str, required=False, help="Name of the dataset to use (e.g., 'CORDIS').", default="CORDIS")
    parser.add_argument(
        "--model_type", 
        type=str, required=True, choices=["traco", "hyperminer"],help="Algorithm to use for training.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations for training.")
    parser.add_argument("--iteration", type=int, help="Iteration number to start training from.", required=False)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training ('cuda' or 'cpu').")
    return parser.parse_args()

# Function to convert Mallet corpus to DataFrame
def mallet_corpus_to_df(corpusFile: pathlib.Path):
    corpus = [line.rsplit(' 0 ')[1].strip() for line in open(corpusFile, encoding="utf-8").readlines()]
    indexes = [line.rsplit(' 0 ')[0].strip() for line in open(corpusFile, encoding="utf-8").readlines()]
    return pd.DataFrame({'id': indexes, 'text': corpus})

# Main training pipeline
def train_pipeline(corpus_name, algorithm, iterations, device, iteration_save):
    dataset_dir = f"/export/usuarios_ml4ds/lbartolome/Datasets/{corpus_name}/models_preproc/iter_0/topmost"
    if corpus_name.lower() == "cordis":
        corpus_save_name = "cordis"
    elif corpus_name.lower() == "s2cs-ai":
        corpus_save_name = "ai"
    elif corpus_name.lower() == "cancer":
        corpus_save_name = "cancer"
    model_save_dir = f"/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models_v5_topmost/{algorithm}/{corpus_save_name}"
    os.makedirs(model_save_dir, exist_ok=True)
    # Preprocessing only on the first iteration
    if not os.path.exists(os.path.join(dataset_dir, "train.jsonlist")):
        print(f"Preprocessing dataset: {corpus_name}")
        corpus_file = pathlib.Path(dataset_dir).parent / "corpus.txt"
        df = mallet_corpus_to_df(corpus_file)

        # Adjust train/test split based on dataset
        max_train_keep = len(df) // (10 if corpus_name.lower() == "cordis" else 100)
        max_test_keep = max_train_keep * 10 // 90

        train_df = df.iloc[:max_train_keep]
        test_df = df.iloc[max_train_keep:max_test_keep + max_train_keep]

        # Generate labeled JSON files
        possible_labels = ["A", "B", "C"]
        train_json = train_df.apply(lambda row: {"label": random.choice(possible_labels), "text": row['text']}, axis=1).tolist()
        test_json = test_df.apply(lambda row: {"label": random.choice(possible_labels), "text": row['text']}, axis=1).tolist()

        with open(os.path.join(dataset_dir, 'train.jsonlist'), 'w') as train_file:
            for item in train_json:
                train_file.write(json.dumps(item) + '\n')

        with open(os.path.join(dataset_dir, 'test.jsonlist'), 'w') as test_file:
            for item in test_json:
                test_file.write(json.dumps(item) + '\n')

        # Bag-of-Words processing
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        bow = vectorizer.fit_transform(pd.concat([train_df, test_df]).text.values.tolist())
        preprocessing = Preprocessing(vocab_size=bow.shape[1])
        
        # save vocabulary size, number of training and test documents
        with open(os.path.join(model_save_dir, 'preprocessing.json'), 'w') as f:
            json.dump({"vocab_size": bow.shape[1], "num_train_docs": len(train_df), "num_test_docs": len(test_df)}, f)
        
        rst = preprocessing.preprocess_jsonlist(dataset_dir=dataset_dir, label_name="label")
        preprocessing.save(dataset_dir, **rst)

    # Load dataset
    dataset = data.BasicDataset(dataset_dir, read_labels=True, device=device)

    # Iterative training
    iters_results = {"pcc": [], "pcd": [], "sibling_td": [], "pn_cd": []}
    for iter in range(1, iterations + 1):
        print(f"Training {algorithm} on {corpus_name}, iteration {iter}")
        if corpus_name.lower() == "cordis":
            num_topics_list= [6, 60] #[10, 50]
        else: 
            num_topics_list=[20, 200]
        
        if algorithm == "traco":
            model = models.TraCo(dataset.vocab_size, num_topics_list=num_topics_list).to(device)
        elif algorithm == "hyperminer":
            model = models.HyperMiner(vocab_size=dataset.vocab_size, num_topics_list=num_topics_list, device=device).to(device)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        trainer = trainers.HierarchicalTrainer(model, dataset, epochs=200)
        top_words, train_theta = trainer.train()

        # Export theta (doc-topic distributions)
        train_theta, test_theta = trainer.export_theta()

        # Compute evaluations
        TD = evaluations.multiaspect_topic_diversity(top_words)
        print(f"TD: {TD}")

        beta_list = trainer.get_beta()
        phi_list = trainer.get_phi()
        annotated_top_words = trainer.get_top_words(annotation=True)

        # Create reference bag-of-words
        reference_bow = np.concatenate((dataset.train_bow, dataset.test_bow), axis=0)

        # Evaluate hierarchy quality
        results, topic_hierarchy = evaluations.hierarchy_quality(
            dataset.vocab, reference_bow, annotated_top_words, beta_list, phi_list
        )

        # Print and save results
        print(json.dumps(topic_hierarchy, indent=4))
        print(results)
        iters_results["pcc"].append(results["PCC"])
        iters_results["pcd"].append(results["PCD"])
        iters_results["sibling_td"].append(results["Sibling_TD"])
        iters_results["pn_cd"].append(results["PnCD"])

        save_data = {
            "model_state_dict": model.state_dict(),
            "trainer_config": {"vocab_size": dataset.vocab_size, "num_topics_list": [10, 50]},
            "top_words": top_words,
            "train_theta": train_theta,
            "test_theta": test_theta,
            "beta_list": beta_list,
            "phi_list": phi_list,
            "annotated_top_words": annotated_top_words,
            "topic_hierarchy": topic_hierarchy,
            "hierarchy_quality_results": results,
            "topic_diversity": TD,
            "dataset": {"vocab": dataset.vocab, "train_bow": dataset.train_bow, "test_bow": dataset.test_bow},
        }

        iter_for_saving = iter if iteration_save is None else iteration_save
        save_path = os.path.join(model_save_dir, f"{corpus_save_name}_trained_model_iter_{iter_for_saving}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Saved model to {save_path}")

    # Print average results
    print(f"Average results for {algorithm} on {corpus_name}:")
    print(f"PCC: {np.mean(iters_results['pcc'])}")
    print(f"PCD: {np.mean(iters_results['pcd'])}")
    print(f"Sibling_TD: {np.mean(iters_results['sibling_td'])}")
    print(f"PnCD: {np.mean(iters_results['pn_cd'])}")

# Entry point
if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args.corpus_name, args.model_type, args.iterations, args.device, args.iteration)
