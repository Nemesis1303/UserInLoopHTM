import pickle
import topmost
from topmost.preprocessing import Preprocessing
import json
import numpy as np
from topmost import evaluations
import torch

device = "cuda"
dataset_dir = "/export/usuarios_ml4ds/lbartolome/Datasets/S2CS-AI/models_preproc/iter_0/topmost"

# Preprocess raw data
#preprocessing = Preprocessing(vocab_size=50711)
#rst = preprocessing.preprocess_jsonlist(dataset_dir=dataset_dir, label_name="label")
#preprocessing.save(dataset_dir, **rst)

dataset = topmost.data.BasicDataset(dataset_dir, read_labels=True, device=device)

iters_results = {
    "pcc": [],
    "pcd": [],
    "sibling_td": [],
    "pn_cd": []
}
for iter in range(1, 4): # 5 iterations
    model = topmost.models.TraCo(dataset.vocab_size, num_topics_list=[20, 50])
    #model = topmost.models.HyperMiner(vocab_size=dataset.vocab_size, num_topics_list=[20, 50], device=device)
    model = model.to(device)

    trainer = topmost.trainers.HierarchicalTrainer(model, dataset)
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

    # Print results
    print(json.dumps(topic_hierarchy, indent=4))
    print(results)
    
    iters_results["pcc"].append(results["PCC"])
    iters_results["pcd"].append(results["PCD"])
    iters_results["sibling_td"].append(results["Sibling_TD"])
    iters_results["pn_cd"].append(results["PnCD"])

    trainer.get_top_words(annotation=True)

    # Save all relevant objects into a pickle file
    save_data = {
        "model_state_dict": model.state_dict(),
        "trainer_config": {
            "vocab_size": dataset.vocab_size,
            "num_topics_list": [20, 50],
        },
        "top_words": top_words,
        "train_theta": train_theta,
        "test_theta": test_theta,
        "beta_list": beta_list,
        "phi_list": phi_list,
        "annotated_top_words": trainer.get_top_words(annotation=True),
        "topic_hierarchy": topic_hierarchy,
        "hierarchy_quality_results": results,
        "topic_diversity": TD,
    }

    # Save the dataset and preprocessing results for reproducibility
    save_data["dataset"] = {
        "vocab": dataset.vocab,
        "train_bow": dataset.train_bow,
        "test_bow": dataset.test_bow,
    }

    # Save all data to a pickle file
    model_type = "traco"
    dataset_name = "ai"
    path_save = f"/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/{model_type}/{dataset_name}/trained_model_iter_{iter}.pkl"
    
    with open(path_save, "wb") as f:
        pickle.dump(save_data, f)
    print(f"All relevant data saved to {path_save}.")
    
    del model, trainer, train_theta, test_theta, beta_list, phi_list, annotated_top_words
    torch.cuda.empty_cache()
    
# Print average results
print("Average results:")
print(f"PCC: {np.mean(iters_results['pcc'])}")
print(f"PCD: {np.mean(iters_results['pcd'])}")
print(f"Sibling_TD: {np.mean(iters_results['sibling_td'])}")
print(f"PnCD: {np.mean(iters_results['pn_cd'])}")