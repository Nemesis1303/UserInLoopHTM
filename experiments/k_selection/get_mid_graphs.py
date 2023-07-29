import pathlib
import sys
import pandas as pd
import scipy.sparse as sparse

sys.path.append('../..')
from src.topicmodeler.src.topicmodeling.manageModels import TMmodel

path_models = "/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_val_mallet"

path_models = pathlib.Path(path_models)

# Read validation corpus
corpusFile = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus_val.txt")
corpus = [line.rsplit(' 0 ')[1].strip() for line in open(
        corpusFile, encoding="utf-8").readlines()]
indexes = [line.rsplit(' 0 ')[0].strip() for line in open(
    corpusFile, encoding="utf-8").readlines()]
corpus_dict = {
    'id': indexes,
    'text': corpus
}
X_val = pd.DataFrame(corpus_dict)
X_val['text'] = X_val['text'].apply(lambda x: x.split())

values = []
for entry in path_models.iterdir():
    TMfolder = entry.joinpath("TMmodel")

    ntopics = int(entry.as_posix().split("ntopics_")[1].split("_alpha")[0])
    alpha = float(entry.as_posix().split("ntopics_")[1].split("_alpha_")[1].split("_optint_")[0])
    opt_int = int(entry.as_posix().split("ntopics_")[1].split("_alpha_")[1].split("_optint_")[1].split("_fold_")[0])
    fold = int(entry.as_posix().split("ntopics_")[1].split("_alpha_")[1].split("_optint_")[1].split("_fold_")[1])


    # Read thetas
    thetas = sparse.load_npz(TMfolder.joinpath('thetas.npz'))
    disp_perc = 100 * thetas.count_nonzero() / (thetas.shape[0] * thetas.shape[1])

    # Calculate coherence score with test corpus
    tm = TMmodel(TMfolder)
    cohr = tm.calculate_topic_coherence(
        metrics=["c_npmi"],
        reference_text=X_val.text.values.tolist(),
        aggregated=True,
    )
    values.append([ntopics, alpha, opt_int, fold, disp_perc, cohr])

df_results = pd.DataFrame(values, columns=['ntopics', 'alpha', 'opt_int', 'fold', 'disp_perc', 'cohr'])
df_results.to_csv(pathlib.Path(path_models).joinpath("mid_results.csv"))    