import pathlib
import sys
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split

# Add src to path
sys.path.append('../..')
from src.utils.misc import mallet_corpus_to_df, corpus_df_to_mallet

def fit():
    # @ TODO: fit the model (train with root model)
    pass

def run_k_fold(corpusFile,val_size=0.2):
    
    scores=[]
    kFold=KFold(n_splits=10,random_state=42,shuffle=True)
    
    # Read corpus as df
    corpusFile = pathlib.Path(corpusFile)
    corpus_df = mallet_corpus_to_df(corpusFile)
    
    # Create val corpus and save it to folder with original corpus
    corpus_train, corpus_val = train_test_split(
        corpus_df, test_size=val_size, random_state=42)
    outFile = corpusFile.parent.joinpath('corpus_val.txt')
    corpus_df_to_mallet(corpus_val, outFile)
    
    #for train_index,test_index in kFold.split(corpus_df):
    #    print("Test Index: ", test_index)
    ##    print("Train Index: ", train_index, "\n")
        
        #X_train, X_test, y_train, y_test = X[train_index], X[test_index], y#[train_index], y[test_index]
        #knn.fit(X_train, y_train)
        #scores.append(knn.score(X_test, y_test))
    #knn.fit(X_train,y_train)
    #print(np.mean(scores))
    ##scores.append(knn.score(X_test,y_test))
    #cross_val_score(knn, X, y, cv=10)
    
    
    pass


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
    
    corpusFile = '/Volumes/usuarios_ml4ds/lbartolome/Datasets/CORDIS/models_preproc/iter_0/corpus.txt'
    run_k_fold(corpusFile)
    
if __name__ == '__main__':
    main()
