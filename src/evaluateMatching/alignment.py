import json
import pathlib

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import normalize

from src.topicmodeler.src.topicmodeling.manageModels import TMmodel


class Alignment(object):
    def __init__(self, logger=None) -> None:

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level=logging.INFO)
            self._logger = logging.getLogger("Alignment")

    def _largest_indices(self,
                         ary: np.array,
                         n: int):  # -> list(tuple(int, int, float)):
        """Returns the n largest indices from a numpy array.

        Parameters
        ----------
        ary : np.array
            Array of values
        n : int
            Number of indices to be returned

        Returns
        -------
        selected_idx : list(tuple(int, int, float))
            List of tuples with the indices of the topics with the highest similarity and their similarity score
        """
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        idx0, idx1 = np.unravel_index(indices, ary.shape)
        idx0 = idx0.tolist()
        idx1 = idx1.tolist()
        selected_idx = []
        for id0, id1 in zip(idx0, idx1):
            if id0 < id1:
                selected_idx.append((id0, id1, ary[id0, id1]))
        return selected_idx

    def _explote_matrix(self, matrix, init_size, id2token1, id2token2):

        # Initialize matrix of dims (n_topics, n_words)
        exp_matrix = np.zeros(init_size, dtype=np.float64)
        print("The shape after explote is: ", exp_matrix.shape)

        # Get idx of the words in initial vocabulary (id2token2) that are in the new vocabulary (id2token1)
        matching_ids = [
            int(key) for key, value in id2token2.items() if value in id2token1.values()]

        # Fill in betas of matching words
        exp_matrix[:, matching_ids] = matrix[:, :len(matching_ids)]

        # Normalize matrix
        exp_matrix = normalize(exp_matrix, axis=1, norm='l1')

        return exp_matrix

    def _sim_word_comp(self,
                       betas1: np.array,
                       betas2: np.array,
                       npairs: int,
                       thr: float = 1e-3):  # -> list(tuple(int, int, float)):
        """Calculates similarities between word distributions of two topic models based on their word composition using Jensen-Shannon distance.

        Parameters
        ----------
        betas1 : np.array
            Topic-word distribution of topic model 1, of dimensions (n_topics, n_words)
        betas2 : np.array
            Topic-word distribution of topic model 2, of dimensions (n_topics, n_words)
        npairs : int
            Number of pairs of words to be returned
        thr : float
            Threshold for removing words with low probability

        Returns
        -------
        selected_worddesc : list(tuple(int, int, float))
            List of tuples with the indices of the topics with the highest similarity and their similarity score
        """

        ntopics = betas1.shape[0]
        ntopics2 = betas2.shape[0]

        js_mat = np.zeros((ntopics, ntopics2))
        for k in range(ntopics):
            for kk in range(ntopics2):
                js_mat[k, kk] = jensenshannon(
                    betas1[k, :], betas2[kk, :])
        JSsim = 1 - js_mat

        return JSsim

    def _wmd(self, model_topics, n_words: int = 15):

        import gensim.downloader as api
        model = api.load('word2vec-google-news-300')

        with open("/Users/lbartolome/Documents/GitHub/UserInLoopHTM/src/evaluateMatching/ai_reference_topics.json", "r") as f:
            data = json.load(f)
        df = pd.DataFrame.from_records(data)
        ref_topics = df["words"].values.tolist()
        from nltk import download
        from nltk.corpus import stopwords
        download('stopwords')  # Download stopwords list.
        stop_words = stopwords.words('english')

        def preprocess(sentence):
            return [w.lower() for w in sentence if w not in stop_words]

        ref_topics = [preprocess(tpc) for tpc in ref_topics]
        all_dist = np.zeros((len(ref_topics), len(model_topics)))
        for idx1, tpc1 in enumerate(ref_topics):
            for idx2, tpc2 in enumerate(model_topics):

                all_dist[idx1, idx2] = model.wmdistance(
                    tpc1[:n_words], tpc2[:n_words])
        all_dist[np.isinf(all_dist)] = 2

        # distances = np.mean(all_dist, axis=0)
        return all_dist

    def do_one_to_one_matching(self,
                               tmModel1: str,
                               tmModel2: str,
                               father: bool = False,
                               method: str = "sim_word_comp"):

        # Check if path to TMmodels exist
        assert pathlib.Path(tmModel1).exists(), self._logger.error(
            "Topic model 1 does not exist")
        assert pathlib.Path(tmModel2).exists(), self._logger.error(
            "Topic model 2 does not exist")

        # Create TMmodel objects
        tm1 = TMmodel(pathlib.Path(tmModel1).joinpath("TMmodel"))
        tm2 = TMmodel(pathlib.Path(tmModel2).joinpath("TMmodel"))

        # Load betas or thetas according to the method chosen
        if method == "sim_word_comp":
            if not father:
                tmModelFather = \
                    TMmodel(pathlib.Path(tmModel1).parent.joinpath("TMmodel"))

            distrib1 = tm1.get_betas()
            print("The shape of distrib1 is: ", distrib1.shape)
            distrib1 = self._explote_matrix(
                matrix=distrib1,
                init_size=(len(distrib1), tmModelFather.get_betas().shape[1]), # ntopics child x nwords father
                id2token1=tm1.get_vocab(),
                id2token2=tmModelFather.get_vocab())

            distrib2 = tm2.get_betas()
            print("The shape of distrib2 is: ", distrib2.shape)
            distrib2 = self._explote_matrix(
                matrix=distrib2,
                init_size=(len(distrib2), tmModelFather.get_betas().shape[1]),
                id2token1=tm2.get_vocab(),
                id2token2=tmModelFather.get_vocab())
        else:
            self._logger.error(
                "Method for calculating similarity not supported")

        # Get topic descriptions
        topic_desc1 = [el[1].split(', ')
                       for el in tm1.get_tpc_word_descriptions()]
        topic_desc2 = [el[1].split(', ')
                       for el in tm2.get_tpc_word_descriptions()]

        wmd1 = self._wmd(topic_desc1, n_words=15)
        wmd2 = self._wmd(topic_desc2, n_words=15)

        # Calculate similarity
        # Between both submodels
        vs_sims = self._sim_word_comp(betas1=distrib1,
                                      betas2=distrib2,
                                      npairs=len(distrib1))

        return vs_sims, wmd1, wmd2
