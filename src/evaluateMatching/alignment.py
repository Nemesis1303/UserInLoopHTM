import pathlib

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import normalize
from tqdm import tqdm
from src.topicmodeling.manageModels import TMmodel


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

        exp_matrix = np.zeros(init_size, dtype=np.float64)
        for i in tqdm(np.arange(init_size[0])):
            for idx, word1 in id2token1.items():
                for j in np.arange(len(id2token2.values())):
                    if list(id2token2.values())[j] == word1:
                        exp_matrix[i, j] = matrix[i][int(idx)]
                        break
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

        Notes
        -----
        In order sim_word_comp to be computed, both topic models should have been trained on the same corpus, using the same vocabulary, i.e., they should have the associated the same words

        Returns
        -------
        selected_worddesc : list(tuple(int, int, float))
            List of tuples with the indices of the topics with the highest similarity and their similarity score
        """

        assert betas1.shape[0] == betas2.shape[0], "Both topic models should have been trained on the same corpus and vocabulary"

        ntopics = betas1.shape[0]
        betas1_aux = betas1[:, np.where(betas1.max(axis=0) > thr)[0]]
        betas2_aux = betas2[:, np.where(betas2.max(axis=0) > thr)[0]]
        js_mat = np.zeros((ntopics, ntopics))
        for k in range(ntopics):
            for kk in range(ntopics):
                js_mat[k, kk] = jensenshannon(
                    betas1_aux[k, :], betas2_aux[kk, :])
        JSsim = 1 - js_mat
        selected_worddesc = self._largest_indices(
            JSsim, ntopics + 2 * npairs)
        selected_worddesc = [(el[0], el[1], el[2].astype(float))
                             for el in selected_worddesc]

        return selected_worddesc

    def _wmd(self, ref_topics, model_topics, n_words: int = 50):

        import gensim.downloader as api
        model = api.load('word2vec-google-news-300')

        model_topics = [el[1].split(', ') for el in model_topics]
        all_dist = np.zeros((len(ref_topics), len(model_topics)))
        for idx1, tpc1 in enumerate(ref_topics):
            for idx2, tpc2 in enumerate(model_topics):
                all_dist[idx1, idx2] = model.wmdistance(
                    tpc1[:n_words], tpc2[:n_words])

        return all_dist

    def do_one_to_one_matching(self,
                               tmModel1: str = "/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/Scholar_AI_mallet_5_topics/HTM-WS_submodel_from_topic4_10_topics",
                               tmModel2: str = "/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/Scholar_AI_mallet_5_topics/HTM-DS_submodel_from_topic4_10_topics",
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
            distrib1 = self._explote_matrix(
                matrix=distrib1,
                init_size=tmModelFather.get_betas().shape,
                id2token1=tm1.get_vocab(),
                id2token2=tmModelFather.get_vocab())

            distrib2 = tm2.get_betas()
            distrib2 = self._explote_matrix(
                matrix=distrib2,
                init_size=tmModelFather.get_betas().shape,
                id2token1=tm2.get_vocab(),
                id2token2=tmModelFather.get_vocab())
        else:
            self._logger.error(
                "Method for calculating similarity not supported")

        # Get topic descriptions
        topic_desc1 = tmModel1.get_tpc_word_descriptions()
        topic_desc2 = tmModel2.get_tpc_word_descriptions()

        # Calculate similarity
        pairs = self._sim_word_comp(betas1=distrib1,
                                    betas2=distrib2,
                                    npairs=len(distrib1))

        new_pairs = [(pair[0], pair[1], pair[2], topic_desc1[i], topic_desc2[i])
                     for i, pair in enumerate(pairs)]

        return new_pairs