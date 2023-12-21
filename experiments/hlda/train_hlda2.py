import logging
import tomotopy as tp
import pathlib
from tqdm import tqdm

import sys
sys.path.append('../..')
from src.utils.misc import mallet_corpus_to_df

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################


path_corpus = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/Datasets/S2CS-AI/models_preproc/iter_0/corpus_train.txt")
df = mallet_corpus_to_df(path_corpus)
df_lemas = df[["text"]].values.tolist()

# HLDA config
tw = tp.TermWeight.ONE  # term weighting scheme in TermWeight
min_cf = 0             # minimum collection frequency of words.
min_df = 0             # minimum document frequency of words.
rm_top = 0             # the number of top words to be removed.
depth = 2              # the maximum depth level of hierarchy between 2 ~ 32767
alpha = 10.0           # document-depth level Dirichlet hyperparameter
eta = 0.1              # hyperparameter of Dirichlet distribution for topic-word
gamma = 1.0            # concentration coeficient of Dirichlet Process
seed = None            # random seed
mycorpus = df_lemas
transform = None       # a callable object to manipulate keyword arguments

mdl = tp.HLDAModel(tw=tp.TermWeight.ONE, min_cf=0, min_df=0,
                   rm_top=0, depth=2, alpha=10.0, eta=0.1, gamma=1.0)

logger.info(f"-- -- HLDA model created.")
logger.info(f"-- -- Adding documents to HLDA model...")

for texts in tqdm(mycorpus):
    mdl.add_doc(texts[0].split())
logger.info(f"-- -- Documents added to HLDA model")
mdl.train(0)
logger.info(f"-- -- HLDA model trained for one epoch.")
print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words))

# Let's train the model
logger.info(f"-- -- Training HLDA model...")
for i in tqdm(range(0, 100, 20)):
    print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(20)
print('Iteration: {:04}, LL per word: {:.4}'.format(1000, mdl.ll_per_word))

# Print summary
mdl.summary()

# Print top 15 words of each topic
for k in range(mdl.k):
    if not mdl.is_live_topic(k):
        continue
    print('Top 10 words of topic #{}'.format(k))
    print(mdl.get_topic_words(k, top_n=15))

# Save model
path_save = "/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/experiments/hlda/output/hlda_ai.bin"
mdl.save(path_save)
