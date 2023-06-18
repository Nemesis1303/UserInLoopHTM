import pdb
import numpy as np
from src.evaluateMatching.alignment import Alignment

import pandas as pd
from src.utils.misc import unpickler, pickler
import pathlib


# Calculate similarity between topics of submodels and WMD between each submodel topics and the reference topics
al = Alignment()
vs_sims, wmd1, wmd2 = al.do_one_to_one_matching()

df = unpickler(
    '/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/info.pkl')

df['father_model'] = [None]*len(df)
df['alphas'] = [None]*len(df)
df['cohrs'] = [None]*len(df)
df['keywords'] = [None]*len(df)

#  Root model
model_path = pathlib.Path(
    "/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/Scholar_AI_mallet_5_topics/TMmodel")
cohrs = model_path.joinpath("topic_coherence.npy")
alphas = model_path.joinpath("alphas.npy")
keywords = model_path.joinpath("tpc_descriptions.txt")

alphas = np.load(alphas).tolist()
alphas = ', '.join(str(element) for element in alphas)

cohrs = np.load(cohrs).tolist()
cohrs = ', '.join(str(element) for element in cohrs)

with open(keywords, "r") as file:
    lines = file.readlines()
keywords = [line.strip().split(", ") for line in lines]

df.loc[df['model'] == 'Scholar_AI_mallet_5_topics', 'alphas'] = alphas
df.loc[df['model'] == 'Scholar_AI_mallet_5_topics', 'cohrs'] = cohrs
df.loc[df['model'] == 'Scholar_AI_mallet_5_topics', 'keywords'] = [keywords]

# Submodel 1: "HTM-DS_submodel_from_topic4_10_topics"
model_path = pathlib.Path(
    "/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/Scholar_AI_mallet_5_topics/HTM-DS_submodel_from_topic4_10_topics/TMmodel")
cohrs = model_path.joinpath("topic_coherence.npy")
alphas = model_path.joinpath("alphas.npy")
keywords = model_path.joinpath("tpc_descriptions.txt")

alphas = np.load(alphas).tolist()
alphas = ', '.join(str(element) for element in alphas)

cohrs = np.load(cohrs).tolist()
cohrs = ', '.join(str(element) for element in cohrs)

with open(keywords, "r") as file:
    lines = file.readlines()
keywords = [line.strip().split(", ") for line in lines]

df.loc[df['model'] == "HTM-DS_submodel_from_topic4_10_topics",
       'father_model'] = "Scholar_AI_mallet_5_topics"
df.loc[df['model'] == "HTM-DS_submodel_from_topic4_10_topics", 'alphas'] = alphas
df.loc[df['model'] == "HTM-DS_submodel_from_topic4_10_topics", 'cohrs'] = cohrs
df.loc[df['model'] == "HTM-DS_submodel_from_topic4_10_topics",
       'keywords'] = [keywords]

# Submodel 2: "HTM-WS_submodel_from_topic4_10_topics"
model_path = pathlib.Path(
    "/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/Scholar_AI_mallet_5_topics/HTM-WS_submodel_from_topic4_10_topics/TMmodel")
model_path = pathlib.Path(
    "/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/Scholar_AI_mallet_5_topics/HTM-DS_submodel_from_topic4_10_topics/TMmodel")
cohrs = model_path.joinpath("topic_coherence.npy")
alphas = model_path.joinpath("alphas.npy")
keywords = model_path.joinpath("tpc_descriptions.txt")

alphas = np.load(alphas).tolist()
alphas = ', '.join(str(element) for element in alphas)

cohrs = np.load(cohrs).tolist()
cohrs = ', '.join(str(element) for element in cohrs)

with open(keywords, "r") as file:
    lines = file.readlines()
keywords = [line.strip().split(", ") for line in lines]

df.loc[df['model'] == "HTM-WS_submodel_from_topic4_10_topics",
       'father_model'] = "Scholar_AI_mallet_5_topics"
df.loc[df['model'] == "HTM-WS_submodel_from_topic4_10_topics", 'alphas'] = alphas
df.loc[df['model'] == "HTM-WS_submodel_from_topic4_10_topics", 'cohrs'] = cohrs
df.loc[df['model'] == "HTM-WS_submodel_from_topic4_10_topics",
       'keywords'] = [keywords]

pickler('/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/info.pkl', df)

df2 = pd.DataFrame(
    {
        'model_1':"HTM-WS_submodel_from_topic4_10_topics",
        'model_2':"HTM-DS_submodel_from_topic4_10_topics",
        'vs_sims': [vs_sims],
        'wmd_1': [wmd1],
        'wmd_2': [wmd2],
    }
)

pickler('/Users/lbartolome/Documents/GitHub/UserInLoopHTM/data/sims_wmds.pkl', df2)