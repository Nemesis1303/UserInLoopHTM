import pandas as pd
import random

df_aux = pd.read_csv("output/candidates_cordis.csv")
df_aux = df_aux[((df_aux.id != 47012) & (df_aux.id != 40234))]

dfs = [df_aux,
       pd.read_csv("output/candidates_cancer.csv"),
       pd.read_csv("output/candidates_ai.csv"),
       ]

print(dfs)

appens = ["COR", "CAN", "AI"]

questions_per_worker = 30

text = ""
# Get 1/3 of the questions from each model type
for el in range(len(dfs)):
    # Select n ids so that each pair of documents associated with id_pair in a corpus is given to the same worker
    df = dfs[el]
    pair_ids = df.id_pair.unique()
    pair_idxs = random.sample(range(len(pair_ids)), int(questions_per_worker/3))
    for pair in pair_idxs:
        df_pair = df[df.id_pair == pair_ids[pair]]
        try:
            df_high = df_pair[df_pair.opt_select == "high"].iloc[0]
            df_low = df_pair[df_pair.opt_select == "low"].iloc[0]

            text += f"{appens[el]}{df_high['id']}. {df_high['clean_text']}\n[[Randomize]]\n\n{df_high['ws_label']} {df_high['ws_desc']}\n{df_high['ds_label']} {df_high['ds_desc']}\n\n"

            text += f"{appens[el]}{df_low['id']}. {df_low['clean_text']}\n[[Randomize]]\n\n{df_low['ws_label']} {df_low['ws_desc']}\n{df_low['ds_label']} {df_low['ds_desc']}\n\n"

            text += "[[PageBreak]]\n"
        except Exception as e:
            print(f"Could not generate question for pair {pair_ids[pair]}: ")

# Specify the file path where you want to save the text
file_path = "output/output.txt"

# Write the text to the file
with open(file_path, "w") as file:
    file.write(text)

print(f"Text has been written to {file_path}")
