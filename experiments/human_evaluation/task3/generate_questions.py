import pandas as pd
import random
import numpy as np

print("Generating questions...")
df = pd.read_csv("output/cordis.csv")
# Remove candidates selected for example questions
df = df[((df['candidate_id'] != 138) & (df['candidate_id'] != 24) & (df['candidate_id'] != 134) )]
print(df)

questions_per_worker = 30

pair_idxs = {
    "hlda": set(),
    "ws": set(),
    "ds": set(),
}

# Add unique indices to each set
for model_type in pair_idxs.keys():
    high_cohrs_idx = set(df[(df.model_type == model_type) & (df.cohrs_cv > 0.5)].index)
    low_cohrs_idx = set(df[(df.model_type == model_type) & (df.cohrs_cv < 0.5)].index)

    # Ensure uniqueness by subtracting already selected indices
    pair_idxs[model_type].update(random.sample(high_cohrs_idx - pair_idxs[model_type], questions_per_worker // 3 // 2))
    pair_idxs[model_type].update(random.sample(low_cohrs_idx - pair_idxs[model_type], questions_per_worker // 3 // 2))

print(pair_idxs)

pair_idxs = {key: list(value) for key, value in pair_idxs.items()}
print(pair_idxs)

question_answer_pairs = []

# Create a list of question-answer pairs for all model types
choices = ["Very Related", "Somewhat Related", "Not Related"]
for model_type, indices in pair_idxs.items():
    for pair in indices:
        df_pair = df[(df.model_type == model_type) & (df.candidate_id == pair)]
        words_list = df_pair.word_description.iloc[0].split(", ")
        random.shuffle(words_list)
        words_list = ", ".join([f"{word}" for word in words_list])
        answers =  "".join([f"{choice}\n" for choice in choices])
        question_answer_pairs.append((f"QR{df_pair.candidate_id.iloc[0]}. {words_list}", answers))

# Shuffle the list of question-answer pairs
random.shuffle(question_answer_pairs)

text = ""
id = 0

# Iterate through the shuffled question-answer pairs
for question, answers in question_answer_pairs:
    text += f"{question}\n\n{answers}\n\n"
    
    if id != 0 and id % 5 == 0:
        text += "[[PageBreak]]\n"
    id += 1
    
file_path = "output/cordis2.txt"

# Write the text to the file
with open(file_path, "w") as file:
    file.write(text)

print(f"Text has been written to {file_path}")