import json
import pandas as pd
import os

def extract_all_labels():
    return [
        "Smears",
        "Loaded Language",
        "Name calling/Labeling",
        "Glittering generalities (Virtue)",
        "Appeal to fear/prejudice",
        "Transfer",
        "Appeal to (Strong) Emotions",
        "Doubt",
        "Exaggeration/Minimisation",
        "Whataboutism",
        "Slogans",
        "Flag-waving",
        "Straw Man",
        "Causal Oversimplification",
        "Appeal to authority",
        "Thought-terminating cliché",
        "Black-and-white Fallacy/Dictatorship",
        "Reductio ad hitlerum",
        "Repetition",
        "Obfuscation, Intentional vagueness, Confusion",
        "Bandwagon",
        "Presenting Irrelevant Data (Red Herring)"
    ]

def read_txt_to_json(txt_filename):
    with open(txt_filename, 'r') as txt_file:
        data = txt_file.read()
    return json.loads(data)

def update_csv_with_new_txt(csv_filename, txt_filename, dir_name):
    # txt -> json
    items = read_txt_to_json(txt_filename)

    # get labels
    all_labels = extract_all_labels()

    # create DataFrame
    df = pd.DataFrame(items)

    # encoder label
    labels_df = pd.DataFrame(0, index=df.index, columns=all_labels)
    for i, item in df.iterrows():
        for label in item['labels']:
            labels_df.at[i, label] = 1

    df = pd.concat([df[['id', 'text', 'image']], labels_df], axis=1)

    # rename image column
    df['image'] = df['image'].apply(lambda x: f'{dir_name}/{x}')

    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename)
        for label in existing_df.columns[3:]:  # Skip 'id', 'text', 'image' columns
            if label not in df.columns:
                df[label] = 0

        for label in df.columns[3:]:  # Skip 'id', 'text', 'image' columns
            if label not in existing_df.columns:
                existing_df[label] = 0

        df = df[existing_df.columns]

        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df

    # drop duplicates
    combined_df['text_clean'] = combined_df['text'].str.strip()
    combined_df = combined_df.drop_duplicates(subset=['text_clean'], keep='last')
    combined_df = combined_df.drop(columns=['text_clean'])

    # write csv
    combined_df.to_csv(csv_filename, index=False)
    

target_data_dir = "persuasive_techniques/persuasive_techniques_processed"

# download and unzip data
os.makedirs(target_data_dir, exist_ok=True)
os.system("git clone https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus.git")
names = ['training_set_task3', 'dev_set_task3', 'test_set_task3']
os.system(f"unzip -d {target_data_dir} SEMEVAL-2021-task6-corpus/data/{names[0]}.zip")
os.system(f"unzip -d {target_data_dir} SEMEVAL-2021-task6-corpus/data/{names[1]}.zip")
os.system(f"unzip -d {target_data_dir} SEMEVAL-2021-task6-corpus/data/{names[2]}.zip")

txt_filename1 = f'{target_data_dir}/{names[0]}/{names[0]}.txt'
json_filename1 = f'{target_data_dir}/{names[0]}/{names[0]}.json'
csv_filename1 = f'{target_data_dir}/train_processed.csv'
txt_filename2 = f'{target_data_dir}/{names[1]}_labeled/{names[1]}_labeled.txt'
json_filename2 = f'{target_data_dir}/{names[1]}_labeled/{names[1]}_labeled.txt'
txt_filename3 = f'{target_data_dir}/{names[2]}/{names[2]}.txt'
json_filename3 = f'{target_data_dir}/{names[2]}/{names[2]}.txt'
csv_filename3 = f'{target_data_dir}/test_processed.csv'
os.system(f"rm -rf {csv_filename1}")
os.system(f"rm -rf {csv_filename3}")

# Process the training txt file and dev txt file, combine them to build a new training set
print(names)
update_csv_with_new_txt(csv_filename1, txt_filename1, names[0])
update_csv_with_new_txt(csv_filename1, txt_filename2, names[1]+"_labeled")

# Process the test txt file
update_csv_with_new_txt(csv_filename3, txt_filename3, names[2])

print(f"Train data has been successfully written in {csv_filename1}")
print(f"Test data has been successfully written in {csv_filename3}")
