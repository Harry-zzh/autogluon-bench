
import pandas as pd
import os

def get_last_underscore_before(s):
    parts = s.rpartition('_')
    return parts[0] if parts[1] else ''
 
# download data from https://www.kaggle.com/datasets/gianmarco96/upmcfood101
# unzip
target_dir = "food101/food101_processed"
os.makedirs(target_dir, exist_ok=True)
os.system(f"unzip UPMC-Food101/upmcfood101.zip -d {target_dir}")

# subsample
for file in ["texts/train_titles.csv", "texts/test_titles.csv"]:
    file_path = f"{target_dir}/{file}"
    key = ""
    if 'train' in file:
        output_file_path = f"{target_dir}/train.csv"
        key = 'train'
    else:
        output_file_path = f"{target_dir}/test.csv"
        key = 'test'
    df = pd.read_csv(file_path, header=None)
    df.columns =  ['image', 'text', 'label']
    df = df.groupby(df.columns[-1]).apply(lambda group: group.sample(frac=0.2, random_state=42)).reset_index(drop=True)
    # print(df.columns)
    df["image"] = df["image"].apply(lambda x: f"images/{key}/{get_last_underscore_before(x)}/{x}")
    
    df.to_csv(output_file_path, index=False,)

    print(f"{key} data has been successfully written in {output_file_path}")
    
