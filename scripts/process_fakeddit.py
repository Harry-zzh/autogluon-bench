import os
import pandas as pd
target_dir = "fakeddit/fakeddit_processed"
os.makedirs(target_dir, exist_ok=True)

# refer to the original dataset link: https://github.com/entitize/Fakeddit
# download "text and metadata" (multimodal_only_samples)
# and "image data" following the README.md

for type in ["train", "validate", "test_public"]:
    tsv_file_path = f'fakeddit/multimodal_{type}.tsv'
    if type == "validate": type = "validation"
    elif type == "test_public": type = "test"
    csv_file_path = f'{target_dir}/{type}.csv'
    df = pd.read_csv(tsv_file_path, delimiter='\t',on_bad_lines='error')

    # subsample 10%/3% of the original data
    if type == "train":
        df = df.groupby(df.columns[-1]).apply(lambda group: group.sample(frac=0.03, random_state=42)).reset_index(drop=True)
    else:
        df = df.groupby(df.columns[-1]).apply(lambda group: group.sample(frac=0.1, random_state=42)).reset_index(drop=True)
    
    imgs = []
    for index, row in df.iterrows():
        imgs.append("images/" + row["id"] + ".jpg")
    df["images"] = imgs
    for index, row in df.iterrows():
        image_path = os.path.join("fakeddit/fakeddit_processed", row['images'])
        if not os.path.exists(image_path):
            df.drop(index, inplace=True)
    
    df.to_csv(csv_file_path, index=False, sep='\t',)
    print(f"{type} data has been successfully written in {csv_file_path}")

