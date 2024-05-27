import os
import random
import pandas as pd

target_dir = "aep/aep_processed"
os.makedirs(target_dir, exist_ok=True)

# refer to the original link of the dataset: https://huggingface.co/datasets/sled-umich/Action-Effect
# download the image data
os.system("wget https://huggingface.co/datasets/sled-umich/Action-Effect/resolve/main/action_effect_image_rs.zip")
os.system(f"unzip action_effect_image_rs.zip -d {target_dir}")
# download the annotation action_effect_sentence_phrase.txt from DropBox: 
# https://www.dropbox.com/s/pi1ckzjipbqxyrw/action_effect_sentence_phrase.txt?dl=0

action_desc_dict = {}
with open("action_effect_sentence_phrase.txt", 'r') as f:
    lines = f.readlines()
for line in lines:
    arr = line.split(",")
    if arr[0] not in action_desc_dict:
        action_desc_dict[arr[0]] = []
    action_desc_dict[arr[0]].append(arr[1].strip())
# print(action_desc_dict)

decription_texts = []
effect_images = []
labels = []
files = os.listdir("action_effect_image_rs")
for file_dir in files:
    path = os.path.join("action_effect_image_rs", file_dir)
    
    for label in ["positive"]:
        image_path = os.path.join(path, label)
        image_files = os.listdir(image_path)
        action = file_dir.replace("+", " ")
        decriptions = action_desc_dict[action]
        # print(len(decriptions))
        # print(len(image_files))
        # print()
        sample_image_files = random.sample(image_files, min(len(decriptions), len(image_files)))
        # for each action, randomly sample the corresponding effect descriptions and images
        for decription, sample_image_file in zip(decriptions, sample_image_files):
            decription_texts.append(decription)
            effect_images.append(os.path.join(os.path.join("action_effect_image_rs", file_dir), os.path.join(label, sample_image_file)))
            labels.append(action)


df = pd.DataFrame({
    "descriptions": decription_texts, # effect descriptions
    "images": effect_images, # effect images
    "label": labels # action
})

# split the training set and test set
train_data = pd.DataFrame()
test_data = pd.DataFrame()
label_name = "label"

labels = df[label_name].unique()

for label in labels:
    subset = df[df["label"] == label]
    sample_size = int(0.25 * len(subset))
    test_subset = subset.sample(n=sample_size, random_state=42)
    train_subset = subset.drop(test_subset.index)
    train_data = pd.concat([train_data, train_subset])
    test_data = pd.concat([test_data, test_subset])

train_file = f"{target_dir}/train.csv"
test_file = f"{target_dir}/test.csv"

train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

print(f"Train data has been successfully written in {train_file}")
print(f"Test data has been successfully written in {test_file}")
