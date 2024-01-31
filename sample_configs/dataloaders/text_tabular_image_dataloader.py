import logging
import os

import pandas as pd
import yaml
import numpy as np
import re
from autogluon.bench.utils.dataset_utils import get_data_home_dir
from autogluon.common.loaders import load_zip
from autogluon.common.loaders._utils import download, protected_zip_extraction


def path_expander(path, base_folder):
    path_l = path.split(";")
    return ";".join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])


logger = logging.getLogger(__name__)


class TextTabularImageDataLoader:
    def __init__(self, dataset_name: str, dataset_config_file: str, dataset_id: int = None, split: str = "train"):
        with open(dataset_config_file, "r") as f:
            config = yaml.safe_load(f)

        self.dataset_config = config[dataset_name]
        if split == "val":
            split = "validation"
        if split not in self.dataset_config["splits"]:
            logger.warning(f"Data split {split} not available.")
            self.data = None
            return
        if split == 'test' and self.dataset_config['test_file'] == 'dev':
            split = 'dev'

        
        if "convert_to_log" in dataset_name:
            # 定义正则表达式模式
            pattern = r'(.+?)_convert_to_log'

            # 使用正则表达式匹配字符串
            match = re.match(pattern, dataset_name)

            # 提取匹配的部分
            if match:
                dataset_name = match.group(1)
                print(dataset_name)
            else:
                print("未找到匹配的部分")
        if self.dataset_config['data_folder']:
            self.data_folder = self.dataset_config['data_folder']
        else:
            self.data_folder = f"{dataset_name}_processed"
        self.name = dataset_name
        print(self.data_folder)
        self.sub_dataset_name = None
        if dataset_id is not None:
            self.sub_dataset_name = self.dataset_config['datasets_id'][dataset_id]

        self.sha1sum = self.dataset_config['sha1sum']
        self.split = split
        self.feature_columns = self.dataset_config["feature_columns"]
        self.label_columns = self.dataset_config["label_columns"]
        self.label_col = self.label_columns[0]
        self.ignore_columns = self.dataset_config["ignore_columns"]
        self.image_columns = self.dataset_config["image_columns"]

        url = self.dataset_config["url"]
        file_name = url.split('/')[-1]
        # download_file_path = os.path.join(get_data_home_dir(), file_name)
        # download(url, download_file_path)
        # protected_zip_extraction(download_file_path, sha1_hash=self.sha1sum, folder=self.local_dir())

        if split == 'train':
            self.data = pd.read_csv(os.path.join(self.base_folder(), self.dataset_config["train_file"]))
            if self.dataset_config["train_subset"]:
                num_sample = self.dataset_config["train_subset"]
                self.data = self.data.iloc[:num_sample]
        elif split == 'validation':
            self.data = pd.read_csv(os.path.join(self.base_folder(), self.dataset_config["val_file"]))
        elif split == 'test':
            self.data = pd.read_csv(os.path.join(self.base_folder(), self.dataset_config["test_file"]))
            if self.dataset_config["test_subset"]:
                num_sample = self.dataset_config["test_subset"]
                self.data = self.data.iloc[:num_sample]
        else:
            raise NotImplementedError(f"Unsupported data split: {split}")
        
        if self.feature_columns is None:
            self.feature_columns = list(self.data.columns.difference(self.ignore_columns + self.label_columns))
        
        if dataset_name != "mocheg" and dataset_name != "iqa" and dataset_name != "snli-ve":
            for img_col in self.image_columns:
                self.data[img_col].fillna("", inplace=True)
                self.data[img_col] = self.data[img_col].apply(lambda ele: path_expander(ele, base_folder=self.base_folder()))
        
        if self.dataset_config["convert_to_log"]:
            print("convert_to_log ...")
            for label_col in self.label_columns:
                self.data = self.data.loc[self.data[label_col] > 0]
                self.data[label_col] = np.log(self.data[label_col])

        # filter data
        use_all_images_datasets = ["id_change_detection", "grocery_image"]
        if any([self.name.lower().startswith(per_dataset) for per_dataset in use_all_images_datasets]):
            self.get_df_with_one_label()
        else:
            self.get_df_with_one_image_and_one_label()

    @property
    def problem_type(self):
        return self.dataset_config["problem_type"]

    @property
    def metric(self):
        return self.dataset_config["metric"]
    
    @property
    def loss_func(self):
        return self.dataset_config["loss_func"]
    
    def local_dir(self):
        path = os.path.join(get_data_home_dir(), self.name)
        return path
    
    def base_folder(self):
        if self.sub_dataset_name:
            path = os.path.join(self.local_dir(), self.data_folder, self.sub_dataset_name)
        else:
            path = os.path.join(self.local_dir(), self.data_folder)
        return path
    
    def get_df_with_single_image(self):
        """get a new dataframe in which we take the first image for all image columns"""
        for col in self.image_columns:
            self.data[col] = self.data[col].apply(lambda ele: ele.split(';')[0])
        return self.data
    
    def get_df_with_one_image_and_one_label(self):
        print("filter the dataset to keep only one image column with one image per sample")
        # If some image column has multiple images, take the first image
        self.get_df_with_single_image()
        assert self.label_columns[0] in list(self.data.columns)
        # keep one label column and one image column
        self.data = self.data[
            list(set(self.feature_columns + self.label_columns[:1])
                - set(self.image_columns[1:]))]
        return

    def get_df_with_one_label(self):
        print("filter the dataset to keep all images and one label column")
        if self.label_columns:
            # keep one label column
            self.data = self.data[list(set(self.feature_columns + self.label_columns[:1]))]
        return
