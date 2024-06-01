import argparse
import csv
import importlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Union
import pandas as pd
from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
from autogluon_local.core.src.autogluon.core.metrics import make_scorer
from autogluon_local.multimodal.src.autogluon.multimodal import MultiModalPredictor
from autogluon_local.multimodal.src.autogluon.multimodal import __version__ as ag_version
from autogluon_local.multimodal.src.autogluon.multimodal.constants import IMAGE_SIMILARITY, IMAGE_TEXT_SIMILARITY, OBJECT_DETECTION, TEXT_SIMILARITY
import yaml
from autogluon_local.multimodal.src.autogluon.multimodal.models.utils import get_pretrained_tokenizer
from autogluon_local.core.src.autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
import numpy as np
from PIL import Image
from autogluon_local.core.src.autogluon.core.metrics import get_metric
from autogluon_local.multimodal.src.autogluon.multimodal.utils.misc import logits_to_prob

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _flatten_dict(data):
    flattened = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value))
        else:
            flattened[key] = value
    return flattened


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset that has been registered with multimodal_dataset_registry.",
    )
    parser.add_argument("--framework", type=str, help="Framework (and) branch/version.")
    parser.add_argument("--benchmark_dir", type=str, default="debug", help="Directory to save benchmarking run.")
    parser.add_argument("--metrics_dir", type=str, default="debug", help="Directory to save benchmarking metrics.")
    parser.add_argument("--constraint", type=str, default=None, help="AWS resources constraint setting.")
    parser.add_argument("--params", type=str, default=None, help="AWS resources constraint setting.")
    parser.add_argument(
        "--custom_dataloader", type=str, default=None, help="Custom dataloader to use in the benchmark."
    )
    parser.add_argument("--custom_metrics", type=str, default=None, help="Custom metrics to use in the benchmark.")
    parser.add_argument(
        "--eval_model_path", type=str, default=None, help="Model checkpoint path for evaluation."
    )

    ### model parmas:
    parser.add_argument(
        "--weight_decay", type=float, default=0.001
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="It is used only when lr_choice is layerwise_decay"
    )
    parser.add_argument(
        "--lr_schedule", type=str, default="cosine_decay"
    )
    parser.add_argument(
        "--warmup_steps", type=float, default=0.1
    )
    parser.add_argument(
        "--gradient_clip_val", default=1.
    )
    parser.add_argument(
        "--top_k_average_method", type=str, default="greedy_soup"
    )
    parser.add_argument(
        "--peft", type=str, default="null", help="Can be 'bit_fit' (only finetune bias), 'norm_fit' (finetune the normalization terms + bias terms), lora (LoRA Adaptations only), lora_bias (LoRA Adaptation + bit_fit), lora_norm (LoRA Adaptation + norm_fit), or null"
    )
    parser.add_argument(
        "--categorical_convert_to_text", action='store_false', default=True, help="convert categorical columns to text or not."
    )
    parser.add_argument(
        "--categorical_convert_to_text_use_header", action='store_true', default=False, help="integrate header information or not."
    )
    parser.add_argument(
        "--categorical_convert_to_text_use_header_template",type=str, default="list"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="num of training epochs."
    )
    parser.add_argument(
        "--hf_text_ckpt", type=str, default="microsoft/deberta-v3-base", help="text ckpt"
    )
    parser.add_argument(
        "--lora_r", type=int, default=8
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument(
        "--resume", action="store_true", default=False
    )
    parser.add_argument(
        "--use_image_aug", action='store_false', default=True,
    )    
    parser.add_argument(
        "--text_trivial_aug_maxscale", type=float, default=0.0
    )

    parser.add_argument(
        "--ft_transformer_ckpt_name", action="store_true", default=False
    )


    parser.add_argument(
        "--use_fusion_transformer", action="store_true", default=False
    )
    parser.add_argument(
        "--fusion_transformer_concat_all_tokens", action="store_true", default=False
    )
    parser.add_argument(
        "--early_fusion", action="store_true", default=False
    )
    parser.add_argument(
        "--sequential_fusion", action="store_true", default=False
    )
    parser.add_argument(
        "--clip_fusion_mlp", action="store_true", default=False, help="Use clip for late fusion model."
    )
    parser.add_argument(
        "--clip_best_quality", action="store_true", default=False, help="Use clip best quality"
    )
    
    parser.add_argument(
        "--clip_high_quality", action="store_true", default=False, help="Use clip high quality"
    )

    parser.add_argument(
        "--use_different_lr_for_each_modality", action="store_true", default=False, 
    )
    parser.add_argument(
        "--image_lr",  type=float, default=0.0
    )
    parser.add_argument(
        "--text_lr",  type=float, default=0.0
    )
    parser.add_argument(
        "--tabular_lr",  type=float, default=0.0
    ) 
    
    parser.add_argument(
        "--numerical_convert_to_text", action='store_true', default=False, help="convert numerical columns to text or not."
    )
    parser.add_argument(
        "--numerical_convert_to_text_use_header", action='store_true', default=False, help="integrate header information or not."
    )
    parser.add_argument(
        "--max_text_len", type=int, default=512, help="max text length."
    )
    parser.add_argument(
        "--auxiliary_weight", type=float, default=0.1, help="auxiliary loss weight for unimodal models."
    )

    parser.add_argument(
        "--get_dataset_info", action='store_true', default=False, help="get dataset information."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )

    

    parser.add_argument(
        "--use_image_only", action='store_true', default=False,
    )
    parser.add_argument(
        "--use_text_only", action='store_true', default=False,
    )
    parser.add_argument(
        "--use_tabular_only", action='store_true', default=False,
    )

    parser.add_argument(
        "--no_hf_text_insert_sep",action='store_false', default=True,
    )

    parser.add_argument(
        "--use_miss_token_embed",action='store_true', default=False,
    )

    parser.add_argument(
        "--use_miss_token_embed_text",action='store_true', default=False,
    )

    parser.add_argument(
        "--use_miss_token_embed_image",action='store_true', default=False,
    )

    parser.add_argument(
        "--use_miss_token_embed_numerical",action='store_true', default=False,
    )
    
    parser.add_argument(
        "--LeMDA", action='store_true', default=False,
    )
    parser.add_argument(
        "--LeMDA_arch", type=str, default="mlp_vae",
    )
    parser.add_argument(
        "--LeMDA_layer", type=int, default=4,
    )
    parser.add_argument(
        "--modality_drop_rate", type=float, default=0.
    )

    parser.add_argument(
        "--self_distill", action='store_true', default=False,
    )

    parser.add_argument(
        "--alignment_loss", type=str, default=None,
    )

    parser.add_argument(
        "--contrastive_loss", type=str, default=None,
    )
    parser.add_argument(
        "--contrastive_loss_w", type=float, default=0.1,
    )

    parser.add_argument(
        "--use_ensemble", action='store_true', default=False, help="ensemble."
    )
    parser.add_argument(
        "--use_avg_ensemble", action='store_true', default=False, help="avg ensemble."
    )

    parser.add_argument(
        "--use_llama", action='store_true', default=False, help="use fusion transformer llama."
    )
    parser.add_argument(
        "--use_llama_7B", action='store_true', default=False, help="use fusion transformer llama."
    )
    parser.add_argument(
        "--no_use_cate_miss_embed",  action='store_true', default=False, help="naive baseline that not using missing embed for categorical."
    )
    parser.add_argument(
        "--manifold_mixup",  action='store_true', default=False, 
    )
    parser.add_argument(
        "--manifold_mixup_a", type=float, default=2.
        
    )
    parser.add_argument(
        "--mixup",  action='store_true', default=False,
        
    )
    
    
    

    args = parser.parse_args()
    return args


def load_dataset(dataset_name: str, custom_dataloader: dict = None):  # dataset name
    """Loads and preprocesses a dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        custom_dataloader (dict): A dictionary containing information about a custom dataloader to use. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test datasets.
    """
    splits = ["train", "val", "test"]
    data = {}
    if dataset_name in multimodal_dataset_registry.list_keys():
        logger.info(f"Loading dataset {dataset_name} from multimodal_dataset_registry")
        for split in splits:
            data[split] = multimodal_dataset_registry.create(dataset_name, split)
    elif custom_dataloader is not None:
        logger.info(f"Loading dataset {dataset_name} from custom dataloader {custom_dataloader}.")
        custom_dataloader_file = custom_dataloader.pop("dataloader_file")
        class_name = custom_dataloader.pop("class_name")
        spec = importlib.util.spec_from_file_location(class_name, custom_dataloader_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        custom_class = getattr(module, class_name)
        for split in splits:
            data[split] = custom_class(dataset_name=dataset_name, split=split, **custom_dataloader)
    else:
        raise ModuleNotFoundError(f"Dataset Loader for dataset {dataset_name} is not available.")

    return data.values()


def load_custom_metrics(custom_metrics: dict):
    """Loads a custom metrics and convert it to AutoGluon Scorer.

    Args:
        custom_metrics (dict): A dictionary containing information about a custom metrics to use. Defaults to None.

    Returns:
        scorer (Scorer)
            scorer: An AutoGluon Scorer object to pass to MultimodalPredictor.
    """

    try:
        custom_metrics_path = custom_metrics.pop("metrics_path")
        func_name = custom_metrics.pop("function_name")
        spec = importlib.util.spec_from_file_location(func_name, custom_metrics_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score_func = getattr(module, func_name)

        scorer = make_scorer(
            name=func_name,
            score_func=score_func,
            **custom_metrics,  # https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-metric.html
        )
    except:
        raise ModuleNotFoundError(f"Unable to load custom metrics function {func_name} from {custom_metrics_path}.")

    return scorer


def save_metrics(metrics_path: str, metrics: dict):
    """Saves evaluation metrics to a JSON file.

    Args:
        metrics_path (str): The path to the directory where the metrics should be saved.
        metrics: The evaluation metrics to save.

    Returns:
        None
    """
    if metrics is None:
        logger.warning("No metrics were created.")
        return

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    file = os.path.join(metrics_path, "results.csv")
    flat_metrics = _flatten_dict(metrics)
    field_names = flat_metrics.keys()

    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerow(flat_metrics)
    logger.info("Metrics saved to %s.", file)
    f.close()


def run(
    dataset_name: Union[str, dict],
    framework: str,
    benchmark_dir: str,
    metrics_dir: str,
    constraint: Optional[str] = None,
    params: Optional[dict] = None,
    custom_dataloader: Optional[dict] = None,
    custom_metrics: Optional[dict] = None,
    eval_model_path: Optional[str] = None,
    resume: bool = False
):
    """Runs the AutoGluon multimodal benchmark on a given dataset.

    Args:
        dataset_name (Union[str, dict]): Dataset that has been registered with multimodal_dataset_registry.

                            To get a list of datasets:

                            from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
                            multimodal_dataset_registry.list_keys()

        benchmark_dir (str): The path to the directory where benchmarking artifacts should be saved.
        constraint (str): The resource constraint used by benchmarking during AWS mode, default: None.
        params (str): The multimodal params, default: {}.
        custom_dataloader (dict): A dictionary containing information about a custom dataloader to use. Defaults to None.
                                To define a custom dataloader in the config file:

                                custom_dataloader:
                                    dataloader_file: path_to/dataloader.py   # relative path to WORKDIR
                                    class_name: DataLoaderClass
                                    dataset_config_file: path_to/dataset_config.yaml
                                    **kwargs (of DataLoaderClass)
        custom_metrics (dict): A dictionary containing information about a custom metrics to use. Defaults to None.
                                To define a custom metrics in the config file:

                                custom_metrics:
                                    metrics_path: path_to/metrics.py   # relative path to WORKDIR
                                    function_name: custom_metrics_function
                                    **kwargs (of autogluon.core.metrics.make_scorer)
    Returns:
        None
    """
    train_data, val_data, test_data = load_dataset(dataset_name=dataset_name, custom_dataloader=custom_dataloader)
    try:
        label_column = train_data.label_columns[0]
    except (AttributeError, IndexError):  # Object Detection does not have label columns
        label_column = None
    if params is None:
        params = {}
    if eval_model_path != None:
        predictor_args = {
            "label": label_column,
            "problem_type": train_data.problem_type,
            "presets": params.pop("presets", None),
        }
    else:
        predictor_args = {
            "label": label_column,
            "problem_type": train_data.problem_type,
            "presets": params.pop("presets", None),
            "path": os.path.join(benchmark_dir, "models"),
        }

    if val_data is not None:
        predictor_args["eval_metric"] = val_data.metric

    if train_data.problem_type == IMAGE_SIMILARITY:
        predictor_args["query"] = train_data.image_columns[0]
        predictor_args["response"] = train_data.image_columns[1]
        predictor_args["match_label"] = train_data.match_label
    elif train_data.problem_type == IMAGE_TEXT_SIMILARITY:
        predictor_args["query"] = train_data.text_columns[0]
        predictor_args["response"] = train_data.image_columns[0]
        del predictor_args["label"]
    elif train_data.problem_type == TEXT_SIMILARITY:
        predictor_args["query"] = train_data.text_columns[0]
        predictor_args["response"] = train_data.text_columns[1]
        predictor_args["match_label"] = train_data.match_label
    elif train_data.problem_type == OBJECT_DETECTION:
        predictor_args["sample_data_path"] = train_data.data

    metrics_func = None
    if custom_metrics is not None and custom_metrics["function_name"] == train_data.metric:
        metrics_func = load_custom_metrics(custom_metrics=custom_metrics)

    # if train_data.loss_func is not None:
    #     params['hyperparameters']['optimization.loss_function'] = train_data.loss_func

    predictor = MultiModalPredictor(**predictor_args)

    get_dataset_info = params.pop("get_dataset_info")
    use_ensemble = params.pop("use_ensemble")
    if use_ensemble:
        use_avg_ensemble = params.pop("use_avg_ensemble")

    if get_dataset_info:

        def get_data_info(train_data, label_col): # 虽然形参名是train_data，但是并不是说只能传train data
            train_tokens = {} # 
            train_null_tokens = 0
            for index, row in train_data.data.iterrows():
                for col_name in train_data.data.columns:
                    if col_name == label_col or 'image' in column_types[col_name]: continue
                    col_text = row[col_name]
                    # if column_types[col_name] != 'text':
                    #     continue
                    if pd.isna(col_text):
                        col_text  = ""
                        train_null_tokens += 1
                        # continue
                    col_tokens = tokenizer.encode(
                        str(col_text),
                        add_special_tokens=False,
                        truncation=False,
                    )
                    if index not in train_tokens:
                        train_tokens[index] = {}
                    # train_tokens[index] += (len(col_tokens) + 1)  # 统计每一行的text总长度
                    if column_types[col_name] not in train_tokens[index]:
                        train_tokens[index][column_types[col_name]] = 0 # 每种类型都需要统计一下长度
                    train_tokens[index][column_types[col_name]] += (len(col_tokens) + 1) 
            return train_tokens, train_null_tokens

        def get_text_len(data_tokens): # 得到拼接后的text长度
            col_types = ['text', 'categorical', 'numerical',]
            res_dict = {} # 记录每个col type下，seq len的相关信息

            text_seq_len = []
            cat_seq_len = []
            num_seq_len = []

            for col_type in col_types:
                if col_type not in data_tokens[0]: # 不存在这一列
                    res_dict[col_type] = {}
                    res_dict[col_type]["max_seq_len"] = "/"
                    res_dict[col_type]["min_seq_len"] = "/"
                    res_dict[col_type]["greater_than_512_ratio"] = "/"
                    continue
                seq_len = []
                
                for i in range(len(data_tokens)):
                    try:
                        seq_len.append(data_tokens[i][col_type])
                    except Exception:
                        print(f"第{i}行, col_type为{col_type}无法读取。") # 因为有一些行读不出来，可能会导致i不存在
                        continue
                max_seq_len = max(seq_len)
                min_seq_len = min(seq_len) 
                if col_type == 'text': text_seq_len = seq_len
                elif col_type == 'categorical': cat_seq_len = seq_len
                else: num_seq_len = seq_len

                res_dict[col_type] = {}
                res_dict[col_type]["max_seq_len"] = max_seq_len
                res_dict[col_type]["min_seq_len"] = min_seq_len
                a = sum([i > 512 for i in seq_len])
                res_dict[col_type]["greater_than_512_ratio"] = f"{a}({np.around( a / len(seq_len) * 100, 2)}\%)"

            # 开始进行text和categorical长度的合并
            col_name = "Text"
            cur_list = None

            if(len(text_seq_len)):
                if cur_list == None:
                    cur_list = text_seq_len
                    
            if(len(cat_seq_len)): # 如果存在categorical列
                col_name += "+Cate"
                if cur_list == None:
                    cur_list = cat_seq_len
                else:
                    cur_list = [cur_list[i] + cat_seq_len[i] for i in range(len(cur_list))]
                    
                res_dict[col_name] = {}
                res_dict[col_name]["max_seq_len"] = max(cur_list)
                res_dict[col_name]["min_seq_len"] = min(cur_list)
                a = sum([i > 512 for i in cur_list])
                res_dict[col_name]["greater_than_512_ratio"] = f"{a}({np.around( a / len(cur_list) * 100, 2)}\%)"
            else:
                col_name += "+Cate"
                res_dict[col_name] = {}
                res_dict[col_name]["max_seq_len"] = "/"
                res_dict[col_name]["min_seq_len"] = "/"
                res_dict[col_name]["greater_than_512_ratio"] = "/"

            if (len(num_seq_len)): # 如果存在numerical列
                col_name += "+Num"
                if cur_list == None:
                    cur_list = num_seq_len
                else:
                    cur_list = [cur_list[i] + num_seq_len[i] for i in range(len(cur_list))]
                
                res_dict[col_name] = {}
                res_dict[col_name]["max_seq_len"] = max(cur_list)
                res_dict[col_name]["min_seq_len"] = min(cur_list)
                a = sum([i > 512 for i in cur_list])
                res_dict[col_name]["greater_than_512_ratio"] =  f"{a}({np.around( a / len(cur_list) * 100, 2)}\%)"
            else:
                col_name += "+Num"
                res_dict[col_name] = {}
                res_dict[col_name]["max_seq_len"] = "/"
                res_dict[col_name]["min_seq_len"] = "/"
                res_dict[col_name]["greater_than_512_ratio"] = "/"

            return res_dict

        def get_image_missing_ratio(data_tokens, column_types, missing_ratios):
            fail_tokens = 0
            for col in data_tokens.columns:
                if "image" in column_types[col]:
                    for a in data_tokens[col]:
                        try:
                            p = Image.open(a)
                        except Exception:
                            fail_tokens += 1
                    break
            missing_ratios[col] = fail_tokens / len(data_tokens)
            return missing_ratios



        def get_type_missing_ratios(missing_ratios, column_types):
            type_missing_ratios = {}
            for col, missing_rate in missing_ratios.items():
                if column_types[col] not in type_missing_ratios:
                    type_missing_ratios[column_types[col]] = []
                type_missing_ratios[column_types[col]].append(missing_rate)
            return type_missing_ratios

        predictor._learner.prepare_train_tuning_data(train_data=train_data.data, tuning_data=val_data.data, seed=params["seed"], holdout_frac=None)
        train_data.data = predictor._learner._train_data
        val_data.data = predictor._learner._tuning_data
        train_data.data = predictor._learner._train_data.reset_index(drop=True)
        val_data.data = predictor._learner._tuning_data.reset_index(drop=True)


        column_types = []
        predictor._learner.infer_column_types(column_types=column_types)
        # 输出当前数据的col types
        column_types = predictor._learner._column_types

        img_num = sum('image' in v for v in column_types.values())
        text_num = sum('text' in v for v in column_types.values())
        cal_num = sum(v == 'categorical' for v in column_types.values())
        numer_num = sum(v == 'numerical' for v in column_types.values())

        if column_types[label_column] == 'categorical': cal_num -= 1
        if column_types[label_column] == 'text': text_num -= 1
        if column_types[label_column] == 'numerical': numer_num -= 1

        #  输出缺失比例
        train_missing_ratios = {col: train_data.data[col].isna().sum() / len(train_data.data) for col in train_data.data.columns}
        val_missing_ratios = {col: val_data.data[col].isna().sum() / len(val_data.data) for col in val_data.data.columns}
        test_missing_ratios =  {col: test_data.data[col].isna().sum() / len(test_data.data) for col in test_data.data.columns}
        # 进一步对image处理
        train_missing_ratios = get_image_missing_ratio(train_data.data, column_types, train_missing_ratios)
        val_missing_ratios = get_image_missing_ratio(val_data.data, column_types, val_missing_ratios)
        test_missing_ratios = get_image_missing_ratio(test_data.data, column_types, test_missing_ratios)

        print("train_missing_ratios: ", train_missing_ratios)
        print("val_missing_ratios: ", val_missing_ratios)
        print("test_missing_ratios: ", test_missing_ratios)

        # 输出每种type的缺失比例(image_text_tabular)
        train_type_missing_ratios =  get_type_missing_ratios(train_missing_ratios, column_types)
        val_type_missing_ratios =  get_type_missing_ratios(val_missing_ratios, column_types)
        test_type_missing_ratios =  get_type_missing_ratios(test_missing_ratios, column_types)
        for cate_type, value in train_type_missing_ratios.items():
            print(f"Missing ratios of {cate_type} in training: {np.round(np.mean(value)*100, 3)}%.")
            v = [np.round(vv*100, 3) for vv in value]
            print(f"Missing ratios of each col of {cate_type} in training: {v}%.")
        print()
        for cate_type, value in val_type_missing_ratios.items():
            print(f"Missing ratios of {cate_type} in validation: {np.round(np.mean(value)*100, 3)}%.")
            v = [np.round(vv*100, 3) for vv in value]
            print(f"Missing ratios of each col of {cate_type} in validation: {v}%.")
        print()
        for cate_type, value in test_type_missing_ratios.items():
            print(f"Missing ratios of {cate_type} in testing: {np.round(np.mean(value)*100, 3)}%.")
            v = [np.round(vv*100, 3) for vv in value]
            print(f"Missing ratios of each col of {cate_type} in testing: {v}%.")
        print()


        tokenizer = get_pretrained_tokenizer(
            tokenizer_name="hf_auto", # 
            checkpoint_name='microsoft/deberta-v3-base', # 
            use_fast=True,
        )

        # 获取基本信息
        train_tokens, train_null_tokens = get_data_info(train_data, label_column)
        test_tokens, test_null_tokens = get_data_info(test_data, label_column)

        # 获取拼接后的text长度

        train_res_dict = get_text_len(train_tokens)
        test_res_dict = get_text_len(test_tokens)
            

        print("dataset_name: ", dataset_name)
        # print("Dataset ID & #Train & #Test & #Img. & #Text. & #Cat. & #Num. & #Train Text Len. > 512 & #Train Text Len. <= 512 & #Test Text Len. > 512 & #Test Text Len. <= 512")
        # print(f"{dataset_name} & {len(train_data.data)} & {len(test_data.data)} & {img_num} & {text_num} & {cal_num} & {numer_num} & {max_train_512_num} & {min_train_512_num} & {max_test_512_num} & {min_test_512_num} ")
        # 这里的text len是拼接过后的长度。
        print("Dataset ID & \#Train & \#Test & \#Img. & \#Text. & \#Cat. & \#Num. "
              "& Train Max T Len. & Train Min T Len. & \#Train T Len. \\textgreater 512 & \#Train T+C Len. \\textgreater 512 & \#Train T+C+N Len. \\textgreater 512 "
              "& Test Max T Len. & Test Min T Len. & \#Test T Len. \\textgreater 512 & \#Test T+C Len. \\textgreater 512 & \#Test T+C+N Len. \\textgreater 512 ")
        print(f"{dataset_name} & {len(train_data.data)} & {len(test_data.data)} & {img_num} & {text_num} & {cal_num} & {numer_num} "
              f"& {train_res_dict['text']['max_seq_len']} & {train_res_dict['text']['min_seq_len']} & {train_res_dict['text']['greater_than_512_ratio']} & {train_res_dict['Text+Cate']['greater_than_512_ratio']}  & {train_res_dict['Text+Cate+Num']['greater_than_512_ratio']} "
              f"& {test_res_dict['text']['max_seq_len']} & {test_res_dict['text']['min_seq_len']} & {test_res_dict['text']['greater_than_512_ratio']} & {test_res_dict['Text+Cate']['greater_than_512_ratio']} & {test_res_dict['Text+Cate+Num']['greater_than_512_ratio']} ")
        return
    elif use_ensemble: # 希望在validation data上evaluate
        zeroshot_configs = [] # 最后要选择的模型config，在我的场景里就是model ckpt name
        
        if dataset_name in ["fake", "qaa", "qaq", "airbnb","channel", "cloth"]:
            prefix_str = f"/home/ubuntu/drive2/ag_bench_runs/multimodal/{dataset_name}/top_k_average_method_greedy_soup/gradient_clip_val_1.0/weight_decay_0.001/warmup_steps_0.1/lr_schedule_cosine_decay/lr_decay_0.9/"
            all_configs = [
                # baseline+
                f"convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/",
                # lf-transformer
                f"convert_to_text_False/ft_transformer_pretrained_False/use_fusion_transformer_True/auxiliary_weight_0.0/max_epochs_20/",
                # lf-aligned
                f"convert_to_text_False/ft_transformer_pretrained_False/use_clip_fusion_mlp/clip_fusion_mlp_quality_high/auxiliary_weight_0.0/max_epochs_20/",
                # lf-llm
                f"convert_to_text_False/ft_transformer_pretrained_False/use_fusion_transformer_True/fusion_transformer_concat_all_tokens_True/auxiliary_weight_0.0/max_epochs_20/use_llama7B_fusion/",
                # early fusion
                f"convert_to_text_False/ft_transformer_pretrained_False/early_fusion_True/auxiliary_weight_0.0/max_epochs_20/",
                # lf-sequential fusion
                f"convert_to_text_False/ft_transformer_pretrained_False/sequential_fusion_True/auxiliary_weight_0.0/max_epochs_20/",
                # positive loss
                f"convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/KL_feature_align_loss/",
                # pos-neg loss
                f"convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/contra_fea_contra_loss/contrastive_loss_w_1.0/",
                # convert-categorical 
                f"convert_to_text_True/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/categorical_template_latex/no_hf_text_insert_sep_False/",
                # input aug
                f"convert_to_text_False/ft_transformer_pretrained_False/text_trivial_aug_maxscale_0.1/auxiliary_weight_0.0/max_epochs_20/",
                # fea independent aug
                f"convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/manifold_mixup/",
                # fea joint aug
                f"convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/LeMDA/lemda_layer_6/",
                
            ]
            if dataset_name in ["airbnb", "channel", "cloth"]:
                all_configs.append(
                    # convert numerical
                    "convert_to_text_False/ft_transformer_pretrained_False/convert_to_text_numerical/auxiliary_weight_0.0/max_epochs_20/"
                )
            if dataset_name in ["fake", "airbnb", "cloth"]:
                all_configs.append(
                    # modality drop=0.3
                    "convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/modality_drop_rate_0.3/"
                )
                if dataset_name in ["airbnb"]:
                    # miss embed
                    all_configs.append(
                        "convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/use_miss_token_True/use_miss_token_True_numerical/"
                    )


        elif dataset_name in ["persuasive_techniques",  "Memotion", "UPMC-Food101","action_effect_pred", "fakeddit"]:
            prefix_str = f"/home/ubuntu/drive2/ag_bench_runs/multimodal/{dataset_name}/top_k_average_method_greedy_soup/gradient_clip_val_1.0/warmup_steps_0.1/lr_schedule_cosine_decay/weight_decay_0.001/lr_decay_0.9/"
            all_configs = [
                # baseline+
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/",
                # lf-transformer
                f"convert_to_text_False/no_img_aug/use_fusion_transformer_True/epoch_20/auxiliary_weight_0.0/",
                # lf-aligned
                f"convert_to_text_False/no_img_aug/epoch_20/use_clip_fusion_mlp/clip_fusion_mlp_quality_high/auxiliary_weight_0.0/",
                # lf-llm
                f"convert_to_text_False/no_img_aug/use_fusion_transformer_True/epoch_20/fusion_transformer_concat_all_tokens_True/auxiliary_weight_0.0/use_llama7B_fusion/",
                # early fusion
                f"convert_to_text_False/no_img_aug/early_fusion_True/epoch_20/auxiliary_weight_0.0/",
                # lf-sequential fusion
                f"convert_to_text_False/no_img_aug/epoch_20/sequential_fusion/auxiliary_weight_0.0/",
                # positive loss
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/KL_feature_align_loss/",
                # pos-neg loss
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/contra_fea_contra_loss/contrastive_loss_w_1.0/",
                # input aug
                f"convert_to_text_False/text_trivial_aug_maxscale_0.1/epoch_20/auxiliary_weight_0.0/",
                # fea independent aug
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/manifold_mixup/",
                # fea joint aug
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/LeMDA/lemda_layer_6/",
                
            ]
            if dataset_name in ["Memotion", "fakeddit"]:
                # modality dropout=0.3
                all_configs.append(
                    "convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/modality_drop_rate_0.3/"
                )
                # learnable embed
                all_configs.append(
                    "convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/use_miss_token_True/"
                )
        elif dataset_name in   ["CCD", "skin_cancer",  "wikiart", "CD18_convert_to_log", "DVM-CAR_convert_to_log",  ]:
            prefix_str = f"ag_bench_runs/multimodal/{dataset_name}/top_k_average_method_greedy_soup/gradient_clip_val_1.0/warmup_steps_0.1/lr_schedule_cosine_decay/weight_decay_0.001/lr_decay_0.9/"
            all_configs = [
                # baseline+
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/",
                # lf-transformer
                f"convert_to_text_False/use_fusion_transformer_True/no_img_aug/epoch_20/auxiliary_weight_0.0/",
                # lf-aligned
                f"convert_to_text_False/no_img_aug/epoch_20/use_clip_fusion_mlp/clip_fusion_mlp_quality_high/auxiliary_weight_0.0/",
                # lf-llm
                f"convert_to_text_False/use_fusion_transformer_True/no_img_aug/epoch_20/fusion_transformer_concat_all_tokens_True/auxiliary_weight_0.0/use_llama7B_fusion/",
                # early fusion
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/early_fusion_True/",
                # lf-sequential fusion
                f"convert_to_text_False/no_img_aug/epoch_20/sequential_fusion/auxiliary_weight_0.0/",
                # positive loss
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/KL_feature_align_loss/",
                # pos-neg loss
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/contra_fea_contra_loss/contrastive_loss_w_1.0/",
                # convert-categorical 
                f"no_img_aug/epoch_20/auxiliary_weight_0.0/categorical_template_latex/no_hf_text_insert_sep_False/",
                # input aug
                f"convert_to_text_False/epoch_20/auxiliary_weight_0.0/",
                # fea independent aug
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/manifold_mixup/",
                # fea joint aug
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/LeMDA/lemda_layer_6/",
                
            ]
            if dataset_name in ["skin_cancer", "CD18_convert_to_log"]:
                # convert numerical
                all_configs.append("convert_to_text_False/no_img_aug/epoch_20/convert_to_text_numerical/")
            if dataset_name in ["skin_cancer", "CD18_convert_to_log", "DVM-CAR"]:
                all_configs.append(
                    # modality drop=0.3
                    "convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/modality_drop_rate_0.3/"
                )

                # miss embed
                all_configs.append(
                    "convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/use_miss_token_True/"
                )

                
        else:
            prefix_str = f"ag_bench_runs/multimodal/{dataset_name}/top_k_average_method_greedy_soup/gradient_clip_val_1.0/warmup_steps_0.1/lr_schedule_cosine_decay/weight_decay_0.001/lr_decay_0.9/"
            all_configs = [
                # baseline+
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/",
                # lf-transformer
                f"convert_to_text_False/use_fusion_transformer_True/no_img_aug/epoch_20/auxiliary_weight_0.0/",
                # lf-aligned
                f"convert_to_text_False/no_img_aug/epoch_20/use_clip_fusion_mlp/clip_fusion_mlp_quality_high/auxiliary_weight_0.0/",
                # lf-llm
                f"convert_to_text_False/use_fusion_transformer_True/no_img_aug/epoch_20/fusion_transformer_concat_all_tokens_True/auxiliary_weight_0.0/use_llama7B_fusion/",
                # early fusion
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/early_fusion_True/",
                # lf-sequential fusion
                f"convert_to_text_False/no_img_aug/epoch_20/sequential_fusion/auxiliary_weight_0.0/",
                # positive loss
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/KL_feature_align_loss/",
                # pos-neg loss
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/contra_fea_contra_loss/contrastive_loss_w_1.0/",
                # convert-categorical 
                f"no_img_aug/epoch_20/auxiliary_weight_0.0/categorical_template_latex/no_hf_text_insert_sep_False/",
                # convert numerical
                "convert_to_text_False/no_img_aug/epoch_20/convert_to_text_numerical/",
                # input aug
                f"convert_to_text_False/epoch_20/auxiliary_weight_0.0/",
                # fea independent aug
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/manifold_mixup/",
                # fea joint aug
                f"convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/LeMDA/lemda_layer_6/",
                
            ]

            if dataset_name in ["petfinder", "covid-chestxray-dataset", "seattle_airbnb_convert_to_log", "KARD"]:
                all_configs.append(
                    # modality drop=0.3
                    "convert_to_text_False/no_img_aug/epoch_20/auxiliary_weight_0.0/modality_drop_rate_0.3/"
                )

                if dataset_name in ["covid-chestxray-dataset", "seattle_airbnb_convert_to_log", ]:
                    # miss embed
                    all_configs.append(
                        "convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/use_miss_token_True/use_miss_token_True_numerical/"
                    )
                if dataset_name in ["KARD"]:
                    # miss embed
                    all_configs.append(
                        "convert_to_text_False/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/use_miss_token_True/use_miss_token_True_image/"
                    )

        if params["seed"] == 0:
            seed_str = ""
        else:
            seed_str = f"seed_{params['seed']}/"
        for i in range(len(all_configs)):
            all_configs[i] = prefix_str+all_configs[i]+ seed_str + "run1/models/model.ckpt"


        num_zeroshot = len(all_configs)
        problem_type = train_data.problem_type

        def score(predictor, config_selected, test_only=False, ensemble_weights=None):
            pred_val = []
            pred_test = []
            for config_s in config_selected:
                basedir = os.path.dirname(config_s)
                if os.path.exists(os.path.join(basedir, "preds_val.npy")) and os.path.exists(os.path.join(basedir, "preds_test.npy")):
                    preds_val = np.load(os.path.join(basedir, "preds_val.npy"))
                    y_val = np.load(os.path.join(basedir, "gt_val.npy"))
                    preds_test = np.load(os.path.join(basedir, "preds_test.npy"))
                    y_test = np.load(os.path.join(basedir, "gt_test.npy"))
                    # print(f"{basedir} already have pred and gt.")
                else:
                    predictor = predictor.load(config_s) # eval_model_path
                    training_duration = 0.
                    predictor._learner.prepare_train_tuning_data(train_data=train_data.data, tuning_data=val_data.data, seed=params["seed"], holdout_frac=None)
                    train_data.data = predictor._learner._train_data
                    val_data.data = predictor._learner._tuning_data
                    train_data.data = predictor._learner._train_data.reset_index(drop=True)
                    val_data.data = predictor._learner._tuning_data.reset_index(drop=True)

                    evaluate_args = {
                        "data": val_data.data,
                        "label": label_column,
                        "metrics": val_data.metric if metrics_func is None else metrics_func,
                        "use_ensemble": True
                    }
                    scores, preds_val, y_val = predictor.evaluate(**evaluate_args)
                    np.save(os.path.join(basedir, "preds_val.npy"), preds_val)
                    np.save(os.path.join(basedir, "gt_val.npy"), y_val)
                    print("validatioin metric:", scores)


                    evaluate_args["data"] = test_data.data
                    scores, preds_test, y_test = predictor.evaluate(**evaluate_args)
                    np.save(os.path.join(basedir, "preds_test.npy"), preds_test)
                    np.save(os.path.join(basedir, "gt_test.npy"), y_test)
                    print("test metric: ", scores)

                pred_val.append(preds_val)
                pred_test.append(preds_test)


            model_list = config_selected
            pred_val = np.stack(pred_val) # y_val不用stack
            pred_test = np.stack(pred_test)
            val_metric = get_metric(metric=val_data.metric, problem_type=problem_type )

            ## weight ensemble
            weighted_ensemble = EnsembleSelection(
                ensemble_size=10,
                problem_type=problem_type,
                metric=val_metric,
                # **self.ensemble_method_kwargs,
            )

            if test_only: # 只需要test
                weighted_ensemble.weights_ = ensemble_weights
                y_test_pred, y_pred_proba = weighted_ensemble.predict(pred_test)
                if problem_type in ["binary"]:
                    y_test_pred = list(logits_to_prob(y_test_pred)[:,1])
                err = val_metric.error(y_test, y_test_pred)
                return err, ensemble_weights
            
            # if problem_type in ["binary", "multiclass"]:
            #     weighted_ensemble.fit(predictions=logits_to_prob(pred_val), labels=y_val) # 这个pred_val应该是model_list里的所有model的pred
            # else:
            weighted_ensemble.fit(predictions=pred_val, labels=y_val)

            y_val_pred, y_pred_proba = weighted_ensemble.predict(pred_val)
            # _calculate_regret方法没问题，算的结果和直接调用error一样，但是在weighted_ensemble.fit里返回的又不一样了。
            # weighted_ensemble._calculate_regret(y_true=y_val, y_pred_proba=y_pred_proba, metric=weighted_ensemble.metric, sample_weight=None)
            
            if problem_type in ["binary"]:
                y_val_pred = list(logits_to_prob(y_val_pred)[:,1])

            err = val_metric.error(y_val, y_val_pred)
            ensemble_weights: np.array = weighted_ensemble.weights_
            # rank = compute_rank_mean(err)
            return err, ensemble_weights
            
        def _select_sequential(configs: list, prior_configs: list, prior_best_score=None):
            best_next_config = None
            best_ensemble_weights = None
            # todo could use np.inf but would need unit-test (also to check that ray/sequential returns the same selection)
            best_score = 999999999
            for config in configs:
                config_selected = prior_configs + [config]
                config_score, ensemble_weights = score(predictor, config_selected) # 在这里会进行weight ensemble，一组组config遍历。得到的config_score 其实是算出来的rank
                if config_score < best_score:
                    best_score = config_score
                    best_next_config = config
                    best_ensemble_weights = ensemble_weights
            return best_next_config, best_score, best_ensemble_weights

        iteration = 0
        if use_ensemble and use_avg_ensemble:
            zeroshot_configs = all_configs
            best_ensemble_weights = [1. / len(all_configs)] * len(all_configs)
            test_score, test_ensemble_weights = score(predictor, zeroshot_configs, ensemble_weights=best_ensemble_weights, test_only=True)
            print("test error: ", test_score)
            print("test metric: ", 1 - test_score)
            print("best_ensemble_weights: ", best_ensemble_weights)
            return
        while len(zeroshot_configs) < num_zeroshot: # 确定一下最终是由几个model进行ensemble
            # greedily search the config that would yield the lowest average rank if we were to evaluate it in combination
            # with previously chosen configs.

            valid_configs = [c for c in all_configs if c not in zeroshot_configs]
            if not valid_configs:
                break
            if iteration == 0:
                prior_best_score = None
            iteration += 1

            time_start = time.time()
            # 再研究一下选择。怎么样避免选择进更差的config。
            best_next_config, best_train_score, best_ensemble_weights = _select_sequential(valid_configs, zeroshot_configs, prior_best_score=prior_best_score)
            time_end = time.time()
            prior_best_score = best_train_score

            zeroshot_configs.append(best_next_config)
            fit_time = time_end - time_start
            msg = f'{iteration}\t: Train: {round(best_train_score, 2)}'

            # test_score = config_scorer_test.score(zeroshot_configs)
            test_score, test_ensemble_weights = score(predictor, zeroshot_configs, ensemble_weights=best_ensemble_weights, test_only=True)
            print("Iteration: ", iteration)
            print("eval error: ", best_train_score)
            print("eval metric: ", 1 - best_train_score)
            print("test error: ", test_score)
            print("test metric: ", 1 - test_score)
            # 这两个
            # print("test_ensemble_weights: ", test_ensemble_weights) # 输出的weight顺序不一定和all_configs一致。
            print("selected: ")
            for c in zeroshot_configs:
                print(c)
            print("best_ensemble_weights: ", best_ensemble_weights)
            print()
            print()
            msg += f' | {round(fit_time, 2)}s | {best_next_config}'
            # print('here, make metadata')
            # metadata_out = dict(
            #     configs=copy.deepcopy(zeroshot_configs),
            #     new_config=best_next_config,
            #     step=iteration,
            #     train_score=best_train_score,
            #     test_score=test_score,
            #     num_configs=len(zeroshot_configs),
            #     fit_time=fit_time,
            # )
            # is_last = len(zeroshot_configs) >= num_zeroshot
            # if return_all_metadata or is_last:
            #     metadata_list.append(metadata_out)

            # print(msg)
        print("final selected: ")
        for c in zeroshot_configs:
            print(c)
        print("best_ensemble_weights: ", best_ensemble_weights)
        
    else:
        fit_args = {"train_data": train_data.data, "tuning_data": val_data.data, **params}

        utc_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        if eval_model_path != None:
            predictor = predictor.load(eval_model_path)
            training_duration = 0.
        else:
            if resume:
                predictor = predictor.load(os.path.join(benchmark_dir,"models/last.ckpt"), resume=True) # 如果不写resume = True，就会加载这个ckpt后从epoch 0开始训练。
            start_time = time.time()
            predictor.fit(**fit_args)
            end_time = time.time()
            training_duration = round(end_time - start_time, 1)

        if isinstance(test_data.data, dict):  # multiple test datasets
            test_data_dict = test_data.data

        else:
            test_data_dict = {dataset_name: test_data}

        for dataset_name, test_data in test_data_dict.items():
            evaluate_args = {
                "data": test_data.data,
                "label": label_column,
                "metrics": test_data.metric if metrics_func is None else metrics_func,
            }

            if test_data.problem_type == IMAGE_TEXT_SIMILARITY:
                evaluate_args["query_data"] = test_data.data[test_data.text_columns[0]].unique().tolist()
                evaluate_args["response_data"] = test_data.data[test_data.image_columns[0]].unique().tolist()
                evaluate_args["cutoffs"] = [1, 5, 10]

            start_time = time.time()
            scores = predictor.evaluate(**evaluate_args)
            end_time = time.time()
            predict_duration = round(end_time - start_time, 1)

            if "#" in framework:
                framework, version = framework.split("#")
            else:
                framework, version = framework, ag_version

            metric_name = test_data.metric if metrics_func is None else metrics_func.name
            metrics = {
                "id": "id/0",  # dummy id to make it align with amlb benchmark output
                "task": dataset_name,
                "framework": framework,
                "constraint": constraint,
                "version": version,
                "fold": 0,
                "type": predictor.problem_type,
                "metric": metric_name,
                "utc": utc_time,
                "training_duration": training_duration,
                "predict_duration": predict_duration,
                "scores": scores,
            }
            subdir = f"{framework}.{dataset_name}.{constraint}.local"
            save_metrics(os.path.join(metrics_dir, subdir, "scores"), metrics)


if __name__ == "__main__":
    args = get_args()
    print("args:")
    print(args)
    if args.params is not None:
        with open(args.params, 'r') as f:
            args.params = yaml.safe_load(f)
    
    args.custom_dataloader = args.params['custom_dataloader']
    args.framework = args.params['framework']
    args.params =  yaml.safe_load(open(os.path.join(args.params['custom_resource_dir'],"multimodal_frameworks.yaml"), 'r'))[args.framework]['params']
    
    ### model params:
    args.params['hyperparameters']["optimization.weight_decay"] = args.weight_decay
    args.params['hyperparameters']["optimization.lr_decay"] = args.lr_decay
    args.params['hyperparameters']["optimization.lr_schedule"] = args.lr_schedule
    args.params['hyperparameters']["optimization.warmup_steps"] = args.warmup_steps
    args.params['hyperparameters']["optimization.gradient_clip_val"] = args.gradient_clip_val
    args.params['hyperparameters']["optimization.top_k_average_method"] = args.top_k_average_method
    args.params['hyperparameters']["optimization.efficient_finetune"] = args.peft
    args.params['hyperparameters']["data.categorical.convert_to_text"] = args.categorical_convert_to_text
    args.params['hyperparameters']["data.categorical.convert_to_text_use_header"] = args.categorical_convert_to_text_use_header
    args.params['hyperparameters']["data.categorical.convert_to_text_use_header_template"] = args.categorical_convert_to_text_use_header_template
    # if args.categorical_convert_to_text_use_header_template == "latex":
    #     assert args.no_hf_text_insert_sep == False
    #     args.params['hyperparameters']["model.hf_text.insert_sep"] = False 
    if  args.no_hf_text_insert_sep == False:
        args.params['hyperparameters']["model.hf_text.insert_sep"] = False 
    args.params['hyperparameters']["data.numerical.convert_to_text"] = args.numerical_convert_to_text
    args.params['hyperparameters']["data.numerical.convert_to_text_use_header"] = args.numerical_convert_to_text_use_header
    args.params['hyperparameters']["optimization.max_epochs"] = args.max_epochs
    # args.params['hyperparameters']['model.hf_text.checkpoint_name'] = args.hf_text_ckpt
    args.params['hyperparameters']['optimization.lora.r'] = args.lora_r
    args.params['hyperparameters']['optimization.learning_rate'] = args.lr
    args.params['get_dataset_info'] = args.get_dataset_info
    args.params['use_ensemble'] = args.use_ensemble
    if args.use_ensemble:
        args.params['use_avg_ensemble'] = args.use_avg_ensemble
    args.params['seed'] = args.seed

    args.params['hyperparameters']['model.hf_text.max_text_len'] = args.max_text_len

    if args.ft_transformer_ckpt_name:
        args.params['hyperparameters']['model.ft_transformer.checkpoint_name'] = "https://automl-mm-bench.s3.amazonaws.com/ft_transformer_pretrained_ckpt/iter_2k.ckpt"
    args.params['hyperparameters']['model.hf_text.text_trivial_aug_maxscale'] = args.text_trivial_aug_maxscale
    
    if args.benchmark_dir == "debug":
        os.system(f"rm -rf  {args.benchmark_dir}")

    use_default_fusion = True # 使用默认的fusion方式

    if args.use_image_aug == False:
        args.params['hyperparameters']['model.timm_image.train_transforms'] = ['resize_shorter_side', 'center_crop']
    if args.use_fusion_transformer:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'fusion_transformer']
        if args.fusion_transformer_concat_all_tokens:
            args.params['hyperparameters']['model.hf_text.pooling_mode'] = "all"
            args.params['hyperparameters']['model.ft_transformer.pooling_mode'] = "all"
            args.params['hyperparameters']['model.timm_image.pooling_mode'] = "all"
        use_default_fusion = False
        if args.use_llama:
            args.params['hyperparameters']['model.fusion_transformer.use_llama'] = True
        if args.use_llama_7B:
            args.params['hyperparameters']['model.fusion_transformer.use_llama_7B'] = True
    else:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'fusion_mlp']
    
    if args.early_fusion:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'fusion_metatransformer']
        for model_name in args.params['hyperparameters']['model.names']:
            args.params['hyperparameters'][f'model.{model_name}.early_fusion'] = True
        use_default_fusion = False
    # print(args.params['hyperparameters']['model.names'])

    if args.sequential_fusion:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'sequential_fusion_mlp']
        if args.auxiliary_weight != 0.1: # 0.1是默认的，把所有有weight这个参数的model都设置一下。
            args.params['hyperparameters']["model.sequential_fusion_mlp.weight"] = args.auxiliary_weight
            # args.params['hyperparameters']["model.sequential_fusion_mlp.weight"] = args.auxiliary_weight
    
        for model_name in args.params['hyperparameters']['model.names']:
            args.params['hyperparameters'][f'model.{model_name}.sequential_fusion'] = True
        use_default_fusion = False

    if args.clip_fusion_mlp:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'clip_fusion_mlp', 'fusion_mlp']
        # use_default_fusion = False

    if args.clip_best_quality:
        args.params['hyperparameters']["model.clip_fusion_mlp.checkpoint_name"] = "openai/clip-vit-large-patch14-336"
        args.params['hyperparameters']["model.clip_fusion_mlp.image_size"] = 336
    if args.clip_high_quality:
        args.params['hyperparameters']["model.clip_fusion_mlp.checkpoint_name"] = "openai/clip-vit-large-patch14"

    if args.use_different_lr_for_each_modality:
        args.params['hyperparameters']["optimization.image_lr"] = args.image_lr
        args.params['hyperparameters']["optimization.text_lr"] = args.text_lr
        args.params['hyperparameters']["optimization.tabular_lr"] = args.tabular_lr

    if args.use_image_only:
        args.params['use_image_only'] = args.use_image_only
    if args.use_text_only:
        args.params['use_text_only'] = args.use_text_only
    if args.use_tabular_only:
        args.params['use_tabular_only'] = args.use_tabular_only

    if use_default_fusion:
        if args.auxiliary_weight != 0.1:
            args.params['hyperparameters']["model.fusion_mlp.weight"] = args.auxiliary_weight
            print("aug loss weight: ", args.params['hyperparameters']["model.fusion_mlp.weight"])
    
    if args.use_miss_token_embed:
        for model_name in args.params['hyperparameters']['model.names']:
            if args.use_miss_token_embed_image or args.use_miss_token_embed_text or args.use_miss_token_embed_numerical:
                if args.use_miss_token_embed_image:
                    if "image" in model_name:
                        args.params['hyperparameters'][f'model.{model_name}.use_miss_token_embed'] = True
                if args.use_miss_token_embed_text:
                    if "text" in model_name:
                        args.params['hyperparameters'][f'model.{model_name}.use_miss_token_embed'] = True
                if args.use_miss_token_embed_numerical:
                    if "ft_transformer" in model_name:
                        args.params['hyperparameters'][f'model.{model_name}.use_miss_token_embed'] = True
                    args.params['hyperparameters']["data.numerical.use_miss_embed"] = True
            else:
                args.params['hyperparameters'][f'model.{model_name}.use_miss_token_embed'] = True
                args.params['hyperparameters']["data.numerical.use_miss_embed"] = True

    if args.LeMDA:
      
        args.params['hyperparameters'][f'model.fusion_mlp.augmenter.turn_on'] = True
        args.params['hyperparameters'][f'optimization.aug_optimizer'] = True
        args.params['hyperparameters'][f'optimization.aug_turn_on'] = True
        args.params['hyperparameters'][f'optimization.aug_learning_rate'] = 1.0e-4
        args.params['hyperparameters'][f'optimization.aug_optim_type'] = "adam"
        args.params['hyperparameters'][f'optimization.aug_weight_decay'] = 1.0e-5
        args.params['hyperparameters'][f'model.fusion_mlp.augmenter.arch'] = args.LeMDA_arch
        args.params['hyperparameters'][f'model.fusion_mlp.augmenter.n_layer'] = args.LeMDA_layer

    if args.modality_drop_rate > 0.:
        args.params['hyperparameters'][f'data.modality_drop_ratio'] = args.modality_drop_rate

    # if args.self_distill:
    #     args.params['teacher_predictor'] = "self_distill"
    #     args.params['hyperparameters'][f'distiller.self_distill'] = True
    # else:
    #     args.params['teacher_predictor'] = None

    if args.alignment_loss != None:
        args.params['hyperparameters'][f'model.fusion_mlp.alignment_loss'] = args.alignment_loss

    if args.contrastive_loss != None:
        args.params['hyperparameters'][f'optimization.contrastive_loss'] =  args.contrastive_loss
        args.params['hyperparameters'][f'optimization.contrastive_loss_w'] =  args.contrastive_loss_w
    
    if args.no_use_cate_miss_embed:
        args.params['hyperparameters'][f'model.ft_transformer.no_use_cate_miss_embed'] = args.no_use_cate_miss_embed
    
    if args.manifold_mixup:
        args.params['hyperparameters'][f'optimization.manifold_mixup'] = True
        args.params['hyperparameters'][f'model.timm_image.manifold_mixup'] = True
        args.params['hyperparameters'][f'model.timm_image.manifold_mixup_a'] = args.manifold_mixup_a
        args.params['hyperparameters'][f'model.hf_text.manifold_mixup'] = True
        args.params['hyperparameters'][f'model.hf_text.manifold_mixup_a'] = args.manifold_mixup_a
        args.params['hyperparameters'][f'model.ft_transformer.manifold_mixup'] = True
        args.params['hyperparameters'][f'model.ft_transformer.manifold_mixup_a'] =  args.manifold_mixup_a
        args.params['hyperparameters']["env.per_gpu_batch_size"] = 2
        args.params['hyperparameters'][f'model.fusion_mlp.manifold_mixup'] = True
        
    
    if args.mixup:
        args.params['hyperparameters']["data.mixup.turn_on"] = True
    # args.params['hyperparameters']["data.mixup.turn_on"] = True
    print(type(args.params['hyperparameters']["optimization.gradient_clip_val"]))
    print(args.params)
    # ['framework']['params']

    


    if args.custom_metrics is not None:
        with open(args.custom_metrics, 'r') as f:
            args.custom_metrics = yaml.safe_load(f)
    print("args:")
    print(args)
    run(
        dataset_name=args.dataset_name,
        framework=args.framework,
        benchmark_dir=args.benchmark_dir,
        metrics_dir=args.metrics_dir,
        constraint=args.constraint,
        params=args.params,
        custom_dataloader=args.custom_dataloader,
        custom_metrics=args.custom_metrics,
        eval_model_path=args.eval_model_path,
        resume=args.resume
    )

   
'''
args:
Namespace(dataset_name='imdb', framework='AutoGluon_stable', 
benchmark_dir='ag_bench_runs/multimodal/ag_bench_20231223T123713', 
metrics_dir='ag_bench_runs/multimodal/ag_bench_20231223T123713/results', 
constraint=None, params={'presets': 'best_quality', 
'hyperparameters': {'optimization.max_epochs': 10}}, 
custom_dataloader={'dataloader_file': 'sample_configs/dataloaders/text_tabular_dataloader.py', 'class_name': 'TextTabularDataLoader', 'dataset_config_file': 'sample_configs/dataloaders/text_tabular_datasets.yaml'}, custom_metrics=None)
'''