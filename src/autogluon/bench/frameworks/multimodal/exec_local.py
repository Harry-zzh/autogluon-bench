import argparse
import csv
import importlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Union

from autogluon.bench.datasets.dataset_registry import multimodal_dataset_registry
from autogluon_local.core.src.autogluon.core.metrics import make_scorer
from autogluon_local.multimodal.src.autogluon.multimodal import MultiModalPredictor
from autogluon_local.multimodal.src.autogluon.multimodal import __version__ as ag_version
from autogluon_local.multimodal.src.autogluon.multimodal.constants import IMAGE_SIMILARITY, IMAGE_TEXT_SIMILARITY, OBJECT_DETECTION, TEXT_SIMILARITY
import yaml
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
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="num of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument(
        "--auxiliary_weight", type=float, default=0., help="auxiliary loss weight for unimodal models."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )

    ### Basic Tricks
    parser.add_argument(
        "--top_k_average_method", type=str, default="greedy_soup"
    )
    parser.add_argument(
        "--gradient_clip_val", default=1.
    )
    parser.add_argument(
        "--warmup_steps", type=float, default=0.1
    )
    parser.add_argument(
        "--categorical_convert_to_text_use_header_template",type=str, default="list"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="num of training epochs."
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="It is used only when lr_choice is layerwise_decay"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False
    )

    ### Multimodal Fusion Strategies
    parser.add_argument(
        "--use_fusion_transformer", action="store_true", default=False
    )
    parser.add_argument(
        "--fusion_transformer_concat_all_tokens", action="store_true", default=False
    )
    parser.add_argument(
        "--clip_fusion_mlp", action="store_true", default=False, help="Use clip for late fusion model."
    )
    parser.add_argument(
        "--clip_high_quality", action="store_true", default=False, help="Use clip high quality"
    )
    parser.add_argument(
        "--early_fusion", action="store_true", default=False
    )
    parser.add_argument(
        "--sequential_fusion", action="store_true", default=False
    )
    
    ### Converting Tabular Data into Text
    parser.add_argument(
        "--categorical_convert_to_text", type=bool, default=True, help="convert categorical columns to text or not."
    )
    parser.add_argument(
        "--categorical_convert_to_text_use_header", action='store_true', default=False, help="integrate header information or not."
    )
    parser.add_argument(
        "--numerical_convert_to_text", action='store_true', default=False, help="convert numerical columns to text or not."
    )
    parser.add_argument(
        "--no_hf_text_insert_sep",action='store_false', default=True,
    )

    ### Data Aug
    parser.add_argument(
        "--text_trivial_aug_maxscale", type=float, default=0.0, help="Text input aug."
    )
    parser.add_argument(
        "--not_use_image_aug", type=bool, default=True,
    )
    parser.add_argument(
        "--LeMDA", action='store_true', default=False, help="Feature Aug(Joint)."
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

    metrics_func = None
    if custom_metrics is not None and custom_metrics["function_name"] == train_data.metric:
        metrics_func = load_custom_metrics(custom_metrics=custom_metrics)

    predictor = MultiModalPredictor(**predictor_args)

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
    
    args.params['hyperparameters']["optimization.max_epochs"] = args.max_epochs
    args.params['hyperparameters']['optimization.learning_rate'] = args.lr
    args.params['seed'] = args.seed

    ### Basic Tricks
    args.params['hyperparameters']["optimization.lr_decay"] = args.lr_decay
    args.params['hyperparameters']["optimization.warmup_steps"] = args.warmup_steps
    args.params['hyperparameters']["optimization.gradient_clip_val"] = args.gradient_clip_val
    args.params['hyperparameters']["optimization.top_k_average_method"] = args.top_k_average_method

    ### Multimodal Fusion Strategies
    use_default_fusion = True # use MLP as fusion module
    if args.use_fusion_transformer:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'fusion_transformer']
        if args.fusion_transformer_concat_all_tokens:
            args.params['hyperparameters']['model.hf_text.pooling_mode'] = "all"
            args.params['hyperparameters']['model.ft_transformer.pooling_mode'] = "all"
        use_default_fusion = False
    else:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'fusion_mlp']
    
    if args.early_fusion:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'fusion_metatransformer']
        for model_name in args.params['hyperparameters']['model.names']:
            args.params['hyperparameters'][f'model.{model_name}.early_fusion'] = True
        use_default_fusion = False

    if args.sequential_fusion:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'sequential_fusion_mlp']
        if args.auxiliary_weight != 0.1:
            args.params['hyperparameters']["model.sequential_fusion_mlp.weight"] = args.auxiliary_weight

        for model_name in args.params['hyperparameters']['model.names']:
            args.params['hyperparameters'][f'model.{model_name}.sequential_fusion'] = True
        use_default_fusion = False

    if args.clip_fusion_mlp:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'clip_fusion_mlp', 'fusion_mlp']
        if args.clip_high_quality:
            args.params['hyperparameters']["model.clip_fusion_mlp.checkpoint_name"] = "openai/clip-vit-large-patch14"
        use_default_fusion = False
    if use_default_fusion:
        if args.auxiliary_weight != 0.1:
            args.params['hyperparameters']["model.fusion_mlp.weight"] = args.auxiliary_weight

    ### Converting Tabular Data into Text
    args.params['hyperparameters']["data.categorical.convert_to_text"] = args.categorical_convert_to_text
    args.params['hyperparameters']["data.categorical.convert_to_text_use_header"] = args.categorical_convert_to_text_use_header
    args.params['hyperparameters']["data.categorical.convert_to_text_use_header_template"] = args.categorical_convert_to_text_use_header_template
    if args.no_hf_text_insert_sep == False:
        args.params['hyperparameters']["model.hf_text.insert_sep"] = False
    args.params['hyperparameters']["data.numerical.convert_to_text"] = args.numerical_convert_to_text

    ### Data Aug
    args.params['hyperparameters']['model.hf_text.text_trivial_aug_maxscale'] = args.text_trivial_aug_maxscale
    if args.not_use_image_aug == False:
        args.params['hyperparameters']['model.timm_image.train_transforms'] = ['resize_shorter_side', 'center_crop']
    if args.LeMDA:
        args.params['hyperparameters'][f'model.fusion_mlp.augmenter.turn_on'] = True
        args.params['hyperparameters'][f'optimization.aug_optimizer'] = True
        args.params['hyperparameters'][f'optimization.aug_turn_on'] = True
        args.params['hyperparameters'][f'optimization.aug_learning_rate'] = 1.0e-4
        args.params['hyperparameters'][f'optimization.aug_optim_type'] = "adam"
        args.params['hyperparameters'][f'optimization.aug_weight_decay'] = 1.0e-5
    
    if args.benchmark_dir == "debug":
        os.system(f"rm -rf  {args.benchmark_dir}")

    if args.custom_metrics is not None:
        with open(args.custom_metrics, 'r') as f:
            args.custom_metrics = yaml.safe_load(f)

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