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
from autogluon_local.multimodal.src.autogluon.multimodal.models.utils import get_pretrained_tokenizer
from autogluon_local.core.src.autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
import numpy as np
from PIL import Image
from autogluon_local.core.src.autogluon.core.metrics import get_metric
from autogluon_local.multimodal.src.autogluon.multimodal.utils.misc import logits_to_prob
import copy
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

def drop_modalities(df, a, column_types, seed=0):
    np.random.seed(seed)
    assert 0 <= a <= 1

    image_columns = [col for col in df.columns if col in column_types and 'image' in column_types[col]]
    text_columns = [col for col in df.columns if col in column_types and 'text' in column_types[col]]
    categorical_columns = [col for col in df.columns if col in column_types and 'categorical' in column_types[col]]
    numerical_columns = [col for col in df.columns if col in column_types and 'numerical' in column_types[col]]
    
    # Tabular columns
    tabular_columns = categorical_columns + numerical_columns
    
    def get_missing_indices(columns, proportion):
        total_elements = len(df) * len(columns)
        n_missing = int(total_elements * proportion)
        missing_indices = np.random.choice(total_elements, n_missing, replace=False, )
        row_indices = missing_indices // len(columns)
        col_indices = missing_indices % len(columns)
        return list(zip(row_indices, col_indices))

    for row_idx, col_idx in get_missing_indices(image_columns, a):
        df.loc[row_idx, image_columns[col_idx]] = np.nan
    
    for row_idx, col_idx in get_missing_indices(text_columns, a):
        df.loc[row_idx, text_columns[col_idx]] = np.nan
        
    for row_idx, col_idx in get_missing_indices(tabular_columns, a):
        df.loc[row_idx, tabular_columns[col_idx]] = np.nan
    
    for index, row in df.iterrows():
        if row[image_columns].isnull().all() and row[text_columns].isnull().all() and row[tabular_columns].isnull().all():
            if len(image_columns) == 0:
                modality_to_retain = np.random.choice(['text', 'tabular'])
            elif len(text_columns) == 0:
                modality_to_retain = np.random.choice(['image', 'tabular'])
            elif len(tabular_columns) == 0:
                modality_to_retain = np.random.choice(['image', 'text'])
            else:
                modality_to_retain = np.random.choice(['image', 'text', 'tabular'])
            if modality_to_retain == 'image':
                df.loc[index, image_columns] = df.loc[index, image_columns].fillna('retained_image_value')
            elif modality_to_retain == 'text':
                df.loc[index, text_columns] = df.loc[index, text_columns].fillna('retained_text_value')
            elif modality_to_retain == 'tabular':
                df.loc[index, tabular_columns] = df.loc[index, tabular_columns].fillna('retained_tabular_value')
    
    return df

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
        "--max_epochs", type=int, default=20, help="num of training epochs."
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
    parser.add_argument(
        "--resume", action="store_true", default=False
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
        "--lr_decay", type=float, default=0.9, help="It is used only when lr_choice is layerwise_decay"
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
        "--use_llama_7B", action='store_true', default=False, help="Use LLM(LLAMA-7B) as the fusion module."
    )
    parser.add_argument(
        "--llama_7B_token", type=str, default=None, help="The token for downloading Llama2-7B model."
    )
    parser.add_argument(
        "--early_fusion", action="store_true", default=False
    )
    parser.add_argument(
        "--meta_transformer_ckpt_path", type=str, default=None, help="The path of the pre-trained checkpoint of Meta-Transformer-L14."
    )
    parser.add_argument(
        "--sequential_fusion", action="store_true", default=False
    )
    
    ### Converting Tabular Data into Text
    parser.add_argument(
        "--categorical_convert_to_text", action='store_true', default=False, help="convert categorical columns to text or not."
    )
    parser.add_argument(
        "--categorical_convert_to_text_template",type=str, default="latex"
    )
    parser.add_argument(
        "--numerical_convert_to_text", action='store_true', default=False, help="convert numerical columns to text or not."
    )
    parser.add_argument(
        "--no_hf_text_insert_sep",action='store_false', default=True,
    )

    ### Cross-modal alignment
    parser.add_argument(
        "--alignment_loss", type=str, default=None, help="Extra loss for cross-modality alignment.", choices=["positive-only", "positive_negative", "all"]
    )
    parser.add_argument(
        "--alignment_loss_w", type=float, default=1., help="The weight of positive+negative alignment loss."
    )

    ### Data Aug
    parser.add_argument(
        "--text_trivial_aug_maxscale", type=float, default=0.0, help="Text input aug."
    )
    parser.add_argument(
        "--use_image_aug", action='store_true', default=False
    )
    parser.add_argument(
        "--LeMDA", action='store_true', default=False, help="Use Feature Aug(Joint) or not."
    )
    parser.add_argument(
        "--LeMDA_layer", type=int, default=6,
    )
    parser.add_argument(
        "--manifold_mixup",  action='store_true', default=False, help="Use Feature Aug(Inde.) or not."
    )
    parser.add_argument(
        "--manifold_mixup_a", type=float, default=2., help= "Alpha params in manifold mixup" 
    )
    

    ### Handling Missingness
    parser.add_argument(
        "--use_miss_token_embed",action='store_true', default=False,
    )
    parser.add_argument(
        "--use_miss_token_embed_image",action='store_true', default=False, help="Use Learnable Embed(Image) or not."
    )

    parser.add_argument(
        "--use_miss_token_embed_numerical",action='store_true', default=False, help="Use LearnableEmbed(Numeric) or not."
    )
    parser.add_argument(
        "--modality_drop_rate", type=float, default=0.
    )
    parser.add_argument(
        "--simulate_missingness",  action='store_true', default=False, help="Simulating modality missingness or not."
    )
    parser.add_argument(
        "--simulate_missingness_drop_rate",  type=float, default=0.
    )


    ### Integrating bag-of-tricks
    parser.add_argument(
        "--use_ensemble", action='store_true', default=False, help="Ensemble Selection."
    )
    parser.add_argument(
        "--avg_all", action='store_true', default=False, help="Average All."
    )
    parser.add_argument(
        "--model_paths", nargs='+', help="List of model ckpt paths for ensembling."
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

    

    utc_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    if eval_model_path != None:
        predictor = predictor.load(eval_model_path)
        training_duration = 0.

    use_ensemble = params.pop("use_ensemble")
    simulate_missingness = params.pop("simulate_missingness")
    if use_ensemble:
        avg_all = params.pop("avg_all")
        ensemble_model_paths = params.pop("model_paths")
    if simulate_missingness:
        simulate_missingness_drop_rate = params.pop("simulate_missingness_drop_rate")

    fit_args = {"train_data": train_data.data, "tuning_data": val_data.data, **params}
    if use_ensemble:
        zeroshot_configs = []
        all_configs = ensemble_model_paths
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
            pred_val = np.stack(pred_val)
            pred_test = np.stack(pred_test)
            val_metric = get_metric(metric=val_data.metric, problem_type=problem_type )

            ## weight ensemble
            weighted_ensemble = EnsembleSelection(
                ensemble_size=10,
                problem_type=problem_type,
                metric=val_metric,
                # **self.ensemble_method_kwargs,
            )

            if test_only:
                weighted_ensemble.weights_ = ensemble_weights
                y_test_pred, y_pred_proba = weighted_ensemble.predict(pred_test)
                if problem_type in ["binary"]:
                    y_test_pred = list(logits_to_prob(y_test_pred)[:,1])
                err = val_metric.error(y_test, y_test_pred)
                return err, ensemble_weights
            
            weighted_ensemble.fit(predictions=pred_val, labels=y_val)

            y_val_pred, y_pred_proba = weighted_ensemble.predict(pred_val)

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
                config_score, ensemble_weights = score(predictor, config_selected) 
                if config_score < best_score:
                    best_score = config_score
                    best_next_config = config
                    best_ensemble_weights = ensemble_weights
            return best_next_config, best_score, best_ensemble_weights

        iteration = 0
        if use_ensemble and avg_all:
            zeroshot_configs = all_configs
            best_ensemble_weights = [1. / len(all_configs)] * len(all_configs)
            test_score, test_ensemble_weights = score(predictor, zeroshot_configs, ensemble_weights=best_ensemble_weights, test_only=True)
            print("test error: ", test_score)
            print("test metric: ", 1 - test_score)
            print("best_ensemble_weights: ", best_ensemble_weights)
            return
        
        final_selected_configs = []
        best_eval_error = 9999999
        final_ensemble_weights = []
        while len(zeroshot_configs) < num_zeroshot: 
            # greedily search the config that would yield the lowest average rank if we were to evaluate it in combination
            # with previously chosen configs.

            valid_configs = [c for c in all_configs if c not in zeroshot_configs]
            if not valid_configs:
                break
            if iteration == 0:
                prior_best_score = None
            iteration += 1

            time_start = time.time()

            best_next_config, best_eval_score, best_ensemble_weights = _select_sequential(valid_configs, zeroshot_configs, prior_best_score=prior_best_score)
            time_end = time.time()
            prior_best_score = best_eval_score

            zeroshot_configs.append(best_next_config)
            fit_time = time_end - time_start
            msg = f'{iteration}\t: Eval: {round(best_eval_score, 2)}'

            test_score, test_ensemble_weights = score(predictor, zeroshot_configs, ensemble_weights=best_ensemble_weights, test_only=True)
            print("Iteration: ", iteration)
            print("eval error: ", best_eval_score)
            print("eval metric: ", 1 - best_eval_score)
            print("test error: ", test_score)
            print("test metric: ", 1 - test_score)

            print("selected: ")
            print(zeroshot_configs)
        
            print("ensemble_weights: ", best_ensemble_weights)
            print()
            print()
            msg += f' | {round(fit_time, 2)}s | {best_next_config}'

            if best_eval_score < best_eval_error:
                best_eval_error = best_eval_score
                final_selected_configs = zeroshot_configs
                final_ensemble_weights = best_ensemble_weights

        print("final selected: ")
        print(final_selected_configs)
    
        print("final ensemble_weights: ", final_ensemble_weights)
        print()

        
    elif simulate_missingness: 
        predictor._learner.prepare_train_tuning_data(train_data=train_data.data, tuning_data=val_data.data, seed=params["seed"], holdout_frac=None)
        column_types = []
        predictor._learner.infer_column_types(column_types=column_types)
        column_types = predictor._learner._column_types

        del column_types[label_column] # do not drop the label
        train_data.data = drop_modalities(train_data.data, a=simulate_missingness_drop_rate, column_types=column_types)
        if val_data.data != None:
            val_data.data = drop_modalities(val_data.data, a=simulate_missingness_drop_rate, column_types=column_types)

        #  
        train_missing_ratios = {col: train_data.data[col].isna().sum() / len(train_data.data) for col in train_data.data.columns}
        print("train col types: ", column_types)
        print("train_missing_ratios: ", train_missing_ratios)

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

        ori_test_data = copy.deepcopy(test_data)
        for drop_rate in [0.1,0.3,0.5]:
            test_data_dict = {dataset_name: ori_test_data}
            for dataset_name, test_data in test_data_dict.items():
                test_data_data = drop_modalities(test_data.data, a=drop_rate,column_types=column_types)
                evaluate_args = {
                    "data": test_data_data,
                    "label": label_column,
                    "metrics": test_data.metric if metrics_func is None else metrics_func,
                }

                if test_data.problem_type == IMAGE_TEXT_SIMILARITY:
                    evaluate_args["query_data"] = test_data_data[test_data.text_columns[0]].unique().tolist()
                    evaluate_args["response_data"] = test_data_data[test_data.image_columns[0]].unique().tolist()
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
                # testing under different missing ratios
                save_metrics(os.path.join(metrics_dir, subdir, f"scores_{drop_rate}"), metrics)

      
    else:
        if resume:
            predictor = predictor.load(os.path.join(benchmark_dir,"models/last.ckpt"), resume=True) 
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
    args.params['use_ensemble'] = args.use_ensemble
    if args.use_ensemble:
        args.params['avg_all'] = args.avg_all
        args.params['model_paths'] = args.model_paths 
    args.params['simulate_missingness'] = args.simulate_missingness
    if args.simulate_missingness:
        args.params['simulate_missingness_drop_rate'] = args.simulate_missingness_drop_rate
    args.params['seed'] = args.seed

    ########## Parameters
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
            args.params['hyperparameters']['model.timm_image.pooling_mode'] = "all"
        use_default_fusion = False
        if args.use_llama_7B:
            args.params['hyperparameters']['model.fusion_transformer.use_llama_7B'] = True
            args.params['hyperparameters']['model.fusion_transformer.llama_7B_token'] = args.llama_7B_token
    else:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'fusion_mlp']
    
    if args.early_fusion:
        args.params['hyperparameters']['model.names'] = ['ft_transformer', 'timm_image', 'hf_text', 'document_transformer', 'fusion_metatransformer']
        for model_name in args.params['hyperparameters']['model.names']:
            args.params['hyperparameters'][f'model.{model_name}.early_fusion'] = True
            args.params['hyperparameters'][f'model.{model_name}.meta_transformer_ckpt_path'] = args.meta_transformer_ckpt_path
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
    args.params['hyperparameters']["data.categorical.convert_to_text_template"] = args.categorical_convert_to_text_template
    if args.no_hf_text_insert_sep == False:
        args.params['hyperparameters']["model.hf_text.insert_sep"] = False
    args.params['hyperparameters']["data.numerical.convert_to_text"] = args.numerical_convert_to_text

    ### Cross-modal alignment
    if args.alignment_loss != None:
        if args.use_fusion_transformer:
            args.params['hyperparameters'][f'model.fusion_transformer.alignment_loss'] = args.alignment_loss
        else:
            args.params['hyperparameters'][f'model.fusion_mlp.alignment_loss'] = args.alignment_loss
        if args.alignment_loss == "positive_negative" or args.alignment_loss == "all":
            args.params['hyperparameters'][f'optimization.contrastive_loss'] =  args.alignment_loss
            args.params['hyperparameters'][f'optimization.contrastive_loss_w'] =  args.alignment_loss_w
    
    ### Data Aug
    args.params['hyperparameters']['model.hf_text.text_trivial_aug_maxscale'] = args.text_trivial_aug_maxscale
    if args.use_image_aug == False:
        args.params['hyperparameters']['model.timm_image.train_transforms'] = ['resize_shorter_side', 'center_crop']
    if args.LeMDA:
        if args.use_fusion_transformer:
            args.params['hyperparameters'][f'model.fusion_transformer.augmenter.turn_on'] = True
            args.params['hyperparameters'][f'model.fusion_transformer.augmenter.n_layer'] = args.LeMDA_layer

        else:
            args.params['hyperparameters'][f'model.fusion_mlp.augmenter.turn_on'] = True
            args.params['hyperparameters'][f'model.fusion_mlp.augmenter.n_layer'] = args.LeMDA_layer

        args.params['hyperparameters'][f'optimization.aug_optimizer'] = True
        args.params['hyperparameters'][f'optimization.aug_turn_on'] = True
        args.params['hyperparameters'][f'optimization.aug_learning_rate'] = 1.0e-4
        args.params['hyperparameters'][f'optimization.aug_optim_type'] = "adam"
        args.params['hyperparameters'][f'optimization.aug_weight_decay'] = 1.0e-5
    
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
        if args.clip_fusion_mlp:
            args.params['hyperparameters'][f'model.clip_fusion_mlp.manifold_mixup'] = True
            
    ### Handling Missingness
    if args.use_miss_token_embed:
        for model_name in args.params['hyperparameters']['model.names']:
            if args.use_miss_token_embed_image or args.use_miss_token_embed_numerical:
                if args.use_miss_token_embed_image:
                    if "image" in model_name:
                        args.params['hyperparameters'][f'model.{model_name}.use_miss_token_embed'] = True
                if args.use_miss_token_embed_numerical:
                    if "ft_transformer" in model_name:
                        args.params['hyperparameters'][f'model.{model_name}.use_miss_token_embed'] = True
                    args.params['hyperparameters']["data.numerical.use_miss_embed"] = True
            else:
                args.params['hyperparameters'][f'model.{model_name}.use_miss_token_embed'] = True
                args.params['hyperparameters']["data.numerical.use_miss_embed"] = True
    if args.modality_drop_rate > 0.:
        args.params['hyperparameters'][f'data.modality_drop_ratio'] = args.modality_drop_rate
    ##########

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