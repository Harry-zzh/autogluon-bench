from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd
import typer
from typing_extensions import Annotated

from autogluon.bench.eval.evaluation.constants import (
    DATASET,
    FOLD,
    FRAMEWORK,
    METRIC,
    METRIC_ERROR,
    PROBLEM_TYPE,
    TIME_INFER_S,
    TIME_TRAIN_S,
)
from autogluon.bench.eval.evaluation.preprocess import preprocess_openml
from autogluon.common.savers import save_pd

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def clean_amlb_results(
    benchmark_name: str = typer.Argument(
        None, help="Benchmark name populated by benchmark run, in format <benchmark_name>_<timestamp>"
    ),
    results_dir: str = typer.Option("data/results/", help="Root directory of raw and prepared results."),
    results_dir_input: str = typer.Option(
        None,
        help="Directory of the results file '<file_prefix><constraint_str><benchmark_name_str>.csv' getting cleaned. Can be an S3 URI. If not provided, it defaults to '<results_dir>input/raw/'",
    ),
    results_dir_output: str = typer.Option(
        None,
        help="Output directory of cleaned file. Can be an S3 URI. If not provided, it defaults to '<results_dir>input/prepared/openml/'",
    ),
    file_prefix: str = typer.Option("results_automlbenchmark", help="File prefix of the input results files."),
    benchmark_name_in_input_path: bool = False,
    benchmark_name_in_output_path: bool = True,
    constraints: Optional[List[str]] = typer.Option(
        None,
        help="List of AMLB constraints, refer to https://github.com/openml/automlbenchmark/blob/master/resources/constraints.yaml",
    ),
    out_path_prefix: str = typer.Option("openml_ag_", help="Prefix of result file."),
    out_path_suffix: str = typer.Option("", help="suffix of result file."),
    framework_suffix_column: str = typer.Option("constraint", help="Framework suffix column."),
):
    """
    Cleans and aggregate results further with unified column names and adds benchmark name into framework column.

    Example:
        agbench clean-and-save-results ag_tabular_20230629T140546 --results-dir-input s3://autogluon-benchmark-metrics/aggregated/tabular/ag_tabular_20230629T140546/ --benchmark-name-in-input-path --constraints constratint_1 --constraints constratint_2
    """
    clean_and_save_results(
        run_name=benchmark_name,
        results_dir=results_dir,
        results_dir_input=results_dir_input,
        results_dir_output=results_dir_output,
        file_prefix=file_prefix,
        run_name_in_input_path=benchmark_name_in_input_path,
        run_name_in_output_path=benchmark_name_in_output_path,
        save=True,
        constraints=constraints if constraints else None,
        out_path_prefix=out_path_prefix,
        out_path_suffix=out_path_suffix,
        framework_suffix_column=framework_suffix_column,
    )


def clean_and_save_results(
    run_name: str,
    results_dir: str = "data/results/",
    results_dir_input: str | None = None,
    results_dir_output: str | None = None,
    file_prefix: str | List[str] = "results_automlbenchmark",
    run_name_in_input_path: bool = True,
    run_name_in_output_path: bool = True,
    save: bool = True,
    save_minimal: bool = True,
    constraints: List[str] | None = None,
    out_path_prefix: str = "openml_ag_",
    out_path_suffix: str = "",
    framework_suffix_column: str = "constraint",
) -> pd.DataFrame:
    if results_dir_input is None:
        results_dir_input = os.path.join(results_dir, "input/raw/")
    if results_dir_output is None:
        results_dir_output = os.path.join(results_dir, "input/prepared/openml/")
    run_name_str = f"_{run_name}" if run_name_in_input_path else ""

    if not isinstance(file_prefix, list):
        file_prefix = [file_prefix]

    results_list = []
    if constraints is None:
        constraints = [None]
    for constraint in constraints:
        constraint_str = f"_{constraint}" if constraint is not None else ""
        for prefix in file_prefix:
            results = preprocess_openml.preprocess_openml_input(
                path=os.path.join(results_dir_input, f"{prefix}{constraint_str}{run_name_str}.csv"),
                framework_suffix=constraint_str,
                framework_suffix_column=framework_suffix_column,
            )
            results_list.append(results)

    results_raw = pd.concat(results_list, ignore_index=True, sort=True)

    if "framework_parent" in results_raw.columns:
        results_raw[FRAMEWORK] = results_raw["framework_parent"] + "_" + run_name + "_" + results_raw[FRAMEWORK]
    else:
        results_raw[FRAMEWORK] = results_raw[FRAMEWORK] + "_" + run_name

    minimal_columns = [
        DATASET,
        FOLD,
        FRAMEWORK,
        "constraint",
        METRIC,
        METRIC_ERROR,
        TIME_TRAIN_S,
        TIME_INFER_S,
        PROBLEM_TYPE,
        "tid",
    ]

    results_raw_columns = list(results_raw.columns)
    results_raw_columns = [c for c in results_raw_columns if c in minimal_columns] + [
        c for c in results_raw_columns if c not in minimal_columns
    ]
    results_raw = results_raw[results_raw_columns]

    if save:
        if run_name_in_output_path:
            save_path = os.path.join(results_dir_output, f"{out_path_prefix}{run_name}{out_path_suffix}")
        else:
            save_path = os.path.join(results_dir_output, f"{out_path_prefix}{out_path_suffix}")
        save_path_file = f"{save_path}.csv"

        save_pd.save(path=save_path_file, df=results_raw)
        logger.log(30, f"Cleaned results are saved in file: {save_path_file}")
        save_path_file_pq = f"{save_path}.parquet"
        save_pd.save(path=save_path_file_pq, df=results_raw)
        if save_minimal:
            results_raw_minimal = results_raw[minimal_columns]

            save_path_file_minimum = f"{save_path}_min.csv"
            save_pd.save(path=save_path_file_minimum, df=results_raw_minimal)
            logger.log(30, f"Cleaned results (minimum) are saved in file: {save_path_file_minimum}")
            save_path_file_minimum_pq = f"{save_path}_min.parquet"
            save_pd.save(path=save_path_file_minimum_pq, df=results_raw_minimal)
    return results_raw


if __name__ == "__main__":
    app()
