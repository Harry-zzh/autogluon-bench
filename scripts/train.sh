### text_tabular_dataloader
python3 src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs_text_tabular.yaml \
--dataset_name imdb --custom_dataloader sample_configs/dataloaders/text_tabular_dataloader.py \
--benchmark_dir ag_bench_runs/multimodal/imdb \
--metrics_dir  ag_bench_runs/multimodal/imdb/results