This folder contains the instructions for reproducing the result from the paper **Bag of Tricks for Multimodal AutoML with Image, Text, and Tabular Data.**

# 1. Installation

```shell
conda create -n ag python=3.10
conda activate ag
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install fastapi==0.110
pip install peft
pip install bitsandbytes
pip install mkl==2024.0

# install autogluon-bench
python3 -m pip install autogluon.bench

# install autogluon
cd src/autogluon/bench/frameworks/multimodal/autogluon_local
bash ./full_install.sh
```

# 2. Datasets Preparation
The urls of our processed datasets are in sample_configs/dataloaders/all_datasets.yaml. Our code already provides the code to prepare the datasets when running the benchmark, so you don't need to download the datasets manually.

Here are the names of datasets:
- Text+Tabular datasets: ["fake", "qaa", "qaq", "airbnb","channel", "cloth"]
- Image+Text datasets: ["ptech", "memotion", "food101", "aep","fakeddit"]
- Image+Tabular datasets: ["ccd", "HAM", "wikiart", "cd18","DVM"]
- Image+Text+Tabular datasets: ["petfinder", "covid", "artm", "seattle", "goodreads", "KARD"]

# 3. Run the Benchmark
```shell
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py --<flag> <value>
```
- `params` is a yaml file that records the basic settings for running autogluon-bench, we use autogluon-bench repository to run our benchmark.
- `seed` determines the random seed. Options are `0,1,2`.
- `benchmark_dir` determines the path of output directory.
- `metrics_dir` determines the path of evaluation results.
- `dataset_name` refers to the dataset name. Options are listed in **2. Datasets Preparation**.

The detailed commands of running the experiments in our paper are as follows.

## Basic Tricks

Flags explanations:
- `top_k_average_method` determines the way of averaging the weights of multiple fine-tuned models. Options are `best, greedy_soup`. `best` means choosing the best checkpoint weight, while `greedy_soup` means using greedy soup.
- `gradient_clip_val` Options are `None, 1.`.`None` means not using gradient clipping, while `1.` means cliping the gradients with a norm threshold of 1.0.
- `warmup_steps` Options are `0., 0.1`. `0` means not using learning rate warmup, while `0.1` means linearly increases the learning rate from 0 to the peak learning rate during the first 10% of training steps.
- `lr_decay` Options are `1., 0.9`. `1.` means not using layerwise learning rate decay, while `0.9` means using a decay rate of 0.9.

```shell
# Baseline
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method best \
--gradient_clip_val None \
--warmup_steps 0. \
--lr_decay 1.

# +Greedy Soup
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val None \
--warmup_steps 0. \
--lr_decay 1.

# +Gradient Clip
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0. \
--lr_decay 1.

# +LR Warmup
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 1.

# +Layerwise Decay (Baseline+)
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9
```

## Multimodal Fusion Strategies

Flags explanations:
- `use_fusion_transformer` determines whether using a transformer-based fusion module (LF-Transformer). 
- `clip_fusion_mlp` determines whether using CLIP’s image and text encoders. `clip_high_quality` means using CLIP ViT-L/14 variant (LF-Aligned).
- `use_llama_7B` determines whether using the 7B version of Llama2 as fusion module (LF-LLM).
- `sequential_fusion` determines whether using sequential fusion (LF-SF).
- `early_fusion` determines whether using early fusion.

```shell
# LF-Transformer
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--use_fusion_transformer 

# LF-Aigned
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--clip_fusion_mlp \
--clip_high_quality

# LF-LLM
## Follow the instructions in https://huggingface.co/meta-llama/Llama-2-7b-hf to get access to this model, then pass your own token for downloading the model to "--llama7B_token".
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--use_fusion_transformer \
--fusion_transformer_concat_all_tokens \
--use_llama_7B \
--llama7B_token {token}

# LF-SF
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs_text_tabular.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--sequential_fusion

# Early-fusion
## Follow the instructions in https://github.com/invictus717/MetaTransformer to download the pre-trained checkpoint of Meta-Transformer-L14, then pass the model path to "--meta_transformer_ckpt_path"
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--early_fusion \
--meta_transformer_ckpt_path {meta_transformer_ckpt_path}
```

## Converting Tabular Data into Text

- `categorical_convert_to_text` determines whether converting categorical data into text. `categorical_convert_to_text_template` determines the used templates when converting, options are `direct, text, list, latex`. Default is `latex`.
- `numerical_convert_to_text` determines whether converting numerical data into text. 

```shell
# Convert Categorical
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--categorical_convert_to_text \
--categorical_convert_to_text_template latex

# Convert Numerical
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--numerical_convert_to_text 
```

## Cross-modal Alignment

- `alignment_loss` determines the type of the extra loss used for cross-modal alignment. Options are `positive-only, positive_negative, all`. 

```shell
# Positive only
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--alignment_loss positive-only

# Positive+Negative
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--alignment_loss positive_negative
```

## Multimodal Data Augmentation

- `text_trivial_aug_maxscale` determines the scale of text input augmentation. Default is 0.1.
- `use_image_aug` determines whether using image input augmentation.
- `manifold_mixup` determines whether using manifold method (Feature Aug.(Inde.)).
- `LeMDA` determines whether using LeMDA method (Feature Aug.(Joint)).

```shell
# Input Aug.
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--text_trivial_aug_maxscale 0.1 \
--use_image_aug

# Feature Aug.(Inde.), use manifold mixup method
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--manifold_mixup

# Feature Aug.(Joint), use LeMDA method
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--LeMDA 
```

## Handling Modality Missingness

- `modality_drop_rate` determines the modality dropout rate. Default is 0.3 (Modality Dropout).
- `use_miss_token_embed_numerical` determines whether using learnable embedding for missing numerical data (LearnableEmbed(Numeric)).
- `use_miss_token_embed_image` determines whether using learnable embedding for missing image data (LearnableEmbed(Image)).
- `simulate_missingness` determines whether simulating various scenarios with different ratios of missing modalities in the training and test sets. `simulate_missingness_drop_rate` is the missing ratio of training set, options are `0.1, 0.3, 0.5`. These two parameters are used only when using the previous tricks of handling modality missingness.

```shell
# Modality Dropout
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--modality_drop_rate 0.3

# Learnable Embed(Numeric)
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--use_miss_token_embed \
--use_miss_token_embed_numerical

# Learnable Embed(Image)
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--use_miss_token_embed \
--use_miss_token_embed_image

# Modality Drop.+Learn. Embed(Image)
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--modality_drop_rate 0.3 \
--use_miss_token_embed \
--use_miss_token_embed_image
```

## Integrating Bag of Tricks
```shell
# Stacking
## For Text+Tabular Datasets
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--use_fusion_transformer \
--categorical_convert_to_text \
--categorical_convert_to_text_template latex \
--LeMDA \
--modality_drop_rate 0.3 

## For Image+Text Datasets
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--clip_fusion_mlp \
--clip_high_quality \
--alignment_loss all \
--text_trivial_aug_maxscale 0.1 \
--use_image_aug \
--manifold_mixup \
--LeMDA \
--modality_drop_rate 0.3 

## For Image+Tabular Datasets
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--clip_fusion_mlp \
--clip_high_quality \
--alignment_loss positive-only \
--categorical_convert_to_text \
--categorical_convert_to_text_template latex \
--text_trivial_aug_maxscale 0.1 \
--use_image_aug \
--use_miss_token_embed \
--use_miss_token_embed_image

## For Image+Text+Tabular Datasets
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--clip_fusion_mlp \
--clip_high_quality \
--alignment_loss positive-only \
--text_trivial_aug_maxscale 0.1 \
--use_image_aug \
--modality_drop_rate 0.3 \
--use_miss_token_embed \
--use_miss_token_embed_image

# Average All
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--use_ensemble \
--avg_all \
--model_paths {1.ckpt 2.ckpt...}

# Ensemble Selection
CUDA_VISIBLE_DEVICES=0 python src/autogluon/bench/frameworks/multimodal/exec_local.py \
--params sample_configs/multimodal_local_configs.yaml \
--seed 0 \
--benchmark_dir {output_dir} \
--metrics_dir {output_dir}/results \
--dataset_name {dataset_name} \
--top_k_average_method greedy_soup \
--gradient_clip_val 1. \
--warmup_steps 0.1 \
--lr_decay 0.9 \
--use_ensemble \
--model_paths {1.ckpt 2.ckpt...}
```