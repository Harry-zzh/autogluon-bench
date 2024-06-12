import numpy as np
import matplotlib.pyplot as plt
text_tabular_dict = {'fake': {0: {'selected_models': ['Baseline+', 'Feature Aug.(Joint)', 'Input Aug.', 'Convert Categorical', 'Positive+Negative'], 'best_ensemble_weights': [0.2, 0.1, 0.5, 0.1, 0.1]}, 1: {'selected_models': ['Positive+Negative', 'Early Fusion', 'Feature Aug.(Joint)'], 'best_ensemble_weights': [0.5, 0.33333333, 0.16666667]}, 2: {'selected_models': ['LF-Transformer', 'Input Aug.', 'LF-Aligned', 'Baseline+', 'LF-SF', 'Convert Categorical'], 'best_ensemble_weights': [0.2, 0.0, 0.2, 0.2, 0.3, 0.1]}}, 'qaa': {0: {'selected_models': ['Input Aug.', 'LF-SF', 'Modality Dropout', 'Feature Aug.(Joint)', 'LF-Transformer'], 'best_ensemble_weights': [0.375, 0.25, 0.125, 0.125, 0.125]}, 1: {'selected_models': ['Modality Dropout', 'Baseline+', 'LF-SF', 'Convert Categorical', 'LF-Aligned'], 'best_ensemble_weights': [0.3, 0.2, 0.2, 0.2, 0.1]}, 2: {'selected_models': ['Positive-only', 'Feature Aug.(Joint)', 'Convert Categorical', 'LF-SF', 'Baseline+'], 'best_ensemble_weights': [0.4, 0.2, 0.2, 0.1, 0.1]}}, 'qaq': {0: {'selected_models': ['Input Aug.', 'Modality Dropout', 'Early Fusion', 'Convert Categorical', 'LF-SF'], 'best_ensemble_weights': [0.4, 0.2, 0.1, 0.2, 0.1]}, 1: {'selected_models': ['Convert Categorical', 'LF-Transformer', 'Positive-only', 'LF-Aligned', 'Feature Aug.(Inde.)', 'LF-SF'], 'best_ensemble_weights': [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]}, 2: {'selected_models': ['Convert Categorical', 'Feature Aug.(Inde.)', 'Positive-only', 'LF-SF'], 'best_ensemble_weights': [0.4, 0.3, 0.2, 0.1]}}, 'airbnb': {0: {'selected_models': ['LF-Transformer', 'Convert Categorical', 'Modality Dropout', 'LF-LLM', 'Baseline+', 'Convert Numerical', 'LF-Aligned', 'Early Fusion', 'LF-SF', 'Positive-only', 'Positive+Negative', 'Input Aug.', 'Feature Aug.(Inde.)', 'Learnable Embed(Numerical)', 'Feature Aug.(Joint)'], 'best_ensemble_weights': [0.1, 0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1]}, 1: {'selected_models': ['Positive+Negative', 'LF-SF', 'Early Fusion', 'Convert Categorical', 'LF-Aligned', 'Learnable Embed(Numerical)', 'Modality Dropout'], 'best_ensemble_weights': [0.1, 0.2, 0.0, 0.3, 0.1, 0.2, 0.1]}, 2: {'selected_models': ['Positive+Negative', 'LF-Aligned', 'Early Fusion', 'Positive-only', 'Convert Numerical'], 'best_ensemble_weights': [0.33333333, 0.16666667, 0.16666667, 0.16666667, 0.16666667]}}, 'channel': {0: {'selected_models': ['Feature Aug.(Inde.)', 'Convert Categorical', 'LF-Transformer', 'Input Aug.', 'Baseline+', 'LF-LLM', 'Convert Numerical', 'Early Fusion', 'LF-SF', 'Positive-only', 'LF-Aligned'], 'best_ensemble_weights': [0.33333333, 0.11111111, 0.0, 0.22222222, 0.0, 0.0]}, 1: {'selected_models': ['Input Aug.', 'Convert Categorical', 'LF-Transformer', 'Baseline+', 'LF-Aligned', 'LF-LLM', 'Positive-only', 'Modality Dropout'], 'best_ensemble_weights': [0.3, 0.2, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1]}, 2: {'selected_models': ['Positive-only', 'Convert Numerical', 'LF-SF', 'LF-Aligned', 'Early Fusion', 'Modality Dropout', 'Positive+Negative', 'Convert Categorical', 'LF-Transformer', 'Feature Aug.(Joint)', 'LF-LLM'], 'best_ensemble_weights': [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.1, 0.1]}}, 'cloth': {0: {'selected_models': ['Input Aug.', 'Convert Numerical', 'LF-Aligned', 'Positive-only', 'LF-Transformer', 'Modality Dropout'], 'best_ensemble_weights': [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]}, 1: {'selected_models': ['Positive+Negative', 'Feature Aug.(Joint)', 'LF-LLM', 'Convert Numerical', 'LF-Aligned', 'Baseline+', 'LF-SF'], 'best_ensemble_weights': [0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1]}, 2: {'selected_models': ['Convert Categorical', 'LF-Transformer', 'Baseline+', 'LF-SF', 'LF-Aligned', 'Feature Aug.(Joint)', 'Modality Dropout', 'Positive+Negative'], 'best_ensemble_weights': [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}}}
img_text_dict = {'persuasive_techniques': {0: {'selected_models': ['LF-Aligned', 'LF-Transformer'], 'best_ensemble_weights': [0.6, 0.4]}, 1: {'selected_models': ['LF-Aligned', 'LF-Transformer', 'LF-SF'], 'best_ensemble_weights': [0.5, 0.16666667, 0.33333333]}, 2: {'selected_models': ['LF-Aligned', 'LF-Transformer'], 'best_ensemble_weights': [0.85714286, 0.14285714]}}, 'Memotion': {0: {'selected_models': ['Positive-only', 'LF-Transformer'], 'best_ensemble_weights': [0.55555556, 0.44444444]}, 1: {'selected_models': ['LF-Transformer', 'Feature Aug.(Joint)'], 'best_ensemble_weights': [0.66666667, 0.33333333]}, 2: {'selected_models': ['Positive+Negative', 'LF-Aligned'], 'best_ensemble_weights': [0.88888889, 0.11111111]}}, 'UPMC-Food101': {0: {'selected_models': ['Modality Dropout', 'LF-Aligned', 'LF-LLM', 'Positive-only', 'Baseline+', 'LF-Transformer', 'Input Aug.', 'Positive+Negative'], 'best_ensemble_weights': [0.2, 0.4, 0.0, 0.1, 0.0, 0.1, 0.1, 0.1]}, 1: {'selected_models': ['Positive-only', 'Baseline+', 'LF-Aligned', 'LF-SF', 'LF-Transformer', 'LF-LLM', 'Positive+Negative', 'Input Aug.', 'Feature Aug.(Inde.)', 'Feature Aug.(Joint)'], 'best_ensemble_weights': [0.1, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1]}, 2: {'selected_models': ['LF-Aligned', 'Baseline+', 'Positive+Negative', 'LF-Transformer', 'LF-LLM', 'Positive-only', 'Input Aug.', 'Feature Aug.(Inde.)', 'Feature Aug.(Joint)'], 'best_ensemble_weights': [0.375, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.25, 0.125]}}, 'action_effect_pred': {0: {'selected_models': ['LF-Aligned', 'Baseline+', 'LF-Transformer', 'LF-LLM'], 'best_ensemble_weights': [0.4, 0.3, 0.2, 0.1]}, 1: {'selected_models': ['LF-Aligned', 'Modality Dropout'], 'best_ensemble_weights': [0.66666667, 0.33333333]}, 2: {'selected_models': ['LF-Aligned', 'Positive-only', 'Positive+Negative'], 'best_ensemble_weights': [0.5, 0.33333333, 0.16666667]}}, 'fakeddit': {0: {'selected_models': ['LF-Aligned', 'LF-SF', 'Positive+Negative'], 'best_ensemble_weights': [0.66666667, 0.16666667, 0.16666667]}, 1: {'selected_models': ['LF-Aligned', 'Positive+Negative', 'Baseline+', 'LF-Transformer', 'LF-LLM', 'Feature Aug.(Inde.)'], 'best_ensemble_weights': [0.6, 0.1, 0.0, 0.0, 0.1, 0.2]}, 2: {'selected_models': ['LF-Aligned', 'LF-LLM', 'Feature Aug.(Inde.)', 'Modality Dropout', 'Baseline+', 'LF-Transformer', 'Positive-only', 'Input Aug.', 'Feature Aug.(Joint)', 'Learnable Embed(Image)', 'LF-SF', 'Positive+Negative'], 'best_ensemble_weights': [0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1]}}}
img_tabular_dict = {'CCD': {0: {'selected_models': ['Input Aug.', 'LF-Aligned', 'Feature Aug.(Inde.)'], 'best_ensemble_weights': [0.33333333, 0.55555556, 0.11111111]}, 1: {'selected_models': ['LF-Aligned', 'Positive+Negative', 'Positive-only'], 'best_ensemble_weights': [0.55555556, 0.33333333, 0.11111111]}, 2: {'selected_models': ['LF-Aligned', 'Feature Aug.(Joint)', 'Modality Dropout'], 'best_ensemble_weights': [0.5, 0.33333333, 0.16666667]}}, 'skin_cancer': {0: {'selected_models': ['Input Aug.', 'Baseline+', 'LF-Aligned', 'Positive+Negative', 'Feature Aug.(Inde.)'], 'best_ensemble_weights': [0.28571429, 0.28571429, 0.14285714, 0.14285714, 0.14285714]}, 1: {'selected_models': ['Baseline+', 'Feature Aug.(Joint)', 'Feature Aug.(Inde.)'], 'best_ensemble_weights': [0.25, 0.375, 0.375]}, 2: {'selected_models': ['Baseline+', 'LF-Transformer', 'Modality Dropout', 'LF-Aligned', 'LF-LLM', 'Positive-only', 'Positive+Negative', 'Input Aug.', 'Convert Categorical', 'Feature Aug.(Inde.)', 'LF-SF'], 'best_ensemble_weights': [0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.1]}}, 'wikiart': {0: {'selected_models': ['LF-Aligned', 'LF-Transformer', 'Feature Aug.(Joint)', 'LF-LLM'], 'best_ensemble_weights': [0.28571429, 0.28571429, 0.14285714, 0.28571429]}, 1: {'selected_models': ['LF-Aligned', 'Feature Aug.(Joint)', 'LF-Transformer', 'Convert Categorical', 'Baseline+', 'LF-LLM'], 'best_ensemble_weights': [0.25, 0.125, 0.125, 0.25, 0.125, 0.125]}, 2: {'selected_models': ['LF-Aligned', 'LF-Transformer', 'LF-LLM', 'Input Aug.', 'Baseline+', 'Positive-only', 'Positive+Negative', 'LF-SF', 'Convert Categorical', 'Modality Dropout'], 'best_ensemble_weights': [0.2, 0.3, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.1, 0.1]}}, 'CD18_convert_to_log': {0: {'selected_models': ['Baseline+', 'LF-Aligned', 'LF-SF', 'Convert Categorical', 'Positive-only', 'Feature Aug.(Joint)', 'Input Aug.'], 'best_ensemble_weights': [0.11111111, 0.33333333, 0.11111111, 0.11111111, 0.11111111, 0.11111111]}, 1: {'selected_models': ['Positive-only', 'LF-Aligned', 'Learnable Embed(Image)', 'Convert Categorical', 'LF-SF', 'LF-LLM'], 'best_ensemble_weights': [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]}, 2: {'selected_models': ['LF-LLM', 'LF-Aligned', 'Convert Categorical', 'Feature Aug.(Joint)'], 'best_ensemble_weights': [0.4, 0.3, 0.2, 0.1]}}, 'DVM-CAR_convert_to_log': {0: {'selected_models': ['Convert Categorical', 'LF-SF', 'Input Aug.'], 'best_ensemble_weights': [0.8, 0.1, 0.1]}, 1: {'selected_models': ['Convert Categorical', 'LF-SF', 'LF-Aligned'], 'best_ensemble_weights': [0.8, 0.1, 0.1]}, 2: {'selected_models': ['Convert Categorical', 'LF-SF'], 'best_ensemble_weights': [0.83333333, 0.16666667]}}}
img_text_tabular_dict = {'petfinder': {0: {'selected_models': ['LF-Aligned', 'Feature Aug.(Joint)', 'Convert Numerical'], 'best_ensemble_weights': [0.6, 0.3, 0.1]}, 1: {'selected_models': ['LF-Aligned', 'Input Aug.', 'LF-LLM', 'Feature Aug.(Inde.)', 'LF-SF'], 'best_ensemble_weights': [0.1, 0.4, 0.1, 0.1, 0.3]}, 2: {'selected_models': ['Modality Dropout', 'LF-Transformer'], 'best_ensemble_weights': [0.5, 0.5]}}, 'covid-chestxray-dataset': {0: {'selected_models': ['LF-Aligned'], 'best_ensemble_weights': [1.0]}, 1: {'selected_models': ['LF-Aligned'], 'best_ensemble_weights': [1.0]}, 2: {'selected_models': ['LF-Aligned', 'LF-Transformer'], 'best_ensemble_weights': [0.75, 0.25]}}, 'art-price-dataset': {0: {'selected_models': ['LF-Transformer', 'LF-Aligned', 'Modality Dropout', 'Baseline+', 'Positive+Negative', 'Convert Categorical', 'Convert Numerical', 'Input Aug.', 'Feature Aug.(Inde.)', 'Positive-only'], 'best_ensemble_weights': [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2]}, 1: {'selected_models': ['LF-Aligned', 'LF-Transformer', 'LF-SF', 'Convert Categorical', 'Baseline+', 'Modality Dropout'], 'best_ensemble_weights': [0.16666667, 0.33333333, 0.0, 0.16666667, 0.16666667, 0.16666667]}, 2: {'selected_models': ['LF-Aligned', 'Feature Aug.(Joint)', 'Baseline+', 'LF-Transformer', 'LF-LLM', 'Feature Aug.(Inde.)', 'Convert Numerical'], 'best_ensemble_weights': [0.25, 0.125, 0.0, 0.0, 0.125, 0.375, 0.125]}}, 'seattle_airbnb_convert_to_log': {0: {'selected_models': ['LF-Aligned', 'LF-LLM', 'Modality Dropout'], 'best_ensemble_weights': [0.6, 0.2, 0.2]}, 1: {'selected_models': ['LF-Aligned', 'LF-LLM', 'Modality Dropout', 'Learnable Embed(Numerical)'], 'best_ensemble_weights': [0.55555556, 0.22222222, 0.11111111, 0.11111111]}, 2: {'selected_models': ['LF-Aligned', 'LF-LLM', 'Modality Dropout'], 'best_ensemble_weights': [0.7, 0.2, 0.1]}}, 'goodreads_convert_to_log': {0: {'selected_models': ['Input Aug.', 'LF-Aligned', 'Positive-only', 'Convert Numerical', 'LF-LLM', 'Convert Categorical'], 'best_ensemble_weights': [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]}, 1: {'selected_models': ['Input Aug.', 'LF-Aligned', 'Positive-only', 'Feature Aug.(Joint)', 'Baseline+'], 'best_ensemble_weights': [0.3, 0.3, 0.2, 0.1, 0.1]}, 2: {'selected_models': ['LF-Aligned', 'Feature Aug.(Joint)', 'Convert Categorical', 'LF-LLM'], 'best_ensemble_weights': [0.44444444, 0.33333333, 0.22222222, 0.0]}}, 'KARD': {0: {'selected_models': ['Convert Numerical', 'Convert Categorical', 'LF-Aligned', 'LF-LLM'], 'best_ensemble_weights': [0.7, 0.1, 0.1, 0.1]}, 1: {'selected_models': ['Convert Numerical', 'LF-Aligned', 'Modality Dropout'], 'best_ensemble_weights': [0.66666667, 0.16666667, 0.16666667]}, 2: {'selected_models': ['Convert Numerical', 'Convert Categorical', 'Feature Aug.(Inde.)'], 'best_ensemble_weights': [0.8, 0.1, 0.1]}}}

dataset_weight_dict = {}
dataset_weight_dict.update(text_tabular_dict)
dataset_weight_dict.update(img_text_dict)
dataset_weight_dict.update(img_tabular_dict)
dataset_weight_dict.update(img_text_tabular_dict)
####### 计算平均被选中的次数
num_datasets = len(dataset_weight_dict)
num_seeds = len(next(iter(dataset_weight_dict.values())))
print("num_seeds: ", num_seeds)
print("num_datasets: ", num_datasets )
print()
# 初始化结果字典
model_weights_per_dataset = {}
total_model_weights = {}

# 如果有模型不存在，给全0权重？
# 遍历每个数据集
for dataset, seeds in dataset_weight_dict.items():
    if dataset not in model_weights_per_dataset:
        model_weights_per_dataset[dataset] = {}

    # 初始化模型权重累加字典
    for seed in seeds:
        models = seeds[seed]['selected_models']
        weights = seeds[seed]['best_ensemble_weights']
        for model, weight in zip(models, weights):
            if "seed" in model:
                model = model.split('seed_')[0]
            else:
                model = model.split("run1/models/model.ckpt")[0]
            model = model.split(dataset)[-1]
            if model not in model_weights_per_dataset[dataset]:
                model_weights_per_dataset[dataset][model] = []
            model_weights_per_dataset[dataset][model].append(weight)
            # if "/home/ubuntu/drive2/ag_bench_runs/multimodal/fake/top_k_average_method_greedy_soup/gradient_clip_val_1.0/weight_decay_0.001/warmup_steps_0.1/lr_schedule_cosine_decay/lr_decay_0.9/convert_to_text_True/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/categorical_template_latex/no_hf_text_insert_sep_False/" in model:
            #     print(model_weights_per_dataset[dataset][model])
            #     print(model)
            #     print()

            
            
            if model not in total_model_weights:
                total_model_weights[model] = []
            total_model_weights[model].append(weight)

# 计算每个模型在每个数据集上的平均权重
average_weights_per_dataset = {}
for dataset, models in model_weights_per_dataset.items():
    average_weights_per_dataset[dataset] = {}
    for model, weights in models.items():
        average_weights_per_dataset[dataset][model] = sum(weights) / num_seeds

# 计算每个模型在所有数据集上的平均权重
average_weights_overall = {}


for model, weights in total_model_weights.items():
    average_weights_overall[model] = sum(weights) / (num_datasets * num_seeds)

# 输出结果
print("每个模型在每个数据集上的平均权重：")
for dataset, models in average_weights_per_dataset.items():
    print(f"{dataset}:")
    for model, avg_weight in models.items():
        print(f"  {model}: {avg_weight:.4f}")
   
    # print(model_weights_per_dataset[dataset]["/home/ubuntu/drive2/ag_bench_runs/multimodal/fake/top_k_average_method_greedy_soup/gradient_clip_val_1.0/weight_decay_0.001/warmup_steps_0.1/lr_schedule_cosine_decay/lr_decay_0.9/convert_to_text_True/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/categorical_template_latex/no_hf_text_insert_sep_False/"])
    # break

print("\n每个模型在所有数据集上的平均权重：")
for model, avg_weight in average_weights_overall.items():
    print(f"{model}: {avg_weight:.4f}")

print("模型总个数：", len(model_weights_per_dataset[dataset]))

# dataset = "fake"
# for seed in [0,1,2]:
#     print(len(dataset_weight_dict[dataset][seed]["selected_models"]))
#     for i, model in enumerate(dataset_weight_dict[dataset][seed]["selected_models"]):
#         if "/home/ubuntu/drive2/ag_bench_runs/multimodal/fake/top_k_average_method_greedy_soup/gradient_clip_val_1.0/weight_decay_0.001/warmup_steps_0.1/lr_schedule_cosine_decay/lr_decay_0.9/convert_to_text_True/ft_transformer_pretrained_False/auxiliary_weight_0.0/max_epochs_20/categorical_template_latex/no_hf_text_insert_sep_False/" in model:
#             print(model)
#             print(dataset_weight_dict[dataset][seed]["best_ensemble_weights"][i])
#             print()
    
# 画柱状图
models = list(average_weights_overall.keys())
print(len(models))
average_weights = list(average_weights_overall.values())

# 先对模型的平均权重按降序排序
sorted_models = sorted(average_weights_overall.items(), key=lambda x: x[1]) #, reverse=True)
models_sorted = [model for model, weight in sorted_models]
average_weights_sorted = [weight for model, weight in sorted_models]


# 计算所有模型的平均权重和中位数
overall_mean = np.mean(average_weights_sorted)
overall_median = np.median(average_weights_sorted)


plt.figure(figsize=(10, 6))
# plt.subplot(151)
plt.barh(models_sorted, average_weights_sorted, color='skyblue')
plt.axvline(overall_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {overall_mean:.4f}')
# plt.axhline(overall_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {overall_median:.4f}')
# plt.xlabel('Model Name')
# plt.ylabel('Average Weight')
plt.title('Average Weight of Models (All)')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.gca().get_yaxis().set_ticks([])

# plt.figure(figsize=(10, 6))
# plt.subplot(151)
# plt.barh(models_sorted, average_weights_sorted, color='skyblue')
# plt.axhline(overall_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {overall_mean:.4f}')
# # plt.axhline(overall_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {overall_median:.4f}')
# # plt.xlabel('Model Name')
# # plt.ylabel('Average Weight')
# # plt.title('Average Weight of Models (All)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# # plt.gca().get_yaxis().set_ticks([])

# plt.subplot(152)
# plt.barh(models_sorted, average_weights_sorted, color='skyblue')
# plt.axhline(overall_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {overall_mean:.4f}')
# # plt.axhline(overall_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {overall_median:.4f}')
# # plt.xlabel('Model Name')
# # plt.ylabel('Average Weight')
# # plt.title('Average Weight of Models (All)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# # plt.gca().get_yaxis().set_ticks([])

# plt.subplot(153)
# plt.barh(models_sorted, average_weights_sorted, color='skyblue')
# plt.axhline(overall_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {overall_mean:.4f}')
# # plt.axhline(overall_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {overall_median:.4f}')
# # plt.xlabel('Model Name')
# # plt.ylabel('Average Weight')
# # plt.title('Average Weight of Models (All)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# # plt.gca().get_yaxis().set_ticks([])

# plt.subplot(154)
# plt.barh(models_sorted, average_weights_sorted, color='skyblue')
# plt.axhline(overall_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {overall_mean:.4f}')
# # plt.axhline(overall_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {overall_median:.4f}')
# # plt.xlabel('Model Name')
# # plt.ylabel('Average Weight')
# # plt.title('Average Weight of Models (All)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# # plt.gca().get_yaxis().set_ticks([])

# plt.subplot(155)
# plt.barh(models_sorted, average_weights_sorted, color='skyblue')
# plt.axhline(overall_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {overall_mean:.4f}')
# # plt.axhline(overall_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {overall_median:.4f}')
# # plt.xlabel('Model Name')
# # plt.ylabel('Average Weight')
# # plt.title('Average Weight of Models (All)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# # plt.gca().get_yaxis().set_ticks([])

# 显示图表
plt.savefig("benchmark_sorted.png")
plt.savefig("benchmark_sorted_1.pdf")
print("save to benchmark_sorted.png")