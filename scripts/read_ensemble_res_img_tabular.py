import numpy as np
import matplotlib.pyplot as plt
def find_max_eval_metric(file_path, use_avg=False):
    max_eval_metric = float('-inf')
    corresponding_test_metric = None
    max_eval_metric_found = False

    with open(file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        if use_avg:
            if lines[i].startswith("test metric:"):
                test_metric = float(lines[i].split(":")[1].strip())
                return test_metric
            else:
                i += 1
        else:
            if lines[i].startswith("eval error:"):
                eval_error = lines[i].split(":")[1].strip()
                eval_metric = float(lines[i+1].split(":")[1].strip())
                test_error = lines[i+2].split(":")[1].strip()
                test_metric = float(lines[i+3].split(":")[1].strip())
                
                # if eval_metric > max_eval_metric:
                #     max_eval_metric = eval_metric
                #     corresponding_test_metric = test_metric
                #     max_eval_metric_found = True

                # i += 4


                selected_models_temp = []
                j = i + 5
                while j < len(lines) and lines[j].strip() and not lines[j].startswith("best_ensemble_weights:"):
                    selected_models_temp.append(lines[j].strip())
                    j += 1
    
                # print(selected_models_temp)
                best_ensemble_weights_temp = []
                if j < len(lines):
                    weights_str = lines[j].split(":")[1].strip().strip('[]')
                    best_ensemble_weights_temp = [float(w) for w in weights_str.split()]


                if eval_metric > max_eval_metric:
                    max_eval_metric = eval_metric
                    corresponding_test_metric = test_metric
                    selected_models = selected_models_temp
                    best_ensemble_weights = best_ensemble_weights_temp
                    max_eval_metric_found = True

                i = j + 2
            else:
                i += 1

    if max_eval_metric_found:
        # print(f"Max eval metric of {file_path}: {max_eval_metric}")
        # print(f"Corresponding test metric: {corresponding_test_metric}")
        return corresponding_test_metric, selected_models, best_ensemble_weights
    else:
        print("No eval metric found.")

use_avg = False
dataset_weight_dict = {}
for seed in [0,1,2]:
    test_res_list = []
    dataset_str = ""
    for dataset in [ "CCD", "skin_cancer",  "wikiart", "CD18_convert_to_log", "DVM-CAR_convert_to_log", ]:
        dataset_str += f"{dataset} & "
        if use_avg:
            file_path = f'ensemble_results/{dataset}/seed_{seed}/use_avg/log.txt'
        else:
            file_path = f'ensemble_results/{dataset}/seed_{seed}/log.txt'

        res = find_max_eval_metric(file_path, use_avg=use_avg)
        if use_avg:
            test_res_list.append(res)
        else:
            test_metric, selected_models, best_ensemble_weights = res
            test_res_list.append(test_metric)
            # print(selected_models)
            # print(best_ensemble_weights)
            # print()
            if dataset not in dataset_weight_dict:
                dataset_weight_dict[dataset] = {}
            if seed not in dataset_weight_dict[dataset]:
                dataset_weight_dict[dataset][seed] = {}
            dataset_weight_dict[dataset][seed]["selected_models"] = selected_models
            dataset_weight_dict[dataset][seed]["best_ensemble_weights"] = best_ensemble_weights

    # print(test_res_list)
    score_str = ""
    for score in test_res_list:
        score_str += f"{np.around(float(score), 3)} & "
    score_str += str(np.around(np.mean(test_res_list), 3))

    print(f"seed {seed}: ")
    dataset_str += "Avg."
    print(dataset_str)
    print(score_str)
    print()

print(dataset_weight_dict)
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
sorted_models = sorted(average_weights_overall.items(), key=lambda x: x[1], reverse=True)
models_sorted = [model for model, weight in sorted_models]
average_weights_sorted = [weight for model, weight in sorted_models]

# 计算所有模型的平均权重和中位数
overall_mean = np.mean(average_weights_sorted)
overall_median = np.median(average_weights_sorted)


plt.figure(figsize=(10, 6))
plt.bar(models_sorted, average_weights_sorted, color='skyblue')
plt.axhline(overall_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {overall_mean:.4f}')
# plt.axhline(overall_median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {overall_median:.4f}')
plt.xlabel('Model Name')
plt.ylabel('Average Weight')
plt.title('Average Weight of Models (Image+Tabular)')
plt.xticks(rotation=45)
plt.tight_layout()
# 显示图表
plt.savefig("image_tabular_sorted.png")
plt.savefig("image_tabular_sorted.pdf")
print("save to image_tabular_sorted.png")