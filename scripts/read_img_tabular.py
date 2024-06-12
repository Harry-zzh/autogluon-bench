import numpy as np
weight_decays = [0., 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
top_k_average_methods = ["best", "best", "best", "best", "best", "greedy_soup", "greedy_soup"]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

results_dict = {}

# for run in [1]:
#     for idx in [1, 2, 3, 4, 5, 6, 7]:
#         # for dataset in ["persuasive_techniques", "yelp", "seattle_airbnb",  "hateful_memes", "mmimdb", "grocery_image","petfinder", "san_francisco_airbnb",]:
#         for dataset in ["persuasive_techniques", "yelp", "hateful_memes", "petfinder", "seattle_airbnb", ]:
#             try:
#                 weight_decay = weight_decays[idx-1]
#                 gradient_clip_val = gradient_clip_vals[idx-1]
#                 lr_decay = lr_decays[idx-1]
#                 warmup_steps = warmup_stepss[idx-1]
#                 lr_schedule = lr_schedules[idx-1]
#                 top_k_average_method = top_k_average_methods[idx-1]


#                 key_str = f"weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
#                 output_dir = f"ag_bench_runs/multimodal/{dataset}/weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
#                 metrics_file = f"{output_dir}/results/AutoGluon_stable.{dataset}.None.local/scores/results.csv"
#                 with open(metrics_file, 'r') as f:
#                     lines = f.readlines()
#                 metric_name = lines[0].strip().split(",")[-1]
#                 score = lines[1].strip().split(",")[-1]

#                 if key_str not in results_dict:
#                     results_dict[key_str] = {}
#                 results_dict[key_str][dataset] = {}
#                 results_dict[key_str][dataset][metric_name] = score
#             except Exception:
#                 continue 
# ### 第三组
# top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
# # weight_decays = [0., 0., 0., 0., 0., 0.001, 0.001]
# # lr_decays = [1., 1., 1., 1., 1., 1., 0.9]
# # weight_decays = [0., 0., 0., 0., 0., 0.0001, 0.01]
# # lr_decays = [1., 1., 1., 1., 1., 1., 1.]
# weight_decays = [0., 0., 0., 0., 0., 0., 0.]
# lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

### 第六组 tune warmup steps
# top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# warmup_stepss = [0., 0., 0., 0.1, 0.2, 0.3, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
# weight_decays = [0., 0., 0., 0., 0., 0., 0.]
# lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

# ### 第七组 不使用cosine decay和weight decay
# top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.3, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay"]
# weight_decays = [0., 0., 0., 0., 0., 0., 0.]
# lr_decays = [1., 1., 1., 1., 0.9, 0.9, 0.9]


top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
weight_decays = [0., 0., 0., 0., 0., 0.001, 0.001]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

convert_to_log = True # 通过baseline的结果，以后跑price数据集都用convert_to_log
# price_datasets = ["seattle_airbnb", "goodreads", "crypto-coven", "CD18", "DVM-CAR", "stylish-product", "art-price-dataset", "nike"]
price_datasets = ["seattle_airbnb", "goodreads", "crypto-coven", "CD18", "DVM-CAR", "stylish-product",  "nike"]

image_tabular_datasets = [
    # "skin_cancer", "Harumanis_mango", "CD18",
    "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "goodreads", "stylish-product", "KARD", 
                          ]

for run in [1]:
    for idx in [1, 2, 3, 4, 5, 6, 7]:
        # for dataset in ["persuasive_techniques", "yelp", "seattle_airbnb",  "hateful_memes", "mmimdb", "grocery_image","petfinder", "san_francisco_airbnb",]:
        # for dataset in ["persuasive_techniques", "yelp", "hateful_memes", "petfinder", "seattle_airbnb", ]:
        # for dataset in ["hateful_memes", "persuasive_techniques", "yelp", "mocheg", "CCD", "action_effect_entailment"]:
        # for dataset in ["seattle_airbnb", "goodreads", "crypto-coven", "nike", "CD18", "DVM-CAR", "stylish-product", "amazon-books-reviews", "KARD", "art-price-dataset", "impressions_dataset"]:
        # for dataset in ["seattle_airbnb_convert_to_log", "goodreads_convert_to_log", "crypto-coven_convert_to_log", "CD18_convert_to_log", "DVM-CAR_convert_to_log", "stylish-product_convert_to_log", "art-price-dataset_convert_to_log"]:
        # for dataset in [# "hateful_memes", "persuasive_techniques", "impressions_dataset"
        #     # "mocheg", "Memotion", "UPMC-Food101","snli-ve", "action_effect_pred", 
        #                 # "yelp", "CCD", "action_effect_entailment", "skin_cancer", "wikiart", "iqa", "Harumanis_mango", "CD18", "DVM-CAR",  
        #                 # "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "goodreads", "stylish-product", "KARD", 
        #                 ]:
        # for dataset in image_tabular_datasets:
        # for dataset in ["petfinder", "covid-chestxray-dataset", "seattle_airbnb", "goodreads", "crypto-coven", "nike", "amazon-books-reviews", "stylish-product", "KARD", "art-price-dataset"]:
        # for dataset in ["nike"]:
        # for dataset in ["hateful_memes", "persuasive_techniques",  "Memotion", "snli-ve", "action_effect_pred",  ]:
        # for dataset in ["skin_cancer", "Harumanis_mango", "CD18",]:
        # for dataset in ["petfinder", "covid-chestxray-dataset", "seattle_airbnb", "stylish-product", "KARD", "art-price-dataset"]:
        # for dataset in ["art-price-dataset"]:
        for dataset in [
            # "hateful_memes", "persuasive_techniques",  "Memotion", "snli-ve", "action_effect_pred", 
            # "yelp", "CCD", "action_effect_entailment", "skin_cancer", "wikiart", "iqa", "Harumanis_mango", "CD18", "DVM-CAR",  
            # "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "stylish-product", "KARD", "art-price-dataset"
            # "fakeddit"
        ]:
            for ft_transformer_pretrained in [False,]:

                try:
                    weight_decay = weight_decays[idx-1]
                    gradient_clip_val = gradient_clip_vals[idx-1]
                    lr_decay = lr_decays[idx-1]
                    warmup_steps = warmup_stepss[idx-1]
                    lr_schedule = lr_schedules[idx-1]
                    top_k_average_method = top_k_average_methods[idx-1]

                    if convert_to_log and dataset in price_datasets:
                        ori_dataset = dataset
                        dataset = dataset + "_convert_to_log"
                        # print(dataset)

                    key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/run{run}"
                    # output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/run{run}"
                    
                    if ft_transformer_pretrained:
                        output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/ft_transformer_pretrained_{ft_transformer_pretrained}/run{run}"
                    else:
                        output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/run{run}"
                    
                    metrics_file = f"{output_dir}/results/AutoGluon_stable.{dataset}.None.local/scores/results.csv"
                    with open(metrics_file, 'r') as f:
                        lines = f.readlines()
                    metric_name = lines[0].strip().split(",")[-1]
                    score = lines[1].strip().split(",")[-1]

                    if key_str not in results_dict:
                        results_dict[key_str] = {}
                    results_dict[key_str][dataset] = {}
                    results_dict[key_str][dataset][metric_name] = score
                except Exception:
                    continue 


for group, dataset_dict in results_dict.items():
    print(group)
    dataset_str = ""
    metric_str = ""
    score_str = ""
    total_score = []
    for dataset, metric_dict in dataset_dict.items():
        dataset_str += f"{dataset} & "
        metric_name, score = list(metric_dict.items())[0]
        metric_str  += f"{metric_name} & "
        score_str += f"{np.around(float(score), 3)} & "
        total_score.append(float(score))
        print(f"{dataset}: {float(score)}")
    # 算平均
    score_str += str(np.around(np.mean(total_score), 3))
    dataset_str += "Avg."
    print(len(total_score))
    print(dataset_str)
    print(metric_str)
    print(score_str)
    print()