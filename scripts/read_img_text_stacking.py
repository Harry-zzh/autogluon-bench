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
# 目前已通过： "UPMC-Food101", "fakeddit" 
# 目前未通过："mocheg", "snli-ve",
# 未知："hateful_memes", "persuasive_techniques", "Memotion", 
# 目前看不懂对方是啥意思的："action_effect_pred", 

# for run in [1,2]:
run = 1
for seed in [0,1,2]:
# for seed in [0]:
    # for idx in [1, 2, 3, 4, 5, 6, 7]:
    for idx in [7]:
        for image_lr in [1e-4, 8e-5, 5e-5]:
            for text_lr in [1e-4, 8e-5, 5e-5]:
                # for tabular_lr in [1e-4, 8e-5, 5e-5]:
                for dataset in [
                    # "persuasive_techniques", "Memotion", "UPMC-Food101","snli-ve", "action_effect_pred", "fakeddit"
                    "persuasive_techniques", "Memotion", "UPMC-Food101", "action_effect_pred", "fakeddit"
                    # "Memotion", "fakeddit" # missing
                    # "persuasive_techniques", "UPMC-Food101", "action_effect_pred"
                    # "hateful_memes", "persuasive_techniques",  "mocheg", "Memotion", "UPMC-Food101","snli-ve", "action_effect_pred", "fakeddit" 
                    # "hateful_memes", "persuasive_techniques","Memotion", "snli-ve", "action_effect_pred", "fakeddit" 
                    # "hateful_memes", "persuasive_techniques",  "mocheg", "Memotion", "UPMC-Food101","snli-ve", "action_effect_pred", 
                    # "hateful_memes", "persuasive_techniques",  "Memotion", "snli-ve", "action_effect_pred", "fakeddit"
                    # "yelp", "CCD", "action_effect_entailment", "skin_cancer", "wikiart", "iqa", "Harumanis_mango", "CD18", "DVM-CAR",  
                    # "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "stylish-product", "KARD", "art-price-dataset"
                    # "yelp", "CCD", "action_effect_entailment", "skin_cancer", "wikiart", "iqa", "Harumanis_mango", "CD18", "DVM-CAR",  
                ]:
                    for convert_to_text in [False]:
                        for ft_transformer_pretrained in [False]:
                            # for text_trivial_aug_maxscale, use_image_aug in zip([0.1, ], [True,]):
                            for text_trivial_aug_maxscale, use_image_aug in zip([0.], [False]):
                                for use_fusion_transformer in [False]:
                                    for early_fusion in [False]:
                                        for epoch_num in [20]:
                                    # if epoch_num == 10 and idx in [5,6,7]:
                                    #     continue
                                                                
                                    # lr_decay = lr_decays[idx-1]
                                    # for weight_decay in [0.0001, 0.01]: ### 这里改变了weight_decay和lr_decay!!
                                    #     for lr_decay in [1., 0.9]:
                                            for clip_fusion_mlp in [False]:
                                                for fusion_transformer_concat_all_tokens in [False]:
                                                    for use_different_lr_for_each_modality in [False]:   
                                                        for clip_fusion_mlp_quality in ['high']:      
                                                            for sequential_fusion in [False]:
                                                                for auxiliary_weight in [0.]:
                                                                # for auxiliary_weight in [0.1, 0.2]:
                                                                    LeMDA = False
                                                                    # LeMDA_arch = "trans_vae"
                                                                    LeMDA_arch = "mlp_vae"
                                                                    modality_drop_rate = 0.
                                                                    # modality_drop_rate = 0.2
                                                                    alignment_loss = "" 
                                                                    # alignment_loss = "KL" 
                                                                    # alignment_loss = "KL_feature" 
                                                                    contrastive_loss = ""
                                                                    # contrastive_loss = "contra_fea"
                                                                    # contrastive_loss = "contra_logit"

                                                                    contrastive_loss_w = 1. # default = 0.1
                                                                    # lemda_layer = 2
                                                                    lemda_layer = 6

                                                                    use_miss_token = False
                                                                    manifold_mixup = False
                                                                    use_llama_7B = False

                                                                    use_miss_token_embed_image = True

                                                                        ###### stacking
                                                                    # lf-aligned
                                                                    clip_fusion_mlp = True
                                                                    clip_fusion_mlp_quality = 'high'

                                                                    # align loss
                                                                    alignment_loss = "KL_feature"
                                                                    contrastive_loss = "contra_fea"
                                                                    contrastive_loss_w = 1.

                                                                    # aug
                                                                    text_trivial_aug_maxscale = 0.1
                                                                    use_image_aug = True

                                                                    manifold_mixup = True
                                                                    LeMDA = True

                                                                    # modality dropout
                                                                    modality_drop_rate = 0.3


                                                                    

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

                                                                        # key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/run{run}"
                                                                        # output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/run{run}"
                                                                        key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/"
                                                                        if convert_to_text is False:
                                                                            key_str += f"convert_to_text_{convert_to_text}/"
                                                                        if ft_transformer_pretrained:
                                                                            key_str += f"ft_transformer_pretrained_{ft_transformer_pretrained}/"
                                                                        if text_trivial_aug_maxscale > 0.:
                                                                            key_str += f"text_trivial_aug_maxscale_{text_trivial_aug_maxscale}/"
                                                                        if use_image_aug is False:
                                                                            key_str += f"no_img_aug/"
                                                                        if use_fusion_transformer:
                                                                            key_str += f"use_fusion_transformer_{use_fusion_transformer}/"
                                                                        if early_fusion:
                                                                            key_str += f"early_fusion_{early_fusion}/"
                                                                        if epoch_num != 10:
                                                                            key_str += f"epoch_{epoch_num}/"
                                                                        if clip_fusion_mlp:
                                                                            key_str += f"use_clip_fusion_mlp/"
                                                                        if fusion_transformer_concat_all_tokens and use_fusion_transformer: # 只有在use fusion transformer的情况下使用
                                                                            key_str += f"fusion_transformer_concat_all_tokens_{fusion_transformer_concat_all_tokens}/"
                                                                        if use_different_lr_for_each_modality:
                                                                            key_str += f"use_different_lr_for_each_modality_imagelr{image_lr}_textlr{text_lr}/"
                                                                        if clip_fusion_mlp and clip_fusion_mlp_quality == 'high':
                                                                            key_str += f"clip_fusion_mlp_quality_high/"
                                                                        if sequential_fusion:
                                                                            key_str += f"sequential_fusion/"
                                                                        if auxiliary_weight != 0.1: # 0.1是默认的weight
                                                                            key_str += f"auxiliary_weight_{auxiliary_weight}/"
                                                                        if LeMDA:
                                                                            key_str += f"LeMDA/"
                                                                            if LeMDA_arch != "mlp_vae":
                                                                                key_str += f"LeMDA_arch_{LeMDA_arch}/"
                                                                            if lemda_layer != 4:
                                                                                key_str  += f"lemda_layer_{lemda_layer}/"
                                                                    
                                                                        if modality_drop_rate > 0.:
                                                                            key_str += f"modality_drop_rate_{modality_drop_rate}/"
                                                                        # if alignment_loss == "KL":
                                                                        if alignment_loss != "":
                                                                            key_str += f"{alignment_loss}_align_loss/"
                                                                        if contrastive_loss != "":
                                                                            key_str += f"{contrastive_loss}_contra_loss/"
                                                                        if contrastive_loss != "" and contrastive_loss_w != 0.1:
                                                                            key_str += f"contrastive_loss_w_{contrastive_loss_w}/"
                                                                        if use_miss_token:
                                                                            key_str += f"use_miss_token_{use_miss_token}/"
                                                                            if use_miss_token_embed_image:
                                                                                key_str += f"use_miss_token_{use_miss_token}_image/"
                                                                        if manifold_mixup:
                                                                            key_str += "manifold_mixup/"
                                                                        if use_llama_7B and use_fusion_transformer:
                                                                            key_str  += f"use_llama7B_fusion/"
                                                                
                                                                        




                                                                        if seed!=0:
                                                                            key_str += f"seed_{seed}/"
                                                                     
                                                                                
                                                                                
                                                                        key_str += f"run{run}"
                                                                        # if ft_transformer_pretrained and convert_to_text is False:
                                                                        #     key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/convert_to_text_{convert_to_text}/ft_transformer_pretrained_{ft_transformer_pretrained}/run{run}"
                                                                            
                                                                        # elif ft_transformer_pretrained:
                                                                        #     key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/ft_transformer_pretrained_{ft_transformer_pretrained}/run{run}"
                                                                        # elif convert_to_text is False:
                                                                        #     key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/convert_to_text_{convert_to_text}/run{run}"
                                                                        # else:
                                                                        #     key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/run{run}"
                                                                        
                                                                        output_dir = f"/home/ubuntu/drive2/ag_bench_runs/multimodal/{dataset}/{key_str}"
                                                                        metrics_file = f"{output_dir}/results/AutoGluon_stable.{dataset}.None.local/scores/results.csv"
                                                                        print(metrics_file)
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


# for group, dataset_dict in results_dict.items():
#     print(group)
#     dataset_str = ""
#     metric_str = ""
#     score_str = ""
#     total_score = []
#     for dataset, metric_dict in dataset_dict.items():
#         dataset_str += f"{dataset} & "
#         metric_name, score = list(metric_dict.items())[0]
#         metric_str  += f"{metric_name} & "
#         score_str += f"{np.around(float(score), 3)} & "
#         total_score.append(float(score))
#         print(f"{dataset}: {float(score)}")
#     # 算平均
#     score_str += str(np.around(np.mean(total_score), 3))
#     dataset_str += "Avg."
#     print(len(total_score))
#     print(dataset_str)
#     print(metric_str)
#     print(score_str)
#     print()



## 如果有多个种子，计算多个种子的平均
cal_std = False # 是否要计算方差
seed_dict = {}
for group, dataset_dict in results_dict.items():
    pos = group.find("seed")

    # 如果找到子字符串，提取之前的部分
    if pos != -1:
        group = group[:pos]
    else:
        pos = group.find("run")
        group = group[:pos]
    if group not in seed_dict:
        seed_dict[group] = []
    seed_dict[group].append(dataset_dict)

for group, seed_dataset_dict in seed_dict.items(): # group是1～7某种组合
    total_score = []
    final_score = [] # 保存每个seed在每个数据集上的平均值
    print("group: ", group)
    for seed, dataset_dict in enumerate(seed_dataset_dict):
        dataset_str = ""
        metric_str = ""
        score_str = ""
        score_list = []

        
        for dataset, metric_dict in dataset_dict.items():
            dataset_str += f"{dataset} & "
            metric_name, score = list(metric_dict.items())[0]
            metric_str  += f"{metric_name} & "
            score_str += f"{np.around(float(score), 3)} & "
            score_list.append(float(score))

        # 算每个seed下的平均
        score_str += str(np.around(np.mean(score_list), 3))
        print(len(score_list))
        print(f"seed {seed}: ")
        dataset_str += "Avg."
        print(dataset_str)
        print(metric_str)
        print(score_str)
        print()

        final_score.append(np.mean(score_list))

        total_score.append(score_list)
    # 每个seed在每种组合下的总分
    dataset_num = len(total_score[0])
    # for i in total_score:
    #     total_score[dataset_num]
    
    score_str = ""
    for i in range(dataset_num):
        cur_seed_score = []
        for seed in range(len(seed_dataset_dict)):
            cur_seed_score.append(total_score[seed][i])
        # final_score.append(np.mean(cur_seed_score))
        # final_std.append(np.std(cur_seed_score))
        if cal_std:
            score_str += f"{np.around(np.mean(cur_seed_score), 3)}({np.around(np.std(cur_seed_score), 3)}) & "
        else:
            score_str += f"{np.around(np.mean(cur_seed_score), 3)} & "
    # 算平均
    if cal_std:
        score_str += f"{np.around(np.mean(final_score), 3)}({np.around(np.std(final_score), 3)})"
    else:
        score_str += str(np.around(np.mean(final_score), 3))
    print(len(final_score))
    print(f"seed total: ")
    print(dataset_str)
    print(metric_str)
    print(score_str)
    print()


   


