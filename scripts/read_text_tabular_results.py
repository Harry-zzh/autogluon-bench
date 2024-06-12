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
#         for dataset in ["imdb", "fake", "qaa", "qaq", "book", "jc","prod",  "salary"]:
#             weight_decay = weight_decays[idx-1]
#             gradient_clip_val = gradient_clip_vals[idx-1]
#             lr_decay = lr_decays[idx-1]
#             warmup_steps = warmup_stepss[idx-1]
#             lr_schedule = lr_schedules[idx-1]
#             top_k_average_method = top_k_average_methods[idx-1]

#             key_str = f"weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
#             output_dir = f"ag_bench_runs/multimodal/{dataset}/weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
#             metrics_file = f"{output_dir}/results/AutoGluon_stable.{dataset}.None.local/scores/results.csv"
#             with open(metrics_file, 'r') as f:
#                 lines = f.readlines()
#             metric_name = lines[0].strip().split(",")[-1]
#             score = lines[1].strip().split(",")[-1]

#             if key_str not in results_dict:
#                 results_dict[key_str] = {}
#             results_dict[key_str][dataset] = {}
#             results_dict[key_str][dataset][metric_name] = score            

### 第二组
# top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# weight_decays = [0., 0., 0., 0.001, 0.001, 0.001, 0.001]
# warmup_stepss = [0., 0., 0., 0., 0.1, 0.1, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay"]
# lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

# for run in [1]:
#     for idx in [1, 2, 3, 4, 5, 6, 7]:
#         for dataset in ["imdb", "fake", "qaa", "qaq", "book", "jc","prod",  "salary"]:
#             try:
#                 weight_decay = weight_decays[idx-1]
#                 gradient_clip_val = gradient_clip_vals[idx-1]
#                 lr_decay = lr_decays[idx-1]
#                 warmup_steps = warmup_stepss[idx-1]
#                 lr_schedule = lr_schedules[idx-1]
#                 top_k_average_method = top_k_average_methods[idx-1]

#                 key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/run{run}"
#                 output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/run{run}"
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


# top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
# lr_decays = [1., 1., 1., 1., 1., 0.9, 0.9]
# weight_decays = [0., 0., 0., 0., 0., 0., 0.0001]

top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "constant"]
weight_decays = [0., 0., 0., 0., 0., 0.001, 0.001]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]
# weight_decays = [0., 0., 0., 0., 0., 0., 0.]
# lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
lr_schedules = ["cosine_decay", "cosine_decay", "cosine_decay", "cosine_decay", "cosine_decay", "cosine_decay", "cosine_decay"]
weight_decays = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

# for run in [1]:
run = 1
for seed in [0,1,2]:
# for seed in [1,2]:
    # for idx in [1, 2, 3, 4, 5, 6, 7]:
    for idx in [7,]:
    # for idx in [1,2,3,4,5,6,7]:
    # for idx in [1,7]:
    # for idx in [1,2,3,4,5,6,7]:
    # for idx in [7]:
        # for dataset in ["qaa", "qaq", "book", "jc", "salary", "cloth", "airbnb","channel", "imdb", "prod", "fake"]:
        # for dataset in ["imdb", "fake", "qaa", "qaq", "book", "jc","prod",  "salary"]:
        # for dataset in ["imdb", "airbnb","channel", "book", "cloth", ]: # jc这个数据集太离谱了。太不稳定了。或许这个数据集就是需要unimodal的loss。
        for dataset in ["fake", "airbnb", "channel", "qaa", "qaq", "cloth"]:
        # for dataset in ["fake", "airbnb", "cloth"]: # missing
        # for dataset in ["channel", "qaa" ,"qaq", ]: 
            for convert_to_text in [False]:
                # for ft_transformer_pretrained in [True]:
                for ft_transformer_pretrained in [False]:
                    for use_fusion_transformer in [False]:
                        for early_fusion in [True]:
                            for text_trivial_aug_maxscale, use_image_aug in zip([0.,], [False, ]):
                            # for text_trivial_aug_maxscale, use_image_aug in zip([0.1,], [False, ]):
                                for fusion_transformer_concat_all_tokens in [False]:
                                    for categorical_convert_to_text_use_header in [False]:
                                        for convert_to_text_numerical in [False]:
                                                for numerical_convert_to_text_use_header in [False]:
                                                    for sequential_fusion in [False]:
                                                        for max_text_len in [512]:
                                                            for auxiliary_weight in [0.]:
                                                            # for auxiliary_weight in [0.1,0.2]:
                                                                for max_epochs in [20]:
                                                                    # for categorical_convert_to_text_use_header_template in ["list", "text", "latex"]:
                                                                    # for categorical_convert_to_text_use_header_template in ["list",]:
                                                                    # for categorical_convert_to_text_use_header_template in ["text"]:
                                                                    for categorical_convert_to_text_use_header_template in ["latex"]:
                                                                        # for no_hf_text_insert_sep in [False]:
                                                                        for LeMDA in [False]:
                                                                            no_hf_text_insert_sep = False
                                                                            LeMDA_arch = "mlp_vae"
                                                                            # LeMDA_arch = "trans_vae"
                                                                            modality_drop_rate = 0.
                                                                            # modality_drop_rate = 0.3
                                                                            
                                                                            alignment_loss = ""
                                                                            # alignment_loss = "KL_feature" 
                                                                            # alignment_loss = "KL"
                                                                            clip_fusion_mlp = False
                                                                            clip_fusion_mlp_quality = 'high'
                                                                            contrastive_loss = ""
                                                                            # contrastive_loss = "contra_logit"
                                                                            # contrastive_loss = "contra_fea"

                                                                            contrastive_loss_w = 1. # default = 0.1
                                                                            use_llama = False
                                                                            lemda_layer = 6
                                                                            no_use_cate_miss_embed = False
                                                                            use_miss_token = False
                                                                            use_miss_token_embed_numerical = True
                                                                            use_miss_token_embed_text = False

                                                                            manifold_mixup = False

                                                                            try:
                                                                                # print(111)
                                                                                weight_decay = weight_decays[idx-1]
                                                                                gradient_clip_val = gradient_clip_vals[idx-1]
                                                                                lr_decay = lr_decays[idx-1]
                                                                                warmup_steps = warmup_stepss[idx-1]
                                                                                lr_schedule = lr_schedules[idx-1]
                                                                                top_k_average_method = top_k_average_methods[idx-1]

                                                                                key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/convert_to_text_{convert_to_text}/ft_transformer_pretrained_{ft_transformer_pretrained}/"
                                                                                if use_fusion_transformer:
                                                                                    key_str += f"use_fusion_transformer_{use_fusion_transformer}/"
                                                                                if early_fusion:
                                                                                    key_str += f"early_fusion_{early_fusion}/"
                                                                                if text_trivial_aug_maxscale > 0.:
                                                                                    key_str += f"text_trivial_aug_maxscale_{text_trivial_aug_maxscale}/"
                                                                                # if use_image_aug is False: ## text+tabular data没有image
                                                                                #     output_dir += f"no_img_aug/"
                                                                                if fusion_transformer_concat_all_tokens and use_fusion_transformer: # 只有在use fusion transformer的情况下使用
                                                                                    key_str += f"fusion_transformer_concat_all_tokens_{fusion_transformer_concat_all_tokens}/"
                                                                                # if convert_to_text and categorical_convert_to_text_use_header: # 只有在convert_to_text的情况下
                                                                                #     key_str += f"categorical_convert_to_text_use_header_{categorical_convert_to_text_use_header}/"
                                                                                if clip_fusion_mlp:
                                                                                    key_str += f"use_clip_fusion_mlp/"
                                                                                if clip_fusion_mlp and clip_fusion_mlp_quality == 'high':
                                                                                    key_str += f"clip_fusion_mlp_quality_high/"
                                                                                if convert_to_text_numerical: # 只有在convert_to_text的情况下
                                                                                    key_str += f"convert_to_text_numerical/"
                                                                                    if numerical_convert_to_text_use_header:
                                                                                        key_str += f"numerical_convert_to_text_use_header_{numerical_convert_to_text_use_header}/"
                                                                                if sequential_fusion:
                                                                                    key_str += f"sequential_fusion_{sequential_fusion}/"
                                                                                if max_text_len not in [512]: # 不是clip 或debertav2
                                                                                    key_str += f"max_text_len_{max_text_len}/"
                                                                                if auxiliary_weight != 0.1: # 0.1是默认的weight
                                                                                    key_str += f"auxiliary_weight_{auxiliary_weight}/"
                                                                                if max_epochs != 10:
                                                                                    key_str += f"max_epochs_{max_epochs}/"
                                                                            
                                                                                if convert_to_text and categorical_convert_to_text_use_header:
                                                                                    key_str += f"categorical_template_{categorical_convert_to_text_use_header_template}/"
                                                                                
                                                                                    key_str += f"no_hf_text_insert_sep_{no_hf_text_insert_sep}/"
                                                                                    # else:
                                                                                if LeMDA:
                                                                                    key_str += "LeMDA/"
                                                                                    if lemda_layer != 4:
                                                                                        key_str += f"lemda_layer_{lemda_layer}/"
                                                                                if modality_drop_rate > 0.:
                                                                                    key_str += f"modality_drop_rate_{modality_drop_rate}/"
                                                                                if alignment_loss != "":
                                                                                    key_str += f"{alignment_loss}_align_loss/"
                                                                                if contrastive_loss != "":
                                                                                    key_str += f"{contrastive_loss}_contra_loss/"
                                                                                if contrastive_loss != "" and contrastive_loss_w != 0.1:
                                                                                    key_str += f"contrastive_loss_w_{contrastive_loss_w}/"
                                                                                if use_fusion_transformer and use_llama:
                                                                                    key_str += f"use_llama_fusion/"
                                                                                
                                                                                if no_use_cate_miss_embed:
                                                                                    key_str += f"no_use_cate_miss_embed_{no_use_cate_miss_embed}/"
                                                                                if use_miss_token:
                                                                                    key_str += f"use_miss_token_{use_miss_token}/"
                                                                                    if use_miss_token_embed_numerical:
                                                                                        key_str += f"use_miss_token_{use_miss_token}_numerical/"
                                                                                    elif use_miss_token_embed_text:
                                                                                        key_str += f"use_miss_token_{use_miss_token}_text/"

                                                                                if manifold_mixup:
                                                                                    key_str += "manifold_mixup/"


                                                                                if seed!=0:
                                                                                    key_str += f"seed_{seed}/"
                                                                                key_str += f"run{run}"
                                                                                # print(key_str)
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

# print()
# for group, dataset_dict in results_dict.items():
#     # print(group)
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
#     # 算平均
#     score_str += str(np.around(np.mean(total_score), 3))
#     print(len(total_score))
#     dataset_str += "Avg."
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


   


