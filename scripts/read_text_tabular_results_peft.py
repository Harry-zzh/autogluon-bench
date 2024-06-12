import numpy as np
results_dict = {}

### 第二组
top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
weight_decays = [0., 0., 0., 0.001, 0.001, 0.001, 0.001]
warmup_stepss = [0., 0., 0., 0., 0.1, 0.1, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", ]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]
# peft_methods = ["lora", "bit_fit"]
peft_methods = ["lora"]

# for run in [1]:
#     for idx in [1,2,3,4,5,6,7]:
#         for peft_method in peft_methods:
#             for epoch in [10, 20, 30 ,40, 50, 60]:
#                 for lora_r in [8]:
#                 # microsoft/deberta-v3-large
#                 # qaa, qaq, prod, fake, salary: only text?
#                 # for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:        
#                     for lr in [0.001]:
#                         # for convert_to_text in [False]:
#                             for hf_text_ckpt in ["microsoft/deberta-v3-base"]:
#                                 for dataset in ["imdb", "fake", "qaa", "qaq", "book", "jc","prod",  "salary"]:
#                                     try:
#                                         weight_decay = weight_decays[idx-1]
#                                         gradient_clip_val = gradient_clip_vals[idx-1]
#                                         lr_decay = lr_decays[idx-1]
#                                         warmup_steps = warmup_stepss[idx-1]
#                                         lr_schedule = lr_schedules[idx-1]
#                                         top_k_average_method = top_k_average_methods[idx-1]

#                                         key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/peft_method_{peft_method}/epoch_{epoch}/lora_r_{lora_r}/lr_{lr}/run{run}"
#                                         output_dir = f"ag_bench_runs/multimodal/{dataset}/{key_str}"
#                                         metrics_file = f"{output_dir}/results/AutoGluon_stable.{dataset}.None.local/scores/results.csv"
#                                         with open(metrics_file, 'r') as f:
#                                             lines = f.readlines()
#                                         metric_name = lines[0].strip().split(",")[-1]
#                                         score = lines[1].strip().split(",")[-1]

#                                         if key_str not in results_dict:
#                                             results_dict[key_str] = {}
#                                         results_dict[key_str][dataset] = {}
#                                         results_dict[key_str][dataset][metric_name] = score
#                                     except Exception:
#                                         continue            


for run in [1]:
    for idx in [7]:
        for peft_method in peft_methods:
            for epoch in [10, 20, 30]:
                for lora_r in [8, 16]:
                # microsoft/deberta-v3-large
                # qaa, qaq, prod, fake, salary: only text?
                # for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:        
                    for lr in [0.01, 0.001]:
# for run in [1]:
#     for idx in [7]:
#         for peft_method in peft_methods:
#             for epoch in [10]:
#                 for lora_r in [8]:
#                 # microsoft/deberta-v3-large
#                 # qaa, qaq, prod, fake, salary: only text?
#                 # for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:        
#                     for lr in [0.001]:
                        for convert_to_text in [True]:
                            for hf_text_ckpt in ["microsoft/deberta-v3-base"]:

                                for dataset in ["imdb", "fake", "qaa", "qaq", "book", "jc","prod",  "salary"]:

                                    try:
                                        weight_decay = weight_decays[idx-1]
                                        gradient_clip_val = gradient_clip_vals[idx-1]
                                        lr_decay = lr_decays[idx-1]
                                        warmup_steps = warmup_stepss[idx-1]
                                        lr_schedule = lr_schedules[idx-1]
                                        top_k_average_method = top_k_average_methods[idx-1]

                                        key_str = f"top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/peft_method_{peft_method}/epoch_{epoch}/lora_r_{lora_r}/lr_{lr}/convert_to_text_{convert_to_text}/run{run}"
                                        output_dir = f"ag_bench_runs/multimodal/{dataset}/{key_str}"
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
    # 算平均
    score_str += str(np.around(np.mean(total_score), 3))
    dataset_str += "Avg."
    print(dataset_str)
    print(metric_str)
    print(score_str)
    print()