import os
from multiprocessing import Pool, current_process, Queue

queue = Queue()
NUM_GPUS = 8
PROC_PER_GPU = 1
def distribute(process_command):
    gpu_id = queue.get()
    try:
        # run processing on GPU <gpu_id>
        ident = current_process().ident
        print('{}: starting process on GPU {}'.format(ident, gpu_id))
        # ... process filename
        print(f"CUDA_VISIBLE_DEVICES={gpu_id} {process_command}")
        os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} {process_command}")
        print('{}: finished'.format(ident))
    finally:
        queue.put(gpu_id)

for gpu_ids in range(NUM_GPUS):
    for _ in range(PROC_PER_GPU):
        queue.put(gpu_ids)

process_list = []

pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)

#  PEFT.USE_LAYERDROP_ST default FIND_UNUSED_PARAMS True PEFT.LAYER_DROP_RATE 0.5
# for peft_mode in ["adapter", "LoRA", "BitFit"]:
# PEFT.LORA.INTERMEDIATE_TRADITIONAL True 


### 每次有新的trick，记得在exec_local里更新args的定义。

# for run in [1]:
#     for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:
#     # for dataset in ["imdb"]:
#         for weight_decay in [0.]:
#             for gradient_clip_val in [None]:
#                 for lr_decay in [1.]:
#                     for warmup_steps in [0.]:
#                         for lr_schedule in ["constant"]:
#                             for top_k_average_method in ["best"]:
#                                 output_dir = f"ag_bench_runs/multimodal/{dataset}/weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
#                                 process_list.append(f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
#                                 f"--params sample_configs/multimodal_local_configs_text_tabular.yaml " \
#                                 f"--dataset_name {dataset} " \
#                                 f"--custom_dataloader sample_configs/dataloaders/text_tabular_dataloader.py " \
#                                 f"--benchmark_dir {output_dir} " \
#                                 f"--metrics_dir {output_dir}/results " \
#                                 f"--weight_decay {weight_decay} " \
#                                 f"--gradient_clip_val {gradient_clip_val} " \
#                                 f"--lr_decay {lr_decay} " \
#                                 f"--warmup_steps {warmup_steps} " \
#                                 f"--lr_schedule {lr_schedule} " \
#                                 f"--top_k_average_method {top_k_average_method} " \
                                
#                                 )

#### 一组一组叠加，目前一共7组  (最原始的加6组参数的叠加。)
# weight_decays = [0., 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
# top_k_average_methods = ["best", "best", "best", "best", "best", "greedy_soup", "greedy_soup"]
# lr_decays = [1., 1., 1., 1., 1., 1., 0.9]


# for run in [1]:
#     for idx in [2, 3, 4, 5, 6, 7]:
#         for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:
#             weight_decay = weight_decays[idx-1]
#             gradient_clip_val = gradient_clip_vals[idx-1]
#             lr_decay = lr_decays[idx-1]
#             warmup_steps = warmup_stepss[idx-1]
#             lr_schedule = lr_schedules[idx-1]
#             top_k_average_method = top_k_average_methods[idx-1]

#             output_dir = f"ag_bench_runs/multimodal/{dataset}/weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
#             process_list.append(f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
#             f"--params sample_configs/multimodal_local_configs_text_tabular.yaml " \
#             f"--dataset_name {dataset} " \
#             f"--custom_dataloader sample_configs/dataloaders/text_tabular_dataloader.py " \
#             f"--benchmark_dir {output_dir} " \
#             f"--metrics_dir {output_dir}/results " \
#             f"--weight_decay {weight_decay} " \
#             f"--gradient_clip_val {gradient_clip_val} " \
#             f"--lr_decay {lr_decay} " \
#             f"--warmup_steps {warmup_steps} " \
#             f"--lr_schedule {lr_schedule} " \
#             f"--top_k_average_method {top_k_average_method} " \
            
#             )

### 第二组
top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
weight_decays = [0., 0., 0., 0.001, 0.001, 0.001, 0.001]
warmup_stepss = [0., 0., 0., 0., 0.1, 0.1, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", ]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]
# peft_methods = ["lora", "bit_fit"]
peft_methods = ["lora"]

# 需要试是否有layerwise decay
for run in [1]:
    for idx in [7]:
        for peft_method in peft_methods:
            for epoch in [10, 20, 30]:
                for lora_r in [8, 16]:
                # microsoft/deberta-v3-large
                # qaa, qaq, prod, fake, salary: only text?
                # for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:        
                    for lr in [0.01, 0.001]:
                        if epoch == 10 and lr == 0.001 and lora_r == 8:
                            continue
                        for convert_to_text in [True]:
                            for hf_text_ckpt in ["microsoft/deberta-v3-base"]:

                                for dataset in ["imdb", "fake", "qaa", "qaq", "book", "jc","prod",  "salary"]: 
                                    weight_decay = weight_decays[idx-1]
                                    gradient_clip_val = gradient_clip_vals[idx-1]
                                    lr_decay = lr_decays[idx-1]
                                    warmup_steps = warmup_stepss[idx-1]
                                    lr_schedule = lr_schedules[idx-1]
                                    top_k_average_method = top_k_average_methods[idx-1]

                                    # output_dir = f"ag_bench_runs_deberta-v3-large/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/peft_method_{peft_method}/epoch_{epoch}/run{run}"
                                    
                                    output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/peft_method_{peft_method}/epoch_{epoch}/lora_r_{lora_r}/lr_{lr}/convert_to_text_{convert_to_text}/run{run}"
                                    os.makedirs(output_dir, exist_ok=True)
                                    log_file = f"{output_dir}/log.txt"

                                    command = f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
                                    f"--params sample_configs/multimodal_local_configs_text_tabular.yaml " \
                                    f"--dataset_name {dataset} " \
                                    f"--custom_dataloader sample_configs/dataloaders/text_tabular_dataloader.py " \
                                    f"--benchmark_dir {output_dir} " \
                                    f"--metrics_dir {output_dir}/results " \
                                    f"--weight_decay {weight_decay} " \
                                    f"--gradient_clip_val {gradient_clip_val} " \
                                    f"--lr_decay {lr_decay} " \
                                    f"--warmup_steps {warmup_steps} " \
                                    f"--lr_schedule {lr_schedule} " \
                                    f"--top_k_average_method {top_k_average_method} " \
                                    f"--peft {peft_method} " \
                                    f"--max_epochs {epoch} " \
                                    f"--hf_text_ckpt {hf_text_ckpt} " \
                                    f"--lora_r {lora_r} "\
                                    f"--lr {lr} "

                                    

                                    if convert_to_text == False:
                                        command += f"--categorical_convert_to_text  "
                                    command += f"> {log_file} 2>&1"

                                    process_list.append(command)
                                    
                            

print(process_list)
print(len(process_list))
# for _ in pool.imap_unordered(distribute, process_list):
#     pass
# pool.close()
# pool.join()

