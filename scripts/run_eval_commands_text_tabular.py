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
output_dir = ""
# for run in [1]:
#     for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:
#         process_list.append(f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
#                             f"--params sample_configs/multimodal_local_configs_text_tabular.yaml " \
#                             f"--dataset_name {dataset} " \
#                             f"--custom_dataloader sample_configs/dataloaders/text_tabular_dataloader.py " \
#                             f"--eval_model_path ag_bench_runs/multimodal/{dataset}/run{run}/models " \
#                             f"--metrics_dir ag_bench_runs/multimodal/{dataset}/run{run}/results ")
        
for run in [1]:
    for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:
    # for dataset in ["imdb"]:
        for weight_decay in [0.]:
            for gradient_clip_val in [None]:
                for lr_decay in [1.]:
                    for warmup_steps in [0.]:
                        for lr_schedule in ["constant"]:
                            for top_k_average_method in ["best"]:
                                output_dir = f"ag_bench_runs/multimodal/{dataset}/weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
                                process_list.append(f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
                                f"--params sample_configs/multimodal_local_configs_text_tabular.yaml " \
                                f"--dataset_name {dataset} " \
                                f"--custom_dataloader sample_configs/dataloaders/text_tabular_dataloader.py " \
                                f"--eval_model_path {output_dir}/models " \
                                f"--metrics_dir {output_dir}/results " \
                                f"--weight_decay {weight_decay} " \
                                f"--gradient_clip_val {gradient_clip_val} " \
                                f"--lr_decay {lr_decay} " \
                                f"--warmup_steps {warmup_steps} " \
                                f"--lr_schedule {lr_schedule} " \
                                f"--top_k_average_method {top_k_average_method} " \
                                
                                )

print(process_list)
print(len(process_list))
# for _ in pool.imap_unordered(distribute, process_list):
#     pass
# pool.close()
# pool.join()

