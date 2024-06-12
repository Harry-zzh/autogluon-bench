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


# for dataset in ["fake", "qaa", "qaq", "airbnb","channel", "cloth"]:
for dataset in ["persuasive_techniques",  "Memotion", "UPMC-Food101","action_effect_pred", "fakeddit"]:
    for seed in [0,1,2]:
        use_avg = True
        if use_avg:
            output_dir = f"ensemble_results/{dataset}/seed_{seed}/use_avg"
            process_list.append(f"python src/autogluon/bench/frameworks/multimodal/exec_local.py --params sample_configs/multimodal_local_configs_img_text_tabular.yaml --dataset_name {dataset} "  \
                            f"--custom_dataloader sample_configs/dataloaders/text_tabular_img_dataloader.py --use_ensemble --use_avg --seed {seed} > {output_dir}/log.txt 2>&1 " \
                            )
        else:
            output_dir = f"ensemble_results/{dataset}/seed_{seed}"
            process_list.append(f"python src/autogluon/bench/frameworks/multimodal/exec_local.py --params sample_configs/multimodal_local_configs_img_text_tabular.yaml --dataset_name {dataset} "  \
                            f"--custom_dataloader sample_configs/dataloaders/text_tabular_img_dataloader.py --use_ensemble --seed {seed} > {output_dir}/log.txt 2>&1 " \
                            )
        os.makedirs(output_dir, exist_ok=True)

print(process_list)
print(len(process_list))
for _ in pool.imap_unordered(distribute, process_list):
    pass
pool.close()
pool.join()

