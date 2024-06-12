import os
from multiprocessing import Pool, current_process, Queue
import time

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
        time.sleep(200)
        print('{}: finished'.format(ident))
    finally:
        queue.put(gpu_id)

BEGIN_GPU=0
for gpu_ids in range(BEGIN_GPU, BEGIN_GPU+ NUM_GPUS):
    for _ in range(PROC_PER_GPU):
        # if gpu_ids in [0, 2, 4, 5, 6, 7]:
        queue.put(gpu_ids)

process_list = []

pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)

### 第六组 tune warmup steps
top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.2, 0.3, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
weight_decays = [0., 0., 0., 0., 0., 0., 0.]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]


### 第七组 不使用cosine decay和weight decay
top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.3, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay"]
weight_decays = [0., 0., 0., 0., 0., 0., 0.]
lr_decays = [1., 1., 1., 1., 0.9, 0.9, 0.9]


top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
weight_decays = [0., 0., 0., 0., 0., 0.001, 0.001]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

convert_to_log = True # 通过baseline的结果，以后跑price数据集都用convert_to_log
# price_datasets = ["seattle_airbnb", "goodreads", "crypto-coven", "CD18", "DVM-CAR", "stylish-product", "art-price-dataset", "nike"]
price_datasets = ["seattle_airbnb", "goodreads", "crypto-coven", "CD18", "DVM-CAR", "stylish-product", "nike"]
image_tabular_datasets = [
    "skin_cancer", "Harumanis_mango", "CD18",
    "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "goodreads", "stylish-product", "KARD", 
                          ] # 这里指的是cate列不会被convert to text的dataset。
#"crypto-coven", "nike", "amazon-books-reviews","art-price-dataset" 效果比较差，需要去掉一些列再看。


for dataset in [ # 去除impressions,  "crypto-coven", "nike", "amazon-books-reviews", 
     "yelp", "CCD", "action_effect_entailment", "skin_cancer", "wikiart", "iqa", "CD18", "DVM-CAR"
                ]:

        if convert_to_log and dataset in price_datasets:
            ori_dataset = dataset
            dataset = dataset + "_convert_to_log"

        output_dir = f"ag_bench_runs_dataset_info/{dataset}"
        log_file = f"{output_dir}/log.txt"
        # os.system(f"rm -rf {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        command = "python3 src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                "--params sample_configs/multimodal_local_configs_img_text_tabular.yaml " \
                "--custom_dataloader sample_configs/dataloaders/text_tabular_img_dataloader.py " \
                "--weight_decay 0.01 --gradient_clip_val 1.0 --lr_decay 0.9 --warmup_steps 0.1 " \
                "--lr_schedule cosine_decay --top_k_average_method greedy_soup --use_image_aug " \
                "--max_epochs 20 --get_dataset_info  "\
                f"--dataset_name {dataset} "
        command += f"> {log_file} 2>&1"
        process_list.append(command)
                                                    

                                                

# process_list.append(f"python train_net_sam_peft.py --config-file configs/sam_peft_{dataset}.yaml --num-gpus 4  --dist-url tcp://127.0.0.1:{port} OUTPUT_DIR checkpoints/sam_{dataset}_{peft_mode}_scale16_train_run1 SOLVER.IMS_PER_BATCH 4 WANDB.NAME sam_{dataset}_{peft_mode}_scale16_train_run1 SEED 18874641 PEFT.MODE {peft_mode}")

# process_list.append(f"python train_net_sam_peft.py --config-file configs/sam_peft_{dataset}.yaml --num-gpus 4  --dist-url tcp://127.0.0.1:{port} OUTPUT_DIR checkpoints/sam_{dataset}_{peft_mode}_r1_train_run2 SOLVER.IMS_PER_BATCH 4 WANDB.NAME sam_{dataset}_{peft_mode}_r1_train_run2 SEED 42686693 PEFT.MODE {peft_mode}")
# process_list.append(f"python train_net_sam_peft.py --config-file configs/sam_peft_{dataset}.yaml --num-gpus 4  --dist-url tcp://127.0.0.1:{port} OUTPUT_DIR checkpoints/sam_{dataset}_{peft_mode}_train_run3 SOLVER.IMS_PER_BATCH 4 WANDB.NAME sam_{dataset}_{peft_mode}_train_run3 SEED 37649630 PEFT.MODE {peft_mode}")
print(process_list)
print(len(process_list))
for _ in pool.imap_unordered(distribute, process_list):
    pass
pool.close()
pool.join()

