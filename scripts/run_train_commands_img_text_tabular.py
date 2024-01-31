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
        time.sleep(100)
        print('{}: finished'.format(ident))
    finally:
        queue.put(gpu_id)

BEGIN_GPU=0
for gpu_ids in range(BEGIN_GPU, BEGIN_GPU+ NUM_GPUS):
    for _ in range(PROC_PER_GPU):
        queue.put(gpu_ids)

process_list = []

pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)

#  PEFT.USE_LAYERDROP_ST default FIND_UNUSED_PARAMS True PEFT.LAYER_DROP_RATE 0.5
# for peft_mode in ["adapter", "LoRA", "BitFit"]:
# PEFT.LORA.INTERMEDIATE_TRADITIONAL True 


### 每次有新的trick，记得在exec_local里更新args的定义。



## img_text_tabular 按照大小来选择，当然也有按照拥有的列数量来选择。
# grocery_image
# coco_caption_subset
# avito_demand_prediction_subset

# for run in [1]:
#     # 0 4 5 6
#     # for dataset in ["persuasive_techniques", "yelp", "seattle_airbnb", "san_francisco_airbnb", "hateful_memes", "petfinder", "mmimdb", "grocery_image"]:
#     for dataset in ["yelp"]:
#         for weight_decay in [0.]:
#             for gradient_clip_val in [None]:
#                 for lr_decay in [1.]:
#                     for warmup_steps in [0.]:
#                         for lr_schedule in ["constant"]:
#                             for top_k_average_method in ["best"]:
#                                 output_dir = f"ag_bench_runs/multimodal/{dataset}/weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
#                                 process_list.append(f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
#                                 f"--params sample_configs/multimodal_local_configs_img_text_tabular.yaml " \
#                                 f"--dataset_name {dataset} " \
#                                 f"--custom_dataloader sample_configs/dataloaders/text_tabular_img_dataloader.py " \
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
#     for idx in [7, 1, 2, 3, 4, 5, 6]:
#     # for idx in [7]:
#         '''
#         grocery_image: 15h / 10 epoch
#         hateful_memes: 2h
#         mmimdb: 6h
#         persuasive_techniques: 0.87h 
#         petfinder: 3.4h
#         san_francisco_airbnb: 2.4h 结果有点问题
#         seattle_airbnb: 1.8h 
#         yelp: 0.99h
        
#         '''
#         # for dataset in ["persuasive_techniques", "yelp", "seattle_airbnb",  "hateful_memes", "mmimdb", "grocery_image","petfinder", "san_francisco_airbnb",]:
#         for dataset in ["san_francisco_airbnb"]:
#             weight_decay = weight_decays[idx-1]
#             gradient_clip_val = gradient_clip_vals[idx-1]
#             lr_decay = lr_decays[idx-1]
#             warmup_steps = warmup_stepss[idx-1]
#             lr_schedule = lr_schedules[idx-1]
#             top_k_average_method = top_k_average_methods[idx-1]

#             output_dir = f"ag_bench_runs/multimodal/{dataset}/weight_decay_{weight_decay}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/top_k_average_method_{top_k_average_method}/lr_decay_{lr_decay}/run{run}"
#             process_list.append(f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
#             f"--params sample_configs/multimodal_local_configs_img_text_tabular.yaml " \
#             f"--dataset_name {dataset} " \
#             f"--custom_dataloader sample_configs/dataloaders/text_tabular_img_dataloader.py " \
#             f"--benchmark_dir {output_dir} " \
#             f"--metrics_dir {output_dir}/results " \
#             f"--weight_decay {weight_decay} " \
#             f"--gradient_clip_val {gradient_clip_val} " \
#             f"--lr_decay {lr_decay} " \
#             f"--warmup_steps {warmup_steps} " \
#             f"--lr_schedule {lr_schedule} " \
#             f"--top_k_average_method {top_k_average_method} " \
            
#             )
            

### 第三组
# top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
# weight_decays = [0., 0., 0., 0., 0., 0.001, 0.001]
# lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

# weight decay调参, lr decay一直为1
# top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
# # weight_decays = [0., 0., 0., 0., 0., 0.0001, 0.01]
# weight_decays = [0., 0., 0., 0., 0., 0., 0.]
# lr_decays = [1., 1., 1., 1., 1., 1., 0.9]


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
price_datasets = ["seattle_airbnb", "goodreads", "crypto-coven", "CD18", "DVM-CAR", "stylish-product", "art-price-dataset"]
image_tabular_datasets = [
    "skin_cancer", "Harumanis_mango", "CD18",
    "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "goodreads", "stylish-product", "KARD", 
                          ] # 这里指的是cate列不会被convert to text的dataset。
#"crypto-coven", "nike", "amazon-books-reviews","art-price-dataset" 效果比较差，需要去掉一些列再看。

for run in [1]:
    # for idx in [1, 2, 3, 4, 5, 6]:
    for idx in [7]:
        ''' 
        跑完全部所用时间：
        grocery_image: 15h / 10 epoch 复现不了精度
        hateful_memes: 2h
        mmimdb: 6h 复现不了精度。
        persuasive_techniques: 0.87h 
        petfinder: 3.4h
        san_francisco_airbnb: 2.4h 结果有点问题
        seattle_airbnb: 1.8h 
        yelp: 0.99h
        
        '''
        # for dataset in ["persuasive_techniques", "yelp", "seattle_airbnb",  "hateful_memes", "mmimdb", "grocery_image","petfinder", "san_francisco_airbnb",]:
        # for dataset in ["persuasive_techniques", "yelp", "hateful_memes", "petfinder", "seattle_airbnb",  ]:
        # for dataset in ["san_francisco_airbnb"]:
        # for dataset in ["persuasive_techniques"]:
        # for dataset in ["gas"]:
        # for dataset in ["skin_cancer"]:
        # for dataset in ["goodreads"]:
        # for dataset in ["crypto-coven"]:
        # for dataset in ["nike"]:
        # for dataset in ["CD18"]:
        # for dataset in ["mocheg"]:
        # for dataset in ["wikiart"]:
        # for dataset in ["DVM-CAR"]:
        # for dataset in ["Memotion"]:
        # for dataset in ["UPMC-Food101"]:
        # for dataset in ["iqa"]:
        # for dataset in ["snli-ve"]:
        # for dataset in ["stylish-product"]:
        # for dataset in ["amazon-books-reviews"]:
        # for dataset in ["KARD"]:
        # for dataset in ["fashion-dataset-img-only"]:
        # for dataset in ["covid-chestxray-dataset"]:
        # for dataset in ["FALL-UP"]:
        # for dataset in ["Fishpond"]:
        # for dataset in ["Harumanis_mango"]:
        # for dataset in ["CCD"]:
        # for dataset in ["price_lookup_codes"]: 
        # for dataset in ["perfume"]:
        # for dataset in ["action_effect_entailment"]:
        # for dataset in ["action_effect_pred"]:
        # for dataset in ["impressions_dataset"]:
        # for dataset in ["art-price-dataset"]:
        # 
        # for dataset in ["hateful_memes", "persuasive_techniques", "yelp", "mocheg", "CCD", "action_effect_entailment"]:
        # price need convert_to_log=True: c
        # and need to try load pre-trained model
        # for dataset in ["seattle_airbnb", "goodreads", "crypto-coven", "nike", "CD18", "DVM-CAR", "stylish-product", "amazon-books-reviews", "KARD", "art-price-dataset", "impressions_dataset"]:
        # for dataset in ["seattle_airbnb", "goodreads", "crypto-coven", "CD18", "DVM-CAR", "stylish-product", "art-price-dataset"]:
        # for dataset in [
        #     # "hateful_memes", "persuasive_techniques",  "mocheg", "Memotion", "UPMC-Food101","snli-ve", "action_effect_pred", "impressions_dataset",  
        #                 #  "yelp", "CCD", "action_effect_entailment", "skin_cancer", "wikiart", "iqa", "Harumanis_mango", "CD18", "DVM-CAR",  
        #                 # "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "goodreads", "crypto-coven", "nike", "amazon-books-reviews", "stylish-product", "KARD", "art-price-dataset"
        #                 ]:
        for dataset in image_tabular_datasets:
            for ft_transformer_pretrained in [True]:
                weight_decay = weight_decays[idx-1]
                gradient_clip_val = gradient_clip_vals[idx-1]
                lr_decay = lr_decays[idx-1]
                warmup_steps = warmup_stepss[idx-1]
                lr_schedule = lr_schedules[idx-1]
                top_k_average_method = top_k_average_methods[idx-1]
                
                if convert_to_log and dataset in price_datasets:
                    ori_dataset = dataset
                    dataset = dataset + "_convert_to_log"
                    print(dataset)
                if ft_transformer_pretrained:
                    output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/ft_transformer_pretrained_{ft_transformer_pretrained}/run{run}"
                else:
                    output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/run{run}"
                log_file = f"{output_dir}/log.txt"
                os.makedirs(output_dir, exist_ok=True)

                # if convert_to_log:
                #     dataset = ori_dataset

                command = f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
                f"--params sample_configs/multimodal_local_configs_img_text_tabular.yaml " \
                f"--dataset_name {dataset} " \
                f"--custom_dataloader sample_configs/dataloaders/text_tabular_img_dataloader.py " \
                f"--benchmark_dir {output_dir} " \
                f"--metrics_dir {output_dir}/results " \
                f"--weight_decay {weight_decay} " \
                f"--gradient_clip_val {gradient_clip_val} " \
                f"--lr_decay {lr_decay} " \
                f"--warmup_steps {warmup_steps} " \
                f"--lr_schedule {lr_schedule} " \
                f"--top_k_average_method {top_k_average_method} " \


                if ft_transformer_pretrained:
                    command += f"--ft_transformer_ckpt_name  "
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

