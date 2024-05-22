import os
from multiprocessing import Pool, current_process, Queue
import time

queue = Queue()
NUM_GPUS = 8
PROC_PER_GPU = 2
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
        # if gpu_ids in [0,1,]: continue
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
# price_datasets = ["seattle_airbnb", "goodreads", "crypto-coven", "CD18", "DVM-CAR", "stylish-product", "art-price-dataset", "nike"]
price_datasets = ["seattle_airbnb", "goodreads", "crypto-coven", "CD18", "DVM-CAR", "stylish-product", "nike"]
image_tabular_datasets = [
    "skin_cancer", "Harumanis_mango", "CD18",
    "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "goodreads", "stylish-product", "KARD", 
                          ] # 这里指的是cate列不会被convert to text的dataset。
#"crypto-coven", "nike", "amazon-books-reviews","art-price-dataset" 效果比较差，需要去掉一些列再看。

# 目前已通过： "UPMC-Food101", "fakeddit" 
# 目前未通过："mocheg", "snli-ve",
# 未知："hateful_memes", "persuasive_techniques", "Memotion", 
# 目前看不懂对方是啥意思的："action_effect_pred", 

run = 1
for seed in [1,2]:
    for idx in [1, 2, 3, 4, 5, 6, 7]:
    # for idx in [7]:
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
      
        # for image_lr in [1e-4, 8e-5, 5e-5]:
        #     for text_lr in [1e-4, 8e-5, 5e-5]:
                # for tabular_lr in [1e-4]: # 没有tabular
        for dataset in [ # 去除impressions,  "crypto-coven", "nike", "amazon-books-reviews", 
            # "persuasive_techniques", "Memotion", "UPMC-Food101","snli-ve", "action_effect_pred", "fakeddit",  #"hateful_memes", "mocheg",
           "persuasive_techniques",  "Memotion", "UPMC-Food101","action_effect_pred", "fakeddit"
        # "persuasive_techniques",  
            # "hateful_memes", "persuasive_techniques",  "mocheg", "Memotion", "UPMC-Food101","snli-ve", "action_effect_pred", "fakeddit" 
                        #  "yelp", "CCD", "action_effect_entailment", "skin_cancer", "wikiart", "iqa", "Harumanis_mango", "CD18", "DVM-CAR",  
                        # "petfinder", "covid-chestxray-dataset", "seattle_airbnb", "goodreads","stylish-product", "KARD", "art-price-dataset"
                        ]:
            for convert_to_text in [False]: # 因为只跑image+text, ft_transformer_pretrained是否为false不影响
                for ft_transformer_pretrained in [False]:
                    # for text_trivial_aug_maxscale in [0.1]:
                    #     for use_image_aug in [True]:
                    for text_trivial_aug_maxscale, use_image_aug in zip([0.,], [False, ]):
                    # for text_trivial_aug_maxscale, use_image_aug in zip([0.1,], [True, ]):
                        for use_fusion_transformer in [False]:
                            for early_fusion in [False]:
                                for epoch_num in [20]: ### 改变weight decay和lr decay看看效果。
                                    # if epoch_num == 10 and dataset != "fakeddit":
                                    #     continue
                                    # if idx in [1,2,3,4] and epoch_num != 10:
                                    #     continue
                                    for clip_fusion_mlp in [False]:
                                        for fusion_transformer_concat_all_tokens in [False]:
                                            for use_different_lr_for_each_modality in [False]:
                                                for clip_fusion_mlp_quality in ['high']:     
                                                    for sequential_fusion in [False]:
                                                        for auxiliary_weight in [0.]:
                                                        # for auxiliary_weight in [0.1, 0.2]:
                                                            for LeMDA in [False]:
                                                                # LeMDA_arch = "trans_vae"
                                                                # modality_drop_rate = 0.
                                                                LeMDA_arch = "mlp_vae"
                                                                modality_drop_rate = 0.
                                                                alignment_loss = ""
                                                                # alignment_loss = "KL"
                                                                # alignment_loss = "KL_feature"
                                                                contrastive_loss = ""
                                                                # contrastive_loss = "contra_logit"
                                                                # contrastive_loss = "contra_fea"

                                                                # contrastive_loss_w = 0.001 # default = 0.1
                                                                contrastive_loss_w = 1.


                                                                weight_decay = weight_decays[idx-1]
                                                                lr_decay = lr_decays[idx-1]
                                                                # for weight_decay in [0.0001, 0.01]: ### 这里改变了weight_decay和lr_decay!!
                                                                #     for lr_decay in [1., 0.9]:
                                                            
                                                                gradient_clip_val = gradient_clip_vals[idx-1]
                                                                warmup_steps = warmup_stepss[idx-1]
                                                                lr_schedule = lr_schedules[idx-1]
                                                                top_k_average_method = top_k_average_methods[idx-1]

                                                                # if not use_different_lr_for_each_modality and image_lr != 1e-4 and text_lr != 1e-4 :
                                                                #     continue
                                                                
                                                            
                                                                # # 不要全相等
                                                                # if use_different_lr_for_each_modality and (image_lr == text_lr):
                                                                #     continue
                                                            
                                                                
                                                                if convert_to_log and dataset in price_datasets:
                                                                    ori_dataset = dataset
                                                                    dataset = dataset + "_convert_to_log"
                                                                    print(dataset)
                                                                output_dir = f"/home/ubuntu/drive2/ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/"
                                                                

                                                                if convert_to_text is False:
                                                                    output_dir += f"convert_to_text_{convert_to_text}/"
                                                                if ft_transformer_pretrained:
                                                                    output_dir += f"ft_transformer_pretrained_{ft_transformer_pretrained}/"
                                                                if text_trivial_aug_maxscale > 0.:
                                                                    output_dir += f"text_trivial_aug_maxscale_{text_trivial_aug_maxscale}/"
                                                                if use_image_aug is False:
                                                                    output_dir += f"no_img_aug/"
                                                                if use_fusion_transformer:
                                                                    output_dir += f"use_fusion_transformer_{use_fusion_transformer}/"
                                                                if early_fusion:
                                                                    output_dir += f"early_fusion_{early_fusion}/"
                                                                if epoch_num != 10:
                                                                    output_dir += f"epoch_{epoch_num}/"
                                                                if clip_fusion_mlp:
                                                                    output_dir += f"use_clip_fusion_mlp/"
                                                                if fusion_transformer_concat_all_tokens and use_fusion_transformer: # 只有在use fusion transformer的情况下使用
                                                                    output_dir += f"fusion_transformer_concat_all_tokens_{fusion_transformer_concat_all_tokens}/"
                                                                if use_different_lr_for_each_modality:
                                                                    output_dir += f"use_different_lr_for_each_modality_imagelr{image_lr}_textlr{text_lr}/"
                                                                if clip_fusion_mlp and clip_fusion_mlp_quality == 'high':
                                                                    output_dir += f"clip_fusion_mlp_quality_high/"
                                                                if sequential_fusion:
                                                                    output_dir += f"sequential_fusion/"
                                                                if auxiliary_weight != 0.1: # 0.1是默认的weight
                                                                    output_dir += f"auxiliary_weight_{auxiliary_weight}/"
                                                                if LeMDA:
                                                                    output_dir += "LeMDA/"
                                                                    if LeMDA_arch != "mlp_vae":
                                                                        output_dir += f"LeMDA_arch_{LeMDA_arch}/"

                                                                if modality_drop_rate > 0.:
                                                                    output_dir += f"modality_drop_rate_{modality_drop_rate}/"
                                                                if alignment_loss != "":
                                                                    output_dir += f"{alignment_loss}_align_loss/"
                                                                if contrastive_loss != "":
                                                                    output_dir += f"{contrastive_loss}_contra_loss/"
                                                                if contrastive_loss != "" and contrastive_loss_w != 0.1:
                                                                    output_dir += f"contrastive_loss_w_{contrastive_loss_w}/"
                                                                if seed!=0:
                                                                    output_dir += f"seed_{seed}/"
                                                                                
                                                                                
                                                                output_dir += f"run{run}"
                                                                
                                                                # if ft_transformer_pretrained and convert_to_text is False:
                                                                #     output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/convert_to_text_{convert_to_text}/ft_transformer_pretrained_{ft_transformer_pretrained}/run{run}"
                                                                # elif ft_transformer_pretrained:
                                                                #     output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/ft_transformer_pretrained_{ft_transformer_pretrained}/run{run}"
                                                                # elif convert_to_text is False:
                                                                #     output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/convert_to_text_{convert_to_text}/run{run}"
                                                                # else:
                                                                #     output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/weight_decay_{weight_decay}/lr_decay_{lr_decay}/run{run}"
                                                                log_file = f"{output_dir}/log.txt"
                                                                # os.system(f"rm -rf {output_dir}")
                                                                os.makedirs(output_dir, exist_ok=True)

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
                                                                if convert_to_text == False:
                                                                    command += f"--categorical_convert_to_text  "
                                                                if text_trivial_aug_maxscale > 0.:
                                                                    command += f"--text_trivial_aug_maxscale {text_trivial_aug_maxscale} "
                                                                if use_image_aug is False:
                                                                    command += "--use_image_aug "
                                                                if use_fusion_transformer:
                                                                    command += f"--use_fusion_transformer "
                                                                if early_fusion:
                                                                    command += f"--early_fusion "
                                                                if epoch_num != 10:
                                                                    command += f"--max_epochs {epoch_num} "
                                                                if clip_fusion_mlp:
                                                                    command += f"--clip_fusion_mlp "
                                                                if fusion_transformer_concat_all_tokens and use_fusion_transformer:
                                                                    command += f"--fusion_transformer_concat_all_tokens "
                                                                if use_different_lr_for_each_modality:
                                                                    command += f"--use_different_lr_for_each_modality "
                                                                    command += f"--image_lr {image_lr} --text_lr {text_lr}  "
                                                                if clip_fusion_mlp and clip_fusion_mlp_quality == 'high':
                                                                    command += f"--clip_high_quality "
                                                                if sequential_fusion:
                                                                    command += f"--sequential_fusion "
                                                                if auxiliary_weight != 0.1: 
                                                                    command += f" --auxiliary_weight {auxiliary_weight} "
                                                                if LeMDA:
                                                                    command += f" --LeMDA "
                                                                    if LeMDA_arch != "mlp_vae":
                                                                        command += f"--LeMDA_arch {LeMDA_arch} "
                                                                if modality_drop_rate > 0.:
                                                                    command += f"--modality_drop_rate {modality_drop_rate} "
                                                                if alignment_loss != "":
                                                                    command += f"--alignment_loss {alignment_loss} "
                                                                if contrastive_loss != "":
                                                                    command += f"--contrastive_loss {contrastive_loss} "
                                                                if contrastive_loss != "" and contrastive_loss_w != 0.1:
                                                                    command += f"--contrastive_loss_w {contrastive_loss_w} "
                                                                if seed != 0:
                                                                    command += f"--seed {seed} "


                                                                command += f"> {log_file} 2>&1"
                                                                process_list.append(command)


# process_list.append(f"python train_net_sam_peft.py --config-file configs/sam_peft_{dataset}.yaml --num-gpus 4  --dist-url tcp://127.0.0.1:{port} OUTPUT_DIR checkpoints/sam_{dataset}_{peft_mode}_scale16_train_run1 SOLVER.IMS_PER_BATCH 4 WANDB.NAME sam_{dataset}_{peft_mode}_scale16_train_run1 SEED 18874641 PEFT.MODE {peft_mode}")

# process_list.append(f"python train_net_sam_peft.py --config-file configs/sam_peft_{dataset}.yaml --num-gpus 4  --dist-url tcp://127.0.0.1:{port} OUTPUT_DIR checkpoints/sam_{dataset}_{peft_mode}_r1_train_run2 SOLVER.IMS_PER_BATCH 4 WANDB.NAME sam_{dataset}_{peft_mode}_r1_train_run2 SEED 42686693 PEFT.MODE {peft_mode}")
# process_list.append(f"python train_net_sam_peft.py --config-file configs/sam_peft_{dataset}.yaml --num-gpus 4  --dist-url tcp://127.0.0.1:{port} OUTPUT_DIR checkpoints/sam_{dataset}_{peft_mode}_train_run3 SOLVER.IMS_PER_BATCH 4 WANDB.NAME sam_{dataset}_{peft_mode}_train_run3 SEED 37649630 PEFT.MODE {peft_mode}")
print(process_list)
print(len(process_list))
# for p in process_list:
#     print(p)
for _ in pool.imap_unordered(distribute, process_list):
    pass
pool.close()
pool.join()

