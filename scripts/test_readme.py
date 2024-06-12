import os
from multiprocessing import Pool, current_process, Queue
import time
queue = Queue()
NUM_GPUS = 8
PROC_PER_GPU = 1
BEGIN_GPU=0
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

for gpu_ids in range(BEGIN_GPU, BEGIN_GPU+NUM_GPUS):
    for _ in range(PROC_PER_GPU):
        # if gpu_ids in [0,1,2,3,]: continue
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
# top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
# gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
# weight_decays = [0., 0., 0., 0.001, 0.001, 0.001, 0.001]
# warmup_stepss = [0., 0., 0., 0., 0.1, 0.1, 0.1]
# lr_schedules = ["constant", "constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay"]
# lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

### 第五组
top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
lr_decays = [1., 1., 1., 1., 1., 0.9, 0.9]
weight_decays = [0., 0., 0., 0., 0., 0., 0.0001]

# for run in [1]:
#     for idx in [4,5,6,7]:
#         for dataset in ["imdb", "qaa", "qaq", "book", "prod", "jc", "fake", "salary"]:
#         # for dataset in ["qaa", ]:
#             for convert_to_text in [True]:
#                 weight_decay = weight_decays[idx-1]
#                 gradient_clip_val = gradient_clip_vals[idx-1]
#                 lr_decay = lr_decays[idx-1]
#                 warmup_steps = warmup_stepss[idx-1]
#                 lr_schedule = lr_schedules[idx-1]
#                 top_k_average_method = top_k_average_methods[idx-1]

#                 output_dir = f"ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/convert_to_text_{convert_to_text}/run{run}"
#                 os.makedirs(output_dir, exist_ok=True)
#                 log_file = f"{output_dir}/log.txt"
                
#                 command = f"python src/autogluon/bench/frameworks/multimodal/exec_local.py "  \
#                 f"--params sample_configs/multimodal_local_configs_text_tabular.yaml " \
#                 f"--dataset_name {dataset} " \
#                 f"--custom_dataloader sample_configs/dataloaders/text_tabular_dataloader.py " \
#                 f"--benchmark_dir {output_dir} " \
#                 f"--metrics_dir {output_dir}/results " \
#                 f"--weight_decay {weight_decay} " \
#                 f"--gradient_clip_val {gradient_clip_val} " \
#                 f"--lr_decay {lr_decay} " \
#                 f"--warmup_steps {warmup_steps} " \
#                 f"--lr_schedule {lr_schedule} " \
#                 f"--top_k_average_method {top_k_average_method} " 
                
#                 if convert_to_text == False:
#                     command += f"--categorical_convert_to_text  "
#                 command += f"> {log_file} 2>&1"

#                 process_list.append(command)

top_k_average_methods = ["best", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup", "greedy_soup"]
gradient_clip_vals = [None, None, 1., 1., 1., 1., 1.]
warmup_stepss = [0., 0., 0., 0.1, 0.1, 0.1, 0.1]
lr_schedules = ["constant", "constant", "constant", "constant", "cosine_decay", "cosine_decay", "cosine_decay"]
weight_decays = [0., 0., 0., 0., 0., 0.001, 0.001]
lr_decays = [1., 1., 1., 1., 1., 1., 0.9]

# approve了：fake, airbnb, channel, qaa, qaq, cloth。样本数太多但是approve了，可以subsample作为备选的：wine, kick, jigsaw, 
# 不通过：book, jc, salary, imdb, prod
# 未知：house, mercari
# 目前只需要重新跑一下fake,  qaa, qaq, 
text_only_dataset = ["qaa", "qaq", "salary", "prod", "fake" ]
# for run in [1]:
run = 1
# for seed in [1,2]:
for seed in [0,]:
    # for idx in [1,2,3,4,5,6,7]:
    for idx in [7]:
    # for idx in [2,3,4,5,6,7]:
        # for dataset in ["qaa", "qaq", "book", "jc", "salary", "cloth", "airbnb","channel", "imdb", "prod", "fake"]:
        # for image_lr in [1e-4, 2e-4, 5e-4]:
        # for text_lr in [1e-4, 2e-4, 5e-4]:
        #     for tabular_lr in [1e-4, 2e-4, 5e-4]:
        # for dataset in ["book", "jc", "cloth", "airbnb","channel", "imdb"]:
        
        for convert_to_text in [False]:
            # for ft_transformer_pretrained in [True]:
            for ft_transformer_pretrained in [False]: # early fusion 为True时，ft_transformer_pretrained似乎也用不上
                for use_fusion_transformer in [False]:
                    for early_fusion in [False]:
                        ### 设置成有text aug
                        for text_trivial_aug_maxscale, use_image_aug in zip([0.,], [False, ]):
                        # for text_trivial_aug_maxscale, use_image_aug in zip([0.1,], [False, ]):
                        # for text_trivial_aug_maxscale, use_image_aug in zip([0.1,], [True, ]):
                            for fusion_transformer_concat_all_tokens in [True]:
                                for categorical_convert_to_text_use_header in [False]:
                                    for use_different_lr_for_each_modality in [False]:
                                        for convert_to_text_numerical in [False]:
                                            for numerical_convert_to_text_use_header in [False]:
                                                for sequential_fusion in [False]:
                                                    # for max_text_len in [77, -1]:
                                                    for max_text_len in [512]:
                                                        for auxiliary_weight in [0.]:
                                                        # for auxiliary_weight in [0.1, 0.2]:
                                                            for max_epochs in [20]:
                                                                # for categorical_convert_to_text_use_header_template in ["list", "text", "latex",]:
                                                                # for categorical_convert_to_text_use_header_template in ["latex", ]:
                                                                # for categorical_convert_to_text_use_header_template in ["list", "text",]:
                                                                    for no_hf_text_insert_sep in [False]:
                                                                        for LeMDA in [True]:
                                                                            categorical_convert_to_text_use_header_template = "latex"
                                                                            # LeMDA_arch = "trans_vae"
                                                                            LeMDA_arch = "mlp_vae"
                                                                            modality_drop_rate = 0.
                                                                            # modality_drop_rate = 0.3
                                                                            # modality_drop_rate = 0.4
                                                                            alignment_loss = ""
                                                                            # alignment_loss = "KL_feature"
                                                                            clip_fusion_mlp = False
                                                                            clip_fusion_mlp_quality = 'high'
                                                                            contrastive_loss = ""
                                                                            # contrastive_loss = "contra_logit"
                                                                            # contrastive_loss = "contra_fea"

                                                                            # contrastive_loss_w = 0.001 # default = 0.1
                                                                            contrastive_loss_w = 1.
                                                                            use_llama = False
                                                                            use_llama_7B = False
                                                                            lemda_layer = 6

                                                                            no_use_cate_miss_embed = False
                                                                            use_miss_token = False
                                                                            use_miss_token_embed_numerical = True
                                                                            use_miss_token_embed_text = False

                                                                            manifold_mixup = False

                                                                            modeling_missingness = False
                                                                            
                                                                            # for modeling_missingness_drop_rate in [0.1,0.3,0.5]:
                                                                            modeling_missingness_drop_rate = 0.1

                                                                            # for dataset in ["book", "cloth", "airbnb","channel", "imdb"]:
                                                                                # for dataset in ["fake", "qaa", "qaq", "airbnb","channel", "cloth"]:
                                                                                # for dataset in ["fake", "airbnb", "cloth"]: # categorical有缺失的
                                                                            
                                                                            # for dataset in ["fake", "qaa", "qaq", "airbnb","channel", "cloth"]:
                                                                            # for dataset in ["ptech", "memotion", "food101", "aep","fakeddit", ]:
                                                                            for dataset in ["ccd", "HAM", "wikiart", "cd18","DVM", ]:
                                                                            # for dataset in ["petfinder", "covid", "artm", "seattle", "goodreads", "KARD"]:
                                                                            # for dataset in ["DVM", ]:
                                
                                                                            # for dataset in ["food101"]:
                                                                                output_dir = f"/home/ubuntu/drive2/test_readme/{dataset}"
                                                                                os.system(f"rm -rf {output_dir}")
                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9"
                                                                                

                                                                                command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                    "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                    "--seed 0 "\
                                                                                    f"--benchmark_dir {output_dir} "\
                                                                                    f"--metrics_dir {output_dir}/results "\
                                                                                    f"--dataset_name {dataset} "\
                                                                                    "--top_k_average_method greedy_soup "\
                                                                                    "--gradient_clip_val 1. "\
                                                                                    "--warmup_steps 0.1 "\
                                                                                    "--lr_decay 0.9  "
                                                                                # print("he")
                                                                                # print(command)
                                                                                
                                                                                # print(process_list)
                                                                                # 

                                                                                command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                    "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                    "--seed 0 "\
                                                                                    f"--benchmark_dir {output_dir} "\
                                                                                    f"--metrics_dir {output_dir}/results "\
                                                                                    f"--dataset_name {dataset} "\
                                                                                    "--top_k_average_method greedy_soup "\
                                                                                    "--gradient_clip_val 1. "\
                                                                                    "--warmup_steps 0.1 "\
                                                                                    "--lr_decay 0.9 --use_fusion_transformer "
                                                                                
                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --clip_fusion_mlp "\
                                                                                #     "--clip_high_quality "
                                                                                
                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --sequential_fusion "

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --early_fusion --meta_transformer_ckpt_path Meta-Transformer_large_patch14_encoder.pth "

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --categorical_convert_to_text "\
                                                                                #     " --categorical_convert_to_text_template latex"

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --numerical_convert_to_text  "\

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --alignment_loss positive-only  "\
                                                                                
                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --alignment_loss positive_negative  "\

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --text_trivial_aug_maxscale 0.1 --use_image_aug "

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --manifold_mixup "

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --LeMDA  "

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --modality_drop_rate 0.3 "

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --use_miss_token_embed_numerical --use_miss_token_embed "

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --use_miss_token_embed_image --use_miss_token_embed "

                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 --use_miss_token_embed_image --use_miss_token_embed --modality_drop_rate 0.3 "
                                                                                
#                                                                                 # stacking
#                                                                                 command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
#                                                                                     "--params sample_configs/multimodal_local_configs.yaml " \
#                                                                                     "--seed 0 "\
#                                                                                     f"--benchmark_dir {output_dir} "\
#                                                                                     f"--metrics_dir {output_dir}/results "\
#                                                                                     f"--dataset_name {dataset} "\
#                                                                                     "--top_k_average_method greedy_soup "\
#                                                                                     "--gradient_clip_val 1. "\
#                                                                                     "--warmup_steps 0.1 "\
#                                                                                     "--lr_decay 0.9 " \
#                                                                                      "--use_fusion_transformer "\
#                                                                                     "--categorical_convert_to_text "\
#                                                                                     "--categorical_convert_to_text_use_header "\
#                                                                                     "--categorical_convert_to_text_use_header_template latex "\
#                                                                                     "--LeMDA "\
#                                                                                     "--modality_drop_rate 0.3 "
                                                                                ## For Image+Text Datasets
                                                                                # command = "python src/autogluon/bench/frameworks/multimodal/exec_local.py " \
                                                                                #     "--params sample_configs/multimodal_local_configs.yaml " \
                                                                                #     "--seed 0 "\
                                                                                #     f"--benchmark_dir {output_dir} "\
                                                                                #     f"--metrics_dir {output_dir}/results "\
                                                                                #     f"--dataset_name {dataset} "\
                                                                                #     "--top_k_average_method greedy_soup "\
                                                                                #     "--gradient_clip_val 1. "\
                                                                                #     "--warmup_steps 0.1 "\
                                                                                #     "--lr_decay 0.9 " \
                                                                                #     "--clip_fusion_mlp "\
                                                                                #     "--clip_high_quality "\
                                                                                #     "--alignment_loss all "\
                                                                                #     "--text_trivial_aug_maxscale 0.1 "\
                                                                                #     "--use_image_aug "\
                                                                                #     "--manifold_mixup "\
                                                                                #     "--LeMDA "\
                                                                                #     "--modality_drop_rate 0.3 "
                                                                                process_list.append(command)
                                                                                
print(process_list)

for p in process_list:
    print(p)
    print()
print(len(process_list))
for _ in pool.imap_unordered(distribute, process_list):
    pass
pool.close()
pool.join()

