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
        if gpu_ids in [0,1,2,3,]: continue
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
                                                                        for LeMDA in [False]:
                                                                            categorical_convert_to_text_use_header_template = "latex"
                                                                            # LeMDA_arch = "trans_vae"
                                                                            LeMDA_arch = "mlp_vae"
                                                                            modality_drop_rate = 0.
                                                                            modality_drop_rate = 0.3
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

                                                                            modeling_missingness = True
                                                                            
                                                                            for modeling_missingness_drop_rate in [0.1,0.3,0.5]:


                                                                            # for dataset in ["book", "cloth", "airbnb","channel", "imdb"]:
                                                                                # for dataset in ["fake", "qaa", "qaq", "airbnb","channel", "cloth"]:
                                                                                # for dataset in ["fake", "airbnb", "cloth"]: # categorical有缺失的
                                                                                for dataset in ["channel", "qaa", "qaq"]:
                                                                                # for dataset in ["cloth", ]:
                                                                                # for dataset in ["airbnb", "channel", "cloth"]: # 有numerical
                                                                                    weight_decay = weight_decays[idx-1]
                                                                                    gradient_clip_val = gradient_clip_vals[idx-1]
                                                                                    lr_decay = lr_decays[idx-1]
                                                                                    warmup_steps = warmup_stepss[idx-1]
                                                                                    lr_schedule = lr_schedules[idx-1]
                                                                                    top_k_average_method = top_k_average_methods[idx-1]

                                                                                    # if not use_different_lr_for_each_modality and text_lr != 1e-4 and tabular_lr != 1e-4:
                                                                                    #     continue
                                                                                    
                                                                                    
                                                                                    # # 不要全相等
                                                                                    # if use_different_lr_for_each_modality and (text_lr == tabular_lr):
                                                                                    #     continue
                                                                                        

                                                                                    output_dir = f"/home/ubuntu/drive2/ag_bench_runs/multimodal/{dataset}/top_k_average_method_{top_k_average_method}/gradient_clip_val_{gradient_clip_val}/weight_decay_{weight_decay}/warmup_steps_{warmup_steps}/lr_schedule_{lr_schedule}/lr_decay_{lr_decay}/convert_to_text_{convert_to_text}/ft_transformer_pretrained_{ft_transformer_pretrained}/"
                                                                                    if use_fusion_transformer:
                                                                                        output_dir += f"use_fusion_transformer_{use_fusion_transformer}/"
                                                                                    if early_fusion:
                                                                                        output_dir += f"early_fusion_{early_fusion}/"
                                                                                    if text_trivial_aug_maxscale > 0.:
                                                                                        output_dir += f"text_trivial_aug_maxscale_{text_trivial_aug_maxscale}/"
                                                                                    # if use_image_aug is False: ## text+tabular data没有image
                                                                                    #     output_dir += f"no_img_aug/"
                                                                                    if fusion_transformer_concat_all_tokens and use_fusion_transformer: # 只有在use fusion transformer的情况下使用
                                                                                        output_dir += f"fusion_transformer_concat_all_tokens_{fusion_transformer_concat_all_tokens}/"
                                                                                    # if convert_to_text and categorical_convert_to_text_use_header: # 只有在convert_to_text的情况下
                                                                                    #     output_dir += f"categorical_convert_to_text_use_header_{categorical_convert_to_text_use_header}/"
                                                                                    if use_different_lr_for_each_modality:
                                                                                        output_dir += f"use_different_lr_for_each_modality_textlr{text_lr}_tabularlr{tabular_lr}/"
                                                                                    
                                                                                    if clip_fusion_mlp:
                                                                                        output_dir += f"use_clip_fusion_mlp/"
                                                                                    if clip_fusion_mlp and clip_fusion_mlp_quality == 'high':
                                                                                        output_dir += f"clip_fusion_mlp_quality_high/"
                                                        
                                                                                    if convert_to_text_numerical: # 只有在convert_to_text的情况下
                                                                                        output_dir += f"convert_to_text_numerical/"
                                                                                        if numerical_convert_to_text_use_header:
                                                                                            output_dir += f"numerical_convert_to_text_use_header_{numerical_convert_to_text_use_header}/"
                                                                                    if sequential_fusion:
                                                                                        output_dir += f"sequential_fusion_{sequential_fusion}/"
                                                                                    # if max_text_len not in [77, 512]: # 不是clip 或debertav2
                                                                                    if max_text_len not in [512]: # 不是clip 或debertav2
                                                                                        output_dir += f"max_text_len_{max_text_len}/"
                                                                                    if auxiliary_weight != 0.1: # 0.1是默认的weight
                                                                                        output_dir += f"auxiliary_weight_{auxiliary_weight}/"
                                                                                    if max_epochs != 10:
                                                                                        output_dir += f"max_epochs_{max_epochs}/"
                                                                                    if convert_to_text and categorical_convert_to_text_use_header:
                                                                                        output_dir += f"categorical_template_{categorical_convert_to_text_use_header_template}/"
                                                                                    
                                                                                        output_dir += f"no_hf_text_insert_sep_{no_hf_text_insert_sep}/"
                                                                                        # else:
                                                                                    if LeMDA:
                                                                                        output_dir += f"LeMDA/"
                                                                                        if LeMDA_arch != "mlp_vae":
                                                                                            output_dir += f"LeMDA_arch_{LeMDA_arch}/"
                                                                                        if lemda_layer != 4:
                                                                                            output_dir += f"lemda_layer_{lemda_layer}/"
                                                                                    if modality_drop_rate > 0.:
                                                                                        output_dir += f"modality_drop_rate_{modality_drop_rate}/"
                                                                                    if alignment_loss != "":
                                                                                        output_dir += f"{alignment_loss}_align_loss/"
                                                                                    if contrastive_loss != "":
                                                                                        output_dir += f"{contrastive_loss}_contra_loss/"
                                                                                    if contrastive_loss != "" and contrastive_loss_w != 0.1:
                                                                                        output_dir += f"contrastive_loss_w_{contrastive_loss_w}/"
                                                                                    if use_fusion_transformer and use_llama:
                                                                                        output_dir += f"use_llama_fusion/"
                                                                                    if use_fusion_transformer and use_llama_7B:
                                                                                        output_dir += f"use_llama7B_fusion/"
                                                                                    if no_use_cate_miss_embed:
                                                                                        output_dir += f"no_use_cate_miss_embed_{no_use_cate_miss_embed}/"
                                                                                    if use_miss_token:
                                                                                        output_dir += f"use_miss_token_{use_miss_token}/"
                                                                                        if use_miss_token_embed_numerical:
                                                                                            output_dir += f"use_miss_token_{use_miss_token}_numerical/"
                                                                                        elif use_miss_token_embed_text:
                                                                                            output_dir += f"use_miss_token_{use_miss_token}_text/"
                                                                                    if manifold_mixup:
                                                                                        output_dir += "manifold_mixup/"
                                                                                    if modeling_missingness:
                                                                                        output_dir += f"modeling_missingness_drop_rate_{modeling_missingness_drop_rate}/"
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    if seed!=0: # seed放在路径最后，方便算平均
                                                                                        output_dir += f"seed_{seed}/"
                                                                                    output_dir += f"run{run}"
                                                                                    # os.system(f"rm -rf {output_dir}")
                                                                                    os.makedirs(output_dir, exist_ok=True)
                                                                                    os.makedirs(os.path.join(output_dir,"models"), exist_ok=True)

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
                                                                                    
                                                                                    if convert_to_text == False:
                                                                                        command += f"--categorical_convert_to_text  "
                                                                                    if ft_transformer_pretrained:
                                                                                        command += f"--ft_transformer_ckpt_name  "
                                                                                    if use_fusion_transformer:
                                                                                        command += f"--use_fusion_transformer "
                                                                                    if early_fusion:
                                                                                        command += f"--early_fusion "
                                                                                    if text_trivial_aug_maxscale > 0.:
                                                                                        command += f"--text_trivial_aug_maxscale {text_trivial_aug_maxscale} "
                                                                                    if fusion_transformer_concat_all_tokens and use_fusion_transformer:
                                                                                        command += f"--fusion_transformer_concat_all_tokens "
                                                                                    if categorical_convert_to_text_use_header:
                                                                                        command += f"--categorical_convert_to_text_use_header "
                                                                                    if use_different_lr_for_each_modality:
                                                                                        command += f"--use_different_lr_for_each_modality "
                                                                                        command += f" --text_lr {text_lr} --tabular_lr {tabular_lr} "
                                                                                    if convert_to_text_numerical: # 只有在convert_to_text的情况下
                                                                                        command += f" --numerical_convert_to_text "
                                                                                        
                                                                                        if numerical_convert_to_text_use_header:
                                                                                            command += f" --numerical_convert_to_text_use_header "
                                                                                    if sequential_fusion:
                                                                                        command += f" --sequential_fusion "
                                                                                    if max_text_len not in [512]: 
                                                                                        command += f" --max_text_len {max_text_len} "
                                                                                    if auxiliary_weight != 0.1: 
                                                                                        command += f" --auxiliary_weight {auxiliary_weight} "
                                                                                    if max_epochs != 10:
                                                                                        command += f" --max_epochs {max_epochs} "
                                                                                    if convert_to_text and categorical_convert_to_text_use_header:
                                                                                        command += f"--categorical_convert_to_text_use_header_template {categorical_convert_to_text_use_header_template} "
                                                                                        # if categorical_convert_to_text_use_header_template == "latex":
                                                                                        if no_hf_text_insert_sep:
                                                                                            command += "--no_hf_text_insert_sep "
                                                                                    if LeMDA:
                                                                                        command += "--LeMDA "
                                                                                        if LeMDA_arch != "mlp_vae":
                                                                                            command += f"--LeMDA_arch {LeMDA_arch} "
                                                                                        if lemda_layer != 4:
                                                                                            command += f"--LeMDA_layer {lemda_layer} " 

                                                                                    if modality_drop_rate > 0.:
                                                                                        command += f"--modality_drop_rate {modality_drop_rate} "
                                                                                    if alignment_loss != "":
                                                                                        command += f"--alignment_loss {alignment_loss} "
                                                                                    if clip_fusion_mlp:
                                                                                        command += f"--clip_fusion_mlp "
                                                                                    if clip_fusion_mlp and clip_fusion_mlp_quality == 'high':
                                                                                        command += f"--clip_high_quality "
                                                                                    if contrastive_loss != "":
                                                                                        command += f"--contrastive_loss {contrastive_loss} "
                                                                                    if contrastive_loss != "" and contrastive_loss_w != 0.1:
                                                                                        command += f"--contrastive_loss_w {contrastive_loss_w} "
                                                                                    if seed != 0:
                                                                                        command += f"--seed {seed} "
                                                                                    if use_fusion_transformer and use_llama:
                                                                                        command += f"--use_llama "
                                                                                    if use_fusion_transformer and use_llama_7B:
                                                                                        command += f"--use_llama_7B "
                                                                                    if no_use_cate_miss_embed:
                                                                                        command += "--no_use_cate_miss_embed "
                                                                                    if use_miss_token:
                                                                                        command += f"--use_miss_token_embed "
                                                                                        if use_miss_token_embed_numerical:
                                                                                            command += f"--use_miss_token_embed_numerical "
                                                                                        elif use_miss_token_embed_text:
                                                                                            command += f"--use_miss_token_embed_text "
                                                                                    if manifold_mixup:
                                                                                        command += "--manifold_mixup "
                                                                                    if modeling_missingness:
                                                                                        command += f"--modeling_missingness --modeling_missingness_drop_rate {modeling_missingness_drop_rate} "
                                                                                    
                                                                    


                                                                                    command += f"> {log_file} 2>&1"

                                                                                    process_list.append(command)

print(process_list)

for p in process_list:
    print(p)
    print()
print(len(process_list))
# for _ in pool.imap_unordered(distribute, process_list):
#     pass
# pool.close()
# pool.join()

