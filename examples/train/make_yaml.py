"""
Make yaml file for training DPO, ORPO, SimPO and SSPO.

base SFT model : 
phi-2 : https://huggingface.co/lole25/phi-2-sft-ultrachat-full (set "trust_remote_code" to False in the yaml file.)
mistral : https://huggingface.co/alignment-handbook/zephyr-7b-sft-full
llama-3 : https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT

or, use any other SFT model.

This code is created based on the official code of LLaMA-Factory and the alignment handbook.
(https://github.com/hiyouga/LLaMA-Factory)
(https://github.com/huggingface/alignment-handbook)

(Zheng, Y., Zhang, R., Zhang, J., Ye, Y., & Luo, Z. (2024). 
Llamafactory: Unified efficient fine-tuning of 100+ language models. 
arXiv preprint arXiv:2403.13372.)

"""

import yaml
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--peft", type=str, default="lora", help="full or lora or q-lora")
parser.add_argument("--method", type=str, default="sspo", help="sft, dpo, orpo, simpo, or sspo")
parser.add_argument("--model_path", type=str, default="princeton-nlp/Llama-3-Base-8B-SFT", help="princeton-nlp/Llama-3-Base-8B-SFT or finetuned version(local)")
args = parser.parse_args()

peft = args.peft
method = args.method
model_path = args.model_path

if peft == "full":
    finetuning_type = "full"
else:
    finetuning_type = "lora"

if method == "sft":
    stage = "sft"
else:
    stage = "dpo"

base_config = {
    "model_name_or_path": model_path,
    "trust_remote_code": True,
    "stage": stage,
    "do_train": True,
    "finetuning_type": finetuning_type,
    "template": "default", # please change this to preferred template. Refer to https://github.com/hiyouga/LLaMA-Factory
    "cutoff_len": 1024,
    "max_samples": 10000000,
    "overwrite_cache": True,
    "preprocessing_num_workers": 12,
    "max_grad_norm": 1.0,
    "logging_steps": 20,
    "save_steps": 1000,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "ddp_timeout": 180000000,
    "val_size": 0.1,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "sspo_base": "simpo"
}

# hyperparameters
datasets = ["ultra_combined_fb0.1_ch0.1"]
fb_ratio = 0.1
ch_ratio = 0.1
learning_rates = [1e-6]
num_train_epochs = [10]
lora_ranks = [8]

sspo_gamma_decays = [0.001]
sspo_priors = [0.8]
sspo_gamma_mins = [round(6113/(6113+20786), 4)] # n_L / (n_L + n_U)
sspo_gamma_0s = [1.0]

per_device_train_batch_sizes = [16]
per_device_eval_batch_sizes = [16]
gradient_accumulation_steps = [1]
pref_betas = [2.0]
cutoff_lens = [1024]
simpo_gammas = [0.5]


# generate yaml files
combinations = []
for dataset in datasets:
    for lr in learning_rates:
        for tb in per_device_train_batch_sizes:
            for eb in per_device_eval_batch_sizes:
                for ga in gradient_accumulation_steps:
                    for epochs in num_train_epochs:
                        for rank in lora_ranks:
                            for sspo_gamma_decay in sspo_gamma_decays:
                                for sspo_gamma_0 in sspo_gamma_0s:
                                    for sspo_prior in sspo_priors:
                                        for sspo_gamma_min in sspo_gamma_mins:
                                            for cutoff_len in cutoff_lens:
                                                for beta in pref_betas:
                                                    for simpo_gamma in simpo_gammas:
                                                        combinations.append((dataset, lr, tb, eb, ga, epochs, rank, sspo_gamma_decay, sspo_gamma_0, sspo_prior, sspo_gamma_min, cutoff_len, beta, simpo_gamma))

print(f"We have {len(combinations)} combinations. Copy and paste the following command to the train-sspo-sweep.sh file.")
print("======================")
yaml_dir = f"./examples/train/sspo_llama3/sweep_yamls/{method}_{model_path.split('/')[-1]}/fb{fb_ratio}_ch{ch_ratio}/"
os.makedirs(yaml_dir, exist_ok=True)

for (dataset, lr, tb, eb, ga, epochs, rank, sspo_gamma_decay, sspo_gamma_0, sspo_prior, sspo_gamma_min, cutoff_len, beta, simpo_gamma) in combinations:
    config = base_config.copy()
    
    if peft == "q-lora":
        config.update({
            "quantization_bit": 4,
            "quantization_method": "bitsandbytes",
            "dataset": dataset,
            "learning_rate": lr,
            "num_train_epochs": epochs,
            "lora_rank": rank,
            "lora_target": "all",
            "per_device_train_batch_size": tb,
            "per_device_eval_batch_size": eb,
            "gradient_accumulation_steps": ga,
            "cutoff_len": cutoff_len,
            "output_dir": f"./saves_{model_path.split('/')[-1]}_llama3/fb{fb_ratio}_ch{ch_ratio}/{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}"
        })

        if method != "sft":
            config.update({
                "pref_beta": beta,
                "pref_loss": method,
                "simpo_gamma": simpo_gamma,
            })
        
        if method == "sspo":
            config.update({
                "sspo_gamma_decay": sspo_gamma_decay,
                "sspo_gamma_0": sspo_gamma_0,
                "sspo_gamma_min": sspo_gamma_min,
                "sspo_prior": sspo_prior,
                "simpo_gamma": simpo_gamma,
            })

        filename = f"fb{fb_ratio}_ch{ch_ratio}_{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}.yaml"

    elif peft == "lora":
        config.update({
            "dataset": dataset,
            "learning_rate": lr,
            "num_train_epochs": epochs,
            "lora_rank": rank,
            "lora_target": "all",
            "per_device_train_batch_size": tb,
            "per_device_eval_batch_size": eb,
            "gradient_accumulation_steps": ga,
            "cutoff_len": cutoff_len,
            "output_dir": f"./saves_{model_path.split('/')[-1]}_llama3/fb{fb_ratio}_ch{ch_ratio}/{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}"
        })

        if method != "sft":
            config.update({
                "pref_beta": beta,
                "pref_loss": method,
                "simpo_gamma": simpo_gamma,
            })
        
        if method == "sspo":
            config.update({
                "sspo_gamma_decay": sspo_gamma_decay,
                "sspo_gamma_0": sspo_gamma_0,
                "sspo_gamma_min": sspo_gamma_min,
                "sspo_prior": sspo_prior,
                "simpo_gamma": simpo_gamma,
            })

        filename = f"fb{fb_ratio}_ch{ch_ratio}_{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}.yaml"

    elif peft == "full":
        config.update({
            "dataset": dataset,
            "learning_rate": lr,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": tb,
            "per_device_eval_batch_size": eb,
            "gradient_accumulation_steps": ga,
            "cutoff_len": cutoff_len,
            "output_dir": f"./saves_{model_path.split('/')[-1]}_llama3/fb{fb_ratio}_ch{ch_ratio}/{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_rank{rank}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}"
        })

        if method != "sft":
            config.update({
                "pref_beta": beta,
                "pref_loss": method,
                "simpo_gamma": simpo_gamma,
            })
        
        if method == "sspo":
            config.update({
                "sspo_gamma_decay": sspo_gamma_decay,
                "sspo_gamma_0": sspo_gamma_0,
                "sspo_gamma_min": sspo_gamma_min,
                "sspo_prior": sspo_prior,
                "simpo_gamma": simpo_gamma,
            })

        filename = f"fb{fb_ratio}_ch{ch_ratio}_{peft}_{model_path.split('/')[-1]}_{method}_lr{lr}_beta{beta}_margins{simpo_gamma}_prior{sspo_prior}_gamma_decay{sspo_gamma_decay}_gamma_init{sspo_gamma_0}_gamma_min{sspo_gamma_min}_cutoff{cutoff_len}_ep{epochs}_tb{tb}_eb{eb}_ga{ga}.yaml"

    filepath = os.path.join(yaml_dir, filename)
    
    print("llamafactory-cli train "+filepath)
    
    try:
        with open(filepath, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        print(f"Error occurred while creating file: {e}")

