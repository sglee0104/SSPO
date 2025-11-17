#!/bin/bash

# """
# Train SSPO with llama3.
# Paste the commands to this file.

# This code is created based on the official code of LLaMA-Factory and the alignment handbook.
# (https://github.com/hiyouga/LLaMA-Factory)
# (https://github.com/huggingface/alignment-handbook)

# (Zheng, Y., Zhang, R., Zhang, J., Ye, Y., & Luo, Z. (2024). 
# Llamafactory: Unified efficient fine-tuning of 100+ language models. 
# arXiv preprint arXiv:2403.13372.)

# """

# Set PYTHONPATH
export PYTHONPATH="./src_sspo:$PYTHONPATH" # for SSPO
# export PYTHONPATH="./src/llamafactory/src:$PYTHONPATH" # for the others

# Use these if required
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export FORCE_TORCHRUN=1
export DISABLE_VERSION_CHECK=1

# Set GPU devices (modify GPU numbers as needed)
# export CUDA_VISIBLE_DEVICES="NUMBERS_OF_GPU_DEVICES"

export WANDB_PROJECT="dpo-sft-mistral-7b-it-fb0.1_ch0.1"
export WANDB_NAME="fb0.1_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr1e-05_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.2273_basesimpo_cutoff1024_ep1_tb4_eb4_ga8"
export WANDB_API_KEY="bab17ee869da570e4aa032aeabf295c64fb590ca"

# Run training : llamafactory-cli train "yaml file path"
# Copy and paste the commands from running the make_yaml.py file to this file.
# You can run multiple commands at once.

# llamafactory-cli train "YOUR_YAML_FILE_PATH"

llamafactory-cli train ./examples/train/mistral-7b-it/dpo_sft/fb0.1_ch0.1/fb0.1_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr1e-05_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.2273_basesimpo_cutoff1024_ep1_tb4_eb4_ga8.yaml
llamafactory-cli train ./examples/train/mistral-7b-it/dpo_sft/fb0.1_ch0.1/fb0.1_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr5e-06_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.2273_basesimpo_cutoff1024_ep1_tb4_eb4_ga8.yaml
llamafactory-cli train ./examples/train/mistral-7b-it/dpo_sft/fb0.1_ch0.1/fb0.1_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr1e-06_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.2273_basesimpo_cutoff1024_ep1_tb4_eb4_ga8.yaml
llamafactory-cli train ./examples/train/mistral-7b-it/dpo_sft/fb0.1_ch0.1/fb0.1_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr5e-07_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.2273_basesimpo_cutoff1024_ep1_tb4_eb4_ga8.yaml

export WANDB_PROJECT="dpo-sft-mistral-7b-it-fb0.01_ch0.1"
export WANDB_NAME="fb0.01_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr1e-05_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.0286_basesimpo_cutoff1024_ep1_tb4_eb4_ga8"
export WANDB_API_KEY="bab17ee869da570e4aa032aeabf295c64fb590ca"

llamafactory-cli train ./examples/train/mistral-7b-it/dpo_sft/fb0.01_ch0.1/fb0.01_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr1e-05_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.0286_basesimpo_cutoff1024_ep1_tb4_eb4_ga8.yaml
llamafactory-cli train ./examples/train/mistral-7b-it/dpo_sft/fb0.01_ch0.1/fb0.01_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr5e-06_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.0286_basesimpo_cutoff1024_ep1_tb4_eb4_ga8.yaml
llamafactory-cli train ./examples/train/mistral-7b-it/dpo_sft/fb0.01_ch0.1/fb0.01_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr1e-06_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.0286_basesimpo_cutoff1024_ep1_tb4_eb4_ga8.yaml
llamafactory-cli train ./examples/train/mistral-7b-it/dpo_sft/fb0.01_ch0.1/fb0.01_ch0.1_lora_Mistral-7B-Instruct-v0.2_dpo_sft_lr5e-07_rank8_beta0.1_margins2.0_prior0.5_gamma_decay0.001_gamma_init1.0_gamma_min0.0286_basesimpo_cutoff1024_ep1_tb4_eb4_ga8.yaml