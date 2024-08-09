#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,3,4
export MODEL_DIR="runwayml/stable-diffusion-v1-5"

# load weights from stage-1
unet_model_name_or_path="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/pre-trained_weight/body/unet"
controlnet_model_name_or_path="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/pre-trained_weight/body/controlnet"

export OUTPUT_DIR="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/finetune_xinlu512"
json_file=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body_train.json

# for validation
val_img1=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body/test/00006/mesh/000163.png
val_img2=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body/test/00020/mesh/000147.png
val_img3=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body/test/00047/mesh/000288.png
val_img4=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body/test/00096/mesh/000082.png

reference_img1=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body/test/00006/mesh/000001.png
reference_img2=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body/test/00020/mesh/000001.png
reference_img3=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body/test/00047/mesh/000001.png
reference_img4=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body/test/00096/mesh/000001.png

tracker_project_name="ft_body_xinlu512"

accelerate launch --main_process_port 65534 ./train/train_body512.py \
 --resume_from_checkpoint latest \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --unet_model_name_or_path $unet_model_name_or_path \
 --controlnet_model_name_or_path $controlnet_model_name_or_path \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=json \
 --dataset_config_name $json_file \
 --image_column target \
 --conditioning_image_column source \
 --caption_column reference \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image $val_img1 $val_img2 $val_img3 $val_img4 \
 --reference_image $reference_img1 $reference_img2 $reference_img3 $reference_img4 \
 --train_batch_size 4 \
 --enable_xformers_memory_efficient_attention \
 --tracker_project_name $tracker_project_name \
 --checkpointing_steps 5000 \
 --validation_steps 1000 \
 --num_train_epochs 60