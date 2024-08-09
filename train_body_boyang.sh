#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_DIR="runwayml/stable-diffusion-v1-5"

# load weights from stage-1
unet_model_name_or_path="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/pre-trained_weight/body/unet"
controlnet_model_name_or_path="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/pre-trained_weight/body/controlnet"

export OUTPUT_DIR="./checkpoints/finetune_boyang_tmp"
json_file="train_data_boyang/body_train.json"

# for validation
val_img1=/NAS5/speech/data/virtual_being/paii/boyang/001/norm/pad_512/clips/000082/mesh/000062.png
val_img2=/NAS5/speech/data/virtual_being/paii/boyang/001/norm/pad_512/clips/000082/mesh/000001.png
val_img3=/NAS5/speech/data/virtual_being/paii/boyang/001/norm/pad_512/clips/000082/mesh/000242.png
val_img4=/NAS5/speech/data/virtual_being/paii/boyang/001/norm/pad_512/clips/000083/mesh/000204.png

reference_img1=/NAS5/speech/data/virtual_being/paii/boyang/001/norm/pad_512/clips/000082/images/000157.png
reference_img2=/NAS5/speech/data/virtual_being/paii/boyang/001/norm/pad_512/clips/000082/images/000108.png
reference_img3=/NAS5/speech/data/virtual_being/paii/boyang/001/norm/pad_512/clips/000082/images/000108.png
reference_img4=/NAS5/speech/data/virtual_being/paii/boyang/001/norm/pad_512/clips/000083/images/000168.png

tracker_project_name="ft_body_boyang"

accelerate launch --main_process_port 65534 ./train/train_body512.py \
 --resume_from_checkpoint latest \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --unet_model_name_or_path $unet_model_name_or_path \
 --controlnet_model_name_or_path $controlnet_model_name_or_path \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=json \
 --dataset_config_name=$json_file \
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