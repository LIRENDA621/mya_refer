#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_DIR="runwayml/stable-diffusion-v1-5"

# load weights from stage-1
unet_model_name_or_path="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/pre-trained_weight/body/unet"
controlnet_model_name_or_path="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/pre-trained_weight/body/controlnet"

export OUTPUT_DIR="./checkpoints/finetune_xinlu3"
json_file="/NAS5/speech/user/ziyichen/paii_mya/train_data_xinlu3/body/body_train.json"

# for validation
val_img1=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips/000084/mesh/000163.png
val_img2=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips/000084/mesh/000147.png
val_img3=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips/000084/mesh/000288.png
val_img4=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips/000084/mesh/000082.png

reference_img1=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips/000084/images/000001.png
reference_img2=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips/000084/images/000001.png
reference_img3=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips/000084/images/000001.png
reference_img4=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips/000084/images/000001.png

tracker_project_name="ft_body_xinlu3"

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