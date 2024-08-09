cd train

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_DIR="runwayml/stable-diffusion-inpainting"
export OMP_NUM_THREADS=1

# load weights from stage-1
unet_model_name_or_path="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/pre-trained_weight/head/unet"
controlnet_model_name_or_path="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/pre-trained_weight/head/controlnet"

export OUTPUT_DIR="/NAS5/speech/user/ziyichen/Make-Your-Anchor/checkpoints/finetune_xinlu2/head"
json_file=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head_train.json

# for validation
val_img1=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/aligned/000010.png
val_img2=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/aligned/000030.png
val_img3=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/aligned/000100.png
val_img4=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/aligned/000200.png

val_mask1=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/raw_aligned_mask/000010.png
val_mask2=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/raw_aligned_mask/000030.png
val_mask3=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/raw_aligned_mask/000100.png
val_mask4=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/raw_aligned_mask/000200.png

reference_img=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head/test/000010/raw_aligned/000001.png

tracker_project_name=ft_head_xinlu2
 
accelerate launch --main_process_port 65535 train_head.py \
 --resume_from_checkpoint latest \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$controlnet_model_name_or_path \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=json \
 --dataset_config_name $json_file \
 --image_column target \
 --conditioning_image_column source \
 --caption_column reference \
 --resolution=256 \
 --learning_rate=1e-5 \
 --validation_image $val_img1 $val_img2 $val_img3 $val_img4 \
 --validation_mask $val_mask1 $val_mask2 $val_mask3 $val_mask4 \
 --reference_image $reference_img $reference_img $reference_img $reference_img \
 --train_batch_size=4 \
 --enable_xformers_memory_efficient_attention \
 --tracker_project_name $tracker_project_name \
 --checkpointing_steps 10000 \
 --validation_steps 1000 \
 --num_train_epochs 60