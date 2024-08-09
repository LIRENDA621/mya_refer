#!/bin/bash
set -e

export LD_LIBRARY_PATH=
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export CUDA_VISIBLE_DEVICES=0

CONFIG_PATH=$1
stage=$2

if [[ "$#" -lt 2 ]]; then
    echo "Usage: bash run_inference.sh <config_path> <stage>"
    exit 1
fi 

# activate your own conda env here (e.g. controlnet here)
CONDA_DIR="/NAS5/speech/user/zhuyixing276/miniconda3"
ENV_NAME=mya
. "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "$(which python3)"

# run training config to obtain parameters
. $CONFIG_PATH

echo "===================================================="
echo "Input parameters: "
echo "Raw video path: $raw_video_path"
echo "Model user id: $user_id  Model id: $model_id  User affiliation: $affi"
echo "Save root: $save_root"
echo "Body appearance image: $body_prompt_img_pth"
echo "Head_prompt_img_pth: $head_prompt_img_pth"
echo "Body model checkpoint: $body_weight_dir"
echo "Head model checkpoint: $head_weight_dir"
echo "Body pose mesh: $body_input_dir"
echo "Body CFG: $body_cfg"
echo "Body condition scale: $body_condition_scale"
echo "Head CFG: $head_cfg"
echo "Head condition scale: $head_condition_scale"
echo "Max #frames: $max_frame"
echo "===================================================="
echo " "

if [[ $stage -le -2 ]]; then
    echo "Stage -2: Preprocess input video file:"
    if [[ ! -f $raw_video_path ]]; then
        echo "Input video does not exist! $raw_video_path"
        exit 1
    fi 
    
    dest_raw_dir=$mesh_image_dir/raw
    if [[ ! -f "$dest_raw_dir/${video_id}.mp4" ]]; then
        mkdir -p $dest_raw_dir
        cp $raw_video_path $dest_raw_dir/ || exit 1
        echo "Copy raw video file to $dest_raw_dir"
    fi
fi

if [[ $stage -le -1 ]]; then
    echo "Stage -1: Check SHOW pose mesh"
    if [[ ! -d $body_input_dir ]]; then
        echo "Pose mesh is not generated yet!"

        clip_dir_dest=$SHOW_PATH/clips_${video_id}
        mkdir -p $clip_dir_dest
        if [[ -f $SHOW_PATH/$clip_list_fn ]]; then
            rm $SHOW_PATH/$clip_list_fn || exit 1
        fi
        cp $dest_raw_dir/${video_id}.mp4 $clip_dir_dest/
        echo "clips_${video_id}/${video_id}.mp4" > $SHOW_PATH/$clip_list_fn

        python3 ../preprocess/prepare_show_config.py \
            --show_config_path $SHOW_PATH/$show_config_name \
            --video_list_fn $clip_list_fn \
            --show_output_dir $show_output_name \
            --max_frames $max_frames \
            --use_ref_expression $use_ref_expression \
            --use_new_orient $use_new_orient \
            --user_id $user_id \
            --model_id $model_id \
            --affi $affi || exit 1
        
        echo "SHOW config file created at $SHOW_PATH/$show_config_name Please run SHOW first"
        exit 1
    fi
fi

if [[ $stage -le 0 ]]; then
    echo "Stage 0: Reorganize pickles and pose meshes:"
    cp -r $SHOW_PATH/$show_output_name/pickles $mesh_image_dir/ || exit 1

    cp -r $SHOW_PATH/$show_config_name $mesh_image_dir/ || exit 1

    echo "Copy the pose mesh back to output directory:"
    cp -r $SHOW_PATH/$show_output_name/poses/${video_id}/mesh $mesh_image_dir/ || exit 1
fi

if [[ $stage -le 1 ]]; then
    echo "Stage 1: Inference on body:"
    python inference_body.py \
        --unet_path $body_unet_path \
        --controlnet_path $body_controlent_path \
        --input_dir $body_input_dir \
        --save_dir $body_save_dir \
        --prompt_img_pth $body_prompt_img_pth \
        --CFG $body_cfg \
        --condition_scale $body_condition_scale \
        --ws 16 \
        --os 8 || exit 1

    ffmpeg -y -r 30 -i $body_save_dir/%06d.png \
        -q:v 0 -pix_fmt yuv420p $body_save_dir.mp4 || exit 1
    ffmpeg -y -r 30 -i $body_input_dir/%06d.png \
        -q:v 0 -pix_fmt yuv420p ${body_save_dir}_mesh.mp4
    ffmpeg -y -i $body_save_dir.mp4 -i ${body_save_dir}_mesh.mp4 \
        -filter_complex hstack=inputs=2 ${body_save_dir}_comb.mp4
fi

if [[ $stage -le 2 ]]; then
    echo "Stage 2: Run face alignment:"
    python face_alignment.py \
        --imgdir_pth $body_input_dir \
        --raw_imgdir_pth $body_save_dir \
        --results_dir $body_headcrop_root \
        --crop_size 512 || exit 1
fi

if [[ $stage -le 3 ]]; then
    echo "Stage 3: Get face mask:"
    python get_mask.py \
        --input_pth $head_dir \
        --mask_pth $head_mask_dir || exit 1
fi

if [[ $stage -le 4 ]]; then
    echo "Stage 4: inference on face inpainting:"
    python inference_face_inpainting.py \
        --unet_path $head_unet_pth \
        --controlnet_path $head_controlnet_path \
        --input_dir $head_input_dir \
        --face_dir $head_dir \
        --mask_dir $head_mask_dir \
        --save_dir $head_save_dir \
        --prompt_img_pth $head_prompt_img_pth \
        --CFG $head_cfg \
        --condition_scale $head_condition_scale \
        --batch_size 30 || exit 1
fi

if [[ $stage -le 5 ]]; then
    echo "Stage 5: face blending:"
    python face_blending_bgr.py \
        --body_dir $body_save_dir \
        --face_dir $head_save_dir \
        --matrix_dir $head_matrix_dir \
        --save_dir $final_save_dir \
        --crop_size 512 || exit 1

    ffmpeg -y -r 30 -i $final_save_dir/%06d.png \
        -q:v 0 -pix_fmt yuv420p $final_save_dir.mp4 || exit 1
    echo "Final video output saved to $final_save_dir.mp4"
fi

echo "Done! $(date)"
