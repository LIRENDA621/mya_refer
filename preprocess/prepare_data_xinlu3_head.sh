#!/bin/bash

input_dir=/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips
seg_mdl=/NAS5/speech/user/ziyichen/Make-Your-Anchor/inference/res/cp/79999_iter.pth
json_dir=/NAS5/speech/user/ziyichen/paii_mya/train_data_xinlu3/head

mkdir -p $json_dir
# python face_alignment.py \
#     --input_dir $input_dir \
#     --head_dir $input_dir \
#     --crop_size 512 || exit 1

# # get face mask
# python get_mask.py \
#     --head_dir $input_dir \
#     --mdl_path $seg_mdl || exit 1


python prepare_json_head.py \
    --head_dir $input_dir \
    --json_dir $json_dir
