#!/bin/bash

body_input_dir=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/body
head_input_dir=/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/head
seg_mdl=/NAS5/speech/user/ziyichen/Make-Your-Anchor/inference/res/cp/79999_iter.pth

python face_alignment.py \
    --input_dir $body_input_dir \
    --head_dir $head_input_dir \
    --crop_size 512 || exit 1

# get face mask
python get_mask.py \
    --head_dir $head_input_dir \
    --mdl_path $seg_mdl || exit 1


python prepare_json_head.py \
    --head_dir $head_input_dir
