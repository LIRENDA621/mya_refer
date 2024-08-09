import os
import shutil
import jsonlines
from glob import glob
import random


main_dir = "/NAS5/speech/data/virtual_being/paii/yuxinlu698/010/norm/VID20240717140840_crop_1080_1080_0_350_resize_512x512/clips"
out_main_dir = "./train_data_xinlu3/body"
splits = ["train", "test"]
# test_idx = [f"{i:06}" for i in [10, 20]]
# print(test_idx)
sub_indice = os.listdir(main_dir)
train_idx = sub_indice[:-2]
test_idx = sub_indice[-2:]

split_dict = {"train": train_idx, "test": test_idx}
os.makedirs(out_main_dir, exist_ok=True)
for split in splits:
  out_json = os.path.join(out_main_dir, f"body_{split}.json")
  with jsonlines.open(out_json, mode="w") as json_w:
    sub_dirs = split_dict[split]
    print(sub_dirs)
    for sub_dir in sub_dirs:
      img_sub_dir = os.path.join(main_dir,sub_dir, "images")
      mesh_sub_dir = os.path.join(main_dir,sub_dir, "mesh")

      # os.makedirs(out_img_sub_dir, exist_ok=True)
      # os.makedirs(out_mesh_sub_dir, exist_ok=True)
      img_files = glob(os.path.join(img_sub_dir, "*.png"))
      for img_name in os.listdir(img_sub_dir):
        img_fn = os.path.join(img_sub_dir, img_name)
        mesh_fn = os.path.join(mesh_sub_dir, img_name)
        # img_basename = os.path.basename(img_file)
        # new_img_basename_int = int(img_basename.split(".")[0])+1
        # new_img_basename = f"{new_img_basename_int:06}" 
        ref_name = random.sample(os.listdir(img_sub_dir), 1)[0]
        ref_fn = os.path.join(img_sub_dir, ref_name)
        # ref_basename = os.path.basename(ref_fn[0])
        # new_ref_basename_int = int(ref_basename.split(".")[0])+1
        # new_ref_basename = f"{new_ref_basename_int:06}"

        # out_img_fn =  os.path.join(out_img_sub_dir, f"{new_img_basename}.png")
        # out_mesh_fn = os.path.join(out_mesh_sub_dir, f"{new_img_basename}.png")
        # out_ref_fn = os.path.join(out_img_sub_dir, f"{new_ref_basename}.png")
        if not os.path.isfile(mesh_fn):
          continue
        json_w.write({"source": mesh_fn, "reference":ref_fn , "target": img_fn})