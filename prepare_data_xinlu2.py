import os
import shutil
import jsonlines
from glob import glob
import random


img_dir = "/NAS0/speech/data/virtual_being/paii/yuxinlu698/009/norm/VID20240627161747_crop_1080_1080_260_0_512/images"
mesh_dir = "/NAS0/speech/data/virtual_being/paii/yuxinlu698/009/norm/VID20240627161747_crop_1080_1080_260_0_512/meshes"
out_dir = "/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/body"
splits = ["train", "test"]
test_idx = [f"{i:06}" for i in [10, 20]]
print(test_idx)

train_idx = [f"{i:06}" for i in range(140) if f"{i:06}" not in test_idx]
print(train_idx)
split_dict = {"train": train_idx, "test": test_idx}
for split in splits:
  out_json = f"/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu2/body_{split}.json"
  with jsonlines.open(out_json, mode="w") as json_w:
    sub_dirs = split_dict[split]
    for sub_dir in sub_dirs:
      out_img_sub_dir = os.path.join(out_dir,split, sub_dir, "image")
      out_mesh_sub_dir = os.path.join(out_dir,split, sub_dir, "mesh")

      os.makedirs(out_img_sub_dir, exist_ok=True)
      os.makedirs(out_mesh_sub_dir, exist_ok=True)
      img_files = glob(os.path.join(img_dir, sub_dir, "*.png"))
      for img_file in img_files:
        img_basename = os.path.basename(img_file)
        new_img_basename_int = int(img_basename.split(".")[0])+1
        new_img_basename = f"{new_img_basename_int:06}" 
        ref_fn = random.sample(img_files, 1)
        ref_basename = os.path.basename(ref_fn[0])
        new_ref_basename_int = int(ref_basename.split(".")[0])+1
        new_ref_basename = f"{new_ref_basename_int:06}"

        out_img_fn =  os.path.join(out_img_sub_dir, f"{new_img_basename}.png")
        out_mesh_fn = os.path.join(out_mesh_sub_dir, f"{new_img_basename}.png")
        out_ref_fn = os.path.join(out_img_sub_dir, f"{new_ref_basename}.png")
        mesh_fn = os.path.join(mesh_dir, sub_dir, f"{new_img_basename}.png")
        if not os.path.isfile(mesh_fn):
          print(mesh_fn)
          continue
        json_w.write({"source": out_mesh_fn, "reference":out_ref_fn , "target": out_img_fn})

        print(f"{img_file}\n{out_img_fn}")
