import os
import shutil
import jsonlines
from glob import glob
import random


img_dir = "/NAS5/speech/user/lirenda621/Data/VID20240401160505_10s_clip_img"
mesh_dir = "/NAS5/speech/user/lirenda621/Data/VID20240401160505_10s_clip_pose_wMesh"
out_dir = "/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body"
splits = ["train", "test"]

for split in splits:
  out_json = f"/NAS5/speech/user/ziyichen/Make-Your-Anchor/train_data_xinlu/body_{split}.json"
  with jsonlines.open(out_json, mode="w") as json_w:
    sub_dirs = os.listdir(os.path.join(img_dir, split))
    # print(sub_dirs)
    for sub_dir in sub_dirs:
      out_img_sub_dir = os.path.join(out_dir,split, sub_dir, "image")
      out_mesh_sub_dir = os.path.join(out_dir,split, sub_dir, "mesh")
      os.makedirs(out_img_sub_dir, exist_ok=True)
      os.makedirs(out_mesh_sub_dir, exist_ok=True)
      img_files = glob(os.path.join(img_dir, split, sub_dir, "*.png"))
      print(img_files)
      for img_file in img_files:
        img_basename = os.path.basename(img_file)
        ref_fn = random.sample(img_files, 1)
        out_img_fn =  os.path.join(out_img_sub_dir, img_basename)
        out_mesh_fn = os.path.join(out_mesh_sub_dir, img_basename)
        # print(out_img_sub_dir)
        # print(ref_fn)
        out_ref_fn = os.path.join(out_img_sub_dir, os.path.basename(ref_fn[0]))
        mesh_fn = os.path.join(mesh_dir, sub_dir, "mesh_img", img_basename[1:])
        json_w.write({"source": out_mesh_fn, "reference":out_ref_fn , "target": out_img_fn})
        shutil.copy(img_file, out_img_fn)
        shutil.copy(mesh_fn, out_mesh_fn)
        

