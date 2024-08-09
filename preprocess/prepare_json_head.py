import os
import shutil
import jsonlines
from glob import glob
import random
import argparse
from tqdm import tqdm


def main(head_dir, json_dir):
  splits = ["train", "test"]
  sub_indice = os.listdir(head_dir)
  train_idx = sub_indice[:-2]
  test_idx = sub_indice[-2:]
  for split in splits:
    out_json = os.path.join(json_dir, f"head_{split}.json")
    with jsonlines.open(out_json, mode="w") as json_w:
      # sub_names = os.listdir(head_dir)
      for sub_name in tqdm(sub_indice):
        img_sub_dir = os.path.join(head_dir,  sub_name, "raw_aligned")
        mesh_sub_dir = os.path.join(head_dir,  sub_name, "aligned")
        mask_sub_dir = os.path.join(head_dir,  sub_name, "raw_aligned_mask")
        img_files = glob(os.path.join(img_sub_dir, "*.png"))
        for img_file in img_files:
          img_basename = os.path.basename(img_file)
          ref_file = random.sample(img_files, 1)[0]
          mesh_file = os.path.join(mesh_sub_dir, img_basename)
          mask_file = os.path.join(mask_sub_dir, img_basename)
          json_w.write({"source": mesh_file, "reference":ref_file , "target": img_file, "mask": mask_file})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get face mask from face image.")
    parser.add_argument( "--head_dir", type=str, default=None, required=True)
    parser.add_argument( "--json_dir", type=str, default=None, required=True)

    args = parser.parse_args()
    head_dir = args.head_dir
    json_dir = args.json_dir

    main(head_dir, json_dir)

