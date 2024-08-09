import argparse
import os


def create_show_config(args):
  """
  Create config file for SHOW inference (see the example file at 
  "paii_virtual_being_3d/third_party/SHOW/inference_config")
  """
  with open(os.path.join(args.show_config_path), "w") as writer:
    writer.write("stage=1\n")
    writer.write(f"video_filelist_path={args.video_list_fn}\n")
    writer.write(f"output_dir={args.show_output_dir}\n")
    writer.write(f"max_frames={args.max_frames}\n")
    writer.write(f"fps=30\n")
    writer.write(f"use_ref_expression={args.use_ref_expression}\n")
    writer.write(f"use_new_orient={args.use_new_orient}\n")
    writer.write(f"user_id={args.user_id}\n")
    writer.write(f"model_id={args.model_id}\n")
    writer.write(f"affi={args.affi}\n")
    writer.write(f"smplx_model_path=../models/smplx/SMPLX_NEUTRAL_2020_org.npz\n")
    writer.write(f"root_data=/data/virtual_being/data\n")
    writer.write(f"pkl_file_ref=$root_data/$affi/$user_id/$model_id/norm/crop_resize/all.pkl\n")
  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Prepare config file for SHOW inference")
  parser.add_argument('--show_config_path', type=str, required=True)
  parser.add_argument('--video_list_fn', type=str, required=True)
  parser.add_argument('--show_output_dir', type=str, required=True)
  parser.add_argument("--max_frames", type=int, default=3600)
  parser.add_argument("--user_id", type=str, required=True)
  parser.add_argument("--model_id", type=str, required=True)
  parser.add_argument("--affi", type=str, default='paii')
  parser.add_argument('--use_ref_expression', type=int, default=0)
  parser.add_argument('--use_new_orient', type=int, default=0)
  args = parser.parse_args()
  print(args, flush=True)
  create_show_config(args)