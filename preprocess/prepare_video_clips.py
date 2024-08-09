import os
import argparse
import subprocess
import cv2
import json
import glob
from tqdm import tqdm


def video2frames(video_fn, img_dir):
  """
  Save video frames to img_dir
  """
  # save cropped video's frames
  os.makedirs(img_dir, exist_ok=True)
  cmd_line = f"ffmpeg -i {video_fn} -start_number 0 {img_dir}/%06d.png"
  subprocess.run(cmd_line, shell=True)
  return 


def main(args):
  """
  Given a raw video mp4 file, prepare input data for Make-Your-Anchor (MYA) 
  model. The input data include: 
    1) Cropped video clips (mp4) as inputs to SHOW, to extract 3D mesh.
    2) Cropped images for each video clip (png format).

  The output resolution is (512, 512) by default.
  """
  os.makedirs(args.output_dir, exist_ok=True)
  resize_res = args.resize_res

  # crop the raw video to (args.crop_res, args.crop_res) at ref_coord
  try:
    ref = eval(args.ref_coord)
  except Exception as e:
    print(str(e))
    print("Error reading ref_coord. Set it to (0,0) instead.")
    ref = (0,0)

  # 1) crop video
  crop_name = "crop"
  crop_video_fn = f"{crop_name}.mp4"
  crop_video_fn = os.path.join(args.output_dir, crop_video_fn)
  cmd_line = f"ffmpeg -i {args.raw_video_path} -vf " \
             f"\"crop={args.crop_res}:{args.crop_res}:{ref[0]}:{ref[1]}\" " \
             f"-c:a copy {crop_video_fn}"
  result1 = subprocess.run(cmd_line, shell=True)
  # Save cropped video's frames
  crop_img_dir = os.path.join(args.output_dir, crop_name, "images")
  video2frames(crop_video_fn, crop_img_dir)
  # Save cropped video clips
  crop_clip_dir = os.path.join(args.output_dir, crop_name, "clips")
  create_video_clips(crop_img_dir, crop_clip_dir, res=args.crop_res, 
                     chunk_size=args.chunk_size)
  
  # 2) crop video --> resized video
  resize_name = f"{crop_name}_resize"
  resize_video_fn = f"{resize_name}.mp4"
  resize_video_fn = os.path.join(args.output_dir, resize_video_fn)
  cmd_line = f"ffmpeg -i {crop_video_fn} -vf scale={resize_res}:{resize_res} "\
             f"{resize_video_fn}"
  result2 = subprocess.run(cmd_line, shell=True)
  # save resized video frames as resized images
  resize_img_dir = os.path.join(args.output_dir, resize_name, "images")
  video2frames(resize_video_fn, resize_img_dir)
  
  # generate video clips and corresponding image directories
  resize_clip_dir = os.path.join(args.output_dir, resize_name, "clips")
  create_video_clips(resize_img_dir, resize_clip_dir, res=args.resize_res, 
                     chunk_size=args.chunk_size)

  with open(os.path.join(args.output_dir, "info.json"), "w") as f:
    output_dict = vars(args)
    output_dict["crop_clip_dir"] = crop_clip_dir
    output_dict["resize_clip_dir"] = resize_clip_dir
    json.dump(vars(args), f, ensure_ascii=False, indent=2)

  return


def create_video_clips(img_dir, clip_dir, res, chunk_size):
  """
  Given video frames in img_dir, create video clips with specified length.
  """
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  num_frame = len(glob.glob(os.path.join(img_dir, "*.png")))
  for i in tqdm(range(num_frame)):
    img = cv2.imread(os.path.join(img_dir, f"{i:06d}.png"))
    h, w, _ = img.shape
    if h != res or w != res:
      img = cv2.resize(img, (res, res))

    out_idx = i % chunk_size + 1
    chunk_num = i // chunk_size
    curr_clip_dir = os.path.join(clip_dir, f"{chunk_num:06d}")
    os.makedirs(os.path.join(curr_clip_dir, "images"), exist_ok=True)

    if out_idx == 1:
      if chunk_num > 0:
        out_video.release()
      out_video = cv2.VideoWriter(os.path.join(curr_clip_dir, f"{chunk_num:06d}.mp4"), 
                                  fourcc, args.fps, (res, res))
    out_video.write(img)
    cv2.imwrite(os.path.join(curr_clip_dir, "images", f"{out_idx:06d}.png"), 
                img)
  out_video.release()
  return clip_dir


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Prepare")
  parser.add_argument('--raw_video_path', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  parser.add_argument("--ref_coord", type=str, default="(0,0)", 
                      help="Top-left corner coords in the raw video for cropping.")
  parser.add_argument("--resize_res", type=int, default=512)
  parser.add_argument("--crop_res", type=int, default=1080)
  parser.add_argument("--fps", type=float, default=30.0)
  parser.add_argument("--chunk_size", type=int, default=900)
  args = parser.parse_args()
  print(args, flush=True)
  main(args)