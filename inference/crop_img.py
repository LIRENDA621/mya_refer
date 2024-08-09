import cv2
import os
from glob import glob

src_dir="/NAS5/speech/user/ziyichen/Make-Your-Anchor/inference/samples/seth_mesh/swap_xinlu_raw"
out_dir="/NAS5/speech/user/ziyichen/Make-Your-Anchor/inference/samples/seth_mesh/swap_xinlu_raw/crop2"
img_fns = glob(os.path.join(src_dir, "*.png"))

os.makedirs(out_dir, exist_ok=True)

for img_fn in img_fns:
  img = cv2.imread(img_fn)
  basename = os.path.basename(img_fn)
  img_crop = img[:, -920:-200,:].copy()
  img_resize = cv2.resize(img_crop, (512,512))
  cv2.imwrite(os.path.join(out_dir, basename), img_resize)