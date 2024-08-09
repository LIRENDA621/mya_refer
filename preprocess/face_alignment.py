import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from insightface_func.face_detect_crop_single import Face_detect_crop
from glob import glob

import argparse

def align(img, M, crop_size):
    align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
    return align_img

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def imgdir_align(input_dir, detect_model, head_dir='./temp_results', crop_size=224):   
    # split_sets = ['train', 'test'] 
    # for split_set in split_sets:
        sub_sets = os.listdir((input_dir))
        for sub_set in sub_sets:
            out_sub_dir = os.path.join(input_dir, sub_set)

            os.makedirs(out_sub_dir, exist_ok=True)
            os.makedirs(os.path.join(out_sub_dir, 'aligned'), exist_ok=True)
            os.makedirs(os.path.join(out_sub_dir, 'raw_aligned'), exist_ok=True)
            os.makedirs(os.path.join(out_sub_dir, 'matrix'), exist_ok=True)
            for img_name in os.listdir(os.path.join(input_dir, sub_set, 'images')):
                raw_img_pth = os.path.join(input_dir, sub_set, 'images', img_name)
                mesh_pth = os.path.join(input_dir,  sub_set, 'mesh', img_name)
                mesh_img = cv2.imread(mesh_pth)
                raw_img = cv2.imread(raw_img_pth)
                print(mesh_pth)
                detect_results = detect_model.get(mesh_img,crop_size)
                if detect_results is not None:
                    mesh_align_crop_list = detect_results[0]
                    mesh_mat_list = detect_results[1]
                    for mesh_align_crop in mesh_align_crop_list:
                        cv2.imwrite(os.path.join(out_sub_dir, 'aligned', f"{img_name}"), mesh_align_crop)
                        raw_img_align_crop = align(raw_img, mesh_mat_list[0], crop_size)
                        cv2.imwrite(os.path.join(out_sub_dir, 'raw_aligned', f'{img_name}'), raw_img_align_crop)
                        np.save(os.path.join(out_sub_dir, 'matrix', f'{img_name[:-4]}.npy'), mesh_mat_list[0])
                        break
                else:
                    print('not detected in {}'.format(raw_img))
                    if not os.path.exists(out_sub_dir):
                        os.mkdir(out_sub_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_dir", type=str, default=None, required=True )
    parser.add_argument( "--head_dir", type=str, default=None, required=True )
    parser.add_argument( "--crop_size", type=int, default=512)

    args = parser.parse_args()


    input_dir = args.input_dir

    crop_size=args.crop_size
    if crop_size == 512:
        mode = 'ffhq'
    else:
        mode = 'None'
    
    app = Face_detect_crop(name='antelopev2', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    head_dir = args.head_dir
    
    imgdir_align(input_dir, app, crop_size=crop_size, head_dir=head_dir)