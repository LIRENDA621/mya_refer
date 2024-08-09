### How to transfer poses from one person's meshes to another person

Go to your own SHOW folder, and activate your SHOW environment. e.g.
```
show_dir=/data/user/chenziyi253/paii_virtual_being_3D/third_party/SHOW
cd $show_dir
source show_venv/bin/activate
```
Edit file locations in render_swap.sh
```
pkl_file_path1: pkl file for body shape (Person 1)
pkl_file_path2: pkl file for gestures (Person 2)
out_path: output directory
out_video: output mp4 file path
smplx_model_path: SMPLX model location
```
The default setup will use 
```
camera parameters: person 1
video size: person 1
betas (shape): person 1
global_orient: person2 
body_pose: person 2
body pose & hand pose & jaw pose: person 2
expression: person 1
```

You may modify the setup by change 224-234 lines in render_swap_pkl.py
Since the output needs both pkl, the output length would be the shortest of the two input pkls.


Finally, run render_swap.sh
```
./render_swap.sh
```

