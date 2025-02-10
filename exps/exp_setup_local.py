import torch, random, os
import numpy as np
from exps.dataset import SCAREDRAWDataset
from exps.options import DefaultOpt
'''
This file is used to set up the environment for the experiments, on my local machine
The setup includes:
    - setting random seeds
    - defining the paths to the dataset and the splits
    - dataset objects
'''
log_path = './logs'
platform = 'local' if os.getcwd().startswith('/mnt') else 'cluster'
if platform == 'local':
    ds_path = '/mnt/c/Users/14152/ZCH/Dev/datasets/SCARED_Images_Resized'
    splits_dir = '/mnt/c/Users/14152/ZCH/Dev/datasets/SCARED_Images_Resized/splits'
else:
    ds_path = '/cluster/project7/Llava_2024/changhao/datasets/SCARED_Images_Resized'
    splits_dir = '/cluster/project7/Llava_2024/changhao/datasets/SCARED_Images_Resized/splits'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
def check_test_only():
    return os.getenv('TEST_ONLY', 'False').lower() == 'true'

def random_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def get_unique_name(name):
    base_name = os.path.join(log_path, name)
    unique_name = base_name
    counter = 1
    while os.path.exists(unique_name):
        unique_name = f"{base_name}_{counter}"
        counter += 1
    return unique_name

random_seeds(233)

"""SCARED DATASET SETUP"""
split_train = os.path.join(ds_path, 'splits/train_files.txt')
split_val = os.path.join(ds_path, 'splits/val_files.txt')
split_test = os.path.join(ds_path, 'splits/test_files.txt')
train_filenames = readlines(split_train)
val_filenames = readlines(split_val)
test_filenames = readlines(split_test)

ds_train = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=train_filenames,
    frame_idxs=DefaultOpt.frame_ids,
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=4,
    is_train=True,
    img_ext='.png'
)

ds_val = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=val_filenames,
    frame_idxs=DefaultOpt.frame_ids,
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=4,
    is_train=False,
    img_ext='.png'
)

ds_test = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=test_filenames,
    frame_idxs=[0],
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=4,
    is_train=False,
    img_ext='.png'
)

ds_test_multi_frame = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=test_filenames,
    frame_idxs=DefaultOpt.frame_ids,
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=4,
    is_train=False,
    img_ext='.png'
)

