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
ds_base = '/mnt/c/Users/14152/ZCH/Dev/datasets' if platform == 'local' else '/cluster/project7/Llava_2024/changhao/datasets'

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
ds_path = os.path.join(ds_base, 'SCARED_Images_Resized')
splits_dir = os.path.join(ds_base, 'SCARED_Images_Resized', 'splits')
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

if platform == 'cluster':
    '''C3VD DATASET SETUP'''
    ds_path = os.path.join(ds_base, 'C3VD_as_SCARED')
    splits_dir = os.path.join(ds_base, 'C3VD_as_SCARED', 'splits')
    split_train = os.path.join(ds_path, 'splits/train_files.txt')
    split_test = os.path.join(ds_path, 'splits/test_files.txt')
    train_filenames = readlines(split_train)
    test_filenames = readlines(split_test)

    ds_train_c3vd = SCAREDRAWDataset(
        data_path=ds_path,
        filenames=train_filenames,
        frame_idxs=DefaultOpt.frame_ids,
        height=DefaultOpt.height,
        width=DefaultOpt.width,
        num_scales=4,
        is_train=True,
        img_ext='.png'
    )

    split_test = SCAREDRAWDataset(
        data_path=ds_path,
        filenames=split_test,
        frame_idxs=[0],
        height=DefaultOpt.height,
        width=DefaultOpt.width,
        num_scales=4,
        is_train=False,
        img_ext='.png'
    )

    '''hamlyn dataset setup'''
    ds_path = os.path.join(ds_base, 'hamlyn_as_SCARED')
    splits_dir = os.path.join(ds_base, 'hamlyn_as_SCARED', 'splits')
    split_train = os.path.join(ds_path, 'splits/train_files.txt')
    train_filenames = readlines(split_train)

    ds_train_hamlyn = SCAREDRAWDataset(
        data_path=ds_path,
        filenames=train_filenames,
        frame_idxs=DefaultOpt.frame_ids,
        height=DefaultOpt.height,
        width=DefaultOpt.width,
        num_scales=4,
        is_train=True,
        img_ext='.png'
    )
    '''PitVis dataset setup'''
    ds_path = os.path.join(ds_base, 'PitVis_as_SCARED')
    splits_dir = os.path.join(ds_base, 'PitVis_as_SCARED', 'splits')
    split_train = os.path.join(ds_path, 'splits/train_files.txt')
    split_test = os.path.join(ds_path, 'splits/test_files.txt')
    train_filenames = readlines(split_train)
    test_filenames = readlines(split_test)

    ds_train_pitvis = SCAREDRAWDataset(
        data_path=ds_path,
        filenames=train_filenames,
        frame_idxs=DefaultOpt.frame_ids,
        height=DefaultOpt.height,
        width=DefaultOpt.width,
        num_scales=4,
        is_train=True,
        img_ext='.png'
    )

    test_filenames = SCAREDRAWDataset(
        data_path=ds_path,
        filenames=test_filenames,
        frame_idxs=[0],
        height=DefaultOpt.height,
        width=DefaultOpt.width,
        num_scales=4,
        is_train=False,
        img_ext='.png'
    )

    '''SyntheticColon dataset setup'''
    ds_path = os.path.join(ds_base, 'SyntheticColon_as_SCARED')
    splits_dir = os.path.join(ds_base, 'SyntheticColon_as_SCARED', 'splits')
    split_train = os.path.join(ds_path, 'splits/train_files.txt')
    split_test = os.path.join(ds_path, 'splits/test_files.txt')
    train_filenames = readlines(split_train)
    test_filenames = readlines(split_test)

    ds_train_syntheticcolon = SCAREDRAWDataset(
        data_path=ds_path,
        filenames=train_filenames,
        frame_idxs=DefaultOpt.frame_ids,
        height=DefaultOpt.height,
        width=DefaultOpt.width,
        num_scales=4,
        is_train=True,
        img_ext='.png'
    )

    test_filenames = SCAREDRAWDataset(
        data_path=ds_path,
        filenames=test_filenames,
        frame_idxs=[0],
        height=DefaultOpt.height,
        width=DefaultOpt.width,
        num_scales=4,
        is_train=False,
        img_ext='.png'
    )