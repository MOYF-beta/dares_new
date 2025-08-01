import sys
import torch, random, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
if 'cuda' in device:
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

random_seeds(23333666)

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
    min_depth=0.1,
    max_depth=150,
    num_scales=4,
    is_train=True,
    load_depth=True,
    img_ext='.png'
)

ds_val = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=val_filenames,
    frame_idxs=DefaultOpt.frame_ids,
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    min_depth=0.1,
    max_depth=150,
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
    min_depth=0.1,
    max_depth=150,
    num_scales=1,
    is_train=False,
    img_ext='.png'
)

ds_test_multi_frame = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=test_filenames,
    frame_idxs=DefaultOpt.frame_ids,
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    min_depth=0.1,
    max_depth=150,
    num_scales=1,
    is_train=False,
    img_ext='.png'
)


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
    min_depth=0.1,
    max_depth=150,
    num_scales=4,
    is_train=True,
    depth_rescale_factor = 150 / 65535,
    load_depth=True,
    img_ext='.png'
)
ds_test_c3vd = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=test_filenames,
    frame_idxs=[0],
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=1,
    depth_rescale_factor = 150 / 65535,
    min_depth=0.1,
    max_depth=150,
    is_train=False,
    load_depth=True,
    img_ext='.png'
)

ds_path = os.path.join(ds_base, 'C3VD_test_time_train_1')
split_train = os.path.join(ds_path, 'splits/train_files.txt')
train_filenames = readlines(split_train)

ds_testtime_train_c3vd_1_train = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=train_filenames,
    frame_idxs=DefaultOpt.frame_ids,
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=4,
    min_depth=0.1,
    max_depth=150,
    load_depth=True,
    is_train=True,
    depth_rescale_factor = 150 / 65535,
    img_ext='.png'
)

split_test = os.path.join(ds_path, 'splits/test_files.txt')
test_filenames = readlines(split_test)
ds_testtime_train_c3vd_1_test = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=test_filenames,
    frame_idxs=[0],
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=1,
    depth_rescale_factor = 150 / 65535,
    min_depth=0.1,
    max_depth=150,
    is_train=False,
    load_depth=True,
    img_ext='.png'
)


ds_path = os.path.join(ds_base, 'C3VD_test_time_train_2')
split_train = os.path.join(ds_path, 'splits/train_files.txt')
train_filenames = readlines(split_train)
ds_testtime_train_c3vd_2_train = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=train_filenames,
    frame_idxs=DefaultOpt.frame_ids,
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=4,
    min_depth=0.1,
    max_depth=150,
    load_depth=True,
    is_train=True,
    depth_rescale_factor = 150 / 65535,
    img_ext='.png'
)

split_test = os.path.join(ds_path, 'splits/test_files.txt')
test_filenames = readlines(split_test)
ds_testtime_train_c3vd_2_test = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=test_filenames,
    frame_idxs=[0],
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=1,
    depth_rescale_factor = 150 / 65535,
    min_depth=0.1,
    max_depth=150,
    is_train=False,
    load_depth=True,
    img_ext='.png'
)


'''hamlyn dataset setup'''
ds_path = os.path.join(ds_base, 'hamlyn_as_SCARED')
splits_dir = os.path.join(ds_base, 'hamlyn_as_SCARED', 'splits')
split_train = os.path.join(ds_path, 'splits/train_files.txt')
split_test = os.path.join(ds_path, 'splits/test_files.txt')
train_filenames = readlines(split_train)
# random.shuffle(train_filenames)
test_filenames = readlines(split_test)
# random.shuffle(test_filenames)
test_filenames = readlines(split_test)

hamlyn_ratio = 0.05
ds_train_hamlyn = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=train_filenames[:int(hamlyn_ratio * len(train_filenames))],
    frame_idxs=DefaultOpt.frame_ids,
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=4,
    min_depth=0.1,
    max_depth=150,
    load_depth=True,
    is_train=True,
    img_ext='.png'
)

ds_test_hamlyn = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=test_filenames[:int(hamlyn_ratio * len(train_filenames))],
    frame_idxs=[0],
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    load_depth=True,
    num_scales=1,
    min_depth=0.1,
    max_depth=200,
    is_train=False,
    img_ext='.png'
)

'''PitVis dataset setup'''
ds_path = os.path.join(ds_base, 'PitVis_as_SCARED')
splits_dir = os.path.join(ds_base, 'PitVis_as_SCARED', 'splits')
split_train = os.path.join(ds_path, 'splits/train_files.txt')
# split_test = os.path.join(ds_path, 'splits/test_files.txt')
train_filenames = readlines(split_train)
# test_filenames = readlines(split_test)
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
# ds_test_pitvis = SCAREDRAWDataset(
#     data_path=ds_path,
#     filenames=test_filenames,
#     frame_idxs=[0],
#     height=DefaultOpt.height,
#     width=DefaultOpt.width,
#     num_scales=1,
#     is_train=False,
#     img_ext='.png'
# )
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
    min_depth=0.1,
    max_depth=20,
    num_scales=4,
    load_depth=True,
    depth_rescale_factor = 20 / 65535,
    is_train=True,
    img_ext='.png'
)
ds_test_syntheticcolon = SCAREDRAWDataset(
    data_path=ds_path,
    filenames=test_filenames,
    frame_idxs=[0],
    height=DefaultOpt.height,
    width=DefaultOpt.width,
    num_scales=1,
    is_train=False,
    load_depth=True,
    min_depth=0.1,
    max_depth=20,
    depth_rescale_factor= 20 / 65535,
    img_ext='.png'
)
ds_base_model_train = torch.utils.data.ConcatDataset([
    ds_train, ds_train_c3vd, ds_train_hamlyn, ds_train_syntheticcolon])# 

def check_ds():
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    print(f"Train: {len(ds_train)}")
    print(f"Val: {len(ds_val)}")
    print(f"Test: {len(ds_test)}")
    print(f"Test Multi Frame: {len(ds_test_multi_frame)}")
    print(f"Train C3VD: {len(ds_train_c3vd)}")
    print(f"Test C3VD: {len(ds_test_c3vd)}")
    print(f"Train Hamlyn: {len(ds_train_hamlyn)}")
    print(f"Test Hamlyn: {len(ds_test_hamlyn)}")
    # print(f"Train PitVis: {len(ds_train_pitvis)}")
    # print(f"Test PitVis: {len(ds_test_pitvis)}")
    print(f"Train SyntheticColon: {len(ds_train_syntheticcolon)}")
    print(f"Test SyntheticColon: {len(ds_test_syntheticcolon)}")
    ds_to_check = [ds_test_syntheticcolon]
    for ds in ds_to_check:
        for i in tqdm(range(len(ds)), desc=f"Checking dataset {ds}"):
            try:
                def load_data(index):
                    try:
                        return ds[index]
                    except Exception as e:
                        print(f"Error at {index}: {e}")
                        return None

                with ThreadPoolExecutor() as executor:
                    list(tqdm(executor.map(load_data, range(len(ds))), total=len(ds), desc=f"Checking dataset {ds}"))
            except Exception as e:
                print(f"Error at {i}: {e}")

if __name__ == "__main__":
    check_ds()
    # split_train = "/mnt/c/Users/14152/ZCH/Dev/datasets/C3VD_as_SCARED/splits/test_files.txt"
    # a = readlines(split_train)
    # print(len(a))

    # hamlyn_sample = ds_train_hamlyn[0]
    # print(hamlyn_sample.keys())
    # print(hamlyn_sample['depth_gt'].shape)
    # print(hamlyn_sample['depth_gt'].max())
    # print(hamlyn_sample['depth_gt'].min())

    # total_depth = 0
    # total_count = 0
    # max_depth = 0
    # min_depth = 100000
    # import matplotlib.pyplot as plt

    # depths = []

    # for i in range(100):
    #     sample = ds_train_hamlyn[i]
    #     depths.extend(sample['depth_gt'].flatten().tolist())

    # plt.hist(depths, bins=50, range=(0, 1000))
    # plt.title('Depth Distribution of ds_train_hamlyn')
    # plt.xlabel('Depth')
    # plt.ylabel('Frequency')
    # plt.show()

    # for i in range(len(ds_train_hamlyn)):
    #     sample = ds_train_hamlyn[i]
    #     total_depth += sample['depth_gt'].sum()
    #     total_count += sample['depth_gt'].numel()
    #     max_depth = max(max_depth, sample['depth_gt'].max())
    #     min_depth = min(min_depth, sample['depth_gt'].min())

    # average_depth = total_depth / total_count
    # print(f"Average depth of ds_train_hamlyn: {average_depth}")
    # print(f"Max depth of ds_train_hamlyn: {max_depth}")
    # print(f"Min depth of ds_train_hamlyn: {min_depth}")