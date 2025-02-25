from __future__ import absolute_import, division, print_function

import os
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from exps.dataset import SCAREDRAWDataset
from torch.utils.data import DataLoader
import numpy as np
from DARES.networks.resnet_encoder import AttentionalResnetEncoder, MultiHeadAttentionalResnetEncoder
from DARES.networks.pose_decoder import PoseDecoder_with_intrinsics as PoseDecoder_i

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from exps.exp_setup_local import ds_base
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        rs.append(cam_to_world[:3, :3])
    return rs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]


def evaluate(opt, ds_path, load_weights_folder, pose_seq=1, gt_path = None, filename_list_path = None):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(load_weights_folder), \
        "Cannot find a folder at {}".format(load_weights_folder)

    if filename_list_path is None:
        filename_list_path = os.path.join(ds_path, "splits",
                     "test_files_sequence{}.txt".format(pose_seq))
    filenames = readlines(filename_list_path)
    dataset = SCAREDRAWDataset(ds_path, filenames, opt.height, opt.width,
                               [0, 1], 1, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    pose_encoder_path = os.path.join(load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(load_weights_folder, "pose.pth")

    pose_encoder = AttentionalResnetEncoder(opt.num_layers, False, num_input_images=2)
    static_dict = torch.load(pose_encoder_path, map_location=device.type)
    if 'module.' in list(static_dict.keys())[0]:
        static_dict = {k.replace('module.', ''): v for k, v in static_dict.items()}

    pose_encoder.load_state_dict(static_dict)

    pose_decoder = PoseDecoder_i(pose_encoder.num_ch_enc, image_width=opt.width, image_height=opt.height, predict_intrinsics=opt.learn_intrinsics, simplified_intrinsic=opt.simplified_intrinsic, num_input_features=1, num_frames_to_predict_for=2)
    pose_decoder_dict = torch.load(pose_decoder_path, map_location=device.type)
    if 'module.' in list(pose_decoder_dict.keys())[0]:
        pose_decoder_dict = {k.replace('module.', ''): v for k, v in pose_decoder_dict.items()}
    pose_decoder.load_state_dict(pose_decoder_dict)

    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()

    pred_poses = []

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)

            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation, intrinsics = pose_decoder(features)

            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)
    # np.savez_compressed(os.path.join(os.path.dirname(__file__), "splits", "endovis", "curve", "pose_our.npz"), data=np.array(pred_poses))
    # np.savez_compressed(os.path.join(ds_path, "splits", "endovis", "pred_pose_sq{}.npz".format(scared_pose_seq)), data=np.array(pred_poses))
    if gt_path is None:
        gt_path = os.path.join(ds_path, "splits", "endovis", "gt_poses_sq{}.npz".format(pose_seq))
        gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
    else:
        gt_local_poses = np.load(gt_path)

    ates = []
    res = []
    num_frames = gt_local_poses.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
        local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
        gt_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        res.append(compute_re(local_rs, gt_rs))

    # print("Trajectory error: {:0.4f}, std: {:0.4f}\n".format(np.mean(ates), np.std(ates)))
    # print("Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res), np.std(res)))
    return np.mean(ates), np.mean(res), np.std(ates), np.std(res)

if __name__ == "__main__":
    DEBUG = True
    from tqdm import tqdm
    model_dit = 'logs/base_model_2/models'
    result_file = model_dit+'/traj_results.txt'
    from exps.attn_encoder_dora.options_attn_encoder import AttnEncoderOpt
    import glob
    min_ate = np.inf
    min_re = np.inf
    ate_std = np.inf
    re_std = np.inf
    best_i = 0
    weight_folders = glob.glob(model_dit+'/weights_*')
    weight_indices = [int(os.path.basename(folder).split('_')[1]) for folder in weight_folders]
    if DEBUG:
        weight_indices = weight_indices[:2]
    with open(result_file, 'w') as f:
        f.write('Best weight results:\n')
        # -- SCARED Traj 1 --
        for i in weight_indices:
            print(f"Evaluating model {i}")
            ate, re, std_ate, std_re = evaluate(AttnEncoderOpt, os.path.join(ds_base, 'SCARED_Images_Resized'),f'{model_dit}/weights_{i}', pose_seq=1)
            if ate < min_ate:
                min_ate = ate
                min_re = re
                best_i = i
        print(f"SCARED Traj 1:Best ATE: {min_ate}+-{std_ate}, Best RE: {min_re}+-{std_re} @ {best_i}")
        f.write(f"SCARED Traj 1:Best ATE: {min_ate}+-{std_ate}, Best RE: {min_re}+-{std_re} @ {best_i}\n")

        # # -- SCARED Traj 2 --
        min_ate = np.inf
        min_re = np.inf
        ate_std = np.inf
        re_std = np.inf
        for i in weight_indices:
            print(f"Evaluating model {i}")
            ate, re , std_ate, std_re = evaluate(AttnEncoderOpt, os.path.join(ds_base, 'SCARED_Images_Resized'), f'{model_dit}/weights_{i}', pose_seq=2)
            if ate < min_ate:
                min_ate = ate
                min_re = re
                ate_std = std_ate
                re_std = std_re
                best_i = i
        print(f"SCARED Traj 2:Best ATE: {min_ate}+-{std_ate}, Best RE: {min_re}+-{std_re} @ {best_i}")
        f.write(f"SCARED Traj 2:Best ATE: {min_ate}+-{std_ate}, Best RE: {min_re}+-{std_re} @ {best_i}\n")

        # -- SimColon --

        test_folders = os.path.join(ds_base, 'SyntheticColon_as_SCARED', 'splits', 'test_folders.txt')
        with open(test_folders, 'r') as f:
            test_folders = f.readlines()

        min_ate = np.inf
        min_re = np.inf
        ate_std = np.inf
        re_std = np.inf
        for i in weight_indices:
            ates = []
            res = []
            print(f"Evaluating model {i}")
            for test_folder in tqdm(test_folders):
                full_test_folder = os.path.join(ds_base, 'SyntheticColon_as_SCARED', test_folder.strip())
                ate, re , std_ate, std_re = evaluate(AttnEncoderOpt, os.path.join(ds_base, 'SyntheticColon_as_SCARED')
                                   , f'{model_dit}/weights_{i}',
                                    filename_list_path=os.path.join(full_test_folder, 'traj_test.txt'),
                                    gt_path=os.path.join(full_test_folder, 'traj.npy'))
                ates.append(ate)
                res.append(re)
            
            if np.mean(ates) < min_ate:
                min_ate = np.mean(ates)
                min_re = np.mean(res)
                best_i = i
                ate_std = std_ate
                re_std = std_re
        print(f"SyntheticColon : Best ATE: {min_ate}+-{std_ate}, Best RE: {min_re}+-{std_re} @ {best_i}")
        f.write(f"SyntheticColon : Best ATE: {min_ate}+-{std_ate}, Best RE: {min_re}+-{std_re} @ {best_i}\n")

        # -- C3VD --
        test_folders = os.path.join(ds_base, 'C3VD_as_SCARED', 'splits', 'test_folders.txt')
        with open(test_folders, 'r') as f:
            test_folders = f.readlines()

        min_ate = np.inf
        min_re = np.inf
        str_ate = None
        str_re = None
        for i in weight_indices:
            ates = []
            res = []
            print(f"Evaluating model {i}")
            for test_folder in tqdm(test_folders):
                full_test_folder = os.path.join(ds_base, 'C3VD_as_SCARED', test_folder.strip())
                ate, re , std_ate, std_re = evaluate(AttnEncoderOpt, os.path.join(ds_base, 'C3VD_as_SCARED'),
                                    f'{model_dit}/weights_{i}',
                                    filename_list_path=os.path.join(full_test_folder, 'traj_test.txt'),
                                    gt_path=os.path.join(full_test_folder, 'traj.npy'))
                ates.append(ate)
                res.append(re)

            if np.mean(ates) < min_ate:
                min_ate = np.mean(ates)
                min_re = np.mean(res)
                best_i = i
                ate_std = std_ate
                re_std = std_re
                
        print(f"C3VD : Best ATE: {min_ate}+-{std_ate}, Best RE: {min_re}+-{std_re} @ {best_i}")
        f.write(f"C3VD : Best ATE: {min_ate}+-{std_ate}, Best RE: {min_re}+-{std_re} @ {best_i}\n")

    # generate figure
    # evaluate(AttnEncoderOpt, f'{model_dit}/weights_{best_i}', pose_seq=1)
