from __future__ import absolute_import, division, print_function

import os
import torch
import sys
import argparse
import numpy as np
import evo
from evo.core import trajectory
from evo.tools import file_interface
from evo.core import metrics
from evo.core import sync
from tqdm import tqdm
from evo.tools import plot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from exps.dataset import SCAREDRAWDataset
from torch.utils.data import DataLoader
from DARES.networks.resnet_encoder import AttentionalResnetEncoder, MultiHeadAttentionalResnetEncoder
from DARES.networks.pose_decoder import PoseDecoder_with_intrinsics as PoseDecoder_i

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from exps.exp_setup_local import ds_base
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transformation_matrix_to_kitti_format(transformations):
    """Convert transformation matrices to KITTI format (r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3)"""
    kitti_data = []
    
    for transformation in transformations:
        # Extract rotation matrix and translation vector
        rotation_matrix = transformation[:3, :3]
        translation = transformation[:3, 3]
        
        # Format: r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
        line = " ".join([str(rotation_matrix[0, 0]), str(rotation_matrix[0, 1]), str(rotation_matrix[0, 2]), str(translation[0]),
                         str(rotation_matrix[1, 0]), str(rotation_matrix[1, 1]), str(rotation_matrix[1, 2]), str(translation[1]),
                         str(rotation_matrix[2, 0]), str(rotation_matrix[2, 1]), str(rotation_matrix[2, 2]), str(translation[2])])
        kitti_data.append(line)
        
    return kitti_data

def transformation_matrix_to_tum_format(transformations):
    """Convert transformation matrices to TUM format (timestamp tx ty tz qx qy qz qw)"""
    from scipy.spatial.transform import Rotation
    
    tum_data = []
    timestamp = 0
    
    for transformation in transformations:
        rotation_matrix = transformation[:3, :3]
        translation = transformation[:3, 3]
        
        # Convert rotation matrix to quaternion
        r = Rotation.from_matrix(rotation_matrix)
        quat = r.as_quat()  # x, y, z, w format
        
        # TUM format uses quaternion in w, x, y, z order
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        
        # Format: timestamp tx ty tz qx qy qz qw
        tum_data.append(f"{timestamp} {translation[0]} {translation[1]} {translation[2]} {qx} {qy} {qz} {qw}")
        timestamp += 0.1  # Increment timestamp
        
    return tum_data

def write_trajectory_to_file(transformation_matrices, output_path, format='kitti'):
    """Write transformation matrices to a file in specified format"""
    if format == 'tum':
        data = transformation_matrix_to_tum_format(transformation_matrices)
    else:  # kitti format
        data = transformation_matrix_to_kitti_format(transformation_matrices)
    
    with open(output_path, 'w') as f:
        for line in data:
            f.write(line + '\n')
    
    return output_path

def load_kitti_poses(pose_file_path):
    """Load KITTI pose file and convert to evo trajectory"""
    poses = []
    with open(pose_file_path, 'r') as f:
        for line in f.readlines():
            values = [float(v) for v in line.strip().split()]
            pose = np.eye(4)
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            poses.append(pose)
    
    if len(poses) == 0:
        raise ValueError(f"No poses found in file: {pose_file_path}")
    
    # Pre-allocate arrays for trajectory
    positions = np.zeros((len(poses), 3))
    orientations = np.zeros((len(poses), 4))
    timestamps = np.zeros(len(poses))
    
    # Fill arrays with pose data
    for i, pose in enumerate(poses):
        # Extract translation
        positions[i] = pose[:3, 3]
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(pose[:3, :3])
        q = r.as_quat()  # x, y, z, w
        orientations[i] = np.array([q[3], q[0], q[1], q[2]])  # convert to w, x, y, z
        timestamps[i] = float(i)
    
    # Create trajectory with all data at once
    traj = trajectory.PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=orientations, 
        timestamps=timestamps
    )
    
    return traj

def evaluate(opt, ds_path, load_weights_folder, pose_seq=1, gt_path=None, filename_list_path=None):
    """Evaluate odometry on the specified dataset using evo
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
    
    # Create temporary files for evaluation
    temp_dir = os.path.join(os.path.dirname(load_weights_folder), "temp_eval")
    os.makedirs(temp_dir, exist_ok=True)
    pred_traj_file = os.path.join(temp_dir, "pred_trajectory.txt")
    
    # Convert predicted poses to trajectory file
    accumulated_poses = [np.eye(4)]
    for i in range(len(pred_poses)):
        accumulated_poses.append(np.dot(accumulated_poses[-1], pred_poses[i]))
    
    # Write trajectory in KITTI format
    write_trajectory_to_file(accumulated_poses[1:], pred_traj_file, format='kitti')
    
    # Load ground truth from KITTI format file
    if gt_path is None:
        gt_path = os.path.join(ds_path, "splits", f"traj_kitti_{pose_seq}.txt")
    
    # Load trajectories with evo
    traj_ref = load_kitti_poses(gt_path)
    traj_est = load_kitti_poses(pred_traj_file)
    
    # Associate, register and align trajectories
    print("Registering and aligning trajectories")
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    traj_est.align(traj_ref, correct_scale=True)
    
    # Calculate ATE
    ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ate_metric.process_data((traj_ref, traj_est))
    ate_statistics = ate_metric.get_all_statistics()
    
    # Calculate RPE
    rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)
    rpe_metric.process_data((traj_ref, traj_est))
    rpe_statistics = rpe_metric.get_all_statistics()

    # Save trajectory visualization
    # Create output folder for visualizations
    vis_output_dir = os.path.join(os.path.dirname(load_weights_folder), "trajectory_vis")
    os.makedirs(vis_output_dir, exist_ok=True)

    # Create a plot collection for visualization
    import matplotlib.pyplot as plt

    plot_collection = plot.PlotCollection(f"Trajectory Evaluation - Seq {pose_seq}")

    # Plot ATE error
    fig_ate = plt.figure(figsize=(10, 8))
    plot.error_array(fig_ate.gca(), ate_metric.error, statistics=ate_statistics,
                    name="APE", title=str(ate_metric))
    plot_collection.add_figure("ate_error", fig_ate)

    # Plot trajectory with ATE error colormap
    fig_traj_ate = plt.figure(figsize=(10, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig_traj_ate, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
    plot.traj_colormap(ax, traj_est, ate_metric.error, plot_mode,
                      min_map=ate_statistics["min"],
                      max_map=ate_statistics["max"],
                      title="ATE mapped onto trajectory")
    plot_collection.add_figure("traj_ate", fig_traj_ate)

    # Plot RPE error
    fig_rpe = plt.figure(figsize=(10, 8))
    plot.error_array(fig_rpe.gca(), rpe_metric.error, statistics=rpe_statistics,
                    name="RPE", title=str(rpe_metric))
    plot_collection.add_figure("rpe_error", fig_rpe)

    # Plot 3D view of trajectories
    fig_3d = plt.figure(figsize=(10, 8))
    ax = plot.prepare_axis(fig_3d, plot.PlotMode.xyz)
    plot.traj(ax, plot.PlotMode.xyz, traj_ref, '--', 'gray', 'reference')
    plot.traj(ax, plot.PlotMode.xyz, traj_est, '-', 'blue', 'estimated')
    ax.legend()
    plot_collection.add_figure("traj_3d", fig_3d)

    # Save all plots to the output directory
    output_filename = os.path.join(vis_output_dir, f"trajectory_seq{pose_seq}")
    plot_collection.export(output_filename, confirm_overwrite=False)
    plt.close('all')
    
    # Print results
    print(f"ATE RMSE: {ate_statistics['rmse']}, std: {ate_statistics['std']}")
    print(f"RPE mean: {rpe_statistics['mean']}, std: {rpe_statistics['std']}")
    
    return ate_statistics['rmse'], rpe_statistics['mean'], ate_statistics['std'], rpe_statistics['std']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pose estimation on different datasets")
    parser.add_argument('--dataset', type=str, choices=['SCARED', 'SyntheticColon', 'C3VD'], default='SCARED',
                        help='Dataset to evaluate on (SCARED, SyntheticColon, or C3VD)')
    parser.add_argument('--model_dir', type=str, default='logs/base_model_2/models',
                        help='Directory containing the model weights')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (limited evaluation)')
    args = parser.parse_args()
    
    DEBUG = args.debug
    model_dir = args.model_dir
    result_file = os.path.join(model_dir, 'traj_results.txt')
    from exps.attn_encoder_dora.options_attn_encoder import AttnEncoderOpt
    import glob
    
    weight_folders = glob.glob(os.path.join(model_dir, 'weights_*'))
    weight_indices = [int(os.path.basename(folder).split('_')[1]) for folder in weight_folders]
    if DEBUG:
        weight_indices = weight_indices[:2]
    
    with open(result_file, 'w') as f_result:
        f_result.write('Best weight results:\n')
        
        if args.dataset == 'SCARED':
            # -- SCARED Traj 1 --
            min_ate = np.inf
            min_re = np.inf
            ate_std = np.inf
            re_std = np.inf
            best_i = 0
            
            for i in weight_indices:
                print(f"Evaluating model {i} on SCARED Trajectory 1")
                ate, re, std_ate, std_re = evaluate(
                    AttnEncoderOpt, 
                    os.path.join(ds_base, 'SCARED_Images_Resized'),
                    f'{model_dir}/weights_{i}', 
                    pose_seq=1
                )
                
                if ate < min_ate:
                    min_ate = ate
                    min_re = re
                    ate_std = std_ate
                    re_std = std_re
                    best_i = i
                    
            print(f"SCARED Traj 1: Best ATE: {min_ate}+-{ate_std}, Best RE: {min_re}+-{re_std} @ {best_i}")
            f_result.write(f"SCARED Traj 1: Best ATE: {min_ate}+-{ate_std}, Best RE: {min_re}+-{re_std} @ {best_i}\n")
            
            # -- SCARED Traj 2 --
            min_ate = np.inf
            min_re = np.inf
            ate_std = np.inf
            re_std = np.inf
            best_i = 0
            
            for i in weight_indices:
                print(f"Evaluating model {i} on SCARED Trajectory 2")
                ate, re, std_ate, std_re = evaluate(
                    AttnEncoderOpt, 
                    os.path.join(ds_base, 'SCARED_Images_Resized'), 
                    f'{model_dir}/weights_{i}', 
                    pose_seq=2
                )
                
                if ate < min_ate:
                    min_ate = ate
                    min_re = re
                    ate_std = std_ate
                    re_std = std_re
                    best_i = i
                    
            print(f"SCARED Traj 2: Best ATE: {min_ate}+-{ate_std}, Best RE: {min_re}+-{re_std} @ {best_i}")
            f_result.write(f"SCARED Traj 2: Best ATE: {min_ate}+-{ate_std}, Best RE: {min_re}+-{re_std} @ {best_i}\n")
            
        elif args.dataset == 'SyntheticColon':
            # -- SimColon --
            test_folders_file = os.path.join(ds_base, 'SyntheticColon_as_SCARED', 'splits', 'test_folders.txt')
            with open(test_folders_file, 'r') as f:
                test_folders = f.readlines()
            
            min_ate = np.inf
            min_re = np.inf
            ate_std = np.inf
            re_std = np.inf
            best_i = 0
            
            for i in weight_indices:
                ates = []
                res = []
                std_ates = []
                std_res = []
                print(f"Evaluating model {i} on SyntheticColon")
                
                
                for test_folder in tqdm(test_folders):
                    full_test_folder = os.path.join(ds_base, 'SyntheticColon_as_SCARED', test_folder.strip())
                    ate, re, std_ate, std_re = evaluate(
                        AttnEncoderOpt, 
                        os.path.join(ds_base, 'SyntheticColon_as_SCARED'),
                        f'{model_dir}/weights_{i}',
                        filename_list_path=os.path.join(full_test_folder, 'traj_test.txt'),
                        gt_path=os.path.join(full_test_folder, 'traj.txt')
                    )
                    ates.append(ate)
                    res.append(re)
                    std_ates.append(std_ate)
                    std_res.append(std_re)
                
                if np.mean(ates) < min_ate:
                    min_ate = np.mean(ates)
                    min_re = np.mean(res)
                    best_i = i
                    ate_std = np.mean(std_ates)
                    re_std = np.mean(std_res)
                    
            print(f"SyntheticColon: Best ATE: {min_ate}+-{ate_std}, Best RE: {min_re}+-{re_std} @ {best_i}")
            f_result.write(f"SyntheticColon: Best ATE: {min_ate}+-{ate_std}, Best RE: {min_re}+-{re_std} @ {best_i}\n")
            
        elif args.dataset == 'C3VD':
            # -- C3VD --
            test_folders_file = os.path.join(ds_base, 'C3VD_as_SCARED', 'splits', 'test_folders.txt')
            with open(test_folders_file, 'r') as f:
                test_folders = f.readlines()
            
            min_ate = np.inf
            min_re = np.inf
            ate_std = np.inf
            re_std = np.inf
            best_i = 0
            
            for i in weight_indices:
                ates = []
                res = []
                std_ates = []
                std_res = []
                print(f"Evaluating model {i} on C3VD")
                

                for test_folder in tqdm(test_folders):
                    full_test_folder = os.path.join(ds_base, 'C3VD_as_SCARED', test_folder.strip())
                    ate, re, std_ate, std_re = evaluate(
                        AttnEncoderOpt, 
                        os.path.join(ds_base, 'C3VD_as_SCARED'),
                        f'{model_dir}/weights_{i}',
                        filename_list_path=os.path.join(full_test_folder, 'traj_test.txt'),
                        gt_path=os.path.join(full_test_folder, 'traj.txt')
                    )
                    ates.append(ate)
                    res.append(re)
                    std_ates.append(std_ate)
                    std_res.append(std_re)
                
                if np.mean(ates) < min_ate:
                    min_ate = np.mean(ates)
                    min_re = np.mean(res)
                    best_i = i
                    ate_std = np.mean(std_ates)
                    re_std = np.mean(std_res)
                    
            print(f"C3VD: Best ATE: {min_ate}+-{ate_std}, Best RE: {min_re}+-{re_std} @ {best_i}")
            f_result.write(f"C3VD: Best ATE: {min_ate}+-{ate_std}, Best RE: {min_re}+-{re_std} @ {best_i}\n")
