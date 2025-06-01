from __future__ import absolute_import, division, print_function

import os
import torch
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
import warnings

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

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_intrinsic_errors(pred_intrinsics, gt_intrinsics):
    """Compute scale-invariant errors between predicted and ground truth intrinsics matrices.
    
    Args:
        pred_intrinsics: Predicted intrinsics matrix or batch of matrices (B, 4, 4)
        gt_intrinsics: Ground truth intrinsics matrix (4, 4)
        
    Returns:
        Dictionary containing various error metrics with scale-invariant calculations
    """
    # If pred_intrinsics is a batch, compute mean errors
    if len(pred_intrinsics.shape) > 2:
        errors = {
            'fx_abs_error': [],
            'fy_abs_error': [],
            'cx_abs_error': [],
            'cy_abs_error': [],
            'fx_rel_error': [],
            'fy_rel_error': [],
            'cx_rel_error': [],
            'cy_rel_error': [],
            'matrix_frobenius_norm': [],
            'scale_factor': [],
            'scale_adjusted_fx_abs_error': [],
            'scale_adjusted_fy_abs_error': [],
            'scale_adjusted_cx_abs_error': [],
            'scale_adjusted_cy_abs_error': [],
            'scale_adjusted_fx_rel_error': [],
            'scale_adjusted_fy_rel_error': [],
            'scale_adjusted_cx_rel_error': [],
            'scale_adjusted_cy_rel_error': [],
            'scale_adjusted_matrix_frobenius_norm': []
        }
        
        for pred_mat in pred_intrinsics:
            # Extract intrinsic parameters
            pred_fx = pred_mat[0, 0].item()
            pred_fy = pred_mat[1, 1].item()
            pred_cx = pred_mat[0, 2].item()
            pred_cy = pred_mat[1, 2].item()
            
            gt_fx = gt_intrinsics[0, 0]
            gt_fy = gt_intrinsics[1, 1]
            gt_cx = gt_intrinsics[0, 2]
            gt_cy = gt_intrinsics[1, 2]
            
            # Standard metrics
            # Compute absolute errors
            errors['fx_abs_error'].append(abs(pred_fx - gt_fx))
            errors['fy_abs_error'].append(abs(pred_fy - gt_fy))
            errors['cx_abs_error'].append(abs(pred_cx - gt_cx))
            errors['cy_abs_error'].append(abs(pred_cy - gt_cy))
            
            # Compute relative errors (%)
            errors['fx_rel_error'].append(abs(pred_fx - gt_fx) / gt_fx * 100)
            errors['fy_rel_error'].append(abs(pred_fy - gt_fy) / gt_fy * 100)
            errors['cx_rel_error'].append(abs(pred_cx - gt_cx) / gt_cx * 100)
            errors['cy_rel_error'].append(abs(pred_cy - gt_cy) / gt_cy * 100)
            
            # Frobenius norm of difference matrix (only the 3x3 part)
            diff_mat = pred_mat[:3, :3].cpu().numpy() - gt_intrinsics[:3, :3]
            errors['matrix_frobenius_norm'].append(np.linalg.norm(diff_mat, 'fro'))
            
            # Calculate optimal scale factor based on focal length ratio
            scale_fx = gt_fx / pred_fx if pred_fx != 0 else 1.0
            scale_fy = gt_fy / pred_fy if pred_fy != 0 else 1.0
            
            # Use the average scale
            scale = (scale_fx + scale_fy) / 2.0
                    # Try different ways to determine scale
                    scale_fx = gt_fx / pred_fx if pred_fx != 0 else 1.0
                    scale_fy = gt_fy / pred_fy if pred_fy != 0 else 1.0
                    
                    # Use the average scale or the one that minimizes the error
                    scale = (scale_fx + scale_fy) / 2.0
                else:
                    # Use a fixed scale of 1.0 if not finding optimal scale
                    scale = 1.0
                
                errors['scale_factor'].append(scale)
                
                # Apply the scale to the predicted intrinsics
                scaled_pred_fx = pred_fx * scale
                scaled_pred_fy = pred_fy * scale
                # Principal points don't necessarily scale the same way as focal lengths
                # but in many cases they might have a proportional relationship
                scaled_pred_cx = pred_cx * scale
                scaled_pred_cy = pred_cy * scale
                
                # Compute scale-adjusted absolute errors
                errors['scale_adjusted_fx_abs_error'].append(abs(scaled_pred_fx - gt_fx))
                errors['scale_adjusted_fy_abs_error'].append(abs(scaled_pred_fy - gt_fy))
                errors['scale_adjusted_cx_abs_error'].append(abs(scaled_pred_cx - gt_cx))
                errors['scale_adjusted_cy_abs_error'].append(abs(scaled_pred_cy - gt_cy))
                
                # Compute scale-adjusted relative errors (%)
                errors['scale_adjusted_fx_rel_error'].append(abs(scaled_pred_fx - gt_fx) / gt_fx * 100)
                errors['scale_adjusted_fy_rel_error'].append(abs(scaled_pred_fy - gt_fy) / gt_fy * 100)
                errors['scale_adjusted_cx_rel_error'].append(abs(scaled_pred_cx - gt_cx) / gt_cx * 100)
                errors['scale_adjusted_cy_rel_error'].append(abs(scaled_pred_cy - gt_cy) / gt_cy * 100)
                
                # Scale-adjusted Frobenius norm
                scaled_pred_mat = pred_mat.clone()
                scaled_pred_mat[0, 0] *= scale  # fx
                scaled_pred_mat[1, 1] *= scale  # fy
                scaled_pred_mat[0, 2] *= scale  # cx
                scaled_pred_mat[1, 2] *= scale  # cy
                diff_mat_scaled = scaled_pred_mat[:3, :3].cpu().numpy() - gt_intrinsics[:3, :3]
                errors['scale_adjusted_matrix_frobenius_norm'].append(np.linalg.norm(diff_mat_scaled, 'fro'))
        
        # Calculate statistics
        result = {}
        for key, values in errors.items():
            result[f'{key}_mean'] = np.mean(values)
            result[f'{key}_std'] = np.std(values)
            result[f'{key}_median'] = np.median(values)
            result[f'{key}_min'] = np.min(values)
            result[f'{key}_max'] = np.max(values)
        
        # Add raw errors for plotting
        result['errors'] = errors
        
        return result    else:
        # Single prediction case
        pred_fx = pred_intrinsics[0, 0].item()
        pred_fy = pred_intrinsics[1, 1].item()
        pred_cx = pred_intrinsics[0, 2].item()
        pred_cy = pred_intrinsics[1, 2].item()
        
        gt_fx = gt_intrinsics[0, 0]
        gt_fy = gt_intrinsics[1, 1]
        gt_cx = gt_intrinsics[0, 2]
        gt_cy = gt_intrinsics[1, 2]
        
        # Compute absolute errors
        fx_abs_error = abs(pred_fx - gt_fx)
        fy_abs_error = abs(pred_fy - gt_fy)
        cx_abs_error = abs(pred_cx - gt_cx)
        cy_abs_error = abs(pred_cy - gt_cy)
        
        # Compute relative errors (%)
        fx_rel_error = abs(pred_fx - gt_fx) / gt_fx * 100
        fy_rel_error = abs(pred_fy - gt_fy) / gt_fy * 100
        cx_rel_error = abs(pred_cx - gt_cx) / gt_cx * 100
        cy_rel_error = abs(pred_cy - gt_cy) / gt_cy * 100
        
        # Frobenius norm of difference matrix (only the 3x3 part)
        diff_mat = pred_intrinsics[:3, :3].cpu().numpy() - gt_intrinsics[:3, :3]
        matrix_frobenius_norm = np.linalg.norm(diff_mat, 'fro')
        
        result = {
            'fx_abs_error': fx_abs_error,
            'fy_abs_error': fy_abs_error,
            'cx_abs_error': cx_abs_error,
            'cy_abs_error': cy_abs_error,
            'fx_rel_error': fx_rel_error,
            'fy_rel_error': fy_rel_error,
            'cx_rel_error': cx_rel_error,
            'cy_rel_error': cy_rel_error,
            'matrix_frobenius_norm': matrix_frobenius_norm
        }
        
        # Scale-invariant metrics (important for monocular depth estimation)
        if scale_invariant:
            # Calculate optimal scale factor based on focal length ratio
            if find_optimal_scale:
                # Try different ways to determine scale
                scale_fx = gt_fx / pred_fx if pred_fx != 0 else 1.0
                scale_fy = gt_fy / pred_fy if pred_fy != 0 else 1.0
                
                # Use the average scale or the one that minimizes the error
                scale = (scale_fx + scale_fy) / 2.0
            else:
                # Use a fixed scale of 1.0 if not finding optimal scale
                scale = 1.0
            
            # Apply the scale to the predicted intrinsics
            scaled_pred_fx = pred_fx * scale
            scaled_pred_fy = pred_fy * scale
            # Principal points don't necessarily scale the same way as focal lengths
            # but in many cases they might have a proportional relationship
            scaled_pred_cx = pred_cx * scale
            scaled_pred_cy = pred_cy * scale
            
            # Compute scale-adjusted absolute errors
            scale_adjusted_fx_abs_error = abs(scaled_pred_fx - gt_fx)
            scale_adjusted_fy_abs_error = abs(scaled_pred_fy - gt_fy)
            scale_adjusted_cx_abs_error = abs(scaled_pred_cx - gt_cx)
            scale_adjusted_cy_abs_error = abs(scaled_pred_cy - gt_cy)
            
            # Compute scale-adjusted relative errors (%)
            scale_adjusted_fx_rel_error = abs(scaled_pred_fx - gt_fx) / gt_fx * 100
            scale_adjusted_fy_rel_error = abs(scaled_pred_fy - gt_fy) / gt_fy * 100
            scale_adjusted_cx_rel_error = abs(scaled_pred_cx - gt_cx) / gt_cx * 100
            scale_adjusted_cy_rel_error = abs(scaled_pred_cy - gt_cy) / gt_cy * 100
              # Scale-adjusted Frobenius norm
            scaled_pred_mat = pred_intrinsics.clone()
            scaled_pred_mat[0, 0] *= scale  # fx
            scaled_pred_mat[1, 1] *= scale  # fy
            scaled_pred_mat[0, 2] *= scale  # cx
            scaled_pred_mat[1, 2] *= scale  # cy
            diff_mat_scaled = scaled_pred_mat[:3, :3].cpu().numpy() - gt_intrinsics[:3, :3]
            scale_adjusted_matrix_frobenius_norm = np.linalg.norm(diff_mat_scaled, 'fro')
            
            result.update({
                'scale_factor': scale,
                'scale_adjusted_fx_abs_error': scale_adjusted_fx_abs_error,
                'scale_adjusted_fy_abs_error': scale_adjusted_fy_abs_error,
                'scale_adjusted_cx_abs_error': scale_adjusted_cx_abs_error,
                'scale_adjusted_cy_abs_error': scale_adjusted_cy_abs_error,
                'scale_adjusted_fx_rel_error': scale_adjusted_fx_rel_error,
                'scale_adjusted_fy_rel_error': scale_adjusted_fy_rel_error,
                'scale_adjusted_cx_rel_error': scale_adjusted_cx_rel_error,
                'scale_adjusted_cy_rel_error': scale_adjusted_cy_rel_error,
                'scale_adjusted_matrix_frobenius_norm': scale_adjusted_matrix_frobenius_norm
            })
        
        return result

def visualize_intrinsics_errors(errors, output_dir, name_prefix=""):
    """Visualize intrinsics errors as boxplots and histograms.
    
    Args:
        errors: Dictionary containing error values from compute_intrinsic_errors
        output_dir: Directory to save visualization plots
        name_prefix: Prefix for saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_errors = errors['errors']
    
    # Create parameter names and descriptions
    param_names = ['fx', 'fy', 'cx', 'cy']
    param_desc = ['Focal Length X', 'Focal Length Y', 'Principal Point X', 'Principal Point Y']
    
    # Plot absolute errors
    plt.figure(figsize=(12, 8))
    abs_data = [raw_errors['fx_abs_error'], raw_errors['fy_abs_error'], 
                raw_errors['cx_abs_error'], raw_errors['cy_abs_error']]
    plt.boxplot(abs_data, labels=param_names)
    plt.title('Absolute Errors in Camera Intrinsic Parameters')
    plt.ylabel('Absolute Error (pixels)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{name_prefix}absolute_errors_boxplot.png"))
    plt.close()
    
    # Plot relative errors
    plt.figure(figsize=(12, 8))
    rel_data = [raw_errors['fx_rel_error'], raw_errors['fy_rel_error'], 
                raw_errors['cx_rel_error'], raw_errors['cy_rel_error']]
    plt.boxplot(rel_data, labels=param_names)
    plt.title('Relative Errors in Camera Intrinsic Parameters')
    plt.ylabel('Relative Error (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{name_prefix}relative_errors_boxplot.png"))
    plt.close()
    
    # If we have scale-invariant metrics, plot those too
    if 'scale_adjusted_fx_abs_error' in raw_errors:
        # Plot scale-adjusted absolute errors
        plt.figure(figsize=(12, 8))
        scale_abs_data = [raw_errors['scale_adjusted_fx_abs_error'], raw_errors['scale_adjusted_fy_abs_error'], 
                          raw_errors['scale_adjusted_cx_abs_error'], raw_errors['scale_adjusted_cy_abs_error']]
        plt.boxplot(scale_abs_data, labels=param_names)
        plt.title('Scale-Adjusted Absolute Errors in Camera Intrinsic Parameters')
        plt.ylabel('Scale-Adjusted Absolute Error (pixels)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f"{name_prefix}scale_adjusted_absolute_errors_boxplot.png"))
        plt.close()
        
        # Plot scale-adjusted relative errors
        plt.figure(figsize=(12, 8))
        scale_rel_data = [raw_errors['scale_adjusted_fx_rel_error'], raw_errors['scale_adjusted_fy_rel_error'], 
                          raw_errors['scale_adjusted_cx_rel_error'], raw_errors['scale_adjusted_cy_rel_error']]
        plt.boxplot(scale_rel_data, labels=param_names)
        plt.title('Scale-Adjusted Relative Errors in Camera Intrinsic Parameters')
        plt.ylabel('Scale-Adjusted Relative Error (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f"{name_prefix}scale_adjusted_relative_errors_boxplot.png"))
        plt.close()
        
        # Plot scale factors
        plt.figure(figsize=(10, 6))
        sns.histplot(raw_errors['scale_factor'], kde=True)
        plt.title('Distribution of Scale Factors')
        plt.xlabel('Scale Factor')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f"{name_prefix}scale_factors.png"))
        plt.close()
      # Create histograms for each parameter
    for i, param in enumerate(param_names):
        # Determine if we have scale-adjusted metrics
        has_scale_metrics = 'scale_adjusted_fx_abs_error' in raw_errors
        
        if has_scale_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            
            # Absolute error histogram
            sns.histplot(abs_data[i], kde=True, ax=axes[0, 0])
            axes[0, 0].set_title(f'Absolute Error Distribution - {param_desc[i]}')
            axes[0, 0].set_xlabel('Absolute Error (pixels)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, linestyle='--', alpha=0.7)
            
            # Relative error histogram
            sns.histplot(rel_data[i], kde=True, ax=axes[0, 1])
            axes[0, 1].set_title(f'Relative Error Distribution - {param_desc[i]}')
            axes[0, 1].set_xlabel('Relative Error (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)
            
            # Scale-adjusted absolute error histogram
            scale_abs_data = [raw_errors['scale_adjusted_fx_abs_error'], 
                             raw_errors['scale_adjusted_fy_abs_error'],
                             raw_errors['scale_adjusted_cx_abs_error'], 
                             raw_errors['scale_adjusted_cy_abs_error']]
            sns.histplot(scale_abs_data[i], kde=True, ax=axes[1, 0])
            axes[1, 0].set_title(f'Scale-Adjusted Absolute Error - {param_desc[i]}')
            axes[1, 0].set_xlabel('Scale-Adjusted Absolute Error (pixels)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, linestyle='--', alpha=0.7)
            
            # Scale-adjusted relative error histogram
            scale_rel_data = [raw_errors['scale_adjusted_fx_rel_error'], 
                              raw_errors['scale_adjusted_fy_rel_error'],
                              raw_errors['scale_adjusted_cx_rel_error'], 
                              raw_errors['scale_adjusted_cy_rel_error']]
            sns.histplot(scale_rel_data[i], kde=True, ax=axes[1, 1])
            axes[1, 1].set_title(f'Scale-Adjusted Relative Error - {param_desc[i]}')
            axes[1, 1].set_xlabel('Scale-Adjusted Relative Error (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Absolute error histogram
            sns.histplot(abs_data[i], kde=True, ax=ax1)
            ax1.set_title(f'Absolute Error Distribution - {param_desc[i]}')
            ax1.set_xlabel('Absolute Error (pixels)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Relative error histogram
            sns.histplot(rel_data[i], kde=True, ax=ax2)
            ax2.set_title(f'Relative Error Distribution - {param_desc[i]}')
            ax2.set_xlabel('Relative Error (%)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name_prefix}{param}_error_distribution.png"))
        plt.close()
      # Create a matrix norm error histogram
    if 'scale_adjusted_matrix_frobenius_norm' in raw_errors:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Standard matrix norm
        sns.histplot(raw_errors['matrix_frobenius_norm'], kde=True, ax=ax1)
        ax1.set_title('Frobenius Norm of Intrinsics Error Matrix')
        ax1.set_xlabel('Frobenius Norm')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Scale-adjusted matrix norm
        sns.histplot(raw_errors['scale_adjusted_matrix_frobenius_norm'], kde=True, ax=ax2)
        ax2.set_title('Scale-Adjusted Frobenius Norm of Intrinsics Error Matrix')
        ax2.set_xlabel('Scale-Adjusted Frobenius Norm')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name_prefix}matrix_norm_error.png"))
        plt.close()
        
        # Comparative boxplot for original vs scale-adjusted
        plt.figure(figsize=(10, 6))
        comp_data = [raw_errors['matrix_frobenius_norm'], raw_errors['scale_adjusted_matrix_frobenius_norm']]
        plt.boxplot(comp_data, labels=['Original', 'Scale-Adjusted'])
        plt.title('Comparison of Original vs Scale-Adjusted Matrix Error')
        plt.ylabel('Frobenius Norm')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f"{name_prefix}matrix_norm_comparison.png"))
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        sns.histplot(raw_errors['matrix_frobenius_norm'], kde=True)
        plt.title('Frobenius Norm of Intrinsics Error Matrix')
        plt.xlabel('Frobenius Norm')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f"{name_prefix}matrix_norm_error.png"))
        plt.close()

def evaluate_intrinsics(opt, ds_path, load_weights_folder, dataset_name="SCARED", pose_seq=1, 
                   filename_list_path=None, scale_invariant=True, find_optimal_scale=True):
    """Evaluate camera intrinsics prediction on the specified dataset
    
    Args:
        opt: Options object containing configuration
        ds_path: Path to dataset
        load_weights_folder: Path to folder containing model weights
        dataset_name: Name of the dataset for result reporting
        pose_seq: Pose sequence number (if applicable)
        filename_list_path: Path to file listing test image filenames
        scale_invariant: Whether to compute scale-invariant metrics (important for monocular depth estimation)
        find_optimal_scale: Whether to find the optimal scale factor between predicted and ground truth intrinsics
        
    Returns:
        Dictionary of error metrics
    """
    assert os.path.isdir(load_weights_folder), \
        f"Cannot find a folder at {load_weights_folder}"
    
    # Setup dataset and dataloader
    if filename_list_path is None:
        filename_list_path = os.path.join(ds_path, "splits",
                                       f"test_files_sequence{pose_seq}.txt")
    
    print(f"Using test files list: {filename_list_path}")
    filenames = readlines(filename_list_path)
    
    dataset = SCAREDRAWDataset(ds_path, filenames, opt.height, opt.width,
                               [0, 1], 1, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    
    # Load ground truth intrinsics
    gt_intrinsics_path = os.path.join(ds_path, "intrinsics.txt")
    if not os.path.exists(gt_intrinsics_path):
        print(f"Warning: Ground truth intrinsics file not found at {gt_intrinsics_path}")
        print("Looking for K.txt instead...")
        gt_intrinsics_path = os.path.join(ds_path, "K.txt")
    
    if not os.path.exists(gt_intrinsics_path):
        raise FileNotFoundError(f"Cannot find ground truth intrinsics at {gt_intrinsics_path} or K.txt")
    
    gt_intrinsics = np.loadtxt(gt_intrinsics_path)
    
    # Check if intrinsics is just 3x3 and convert to 4x4 if needed
    if gt_intrinsics.shape == (3, 3):
        temp = np.eye(4)
        temp[:3, :3] = gt_intrinsics
        gt_intrinsics = temp
    
    print(f"Loaded ground truth intrinsics:\n{gt_intrinsics}")
    
    # Load models
    pose_encoder_path = os.path.join(load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(load_weights_folder, "pose.pth")
    
    # Check if the model supports intrinsics prediction
    if not opt.learn_intrinsics:
        raise ValueError("The selected model is not configured to predict intrinsics. "
                         "Set learn_intrinsics=True in options.")
    
    # Load encoder
    pose_encoder = AttentionalResnetEncoder(opt.num_layers, False, num_input_images=2)
    encoder_dict = torch.load(pose_encoder_path, map_location=device.type)
    if 'module.' in list(encoder_dict.keys())[0]:
        encoder_dict = {k.replace('module.', ''): v for k, v in encoder_dict.items()}
    pose_encoder.load_state_dict(encoder_dict)
    
    # Load decoder with intrinsics prediction
    pose_decoder = PoseDecoder_i(
        pose_encoder.num_ch_enc, 
        image_width=opt.width, 
        image_height=opt.height, 
        predict_intrinsics=opt.learn_intrinsics,
        simplified_intrinsic=opt.simplified_intrinsic,
        num_input_features=1, 
        num_frames_to_predict_for=2
    )
    decoder_dict = torch.load(pose_decoder_path, map_location=device.type)
    if 'module.' in list(decoder_dict.keys())[0]:
        decoder_dict = {k.replace('module.', ''): v for k, v in decoder_dict.items()}
    pose_decoder.load_state_dict(decoder_dict)
    
    # Move models to device and set to eval mode
    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()
    
    # Collect predictions
    all_pred_intrinsics = []
    
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Evaluating intrinsics"):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)
            
            # Prepare inputs and run the model
            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)
            features = [pose_encoder(all_color_aug)]
            
            # Get predictions (axisangle, translation, intrinsics)
            _, _, pred_intrinsics = pose_decoder(features)
            
            # Save predicted intrinsics
            all_pred_intrinsics.append(pred_intrinsics)
    
    # Concatenate all predictions
    all_pred_intrinsics = torch.cat(all_pred_intrinsics, dim=0)
      # Compute error metrics
    print(f"\nComputing error metrics for {all_pred_intrinsics.shape[0]} predictions...")
    error_metrics = compute_intrinsic_errors(all_pred_intrinsics, gt_intrinsics, 
                                            scale_invariant=scale_invariant,
                                            find_optimal_scale=find_optimal_scale)
    
    # Create visualization directory
    vis_dir = os.path.join(os.path.dirname(load_weights_folder), "intrinsics_vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize errors
    prefix = f"{dataset_name}_{pose_seq}_"
    visualize_intrinsics_errors(error_metrics, vis_dir, prefix)
    
    # Calculate mean predicted intrinsics for better comparison
    mean_pred_intrinsics = torch.mean(all_pred_intrinsics, dim=0).cpu().numpy()
    
    # Print the mean predicted intrinsics and ground truth for comparison
    print("\nMean Predicted Intrinsics:")
    print(mean_pred_intrinsics)
    print("\nGround Truth Intrinsics:")
    print(gt_intrinsics)
      # Print summary of errors
    print("\nIntrinsics Error Summary:")
    print(f"Focal Length X: {error_metrics['fx_abs_error_mean']:.2f} ± {error_metrics['fx_abs_error_std']:.2f} pixels "
          f"({error_metrics['fx_rel_error_mean']:.2f}% relative)")
    print(f"Focal Length Y: {error_metrics['fy_abs_error_mean']:.2f} ± {error_metrics['fy_abs_error_std']:.2f} pixels "
          f"({error_metrics['fy_rel_error_mean']:.2f}% relative)")
    print(f"Principal Point X: {error_metrics['cx_abs_error_mean']:.2f} ± {error_metrics['cx_abs_error_std']:.2f} pixels "
          f"({error_metrics['cx_rel_error_mean']:.2f}% relative)")
    print(f"Principal Point Y: {error_metrics['cy_abs_error_mean']:.2f} ± {error_metrics['cy_abs_error_std']:.2f} pixels "
          f"({error_metrics['cy_rel_error_mean']:.2f}% relative)")
    print(f"Matrix Frobenius Norm: {error_metrics['matrix_frobenius_norm_mean']:.2f} ± "
          f"{error_metrics['matrix_frobenius_norm_std']:.2f}")
    
    # Print scale-invariant metrics summary if available
    if scale_invariant and 'scale_factor_mean' in error_metrics:
        print("\nScale-Invariant Metrics:")
        print(f"Mean Scale Factor: {error_metrics['scale_factor_mean']:.4f} ± {error_metrics['scale_factor_std']:.4f}")
        print(f"Scale-Adjusted Focal Length X: {error_metrics['scale_adjusted_fx_abs_error_mean']:.2f} ± "
              f"{error_metrics['scale_adjusted_fx_abs_error_std']:.2f} pixels "
              f"({error_metrics['scale_adjusted_fx_rel_error_mean']:.2f}% relative)")
        print(f"Scale-Adjusted Focal Length Y: {error_metrics['scale_adjusted_fy_abs_error_mean']:.2f} ± "
              f"{error_metrics['scale_adjusted_fy_abs_error_std']:.2f} pixels "
              f"({error_metrics['scale_adjusted_fy_rel_error_mean']:.2f}% relative)")
        print(f"Scale-Adjusted Principal Point X: {error_metrics['scale_adjusted_cx_abs_error_mean']:.2f} ± "
              f"{error_metrics['scale_adjusted_cx_abs_error_std']:.2f} pixels "
              f"({error_metrics['scale_adjusted_cx_rel_error_mean']:.2f}% relative)")
        print(f"Scale-Adjusted Principal Point Y: {error_metrics['scale_adjusted_cy_abs_error_mean']:.2f} ± "
              f"{error_metrics['scale_adjusted_cy_abs_error_std']:.2f} pixels "
              f"({error_metrics['scale_adjusted_cy_rel_error_mean']:.2f}% relative)")
        print(f"Scale-Adjusted Matrix Frobenius Norm: {error_metrics['scale_adjusted_matrix_frobenius_norm_mean']:.2f} ± "
              f"{error_metrics['scale_adjusted_matrix_frobenius_norm_std']:.2f}")
    
    # Save summary to file
    summary_file = os.path.join(vis_dir, f"{prefix}summary.txt")
    with open(summary_file, "w") as f:
        f.write("Intrinsics Error Summary\n")
        f.write("=======================\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Pose Sequence: {pose_seq}\n")
        f.write(f"Model: {os.path.basename(load_weights_folder)}\n\n")
        
        f.write("Mean Predicted Intrinsics:\n")
        f.write(f"{mean_pred_intrinsics}\n\n")
        
        f.write("Ground Truth Intrinsics:\n")
        f.write(f"{gt_intrinsics}\n\n")
        
        f.write("Standard Error Metrics:\n")
        f.write(f"Focal Length X: {error_metrics['fx_abs_error_mean']:.2f} ± {error_metrics['fx_abs_error_std']:.2f} pixels "
                f"({error_metrics['fx_rel_error_mean']:.2f}% relative)\n")
        f.write(f"Focal Length Y: {error_metrics['fy_abs_error_mean']:.2f} ± {error_metrics['fy_abs_error_std']:.2f} pixels "
                f"({error_metrics['fy_rel_error_mean']:.2f}% relative)\n")
        f.write(f"Principal Point X: {error_metrics['cx_abs_error_mean']:.2f} ± {error_metrics['cx_abs_error_std']:.2f} pixels "
                f"({error_metrics['cx_rel_error_mean']:.2f}% relative)\n")
        f.write(f"Principal Point Y: {error_metrics['cy_abs_error_mean']:.2f} ± {error_metrics['cy_abs_error_std']:.2f} pixels "
                f"({error_metrics['cy_rel_error_mean']:.2f}% relative)\n")
        f.write(f"Matrix Frobenius Norm: {error_metrics['matrix_frobenius_norm_mean']:.2f} ± "
                f"{error_metrics['matrix_frobenius_norm_std']:.2f}\n")
        
        # Include scale-invariant metrics in the file if available
        if scale_invariant and 'scale_factor_mean' in error_metrics:
            f.write("\nScale-Invariant Metrics:\n")
            f.write(f"Mean Scale Factor: {error_metrics['scale_factor_mean']:.4f} ± {error_metrics['scale_factor_std']:.4f}\n")
            f.write(f"Scale-Adjusted Focal Length X: {error_metrics['scale_adjusted_fx_abs_error_mean']:.2f} ± "
                   f"{error_metrics['scale_adjusted_fx_abs_error_std']:.2f} pixels "
                   f"({error_metrics['scale_adjusted_fx_rel_error_mean']:.2f}% relative)\n")
            f.write(f"Scale-Adjusted Focal Length Y: {error_metrics['scale_adjusted_fy_abs_error_mean']:.2f} ± "
                   f"{error_metrics['scale_adjusted_fy_abs_error_std']:.2f} pixels "
                   f"({error_metrics['scale_adjusted_fy_rel_error_mean']:.2f}% relative)\n")
            f.write(f"Scale-Adjusted Principal Point X: {error_metrics['scale_adjusted_cx_abs_error_mean']:.2f} ± "
                   f"{error_metrics['scale_adjusted_cx_abs_error_std']:.2f} pixels "
                   f"({error_metrics['scale_adjusted_cx_rel_error_mean']:.2f}% relative)\n")
            f.write(f"Scale-Adjusted Principal Point Y: {error_metrics['scale_adjusted_cy_abs_error_mean']:.2f} ± "
                   f"{error_metrics['scale_adjusted_cy_abs_error_std']:.2f} pixels "
                   f"({error_metrics['scale_adjusted_cy_rel_error_mean']:.2f}% relative)\n")
            f.write(f"Scale-Adjusted Matrix Frobenius Norm: {error_metrics['scale_adjusted_matrix_frobenius_norm_mean']:.2f} ± "
                   f"{error_metrics['scale_adjusted_matrix_frobenius_norm_std']:.2f}\n")
    
    print(f"Summary saved to {summary_file}")
    
    # Return error metrics for further processing
    return error_metrics

if __name__ == "__main__":    parser = argparse.ArgumentParser(description="Evaluate camera intrinsics prediction")
    parser.add_argument('--dataset', type=str, choices=['SCARED', 'SyntheticColon', 'C3VD'], default='SCARED',
                        help='Dataset to evaluate on (SCARED, SyntheticColon, or C3VD)')
    parser.add_argument('--model_dir', type=str, default='logs/base_model_2/models',
                        help='Directory containing the model weights')
    parser.add_argument('--weight_idx', type=int, default=None, 
                        help='Specific model weight index to evaluate. If not provided, all weights are evaluated.')
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode (limited evaluation)')
    parser.add_argument('--scale_invariant', action='store_true', default=True,
                        help='Compute scale-invariant metrics (important for monocular depth estimation)')
    parser.add_argument('--no_scale_invariant', dest='scale_invariant', action='store_false',
                        help='Disable scale-invariant metrics computation')
    parser.add_argument('--find_optimal_scale', action='store_true', default=True,
                        help='Find optimal scale factor between predicted and ground truth intrinsics')
    parser.add_argument('--no_find_optimal_scale', dest='find_optimal_scale', action='store_false',
                        help='Disable finding optimal scale factor')
    args = parser.parse_args()
    
    # Import options from the appropriate experiment
    from exps.attn_encoder_dora.options_attn_encoder import AttnEncoderOpt
    
    # Ensure the options have intrinsics prediction enabled
    AttnEncoderOpt.learn_intrinsics = True
    
    DEBUG = args.debug
    model_dir = args.model_dir
    result_file = os.path.join(model_dir, 'intrinsics_results.txt')
    
    # Find all model weight folders or use a specific one
    if args.weight_idx is not None:
        weight_folders = [os.path.join(model_dir, f'weights_{args.weight_idx}')]
        weight_indices = [args.weight_idx]
    else:
        weight_folders = glob.glob(os.path.join(model_dir, 'weights_*'))
        weight_indices = [int(os.path.basename(folder).split('_')[1]) for folder in weight_folders]
        if DEBUG:
            weight_indices = weight_indices[:2]
            weight_folders = weight_folders[:2]
    
    # Sort weights by index
    sorted_idx = np.argsort(weight_indices)
    weight_indices = [weight_indices[i] for i in sorted_idx]
    weight_folders = [weight_folders[i] for i in sorted_idx]
    
    print(f"Found {len(weight_folders)} weight folders to evaluate")
    
    # Open results file
    with open(result_file, 'w') as f_result:
        f_result.write('Intrinsics Evaluation Results:\n')
        
        if args.dataset == 'SCARED':
            # -- SCARED Dataset --
            print("Evaluating on SCARED dataset")
            
            # Find best model for sequence 1
            min_error = np.inf
            best_i = 0
            
            for i, folder in zip(weight_indices, weight_folders):
                print(f"Evaluating model {i} on SCARED Sequence 1")
                
                try:                    error_metrics = evaluate_intrinsics(
                        AttnEncoderOpt, 
                        os.path.join(ds_base, 'SCARED_Images_Resized'),
                        folder,
                        dataset_name="SCARED",
                        pose_seq=1,
                        scale_invariant=args.scale_invariant,
                        find_optimal_scale=args.find_optimal_scale
                    )
                    
                    # Use matrix Frobenius norm as the main error metric
                    current_error = error_metrics['matrix_frobenius_norm_mean']
                    
                    if current_error < min_error:
                        min_error = current_error
                        best_i = i
                          # Write per-model results
                    f_result.write(f"Model {i} - Sequence 1 - Matrix Norm: {current_error:.4f}\n")
                    f_result.write(f"  fx error: {error_metrics['fx_abs_error_mean']:.2f} pixels ({error_metrics['fx_rel_error_mean']:.2f}%)\n")
                    f_result.write(f"  fy error: {error_metrics['fy_abs_error_mean']:.2f} pixels ({error_metrics['fy_rel_error_mean']:.2f}%)\n")
                    f_result.write(f"  cx error: {error_metrics['cx_abs_error_mean']:.2f} pixels ({error_metrics['cx_rel_error_mean']:.2f}%)\n")
                    f_result.write(f"  cy error: {error_metrics['cy_abs_error_mean']:.2f} pixels ({error_metrics['cy_rel_error_mean']:.2f}%)\n")
                    
                    # Add scale-adjusted metrics if available
                    if args.scale_invariant and 'scale_factor_mean' in error_metrics:
                        f_result.write(f"  Scale factor: {error_metrics['scale_factor_mean']:.4f}\n")
                        f_result.write(f"  Scale-adjusted matrix norm: {error_metrics['scale_adjusted_matrix_frobenius_norm_mean']:.4f}\n")
                        f_result.write(f"  Scale-adjusted fx error: {error_metrics['scale_adjusted_fx_abs_error_mean']:.2f} pixels ({error_metrics['scale_adjusted_fx_rel_error_mean']:.2f}%)\n")
                        f_result.write(f"  Scale-adjusted fy error: {error_metrics['scale_adjusted_fy_abs_error_mean']:.2f} pixels ({error_metrics['scale_adjusted_fy_rel_error_mean']:.2f}%)\n")
                    f_result.write("\n")
                
                except Exception as e:
                    print(f"Error evaluating model {i}: {str(e)}")
                    f_result.write(f"Model {i} - Evaluation failed: {str(e)}\n\n")
            
            print(f"SCARED Sequence 1: Best Model @ {best_i} with Matrix Norm Error: {min_error:.4f}")
            f_result.write(f"SCARED Sequence 1: Best Model @ {best_i} with Matrix Norm Error: {min_error:.4f}\n\n")
            
            # Also evaluate on sequence 2 if not in debug mode
            if not DEBUG:
                min_error = np.inf
                best_i = 0
                
                for i, folder in zip(weight_indices, weight_folders):
                    print(f"Evaluating model {i} on SCARED Sequence 2")
                    
                    try:                        error_metrics = evaluate_intrinsics(
                            AttnEncoderOpt, 
                            os.path.join(ds_base, 'SCARED_Images_Resized'),
                            folder,
                            dataset_name="SCARED",
                            pose_seq=2,
                            scale_invariant=args.scale_invariant,
                            find_optimal_scale=args.find_optimal_scale
                        )
                        
                        current_error = error_metrics['matrix_frobenius_norm_mean']
                        
                        if current_error < min_error:
                            min_error = current_error
                            best_i = i
                            
                        # Write per-model results
                        f_result.write(f"Model {i} - Sequence 2 - Matrix Norm: {current_error:.4f}\n")
                        f_result.write(f"  fx error: {error_metrics['fx_abs_error_mean']:.2f} pixels ({error_metrics['fx_rel_error_mean']:.2f}%)\n")
                        f_result.write(f"  fy error: {error_metrics['fy_abs_error_mean']:.2f} pixels ({error_metrics['fy_rel_error_mean']:.2f}%)\n")
                        f_result.write(f"  cx error: {error_metrics['cx_abs_error_mean']:.2f} pixels ({error_metrics['cx_rel_error_mean']:.2f}%)\n")
                        f_result.write(f"  cy error: {error_metrics['cy_abs_error_mean']:.2f} pixels ({error_metrics['cy_rel_error_mean']:.2f}%)\n\n")
                    
                    except Exception as e:
                        print(f"Error evaluating model {i}: {str(e)}")
                        f_result.write(f"Model {i} - Evaluation failed: {str(e)}\n\n")
                
                print(f"SCARED Sequence 2: Best Model @ {best_i} with Matrix Norm Error: {min_error:.4f}")
                f_result.write(f"SCARED Sequence 2: Best Model @ {best_i} with Matrix Norm Error: {min_error:.4f}\n\n")
            
        elif args.dataset == 'SyntheticColon':
            # -- SyntheticColon Dataset --
            test_folders_file = os.path.join(ds_base, 'SyntheticColon_as_SCARED', 'splits', 'test_folders.txt')
            with open(test_folders_file, 'r') as f:
                test_folders = f.readlines()
            
            if DEBUG:
                test_folders = test_folders[:2]  # Limit to first two test folders in debug mode
            
            min_error = np.inf
            best_i = 0
            
            for i, folder in zip(weight_indices, weight_folders):
                print(f"Evaluating model {i} on SyntheticColon")
                
                all_errors = []
                all_metrics = {}
                
                for test_folder in tqdm(test_folders, desc=f"Model {i} - Test folders"):
                    full_test_folder = os.path.join(ds_base, 'SyntheticColon_as_SCARED', test_folder.strip())
                    try:                        error_metrics = evaluate_intrinsics(
                            AttnEncoderOpt, 
                            os.path.join(ds_base, 'SyntheticColon_as_SCARED'),
                            folder,
                            dataset_name="SyntheticColon",
                            filename_list_path=os.path.join(full_test_folder, 'traj_test.txt'),
                            scale_invariant=args.scale_invariant,
                            find_optimal_scale=args.find_optimal_scale
                        )
                        
                        all_errors.append(error_metrics['matrix_frobenius_norm_mean'])
                        
                        # Aggregate metrics
                        for key in error_metrics:
                            if key != 'errors':
                                if key not in all_metrics:
                                    all_metrics[key] = []
                                all_metrics[key].append(error_metrics[key])
                    except Exception as e:
                        print(f"Error evaluating folder {test_folder.strip()}: {str(e)}")
                
                # Calculate average errors across all test folders
                if all_errors:
                    avg_error = np.mean(all_errors)
                    
                    if avg_error < min_error:
                        min_error = avg_error
                        best_i = i
                    
                    # Write per-model results with average metrics
                    f_result.write(f"Model {i} - SyntheticColon - Avg Matrix Norm: {avg_error:.4f}\n")
                    f_result.write(f"  fx error: {np.mean(all_metrics['fx_abs_error_mean']):.2f} pixels ({np.mean(all_metrics['fx_rel_error_mean']):.2f}%)\n")
                    f_result.write(f"  fy error: {np.mean(all_metrics['fy_abs_error_mean']):.2f} pixels ({np.mean(all_metrics['fy_rel_error_mean']):.2f}%)\n")
                    f_result.write(f"  cx error: {np.mean(all_metrics['cx_abs_error_mean']):.2f} pixels ({np.mean(all_metrics['cx_rel_error_mean']):.2f}%)\n")
                    f_result.write(f"  cy error: {np.mean(all_metrics['cy_abs_error_mean']):.2f} pixels ({np.mean(all_metrics['cy_rel_error_mean']):.2f}%)\n\n")
            
            print(f"SyntheticColon: Best Model @ {best_i} with Matrix Norm Error: {min_error:.4f}")
            f_result.write(f"SyntheticColon: Best Model @ {best_i} with Matrix Norm Error: {min_error:.4f}\n\n")
            
        elif args.dataset == 'C3VD':
            # -- C3VD Dataset --
            test_folders_file = os.path.join(ds_base, 'C3VD_as_SCARED', 'splits', 'test_folders.txt')
            with open(test_folders_file, 'r') as f:
                test_folders = f.readlines()
            
            if DEBUG:
                test_folders = test_folders[:2]  # Limit to first two test folders in debug mode
            
            min_error = np.inf
            best_i = 0
            
            for i, folder in zip(weight_indices, weight_folders):
                print(f"Evaluating model {i} on C3VD")
                
                all_errors = []
                all_metrics = {}
                
                for test_folder in tqdm(test_folders, desc=f"Model {i} - Test folders"):
                    full_test_folder = os.path.join(ds_base, 'C3VD_as_SCARED', test_folder.strip())
                    try:                        error_metrics = evaluate_intrinsics(
                            AttnEncoderOpt, 
                            os.path.join(ds_base, 'C3VD_as_SCARED'),
                            folder,
                            dataset_name="C3VD",
                            filename_list_path=os.path.join(full_test_folder, 'traj_test.txt'),
                            scale_invariant=args.scale_invariant,
                            find_optimal_scale=args.find_optimal_scale
                        )
                        
                        all_errors.append(error_metrics['matrix_frobenius_norm_mean'])
                        
                        # Aggregate metrics
                        for key in error_metrics:
                            if key != 'errors':
                                if key not in all_metrics:
                                    all_metrics[key] = []
                                all_metrics[key].append(error_metrics[key])
                    except Exception as e:
                        print(f"Error evaluating folder {test_folder.strip()}: {str(e)}")
                
                # Calculate average errors across all test folders
                if all_errors:
                    avg_error = np.mean(all_errors)
                    
                    if avg_error < min_error:
                        min_error = avg_error
                        best_i = i
                    
                    # Write per-model results with average metrics
                    f_result.write(f"Model {i} - C3VD - Avg Matrix Norm: {avg_error:.4f}\n")
                    f_result.write(f"  fx error: {np.mean(all_metrics['fx_abs_error_mean']):.2f} pixels ({np.mean(all_metrics['fx_rel_error_mean']):.2f}%)\n")
                    f_result.write(f"  fy error: {np.mean(all_metrics['fy_abs_error_mean']):.2f} pixels ({np.mean(all_metrics['fy_rel_error_mean']):.2f}%)\n")
                    f_result.write(f"  cx error: {np.mean(all_metrics['cx_abs_error_mean']):.2f} pixels ({np.mean(all_metrics['cx_rel_error_mean']):.2f}%)\n")
                    f_result.write(f"  cy error: {np.mean(all_metrics['cy_abs_error_mean']):.2f} pixels ({np.mean(all_metrics['cy_rel_error_mean']):.2f}%)\n\n")
            
            print(f"C3VD: Best Model @ {best_i} with Matrix Norm Error: {min_error:.4f}")
            f_result.write(f"C3VD: Best Model @ {best_i} with Matrix Norm Error: {min_error:.4f}\n\n")
    
    print(f"\nResults written to {result_file}")
