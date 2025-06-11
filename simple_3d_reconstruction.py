#!/usr/bin/env python3
"""
Simple 3D reconstruction script using DARES PEFT depth model
Based on the evaluate_3d_reconstruction.py script from DARES
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

# Add DARES path
import sys
sys.path.append("/mnt/c/Users/14152/Desktop/new_code")
sys.path.append("/mnt/c/Users/14152/Desktop/new_code/DARES")

from networks.dares_peft import DARES
from layers import disp_to_depth


def load_image(image_path, target_size=(256, 320)):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize to target size
    image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # Convert to tensor and normalize
    image_array = np.array(image) / 255.0
    image_tensor = torch.from_numpy(image_array).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    return image_tensor, original_size, np.array(image)


def load_dares_model(model_path, device='cuda'):
    """Load DARES PEFT model"""
    print(f"Loading DARES PEFT model from {model_path}")
    
    # Create model
    model = DARES(
        r=[14,14,12,12,10,10,8,8,8,8,8,8], 
        target_modules=['query', 'value'],
        use_dora=True,
        full_finetune=False
    )
    
    # Load weights
    if os.path.isdir(model_path):
        # Load from directory (PEFT format)
        try:
            model.load_parameters(model_path)
        except:
            # Try alternative loading method
            model_file = os.path.join(model_path, "depth_model.pth")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"No valid model found in {model_path}")
    else:
        # Load from single file
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def predict_depth(model, image_tensor, min_depth=1e-3, max_depth=150):
    """Predict depth from image"""
    with torch.no_grad():
        image_tensor = image_tensor.cuda()
        
        # Forward pass
        output = model(image_tensor)
        
        # Convert disparity to depth
        pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], min_depth, max_depth)
        
        return pred_disp.cpu().numpy(), pred_depth.cpu().numpy()


def render_depth(depth_array, colormap_name="plasma"):
    """Render depth as colored image"""
    depth_array = depth_array.squeeze()
    
    # Normalize depth values
    min_val, max_val = depth_array.min(), depth_array.max()
    normalized_depth = (depth_array - min_val) / (max_val - min_val)
    
    # Apply colormap
    import matplotlib.cm as cm
    colormap = cm.get_cmap(colormap_name)
    colored_depth = colormap(normalized_depth)
    
    # Convert to PIL Image
    colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_depth)


def create_intrinsic_matrix(image_width, image_height, fov_degrees=60):
    """Create a simple intrinsic camera matrix"""
    # Convert FOV to focal length
    fov_rad = np.radians(fov_degrees)
    focal_length = image_width / (2 * np.tan(fov_rad / 2))
    
    # Create intrinsic matrix
    K = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return K


def reconstruct_pointcloud(rgb_image, depth_array, camera_intrinsics, depth_scale=1.0, depth_range=(0.1, 50.0)):
    """Create 3D point cloud from RGB image and depth"""
    height, width = depth_array.shape
    
    # Ensure RGB image matches depth dimensions
    if rgb_image.shape[:2] != (height, width):
        rgb_image = cv2.resize(rgb_image, (width, height))
    
    # Scale and clamp depth values
    depth_array = depth_array * depth_scale
    depth_array = np.clip(depth_array, depth_range[0], depth_range[1])
    
    # Create Open3D images
    rgb_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_array.astype(np.float32))
    
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, 
        depth_o3d, 
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=depth_range[1]
    )
    
    # Create camera intrinsic
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        camera_intrinsics[0, 0], camera_intrinsics[1, 1],
        camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    )
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )
    
    return pcd


def save_reconstruction_results(rgb_image, depth_array, pcd, output_dir, basename):
    """Save reconstruction results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save RGB image
    rgb_path = os.path.join(output_dir, f"{basename}_rgb.png")
    Image.fromarray(rgb_image).save(rgb_path)
    
    # Save depth visualization
    depth_vis = render_depth(depth_array)
    depth_path = os.path.join(output_dir, f"{basename}_depth.png")
    depth_vis.save(depth_path)
    
    # Save raw depth as NPY
    depth_npy_path = os.path.join(output_dir, f"{basename}_depth.npy")
    np.save(depth_npy_path, depth_array)
    
    # Save point cloud
    pcd_path = os.path.join(output_dir, f"{basename}_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    
    print(f"Results saved to {output_dir}:")
    print(f"  - RGB image: {rgb_path}")
    print(f"  - Depth visualization: {depth_path}")
    print(f"  - Raw depth: {depth_npy_path}")
    print(f"  - Point cloud: {pcd_path}")


def visualize_3d(pcd, window_name="3D Reconstruction"):
    """Visualize 3D point cloud"""
    print("Displaying 3D reconstruction. Close the window to continue...")
    
    # Optional: downsample for better performance
    if len(pcd.points) > 100000:
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    # Remove outliers
    pcd, _ = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
    
    # Estimate normals for better visualization
    pcd.estimate_normals()
    
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd], 
        window_name=window_name,
        point_show_normal=False
    )


def main():
    parser = argparse.ArgumentParser(description='Simple 3D reconstruction using DARES PEFT model')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', required=True, help='Path to DARES PEFT model weights')
    parser.add_argument('--output', default='./reconstruction_output', help='Output directory')
    parser.add_argument('--min_depth', type=float, default=0.1, help='Minimum depth value')
    parser.add_argument('--max_depth', type=float, default=50.0, help='Maximum depth value')
    parser.add_argument('--depth_scale', type=float, default=1.0, help='Depth scaling factor')
    parser.add_argument('--fov', type=float, default=60.0, help='Camera field of view in degrees')
    parser.add_argument('--no_visualization', action='store_true', help='Skip 3D visualization')
    parser.add_argument('--device', default='cuda', help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=== Simple 3D Reconstruction with DARES PEFT ===")
    print(f"Input image: {args.image}")
    print(f"Model path: {args.model}")
    print(f"Output directory: {args.output}")
    print(f"Device: {args.device}")
    
    # Load and preprocess image
    print("\n1. Loading and preprocessing image...")
    image_tensor, original_size, rgb_array = load_image(args.image)
    print(f"Image size: {image_tensor.shape}")
    
    # Load model
    print("\n2. Loading DARES PEFT model...")
    model = load_dares_model(args.model, args.device)
    
    # Predict depth
    print("\n3. Predicting depth...")
    pred_disp, pred_depth = predict_depth(
        model, image_tensor, 
        min_depth=args.min_depth, 
        max_depth=args.max_depth
    )
    depth_array = pred_depth.squeeze()
    
    print(f"Predicted depth range: {depth_array.min():.3f} - {depth_array.max():.3f}")
    
    # Create camera intrinsics
    print("\n4. Creating camera intrinsics...")
    height, width = depth_array.shape
    camera_intrinsics = create_intrinsic_matrix(width, height, args.fov)
    print(f"Camera intrinsics:\n{camera_intrinsics}")
    
    # Reconstruct 3D point cloud
    print("\n5. Reconstructing 3D point cloud...")
    pcd = reconstruct_pointcloud(
        rgb_array, depth_array, camera_intrinsics,
        depth_scale=args.depth_scale,
        depth_range=(args.min_depth, args.max_depth)
    )
    print(f"Point cloud contains {len(pcd.points)} points")
    
    # Save results
    print("\n6. Saving results...")
    basename = os.path.splitext(os.path.basename(args.image))[0]
    save_reconstruction_results(rgb_array, depth_array, pcd, args.output, basename)
    
    # Visualize 3D reconstruction
    if not args.no_visualization:
        print("\n7. Visualizing 3D reconstruction...")
        visualize_3d(pcd, f"3D Reconstruction - {basename}")
    
    print("\n✅ 3D reconstruction completed successfully!")


if __name__ == "__main__":
    main()
