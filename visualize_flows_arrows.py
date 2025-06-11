#!/usr/bin/env python3
"""
改进的光流和外观流可视化脚本 - 箭头阵列版本 (包含遮挡掩码)

用法:
python visualize_flows_arrows.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg --model_path path/to/models

新的可视化方式：
1. 光流 (Optical Flow) - 前向和反向光流使用不同颜色箭头阵列
2. 外观流 (Appearance Flow) - 使用彩色箭头阵列（RGB通道）
3. 遮挡掩码 (Occlusion Masks) - 后向遮挡掩码和双向遮挡图
4. 更清晰的向量场可视化
5. 基于trainer_abc实现的遮挡检测
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'DARES')))

# 导入必要的模块
from DARES.networks.resnet_encoder import AttentionalResnetEncoder
from DARES.networks.optical_flow_decoder import PositionDecoder
from DARES.networks.appearance_flow_decoder import TransformDecoder
from DARES.layers import SpatialTransformer, get_occu_mask_backward, get_occu_mask_bidirection


def load_image(img_path, height=256, width=320, device='cuda'):
    """加载并预处理图像"""
    # 读取图像
    img = Image.open(img_path).convert('RGB')
    
    # 调整大小
    img = img.resize((width, height), Image.BILINEAR)
    
    # 转换为tensor并归一化
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
    
    return img_tensor.to(device)


def load_models(model_path, device='cuda'):
    """加载预训练模型"""
    print(f"从 {model_path} 加载模型...")
    
    models = {}
    
    # 初始化编码器和解码器
    models["position_encoder"] = AttentionalResnetEncoder(
        num_layers=18, 
        pretrained=False, 
        num_input_images=2
    )
    
    models["transform_encoder"] = AttentionalResnetEncoder(
        num_layers=18, 
        pretrained=False, 
        num_input_images=2
    )
    
    models["position"] = PositionDecoder(
        models["position_encoder"].num_ch_enc, 
        scales=[0, 1, 2, 3]
    )
    
    models["transform"] = TransformDecoder(
        models["transform_encoder"].num_ch_enc, 
        scales=[0, 1, 2, 3]
    )
    
    # 加载预训练权重
    model_files = {
        "position_encoder": "position_encoder.pth",
        "position": "position.pth", 
        "transform_encoder": "transform_encoder.pth",
        "transform": "transform.pth"
    }
    
    for model_name, filename in model_files.items():
        model_file = os.path.join(model_path, filename)
        if os.path.exists(model_file):
            print(f"✓ 加载 {model_name}")
            state_dict = torch.load(model_file, map_location=device)
            models[model_name].load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠ 未找到 {model_file}")
    
    # 移动模型到设备并设置为评估模式
    for model in models.values():
        model.to(device)
        model.eval()
    
    return models


def create_occlusion_layers(height, width, device):
    """创建遮挡掩码计算层"""
    get_occu_mask_backward_layer = get_occu_mask_backward((height, width)).to(device)
    get_occu_mask_bidirection_layer = get_occu_mask_bidirection((height, width)).to(device)
    return get_occu_mask_backward_layer, get_occu_mask_bidirection_layer


def flow_to_color_hsv(flow_np):
    """将光流转换为HSV颜色可视化"""
    # flow_np: [2, H, W]
    magnitude = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    angle = np.arctan2(flow_np[1], flow_np[0])
    
    # 创建HSV图像
    hsv = np.zeros((flow_np.shape[1], flow_np.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 255  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = np.clip(magnitude * 255 / (np.max(magnitude) + 1e-8), 0, 255)  # Value
    
    # 转换为RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def draw_flow_arrows(ax, flow, step=16, scale=1.0, color='black', alpha=0.7, width=0.002):
    """在图像上绘制光流箭头"""
    H, W = flow.shape[1], flow.shape[2]
    
    # 创建采样网格
    Y, X = np.mgrid[step//2:H:step, step//2:W:step]
    
    # 采样光流
    U = flow[0][Y, X] * scale
    V = flow[1][Y, X] * scale
    
    # 计算幅度用于过滤
    magnitude = np.sqrt(U**2 + V**2)
    threshold = np.percentile(magnitude, 75)  # 只显示较大的光流
    
    # 绘制箭头
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if magnitude[i, j] > threshold * 0.3:  # 过滤掉太小的光流
                ax.arrow(X[i, j], Y[i, j], U[i, j], V[i, j], 
                        head_width=3, head_length=3, fc=color, ec=color, 
                        alpha=alpha, width=width, length_includes_head=True)


def draw_appearance_flow_arrows(ax, app_flow, step=16, scale=20.0, alpha=0.8):
    """绘制外观流的彩色箭头"""
    H, W = app_flow.shape[0], app_flow.shape[1]
    
    # 创建采样网格
    Y, X = np.mgrid[step//2:H:step, step//2:W:step]
    
    # 采样外观流 (3通道)
    R = app_flow[Y, X, 0] * scale  # Red channel
    G = app_flow[Y, X, 1] * scale  # Green channel  
    B = app_flow[Y, X, 2] * scale  # Blue channel
    
    # 计算合成的幅度和方向
    magnitude = np.sqrt(R**2 + G**2 + B**2)
    threshold = np.percentile(magnitude, 60)
    
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if magnitude[i, j] > threshold * 0.2:
                # 使用RGB值作为颜色，归一化到[0,1]
                r_norm = np.clip((app_flow[Y[i, j], X[i, j], 0] + 1) / 2, 0, 1)
                g_norm = np.clip((app_flow[Y[i, j], X[i, j], 1] + 1) / 2, 0, 1)
                b_norm = np.clip((app_flow[Y[i, j], X[i, j], 2] + 1) / 2, 0, 1)
                
                color = (r_norm, g_norm, b_norm)
                
                # 计算箭头方向（使用前两个通道）
                dx = R[i, j]
                dy = G[i, j]
                
                if abs(dx) > 0.1 or abs(dy) > 0.1:  # 只绘制有意义的变化
                    ax.arrow(X[i, j], Y[i, j], dx, dy,
                            head_width=4, head_length=4, fc=color, ec=color,
                            alpha=alpha, width=0.8, length_includes_head=True)


def create_quiver_plot(flow, title, step=12, scale=15):
    """创建向量场图"""
    H, W = flow.shape[1], flow.shape[2]
    
    # 创建采样网格
    Y, X = np.mgrid[0:H:step, 0:W:step]
    
    # 采样光流
    U = flow[0][Y, X]
    V = flow[1][Y, X]
    
    # 计算幅度
    magnitude = np.sqrt(U**2 + V**2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制向量场
    q = ax.quiver(X, Y, U, V, magnitude, scale=scale, alpha=0.8, cmap='viridis')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # 翻转Y轴
    ax.set_aspect('equal')
    
    # 添加颜色条
    plt.colorbar(q, ax=ax, label='Flow Magnitude')
    
    return fig


def visualize_occlusion_maps(ax, occu_mask, occu_map, title):
    """可视化遮挡掩码和遮挡图"""
    # 创建子图
    if hasattr(ax, '__len__'):  # 如果ax是数组
        # 遮挡掩码 (二值化)
        im1 = ax[0].imshow(occu_mask, cmap='gray', vmin=0, vmax=1)
        ax[0].set_title(f'{title} - Mask (Binary)', fontsize=10)
        ax[0].axis('off')
        
        # 遮挡图 (连续值)
        im2 = ax[1].imshow(occu_map, cmap='viridis', vmin=0, vmax=1)
        ax[1].set_title(f'{title} - Map (Continuous)', fontsize=10)
        ax[1].axis('off')
        
        return im1, im2
    else:
        # 只显示掩码
        im = ax.imshow(occu_mask, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        return im


def predict_and_visualize_arrows(img1_path, img2_path, model_path, save_dir, 
                                height=256, width=320, device='cuda'):
    """预测并使用箭头可视化光流和外观流，包括遮挡掩码"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载图像
    print("加载图像...")
    img1_tensor = load_image(img1_path, height, width, device)
    img2_tensor = load_image(img2_path, height, width, device)
    
    # 加载模型
    models = load_models(model_path, device)
    
    # 初始化空间变换器和遮挡掩码层
    spatial_transform = SpatialTransformer((height, width)).to(device)
    get_occu_mask_backward_layer, get_occu_mask_bidirection_layer = create_occlusion_layers(
        height, width, device)
    
    print("开始推理...")
    with torch.no_grad():
        # === 计算光流 (前向和反向) ===
        print("  计算光流...")
        # 前向光流 (img2 -> img1)
        inputs_forward = torch.cat([img2_tensor, img1_tensor], 1)  # [B, 6, H, W]
        position_features_forward = models["position_encoder"](inputs_forward)
        position_outputs_forward = models["position"](position_features_forward)
        optical_flow_forward = position_outputs_forward[("position", 0)]  # [B, 2, H, W]
        optical_flow_forward_high = F.interpolate(
            optical_flow_forward, [height, width], mode="bilinear", align_corners=True
        )
        
        # 反向光流 (img1 -> img2)  
        inputs_backward = torch.cat([img1_tensor, img2_tensor], 1)  # [B, 6, H, W]
        position_features_backward = models["position_encoder"](inputs_backward)
        position_outputs_backward = models["position"](position_features_backward)
        optical_flow_backward = position_outputs_backward[("position", 0)]  # [B, 2, H, W]
        optical_flow_backward_high = F.interpolate(
            optical_flow_backward, [height, width], mode="bilinear", align_corners=True
        )
        
        # === 计算遮挡掩码 ===
        print("  计算遮挡掩码...")
        # 后向遮挡掩码 (基于反向光流)
        occu_mask_backward, occu_map_backward = get_occu_mask_backward_layer(
            optical_flow_backward_high, th=0.95)
        
        # 双向遮挡图 (基于前向和反向光流)
        occu_map_bidirection = get_occu_mask_bidirection_layer(
            optical_flow_forward_high, optical_flow_backward_high, scale=0.01, bias=0.5)
        
        # === 图像配准 ===
        print("  图像配准...")
        registered_img = spatial_transform(img2_tensor, optical_flow_forward_high)
        
        # === 计算外观流 ===  
        print("  计算外观流...")
        # 准备输入 (配准后的图像 + 参考图像)
        transform_input = torch.cat([registered_img, img1_tensor], 1)  # [B, 6, H, W]
        
        # 外观流编码和解码
        transform_features = models["transform_encoder"](transform_input)
        transform_outputs = models["transform"](transform_features)
        
        # 获取最高分辨率的外观流
        appearance_flow = transform_outputs[("transform", 0)]  # [B, 3, H, W]
        appearance_flow_high = F.interpolate(
            appearance_flow, [height, width], mode="bilinear", align_corners=True
        )
        
        # 生成精细化结果 (使用遮挡掩码)
        refined_img = (appearance_flow_high * occu_mask_backward.detach() + img1_tensor)
        refined_img = torch.clamp(refined_img, 0.0, 1.0)
    
    print("生成箭头可视化...")
    
    # 转换为numpy用于可视化
    img1_np = img1_tensor[0].cpu().permute(1, 2, 0).numpy()
    img2_np = img2_tensor[0].cpu().permute(1, 2, 0).numpy()
    registered_np = registered_img[0].cpu().permute(1, 2, 0).numpy()
    refined_np = refined_img[0].cpu().permute(1, 2, 0).numpy()
    
    optical_flow_forward_np = optical_flow_forward_high[0].cpu().numpy()  # [2, H, W]
    optical_flow_backward_np = optical_flow_backward_high[0].cpu().numpy()  # [2, H, W]
    appearance_flow_np = appearance_flow_high[0].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
    
    # 遮挡掩码和遮挡图
    occu_mask_backward_np = occu_mask_backward[0, 0].cpu().numpy()  # [H, W]
    occu_map_backward_np = occu_map_backward[0, 0].cpu().numpy()  # [H, W]
    occu_map_bidirection_np = occu_map_bidirection[0, 0].cpu().numpy()  # [H, W]
    
    # === 创建增强的箭头可视化 (包含遮挡信息) ===
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    
    # 第一行: 原始图像和基本光流
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title('Image 1 (Reference)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title('Image 2 (Target)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 传统光流颜色可视化
    flow_color = flow_to_color_hsv(optical_flow_forward_np)
    axes[0, 2].imshow(flow_color)
    axes[0, 2].set_title('Forward Optical Flow (HSV)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 第二行: 光流箭头可视化
    axes[1, 0].imshow(img1_np)
    draw_flow_arrows(axes[1, 0], optical_flow_forward_np, step=20, scale=3.0, 
                    color='red', alpha=0.8, width=0.5)
    axes[1, 0].set_title('Forward Flow Arrows', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img1_np)
    draw_flow_arrows(axes[1, 1], optical_flow_backward_np, step=20, scale=3.0, 
                    color='blue', alpha=0.8, width=0.5)
    axes[1, 1].set_title('Backward Flow Arrows', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.clip(registered_np, 0, 1))
    axes[1, 2].set_title('Registered Image 2→1', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # 第三行: 遮挡掩码可视化
    im1 = axes[2, 0].imshow(occu_mask_backward_np, cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('Backward Occlusion Mask', fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(im1, ax=axes[2, 0], shrink=0.8)
    
    im2 = axes[2, 1].imshow(occu_map_backward_np, cmap='viridis', vmin=0, vmax=1)
    axes[2, 1].set_title('Backward Occlusion Map', fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')
    plt.colorbar(im2, ax=axes[2, 1], shrink=0.8)
    
    # 双向遮挡图的范数可视化
    bidirection_norm = np.linalg.norm(occu_map_bidirection_np, axis=None) if occu_map_bidirection_np.ndim > 2 else occu_map_bidirection_np
    im3 = axes[2, 2].imshow(bidirection_norm, cmap='plasma')
    axes[2, 2].set_title('Bidirection Occlusion Map', fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')
    plt.colorbar(im3, ax=axes[2, 2], shrink=0.8)
    
    # 第四行: 外观流和最终结果
    axes[3, 0].imshow(img1_np)
    draw_appearance_flow_arrows(axes[3, 0], appearance_flow_np, step=18, scale=30.0)
    axes[3, 0].set_title('Appearance Flow Arrows', fontsize=12, fontweight='bold')
    axes[3, 0].axis('off')
    
    # 外观流传统可视化
    appearance_vis = (appearance_flow_np + 1.0) / 2.0
    axes[3, 1].imshow(np.clip(appearance_vis, 0, 1))
    axes[3, 1].set_title('Appearance Flow (RGB)', fontsize=12, fontweight='bold')
    axes[3, 1].axis('off')
    
    axes[3, 2].imshow(np.clip(refined_np, 0, 1))
    axes[3, 2].set_title('Final Refined Result', fontsize=12, fontweight='bold')
    axes[3, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存完整可视化
    output_path = os.path.join(save_dir, 'flow_arrows_with_occlusion_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存增强箭头可视化: {output_path}")
    
    plt.show()
    
    # === 创建单独的向量场图 ===
    print("生成向量场图...")
    
    # 前向光流向量场
    fig_quiver1 = create_quiver_plot(optical_flow_forward_np, 'Forward Optical Flow Vector Field', step=16, scale=20)
    fig_quiver1.savefig(os.path.join(save_dir, 'forward_optical_flow_quiver.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_quiver1)
    
    # 反向光流向量场
    fig_quiver1b = create_quiver_plot(optical_flow_backward_np, 'Backward Optical Flow Vector Field', step=16, scale=20)
    fig_quiver1b.savefig(os.path.join(save_dir, 'backward_optical_flow_quiver.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_quiver1b)
    
    # 外观流向量场 (使用前两个通道)
    app_flow_2d = np.stack([appearance_flow_np[:,:,0], appearance_flow_np[:,:,1]], axis=0)
    fig_quiver2 = create_quiver_plot(app_flow_2d, 'Appearance Flow Vector Field (RG channels)', step=16, scale=30)
    fig_quiver2.savefig(os.path.join(save_dir, 'appearance_flow_quiver.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_quiver2)
    
    # === 创建高密度箭头图 ===
    print("生成高密度箭头图...")
    
    # 高密度前向光流箭头
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img1_np, alpha=0.7)
    draw_flow_arrows(ax, optical_flow_forward_np, step=12, scale=4.0, 
                    color='yellow', alpha=0.9, width=0.8)
    ax.set_title('High Density Forward Optical Flow Arrows', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.savefig(os.path.join(save_dir, 'forward_optical_flow_dense_arrows.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 高密度反向光流箭头
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img1_np, alpha=0.7)
    draw_flow_arrows(ax, optical_flow_backward_np, step=12, scale=4.0, 
                    color='cyan', alpha=0.9, width=0.8)
    ax.set_title('High Density Backward Optical Flow Arrows', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.savefig(os.path.join(save_dir, 'backward_optical_flow_dense_arrows.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 高密度外观流箭头
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img1_np, alpha=0.7)
    draw_appearance_flow_arrows(ax, appearance_flow_np, step=12, scale=40.0, alpha=1.0)
    ax.set_title('High Density Appearance Flow Arrows (RGB colored)', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.savefig(os.path.join(save_dir, 'appearance_flow_dense_arrows.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === 保存其他结果 ===
    print("保存其他结果...")
    
    # 原始图像
    Image.fromarray((img1_np * 255).astype(np.uint8)).save(
        os.path.join(save_dir, 'img1_reference.png'))
    Image.fromarray((img2_np * 255).astype(np.uint8)).save(
        os.path.join(save_dir, 'img2_target.png'))
    
    # 配准图像
    Image.fromarray((np.clip(registered_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(save_dir, 'registered_image.png'))
    
    # 精细化结果
    Image.fromarray((np.clip(refined_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(save_dir, 'refined_result.png'))
    
    print(f"✓ 所有结果已保存到: {save_dir}")
    
    # 打印统计信息
    flow_forward_magnitude = np.sqrt(optical_flow_forward_np[0]**2 + optical_flow_forward_np[1]**2)
    flow_backward_magnitude = np.sqrt(optical_flow_backward_np[0]**2 + optical_flow_backward_np[1]**2)
    app_magnitude = np.sqrt(np.sum(appearance_flow_np**2, axis=2))
    
    print(f"\n=== 统计信息 ===")
    print(f"前向光流幅度: 最小={flow_forward_magnitude.min():.3f}, 最大={flow_forward_magnitude.max():.3f}, 平均={flow_forward_magnitude.mean():.3f}")
    print(f"反向光流幅度: 最小={flow_backward_magnitude.min():.3f}, 最大={flow_backward_magnitude.max():.3f}, 平均={flow_backward_magnitude.mean():.3f}")
    print(f"外观流幅度: 最小={app_magnitude.min():.3f}, 最大={app_magnitude.max():.3f}, 平均={app_magnitude.mean():.3f}")
    print(f"后向遮挡掩码覆盖率: {occu_mask_backward_np.mean():.3f}")
    print(f"后向遮挡图平均值: {occu_map_backward_np.mean():.3f}")
    print(f"配准误差 (MSE): {np.mean((img1_np - np.clip(registered_np, 0, 1))**2):.6f}")
    print(f"精细化误差 (MSE): {np.mean((img1_np - np.clip(refined_np, 0, 1))**2):.6f}")


def main():
    parser = argparse.ArgumentParser(description='箭头阵列可视化光流和外观流')
    parser.add_argument('--img1', required=True, help='第一张图片路径')
    parser.add_argument('--img2', required=True, help='第二张图片路径')
    parser.add_argument('--model_path', required=True, help='模型权重文件夹路径')
    parser.add_argument('--height', type=int, default=256, help='图片高度')
    parser.add_argument('--width', type=int, default=320, help='图片宽度')
    parser.add_argument('--save_dir', default='./flow_arrows_visualization', help='结果保存目录')
    parser.add_argument('--device', default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.img1):
        raise FileNotFoundError(f"图片1不存在: {args.img1}")
    if not os.path.exists(args.img2):
        raise FileNotFoundError(f"图片2不存在: {args.img2}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型路径不存在: {args.model_path}")
    
    print("=" * 60)
    print("箭头阵列可视化光流和外观流")
    print("=" * 60)
    
    predict_and_visualize_arrows(
        img1_path=args.img1,
        img2_path=args.img2,
        model_path=args.model_path,
        save_dir=args.save_dir,
        height=args.height,
        width=args.width,
        device=args.device
    )
    
    print("=" * 60)
    print("完成! 检查输出目录以查看箭头阵列可视化结果")
    print("=" * 60)


if __name__ == "__main__":
    main()
