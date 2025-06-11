"""
Flow-based Pose Decoder
基于光流和外观流信息直接进行姿态估计的解码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import ConvBlock, Conv3x3


class FlowBasedPoseDecoder(nn.Module):
    """
    基于optical flow (2通道) 和 appearance flow (3通道) 的pose预测器
    
    输入: 
    - optical_flow: [B, 2, H, W] 光流信息
    - appearance_flow: [B, 3, H, W] 外观流信息
    
    输出:
    - axisangle: [B, num_frames, 1, 3] 旋转
    - translation: [B, num_frames, 1, 3] 平移
    """
    
    def __init__(self, 
                 input_height=256, 
                 input_width=320,
                 num_frames_to_predict_for=1,
                 predict_intrinsics=False,
                 image_width=None, 
                 image_height=None,
                 auto_scale=True):
        super(FlowBasedPoseDecoder, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.predict_intrinsics = predict_intrinsics
        self.auto_scale = auto_scale
        
        if predict_intrinsics:
            assert image_width is not None and image_height is not None
            self.image_width = image_width
            self.image_height = image_height
        
        # 1. Flow融合网络
        # 将5通道flow信息(2+3)融合成语义特征
        self.flow_fusion = nn.Sequential(
            # 输入: [B, 5, H, W] (2通道optical + 3通道appearance)
            Conv3x3(5, 64),
            nn.ReLU(inplace=True),
            Conv3x3(64, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 128, H/2, W/2]
            
            Conv3x3(128, 256),
            nn.ReLU(inplace=True),
            Conv3x3(256, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 256, H/4, W/4]
            
            Conv3x3(256, 512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 512, H/8, W/8]
        )
        
        # 2. 全局特征提取
        # 将空间特征聚合为全局描述符
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 512, 1, 1]
            nn.Flatten(),  # [B, 512]
        )
        
        # 3. Pose预测头
        # 从全局特征预测pose参数
        self.pose_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6 * num_frames_to_predict_for)  # 6DOF per frame
        )
        
        # 4. 尺度预测分支(可选)
        if auto_scale:
            self.scale_head = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()  # 输出0-1，后续映射到合理范围
            )
        
        # 5. 内参预测分支(可选)
        if predict_intrinsics:
            num_intrinsic_params = 4  # fx, fy, cx, cy
            self.intrinsics_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_intrinsic_params)
            )
            self.softplus = nn.Softplus()
    
    def forward(self, optical_flow, appearance_flow):
        """
        Args:
            optical_flow: [B, 2, H, W] 光流
            appearance_flow: [B, 3, H, W] 外观流
        """
        batch_size = optical_flow.size(0)
        device = optical_flow.device
        
        # 1. 融合两种flow信息
        # 将optical flow和appearance flow在通道维度拼接
        flow_combined = torch.cat([optical_flow, appearance_flow], dim=1)  # [B, 5, H, W]
        
        # 2. 提取flow特征
        flow_features = self.flow_fusion(flow_combined)  # [B, 512, H/8, W/8]
        
        # 3. 全局特征聚合
        global_features = self.global_pool(flow_features)  # [B, 512]
        
        # 4. Pose预测
        pose_raw = self.pose_head(global_features)  # [B, 6*num_frames]
        
        # 5. 尺度处理
        if self.auto_scale:
            scale_factor = self.scale_head(global_features)  # [B, 1]
            # 将0-1映射到合理的尺度范围 (1e-3 到 1e-1)
            scale_factor = 1e-3 + scale_factor * (1e-1 - 1e-3)
        else:
            scale_factor = torch.ones(batch_size, 1, device=device) * 1e-3
        
        # 应用尺度并重塑
        pose_scaled = pose_raw * scale_factor  # [B, 6*num_frames]
        pose_reshaped = pose_scaled.view(batch_size, self.num_frames_to_predict_for, 1, 6)
        
        # 分离旋转和平移
        axisangle = pose_reshaped[..., :3]    # [B, num_frames, 1, 3]
        translation = pose_reshaped[..., 3:]  # [B, num_frames, 1, 3]
        
        # 6. 内参预测(可选)
        if self.predict_intrinsics:
            intrinsics_raw = self.intrinsics_head(global_features)  # [B, 4]
            intrinsics_mat = self._construct_intrinsics_matrix(intrinsics_raw, batch_size, device)
            return axisangle, translation, intrinsics_mat
        else:
            return axisangle, translation
    
    def _construct_intrinsics_matrix(self, intrinsics_raw, batch_size, device):
        """构建内参矩阵"""
        # 归一化到[0,1]
        intrinsics_normalized = torch.sigmoid(intrinsics_raw)
        
        # 构建4x4内参矩阵
        intrinsics_mat = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 计算焦距 (映射到合理范围)
        foci_scale = 0.5 + intrinsics_normalized[:, :2] * 2.0  # [0.5, 2.5]倍图像尺寸
        fx = foci_scale[:, 0] * self.image_width
        fy = foci_scale[:, 1] * self.image_height
        
        # 计算主点 (映射到图像中心附近)
        offset_scale = 0.3 + intrinsics_normalized[:, 2:] * 0.4  # [0.3, 0.7]倍图像尺寸
        cx = offset_scale[:, 0] * self.image_width
        cy = offset_scale[:, 1] * self.image_height
        
        # 填充内参矩阵
        intrinsics_mat[:, 0, 0] = self.softplus(fx) + 1e-3  # 确保正值
        intrinsics_mat[:, 1, 1] = self.softplus(fy) + 1e-3
        intrinsics_mat[:, 0, 2] = cx
        intrinsics_mat[:, 1, 2] = cy
        
        return intrinsics_mat


class LightweightFlowPoseDecoder(nn.Module):
    """
    轻量级版本：直接从flow信息预测pose，计算量更小
    """
    
    def __init__(self, 
                 num_frames_to_predict_for=1,
                 auto_scale=True):
        super(LightweightFlowPoseDecoder, self).__init__()
        
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.auto_scale = auto_scale
        
        # 简化的flow处理网络
        self.flow_processor = nn.Sequential(
            # 输入: [B, 5, H, W]
            nn.Conv2d(5, 32, 7, stride=4, padding=3),  # 大核快速下采样
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 简化的pose预测
        self.pose_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6 * num_frames_to_predict_for)
        )
        
        if auto_scale:
            self.scale_predictor = nn.Sequential(
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
    def forward(self, optical_flow, appearance_flow):
        batch_size = optical_flow.size(0)
        device = optical_flow.device
        
        # 合并flow
        flow_combined = torch.cat([optical_flow, appearance_flow], dim=1)
        
        # 处理
        features = self.flow_processor(flow_combined)
        pose_raw = self.pose_predictor(features)
        
        # 尺度
        if self.auto_scale:
            scale = self.scale_predictor(features) * 1e-2
        else:
            scale = torch.ones(batch_size, 1, device=device) * 1e-3
        
        # 输出
        pose_scaled = pose_raw * scale
        pose_reshaped = pose_scaled.view(batch_size, self.num_frames_to_predict_for, 1, 6)
        
        axisangle = pose_reshaped[..., :3]
        translation = pose_reshaped[..., 3:]
        
        return axisangle, translation


class FlowBasedMultiScalePoseDecoder(nn.Module):
    """
    多尺度版本：处理不同分辨率的flow信息，更鲁棒
    """
    
    def __init__(self, 
                 scales=[0, 1, 2, 3],  # 对应不同分辨率的flow
                 num_frames_to_predict_for=1,
                 auto_scale=True):
        super(FlowBasedMultiScalePoseDecoder, self).__init__()
        
        self.scales = scales
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.auto_scale = auto_scale
        
        # 每个尺度的处理网络
        self.scale_processors = nn.ModuleDict()
        for scale in scales:
            self.scale_processors[str(scale)] = nn.Sequential(
                Conv3x3(5, 32),
                nn.ReLU(inplace=True),
                Conv3x3(32, 64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        
        # 多尺度特征融合
        total_features = 64 * len(scales)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        # Pose预测
        self.pose_head = nn.Linear(128, 6 * num_frames_to_predict_for)
        
        if auto_scale:
            self.scale_head = nn.Sequential(
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
    def forward(self, optical_flows, appearance_flows):
        """
        Args:
            optical_flows: dict of {scale: [B, 2, H, W]}
            appearance_flows: dict of {scale: [B, 3, H, W]}
        """
        batch_size = optical_flows[self.scales[0]].size(0)
        device = optical_flows[self.scales[0]].device
        
        # 处理每个尺度
        scale_features = []
        for scale in self.scales:
            # 合并该尺度的flow
            flow_combined = torch.cat([
                optical_flows[scale], 
                appearance_flows[scale]
            ], dim=1)
            
            # 提取特征
            features = self.scale_processors[str(scale)](flow_combined)
            scale_features.append(features)
        
        # 融合多尺度特征
        combined_features = torch.cat(scale_features, dim=1)
        fused_features = self.fusion(combined_features)
        
        # 预测pose
        pose_raw = self.pose_head(fused_features)
        
        # 尺度处理
        if self.auto_scale:
            scale_factor = self.scale_head(fused_features) * 1e-2
        else:
            scale_factor = torch.ones(batch_size, 1, device=device) * 1e-3
        
        # 输出
        pose_scaled = pose_raw * scale_factor
        pose_reshaped = pose_scaled.view(batch_size, self.num_frames_to_predict_for, 1, 6)
        
        axisangle = pose_reshaped[..., :3]
        translation = pose_reshaped[..., 3:]
        
        return axisangle, translation
