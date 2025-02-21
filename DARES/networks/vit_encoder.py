import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class ViTMultiImageInput(nn.Module):
    """支持多图输入的ViT编码器"""
    def __init__(self, num_input_images=1, img_size=(320,256), patch_size=16):
        super().__init__()
        
        # 原始ViT配置
        original_vit = vit_b_16(weights='DEFAULT')
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        # 修改输入层适应多图输入
        in_channels = num_input_images * 3
        self.num_input_images = num_input_images
        self.patch_embed = nn.Conv2d(in_channels, original_vit.conv_proj.out_channels,
                                   kernel_size=patch_size, stride=patch_size)
        
        # 复制其他组件
        self.class_token = original_vit.class_token
        
        # 位置编码调整 ------------------------------------------------------
        # 获取原始位置编码并分离class token
        original_pos_embed = original_vit.encoder.pos_embedding
        embed_dim = original_pos_embed.shape[-1]
        
        # 计算新尺寸下的网格参数
        grid_h = img_size[0] // patch_size
        grid_w = img_size[1] // patch_size
        
        # 分离class token和patch位置编码
        original_cls_pos = original_pos_embed[:, :1, :]  # 保留class token位置
        original_patch_pos = original_pos_embed[:, 1:, :]
        
        # 将原始patch位置编码转换为2D图像格式
        orig_grid_size = int(original_patch_pos.shape[1]**0.5)
        original_patch_pos = original_patch_pos.reshape(
            1, orig_grid_size, orig_grid_size, embed_dim
        ).permute(0, 3, 1, 2)  # [1, embed_dim, grid, grid]
        
        # 使用双三次插值调整空间维度
        new_patch_pos = torch.nn.functional.interpolate(
            original_patch_pos,
            size=(grid_h, grid_w),
            mode='bicubic',
            align_corners=False
        )
        
        # 转换回序列格式并与class token拼接
        new_patch_pos = new_patch_pos.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.cat([original_cls_pos, new_patch_pos], dim=1)
        )
        # 位置编码调整结束 --------------------------------------------------
        
        self.transformer = original_vit.encoder
        self.ln = original_vit.encoder.ln
        
        # 特征金字塔配置
        self.num_ch_enc = np.array([256, 512, 768, 768])  # 各层特征通道数
        self.feature_levels = [3, 6, 9, 11]  # 选择中间层作为特征输出

        # 初始化多图权重
        if num_input_images > 1:
            self._create_multiimage_weights(original_vit)

    def _create_multiimage_weights(self,original_vit):
        """处理多图输入的权重初始化"""
        old_weight = original_vit.conv_proj.weight.data
        new_weight = old_weight.repeat(1, self.num_input_images, 1, 1)
        new_weight /= self.num_input_images
        self.patch_embed.weight.data = new_weight

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.positional_embedding
        
        # 特征收集
        features = []
        for i, blk in enumerate(self.transformer.layers):
            x = blk(x)
            if i in self.feature_levels:
                # 移除cls_token并reshape为特征图
                # 使用保存的网格尺寸
                feature = x[:, 1:]  # 移除class token [B, num_patches, embed_dim]
                feature = feature.transpose(1, 2)  # [B, embed_dim, num_patches]
                feature = feature.view(B, -1, self.grid_h, self.grid_w)  # 正确reshape
                features.append(feature)
        
        # 调整特征金字塔尺寸
        feature_pyramid = []
        for feat in features:
            if feature_pyramid:
                # 自动匹配前一层的尺寸
                target_size = feature_pyramid[-1].shape[-2:]
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            feature_pyramid.append(feat)
        
        return feature_pyramid

class ViTMotionEmbedding(nn.Module):
    """支持时序信息的ViT编码器，通过motion embedding处理多帧输入"""
    def __init__(self, num_frames=2, img_size=(320,256), patch_size=16, 
                 motion_dim=64, temporal_attention=True):
        super().__init__()
        
        # 基础ViT配置
        original_vit = vit_b_16(weights='DEFAULT')
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        
        # 修改输入层处理单帧
        self.patch_embed = nn.Conv2d(3, original_vit.conv_proj.out_channels,
                                   kernel_size=patch_size, stride=patch_size)
        
        # Motion Embedding层
        embed_dim = original_vit.hidden_dim
        self.motion_embed = nn.Sequential(
            nn.Conv2d(2 * embed_dim, motion_dim, 1),  # 压缩相邻帧差异
            nn.GELU(),
            nn.Conv2d(motion_dim, embed_dim, 1)  # 映射回原始维度
        )
        
        # 时序注意力模块（可选）
        self.temporal_attention = temporal_attention
        if temporal_attention:
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.temporal_ln = nn.LayerNorm(embed_dim)
        
        # 复制基础组件
        self.class_token = original_vit.class_token
        
        # 位置编码调整
        original_pos_embed = original_vit.encoder.pos_embedding
        embed_dim = original_pos_embed.shape[-1]
        
        # 计算新尺寸下的网格参数
        grid_h = img_size[0] // patch_size
        grid_w = img_size[1] // patch_size
        
        # 分离并调整位置编码
        original_cls_pos = original_pos_embed[:, :1, :]
        original_patch_pos = original_pos_embed[:, 1:, :]
        
        orig_grid_size = int(original_patch_pos.shape[1]**0.5)
        original_patch_pos = original_patch_pos.reshape(
            1, orig_grid_size, orig_grid_size, embed_dim
        ).permute(0, 3, 1, 2)
        
        new_patch_pos = torch.nn.functional.interpolate(
            original_patch_pos,
            size=(grid_h, grid_w),
            mode='bicubic',
            align_corners=False
        )
        
        new_patch_pos = new_patch_pos.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.cat([original_cls_pos, new_patch_pos], dim=1)
        )
        
        # 时序位置编码
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, num_frames, 1, embed_dim) * 0.02
        )
        
        self.transformer = original_vit.encoder
        self.ln = original_vit.encoder.ln
        
        # 特征金字塔配置
        self.num_ch_enc = np.array([256, 512, 768, 768])
        self.feature_levels = [3, 6, 9, 11]

    def compute_motion_features(self, x1, x2):
        """计算相邻帧之间的运动特征"""
        # 拼接相邻帧的特征
        concat_features = torch.cat([x1, x2], dim=1)
        # 通过motion embedding网络
        motion_feat = self.motion_embed(concat_features)
        return motion_feat

    def forward(self, x):
        """
        输入: [B, T, C, H, W] 格式的多帧图像
        """
        if x.dim() == 4:# 此时输入T维度在C维度叠加了
            B, C, H, W = x.shape
            assert C // 3 == self.num_frames, f"Expected {self.num_frames} frames, got {C // 3}"
            x = x.view(B, self.num_frames, 3, H, W)
            
        B, T, C, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"
        
        # 逐帧处理
        frame_features = []
        for t in range(T):
            # 提取patch embeddings
            feat = self.patch_embed(x[:, t])  # [B, embed_dim, grid_h, grid_w]
            frame_features.append(feat)
        
        # 计算motion features
        motion_features = []
        for t in range(T-1):
            motion_feat = self.compute_motion_features(
                frame_features[t], frame_features[t+1]
            )
            motion_features.append(motion_feat)
        
        # 融合motion信息
        enhanced_features = []
        enhanced_features.append(frame_features[0])
        for t in range(T-1):
            enhanced_feat = frame_features[t+1] + motion_features[t]
            enhanced_features.append(enhanced_feat)
        
        # 处理每一帧
        outputs = []
        for t in range(T):
            curr_feat = enhanced_features[t]
            # 展平空间维度
            curr_feat = curr_feat.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            
            # 添加class token
            cls_tokens = self.class_token.expand(B, -1, -1)
            curr_feat = torch.cat([cls_tokens, curr_feat], dim=1)
            
            # 添加位置编码和时序编码
            curr_feat = curr_feat + self.positional_embedding
            curr_feat = curr_feat + self.temporal_embedding[:, t]
            
            outputs.append(curr_feat)
        
        # 堆叠所有帧的特征
        x = torch.stack(outputs, dim=1)  # [B, T, num_patches+1, embed_dim]
        
        # 时序注意力（可选）
        if self.temporal_attention:
            B, T, N, C = x.shape
            x = x.view(B * N, T, C)  # 重排以便计算时序注意力
            x = self.temporal_ln(x)
            x, _ = self.temporal_attn(x, x, x)
            x = x.view(B, N, T, C).transpose(1, 2)  # [B, T, N, C]
        
        # 特征金字塔构建
        feature_pyramid = []
        for t in range(T):
            curr_feat = x[:, t]  # 处理当前帧
            curr_features = []
            
            for i, blk in enumerate(self.transformer.layers):
                curr_feat = blk(curr_feat)
                if i in self.feature_levels:
                    # 提取特征图
                    feature = curr_feat[:, 1:]
                    feature = feature.transpose(1, 2)
                    feature = feature.view(B, -1, self.grid_h, self.grid_w)
                    curr_features.append(feature)
            
            # 调整特征金字塔尺寸
            curr_pyramid = []
            for feat in curr_features:
                if curr_pyramid:
                    target_size = curr_pyramid[-1].shape[-2:]
                    feat = nn.functional.interpolate(
                        feat, size=target_size, mode='bilinear', align_corners=False
                    )
                curr_pyramid.append(feat)
            
            feature_pyramid.append(curr_pyramid)
        
        # 合并所有帧的特征金字塔
        final_pyramid = []
        for level in range(len(feature_pyramid[0])):
            level_features = torch.stack(
                [fp[level] for fp in feature_pyramid], dim=1
            )  # [B, T, C, H, W]
            level_features = level_features.view(B, -1, *level_features.shape[3:])
            final_pyramid.append(level_features)
        
        return final_pyramid

class VitEncoder(nn.Module):
    """兼容ResNet接口的ViT编码器"""
    def __init__(self, num_input_images=1, pretrained=True, img_size=(256,320), motion_embedding=False):
        super().__init__()
        if motion_embedding:
            self.encoder = ViTMotionEmbedding(num_input_images, img_size)
            self.num_ch_enc = self.encoder.num_ch_enc*2
        else:
            self.encoder = ViTMultiImageInput(num_input_images, img_size)
            self.num_ch_enc = self.encoder.num_ch_enc
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """加载预训练权重"""
        pretrained_dict = vit_b_16(weights='DEFAULT').state_dict()
        model_dict = self.encoder.state_dict()
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                         if k in model_dict and "patch_embed" not in k}
        model_dict.update(pretrained_dict)         
        self.encoder.load_state_dict(model_dict)

    def forward(self, input_image):
        x = input_image
        features = self.encoder(x)

        output_features = [features[0]]  # 第一层
        for f in features[1:]:
            output_features.append(output_features[-1])
        
        return output_features

