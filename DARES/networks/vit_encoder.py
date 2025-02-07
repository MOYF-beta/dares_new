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

class VitEncoder(nn.Module):
    """兼容ResNet接口的ViT编码器"""
    def __init__(self, num_input_images=1, pretrained=True, img_size=(256,320)):
        super().__init__()
        self.encoder = ViTMultiImageInput(num_input_images, img_size)
        self.num_ch_enc = self.encoder.num_ch_enc
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """加载预训练权重"""
        pretrained_dict = vit_b_16(weights='DEFAULT').state_dict()
        model_dict = self.encoder.state_dict()
        
        # 过滤可加载参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                         if k in model_dict and "patch_embed" not in k}
        model_dict.update(pretrained_dict)
        self.encoder.load_state_dict(model_dict)

    def forward(self, input_image):
        # 标准化处理 (与ResNet保持一致)
        # x = (input_image - 0.45) / 0.225
        x = input_image
        features = self.encoder(x)
        
        # 对齐ResNet的特征层级
        output_features = [features[0]]  # 第一层
        for f in features[1:]:
            output_features.append(output_features[-1])
        
        return output_features