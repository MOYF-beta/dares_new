from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from exps.exp_setup_local import device
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlockWithAttention(nn.Module):
    """Layer to perform a convolution followed by ELU and multi-head attention
    """
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(ConvBlockWithAttention, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        
        # Store original shape for restoration
        B, C, H, W = out.shape
        
        # Flatten spatial dimensions for attention
        out_flat = out.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Apply multi-head attention
        attn_out, _ = self.attention(out_flat, out_flat, out_flat)
        
        # Restore original shape
        out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        return out


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        # Ensure model layers are on the same device as input
        device = last_features[0].device
        self.convs["squeeze"] = self.convs["squeeze"].to(device)
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = 0.001*out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

class PoseDecoder_with_intrinsics(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,
                 predict_intrinsics=False, simplified_intrinsic=False, image_width=None, image_height=None, auto_scale=True):
        super(PoseDecoder_with_intrinsics, self).__init__()
        self.auto_scale = auto_scale
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        if predict_intrinsics:
            assert image_width is not None and image_height is not None\
                , "image_width and image_height must be provided if predict_intrinsics is True"
            self.image_width = image_width
            self.image_height = image_height

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.predict_intrinsics = predict_intrinsics

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)        # if predict_intrinsics, the feature is extracted here
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.relu = nn.ReLU()
        self.softlpus = nn.Softplus()
        
        if self.predict_intrinsics:
            if simplified_intrinsic:
                # fx, fy = ? ; cx = cy = 0.5
                self.num_param_to_predict = 2
            else:
                # fx, fy, cx, cy = ?
                self.num_param_to_predict = 4
            
            self.intrinsics_layers = nn.Sequential(
                ConvBlock(256, 256),
                ConvBlock(256, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.num_param_to_predict),
            )
        
        # 添加scale_factor预测分支
        if self.auto_scale:
            self.trans_scale_factor_layers = nn.Sequential(
                ConvBlock(256, 128),
                ConvBlock(128, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
            self.axis_scale_factor_layers = nn.Sequential(
                ConvBlock(256, 128),
                ConvBlock(128, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
        
        self.net = nn.ModuleList(list(self.convs.values()))

    def _fuse_features(self, features_list):
        """
        Fuse multi-scale features using the specified fusion method
        
        Args:
            features_list: List of feature tensors [B, hidden_dim, H, W]
        
        Returns:
            fused_features: Fused feature tensor [B, hidden_dim, H, W]
        """
        if len(features_list) == 1:
            return features_list[0]
        
        if self.fusion_method == 'weighted_sum':
            # Weighted sum with learnable weights
            weights = F.softmax(self.scale_weights, dim=0)
            fused_features = sum(w * feat for w, feat in zip(weights, features_list))
            
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            attention_scores = []
            for feat in features_list:
                score = self.scale_attention(feat)  # [B, 1]
                attention_scores.append(score)
            
            # Normalize attention scores
            attention_weights = F.softmax(torch.stack(attention_scores, dim=1), dim=1)  # [B, num_scales, 1]
            
            # Apply attention weights
            stacked_features = torch.stack(features_list, dim=1)  # [B, num_scales, hidden_dim, H, W]
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_scales, 1, 1, 1]
            fused_features = torch.sum(stacked_features * attention_weights, dim=1)  # [B, hidden_dim, H, W]
            
        elif self.fusion_method == 'conv':
            # Convolutional fusion
            concatenated = torch.cat(features_list, dim=1)  # [B, hidden_dim*num_scales, H, W]
            fused_features = self.fusion_conv(concatenated)  # [B, hidden_dim, H, W]
            
        elif self.fusion_method == 'avg':
            # Simple average
            fused_features = torch.stack(features_list, dim=0).mean(dim=0)
            
        elif self.fusion_method == 'max':
            # Element-wise maximum
            fused_features = torch.stack(features_list, dim=0).max(dim=0)[0]
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_features

    def forward(self, input_features):
        def predict_intrinsics(feature_for_intrinsics):
            batch_size = feature_for_intrinsics.shape[0]
            
            # prepare intrinsics matrix
            intrinsics_mat = torch.eye(4, device=device).unsqueeze(0)
            intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            
            # do the prediction
            intrinsics = self.intrinsics_layers(feature_for_intrinsics)
            
            # Apply sigmoid to normalize predictions to [0, 1] range for numerical stability
            intrinsics_normalized = torch.sigmoid(intrinsics)
            
            # Scale focal lengths to reasonable range [0.5, 2.5] times image dimensions
            # This ensures positive values and prevents singularity
            foci_scale_min, foci_scale_max = 0.5, 2.5
            foci_normalized = intrinsics_normalized[:, :2]
            foci_scaled = foci_scale_min + foci_normalized * (foci_scale_max - foci_scale_min)
            foci = foci_scaled * torch.tensor([self.image_width, self.image_height], device=device)
            
            # Apply softplus and add minimum threshold to ensure positive focal lengths
            foci = self.softlpus(foci) + 1e-3  # Minimum focal length to prevent singularity
            if self.num_param_to_predict == 4:
                # For full intrinsics, ensure principal points are within valid range
                offsets_normalized = intrinsics_normalized[:, 2:]
                # Map to [0.1, 0.9] of image dimensions to avoid edge cases
                offsets = (0.1 + offsets_normalized * 0.8) * torch.tensor([self.image_width, self.image_height], device=device)
            else:
                # For simplified intrinsics, set principal points to image center with proper scaling
                offsets = torch.ones((batch_size, 2), device=device) * torch.tensor([self.image_width / 2.0, self.image_height / 2.0], device=device)
            
            # Construct intrinsics matrix safely with bounds checking
            # 取batch内平均值，所有样本用同一个intrinsics
            mean_foci = torch.mean(foci, dim=0)
            mean_offsets = torch.mean(offsets, dim=0)
            intrinsics_mat[:, 0, 0] = torch.clamp(mean_foci[0], min=1e-3)  # fx
            intrinsics_mat[:, 1, 1] = torch.clamp(mean_foci[1], min=1e-3)  # fy
            intrinsics_mat[:, 0, 2] = torch.clamp(mean_offsets[0], min=1.0, max=self.image_width-1.0)  # cx
            intrinsics_mat[:, 1, 2] = torch.clamp(mean_offsets[1], min=1.0, max=self.image_height-1.0)  # cy
            # print(f'fx: {intrinsics_mat[:, 0, 0].mean().item():.2f}, fy: {intrinsics_mat[:, 1, 1].mean().item():.2f}, cx: {intrinsics_mat[:, 0, 2].mean().item():.2f}, cy: {intrinsics_mat[:, 1, 2].mean().item():.2f}', end='\t')
            return intrinsics_mat

        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
            # if i == 0 and self.predict_intrinsics:
            #     feature_for_intrinsics = out

        out = out.mean(3).mean(2)
        if self.auto_scale:
            trans_scale_factor = self.trans_scale_factor_layers(cat_features)
            axis_scale_factor = self.axis_scale_factor_layers(cat_features)
            trans_scale_factor = torch.exp(trans_scale_factor*5 + 0.5) * 1e-2
            axis_scale_factor = torch.exp(axis_scale_factor*5 + 0.5) * 1e-2
        else:
            trans_scale_factor = torch.ones(out.shape[0], 1, device=device)* 1e-2
            axis_scale_factor = torch.ones(out.shape[0], 1, device=device)* 1e-2
        
        # NOTE : Here is a key issue of training on different datasets. On previous studies this is fixed
        # Beili says this vary from dataset to dataset.
        # 应用通过网络估计的scale_factor
        out = out.view(-1, self.num_frames_to_predict_for, 1, 6) 
        
        axisangle = out[..., :3]* trans_scale_factor.unsqueeze(1).unsqueeze(2)
        translation = out[..., 3:]* axis_scale_factor.unsqueeze(1).unsqueeze(2)
        print(f'axis_scale: {axis_scale_factor.mean().item():.4f}, trans_scale: {trans_scale_factor.mean().item():.4f}', end='\r')
        if self.predict_intrinsics:
            return axisangle, translation, predict_intrinsics(cat_features)
        else:
            return axisangle, translation


class SimplifiedPoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None,
                 predict_intrinsics=False, image_width=None, image_height=None):
        super(SimplifiedPoseDecoder, self).__init__()
        
        # Basic initialization
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.predict_intrinsics = predict_intrinsics
        
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        
        if predict_intrinsics:
            self.image_width = image_width
            self.image_height = image_height
        
        # Simplified convolution layers
        self.convs = OrderedDict()
        # Reduce initial channels to 128 instead of 256
        self.convs["squeeze"] = nn.Conv2d(self.num_ch_enc[-1], 128, 1)
        # Simplified pose prediction pathway
        self.convs["pose_pred"] = nn.Sequential(
            nn.Conv2d(num_input_features * 128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 6 * num_frames_to_predict_for, 1)
        )
        
        # Simplified intrinsics prediction (if enabled)
        if self.predict_intrinsics:
            self.intrinsics_layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(num_input_features * 128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 4)  # Predicting fx, fy, cx, cy
            )
        
        self.relu = nn.ReLU()
        
    def forward(self, input_features):
        # Process input features
        last_features = [f[-1] for f in input_features]
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        
        # Pose prediction
        pose_out = self.convs["pose_pred"](cat_features)
        pose_out = pose_out.mean(3).mean(2)
        pose_out = 0.001 * pose_out.view(-1, self.num_frames_to_predict_for, 1, 6)
        
        # Split pose into rotation and translation
        axisangle = pose_out[..., :3]
        translation = pose_out[..., 3:]
        
        # Intrinsics prediction (if enabled)
        if self.predict_intrinsics:
            batch_size = cat_features.shape[0]
            intrinsics = self.intrinsics_layers(cat_features)
            
            # Construct intrinsics matrix
            intrinsics_mat = torch.eye(4).unsqueeze(0).to(device)
            intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            
            # Scale predictions to image dimensions
            foci = (intrinsics[:, :2] + 0.5) * torch.Tensor(
                [self.image_width, self.image_height],
                
            ).to(device)
            offsets = (intrinsics[:, 2:] + 0.5) * torch.Tensor(
                [self.image_width, self.image_height],
            ).to(device)
            
            # Fill intrinsics matrix
            intrinsics_mat[:, 0, 0] = foci[:, 0]  # fx
            intrinsics_mat[:, 1, 1] = foci[:, 1]  # fy
            intrinsics_mat[:, 0, 2] = offsets[:, 0]  # cx
            intrinsics_mat[:, 1, 2] = offsets[:, 1]  # cy
            
            return axisangle, translation, intrinsics_mat
        
        return axisangle, translation


class CrossAttnPoseDecoder_with_intrinsics(nn.Module):
    """
    Cross Attention Pose Decoder that processes hidden states from two DARES models (ref and tar)
    using cross attention to extract features for pose and intrinsics prediction.
    """
    def __init__(self, num_ch_enc, num_input_features=2, num_frames_to_predict_for=None, stride=1,
                 predict_intrinsics=False, simplified_intrinsic=False, image_width=None, image_height=None, 
                 auto_scale=True, num_heads=8, hidden_dim=256, use_scales=None, fusion_method='weighted_sum'):
        super(CrossAttnPoseDecoder_with_intrinsics, self).__init__()
        
        self.auto_scale = auto_scale
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features  # Should be 2 for ref and tar
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        # Determine which scales to use (default: all scales)
        if use_scales is None:
            self.use_scales = list(range(len(num_ch_enc)))  # Use all scales by default
        else:
            self.use_scales = use_scales
        self.num_scales = len(self.use_scales)
        
        if predict_intrinsics:
            assert image_width is not None and image_height is not None, \
                "image_width and image_height must be provided if predict_intrinsics is True"
            self.image_width = image_width
            self.image_height = image_height

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.predict_intrinsics = predict_intrinsics

        # Feature extraction layers for ref and tar hidden states (for each scale)
        self.ref_squeeze_layers = nn.ModuleList([
            nn.Conv2d(self.num_ch_enc[scale_idx], hidden_dim, 1) 
            for scale_idx in self.use_scales
        ])
        self.tar_squeeze_layers = nn.ModuleList([
            nn.Conv2d(self.num_ch_enc[scale_idx], hidden_dim, 1) 
            for scale_idx in self.use_scales
        ])
        
        # Multi-scale feature fusion
        if self.fusion_method == 'weighted_sum':
            # Learnable weights for each scale
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            self.scale_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
        elif self.fusion_method == 'conv':
            # Convolutional fusion
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(hidden_dim * self.num_scales, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
            )
        
        # Cross attention layers
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Pose prediction layers
        self.convs = OrderedDict()
        self.convs[("pose", 0)] = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(hidden_dim, 6 * num_frames_to_predict_for, 1)
        
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        
        # Intrinsics prediction layers
        if self.predict_intrinsics:
            if simplified_intrinsic:
                # fx, fy = ? ; cx = cy = 0.5
                self.num_param_to_predict = 2
            else:
                # fx, fy, cx, cy = ?
                self.num_param_to_predict = 4
            
            self.intrinsics_layers = nn.Sequential(
                ConvBlock(hidden_dim, hidden_dim),
                ConvBlock(hidden_dim, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.num_param_to_predict),
            )
        
        # Scale factor prediction layers
        if self.auto_scale:
            self.trans_scale_factor_layers = nn.Sequential(
                ConvBlock(hidden_dim, 128),
                ConvBlock(128, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
            self.axis_scale_factor_layers = nn.Sequential(
                ConvBlock(hidden_dim, 128),
                ConvBlock(128, 64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
        
        self.net = nn.ModuleList(list(self.convs.values()))

    def _fuse_features(self, features_list):
        """
        Fuse multi-scale features using the specified fusion method
        
        Args:
            features_list: List of feature tensors [B, hidden_dim, H, W]
        
        Returns:
            fused_features: Fused feature tensor [B, hidden_dim, H, W]
        """
        if len(features_list) == 1:
            return features_list[0]
        
        if self.fusion_method == 'weighted_sum':
            # Weighted sum with learnable weights
            weights = F.softmax(self.scale_weights, dim=0)
            fused_features = sum(w * feat for w, feat in zip(weights, features_list))
            
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            attention_scores = []
            for feat in features_list:
                score = self.scale_attention(feat)  # [B, 1]
                attention_scores.append(score)
            
            # Normalize attention scores
            attention_weights = F.softmax(torch.stack(attention_scores, dim=1), dim=1)  # [B, num_scales, 1]
            
            # Apply attention weights
            stacked_features = torch.stack(features_list, dim=1)  # [B, num_scales, hidden_dim, H, W]
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_scales, 1, 1, 1]
            fused_features = torch.sum(stacked_features * attention_weights, dim=1)  # [B, hidden_dim, H, W]
            
        elif self.fusion_method == 'conv':
            # Convolutional fusion
            concatenated = torch.cat(features_list, dim=1)  # [B, hidden_dim*num_scales, H, W]
            fused_features = self.fusion_conv(concatenated)  # [B, hidden_dim, H, W]
            
        elif self.fusion_method == 'avg':
            # Simple average
            fused_features = torch.stack(features_list, dim=0).mean(dim=0)
            
        elif self.fusion_method == 'max':
            # Element-wise maximum
            fused_features = torch.stack(features_list, dim=0).max(dim=0)[0]
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_features

    def forward(self, ref_hidden_states, tar_hidden_states):
        """
        Forward pass with cross attention between ref and tar hidden states
        
        Args:
            ref_hidden_states: List of hidden states from reference DARES model
            tar_hidden_states: List of hidden states from target DARES model
        
        Returns:
            axisangle: Predicted axis-angle rotation
            translation: Predicted translation
            intrinsics_mat: Predicted intrinsics matrix (if predict_intrinsics=True)
        """
        def predict_intrinsics(feature_for_intrinsics):
            batch_size = feature_for_intrinsics.shape[0]
            
            # Prepare intrinsics matrix
            intrinsics_mat = torch.eye(4, device=device).unsqueeze(0)
            intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            
            # Do the prediction
            intrinsics = self.intrinsics_layers(feature_for_intrinsics)
            
            # Apply sigmoid to normalize predictions to [0, 1] range for numerical stability
            intrinsics_normalized = torch.sigmoid(intrinsics)
            
            # Scale focal lengths to reasonable range [0.5, 2.5] times image dimensions
            foci_scale_min, foci_scale_max = 0.5, 2.5
            foci_normalized = intrinsics_normalized[:, :2]
            foci_scaled = foci_scale_min + foci_normalized * (foci_scale_max - foci_scale_min)
            foci = foci_scaled * torch.tensor([self.image_width, self.image_height], device=device)
            
            # Apply softplus and add minimum threshold to ensure positive focal lengths
            foci = self.softplus(foci) + 1e-3
            
            if self.num_param_to_predict == 4:
                # For full intrinsics, ensure principal points are within valid range
                offsets_normalized = intrinsics_normalized[:, 2:]
                offsets = (0.1 + offsets_normalized * 0.8) * torch.tensor([self.image_width, self.image_height], device=device)
            else:
                # For simplified intrinsics, set principal points to image center
                offsets = torch.ones((batch_size, 2), device=device) * torch.tensor([self.image_width / 2.0, self.image_height / 2.0], device=device)
            
            # Construct intrinsics matrix safely with bounds checking
            mean_foci = torch.mean(foci, dim=0)
            mean_offsets = torch.mean(offsets, dim=0)
            intrinsics_mat[:, 0, 0] = torch.clamp(mean_foci[0], min=1e-3)  # fx
            intrinsics_mat[:, 1, 1] = torch.clamp(mean_foci[1], min=1e-3)  # fy
            intrinsics_mat[:, 0, 2] = torch.clamp(mean_offsets[0], min=1.0, max=self.image_width-1.0)  # cx
            intrinsics_mat[:, 1, 2] = torch.clamp(mean_offsets[1], min=1.0, max=self.image_height-1.0)  # cy
            
            return intrinsics_mat        # Extract and process features from specified scales
        ref_features_list = []
        tar_features_list = []
        
        # Get target spatial size from the largest scale (assuming index 0 is largest)
        target_scale_idx = self.use_scales[0] if len(self.use_scales) > 0 else 0
        target_size = ref_hidden_states[target_scale_idx].shape[2:]  # (H, W)
        
        for i, scale_idx in enumerate(self.use_scales):
            # Extract features from specified scale
            ref_scale_features = ref_hidden_states[scale_idx]  # [B, C_i, H_i, W_i]
            tar_scale_features = tar_hidden_states[scale_idx]  # [B, C_i, H_i, W_i]
            
            # Apply squeeze convolution to unify channels
            ref_squeezed = self.relu(self.ref_squeeze_layers[i](ref_scale_features))  # [B, hidden_dim, H_i, W_i]
            tar_squeezed = self.relu(self.tar_squeeze_layers[i](tar_scale_features))  # [B, hidden_dim, H_i, W_i]
            
            # Resize to target size if necessary
            if ref_squeezed.shape[2:] != target_size:
                ref_squeezed = F.interpolate(ref_squeezed, size=target_size, mode='bilinear', align_corners=False)
                tar_squeezed = F.interpolate(tar_squeezed, size=target_size, mode='bilinear', align_corners=False)
            
            ref_features_list.append(ref_squeezed)
            tar_features_list.append(tar_squeezed)
        
        # Fuse multi-scale features
        ref_features = self._fuse_features(ref_features_list)  # [B, hidden_dim, H, W]
        tar_features = self._fuse_features(tar_features_list)  # [B, hidden_dim, H, W]
        
        # Reshape for attention: [B, H*W, hidden_dim]
        B, C, H, W = ref_features.shape
        ref_features_flat = ref_features.view(B, C, H*W).transpose(1, 2)  # [B, H*W, hidden_dim]
        tar_features_flat = tar_features.view(B, C, H*W).transpose(1, 2)  # [B, H*W, hidden_dim]
        
        # Cross attention: tar attends to ref
        cross_attended, _ = self.cross_attention(
            query=tar_features_flat,
            key=ref_features_flat,
            value=ref_features_flat
        )
        
        # Residual connection and layer norm
        cross_attended = self.layer_norm1(cross_attended + tar_features_flat)
        
        # Self attention on cross-attended features
        self_attended, _ = self.self_attention(
            query=cross_attended,
            key=cross_attended,
            value=cross_attended
        )
        
        # Residual connection and layer norm
        self_attended = self.layer_norm2(self_attended + cross_attended)
        
        # Feed forward network
        ffn_output = self.ffn(self_attended)
        attended_features = self_attended + ffn_output
        
        # Reshape back to spatial dimensions: [B, hidden_dim, H, W]
        attended_features = attended_features.transpose(1, 2).view(B, C, H, W)
        
        # Pose prediction through conv layers
        out = attended_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        
        # Global average pooling
        out = out.mean(3).mean(2)
        
        # Scale factor prediction
        if self.auto_scale:
            trans_scale_factor = self.trans_scale_factor_layers(attended_features)
            axis_scale_factor = self.axis_scale_factor_layers(attended_features)
            trans_scale_factor = torch.exp(trans_scale_factor * 5 + 0.5) * 1e-2
            axis_scale_factor = torch.exp(axis_scale_factor * 5 + 0.5) * 1e-2
        else:
            trans_scale_factor = torch.ones(out.shape[0], 1, device=device) * 1e-2
            axis_scale_factor = torch.ones(out.shape[0], 1, device=device) * 1e-2
        
        # Apply scale factors
        out = out.view(-1, self.num_frames_to_predict_for, 1, 6)
        
        axisangle = out[..., :3] * axis_scale_factor.unsqueeze(1).unsqueeze(2)
        translation = out[..., 3:] * trans_scale_factor.unsqueeze(1).unsqueeze(2)
        
        print(f'axis_scale: {axis_scale_factor.mean().item():.4f}, trans_scale: {trans_scale_factor.mean().item():.4f}', end='\r')
        
        if self.predict_intrinsics:
            return axisangle, translation, predict_intrinsics(attended_features)
        else:
            return axisangle, translation
