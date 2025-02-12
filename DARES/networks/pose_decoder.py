from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
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
                 predict_intrinsics=False, simplified_intrinsic=False, image_width=None, image_height=None):
        super(PoseDecoder_with_intrinsics, self).__init__()
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
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        # if predict_intrinsics, the feature is extracted here
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

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


        self.net = nn.ModuleList(list(self.convs.values()))
    def forward(self, input_features):
        def predict_intrinsics(feature_for_intrinsics):
            batch_size = feature_for_intrinsics.shape[0]
            # prepare intrinsics matrix
            intrinsics_mat = torch.eye(4).unsqueeze(0).to(device)
            intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            # do the prediction
            intrinsics = self.intrinsics_layers(feature_for_intrinsics)
            # construct the intrinsics matrix
            foci = (intrinsics[:, :2] + 0.5) * torch.Tensor([self.image_width, self.image_height]).to(device)
            foci_mat = self.softlpus(torch.diag_embed(foci))
            if self.num_param_to_predict == 4:
                offsets = (intrinsics[:, 2:] + 0.5) * torch.Tensor([self.image_width, self.image_height]).to(device)
            else:
                offsets = torch.ones((batch_size,2)) * 0.5
            intrinsics_mat[:, :2, :2] = foci_mat
            intrinsics_mat[:, :2, 2:3] = offsets.unsqueeze(-1)

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

        out = 0.001*out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]
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