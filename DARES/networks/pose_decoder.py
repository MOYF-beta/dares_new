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

class PoseDecoder_with_intrinsics2(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,
                 predict_intrinsics=False, simplified_intrinsic=False, image_width=None, image_height=None):
        super(PoseDecoder_with_intrinsics2, self).__init__()
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
        self.convs[("pose_f", 0)] = nn.Conv2d(num_ch_enc, 256, 3, stride, 1)
        self.convs[("pose_f", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        # if predict_intrinsics, the feature is extracted here
        self.convs[("pose_f", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.convs[("pose_b", 0)] = nn.Conv2d(num_ch_enc, 256, 3, stride, 1)
        self.convs[("pose_b", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose_b", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

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
                ConvBlock(num_ch_enc, 256),
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
        
        out_f = input_features
        out_b = input_features.clone()
        for i in range(3):
            out_b = self.convs[("pose_b", i)](out_b)
            if i != 2:
                out_b = self.relu(out_b)

        out_b = out_b.mean(3).mean(2)

        out_b = 0.001 * out_b.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle_b = out_b[..., :3]
        translation_b = out_b[..., 3:]
        for i in range(3):
            out_f = self.convs[("pose_f", i)](out_f)
            if i != 2:
                out_f = self.relu(out_f)
            # if i == 0 and self.predict_intrinsics:
            #     feature_for_intrinsics = out

        out_f = out_f.mean(3).mean(2)

        out_f = 0.001*out_f.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle_f = out_f[..., :3]
        translation_f = out_f[..., 3:]
        if self.predict_intrinsics:
            return axisangle_f, translation_f, axisangle_b, translation_b, predict_intrinsics(input_features)
        else:
            return axisangle_f, translation_f, axisangle_b, translation_b