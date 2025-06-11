"""
DSFM Trainer - Direct Structure from Motion without pose encoder
This trainer eliminates the need for a pose encoder by directly using
optical flow (position) and appearance flow (transform) as inputs to
the AF_OF_Posedecoder_with_intrinsics module.
"""


import os
import torch
import torch.nn.functional as F

from DARES.layers import *
from DARES.utils import *

from exps.trainer_abc import Trainer
from DARES.networks.dares_peft import DARES
from DARES.networks.resnet_encoder import AttentionalResnetEncoder
from DARES.networks.optical_flow_decoder import PositionDecoder
from DARES.networks.appearance_flow_decoder import TransformDecoder
from DARES.networks.pose_decoder import AF_OF_Posedecoder_with_intrinsics


class TrainerDSFM(Trainer):
    """
    Direct Structure from Motion Trainer
    
    This trainer uses the AF_OF_Posedecoder_with_intrinsics module to predict poses
    directly from optical flow and appearance flow, eliminating the need for a pose encoder.
    """

    def compare_weight_size(self):
        # legacy: pose_encoder, pose_decoder, depth_model, position_encoder, position_decoder, transform_encoder, transform_decoder
        # current: position_encoder, position_decoder, transform_encoder, transform_decoder, af_of_pose_decoder, depth_model
        pose_encoder = AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2)
        position_encoder = AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2)
        transform_encoder = AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2)
        af_of_pose_decoder = AF_OF_Posedecoder_with_intrinsics(
            af_channels=3,  # Appearance flow channels
            of_channels=2,  # Optical flow channels
            patch_size=16,
            embed_dim=256,
            num_vit_layers=2,
            num_heads=4,
            num_frames_to_predict_for=1,
            predict_intrinsics=self.opt.learn_intrinsics,
            simplified_intrinsic=self.opt.simplified_intrinsic,
            image_width=self.opt.width,
            image_height=self.opt.height,
            auto_scale=True,
            dropout=0.1,
            input_height=self.opt.height // 2,
            input_width=self.opt.width // 2,
        )
        position_decoder = PositionDecoder(position_encoder.num_ch_enc, self.opt.scales)
        transform_decoder = TransformDecoder(transform_encoder.num_ch_enc, self.opt.scales)
        depth_model = DARES()
        print("Legacy model weights total size:")
        total_size_legacy = (
            sum(p.numel() for p in pose_encoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in position_encoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in position_decoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in transform_encoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in transform_decoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in af_of_pose_decoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in depth_model.parameters() if p.requires_grad)
        )
        print(f"Legacy model total size: {total_size_legacy / 1e6:.2f}M")
        print("Current model weights total size:")
        total_size_current = (
            sum(p.numel() for p in position_encoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in position_decoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in transform_encoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in transform_decoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in af_of_pose_decoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in depth_model.parameters() if p.requires_grad)
        )
        print(f"Current model total size: {total_size_current / 1e6:.2f}M")
    def load_model(self):
        """Load models for DSFM approach"""
        # Initialize depth model and flow encoders
        encoders = {
            "depth_model": DARES(),
            "position_encoder": AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2),
            "transform_encoder": AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2)
        }

        # Initialize decoders
        decoders = {
            "position": PositionDecoder(encoders["position_encoder"].num_ch_enc, self.opt.scales),
            "transform": TransformDecoder(encoders["transform_encoder"].num_ch_enc, self.opt.scales)
        }
        
        # Initialize AF_OF pose decoder (no pose encoder needed)
        self.af_of_pose_decoder = AF_OF_Posedecoder_with_intrinsics(
            af_channels=3,  # Appearance flow channels
            of_channels=2,  # Optical flow channels
            patch_size=16,
            embed_dim=256,
            num_vit_layers=2,
            num_heads=4,
            num_frames_to_predict_for=1,
            predict_intrinsics=self.opt.learn_intrinsics,
            simplified_intrinsic=self.opt.simplified_intrinsic,
            image_width=self.opt.width,
            image_height=self.opt.height,
            auto_scale=True,
            dropout=0.1,
            input_height=self.opt.height // 2,
            input_width=self.opt.width // 2,
        )

        # Register models for training
        all_models = [
            ("depth_model",         encoders["depth_model"],        self.param_monodepth),
            ("position_encoder",    encoders["position_encoder"],   self.param_pose_net),
            ("position",            decoders["position"],           self.param_pose_net),
            ("transform_encoder",   encoders["transform_encoder"],  self.param_pose_net),
            ("transform",           decoders["transform"],          self.param_pose_net),
            ("pose",                self.af_of_pose_decoder,        self.param_monodepth)  # AF_OF decoder as pose model
        ]
        
        # Initialize models and parameters, and move to device
        for model_name, model_instance, param_list in all_models:
            self.models[model_name] = model_instance
            param_list += list(filter(lambda p: p.requires_grad, model_instance.parameters()))
            self.models[model_name].to(self.device)
        
        # Load pretrained weights if available
        if self.pretrained_root_dir is not None:
            model_paths = [
                "depth_model",
                "position_encoder",
                "position",
                "transform_encoder",
                "transform"
                # Note: No pose_encoder or pose since we're using AF_OF decoder
            ]
            for model_name in model_paths:
                model_path = os.path.join(self.pretrained_root_dir, "best", f"{model_name}.pth")
                if os.path.exists(model_path):
                    self.models[model_name].load_state_dict(
                        torch.load(model_path, map_location=self.device, weights_only=True), 
                        strict=False
                    )
                    print(f"\033[94mLoaded pretrained weights for {model_name} from {model_path}\033[0m")
                else:
                    print(f"\033[93mNo pretrained weights found for {model_name} at {model_path}\033[0m")
    # def set_train(self, train_position_only=False, freeze_depth=False):
    #     """Enable or disable gradients based on optimizer groups."""
    #     # Switch model modes: when training position only, keep depth_model in eval, others in train; otherwise all train
    #     for name, model in self.models.items():
    #         model.train()
    #     # Freeze/unfreeze depth parameters
    #     for group in self.optimizer_depth.param_groups:
    #         for p in group['params']:
    #             p.requires_grad = not train_position_only
    #     # Freeze/unfreeze pose parameters
    #     for group in self.optimizer_pose.param_groups:
    #         for p in group['params']:
    #             p.requires_grad = True #  train_position_only
    #     if freeze_depth:
    #         for param in self.models["depth_model"].parameters():
    #             param.requires_grad = False
    #         self.models["depth_model"].eval()
    def get_depth_input(self, inputs):
        """Get depth input tensor from inputs - DARES expects 4D tensor (B, C, H, W)"""
        # DARES is a single-frame depth estimation model, so we use the reference frame (index 0)
        depth_input = inputs[("color_aug", 0, 0)]  # Reference frame
        return depth_input

    def predict_poses(self, inputs, disps=None):
        """
        Predict poses using optical flow and appearance flow with AF_OF decoder
        """
        outputs = {}

        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # Generate optical flow (position)
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:
                        # Forward position (optical flow)
                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear",
                            align_corners=True
                        )
                        outputs[("registration", scale, f_i)] = self.spatial_transform(
                            inputs[("color", f_i, 0)],
                            outputs[("position", "high", scale, f_i)]
                        )

                        # Reverse position
                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear",
                            align_corners=True
                        )
                        outputs[("occu_mask_backward", scale, f_i)], outputs[("occu_map_backward", scale, f_i)] = \
                            self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)]
                        )

                    # Transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:
                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear",
                            align_corners=True
                        )
                        outputs[("refined", scale, f_i)] = (
                            outputs[("transform", "high", scale, f_i)] * outputs[("occu_mask_backward", 0, f_i)].detach()
                            + inputs[("color", 0, 0)]
                        )
                        outputs[("refined", scale, f_i)] = torch.clamp(
                            outputs[("refined", scale, f_i)], min=0.0, max=1.0
                        )

                    # Pose prediction using AF_OF decoder (if disps is provided)
                    if disps is not None:
                        # Extract optical flow (2-channel) and appearance flow (3-channel) at highest resolution
                        optical_flow = F.interpolate(
                            outputs[("position", "high", 0, f_i)],
                            [self.models["pose"].input_height, self.models["pose"].input_width],
                            mode="bilinear",
                            align_corners=True
                        )  # [B, 2, H, W]
                        appearance_flow = F.interpolate(
                            outputs[("transform", "high", 0, f_i)],
                            [self.models["pose"].input_height, self.models["pose"].input_width],
                            mode="bilinear",
                            align_corners=True
                        )  # [B, 3, H, W]
                        
                        # Use AF_OF decoder for pose prediction
                        if self.opt.learn_intrinsics:
                            axisangle, translation, intrinsics = self.models["pose"](
                                appearance_flow, 
                                optical_flow
                            )
                            outputs["estimated_intrinsics"] = intrinsics
                        else:
                            axisangle, translation = self.models["pose"](
                                appearance_flow, 
                                optical_flow
                            )

                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0]
                        )
        return outputs
