import os
import torch
import torch.nn.functional as F
from DARES.networks.dares_peft import DARES
from DARES.networks.resnet_encoder import AttentionalResnetEncoder,MultiHeadAttentionalResnetEncoder
from DARES.networks.optical_flow_decoder import PositionDecoder
from DARES.networks.appearance_flow_decoder import TransformDecoder
from DARES.layers import transformation_from_parameters
from exps.trainer_dares import DARESTrainer

class TrainerNoPose(DARESTrainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize training state
        self.training = True

    def load_model(self):
        # Initialize depth model
        dares_depth = DARES(use_dora=True,target_modules=['query', 'value'],full_finetune=False)
        dares_pose = DARES(use_dora=True,target_modules=['query', 'value'],full_finetune=True)
        encoders = {
            "depth_model": dares_depth,
            "dares_tar": dares_pose,
            "dares_ref": dares_pose,
            "position_encoder": MultiHeadAttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2),
            "transform_encoder": MultiHeadAttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2)
        }
        
        # Initialize pose decoder
        num_ch_enc = [64, 64, 64, 64]  # This should match DARES output
        
        # Create cross-attention pose decoder
        from DARES.networks.pose_decoder import CrossAttnPoseDecoder_with_intrinsics
        self.models["pose"] = CrossAttnPoseDecoder_with_intrinsics(
            num_ch_enc=num_ch_enc,
            num_input_features=2,  # ref and tar
            predict_intrinsics=self.opt.learn_intrinsics,
            image_width=self.opt.width,
            image_height=self.opt.height,
            auto_scale=True,
            num_heads=self.dares_config['num_heads'],
            hidden_dim=self.dares_config['hidden_dim'],
            use_scales=self.dares_config['use_scales'],
            fusion_method=self.dares_config['fusion_method']
        )
        
        # No pose encoder/decoder since this is a "no pose" trainer
        decoders = {
            "position": PositionDecoder(encoders["position_encoder"].num_ch_enc, self.opt.scales),
            "transform": TransformDecoder(encoders["transform_encoder"].num_ch_enc, self.opt.scales)
        }

        all_models = [
            ("depth_model",         encoders["depth_model"],        self.param_monodepth),
            ("dares_tar",           encoders["dares_tar"],          self.param_monodepth),
            ("dares_ref",           encoders["dares_ref"],          self.param_monodepth),
            ("pose",                self.models["pose"],            self.param_monodepth),
            ("position_encoder",    encoders["position_encoder"],   self.param_pose_net),
            ("position",            decoders["position"],           self.param_pose_net),
            ("transform_encoder",   encoders["transform_encoder"],  self.param_pose_net),
            ("transform",           decoders["transform"],          self.param_pose_net)
        ]
        # Initialize models and parameters, and move to device
        for model_name, model_instance, param_list in all_models:
            self.models[model_name] = model_instance
            param_list += list(filter(lambda p: p.requires_grad, model_instance.parameters()))
            # for name, param in model_instance.named_parameters():
            #     print(f"\033[94m{name} requires grad: {param.requires_grad}\033[0m")
            self.models[model_name].to(self.device)
        # Load pretrained weights if available
        if self.pretrained_root_dir is not None:
            model_paths = [
            "depth_model",
            "pose",
            "position_encoder",
            "position",
            "transform_encoder",
            "transform"
            ]
            for model_name in model_paths:
                model_path = os.path.join(self.pretrained_root_dir, "best", f"{model_name}.pth")
                if os.path.exists(model_path):
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                    model_state_dict = self.models[model_name].state_dict()
                    filtered_state_dict = {}
                    for key, value in state_dict.items():
                        if key in model_state_dict:
                            model_shape = model_state_dict[key].shape
                            value_shape = value.shape
                            
                            if model_shape == value_shape:
                                filtered_state_dict[key] = value
                            else:
                                # Try to patch size mismatches by padding with near-zero values
                                try:
                                    if len(model_shape) == len(value_shape):
                                        # Create tensor with model shape, filled with small random values
                                        patched_tensor = torch.randn(model_shape, device=value.device) * 1e-6
                                        
                                        # Copy original values to the overlapping region
                                        slices = tuple(slice(0, min(m, v)) for m, v in zip(model_shape, value_shape))
                                        patched_tensor[slices] = value[slices]
                                        
                                        filtered_state_dict[key] = patched_tensor
                                        print(f"\033[92mPatched parameter {key} from {value_shape} to {model_shape}\033[0m")
                                    else:
                                        print(f"\033[93mSkipping parameter {key} due to dimension mismatch: {value_shape} vs {model_shape}\033[0m")
                                except Exception as e:
                                    print(f"\033[93mFailed to patch parameter {key}: {e}\033[0m")
                        else:
                            print(f"\033[93mSkipping parameter {key} - not found in model\033[0m")
                    self.models[model_name].load_state_dict(filtered_state_dict, strict=False)
                    print(f"\033[94mLoaded pretrained weights for {model_name} from {model_path}\033[0m")
                else:
                    print(f"\033[93mNo pretrained weights found for {model_name} at {model_path}\033[0m")
    
    def _setup_parameter_groups(self):
        """Setup parameter groups for depth optimization only (no pose)"""
        # Depth model parameters
        self.param_monodepth = list(self.models["depth_model"].parameters())
        
        # Position and transform model parameters (including pose decoder)
        self.param_pose_net = (
            list(self.models["pose"].parameters()) +
            list(self.models["position_encoder"].parameters()) +
            list(self.models["position"].parameters()) +
            list(self.models["transform_encoder"].parameters()) +
            list(self.models["transform"].parameters())
        )
        
        print(f"Depth model parameters: {len(self.param_monodepth)}")
        print(f"Position/Transform/Pose model parameters: {len(self.param_pose_net)}")
    
    def get_depth_input(self, inputs):
        return inputs['color_aug', 0, 0]
    
    def set_train(self, train_position_only=False, freeze_depth=False):
        """Enable or disable gradients based on optimizer groups."""
        # Set trainer training state
        self.training = True
        # Call parent implementation but handle missing pose models gracefully
        for name, model in self.models.items():
            model.train()
        # Freeze/unfreeze depth parameters
        for group in self.optimizer_depth.param_groups:
            for p in group['params']:
                p.requires_grad = not train_position_only
        # Freeze/unfreeze pose parameters
        for group in self.optimizer_pose.param_groups:
            for p in group['params']:
                p.requires_grad = train_position_only
        if freeze_depth:
            for param in self.models["depth_model"].parameters():
                param.requires_grad = False
            self.models["depth_model"].eval()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        # Set trainer training state
        self.training = False
        for name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
    
    def predict_poses(self, inputs, outputs):
        """
        No-pose implementation: generate optical flow and appearance flow outputs 
        but use dummy poses (no camera movement assumption)
        """
        
        if outputs is None:
            outputs = {}
        
        # Generate position/optical flow and appearance flow outputs using traditional encoders
        # This follows the same structure as trainer_abc.py but without pose estimation
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs[("color_aug", f_i, 0)] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    batch_size = inputs[("color_aug", 0, 0)].shape[0]
                    device = inputs[("color_aug", 0, 0)].device
                    
                    # === Generate optical flow outputs using position encoder/decoder ===
                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # Position (optical flow)
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:
                        # Forward position/optical flow
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

                        # Reverse position/optical flow
                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear",
                            align_corners=True
                        )
                        
                        # Generate occlusion masks
                        outputs[("occu_mask_backward", scale, f_i)], outputs[("occu_map_backward", scale, f_i)] = \
                            self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)]
                        )

                    # === Generate appearance flow outputs using transform encoder/decoder ===
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
                    
                    # === Generate dummy pose outputs (no camera movement assumption) ===
                    # Create identity/dummy poses representing no camera movement
                    outputs[("axisangle", 0, f_i)] = torch.zeros((batch_size, 1, 1, 3), device=device)
                    outputs[("translation", 0, f_i)] = torch.zeros((batch_size, 1, 1, 3), device=device)
                    
                    # Create transformation matrix using the same function as other trainers
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        outputs[("axisangle", 0, f_i)][:, 0], 
                        outputs[("translation", 0, f_i)][:, 0]
                    )
                    
                    # If intrinsics are learned, provide dummy intrinsics
                    if self.opt.learn_intrinsics:
                        outputs["estimated_intrinsics"] = torch.tensor([
                            [self.opt.width/2, 0, self.opt.width/2],
                            [0, self.opt.height/2, self.opt.height/2],
                            [0, 0, 1]
                        ], device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        return outputs

    def compute_depth_losses(self, inputs, outputs):
        """Compute losses for depth optimization"""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_reprojection = 0
            # Add self-ssi accumulation
            loss_self_ssi = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                
                # Compute reprojection loss
                loss_reprojection += (
                    self.compute_reprojection_loss(
                        outputs[("color", frame_id, scale)], 
                        outputs[("refined", scale, frame_id)]
                    ) * occu_mask_backward
                ).sum() / occu_mask_backward.sum()
                # Add self-ssi loss term
                if self.opt.self_ssi and self.epoch >= self.opt.warm_up_step:
                    loss_self_ssi += self.self_ssi(
                        outputs[("disp", scale)],
                        outputs[("t_to_s_disp", frame_id, scale)],
                        occu_mask_backward)

            # Add weighted self-ssi constraint
            if self.opt.self_ssi and self.epoch >= self.opt.warm_up_step:
                loss += self.opt.self_ssi_constraint * loss_self_ssi

            # Compute disparity smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            from DARES.layers import get_smooth_loss
            disp_smooth_loss = get_smooth_loss(norm_disp, color)

            # Combine losses
            loss += loss_reprojection / 2.0
            loss += self.opt.disparity_smoothness * disp_smooth_loss / (2 ** scale)

            # supervised loss
            if self.use_supervised_loss:
                loss += self.compute_supervised_loss(inputs, outputs)  * 0.001
            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_position_losses(self, inputs, outputs):
        """Compute losses for position optimization"""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                from DARES.layers import get_smooth_loss
                loss_smooth_registration += get_smooth_loss(outputs[("position", scale, frame_id)], color)
                loss_registration += (
                    self.compute_reprojection_loss(
                        outputs[("registration", scale, frame_id)], 
                        outputs[("refined", scale, frame_id)].detach()
                    ) * occu_mask_backward
                ).sum() / occu_mask_backward.sum()

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)
            
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses