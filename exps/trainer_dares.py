"""
DARES-based Trainer for Pose Estimation

This trainer inherits from the base Trainer class and replaces the traditional 
pose encoder with DARES (Depth Anything with PEFT) as a feature extractor for pose prediction.
"""

import torch
from exps.trainer_abc import Trainer
from DARES.networks.pose_decoder import CrossAttnPoseDecoder_with_intrinsics
from DARES.layers import *
from DARES.utils import *


class DARESTrainer(Trainer):
    """
    Trainer that uses DARES models as feature extractors for pose prediction.
    Inherits from the base Trainer class and overrides pose-related methods.
    """
    
    def __init__(self, model_name, log_dir, options, train_eval_ds={},
                 pretrained_root_dir=None, merge_val_as_train=False, 
                 use_supervised_loss=True, debug=False, use_af_pose=False,
                 dares_config=None):
        """
        Initialize DARES trainer
        
        Args:
            dares_config: Dictionary containing DARES configuration
                - use_dora: Whether to use DoRA instead of LoRA
                - r: LoRA rank values for different layers
                - target_modules: Target modules for PEFT
                - fusion_method: Feature fusion method for pose decoder
                - use_scales: Which scales to use for pose prediction
        """
        self.dares_config = dares_config or {
            'use_dora': True,
            'r': [14, 14, 12, 12, 10, 10, 8, 8, 8, 8, 8, 8],
            'target_modules': ['query', 'value'],
            'fusion_method': 'weighted_sum',
            'use_scales': None,  # Use all scales by default
            'hidden_dim': 256,
            'num_heads': 8
        }
        
        # Call parent constructor
        super().__init__(model_name, log_dir, options, train_eval_ds,
                        pretrained_root_dir, merge_val_as_train, 
                        use_supervised_loss, debug, use_af_pose)
        
        # Get channel dimensions from DARES backbone
        # DARES typically outputs 4 scales with 64 channels each

        
        # Move models to device
        for model_name, model in self.models.items():
            model.to(self.device)
        
        # Setup parameter groups for optimizers
        # self._setup_parameter_groups()
        
        # Load pretrained weights if provided
        if self.pretrained_root_dir:
            self._load_pretrained_weights()
    
    # def _setup_parameter_groups(self):
    #     """Setup parameter groups for depth and pose optimization"""
    #     # Depth model parameters
    #     self.param_monodepth = list(self.models["depth_model"].parameters())
        
    #     # Pose model parameters (DARES feature extractors + pose decoder)
    #     self.param_pose_net = (
    #         list(self.models["dares_ref"].parameters()) +
    #         list(self.models["dares_tar"].parameters()) +
    #         list(self.models["pose"].parameters())
    #     )
        
    #     print(f"Depth model parameters: {len(self.param_monodepth)}")
    #     print(f"Pose model parameters: {len(self.param_pose_net)}")
    
    def _load_pretrained_weights(self):
        """Load pretrained weights for DARES models"""
        print(f"Loading pretrained weights from {self.pretrained_root_dir}")
        
        # Load depth model weights
        try:
            self.models["depth_model"].load_parameters(
                f"{self.pretrained_root_dir}/depth_model"
            )
        except Exception as e:
            print(f"Warning: Could not load depth model weights: {e}")
        
        # Load pose model weights if available
        try:
            self.models["dares_ref"].load_parameters(
                f"{self.pretrained_root_dir}/dares_ref"
            )
            self.models["dares_tar"].load_parameters(
                f"{self.pretrained_root_dir}/dares_tar"
            )
        except Exception as e:
            print(f"Warning: Could not load pose model weights: {e}")
    
    def get_depth_input(self, inputs):
        """Prepare input for depth model (DARES)"""
        # Use center frame for depth prediction
        return inputs[("color_aug", 0, 0)]
    
    def predict_poses(self, inputs, outputs=None):
        """
        Predict poses using DARES feature extractors and cross-attention decoder
        """
        if outputs is None:
            outputs = {}
        
        if self.num_pose_frames == 2:
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # Get reference and target frames
                    ref_frame = inputs[("color_aug", 0, 0)]  # Center frame as reference
                    tar_frame = inputs[("color_aug", f_i, 0)]  # Adjacent frame as target
                    
                    # Extract features using DARES models
                    with torch.no_grad() if not self.training else torch.enable_grad():
                        # Extract hidden states from DARES models
                        ref_hidden_states = self._extract_hidden_states(self.models["dares_ref"], ref_frame)
                        tar_hidden_states = self._extract_hidden_states(self.models["dares_tar"], tar_frame)
                    
                    # Predict pose using cross-attention decoder
                    if self.opt.learn_intrinsics:
                        axisangle, translation, intrinsics = self.models["pose"](
                            ref_hidden_states, tar_hidden_states
                        )
                        outputs["estimated_intrinsics"] = intrinsics
                    else:
                        axisangle, translation = self.models["pose"](
                            ref_hidden_states, tar_hidden_states
                        )
                    
                    # Store pose outputs
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0]
                    )
        
        return outputs
    
    def _extract_hidden_states(self, dares_model, input_tensor):
        """
        Extract hidden states from DARES backbone
        
        Args:
            dares_model: DARES model instance
            input_tensor: Input image tensor [B, 3, H, W]
        
        Returns:
            hidden_states: List of feature tensors from different scales
        """
        # Get outputs from DARES backbone
        outputs = dares_model.backbone.forward_with_filtered_kwargs(
            input_tensor, output_hidden_states=True, output_attentions=None
        )
        
        # Extract feature maps (hidden states)
        hidden_states = outputs.feature_maps
        
        # Process through neck to get multi-scale features
        _, _, height, width = input_tensor.shape
        patch_size = dares_model.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        
        processed_hidden_states = dares_model.neck(hidden_states, patch_height, patch_width)
        
        return processed_hidden_states
    
    def process_batch_position(self, inputs):
        """
        Process batch for position optimization (appearance flow)
        For DARES trainer, we still use the original position estimation
        """
        # For now, keep the original position/transform pipeline
        # This can be extended to use DARES features if needed
        return super().process_batch_position(inputs)
    
    def save_model(self):
        """Save DARES model weights"""
        save_folder = f"{self.log_path}/models/weights_{self.epoch}"
        import os
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Save DARES models using their custom save method
        for model_name in ["depth_model", "dares_ref", "dares_tar"]:
            if model_name in self.models:
                try:
                    model_save_path = f"{save_folder}/{model_name}"
                    # Create parent directory for the model save path
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    self.models[model_name].save_parameters(model_save_path)
                except Exception as e:
                    print(f"Warning: Could not save {model_name}: {e}")
        
        # Save pose decoder normally
        if "pose" in self.models:
            pose_save_path = f"{save_folder}/pose.pth"
            torch.save(self.models["pose"].state_dict(), pose_save_path)
        
        # Save optimizers
        torch.save(self.optimizer_pose.state_dict(), f"{save_folder}/adam_pose.pth")
        torch.save(self.optimizer_depth.state_dict(), f"{save_folder}/adam_depth.pth")
        
        print(f"Saved DARES model weights to {save_folder}")


