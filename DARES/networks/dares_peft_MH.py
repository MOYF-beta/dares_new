# %%

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))

from transformers import DepthAnythingForDepthEstimation
import torch
import torch.nn as nn
import copy
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

class DualFrameEmbeddings(nn.Module):
    def __init__(self, original_embeddings):
        super(DualFrameEmbeddings, self).__init__() 
        self.embeddings_a = original_embeddings
        # Deep copy the original embeddings for the second frame
        self.embeddings_b = copy.deepcopy(original_embeddings)
        
        # 3D convolution to fuse features from two frames
        self.fusion_conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
        self.two_frame = True  
    
    def set_two_frame_mode(self, mode):
        self.two_frame = mode
    def forward(self, x):
        if not self.two_frame:
            # Single frame mode - use original embeddings
            return self.embeddings_a(x)
        else:
            # Two frame mode
            B, C, H, W = x.shape
            x1, x2 = torch.split(x, C//2, dim=1)
            
            embeddings1 = self.embeddings_a(x1)
            embeddings2 = self.embeddings_b(x2)
            
            # Stack embeddings along a new dimension
            stacked = torch.stack([embeddings1, embeddings2], dim=1)  
            fused = self.fusion_conv(stacked)
            return fused.squeeze(1) 

class DepthAnythingDepthEstimationHead(nn.Module):
    def __init__(self, model_head):
        super().__init__()
        self.conv1 = model_head.conv1
        self.conv2 = model_head.conv2
        self.activation1 = nn.ReLU()
        self.conv3 = model_head.conv3
        self.activation2 = nn.Sigmoid()

    def forward(self, hidden_states, height, width):
        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (int(height), int(width)),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        return predicted_depth


class DepthAnythingPoseEstimationHead(nn.Module):
    def __init__(self, 
                 num_frames_to_predict_for=1,
                 predict_intrinsics=True,
                 simplified_intrinsic=False,
                 image_width=None,
                 image_height=None,
                 scale_range=(1e-3, 1e3)):
        super().__init__()
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.predict_intrinsics = predict_intrinsics
        self.scele_range = scale_range
        if predict_intrinsics:
            assert image_width is not None and image_height is not None, \
                "image_width and image_height must be provided if predict_intrinsics is True"
            self.image_width = image_width
            self.image_height = image_height
            
            if simplified_intrinsic:
                self.num_param_to_predict = 2
            else:
                self.num_param_to_predict = 4
        
        # Reduced to just one main convolutional layer instead of two
        self.conv = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.pose_conv = nn.Conv2d(256, 6 * num_frames_to_predict_for, kernel_size=1)
        
        # Simplified scale prediction path
        self.scale_pool = nn.AdaptiveAvgPool2d(1)
        self.scale_fc = nn.Linear(256, num_frames_to_predict_for)
        self.scale_sigmoid = nn.Sigmoid()
        
        if predict_intrinsics:
            # Simplified intrinsics prediction path
            self.intrinsics_pool = nn.AdaptiveAvgPool2d(1)
            self.intrinsics_fc = nn.Linear(256, self.num_param_to_predict)
            self.softplus = nn.Softplus()

    def forward(self, hidden_states, height, width):
        x = self.conv(hidden_states)
        x = nn.functional.interpolate(
            x,
            (int(height), int(width)),
            mode="bilinear",
            align_corners=True,
        )
        x = self.activation(x)
        features = x  # Save features for other prediction paths
        
        # Pose prediction
        pose_out = self.pose_conv(features)
        pose_out = pose_out.mean(3).mean(2)
        
        # NOTE new compoents: dynamic scale prediction
        scale_feat = self.scale_pool(features).view(features.size(0), -1)
        dynamic_scale = self.scale_fc(scale_feat)
        dynamic_scale = self.scale_sigmoid(dynamic_scale) * self.scele_range[1] + self.scele_range[0]
        
        # Apply dynamic scaling
        pose_out = pose_out.view(-1, self.num_frames_to_predict_for, 6)
        dynamic_scale = dynamic_scale.unsqueeze(-1)  # [B, num_frames, 1]
        pose_out = dynamic_scale * pose_out
        pose_out = pose_out.view(-1, self.num_frames_to_predict_for, 1, 6)
        
        axisangle = pose_out[..., :3]
        translation = pose_out[..., 3:]
        
        if self.predict_intrinsics:
            intrinsics_feat = self.intrinsics_pool(features).view(features.size(0), -1)
            intrinsics = self.intrinsics_fc(intrinsics_feat)
            
            batch_size = hidden_states.shape[0]
            device = hidden_states.device
            
            intrinsics_mat = torch.eye(4).unsqueeze(0).to(device)
            intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
            
            foci = (intrinsics[:, :2] + 0.5) * torch.tensor([self.image_width, self.image_height]).to(device)
            foci_mat = self.softplus(torch.diag_embed(foci))
            
            if self.num_param_to_predict == 4:
                offsets = (intrinsics[:, 2:] + 0.5) * torch.tensor([self.image_width, self.image_height]).to(device)
            else:
                offsets = torch.ones((batch_size, 2), device=device) * 0.5

            intrinsics_mat[:, :2, :2] = foci_mat
            intrinsics_mat[:, :2, 2:3] = offsets.unsqueeze(-1)
            
            return axisangle, translation, intrinsics_mat
        else:
            return axisangle, translation


class LightOpticalFlowEstimationHead(nn.Module):
    """轻量化光流估计Head，输出2通道光流，结构简洁"""
    def __init__(self, in_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
    def forward(self, hidden_states, height, width):
        x = self.conv1(hidden_states)
        x = self.act1(x)
        x = self.conv2(x)
        x = nn.functional.interpolate(x, (int(height), int(width)), mode="bilinear", align_corners=True)
        return x

class LightAppearanceFlowEstimationHead(nn.Module):
    """轻量化外观流估计Head，输出3通道，结构简洁"""
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
    def forward(self, hidden_states, height, width):
        x = self.conv1(hidden_states)
        x = self.act1(x)
        x = self.conv2(x)
        x = nn.functional.interpolate(x, (int(height), int(width)), mode="bilinear", align_corners=True)
        x = self.tanh(x)
        return x

class DARES_MH(nn.Module):
    def hidden_states_channels(self,model_head):
        # 假设与DepthAnythingDepthEstimationHead输入一致
        # 取conv1的输入通道
        return model_head.conv1.in_channels

    def __init__(self, 
                 r=[14,14,12,12,10,10,8,8,8,8,8,8], 
                 target_modules=['query', 'value'],
                 pretrained_path=None,
                 use_dora=True, 
                 full_finetune=False,
                 image_size=(256, 320),
                 heads=["depth", "pose"]):
        super(DARES_MH, self).__init__()
        self.image_size = image_size
        # Load base model
        base_model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.config = base_model.config
        self.backbone = base_model.backbone

        # Change the backbone to support two-frame input
        original_embeddings = self.backbone.embeddings
        self.backbone.embeddings = DualFrameEmbeddings(original_embeddings)

        # 配置4个adapter，分别对应4种模式
        self.adapter_map = {
            "depth": "adapter_depth",
            "pose": "adapter_pose",
            "optical_flow": "adapter_optical_flow",
            "appearance_flow": "adapter_appearance_flow"
        }
        for mode, adapter_name in self.adapter_map.items():
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=r[0],
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=target_modules,
                use_dora=use_dora,
            )
            if mode == "depth":
                self.backbone = get_peft_model(self.backbone, config, adapter_name=adapter_name)
            else:
                self.backbone.add_adapter(adapter_name, config)
        # 默认使用depth adapter
        self.backbone.set_adapter(self.adapter_map["depth"])
        self.current_adapter = self.adapter_map["depth"]

        # Freeze base model parameters
        for param in self.backbone.parameters():
            param.requires_grad = full_finetune
        # Unfreeze LoRA/DoRA parameters for all adapters
        for name, param in self.backbone.named_parameters():
            if "lora" in name or (use_dora and "dora" in name):
                param.requires_grad = True
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, weights_only=True)
            base_model.load_state_dict(state_dict, strict=False)
        self.neck = base_model.neck
        for param in self.neck.parameters():
            param.requires_grad = full_finetune
        model_head = base_model.head
        # 动态初始化head
        self.heads_dict = nn.ModuleDict()
        self.available_heads = []
        if "depth" in heads:
            self.heads_dict["depth"] = DepthAnythingDepthEstimationHead(model_head)
            self.available_heads.append("depth")
        if "pose" in heads:
            self.heads_dict["pose"] = DepthAnythingPoseEstimationHead(
                num_frames_to_predict_for=1,
                predict_intrinsics=True,
                image_width=self.image_size[0],
                image_height=self.image_size[1]
            )
            self.available_heads.append("pose")
        if "optical_flow" in heads:
            self.heads_dict["optical_flow"] = LightOpticalFlowEstimationHead(in_channels=self.hidden_states_channels(model_head))
            self.available_heads.append("optical_flow")
        if "appearance_flow" in heads:
            self.heads_dict["appearance_flow"] = LightAppearanceFlowEstimationHead(in_channels=self.hidden_states_channels(model_head))
            self.available_heads.append("appearance_flow")

    def forward(self, pixel_values, mode="depth"):
        # 根据mode自动切换adapter和embedding帧数
        if mode not in self.adapter_map:
            raise ValueError(f"Unknown mode: {mode}, available: {list(self.adapter_map.keys())}")
        adapter_name = self.adapter_map[mode]
        if self.current_adapter != adapter_name:
            self.backbone.set_adapter(adapter_name)
            self.current_adapter = adapter_name
        # 深度模式用单帧embedding，其余用双帧
        use_two_frame = (mode != "depth")
        self.backbone.embeddings.set_two_frame_mode(use_two_frame)
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=None, output_attentions=None
        )
        hidden_states = outputs.feature_maps
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        # 选择head进行推理
        if mode == "depth":
            out = {}
            out[("disp", 0)] = self.heads_dict["depth"](hidden_states[3], height, width)
            out[("disp", 1)] = self.heads_dict["depth"](hidden_states[2], height/2, width/2)
            out[("disp", 2)] = self.heads_dict["depth"](hidden_states[1], height/4, width/4)
            out[("disp", 3)] = self.heads_dict["depth"](hidden_states[0], height/8, width/8)
            return out
        elif mode == "pose":
            return self.heads_dict["pose"](hidden_states[3], height, width)
        elif mode == "optical_flow":
            out = {}
            out[("flow", 0)] = self.heads_dict["optical_flow"](hidden_states[3], height, width)
            out[("flow", 1)] = self.heads_dict["optical_flow"](hidden_states[2], height/2, width/2)
            out[("flow", 2)] = self.heads_dict["optical_flow"](hidden_states[1], height/4, width/4)
            out[("flow", 3)] = self.heads_dict["optical_flow"](hidden_states[0], height/8, width/8)
            return out
        elif mode == "appearance_flow":
            out = {}
            out[("transform", 0)] = self.heads_dict["appearance_flow"](hidden_states[3], height, width)
            out[("transform", 1)] = self.heads_dict["appearance_flow"](hidden_states[2], height/2, width/2)
            out[("transform", 2)] = self.heads_dict["appearance_flow"](hidden_states[1], height/4, width/4)
            out[("transform", 3)] = self.heads_dict["appearance_flow"](hidden_states[0], height/8, width/8)
            return out
        else:
            raise ValueError(f"Unknown mode: {mode}, available: {self.available_heads}")
# %% 

if __name__ == "__main__":
    test_input = torch.randn(1, 3, 256, 320)
    test_dual_frame_input = torch.randn(1, 6, 256, 320)  # Two frames concatenated
    model_dora = DARES_MH(
        r=[14,14,12,12,10,10,8,8,8,8,8,8],
        target_modules=['query', 'value'],
        use_dora=True,
        heads=["depth", "pose", "optical_flow", "appearance_flow"]
    )
    print("\nTesting DoRA model (depth mode):")
    output_dora = model_dora(test_input, mode="depth")
    print(f"Depth output shape: {output_dora[('disp', 0)].shape}")

    print("\nTesting pose mode:")
    pose_out = model_dora(test_dual_frame_input, mode="pose")
    if isinstance(pose_out, tuple):
        print(f"Axis-angle shape: {pose_out[0].shape}")
        print(f"Translation shape: {pose_out[1].shape}")
        print(f"Intrinsics shape: {pose_out[2].shape if len(pose_out) > 2 else 'N/A'}")
    else:
        print(f"Pose output shape: {pose_out.shape}")

    print("\nTesting optical flow mode:")
    flow_out = model_dora(test_dual_frame_input, mode="optical_flow")
    print(f"Optical flow output shape: {flow_out.shape}")

    print("\nTesting appearance flow mode:")
    af_out = model_dora(test_dual_frame_input, mode="appearance_flow")
    print(f"Appearance flow output shape: {af_out.shape}")

    # Test single frame input (depth)
    print("\nTesting single frame input (depth mode):")
    single_frame_input = torch.randn(8, 3, 256, 320)
    single_frame_output = model_dora(single_frame_input, mode="depth")
    print(f"Single frame output shape: {single_frame_output[('disp', 0)].shape}")

    # Test two frame input (other modes)
    print("\nTesting two frame input (pose mode):")
    frame1 = torch.randn(8, 3, 256, 320)
    frame2 = torch.randn(8, 3, 256, 320)
    two_frame_input = torch.cat([frame1, frame2], dim=1)  # [8, 6, 256, 320]
    pose_out = model_dora(two_frame_input, mode="pose")
    if isinstance(pose_out, tuple):
        print(f"Axis-angle shape: {pose_out[0].shape}")
        print(f"Translation shape: {pose_out[1].shape}")
        print(f"Intrinsics shape: {pose_out[2].shape if len(pose_out) > 2 else 'N/A'}")
    else:
        print(f"Pose output shape: {pose_out.shape}")
    print("\nTesting two frame input (optical_flow mode):")
    flow_out = model_dora(two_frame_input, mode="optical_flow")
    print(f"Optical flow output shape: {flow_out.shape}")
    print("\nTesting two frame input (appearance_flow mode):")
    af_out = model_dora(two_frame_input, mode="appearance_flow")
    print(f"Appearance flow output shape: {af_out.shape}")