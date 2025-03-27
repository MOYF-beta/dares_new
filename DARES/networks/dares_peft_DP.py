# %%
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
                 predict_intrinsics=False,
                 simplified_intrinsic=False,
                 image_width=None,
                 image_height=None):
        super().__init__()
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.predict_intrinsics = predict_intrinsics
        
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
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
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
        dynamic_scale = self.scale_sigmoid(dynamic_scale) * 10.0 + 0.1  # Scale between 0.1 and 10.1
        
        # Apply dynamic scaling
        pose_out = pose_out.view(-1, self.num_frames_to_predict_for, 6)
        dynamic_scale = dynamic_scale.unsqueeze(-1)  # [B, num_frames, 1]
        pose_out = 0.001 * dynamic_scale * pose_out
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

class DARES(nn.Module):
    def __init__(self, 
                 r=[14,14,12,12,10,10,8,8,8,8,8,8], 
                 target_modules=['query', 'value'],
                 pretrained_path=None,
                 use_dora=True, 
                 full_finetune=False):
        super(DARES, self).__init__()
        
        # Load base model
        base_model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.config = base_model.config
        self.backbone = base_model.backbone

        # change the backbone to support two-frame input
        original_embeddings = self.backbone.embeddings
        self.backbone.embeddings = DualFrameEmbeddings(original_embeddings)
        # Configure PEFT with DoRA support
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=r[0],  # Using first r value as default
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules,
            use_dora=use_dora,
        )
        
        # Apply PEFT to backbone
        self.backbone = get_peft_model(self.backbone, peft_config)
        
        # Freeze base model parameters
        for param in self.backbone.parameters():
            param.requires_grad = full_finetune
            
        # Unfreeze LoRA/DoRA parameters
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
        self.head = DepthAnythingDepthEstimationHead(model_head)

    def save_parameters(self, filename: str) -> None:
        """Save PEFT parameters"""
        self.backbone.save_pretrained(filename)
        print(f'Saved {"DoRA" if self.backbone.peft_config.use_dora else "LoRA"} parameters to {filename}')

    def load_parameters(self, filename: str) -> None:
        """Load PEFT parameters"""
        self.backbone.load_adapter(filename, "default")
        print(f'Loaded {"DoRA" if self.backbone.peft_config.use_dora else "LoRA"} parameters from {filename}')

    def forward(self, pixel_values, two_frame=False):

        self.backbone.embeddings.set_two_frame_mode(two_frame)
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=None, output_attentions=None
        )
        hidden_states = outputs.feature_maps
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        
        outputs = {}
        outputs[("disp", 0)] = self.head(hidden_states[3], height, width)
        outputs[("disp", 1)] = self.head(hidden_states[2], height/2, width/2)
        outputs[("disp", 2)] = self.head(hidden_states[1], height/4, width/4)
        outputs[("disp", 3)] = self.head(hidden_states[0], height/8, width/8)

        return outputs

# %% 

if __name__ == "__main__":
    test_input = torch.randn(1, 3, 256, 320)
    model_dora = DARES(
        r=[14,14,12,12,10,10,8,8,8,8,8,8],
        target_modules=['query', 'value'],
        use_dora=True
    )
    print("\nTesting DoRA model:")
    output_dora = model_dora(test_input)
    print(output_dora[("disp", 0)].shape)

# %%
print(model_dora.backbone.embeddings)
'''
Dinov2Embeddings(
  (patch_embeddings): Dinov2PatchEmbeddings(
    (projection): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
  )
  (dropout): Dropout(p=0.0, inplace=False)
)
'''

# Replace the backbone embeddings with our dual frame embeddings

# %%
# Test single frame input
print("\nTesting single frame input:")
single_frame_input = torch.randn(2, 3, 256, 320)
model_dora.backbone.embeddings.set_two_frame_mode(False)
single_frame_output = model_dora(single_frame_input)
print(f"Single frame output shape: {single_frame_output[('disp', 0)].shape}")

# Test two frame input (concatenated along channel dimension)
print("\nTesting two frame input:")
frame1 = torch.randn(2, 3, 256, 320)
frame2 = torch.randn(2, 3, 256, 320)
two_frame_input = torch.cat([frame1, frame2], dim=1)  # Concatenate along channel dimension to get [1, 6, 256, 320]
print(f"Two frame input shape: {two_frame_input.shape}")
model_dora.backbone.embeddings.set_two_frame_mode(True)
two_frame_output = model_dora(two_frame_input, two_frame=True)
print(f"Two frame output shape: {two_frame_output[('disp', 0)].shape}")