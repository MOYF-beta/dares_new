from transformers import DepthAnythingForDepthEstimation
import torch
import torch.nn as nn
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

from DARES.networks.refine_net import Refine_net

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

class DARES(nn.Module):
    def __init__(self, 
                 r=[14,14,12,12,10,10,8,8,8,8,8,8], 
                 target_modules=['query', 'value'],
                 pretrained_path=None, 
                 enable_refine_net=False, 
                 num_blocks=4, 
                 feat_channels=64,
                 use_dora=True, full_finetune = False):  # 添加use_dora参数
        super(DARES, self).__init__()
        
        # Load base model
        base_model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.config = base_model.config
        self.backbone = base_model.backbone
        
        # Configure PEFT with DoRA support
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=r[0],  # Using first r value as default
            lora_alpha=16,  # 增加alpha值以匹配示例
            lora_dropout=0.1,
            target_modules=target_modules,
            use_dora=use_dora,  # 启用或禁用DoRA
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
        
        if enable_refine_net:
            self.refine_net = Refine_net(num_blocks, feat_channels)
        self.enable_refine_net = enable_refine_net

    def save_parameters(self, filename: str) -> None:
        """Save PEFT parameters"""
        self.backbone.save_pretrained(filename)
        print(f'Saved {"DoRA" if self.backbone.peft_config.use_dora else "LoRA"} parameters to {filename}')

    def load_parameters(self, filename: str) -> None:
        """Load PEFT parameters"""
        self.backbone.load_adapter(filename, "default")
        print(f'Loaded {"DoRA" if self.backbone.peft_config.use_dora else "LoRA"} parameters from {filename}')

    def forward(self, pixel_values):
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

        if self.enable_refine_net:
            for i in range(4):
                outputs[("disp", i)] = self.refine_net(outputs[("disp", i)], pixel_values)
        return outputs


if __name__ == "__main__":
    # Test the model with both LoRA and DoRA configurations
    # Test LoRA
    model_lora = DARES(
        r=[14,14,12,12,10,10,8,8,8,8,8,8],
        target_modules=['query', 'value'],
        use_dora=False
    )
    print("Testing LoRA model:")
    test_input = torch.randn(1, 3, 256, 320)
    output_lora = model_lora(test_input)
    print(output_lora[("disp", 0)].shape)

    # Test DoRA
    model_dora = DARES(
        r=[14,14,12,12,10,10,8,8,8,8,8,8],
        target_modules=['query', 'value'],
        use_dora=True
    )
    print("\nTesting DoRA model:")
    output_dora = model_dora(test_input)
    print(output_dora[("disp", 0)].shape)