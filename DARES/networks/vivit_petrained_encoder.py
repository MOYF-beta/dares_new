import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import VivitForVideoClassification, VivitModel

class VivitLoraEncoder(nn.Module):
    """ViViT encoder with LoRA support and ResNet-compatible interface"""
    def __init__(self, num_input_images=2, pretrained=True, img_size=(224, 224)):
        super().__init__()
        
        # Initialize base model
        pretrained_model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
        config = pretrained_model.config
        config.video_size = [num_input_images, img_size[0], img_size[1]]
        
        # Create base model
        self.base_model = VivitModel(config)
        if pretrained:
            self.base_model.load_state_dict(pretrained_model.state_dict(), strict=False)
        
        # Enable gradients for embeddings
        for param in self.base_model.embeddings.parameters():
            param.requires_grad = True
            
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["query", "value"],  # Apply LoRA to attention layers
            lora_dropout=0.1,
            bias="none"
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Define output channels (matching hidden size across layers)
        self.num_ch_enc = [768] * 5  # Assuming 5 output features needed to match ResNet interface
        
    def process_features(self, hidden_states):
        """Convert encoder output to spatial features"""
        # Remove class token and reshape
        features = hidden_states[:, 1:, :]
        batch_size, seq_len, hidden_size = features.shape
        h = w = int((seq_len) ** 0.5)  # Assuming square image
        
        # Reshape to spatial format (B, H, W, C)
        features = features.view(batch_size, h, w, hidden_size)
        
        # Reorder to (B, C, H, W) for convolution compatibility
        features = features.permute(0, 3, 1, 2)
        
        return features
        
    def forward(self, input_images):
        """
        Forward pass returning list of features at different scales
        Args:
            input_images: Tensor of shape (B, num_input_images, C, H, W)
        Returns:
            list of features at different scales
        """
        B, N, C, H, W = input_images.shape
        
        # Reshape input to match ViViT's expected format
        # From (B, N, C, H, W) to (B, C, N, H, W)
        pixel_values = input_images.permute(0, 2, 1, 3, 4)
        
        # Get embeddings
        embedding_output = self.model.embeddings(pixel_values, interpolate_pos_encoding=True)
        
        # Pass through encoder
        encoder_outputs = self.model.encoder(
            embedding_output,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Process all hidden states
        features = []
        for hidden_states in encoder_outputs.hidden_states:
            features.append(self.process_features(hidden_states))
            
        # Match ResNet interface expectation of 5 feature levels
        output_features = [features[0]]  # First layer
        for _ in range(4):  # Repeat last feature to match expected 5 levels
            output_features.append(output_features[-1])
            
        return output_features

# Example usage:
if __name__ == "__main__":
    # Create model
    model = VivitLoraEncoder(num_input_images=2)
    
    # Create sample input (B, N, C, H, W)
    x = torch.randn(1, 3, 2, 224, 224)
    
    # Get features
    features = model(x)
    
    # Print shapes
    for i, feat in enumerate(features):
        print(f"Feature level {i} shape:", feat.shape)