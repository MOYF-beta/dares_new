# %% 
from transformers import DepthAnythingForDepthEstimation
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from DinoV2_embedding import Dinov2Embeddings_MultiFrame

class _LoRA_blk(nn.Module):

    def __init__(
            self,
            w: nn.Module,
            linear_a: nn.Module,
            linear_b: nn.Module
    ):
        super().__init__()
        self.w = w
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dim = w.in_features

    def forward(self, x):
        W = self.w(x)
        deltaW = self.linear_b(self.linear_a(x))

        W += deltaW
        return W

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

class LoRAInitializer:
    def __init__(self, model, r=[14,14,12,12,10,10,8,8,8,8,8,8],
                  lora=['mlp','q','v']):
        self.model = model
        self.r = r
        self.lora = lora
        self.w_As = []
        self.w_Bs = []
        self.initialize_lora()

    def initialize_lora(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.model.backbone.encoder.layer):
            dim = blk.attention.attention.query.in_features

            if 'q' in self.lora:
                w_q = blk.attention.attention.query
                w_a_linear_q = nn.Linear(dim, self.r[t_layer_i], bias=False)
                w_b_linear_q = nn.Linear(self.r[t_layer_i], dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                blk.attention.attention.query = _LoRA_blk(w_q, w_a_linear_q, w_b_linear_q)

            if 'v' in self.lora:
                w_v = blk.attention.attention.value
                w_a_linear_v = nn.Linear(dim, self.r[t_layer_i], bias=False)
                w_b_linear_v = nn.Linear(self.r[t_layer_i], dim, bias=False)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attention.attention.value = _LoRA_blk(w_v, w_a_linear_v, w_b_linear_v)

            if 'k' in self.lora:
                w_k = blk.attention.attention.key
                w_a_linear_k = nn.Linear(dim, self.r[t_layer_i], bias=False)
                w_b_linear_k = nn.Linear(self.r[t_layer_i], dim, bias=False)
                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)
                blk.attention.attention.key = _LoRA_blk(w_k, w_a_linear_k, w_b_linear_k)

            if 'mlp' in self.lora:
                mlp_fc1 = blk.mlp.fc1
                mlp_dim_in = mlp_fc1.in_features
                mlp_dim_out = mlp_fc1.out_features
                w_a_linear_mlp_fc1 = nn.Linear(mlp_dim_in, self.r[t_layer_i], bias=False)
                w_b_linear_mlp_fc1 = nn.Linear(self.r[t_layer_i], mlp_dim_out, bias=False)
                self.w_As.append(w_a_linear_mlp_fc1)
                self.w_Bs.append(w_b_linear_mlp_fc1)
                blk.mlp.fc1 = _LoRA_blk(mlp_fc1, w_a_linear_mlp_fc1, w_b_linear_mlp_fc1)

                mlp_fc2 = blk.mlp.fc2
                mlp_dim_in = mlp_fc2.in_features
                mlp_dim_out = mlp_fc2.out_features
                w_a_linear_mlp_fc2 = nn.Linear(mlp_dim_in, self.r[t_layer_i], bias=False)
                w_b_linear_mlp_fc2 = nn.Linear(self.r[t_layer_i], mlp_dim_out, bias=False)
                self.w_As.append(w_a_linear_mlp_fc2)
                self.w_Bs.append(w_b_linear_mlp_fc2)
                blk.mlp.fc2 = _LoRA_blk(mlp_fc2, w_a_linear_mlp_fc2, w_b_linear_mlp_fc2)

        self.reset_parameters()
        print("LoRA params initialized!")

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class DARES_cpe(nn.Module):
    def __init__(self, frame_ids, other_frame_init_weight, 
                 r = [14,14,12,12,10,10,8,8,8,8,8,8], lora = ['q', 'v', 'mlp'],
                 pretrained_path = None,
                 debug_output=False):
        super(DARES_cpe, self).__init__()
        model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.r = r
        self.lora = lora
        self.config = model.config
        self.backbone = model.backbone
        if debug_output:
            with open("model.txt", "w") as f:
                f.write(str(model) + "\n")
            with open("config.txt", "w") as f:
                f.write(str(model.config) + "\n")
            with open("backbone.txt", "w") as f:
                f.write(str(model.backbone) + "\n")
            with open("head.txt", "w") as f:
                f.write(str(model.head) + "\n")
            with open("neck.txt", "w") as f:
                f.write(str(model.neck) + "\n")
        # Initialize LoRA parameters
        self.lora_initializer = LoRAInitializer(model, r, lora)
        # load pretrained model
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path,weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        
        # replace backbone's embedding layer with CPE embedding layer
        self.backbone.embeddings = Dinov2Embeddings_MultiFrame(
            self.backbone.embeddings,
            time_indexs=frame_ids,other_frame_init_weight=other_frame_init_weight
        )
        self.neck = model.neck
        model_head = model.head
        self.head = DepthAnythingDepthEstimationHead(model_head)
        if debug_output:
            with open("model_new.txt", "a") as f:
                f.write(str(self) + "\n")
        model.post_init()

    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.lora_initializer.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.lora_initializer.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.lora_initializer.w_Bs[i].weight for i in range(num_layer)}
        decode_head_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.decode_head, torch.nn.DataParallel) or isinstance(self.decode_head, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.decode_head.module.state_dict()
        else:
            state_dict = self.decode_head.state_dict()
        for key, value in state_dict.items():
            decode_head_tensors[key] = value
        # save embeddings weights
        embedding_layers = self.backbone.embeddings
        embedding_layers_dict = embedding_layers.state_dict()
        merged_dict = {**a_tensors, **b_tensors, **decode_head_tensors, 'embed':embedding_layers_dict}
        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.lora_initializer.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.lora_initializer.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        decode_head_dict = self.decode_head.state_dict()
        decode_head_keys = decode_head_dict.keys()

        # load decode head
        decode_head_keys = [k for k in decode_head_keys]
        decode_head_values = [state_dict[k] for k in decode_head_keys]
        decode_head_new_state_dict = {k: v for k, v in zip(decode_head_keys, decode_head_values)}
        decode_head_dict.update(decode_head_new_state_dict)
        self.decode_head.load_state_dict(decode_head_dict)

        # load embeddings weights
        embedding_layers = self.backbone.embeddings
        embedding_layers.load_state_dict(state_dict['embed'])

        print('loaded lora parameters from %s.' % filename)

    def forward(self, pixel_values):
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=None, output_attentions=None
        )
        hidden_states = outputs.feature_maps
        _, _, time, height, width = pixel_values.shape
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
    model = DARES_cpe(frame_ids=[0,-1,1],other_frame_init_weight = 1e-3, debug_output=True)
    test_input = torch.randn(1, 3, 3, 256, 256)
    backbone_output = model.backbone.forward_with_filtered_kwargs(test_input, output_hidden_states=None, output_attentions=None)
    print(backbone_output.keys())
# %%