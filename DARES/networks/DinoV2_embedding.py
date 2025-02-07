# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch DINOv2 model."""

import collections
from typing import Optional
import torch
from torch import nn
from cpe_models import ComplexConv2d
'''
------------------------------------原始定义------------------------------------
放在这里仅用于获取类型提示，实际使用时从transformers.models加载
'''
class Dinov2Embeddings(nn.Module):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """
    @staticmethod
    def torch_int(x):
        """
        Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
        """
        return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)

    def __init__(self, config) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and interpolation at torch.float32 precision.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = Dinov2Embeddings.torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings


class Dinov2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

'''
------------------------------------以下为修改部分------------------------------------
'''

import torch
import torch.nn as nn
from typing import Optional

class Dinov2Embeddings_MultiFrame(nn.Module):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    Here we add time information to the embeddings.
    """

    def __init__(self, original_patch_embeddings: Dinov2Embeddings, time_indexs: list[int],
                  other_frame_init_weight = 1e-4,
                 old_embed_requires_grad = False,
                 use_res_connect = True,
                 use_old_for_debug = False,
                 seperate_embed = False
                 ) -> None:
        super().__init__()
        '''
        take the original patch embeddings and add time information to it.
        This model then can be used to replace the original Dinov2Embeddings in Dinov2 model.
        '''
        
        self.original_patch_embeddings = original_patch_embeddings
        self.cls_token = original_patch_embeddings.cls_token
        self.mask_token = original_patch_embeddings.mask_token
        self.position_embeddings = original_patch_embeddings.position_embeddings
        self.dropout = original_patch_embeddings.dropout
        self.patch_size = original_patch_embeddings.patch_size
        self.config = original_patch_embeddings.config
        self.use_old_for_debug = use_old_for_debug
        self.seperate_embed = seperate_embed

        # Set requires_grad to False for all parameters
        if old_embed_requires_grad:
            print("\033[93m!We are now training old embed now!\033[0m")
            for param in self.original_patch_embeddings.parameters():
                param.requires_grad = True
            self.cls_token.requires_grad = True
            self.mask_token.requires_grad = True
            self.position_embeddings.requires_grad = True
        if use_old_for_debug:
            print("\033[91m!!Using original Dinov2Embeddings for debug!!\033[0m")
            return

        # Initialize separate embeddings if seperate_embed is True
        if seperate_embed:
            print("~~We are now using seperate embeddings for depth and pose~~")
            depth_patch_embeddings_copy = Dinov2PatchEmbeddings(original_patch_embeddings.config)
            depth_patch_embeddings_copy.load_state_dict(original_patch_embeddings.patch_embeddings.state_dict())
            pose_patch_embeddings_copy = Dinov2PatchEmbeddings(original_patch_embeddings.config)
            pose_patch_embeddings_copy.load_state_dict(original_patch_embeddings.patch_embeddings.state_dict())
            self.depth_patch_embeddings = Dinov2PatchEmbeddings_MultiFrame(
                depth_patch_embeddings_copy, time_indexs,
                other_frame_init_weight, use_res_connect)
            self.pose_patch_embeddings = Dinov2PatchEmbeddings_MultiFrame(
                pose_patch_embeddings_copy, time_indexs,
                other_frame_init_weight, use_res_connect)
            self.current_embed = 'depth'  # Default to 'depth'
        else:
            print("~~We are now using shared embeddings for depth and pose~~")
            self.depth_patch_embeddings = Dinov2PatchEmbeddings_MultiFrame(
                original_patch_embeddings.patch_embeddings, time_indexs,
                other_frame_init_weight, use_res_connect)
            self.current_embed = 'depth'  # Only 'depth' is available when seperate_embed is False

    def use_embed(self, embed_name: str):
        if not self.seperate_embed:
            if embed_name != 'depth':
                raise ValueError("When seperate_embed is False, only 'depth' embedding is available.")
            return
        if embed_name not in ['depth', 'pose']:
            raise ValueError("embed_name must be either 'depth' or 'pose'.")
        self.current_embed = embed_name

    def forward(self, time_pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_old_for_debug:
            return self.original_patch_embeddings(time_pixel_values, bool_masked_pos)
        
        batch_size, channel, time, height, width = time_pixel_values.shape
        assert time == len(self.depth_patch_embeddings.time_indexs), "The number of time indexs should match the time dimension of the input"
        
        if self.seperate_embed:
            if self.current_embed == 'depth':
                embeddings = self.depth_patch_embeddings(time_pixel_values.to(dtype=self.depth_patch_embeddings.dtype))
            else:
                embeddings = self.pose_patch_embeddings(time_pixel_values.to(dtype=self.pose_patch_embeddings.dtype))
        else:
            embeddings = self.depth_patch_embeddings(time_pixel_values.to(dtype=self.depth_patch_embeddings.dtype))

        # from here on, the time dimension is merged and everything is 2D again
        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.original_patch_embeddings.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings
class Dinov2PatchEmbeddings_MultiFrame(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, old_Dinov2PatchEmbeddings : Dinov2PatchEmbeddings, time_indexs: list[int], 
                 other_frame_init_weight, use_res_connect) -> None:
        super().__init__()

        self.image_size = old_Dinov2PatchEmbeddings.image_size
        self.patch_size = old_Dinov2PatchEmbeddings.patch_size
        self.num_channels = old_Dinov2PatchEmbeddings.num_channels
        self.num_patches = old_Dinov2PatchEmbeddings.num_patches
        self.time_indexs = time_indexs
        self.num_time_indexs = len(time_indexs)
        self.dtype = old_Dinov2PatchEmbeddings.projection.weight.dtype
        # 接受原本的patch_embeddings，传给Dinov2Complex_Conv，由它转为复数
        self.projection = Dinov2ComplexRes_Proj(old_Dinov2PatchEmbeddings.projection,
                                                 time_indexs, self.patch_size, use_res_connect,
                                                 other_frame_init_weight)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
    
class Dinov2ComplexRes_Proj(nn.Module):
    '''
    This class takes a nn.Conv2d from Dinov2PatchEmbeddings_MultiFrame.projection, 
    convert it to Complex domain and preserve the loaded real weights. Then we use this
    ComplexConv2d to process each frame's pixel values and merge them in time domain, 
    getting 2D embeddings.

    the time_merge_layer's output is regarded as resiudal and should be close to 0 at the beginning.

    This modification is intended to make Vit support video processing. adjacent frames's
    pixel are added a learnable imaginary part, multiplied by the time index. This should
    give the model a sense of time.
    '''

    def __init__(self, old_projection: nn.Conv2d,
                  time_indexs: list[int], 
                  patch_size, 
                  use_res_connect,
                  other_frame_init_weight,
                  old_embed_requires_grad = False,
                  ) -> None:
        super().__init__()
        self.res_connect = use_res_connect
        self.projection_c = ComplexConv2d(old_projection, imag_init_std=other_frame_init_weight)
        self.olf_projection = old_projection
        self.olf_projection.weight.requires_grad = old_embed_requires_grad
        self.num_time_indexs = len(time_indexs)
        assert time_indexs[0] == 0, "The first time index should be 0"
        # time_merge_layer input is stacked embeddings of real and imaginary parts,
        # there are total 2 * num_time_indexs embeddings, half of them are real and 
        # the other half are imaginary.
        hidden_size = old_projection.out_channels
        # (batch,channel,time,H,W) --Conv3d-> (batch,channel,1,H,W) --sequeeze-> (batch,channel,H,W)
        self.time_merge_layer = nn.Conv3d(2 * hidden_size,
                                            hidden_size, 
                                           kernel_size=(self.num_time_indexs,1, 1),
                                           stride=(1,1,1), padding=(0,0,0)
                                           )
        # initialize the weights of time_merge_layer around 0 with a standard deviation of other_frame_init_weight
        nn.init.normal_(self.time_merge_layer.weight, mean=0.0, std=other_frame_init_weight)

        # learnable imaginary patch
        self.imaginary_patchs_convs = []
        for i, time_index in enumerate(time_indexs):
            self.imaginary_patchs_convs.append(
                nn.Conv2d(old_projection.in_channels, old_projection.in_channels,
                           kernel_size=(1,1), stride=1))
            # initialize the weights of imaginary_patchs_convs around time_index with a standard deviation of other_frame_init_weight
            nn.init.normal_(self.imaginary_patchs_convs[-1].weight, mean=time_index, std=other_frame_init_weight)
            self.imaginary_patchs_convs[-1]
    def img_to_complex(self, pixel_values: torch.Tensor, time_index : int, ) -> torch.Tensor:
        '''
        Convert the pixel_values to complex domain
        '''

        real = pixel_values
        imag_expanded = real.clone()
        imag_expanded = self.imaginary_patchs_convs[time_index](imag_expanded)

        return real, imag_expanded

    def forward(self, time_pixel_values: torch.Tensor) -> torch.Tensor:
        '''
        pixel_values shape: (batch_size, num_channels, time_index, height, width)
        for each time_index, convert the pixel_values to complex domain, then use projection_c
        to get the seperate embeddings. finally, merge the embeddings in time domain using 
        time_merge_layer.
        '''
        batch_size, num_channels, time, height, width = time_pixel_values.shape
        embeded_patches_re = []
        embeded_patches_im = []

        for frame_index in range(time):
            real, imag = self.img_to_complex(time_pixel_values[:, :, frame_index], frame_index)
            embeded_re, embeded_im = self.projection_c(real, imag)
            embeded_patches_re.append(embeded_re)
            embeded_patches_im.append(embeded_im)
        
        embeded_patches_re = torch.stack(embeded_patches_re, dim=2)
        embeded_patches_im = torch.stack(embeded_patches_im, dim=2)
        embeded_patches = torch.cat((embeded_patches_re, embeded_patches_im), dim=1)

        # Add the real part of the 0th frame to the merged embeddings
        merged_embeddings = self.time_merge_layer(embeded_patches).squeeze(2)
        
        if self.res_connect:
            old_embedding = self.olf_projection(time_pixel_values[:, :, 0])
            return merged_embeddings + old_embedding
        else:
            return merged_embeddings
        '''
        应该也能这么算，等上面的测试过了再试试这个
        all_frames = time_pixel_values.permute(0,2,1,3,4)  # [B,T,C,H,W]
        real, imag = self.img_to_complex(all_frames)  # 修改后的批量处理
        embeded_re, embeded_im = self.projection_c(real, imag)
        '''