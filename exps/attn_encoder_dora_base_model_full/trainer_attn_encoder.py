import os
import torch
from DARES.networks.dares_peft import DARES
from DARES.networks.resnet_encoder import AttentionalResnetEncoder
from DARES.networks.pose_decoder import PoseDecoder_with_intrinsics as PoseDecoder_i
from DARES.networks.optical_flow_decoder import PositionDecoder
from DARES.networks.appearance_flow_decoder import TransformDecoder
from exps.trainer_abc import Trainer

class TrainerAttnEncoder(Trainer):
    def load_model(self):
        # Initialize depth model
        encoders = {
            "depth_model": DARES(use_dora=True,target_modules=['query', 'value'], full_finetune=True),
            "pose_encoder": AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=self.num_pose_frames),
            "position_encoder": AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2),
            "transform_encoder": AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2)
        }

        decoders = {
            "pose": PoseDecoder_i(encoders["pose_encoder"].num_ch_enc, image_width=self.opt.width, image_height=self.opt.height, predict_intrinsics=self.opt.learn_intrinsics, simplified_intrinsic=self.opt.simplified_intrinsic, num_input_features=1, num_frames_to_predict_for=2),
            "position": PositionDecoder(encoders["position_encoder"].num_ch_enc, self.opt.scales),
            "transform": TransformDecoder(encoders["transform_encoder"].num_ch_enc, self.opt.scales)
        }

        all_models = [
            ("depth_model",         encoders["depth_model"],        self.param_monodepth),
            ("pose_encoder",        encoders["pose_encoder"],       self.param_monodepth),
            ("pose",                decoders["pose"],               self.param_monodepth),
            ("position_encoder",    encoders["position_encoder"],   self.param_pose_net),
            ("position",            decoders["position"],           self.param_pose_net),
            ("transform_encoder",   encoders["transform_encoder"],  self.param_monodepth),
            ("transform",           decoders["transform"],          self.param_monodepth)
        ]
        # Initialize models and parameters, and move to device
        for model_name, model_instance, param_list in all_models:
            self.models[model_name] = model_instance
            param_list += list(filter(lambda p: p.requires_grad, model_instance.parameters()))

            self.models[model_name] = torch.nn.DataParallel(self.models[model_name])
            self.models[model_name].to("cuda")
        # Load pretrained weights if available
        if self.pretrained_root_dir is not None:
            model_paths = [
            "depth_model",
            "position_encoder",
            "position",
            "transform_encoder",
            "transform",
            "pose_encoder",
            "pose"
            ]
            for model_name in model_paths:
                model_path = os.path.join(self.pretrained_root_dir, "best", f"{model_name}.pth")
                if os.path.exists(model_path):
                    self.models[model_name].load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)
                    print(f"\033[94mLoaded pretrained weights for {model_name} from {model_path}\033[0m")
                else:
                    print(f"\033[93mNo pretrained weights found for {model_name} at {model_path}\033[0m")
        
        print(f'training on {torch.cuda.device_count()} GPUs')
    
    def get_depth_input(self, inputs):
        return inputs['color_aug', 0, 0]