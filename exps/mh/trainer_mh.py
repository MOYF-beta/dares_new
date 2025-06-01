import os
import torch
from exps.trainerMH_abc import TrainerMH as BaseTrainerMH
from DARES.networks.dares_peft_MH import DARES_MH
from DARES.networks.resnet_encoder import AttentionalResnetEncoder
from DARES.networks.optical_flow_decoder import PositionDecoder
from DARES.networks.appearance_flow_decoder import TransformDecoder

class TrainerMH(BaseTrainerMH):
    def load_model(self):
        # Step-wise model initialization
        # 1) Encoders
        encoders = {
            "dares_mh": DARES_MH(
                r=8, target_modules=["query","value"], use_dora=True,
                full_finetune=False, image_size=(self.opt.height,self.opt.width),
                heads=["depth","pose"],
                pretrained_path=(os.path.join(self.pretrained_root_dir,"best","depth_model.pth")
                                 if self.pretrained_root_dir else None)
            ).to(self.device),
            "position_encoder": AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2).to(self.device),
            "transform_encoder": AttentionalResnetEncoder(self.opt.num_layers, False, num_input_images=2).to(self.device)
        }
        # 2) Decoders
        decoders = {
            "position": PositionDecoder(encoders["position_encoder"].num_ch_enc, self.opt.scales).to(self.device),
            "transform": TransformDecoder(encoders["transform_encoder"].num_ch_enc, self.opt.scales).to(self.device)
        }
        # 3) Register models and parameters
        all_models = [
            ("dares_mh", encoders["dares_mh"], self.param_monodepth),
            ("position_encoder", encoders["position_encoder"], self.param_pose_net),
            ("position", decoders["position"], self.param_pose_net),
            ("transform_encoder", encoders["transform_encoder"], self.param_monodepth),
            ("transform", decoders["transform"], self.param_monodepth),
        ]
        for name, mdl, plist in all_models:
            self.models[name] = mdl
            plist += list(filter(lambda p: p.requires_grad, mdl.parameters()))
        # 4) Load pretrained weights
        if self.pretrained_root_dir:
            for name in [m[0] for m in all_models]:
                path = os.path.join(self.pretrained_root_dir, "best", f"{name}.pth")
                if os.path.exists(path):
                    self.models[name].load_state_dict(
                        torch.load(path, map_location=self.device, weights_only=True), strict=False)
                    print(f"Loaded pretrained weights for {name} from {path}")