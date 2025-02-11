import os,sys
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
this script is used to load the pose encoder and decoder for other models
'''
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exps/DARES_pro')))


def load_pose_encoder_decoder_DARES(opt ):
    from DARES.networks import ResnetEncoder, PoseDecoder
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location=device.type))

    pose_decoder = PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path, map_location=device.type))

    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()
    return pose_encoder, pose_decoder

def load_DARES(opt, weight_path=None, pth_name='depth_model.pth', refine=True, peft=False):
    if peft:
        from DARES.networks.dares_peft import DARES
    else:
        from DARES.networks.dares import DARES
    if weight_path is None:
        weight_path = opt.load_weights_folder
    depth_model_path = os.path.join(weight_path, pth_name)
    if not os.path.exists(depth_model_path):
        depth_model_path = os.path.join(weight_path, 'depth.pth')
    depth_model_dict = torch.load(depth_model_path)
    depth_model = DARES(enable_refine_net=refine)
    model_dict = depth_model.state_dict()
    depth_model.load_state_dict({k: v for k, v in depth_model_dict.items() if k in model_dict})
    depth_model.cuda()
    depth_model.eval()

    return depth_model

def load_DARES_CPE(opt, weight_path=None, pth_name='depth_model.pth'):
    from exps.dares_cpe.dares_cpe import DARES_cpe
    if weight_path is None:
        weight_path = opt.load_weights_folder
    depth_model_path = os.path.join(weight_path, pth_name)
    if not os.path.exists(depth_model_path):
        depth_model_path = os.path.join(weight_path, 'depth.pth')
    depth_model = DARES_cpe(opt.frame_ids, opt.other_frame_init_weight)
    state_dict = torch.load(depth_model_path, map_location=device, weights_only=True)
    depth_model.load_state_dict(state_dict)
    depth_model.cuda()
    depth_model.eval()

    return depth_model