import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
from exps.shared_vit_encoder_with_pose_no_multiscale.trainer_vit_encoder import TrainerVitEncoder
from exps.shared_vit_encoder_with_pose_no_multiscale.options_vit_encoder import VitEndoderOpt
from exps.exp_setup_local import ds_train, ds_val, check_test_only, get_unique_name, log_path
from exps.find_best import find_best

opt = VitEndoderOpt
model_name = 'dares_shared_vit_encoder_with_pose_no_multiscale'
pretrained_root_dir = './logs/dares_attn_encoder/models'
if __name__ == "__main__":
    if not check_test_only():
        trainer = TrainerVitEncoder(model_name, log_path, opt, 
                          train_eval_ds={'train': ds_train, 'val': ds_val},
                          pretrained_root_dir=pretrained_root_dir)
        trainer.train()
    find_best('DARES', model_name, only_keep_best=True)