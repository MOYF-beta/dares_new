import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
from exps.no_pose_encoder.trainer_no_pose import TrainerNoPose
from exps.no_pose_encoder.options_no_pose import AttnEncoderOpt
from exps.exp_setup_local import (ds_val, check_test_only, get_unique_name, log_path, 
                                  ds_train, ds_train_c3vd, ds_train_syntheticcolon,
                                  ds_test , ds_test_c3vd,  ds_test_syntheticcolon)
from exps.find_best_parametric import find_best_parametric
from exps.load_other_models import load_DARES
opt = AttnEncoderOpt

pretrained_root_dir = './DARES/af_sfmlearner_weights'
if __name__ == "__main__":

    exp_num = 1
    # SCARED
    model_name = f'no_pose_{exp_num}_scared'
    trainer = TrainerNoPose(model_name, log_path, opt, 
                      train_eval_ds={'train': ds_train, 'val': ds_val},
                      pretrained_root_dir=pretrained_root_dir)
    trainer.train()
    find_best_parametric(load_DARES, model_name,
                          only_keep_best=False, ds_name='SCARED', dataset=ds_test, peft=True, pose_seq = 1)
    find_best_parametric(load_DARES, model_name,
                          only_keep_best=False, ds_name='SCARED', dataset=ds_test, peft=True, pose_seq = 2)
    # C3VD
    model_name = f'no_pose_{exp_num}_c3vd'
    trainer = TrainerNoPose(model_name, log_path, opt, 
                      train_eval_ds={'train': ds_train_c3vd, 'val': ds_test_c3vd},
                      pretrained_root_dir=pretrained_root_dir)
    trainer.train()
    find_best_parametric(load_DARES, model_name,
                          only_keep_best=False, ds_name='C3VD', dataset=ds_test_c3vd, peft=True)