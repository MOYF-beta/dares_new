"""
Training script for DSFM (Direct Structure from Motion) trainer
This script demonstrates how to use the TrainerDSFM that eliminates
the need for a pose encoder by using optical flow and appearance flow directly.
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Add DARES directory to Python path
dares_path = os.path.join(project_root, 'DARES')
sys.path.insert(0, dares_path)

from exps.DSfm.trainer_dsfm import TrainerDSFM
from exps.DSfm.options_dsfm import DSFMOpt
from exps.exp_setup_local import (ds_val, check_test_only, get_unique_name, log_path, 
                                  ds_train, ds_train_c3vd, ds_train_syntheticcolon,
                                  ds_test, ds_test_c3vd, ds_test_syntheticcolon,
                                  ds_train_pitvis)
from exps.find_best_parametric import find_best_parametric
from exps.load_other_models import load_DARES

opt = DSFMOpt

pretrained_root_dir = './DARES/af_sfmlearner_weights'

if __name__ == "__main__":
    exp_num = 1
    
    # SCARED dataset
    # model_name = f'dsfm_{exp_num}_scared'
    # print(f"\033[96m=== Training DSFM model: {model_name} ===\033[0m")
    # print(f"\033[93mThis trainer eliminates pose encoder and uses AF_OF_Posedecoder_with_intrinsics\033[0m")
    
    # trainer = TrainerDSFM(model_name, log_path, opt, 
    #                   train_eval_ds={'train': ds_train, 'val': ds_val},
    #                   pretrained_root_dir=pretrained_root_dir,
    #                   use_af_pose=False)  # Not needed since we don't use pose encoder
    # trainer.train()
    # trainer.compare_weight_size()
    
    # # Evaluate on test sets
    # find_best_parametric(load_DARES, model_name,
    #                       only_keep_best=False, ds_name='SCARED', dataset=ds_test, peft=True, pose_seq=1)
    # find_best_parametric(load_DARES, model_name,
    #                       only_keep_best=False, ds_name='SCARED', dataset=ds_test, peft=True, pose_seq=2)
    
    # # C3VD dataset
    # model_name = f'dsfm_{exp_num}_c3vd'
    # print(f"\033[96m=== Training DSFM model: {model_name} ===\033[0m")
    
    # trainer = TrainerDSFM(model_name, log_path, opt, 
    #                   train_eval_ds={'train': ds_train_c3vd, 'val': ds_test_c3vd},
    #                   pretrained_root_dir=pretrained_root_dir,
    #                   use_af_pose=False)
    # trainer.train()
    
    # find_best_parametric(load_DARES, model_name,
    #                       only_keep_best=False, ds_name='C3VD', dataset=ds_test_c3vd, peft=True)

    # PitVis dataset
    model_name = f'dsfm_{exp_num}_pitvis'
    print(f"\033[96m=== Training DSFM model: {model_name} ===\033[0m")
    trainer = TrainerDSFM(model_name, log_path, opt, 
                      train_eval_ds={'train': ds_train_pitvis, 'val': ds_val},
                      pretrained_root_dir=pretrained_root_dir,
                      use_af_pose=False)
    trainer.train()
    
