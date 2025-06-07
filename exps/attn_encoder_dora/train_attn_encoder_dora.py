import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
from exps.attn_encoder_dora.trainer_attn_encoder import TrainerAttnEncoder
from exps.attn_encoder_dora.options_attn_encoder import AttnEncoderOpt
from exps.exp_setup_local import (ds_val, check_test_only, get_unique_name, log_path, 
                                  ds_train, ds_train_c3vd, ds_train_syntheticcolon,
                                  ds_test , ds_test_c3vd,  ds_test_syntheticcolon)
from exps.find_best_parametric import find_best_parametric
from exps.load_other_models import load_DARES
opt = AttnEncoderOpt

pretrained_root_dir = 'logs/baidalan1_syntheticcolon/models'
if __name__ == "__main__":
    我有多摆 = 'baidalan2'
    我想做实验嘛 = False
    这些代码是不是垃圾 = True
    # SyntheticColon
    model_name = f'{我有多摆}_syntheticcolon'
    trainer = TrainerAttnEncoder(model_name, log_path, opt, 
                      train_eval_ds={'train': ds_train_syntheticcolon, 'val': ds_test_syntheticcolon},
                      pretrained_root_dir=pretrained_root_dir)
    trainer.train()
    find_best_parametric(load_DARES, model_name,
                            only_keep_best=False, ds_name='SyntheticColon', dataset=ds_test_syntheticcolon, peft=True)
    # # SCARED
    # model_name = f'{我有多摆}_scared'
    # trainer = TrainerAttnEncoder(model_name, log_path, opt, 
    #                   train_eval_ds={'train': ds_train, 'val': ds_val},
    #                   pretrained_root_dir=pretrained_root_dir)
    # trainer.train()
    # find_best_parametric(load_DARES, model_name,
    #                       only_keep_best=False, ds_name='SCARED', dataset=ds_test, peft=True, pose_seq = 1)
    # find_best_parametric(load_DARES, model_name,
    #                       only_keep_best=False, ds_name='SCARED', dataset=ds_test, peft=True, pose_seq = 2)
    # # C3VD
    # model_name = f'{我有多摆}_c3vd'
    # trainer = TrainerAttnEncoder(model_name, log_path, opt, 
    #                   train_eval_ds={'train': ds_train_c3vd, 'val': ds_test_c3vd},
    #                   pretrained_root_dir=pretrained_root_dir)
    # trainer.train()
    # find_best_parametric(load_DARES, model_name,
    #                       only_keep_best=False, ds_name='C3VD', dataset=ds_test_c3vd, peft=True)