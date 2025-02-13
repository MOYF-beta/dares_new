import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
from exps.attn_encoder_dora_base_model.trainer_attn_encoder import TrainerAttnEncoder
from exps.attn_encoder_dora_base_model.options_attn_encoder import AttnEncoderOpt
from exps.exp_setup_local import ds_val, check_test_only, get_unique_name, log_path, ds_base_model_train
from exps.find_best import find_best

opt = AttnEncoderOpt
model_name = 'base_model_attn_encoder_dora'
pretrained_root_dir = './DARES/af_sfmlearner_weights'
if __name__ == "__main__":
    if not check_test_only():
        trainer = TrainerAttnEncoder(model_name, log_path, opt, 
                          train_eval_ds={'train': ds_base_model_train, 'val': ds_val},
                          pretrained_root_dir=pretrained_root_dir)
        trainer.train()
    find_best('DARES_peft', model_name, only_keep_best=True)