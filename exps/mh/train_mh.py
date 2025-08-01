import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
from exps.mh.trainer_mh import TrainerMH
from exps.mh.options_mh import MHOpt
from exps.exp_setup_local import (ds_val, check_test_only, get_unique_name, log_path, 
                                  ds_train, 
                                  ds_test , ds_test_c3vd, ds_test_syntheticcolon)
from exps.find_best import find_best

opt = MHOpt
model_name = 'dares_mh'
pretrained_root_dir = None
if __name__ == "__main__":
    if not check_test_only():
        trainer = TrainerMH(model_name, log_path, opt, 
                          train_eval_ds={'train': ds_train, 'val': ds_val},
                          pretrained_root_dir=pretrained_root_dir)
        trainer.train()
    # find_best('DARES_peft', model_name, dataset=ds_test, ds_name='SCARED')
    # find_best('DARES_peft', model_name, dataset=ds_test_c3vd, ds_name='C3VD')
    # find_best('DARES_peft', model_name, dataset=ds_test_hamlyn, ds_name='Hamlyn')
    # find_best('DARES_peft', model_name, dataset=ds_test_syntheticcolon, ds_name='SyntheticColon')