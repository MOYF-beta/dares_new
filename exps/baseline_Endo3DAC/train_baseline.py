import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
from exps.baseline_Endo3DAC.trainer_baseline import TrainerBaseline
from exps.baseline_Endo3DAC.options_baseline import BaselineOpt
from exps.exp_setup_local import ds_train, ds_test,ds_train_syntheticcolon, ds_test_c3vd, check_test_only, ds_test_syntheticcolon, ds_train_c3vd, log_path
from exps.find_best import find_best
from exps.find_best_parametric import find_best_parametric
from exps.load_other_models import load_DARES

opt = BaselineOpt
model_name = 'base_model_1'
pretrained_root_dir = './DARES/af_sfmlearner_weights'
if __name__ == "__main__":
    # if not check_test_only():
    #     trainer = TrainerBaseline(model_name, log_path, opt, 
    #                       train_eval_ds={'train': ds_train, 'val': ds_test},
    #                       pretrained_root_dir=pretrained_root_dir)
    #     trainer.train()
    find_best_parametric(load_DARES, model_name, only_keep_best=False, ds_name='SCARED', dataset=ds_test)