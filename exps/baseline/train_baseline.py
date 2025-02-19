import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
from exps.baseline.trainer_baseline import TrainerBaseline
from exps.baseline.options_baseline import BaselineOpt
from exps.exp_setup_local import ds_train, ds_val, check_test_only, get_unique_name, log_path
from exps.find_best import find_best

opt = BaselineOpt
model_name = 'baseline'
pretrained_root_dir = './DARES/af_sfmlearner_weights'
if __name__ == "__main__":
    # if not check_test_only():
    #     trainer = TrainerBaseline(model_name, log_path, opt, 
    #                       train_eval_ds={'train': ds_train, 'val': ds_val},
    #                       pretrained_root_dir=pretrained_root_dir)
    #     trainer.train()
    find_best('DARES', model_name, only_keep_best=True)