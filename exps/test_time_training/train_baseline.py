import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DARES')))
from exps.test_time_training.trainer_baseline import TrainerBaseline
from exps.test_time_training.options_baseline import BaselineOpt
from exps.exp_setup_local import ds_testtime_train_c3vd_2_train, ds_testtime_train_c3vd_2_test, check_test_only, log_path
from exps.find_best import find_best
from exps.find_best_parametric import find_best_parametric
from exps.load_other_models import load_DARES

opt = BaselineOpt
model_name = 'test_time_training_2'
pretrained_root_dir = './logs/baseline/models'
if __name__ == "__main__":
    if not check_test_only():
        trainer = TrainerBaseline(model_name, log_path, opt, 
                          train_eval_ds={'train': ds_testtime_train_c3vd_2_train, 'val': ds_testtime_train_c3vd_2_test},
                          pretrained_root_dir=pretrained_root_dir)
        trainer.train()
    find_best_parametric(load_DARES, model_name,  peft = False, only_keep_best=False, ds_name='test_time_train_2', dataset=ds_testtime_train_c3vd_2_test)