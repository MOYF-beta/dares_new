import os
import argparse
import warnings
import json
from exps.dataset import DataLoaderX as DataLoader
from exps.load_other_models import *
import shutil


from exps.options import DotDict
from exps.exp_setup_local import log_path
def find_best(model_type, model_name, only_keep_best=False, ds_name='SCARED', dataset = None):
    warnings.filterwarnings("ignore")
    if model_type == 'DARES' or model_type == 'DARES_CPE' or model_type == 'DARES_peft':
        if dataset is None:
            from exps.exp_setup_local import ds_test, ds_test_multi_frame
        else:
            ds_test = dataset
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    model_path = os.path.join(log_path, model_name, 'models')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    opt_run = json.load(open(os.path.join(model_path, 'opt.json'), 'r'))
    opt_dict = DotDict(opt_run)
    # Helper function to evaluate a model with a specific set of weights
    
    def evaluate_model(model_type, weight_path, dataset = ds_test, load_depth_from_npz = ds_name=='SCARED'):
        if ds_name=='SCARED':
            from DARES.evaluate_depth_scared import evaluate
        else:
            from DARES.evaluate_depth import evaluate
        if model_type == 'DARES':
            test_dataloader = DataLoader(dataset, 16, shuffle=False, pin_memory=True, drop_last=False, num_workers=10)
            depth_model = load_DARES(opt_dict, weight_path, 'depth_model.pth', refine=False)
            ds_and_model = {
                'dataloader': test_dataloader,
                'depth_model': depth_model
            }
            return evaluate(opt_dict, ds_and_model=ds_and_model, load_depth_from_npz=load_depth_from_npz)
        elif model_type == 'DARES_peft':
            test_dataloader = DataLoader(dataset, 16, shuffle=False, pin_memory=True, drop_last=False, num_workers=10)
            depth_model = load_DARES(opt_dict, weight_path, 'depth_model.pth', refine=False, peft=True)
            ds_and_model = {
                'dataloader': test_dataloader,
                'depth_model': depth_model
            }
            return evaluate(opt_dict, ds_and_model=ds_and_model, load_depth_from_npz=load_depth_from_npz)        
        elif model_type == 'DARES_refine':
            test_dataloader = DataLoader(dataset, 16, shuffle=False, pin_memory=True, drop_last=False, num_workers=10)
            depth_model = load_DARES(opt_dict, weight_path, 'depth_model.pth', refine=True)
            ds_and_model = {
                'dataloader': test_dataloader,
                'depth_model': depth_model
            }
            return evaluate(opt_dict, ds_and_model=ds_and_model, load_depth_from_npz=load_depth_from_npz) 
        elif model_type == 'DARES_CPE':
            test_dataloader = DataLoader(ds_test_multi_frame, 16, shuffle=False, pin_memory=True, drop_last=False, num_workers=10)
        
            depth_model = load_DARES_CPE(opt_dict, weight_path, 'depth_model.pth')
            ds_and_model = {
                'dataloader': test_dataloader,
                'depth_model': depth_model
            }
            return evaluate(opt_dict, ds_and_model=ds_and_model, frames_input=opt_dict.frame_ids, load_depth_from_npz=load_depth_from_npz)    
    
    print(f"Testing {model_type} ({model_name})")
    weights = [w for w in os.listdir(model_path) if w.startswith('weights_')]
    print(f"Found {len(weights)} weights for {model_name}")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    best_score = float('inf')
    best_weight = None

    METRIC = 'abs_rel'
    metric_indices = {
        'abs_rel': 0,
        'sq_rel': 1,
        'rmse': 2,
        'rmse_log': 3,
        'a1': 4,
        'a2': 5,
        'a3': 6
    }
    index = metric_indices[METRIC]

    for weight in weights:
        weight_path = os.path.join(model_path, weight)

        score = evaluate_model(model_type, weight_path)
        print(f"\t{weight}")
        if score[index] < best_score:
            best_score = score[index]
            best_weight = weight


    print(f"Best weight for {model_type}: {best_weight} with score: {best_score}")
    print('-' * 50)

    # Handle best weight file
    best_weight_path = os.path.join(model_path, best_weight)
    best_dir = os.path.join(model_path, 'best')

    if only_keep_best:
        # Remove all other weights
        for weight in weights:
            weight_path = os.path.join(model_path, weight)
            if weight != best_weight:
                shutil.rmtree(weight_path)
        os.rename(best_weight_path, best_dir)
    else:
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(best_weight_path, best_dir)

    # Output best weights and scores to a file
    result_file = os.path.join(model_path, 'best_results.txt')
    with open(result_file, 'a') as f:
        f.write(f'Best weight path: {best_weight} for dataset {ds_name}\n')
        f.write('| Method        | abs_rel | sq_rel | rmse   | rmse_log | a1    | a2    | a3    |\n')
        f.write('|---------------|---------|--------|--------|----------|-------|-------|-------|\n')
        f.write(f'| {model_type:<13} | {best_score:<7.4f} | {score[1]:<7.4f} | {score[2]:<7.4f} | {score[3]:<7.4f} | {score[4]:<7.4f} | {score[5]:<7.4f} | {score[6]:<7.4f} |\n')

    print(f"Results appended to {result_file}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a specific model with specified weights.")
    parser.add_argument('--model_type', required=True, choices=['DARES', 'monodepth2', 'AF_SfMLearner'], help="Type of the model to evaluate.")
    parser.add_argument('--model_name', required=True, help="Name of the model directory under log_path.")
    args = parser.parse_args()

    find_best(args.model_type, args.model_name)
