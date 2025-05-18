import os
import argparse
import warnings
import json
import shutil
from exps.dataset import DataLoaderX as DataLoader
from exps.options import DotDict
from exps.exp_setup_local import log_path

def find_best_parametric(model_loader_fn, model_name, only_keep_best=False, ds_name='SCARED', dataset=None, eval_kwargs=None):
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DARES')))
    from DARES import evaluate_pose_and_intrinsics
    warnings.filterwarnings("ignore")
    if dataset is None:
        from exps.exp_setup_local import ds_test
        dataset = ds_test
    model_path = os.path.join(log_path, model_name, 'models')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    opt_run = json.load(open(os.path.join(model_path, 'opt.json'), 'r'))
    opt_dict = DotDict(opt_run)
    def evaluate_model(weight_path, dataset=dataset, load_depth_from_npz=ds_name=='SCARED'):
        if ds_name == 'SCARED':
            from DARES.evaluate_depth_scared import evaluate
        else:
            from DARES.evaluate_depth import evaluate
        test_dataloader = DataLoader(dataset, 16, shuffle=False, pin_memory=True, drop_last=False, num_workers=10)
        depth_model = model_loader_fn(opt_dict, weight_path)
        ds_and_model = {
            'dataloader': test_dataloader,
            'depth_model': depth_model
        }
        eval_args = dict(ds_and_model=ds_and_model, load_depth_from_npz=load_depth_from_npz)
        if eval_kwargs:
            eval_args.update(eval_kwargs)
        return evaluate(opt_dict, **eval_args)
    print(f"Testing {model_name}")
    weights = [w for w in os.listdir(model_path) if w.startswith('weights_')]
    if not weights:
        print(f"No weights found in {model_path} for model {model_name}.")
        return
    print(f"Found {len(weights)} weights for {model_name}")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    best_score = float('inf')
    best_weight = None
    best_pose_score = float('inf')
    best_pose_weight = None
    best_pose_ate = None
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
    pose_ate_dict = {}
    for weight in weights:
        weight_path = os.path.join(model_path, weight)
        # 深度评估
        score = evaluate_model(weight_path)
        print(f"\t{weight}")
        if score[index] < best_score:
            best_score = score[index]
            best_weight = weight
        # 位姿/内参评估
        try:
            pose_result = evaluate_pose_and_intrinsics.evaluate(
                opt=opt_dict,  # 可根据需要传递opt
                load_weights_folder=weight_path,
                dataset_name=ds_name,
                evaluate_pose=True,
                evaluate_intrinsics=True
            )
            if pose_result and 'ate_metric' in pose_result:
                ate = pose_result['ate_metric'] if isinstance(pose_result['ate_metric'], float) else pose_result['ate_metric'].get('rmse', None)
                pose_ate_dict[weight] = ate
                if ate is not None and ate < best_pose_score:
                    best_pose_score = ate
                    best_pose_weight = weight
                    best_pose_ate = ate
        except Exception as e:
            print(f"Pose evaluation failed for {weight}: {e}")
    print(f"Best weight for {model_name} (depth): {best_weight} with score: {best_score}")
    print(f"Best weight for {model_name} (pose): {best_pose_weight} with ATE: {best_pose_ate}")
    print('-' * 50)
    # 只保留最佳深度和最佳位姿权重
    best_weight_path = os.path.join(model_path, best_weight)
    best_pose_weight_path = os.path.join(model_path, best_pose_weight) if best_pose_weight else None
    best_dir = os.path.join(model_path, f'best_{ds_name}_depth')
    best_pose_dir = os.path.join(model_path, f'best_{ds_name}_pose')
    if only_keep_best:
        for weight in weights:
            weight_path = os.path.join(model_path, weight)
            if weight != best_weight and (best_pose_weight is None or weight != best_pose_weight):
                shutil.rmtree(weight_path)
        os.rename(best_weight_path, best_dir)
        if best_pose_weight_path and best_pose_weight != best_weight:
            os.rename(best_pose_weight_path, best_pose_dir)
    else:
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(best_weight_path, best_dir)
        if best_pose_weight_path and best_pose_weight != best_weight:
            if os.path.exists(best_pose_dir):
                shutil.rmtree(best_pose_dir)
            shutil.copytree(best_pose_weight_path, best_pose_dir)
    # 输出结果
    result_file = os.path.join(model_path, 'best_results.txt')
    if not os.path.exists(result_file):
        with open(result_file, 'w') as f:
            f.write('Best weight results:\n')
    with open(result_file, 'a') as f:
        f.write(f'Best weight path (depth): {best_weight} for dataset {ds_name}\n')
        f.write(f'Best weight path (pose): {best_pose_weight} for dataset {ds_name}\n')
        f.write('| Model Name    | abs_rel | sq_rel | rmse   | rmse_log | a1    | a2    | a3    |  ATE   |\n')
        f.write('|---------------|---------|--------|--------|----------|-------|-------|-------|--------|\n')
        f.write(f'| {model_name:<13} | {best_score:<7.4f} | {score[1]:<7.4f} | {score[2]:<7.4f} | {score[3]:<7.4f} | {score[4]:<7.4f} | {score[5]:<7.4f} | {score[6]:<7.4f} | {best_pose_ate if best_pose_ate is not None else "-":<6} |\n')
        # 输出pose最优模型的内参评估结果
        if best_pose_weight in pose_ate_dict:
            pose_result = None
            try:
                pose_result = evaluate_pose_and_intrinsics.evaluate(
                    opt=opt_dict,
                    load_weights_folder=os.path.join(model_path, best_pose_weight),
                    dataset_name=ds_name,
                    evaluate_pose=True,
                    evaluate_intrinsics=True
                )
            except Exception as e:
                f.write(f"[WARN] Failed to get intrinsics for best pose: {e}\n")
            if pose_result and 'intrinsics_error' in pose_result:
                intr = pose_result['intrinsics_error']
                mean_intr = pose_result.get('mean_pred_intrinsics', None)
                f.write('\n[Best Pose Intrinsics Evaluation]\n')
                if mean_intr is not None:
                    f.write(f"Mean Predicted Intrinsics:\n{mean_intr}\n")
                f.write(f"Mean Scale Factor: {intr.get('scale_factor_mean', '-'):.4f}\n")
                f.write(f"Scale-Adjusted Focal Length X Error: {intr.get('scale_adjusted_fx_abs_error_mean', '-'):.2f} pixels ({intr.get('scale_adjusted_fx_rel_error_mean', '-'):.2f}% relative)\n")
                f.write(f"Scale-Adjusted Focal Length Y Error: {intr.get('scale_adjusted_fy_abs_error_mean', '-'):.2f} pixels ({intr.get('scale_adjusted_fy_rel_error_mean', '-'):.2f}% relative)\n")
                f.write(f"Scale-Adjusted Principal Point X Error: {intr.get('scale_adjusted_cx_abs_error_mean', '-'):.2f} pixels ({intr.get('scale_adjusted_cx_rel_error_mean', '-'):.2f}% relative)\n")
                f.write(f"Scale-Adjusted Principal Point Y Error: {intr.get('scale_adjusted_cy_abs_error_mean', '-'):.2f} pixels ({intr.get('scale_adjusted_cy_rel_error_mean', '-'):.2f}% relative)\n")
                f.write(f"Scale-Adjusted Matrix Frobenius Norm: {intr.get('scale_adjusted_matrix_frobenius_norm_mean', '-'):.2f}\n")
    print(f"Results appended to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific model with specified weights (parametric version).")
    parser.add_argument('--model_name', required=True, help="Name of the model directory under log_path.")
    args = parser.parse_args()
    from exps.load_other_models import load_DARES
    model_loader_fn = lambda opt, w: load_DARES(opt, w, 'depth_model.pth', refine=False)
    find_best_parametric(model_loader_fn, args.model_name)
