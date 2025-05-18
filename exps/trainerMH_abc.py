import json
import os
import random
import time

import torch
import torch.optim as optim
from exps.dataset import DataLoaderX as DataLoader
from torch.utils.data import RandomSampler, Sampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from DARES.layers import *
from DARES.utils import *
from DARES.networks.dares_peft_MH import DARES_MH


class GlobalRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

class ConcatDatasetSampler(Sampler):
    def __init__(self, dataset):
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            self.concat_dataset = dataset
            self.cumulative_sizes = dataset.cumulative_sizes
            print(f'cumulative_sizes: {self.cumulative_sizes}')
        else:
            self.concat_dataset = None
            self.dataset = dataset
            print("\033[91m WARNING: Dataset is not a ConcatDataset, using GlobalRandomSampler instead.\033[0m")

    def __iter__(self):
        if self.concat_dataset:
            indices = []
            start = 0
            for end in self.cumulative_sizes:
                dataset_len = end - start
                perm = torch.randperm(dataset_len).tolist()
                global_perm = [p + start for p in perm]
                indices.extend(global_perm)
                start = end
            return iter(indices)
        else:
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            return iter(indices)

    def __len__(self):
        if self.concat_dataset:
            return len(self.concat_dataset)
        else:
            return len(self.dataset)
    
from exps.exp_setup_local import device
from abc import ABC, abstractmethod
class TrainerMH(ABC):

    @abstractmethod
    def load_model(self):
        pass

    def __init__(self, model_name, log_dir, options, train_eval_ds={},
                  pretrained_root_dir=None, merge_val_as_train=False, use_supervised_loss = True, debug = False):
        if debug:
            print("\033[91m WARNING: Debug mode activated, only train 1 epoch\033[0m")
            options.num_epochs = 2
            options.batch_size = 8
        self.opt = options
        self.log_path = os.path.join(log_dir, model_name)
        self.use_supervised_loss = use_supervised_loss
        self.models = {}  
        self.param_monodepth = []  
        self.param_pose_net = []
        
        self.device = device
        self.train_pos = False
        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        self.pretrained_root_dir = pretrained_root_dir
        self.load_model()

        # Initialize optimizer and scheduler
        self.optimizer_pose = optim.Adam(self.param_pose_net, self.opt.pos_learning_rate, eps=1e-7, weight_decay=self.opt.weight_decay)
        self.lr_scheduler_pose = optim.lr_scheduler.StepLR(self.optimizer_pose, self.opt.scheduler_step_size, 0.1)
        
        self.optimizer_depth = optim.Adam(self.param_monodepth, self.opt.learning_rate, eps=1e-7, weight_decay=self.opt.weight_decay_pose)
        self.lr_scheduler_depth = optim.lr_scheduler.StepLR(self.optimizer_depth, self.opt.scheduler_step_size, 0.1)


        train_dataset = train_eval_ds["train"]
        val_dataset = train_eval_ds["val"]
        
        if 'train_sampler' in train_eval_ds:
            train_sampler = train_eval_ds['train_sampler'](train_dataset)
        else:
            train_sampler = GlobalRandomSampler(train_dataset)
        if merge_val_as_train:
            print("\033[91m WARNING: Merging validation dataset into training dataset\033[0m")
            combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
            self.train_loader = DataLoader(
            combined_dataset, self.opt.batch_size,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        else:
            self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
            self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
            self.val_iter = iter(self.val_loader)

        
        num_train_samples = len(train_eval_ds['train'])
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
            self.ms_ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
    
    def load_model(self):
        """加载 DARES_MH 单模型多任务结构"""
        heads = ["depth", "pose", "optical_flow", "appearance_flow"]
        self.models["dares_mh"] = DARES_MH(
            r=[14,14,12,12,10,10,8,8,8,8,8,8],
            target_modules=['query', 'value'],
            use_dora=True,
            full_finetune=False,
            image_size=(self.opt.height, self.opt.width),
            heads=heads,
            pretrained_path=os.path.join(self.pretrained_root_dir, "best", "depth_model.pth") if self.pretrained_root_dir else None
        ).to(self.device)
        # 只需一个参数组
        self.param_monodepth += list(filter(lambda p: p.requires_grad, self.models["dares_mh"].parameters()))
        self.param_pose_net = self.param_monodepth

    def set_train(self, train_position_only=False):
        """
        设置 DARES_MH 的 adapter/head 是否可训练，双帧embedding始终可训练。
        train_position_only=True 时，仅训练pose/optical_flow/appearance_flow相关adapter和head；
        train_position_only=False 时，训练全部（包括depth）。
        """
        model = self.models["dares_mh"]
        # 1. 先全部冻结
        for param in model.parameters():
            param.requires_grad = False
        # 2. LoRA/DoRA参数按adapter解冻
        if train_position_only:
            # 只训练非depth adapter
            for name, param in model.backbone.named_parameters():
                if any([adapter in name for adapter in ["adapter_pose", "adapter_optical_flow", "adapter_appearance_flow"]]):
                    if ("lora" in name or "dora" in name):
                        param.requires_grad = True
            # 只训练相关head
            for head_name in ["pose", "optical_flow", "appearance_flow"]:
                if head_name in model.heads_dict:
                    for param in model.heads_dict[head_name].parameters():
                        param.requires_grad = True
        else:
            # 训练全部adapter
            for name, param in model.backbone.named_parameters():
                if any([adapter in name for adapter in model.adapter_map.values()]):
                    if ("lora" in name or "dora" in name):
                        param.requires_grad = True
            # 训练全部head
            for head in model.heads_dict.values():
                for param in head.parameters():
                    param.requires_grad = True
        # 3. 双帧embedding始终可训练
        for param in model.backbone.embeddings.parameters():
            param.requires_grad = True
        # 4. 设置train/eval模式
        model.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        self.models["dares_mh"].eval()

    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation"""
        print("-"*20 + "Training"+ "-"*20)
        summed_loss = 0
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            
            # Train position
            self.set_train(train_position_only=True)
            outputs, losses = self.process_batch_position(inputs)
            loss_pose = losses["loss"]

            self.optimizer_pose.zero_grad()
            loss_pose.backward()
            self.optimizer_pose.step()

            # Train depth
            self.set_train(train_position_only=False)
            outputs, losses = self.process_batch_depth(inputs)
            loss_depth = losses["loss"]

            self.optimizer_depth.zero_grad()
            loss_depth.backward()
            self.optimizer_depth.step()

            duration = time.time() - before_op_time

            # Log training
            phase = batch_idx % self.opt.log_frequency == 0
            summed_loss += loss_depth.item()
            if phase:
                avg_loss = loss_depth.item() if batch_idx == 0 else summed_loss/self.opt.log_frequency

                self.log_time(batch_idx, duration, avg_loss)
                self.log("train", outputs, losses)
                summed_loss = 0

            self.step += 1

        self.lr_scheduler_pose.step()
        self.lr_scheduler_depth.step()

    def process_batch_position(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = {}
        # DARES_MH pose 模式，输入为双帧
        pose_input = self.get_dual_frame_input(inputs, frame0=0, frame1=1)
        pose_out = self.models["dares_mh"](pose_input, mode="pose")
        if isinstance(pose_out, tuple):
            outputs["axisangle"], outputs["translation"] = pose_out[:2]
            if len(pose_out) > 2:
                outputs["estimated_intrinsics"] = pose_out[2]
        else:
            outputs["pose"] = pose_out        # === 补全传统outputs内容 ===
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # 获取光流特征
                optical_flow_input = self.get_dual_frame_input(inputs, frame0=f_i, frame1=0)
                optical_flow_reverse_input = self.get_dual_frame_input(inputs, frame0=0, frame1=f_i)
                
                # 计算光流
                optical_flow = self.models["dares_mh"](optical_flow_input, mode="optical_flow")
                optical_flow_reverse = self.models["dares_mh"](optical_flow_reverse_input, mode="optical_flow")
                
                for scale in self.opt.scales:
                    # 设置position和position_reverse
                    outputs[("position", scale, f_i)] = optical_flow[("flow", scale)] if isinstance(optical_flow, dict) else optical_flow
                    outputs[("position_reverse", scale, f_i)] = optical_flow_reverse[("flow", scale)] if isinstance(optical_flow_reverse, dict) else optical_flow_reverse
                    
                    # 计算high resolution版本
                    outputs[("position", "high", scale, f_i)] = F.interpolate(
                        outputs[("position", scale, f_i)],
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=True
                    )
                    outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                        outputs[("position_reverse", scale, f_i)],
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=True
                    )
                    
                    # 计算registration
                    outputs[("registration", scale, f_i)] = self.spatial_transform(
                        inputs[("color", f_i, 0)],
                        outputs[("position", "high", scale, f_i)]
                    )
                    
                    # 计算occu_mask
                    outputs[("occu_mask_backward", scale, f_i)], outputs[("occu_map_backward", scale, f_i)] = \
                        self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                    outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                        outputs[("position", "high", scale, f_i)],
                        outputs[("position_reverse", "high", scale, f_i)]
                    )
                    
                    # 确保我们有0尺度的遮挡掩码
                    if scale != 0:
                        outputs[("occu_mask_backward", 0, f_i)] = outputs[("occu_mask_backward", scale, f_i)]
                
                # 计算appearance flow (transform)
                appearance_flow_input = torch.cat([outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]], 1)
                appearance_flow = self.models["dares_mh"](appearance_flow_input, mode="appearance_flow")
                
                # 处理每个尺度
                for scale in self.opt.scales:
                    # 设置transform
                    outputs[("transform", scale, f_i)] = appearance_flow[("transform", scale)] if isinstance(appearance_flow, dict) else appearance_flow
                    outputs[("transform", "high", scale, f_i)] = F.interpolate(
                        outputs[("transform", scale, f_i)],
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=True
                    )
                    
                    # 计算最终的refined图像
                    outputs[("refined", scale, f_i)] = (
                        outputs[("transform", "high", scale, f_i)] * outputs[("occu_mask_backward", 0, f_i)].detach()
                        + inputs[("color", 0, 0)]
                    )
                    outputs[("refined", scale, f_i)] = torch.clamp(
                        outputs[("refined", scale, f_i)], min=0.0, max=1.0
                    )
        losses = self.compute_position_losses(inputs, outputs)

        return outputs, losses

    def get_depth_input(self, inputs):
        # 单帧输入 [B, 3, H, W]
        return inputs[("color_aug", 0, 0)]

    def get_dual_frame_input(self, inputs, frame0=0, frame1=1):
        # 双帧输入 [B, 6, H, W]，frame0/1 可指定
        return torch.cat([inputs[("color_aug", frame0, 0)], inputs[("color_aug", frame1, 0)]], dim=1)

    def process_batch_depth(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = {}
        depth_input = self.get_depth_input(inputs)
        # DARES_MH depth 模式
        outputs = self.models["dares_mh"](depth_input, mode="depth")
        outputs.update(self.predict_poses(inputs, outputs))        # === 补全传统outputs内容 ===
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # 获取光流特征
                optical_flow_input = self.get_dual_frame_input(inputs, frame0=f_i, frame1=0)
                optical_flow_reverse_input = self.get_dual_frame_input(inputs, frame0=0, frame1=f_i)
                
                # 计算光流
                optical_flow = self.models["dares_mh"](optical_flow_input, mode="optical_flow")
                optical_flow_reverse = self.models["dares_mh"](optical_flow_reverse_input, mode="optical_flow")
                
                for scale in self.opt.scales:
                    # 设置position和position_reverse
                    outputs[("position", scale, f_i)] = optical_flow[("flow", scale)] if isinstance(optical_flow, dict) else optical_flow
                    outputs[("position_reverse", scale, f_i)] = optical_flow_reverse[("flow", scale)] if isinstance(optical_flow_reverse, dict) else optical_flow_reverse
                    
                    # 计算high resolution版本
                    outputs[("position", "high", scale, f_i)] = F.interpolate(
                        outputs[("position", scale, f_i)],
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=True
                    )
                    outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                        outputs[("position_reverse", scale, f_i)],
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=True
                    )
                    
                    # 计算registration
                    outputs[("registration", scale, f_i)] = self.spatial_transform(
                        inputs[("color", f_i, 0)],
                        outputs[("position", "high", scale, f_i)]
                    )
                    
                    # 计算occu_mask
                    outputs[("occu_mask_backward", scale, f_i)], outputs[("occu_map_backward", scale, f_i)] = \
                        self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                    outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                        outputs[("position", "high", scale, f_i)],
                        outputs[("position_reverse", "high", scale, f_i)]
                    )
                    
                    # 确保我们有0尺度的遮挡掩码
                    if scale != 0:
                        outputs[("occu_mask_backward", 0, f_i)] = outputs[("occu_mask_backward", scale, f_i)]
                
                # 计算appearance flow (transform)
                appearance_flow_input = torch.cat([outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]], 1)
                appearance_flow = self.models["dares_mh"](appearance_flow_input, mode="appearance_flow")
                
                # 处理每个尺度
                for scale in self.opt.scales:
                    # 设置transform
                    outputs[("transform", scale, f_i)] = appearance_flow[("transform", scale)] if isinstance(appearance_flow, dict) else appearance_flow
                    outputs[("transform", "high", scale, f_i)] = F.interpolate(
                        outputs[("transform", scale, f_i)],
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=True
                    )
                    
                    # 计算最终的refined图像
                    outputs[("refined", scale, f_i)] = (
                        outputs[("transform", "high", scale, f_i)] * outputs[("occu_mask_backward", 0, f_i)].detach()
                        + inputs[("color", 0, 0)]
                    )
                    outputs[("refined", scale, f_i)] = torch.clamp(
                        outputs[("refined", scale, f_i)], min=0.0, max=1.0
                    )
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_depth_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, disps=None):
        """Predict poses and other outputs for monocular sequences."""
        outputs = {}
        
        # Get all relevant frame IDs
        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                # Handle stereo vision case
                continue
                
            # For negative frame_ids (-1), we need to calculate differently
            if frame_id < 0:
                pose_input = self.get_dual_frame_input(inputs, frame0=0, frame1=frame_id)
                pose_out = self.models["dares_mh"](pose_input, mode="pose")

                if isinstance(pose_out, tuple):
                    axisangle, translation = pose_out[:2]
                    outputs[("axisangle", 0, frame_id)] = axisangle
                    outputs[("translation", 0, frame_id)] = translation

                    # Calculate transformation matrix with invert=True for negative frame IDs
                    outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True
                    )

                    if len(pose_out) > 2:
                        outputs["estimated_intrinsics"] = pose_out[2]
                else:
                    # If pose_out is not a tuple, but a direct transformation matrix
                    # We need to invert it for negative frame IDs
                    outputs[("cam_T_cam", 0, frame_id)] = torch.inverse(pose_out)
            else:
                # For positive frame_ids (1), calculate frame_id -> 0 relative pose
                pose_input = self.get_dual_frame_input(inputs, frame0=frame_id, frame1=0)
                pose_out = self.models["dares_mh"](pose_input, mode="pose")
                
                if isinstance(pose_out, tuple):
                    axisangle, translation = pose_out[:2]
                    outputs[("axisangle", 0, frame_id)] = axisangle
                    outputs[("translation", 0, frame_id)] = translation
                    
                    # Calculate transformation matrix
                    outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False
                    )
                    
                    if len(pose_out) > 2:
                        outputs["estimated_intrinsics"] = pose_out[2]
                else:
                    # If pose_out is not a tuple, but a direct transformation matrix
                    outputs[("cam_T_cam", 0, frame_id)] = pose_out
        
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
            
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            source_scale = 0
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                if self.opt.learn_intrinsics:
                    K = outputs["estimated_intrinsics"]
                    inv_K = torch.inverse(K)
                

                    cam_points = self.backproject_depth[source_scale](
                        depth, inv_K)
                    pix_coords = self.project_3d[source_scale](
                        cam_points, K, T)
                else:
                    cam_points = self.backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                        cam_points, inputs[("K", source_scale)], T)
                
    def compute_supervised_loss(self, inputs, outputs, debug=False):
        if 'depth_gt' in inputs.keys() and inputs['depth_gt'].any():
            depth_gt = inputs['depth_gt']
        else:
            return 0
        depth_pred = outputs[("depth", 0, 0)] * self.opt.max_depth
        mask = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)
            
        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]

        return nn.L1Loss()(depth_pred, depth_gt)
        

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ms_ssim_loss = 1 - self.ms_ssim(pred, target)
            reprojection_loss = 0.9 * ms_ssim_loss + 0.1 * l1_loss

        return reprojection_loss

    def compute_position_losses(self, inputs, outputs):
        """Compute losses for position optimization"""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += get_smooth_loss(outputs[("position", scale, frame_id)], color)
                loss_registration += (
                    self.compute_reprojection_loss(
                        outputs[("registration", scale, frame_id)], 
                        outputs[("refined", scale, frame_id)].detach()
                    ) * occu_mask_backward
                ).sum() / occu_mask_backward.sum()

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)
            
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs):
        """Compute losses for depth optimization"""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_reprojection = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                
                # Compute reprojection loss
                loss_reprojection += (
                    self.compute_reprojection_loss(
                        outputs[("color", frame_id, scale)], 
                        outputs[("refined", scale, frame_id)]
                    ) * occu_mask_backward
                ).sum() / occu_mask_backward.sum()
                
            # Compute disparity smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            disp_smooth_loss = get_smooth_loss(norm_disp, color)

            # Combine losses
            loss += loss_reprojection / 2.0
            loss += self.opt.disparity_smoothness * disp_smooth_loss / (2 ** scale)

            # supervised loss
            if self.use_supervised_loss:
                loss += self.compute_supervised_loss(inputs, outputs)  * 0.001
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss depth: {:.5f} | time elapsed: {} | time left: {} | lr_pos: {} | lr_depth: {}"
        
        lr_pos = str(self.optimizer_pose.param_groups[0]['lr'])
        lr_depth = str(self.optimizer_depth.param_groups[0]['lr'])
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)
                                  ,lr_pos, lr_depth))

    def log(self, mode, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
            if ("disp", 0) in outputs:
                for j in range(min(4, self.opt.batch_size)):  # write a maximum of four images
                    writer.add_image(
                        "disp_0",
                        normalize_image(outputs[("disp", 0)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer_pose.state_dict(), save_path)