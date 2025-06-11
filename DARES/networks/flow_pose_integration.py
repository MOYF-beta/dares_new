"""
Flow-based Pose Prediction Integration
集成基于flow的pose预测到现有训练框架中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowBasedPoseTrainer:
    """
    基于flow的pose预测训练器
    与现有的trainer集成，支持flow-based pose estimation
    """
    
    def __init__(self, opt, models):
        self.opt = opt
        self.models = models
        
        # 添加flow-based pose decoder
        from .flow_based_pose_decoder import FlowBasedPoseDecoder
        self.flow_pose_decoder = FlowBasedPoseDecoder(
            input_height=opt.height,
            input_width=opt.width,
            num_frames_to_predict_for=len(opt.frame_ids) - 1,
            predict_intrinsics=opt.predict_intrinsics if hasattr(opt, 'predict_intrinsics') else False,
            image_width=opt.width,
            image_height=opt.height,
            auto_scale=True
        )
        
        if torch.cuda.is_available():
            self.flow_pose_decoder = self.flow_pose_decoder.cuda()
    
    def predict_poses_from_flows(self, inputs):
        """
        基于已预测的flow信息进行pose预测
        
        Args:
            inputs: 包含optical flow和appearance flow的字典
                   预期包含以下keys:
                   - ("flow", scale, frame_id): optical flow [B, 2, H, W]
                   - ("transform", scale, frame_id): appearance flow [B, 3, H, W]
        
        Returns:
            outputs: 包含pose预测结果的字典
        """
        outputs = {}
        
        # 遍历每个frame id (除了参考帧0)
        for f_i in self.opt.frame_ids[1:]:
            if f_i == "s":
                continue
                
            # 获取最高分辨率的flow信息 (scale=0)
            optical_flow = inputs[("flow", 0, f_i)]      # [B, 2, H, W]
            appearance_flow = inputs[("transform", 0, f_i)]  # [B, 3, H, W]
            
            # 使用flow-based decoder预测pose
            if hasattr(self.flow_pose_decoder, 'predict_intrinsics') and self.flow_pose_decoder.predict_intrinsics:
                axisangle, translation, intrinsics = self.flow_pose_decoder(optical_flow, appearance_flow)
                outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0])
                outputs[("K", f_i)] = intrinsics
            else:
                axisangle, translation = self.flow_pose_decoder(optical_flow, appearance_flow)
                outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0])
        
        return outputs
    
    def predict_poses_hybrid(self, inputs, use_flow_pose=True, use_image_pose=True, fusion_weight=0.5):
        """
        混合方法：同时使用flow-based和image-based pose预测，然后融合
        
        Args:
            use_flow_pose: 是否使用flow-based pose预测
            use_image_pose: 是否使用传统image-based pose预测  
            fusion_weight: flow-based预测的权重 (0-1)
        """
        outputs = {}
        
        for f_i in self.opt.frame_ids[1:]:
            if f_i == "s":
                continue
            
            poses = []
            weights = []
            
            # 1. Flow-based pose prediction
            if use_flow_pose and ("flow", 0, f_i) in inputs and ("transform", 0, f_i) in inputs:
                optical_flow = inputs[("flow", 0, f_i)]
                appearance_flow = inputs[("transform", 0, f_i)]
                
                axisangle_flow, translation_flow = self.flow_pose_decoder(optical_flow, appearance_flow)
                pose_flow = self.transformation_from_parameters(
                    axisangle_flow[:, 0], translation_flow[:, 0])
                
                poses.append(pose_flow)
                weights.append(fusion_weight)
            
            # 2. Traditional image-based pose prediction  
            if use_image_pose:
                # 使用原始的pose预测方法
                pose_inputs = [inputs["color_aug", f_i, 0], inputs["color_aug", 0, 0]]
                
                if "pose" in self.models:
                    # 使用独立的pose网络
                    pose_features = self.models["pose_encoder"](torch.cat(pose_inputs, 1))
                    axisangle_img, translation_img = self.models["pose"](pose_features)
                    pose_img = self.transformation_from_parameters(
                        axisangle_img[:, 0], translation_img[:, 0])
                else:
                    # 使用多任务模型的pose预测
                    dual_frame_input = torch.cat(pose_inputs, dim=1)  # [B, 6, H, W]
                    pose_outputs = self.models["encoder"](dual_frame_input, mode="pose")
                    axisangle_img, translation_img = pose_outputs[0], pose_outputs[1]
                    pose_img = self.transformation_from_parameters(
                        axisangle_img[:, 0], translation_img[:, 0])
                
                poses.append(pose_img)
                weights.append(1.0 - fusion_weight)
            
            # 3. 融合预测结果
            if len(poses) == 1:
                final_pose = poses[0]
            elif len(poses) == 2:
                # 加权融合两个pose矩阵
                final_pose = self.weighted_pose_fusion(poses[0], poses[1], weights[0])
            else:
                raise ValueError("No valid pose predictions available")
            
            outputs[("cam_T_cam", 0, f_i)] = final_pose
        
        return outputs
    
    def weighted_pose_fusion(self, pose1, pose2, weight1):
        """
        加权融合两个pose矩阵
        
        Args:
            pose1, pose2: [B, 4, 4] transformation matrices
            weight1: pose1的权重
        """
        weight2 = 1.0 - weight1
        
        # 提取旋转和平移
        R1 = pose1[:, :3, :3]
        t1 = pose1[:, :3, 3:4]
        R2 = pose2[:, :3, :3]
        t2 = pose2[:, :3, 3:4]
        
        # 平移直接加权平均
        t_fused = weight1 * t1 + weight2 * t2
        
        # 旋转矩阵融合（使用四元数插值会更准确，这里简化处理）
        # 简化方法：对旋转矩阵做加权平均然后重新正交化
        R_weighted = weight1 * R1 + weight2 * R2
        
        # SVD重新正交化
        U, _, V = torch.svd(R_weighted)
        R_fused = torch.bmm(U, V.transpose(-2, -1))
        
        # 重构变换矩阵
        batch_size = pose1.size(0)
        device = pose1.device
        fused_pose = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        fused_pose[:, :3, :3] = R_fused
        fused_pose[:, :3, 3:4] = t_fused
        
        return fused_pose
    
    def transformation_from_parameters(self, axisangle, translation, invert=False):
        """Convert axis-angle and translation to transformation matrix"""
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()

        if invert:
            R = R.transpose(1, 2)
            t *= -1

        T = self.get_translation_matrix(t)
        
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)

        return M
    
    def get_translation_matrix(self, translation_vector):
        """Convert translation vector to transformation matrix"""
        T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3] = translation_vector[:, :, 0]
        return T
    
    def rot_from_axisangle(self, vec):
        """Convert axis-angle to rotation matrix"""
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)

        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca

        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)

        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC

        rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1

        return rot
    
    def compute_flow_pose_loss(self, outputs, inputs):
        """
        计算flow-based pose预测的损失
        可以与传统的重投影损失结合使用
        """
        losses = {}
        
        # 如果有ground truth pose，可以直接计算pose损失
        if "gt_pose" in inputs:
            pose_loss = 0
            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue
                
                pred_pose = outputs[("cam_T_cam", 0, f_i)]
                gt_pose = inputs["gt_pose"][f_i]
                
                # 计算pose距离
                pose_loss += self.pose_distance(pred_pose, gt_pose)
            
            losses["pose_loss"] = pose_loss / len(self.opt.frame_ids[1:])
        
        # 也可以使用重投影一致性作为监督信号
        if "depth" in outputs:
            reproj_loss = self.compute_reprojection_loss(outputs, inputs)
            losses["reproj_loss"] = reproj_loss
        
        return losses
    
    def pose_distance(self, pose1, pose2):
        """计算两个pose之间的距离"""
        # 旋转部分的距离
        R1 = pose1[:, :3, :3]
        R2 = pose2[:, :3, :3]
        R_diff = torch.bmm(R1, R2.transpose(-2, -1))
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
        rot_dist = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        # 平移部分的距离  
        t1 = pose1[:, :3, 3]
        t2 = pose2[:, :3, 3]
        trans_dist = torch.norm(t1 - t2, dim=1)
        
        return rot_dist.mean() + trans_dist.mean()


# 使用示例
def integrate_flow_pose_prediction(trainer_class):
    """
    将flow-based pose预测集成到现有trainer中
    """
    
    class EnhancedTrainer(trainer_class):
        def __init__(self, options):
            super().__init__(options)
            
            # 初始化flow-based pose预测器
            self.flow_pose_trainer = FlowBasedPoseTrainer(self.opt, self.models)
        
        def predict_poses(self, inputs, disps=None):
            """重写pose预测方法，支持flow-based预测"""
            
            # 首先运行原始的flow预测
            flow_outputs = super().predict_poses(inputs, disps)
            
            # 然后基于flow进行pose预测
            if self.opt.use_flow_pose:
                flow_pose_outputs = self.flow_pose_trainer.predict_poses_from_flows(flow_outputs)
                
                # 选择使用哪种pose预测结果
                if self.opt.pose_fusion_mode == "flow_only":
                    # 只使用flow-based pose
                    for key in flow_pose_outputs:
                        if key.startswith("cam_T_cam"):
                            flow_outputs[key] = flow_pose_outputs[key]
                
                elif self.opt.pose_fusion_mode == "hybrid":
                    # 混合使用
                    hybrid_outputs = self.flow_pose_trainer.predict_poses_hybrid(
                        {**inputs, **flow_outputs}, 
                        fusion_weight=self.opt.flow_pose_weight
                    )
                    flow_outputs.update(hybrid_outputs)
            
            return flow_outputs
        
        def compute_losses(self, inputs, outputs):
            """增强损失计算，包含flow-pose损失"""
            losses = super().compute_losses(inputs, outputs)
            
            # 添加flow-based pose损失
            if self.opt.use_flow_pose:
                flow_pose_losses = self.flow_pose_trainer.compute_flow_pose_loss(outputs, inputs)
                losses.update(flow_pose_losses)
            
            return losses
    
    return EnhancedTrainer
