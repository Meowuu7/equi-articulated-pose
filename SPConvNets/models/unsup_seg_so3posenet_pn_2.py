import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import time
from collections import OrderedDict
import json
import vgtk
import SPConvNets.utils.base_so3poseconv as M
import vgtk.so3conv.functional as L
import vgtk.so3conv as sptk
from SPConvNets.utils.slot_attention import SlotAttention
import vgtk.spconv as zptk
from SPConvNets.utils.loss_util import batched_index_select
from extensions.chamfer_dist import ChamferDistance

from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
from model_util import farthest_point_sampling



class ClsSO3ConvModel(nn.Module): # SO(3) equi-conv-network #
    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()

        # get backbone model
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']: # backbone
            self.backbone.append(M.BasicSO3PoseConvBlock(block_param))
        print(f"number of convs in the backbone: {len(self.backbone)}")
        # self.outblock = M.ClsOutBlockR(params['outblock'])
        # output classification block
        #
        self.chamfer_dist = ChamferDistance()
        ''' Get anchors '''
        self.anchors = torch.from_numpy(L.get_anchors(params['outblock']['kanchor'])).cuda()
        self.n_reconstructed = 128
        self.outblock = M.ClsOutBlockPointnet(params['outblock'], down_task=False) # clsoutblockpointnet?
        # PointNet Encoder
        # self.pointnetenc = sptk.PointnetSO3Conv(dim_in=256, dim_out=1024, kanchor=60)
        # Need a decoder for position and latent variant features --- but it is what makes it tricky --- we need implicit shape decoded from invariant features as well as each point's variant implicit features, we should factorize positiona and pose to a canonical frame with position and pose from the equivariant features --- position & variant features
        # a equivariant point completion
        # todo: better MLP models
        ''' Construct canonical position decoding block '''
        # encoded feature dimension
        self.encoded_feat_dim = params['outblock']['dim_in']
        self.kanchor = params['outblock']['kanchor']
        self.xyz_canon_in_feat_dim = self.encoded_feat_dim * self.kanchor
        self.xyz_canon_in_feat_dim = self.encoded_feat_dim
        self.xyz_canon_block = nn.Sequential(
            nn.Conv2d(in_channels=self.xyz_canon_in_feat_dim, out_channels=self.xyz_canon_in_feat_dim // 2, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.xyz_canon_in_feat_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.xyz_canon_in_feat_dim // 2, out_channels=self.xyz_canon_in_feat_dim // 4, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.xyz_canon_in_feat_dim // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.xyz_canon_in_feat_dim // 4, out_channels=3, kernel_size=(1, 1), stride=(1, 1), bias=True),
        )
        ''' Construct pose estimation block '''
        self.pose_estimation_block = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim // 2, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 2, out_channels=self.encoded_feat_dim // 4,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 4, out_channels=12, kernel_size=(1, 1), stride=(1, 1),
                      bias=True),
        )
        ''' Construct slot-attention module now.. '''
        ### eps is set to default; we may need to tune `dim` and `hidden_dim` ###
        ### output feature shape: bz x num_slots x dim ###
        self.num_slots = params['outblock']['k']
        self.slot_attention = SlotAttention(num_slots=params['outblock']['k'], dim=self.encoded_feat_dim, hidden_dim=self.encoded_feat_dim)
        ''' Construct per-point variant feature transformation MLP '''
        self.variant_feat_trans = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
        )
        ''' Construct per-slot rotation and translation prediction MLP '''
        self.transformation_dim = 7
        self.transformation_prediction = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim // 2, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 2, out_channels=self.encoded_feat_dim // 4,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 4, out_channels=self.transformation_dim,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
        )
        ''' Construct part point construction network '''
        # todo: better reconstruction process
        self.recon_part_M = 96
        self.part_reconstruction_net = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.recon_part_M * 3,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
        )

        # self.xyz_canon_block = nn.Linear(self.encoded_feat_dim * self.kanchor, 3)
        self.na_in = params['na'] # na_in
        # todo: what does this parameter used for?
        self.invariance = True


    # R.size = bz x N x na x 3 x 3
    def get_rotation_matrix(self, Rs):
        a1s = Rs[:, :, :, :, 0].unsqueeze(-1)
        a2s = Rs[:, :, :, :, 1].unsqueeze(-1)
        b1s = a1s / torch.norm(a1s, dim=3, p=2, keepdim=True)
        b2s = a2s - (torch.sum(b1s * a2s, dim=3, keepdim=True)) * b1s
        b2s = b2s / torch.norm(b2s, dim=3, p=2, keepdim=True)
        b3s = torch.zeros_like(b2s)
        b3s[..., 0, 0] = b1s[..., 1, 0] * b2s[..., 2, 0] - b1s[..., 2, 0] * b2s[..., 1, 0]
        b3s[..., 1, 0] = -(b1s[..., 0, 0] * b2s[..., 2, 0] - b1s[..., 2, 0] * b2s[..., 0, 0])
        b3s[..., 2, 0] = b1s[..., 0, 0] * b2s[..., 1, 0] - b1s[..., 1, 0] * b2s[..., 0, 0]
        # tb1, tb2, tb3 = b1s[0, 0, :, 0], b2s[0, 0, :, 0], b3s[0, 0, :, 0]
        Rs = torch.cat([b1s, b2s, b3s], dim=-1)
        # print(torch.sum((torch.det(Rs) < 0).long()))
        return Rs

    def forward_one_iter(self, x, pose, rlabel=None): # rotation label

        output = {}
        #
        # should with pose as input; actually an orientation of each point --- not orientation... te rotation has been done? fix the global pose of each input point; relative pose relations --- how to compute? create laptops and put the joint-center to the original point
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        ori_pts = x.clone()
        bz, npoints = x.size(0), x.size(2)
        # input_x = x # preprocess input and equivariant
        #### Preprocess input and equivariant feature transformation ####
        x = M.preprocess_input(x, self.na_in, pose, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)

        # x: (out_feature: bz x C x N x 1; atten_weight: bz x N x A)
        # get position information
        x_xyz = x.xyz; x_anchors = x.anchors; x_pose = x.pose

        ''' Get canonicalized xyz '''
        # # a flatten of [c_out x na]-dim feature for invariant position estimation
        # # function(per-point-flatten-feature) -> per-point position
        # # x.feat: bz x c_out x N x na
        # x_feat_expand = x.feats.contiguous().permute(0, 2, 1, 3).contiguous()
        # x_feat_expand = x_feat_expand.view(x_feat_expand.size(0), x_feat_expand.size(1), self.xyz_canon_in_feat_dim)
        # # x_feat_expand: bz x feat_dim x N x 1
        # x_feat_expand = x_feat_expand.contiguous().transpose(1, 2).contiguous().unsqueeze(-1)
        # # canonicalized_xyz: bz x 3 x N x 1 -> bz x 3 x N
        # canonicalized_xyz = self.xyz_canon_block(x_feat_expand).squeeze(-1)
        #
        # canonicalized_xyz = self.xyz_canon_block(x.feats) # not one-to-one, but generate points
        ''' Cluster points via slot attention; clustering; inv_feat '''
        # inv_feat
        inv_feat, atten_score = self.outblock(x, rlabel)
        # inv_feat: bz x c_out x N
        inv_feat = inv_feat.squeeze(-1) # squeeze invariant features
        ''' Cluster '''
        # rep_slots: bz x num_slots x c_out; attn: bz x N x num_slots
        # slot representation vectors and slot attention vectors
        # invariant features for clustering via slot-attention
        rep_slots, attn_ori = self.slot_attention(inv_feat.contiguous().transpose(1, 2).contiguous())
        # x.feat: bz x c_out x N x na
        # aggregated_feat: bz x c_out x N x num_slots x na; attn: bz x n_s x N
        ''' Attention from each point to each cluster '''
        # attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)
        # attn: bz x N x num_slots
        ### for vis ###
        point_label = torch.argmax(attn_ori, dim=1)
        # if not os.path.exists("vis_pts.npy"):
        np.save("vis_pts.npy", ori_pts.detach().cpu().numpy())
        np.save("vis_labels.npy", point_label.detach().cpu().numpy())
        ### for vis ###
        attn = attn_ori.contiguous().transpose(1, 2).contiguous()
        # slot_weights: bz x num_slots
        slot_weights = attn.sum(dim=1)
        slot_weights = slot_weights / torch.sum(slot_weights, dim=-1, keepdim=True)
        # attn_ori: bz x num_slots x N -> bz x N x num_slots
        # attn_slot: bz x num_slots x N
        attn_slot = attn_ori / attn_ori.sum(dim=-1, keepdim=True)
        attn_ori = attn_ori.contiguous().transpose(1, 2).contiguous()


        ''' Get variant features for each slot '''
        transformed_variant_feats = self.variant_feat_trans(x.feats)
        ''' Aggregate variant feats for each slot '''
        # todo: is it the most suitable way to aggegate features?
        # transformed_variant_feats: bz x c_out x N x na; attn_ori: bz x num_slots x N
        # variant_feats_slot: bz x c_out x num_slots x N x na -> bz x c_out x num_slots x na
        # print(transformed_variant_feats.size(), attn_ori.size())
        # print(transformed_variant_feats.size(), attn_slot.size())
        variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * attn_slot.unsqueeze(1).unsqueeze(-1), dim=3)
        ''' Predict \delta_q and translation for each rotation state q '''
        per_slot_transformation = self.transformation_prediction(variant_feats_slot)
        # pred_R: bz x 4 x num_slots x na
        pred_R, pred_T = per_slot_transformation[:, :4, ...], per_slot_transformation[:,4:, ...]
        output['pred_R'], output['pred_T'] = pred_R, pred_T
        R_reg_loss = torch.mean(torch.sqrt((pred_R ** 2).sum(dim=1)) - 1.) # reg R
        ''' Predict points for each part '''
        slot_points = self.part_reconstruction_net(rep_slots.contiguous().transpose(1, 2).contiguous().unsqueeze(-1)).contiguous().squeeze(-1)
        # slot_points: bz x num_slots x M x 3
        slot_points = slot_points.contiguous().transpose(1, 2).contiguous().view(bz, self.num_slots, self.recon_part_M, 3)
        np.save("slot_pts.npy", slot_points.detach().cpu().numpy())
        ''' Apply estimated rotations and translations on the estimated points '''
        # transfer estimated qua to rotation matrix & use rotation matrix and translation vector to transform position matrix
        # calculate chamber distance between shape and each estimated parts & overall chamfer distance between downsampled shape and original shape
        # may use compute_rotation_matrix_from_quaternion, self.anchors
        # pred_T          = self.output_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous()
        #                 if na > 1:
        #                     pred_T      = torch.matmul(anchors, pred_T) # nb, na, 3, 1,
        ''' From predicted res_R to residual rotation matrix '''
        # pred_res_R: bz x num_slots x na x 3 x 3
        # print(pred_R.size())
        # print(compute_rotation_matrix_from_quaternion(pred_R.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, 4)).size())
        pred_res_R = compute_rotation_matrix_from_quaternion(pred_R.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, 4)).contiguous().view(bz, self.num_slots, self.kanchor, 3, 3)

        pred_R = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), pred_res_R)

        ### pred_res_R: bz x 4 x n_s x n_a


        ''' From predicted T to the real translation vector --- part rotation modeling '''
        # pred_res_T: bz x num_slots x na x 3
        pred_res_T = pred_T.contiguous().permute(0, 2, 3, 1).contiguous()
        # pred_T: bz x num_slots x na x 3
        pred_T = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), pred_res_T.unsqueeze(-1)).squeeze(-1)
        # todo: other transformation strategy, like those used in equi-pose?
        ''' From predicted rotation matrix and translation matrix to transformed points  '''
        # transformed_slot_pts: bz x num_slots x na x M x 3
        transformed_slot_pts = torch.matmul(pred_R, slot_points.contiguous().transpose(-1, -2).unsqueeze(2)).contiguous().transpose(-1, -2) + pred_T.unsqueeze(3)
        ''' Repeat input points for further chamfer distance computation '''
        # ori_pts: bz x 3 x N
        input_repeat_pts = ori_pts.contiguous().transpose(1, 2).contiguous().unsqueeze(1).unsqueeze(1).repeat(1, self.num_slots, self.kanchor, 1, 1)
        # dist1: -1 x M; dist2: -1 x N
        dist1, dist2 = self.chamfer_dist(transformed_slot_pts.view(-1, self.recon_part_M, 3).contiguous(),
                                         input_repeat_pts.contiguous().view(-1, input_repeat_pts.size(-2), 3).contiguous(),
                                         return_raw=True)
        # dist1 = dist1.contiguous().view(bz, self.num_slots, self.kanchor, self.recon_part_M)
        dist2 = dist2.contiguous().view(bz, self.num_slots, self.kanchor, npoints)
        # attn_ori: bz x N x ns --> bz x ns x 1 x N
        # dist2: bz x num_slots x na x N
        # print(attn_ori.size(), dist2.size())
        dist2 = torch.sum(attn_slot.contiguous().unsqueeze(2) * dist2, dim=-1)
        ''' Get used distances and selected rotation state for each slot '''
        # dist2: bz x num_slots
        # dist2 = dist2.min(dim=-1)
        dist2, rot_state_per_slot = torch.min(dist2, dim=-1)
        ''' Select transformed points, rotation matrix and translation vectors '''
        # transformed_slot_pts: bz x num_slots x M x 3
        transformed_slot_pts = batched_index_select(transformed_slot_pts, rot_state_per_slot.unsqueeze(-1), dim=2)
        # selected_R: bz x num_slots x 3 x 3
        selected_R = batched_index_select(pred_R, rot_state_per_slot.unsqueeze(-1), dim=2)
        # selected_T: bz x num_slots x 3
        selected_T = batched_index_select(pred_T, rot_state_per_slot.unsqueeze(-1), dim=2)
        # print(selected_R.size(), selected_T.size())
        selected_R_expanded = selected_R.repeat(1, 1, self.recon_part_M, 1, 1)
        selected_T_expanded = selected_T.repeat(1, 1, self.recon_part_M, 1)
        if len(transformed_slot_pts.size()) > 4:
            transformed_slot_pts = transformed_slot_pts.squeeze(2)
        transformed_slot_pts = transformed_slot_pts.contiguous().view(bz, -1, 3)
        # bz x n_recon x 3
        print(torch.mean(torch.mean(transformed_slot_pts, dim=0), dim=0))
        print("ori-shape: ", torch.mean(torch.mean(ori_pts, dim=-1), dim=0))
        ''' Sample points '''
        fps_idx = farthest_point_sampling(transformed_slot_pts, self.n_reconstructed)
        downsampled_transformed_pts = transformed_slot_pts.contiguous().view(bz * self.recon_part_M * self.num_slots, -1)[fps_idx, :].contiguous().view(bz, self.n_reconstructed, -1)
        downsampled_R = selected_R_expanded.contiguous().view(bz * self.recon_part_M * self.num_slots, 3, 3)[fps_idx, :, :].contiguous().view(bz, self.n_reconstructed, 3, 3)
        downsampled_T = selected_T_expanded.contiguous().view(bz * self.recon_part_M * self.num_slots, 3)[fps_idx, :].contiguous().view(bz, self.n_reconstructed, 3)

        ''' Get reconstruction loss between transformed points and original input points '''
        # dist1_glb: bz x n_recon; dist2_glb: bz x npoints
        # print(downsampled_transformed_pts.size(), input_repeat_pts.size())
        dist1_glb, dist2_glb = self.chamfer_dist(
            downsampled_transformed_pts, ori_pts.contiguous().transpose(1, 2).contiguous(), return_raw=True
        )

        ''' Get reconstruction loss for global shape and local part '''
        glb_recon_loss = dist1_glb.mean(-1) + dist2_glb.mean(-1)
        # print(slot_weights.size(), dist2.size())
        lal_recon_loss = torch.sum(slot_weights * dist2, dim=-1)
        recon_loss = (glb_recon_loss + lal_recon_loss).mean() + 0.1 * R_reg_loss

        ''' Get down-sampled points and pose '''
        downsampled_transformed_pts = downsampled_transformed_pts.contiguous().transpose(1, 2).contiguous()
        downsampled_pose = torch.cat([downsampled_R, downsampled_T.unsqueeze(-1)], dim=-1)
        downsampled_pose = torch.cat([downsampled_pose, torch.zeros((bz, self.n_reconstructed, 1, 4)).cuda()], dim=-2)

        return recon_loss, attn, downsampled_transformed_pts, downsampled_pose

    def get_rotation_sims(self, gt_rot, pred_rot):
        if gt_rot.size(-1) > 3:
            gt_rot = gt_rot[..., :3, :3]
            pred_rot = pred_rot[..., :3, :3]
        # gt_rot: bz x npoints x 3 x 3;
        def get_trace(a):
            return a[..., 0, 0] + a[..., 1, 1] + a[..., 2, 2]
        inv_gt_rot = gt_rot.contiguous().transpose(2, 3).contiguous()
        # bz x npoints x 3 x 3
        rel_mtx = torch.matmul(pred_rot, inv_gt_rot)
        # traces: bz x npoints
        traces = get_trace(rel_mtx)
        traces = (traces - 1) / 2.
        print("Similartiy with gt_rot", torch.mean(traces).item())
        return torch.mean(traces).item()

    def forward(self, x, pose, rlabel=None, nn_inter=2):

        # loss, attn = self.forward_one_iter(x, pose, rlabel=rlabel)
        # return loss, attn

        bz, np = x.size(0), x.size(2)
        init_pose = torch.zeros([bz, np, 4, 4], dtype=torch.float32).cuda()
        init_pose[..., 0, 0] = 1.; init_pose[..., 1, 1] = 1.; init_pose[..., 2, 2] = 1.
        tot_loss = 0.0
        cur_transformed_points = x
        cur_estimated_pose = init_pose
        nn_inter = 1
        cur_estimated_pose = pose
        for i in range(nn_inter):
            cur_reconstructed_loss_orbit, attn, cur_transformed_points, cur_estimated_pose = self.forward_one_iter(cur_transformed_points, cur_estimated_pose, rlabel=rlabel)
            tot_loss += cur_reconstructed_loss_orbit
            cur_gt_rot_dis = self.get_rotation_sims(pose, cur_estimated_pose)
            torch.cuda.empty_cache()

        return tot_loss / nn_inter, attn # range(n_iter)

    def get_anchor(self):
        return self.backbone[-1].get_anchor()


# Full Version
def build_model(opt,
                mlps=[[64,64], [128,128], [256,256],[256]],
                out_mlps=[256],
                strides=[2,2,2,2],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.4,
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                input_radius = 1.0,
                sigma_ratio= 0.5, # 0.1
                xyz_pooling = None,
                so3_pooling = "max",
                to_file=None):
    # initial_radius_ratio = 0.05
    initial_radius_ratio = 0.15
    initial_radius_ratio = 0.20
    device = opt.device
    input_num = opt.model.input_num # 1024
    dropout_rate = opt.model.dropout_rate # default setting: 0.0
    # temperature
    temperature = opt.train_loss.temperature # set temperature
    so3_pooling = 'attention' #  opt.model.flag # model flag
    na = 1 if opt.model.kpconv else opt.model.kanchor # how to represent rotation possibilities? --- sampling from the sphere ---- points!
    nmasks = opt.nmasks

    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    params = {'name': 'Invariant ZPConv Model',
              'backbone': [],
              'na': na
              }

    dim_in = 1

    # process args
    n_layer = len(mlps)
    stride_current = 1 # stride_current_--
    stride_multipliers = [stride_current]
    # for i in range(n_layer):
    #     stride_current *= 2 # strides[i]
    #     stride_multipliers += [stride_current]
    # todo: use ohter strides? --- possible other choices?
    for i in range(n_layer):
        stride_current *= 1 # strides[i]
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]
    # radius ratio should increase as the stride increases to sample more reasonable points
    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]
    # radius_ratio = [0.25, 0.5]
    # set radius for each layer
    radii = [r * input_radius for r in radius_ratio]
    # Compute sigma
    # weighted_sigma = [sigma_ratio * radii[i]**2 * stride_multipliers[i] for i in range(n_layer + 1)]
    # sigma for radius and points
    weighted_sigma = [sigma_ratio * radii[0]**2]

    for idx, s in enumerate(strides):
        # weighted_sigma.append(weighted_sigma[idx] * 2)
        weighted_sigma.append(weighted_sigma[idx] * 1) #

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0
            stride_conv = i == 0 or xyz_pooling != 'stride'
            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            # neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
            neighbor = 32 # int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
            # if i==0 and j==0:
            #    neighbor *= int(input_num/1024)
            kernel_size = 1
            # if j == 0:
            #     # stride at first (if applicable), enforced at first layer
            #     inter_stride = strides[i]
            #     nidx = i if i == 0 else i+1
            #     if stride_conv:
            #         neighbor *= 2 # = 2 * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
            #         # kernel_size = 1 # if inter_stride < 4 else 3
            # else:
            inter_stride = 1
            nidx = i+1

            print(f"At block {i}, layer {j}!")
            print(f'neighbor: {neighbor}')
            print(f'stride: {inter_stride}')
            sigma_to_print = weighted_sigma[nidx]**2 / 3
            print(f'sigma: {sigma_to_print}')
            print(f'radius ratio: {radius_ratio[nidx]}')

            # one-inter one-intra policy
            block_type = 'inter_block' if na<60 else 'separable_block' # point-conv and group-conv separable conv
            print(f"layer {i}, block {j}, block_type: {block_type}")
            conv_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': inter_stride,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'n_neighbor': neighbor,
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling':  xyz_pooling,
                    'kanchor': na,
                    'norm': 'BatchNorm2d',
                }
            }
            block_param.append(conv_param)
            dim_in = dim_out

        params['backbone'].append(block_param)

    # kernels here are defined as kernel points --- explicit kernels --- each with a [dim_in, dim_out] weight matrix
    params['outblock'] = {
            'dim_in': dim_in,
            'mlp': out_mlps,
            'fc': [64],
            'k': nmasks, # 40,
            'pooling': so3_pooling,
            'temperature': temperature,
            'kanchor':na,
    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
