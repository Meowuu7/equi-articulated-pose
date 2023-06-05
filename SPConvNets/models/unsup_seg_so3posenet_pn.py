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
import vgtk.spconv.functional as L
import vgtk.so3conv as sptk
# from SPConvNets.utils.slot_attention import SlotAttention
import vgtk.spconv as zptk
from SPConvNets.utils.loss_util import batched_index_select


class ClsSO3ConvModel(nn.Module): # SO(3) equi-conv-network #
    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()

        # get backbone model
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']: # backbone
            self.backbone.append(M.BasicSO3PoseConvBlock(block_param))
        # self.outblock = M.ClsOutBlockR(params['outblock'])
        # output classification block
        #
        self.outblock = M.ClsOutBlockPointnet(params['outblock'], down_task=False) # clsoutblockpointnet?
        # PointNet Encoder
        # self.pointnetenc = sptk.PointnetSO3Conv(dim_in=256, dim_out=1024, kanchor=60)
        # Need a decoder for position and latent variant features --- but it is what makes it tricky --- we need implicit shape decoded from invariant features as well as each point's variant implicit features, we should factorize positiona and pose to a canonical frame with position and pose from the equivariant features --- position & variant features
        # a equivariant point completion
        # todo: it seems that it is not an easy thing to do the point number transformation via an easy way; so we first focus on complete shapes
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
        #
        # should with pose as input; actually an orientation of each point --- not orientation... te rotation has been done? fix the global pose of each input point; relative pose relations --- how to compute? create laptops and put the joint-center to the original point
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        bz, npoints = x.size(0), x.size(2)
        # input_x = x
        #### Preprocess input and equivariant feature transformation ####
        x = M.preprocess_input(x, self.na_in, pose, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        #### Preprocess input and equivariant feature transformation ####
        #### Get equi-feats for further processing ####
        equi_x_feat = x.feats.detach().cpu()

        # from the output feature to others?
        # no movign points and moving points?
        # equi_x_feat.size = bz x c_in x N x na
        equi_feats = []
        for i in range(equi_x_feat.size(0)):
            equi_feats.append(equi_x_feat[i])
        equi_feat_a, equi_feat_b = equi_feats[0], equi_feats[1]
        diffs = []
        coo = []
        for j in range(equi_x_feat.size(2)):
            # c_in x na; c_in x na
            a_p_feat, b_p_feat = equi_feat_a[:, j, :], equi_feat_b[:, j, :]
            a_p_b_p_feat = torch.sum((a_p_feat.unsqueeze(-1) - b_p_feat.unsqueeze(1)) ** 2, dim=0) # na x na
            # a_p_b_p_feat_min_dist = torch.min(a_p_b_p_feat).item()
            b_min_value, b_min_idx = torch.min(a_p_b_p_feat, dim=1)
            a_min_value, a_min_idx = torch.min(b_min_value, dim=0)
            diffs.append(a_min_value)
            coo.append(float(abs(a_min_idx.item() - b_min_idx[a_min_idx.item()].item())))
        print(sum(diffs), sum(coo) / equi_x_feat.size(2))

        #### Get equi-feats for further processing ####


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

        canonicalized_xyz = self.xyz_canon_block(x.feats)
        ''' Cluster points via slot attention '''
        inv_feat, atten_score = self.outblock(x, rlabel)
        # inv_feat: bz x c_out x N
        inv_feat = inv_feat.squeeze(-1) # squeeze invariant features
        # rep_slots: bz x num_slots x c_out; attn: bz x N x num_slots # slot representation vectors and slot attention vectors
        rep_slots, attn_ori = self.slot_attention(inv_feat.contiguous().transpose(1, 2).contiguous())
        # x.feat: bz x c_out x N x na
        # aggregated_feat: bz x c_out x N x num_slots x na
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)
        attn = attn.contiguous().transpose(1, 2).contiguous()
        attn_ori = attn_ori.contiguous().transpose(1, 2).contiguous()

        # print(x.feats.size(), attn.size())
        # attn: bz x N x num_slots
        # x.feats: bz x c_out x N x na
        aggregated_feat = x.feats.unsqueeze(3) * attn.unsqueeze(1).unsqueeze(-1)
        # aggregated_feat: bz x c_out x num_slots x na
        aggregated_feat = aggregated_feat.sum(2)

        ''' Estimate pose from aggregated feature '''
        # estimated_pose: bz x 12 x num_slots x na
        estimated_pose_per_slot = self.pose_estimation_block(aggregated_feat)
        #
        estimated_rot = estimated_pose_per_slot[:, :9, :, :].contiguous().permute(0, 2, 3, 1).contiguous().view(bz, self.num_slots, self.kanchor, 3, 3).contiguous()
        # estimated_rot: bz x num_slots x na x 3 x 3
        estimated_rot = self.get_rotation_matrix(estimated_rot)
        estimated_trans = estimated_pose_per_slot[:, 9:, :, :].contiguous().permute(0, 2, 3, 1).contiguous()
        # point_slot: bz x N x 1 --- for the index of slots
        # point_slot = torch.argmax(attn, dim=-1)
        point_slot = torch.argmax(attn_ori, dim=-1)
        # estimated_rot_point: bz x N x na x 3 x 3
        # estimated_trans_point: bz x N x na x 3
        estimated_rot_point = batched_index_select(values=estimated_rot, indices=point_slot, dim=1)
        estimated_trans_point = batched_index_select(values=estimated_trans, indices=point_slot, dim=1)

        ''' Calculate reconstruction loss '''
        ### Calculate reconstructed positions ###
        # reconstructed_points: bz x N x na x 3

        # canonicalized_xyz = canonicalized_xyz.contiguous().transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, self.kanchor, 1)
        canonicalized_xyz = canonicalized_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # print(estimated_rot_point.size(), canonicalized_xyz.size(), estimated_trans_point.size())
        # rot: bz x N x na x 3 x 3; xyz: bz x N x 3

        reconstructed_points = torch.matmul(estimated_rot_point, canonicalized_xyz.unsqueeze(-1)).contiguous().squeeze(-1) + estimated_trans_point
        #
        reconstructed_loss_orbit = torch.sum((reconstructed_points - x_xyz.contiguous().transpose(1, 2).unsqueeze(2)) ** 2, dim=-1).mean(1)
        # selected_orbit: bz
        reconstructed_loss_orbit, selected_orbit = torch.min(reconstructed_loss_orbit, dim=-1)
        # selected_reconstructed_points: bz x N x 3
        # print(reconstructed_points.size(), selected_orbit.size())
        selected_orbit = selected_orbit.unsqueeze(-1)
        selected_reconstructed_points = batched_index_select(reconstructed_points.contiguous().transpose(1, 2).contiguous(), selected_orbit, dim=1)
        # selected_estimated_rot: bz x N x 3 x 3
        selected_estimated_rot = batched_index_select(estimated_rot_point.contiguous().transpose(1, 2).contiguous(), selected_orbit, dim=1)
        # selected_estimated_trans: bz x N x 3
        selected_estimated_trans = batched_index_select(estimated_trans_point.contiguous().transpose(1, 2).contiguous(), selected_orbit, dim=1)
        # selected_transformed_points: bz x N x 3
        selected_transformed_points = torch.matmul(selected_estimated_rot, selected_reconstructed_points.unsqueeze(-1)).squeeze(-1) + selected_estimated_trans
        selected_estimated_pose = torch.cat([selected_estimated_rot, selected_estimated_trans.unsqueeze(-1)], dim=-1)
        # selected_estimated_pose: bz x N x 4 x 4
        # print(selected_estimated_pose.size())
        selected_estimated_pose = selected_estimated_pose.squeeze(1)
        selected_transformed_points = selected_transformed_points.squeeze(1).contiguous().transpose(1, 2).contiguous()
        selected_estimated_pose = torch.cat([selected_estimated_pose, torch.zeros([bz, npoints, 1, 4], dtype=torch.float32).cuda()], dim=2)

        reconstructed_loss_orbit = reconstructed_loss_orbit.mean()
        return reconstructed_loss_orbit, selected_transformed_points, selected_estimated_pose, attn_ori

        # x = self.outblock(x, rlabel)
        # # Get invariant features and attention scores (equi-variant features)
        # inv_feat, atten_socre = x
        # # Set object x using original xyz matrix, invariant feature matrix, anchors and pose
        # obj_x = zptk.SphericalPointCloudPose(x_xyz, inv_feat, x_anchors, x_pose)
        # # Get invariant global feature from PointNet encoder; inv_glb_feat: bz x C_glb x 1
        # inv_glb_feat = self.pointnetenc(obj_x)
        # # inv_glb_feat: bz x C_glb
        # inv_glb_feat = inv_glb_feat.squeeze(-1) # encoder, then we need to get decoder
        # #
        #
        #
        #
        #
        # # equivariance features and invariance features;
        # #### An implicit canonical frame ####
        # # feed into a reconstruction module; bz x pos_dim x Z --- feed into the slot-attention-2 module for clustering
        # # feed into a pose transformation module: bz x implicit_feat_dim (perhaps equal to A) x Z --- soft aggregation implicit features for each cluster and then transform them to the estimated pose of this cluster
        # # estimated pose for each cluster is then assigned to each point for the next iteration
        # #### An implicit canonical frame ####
        # # we can use the reconstructed shape from the implicit shape directly into the next iteration
        # return x

    def forward(self, x, pose, rlabel=None, nn_inter=2):

        bz, np = x.size(0), x.size(2)
        init_pose = torch.zeros([bz, np, 4, 4], dtype=torch.float32).cuda()
        init_pose[..., 0, 0] = 1.; init_pose[..., 1, 1] = 1.; init_pose[..., 2, 2] = 1.
        tot_loss = 0.0
        cur_transformed_points = x
        cur_estimated_pose = init_pose
        nn_inter = 1
        cur_estimated_pose = pose
        for i in range(nn_inter):
            cur_reconstructed_loss_orbit, cur_transformed_points, cur_estimated_pose, attn = self.forward_one_iter(cur_transformed_points, cur_estimated_pose, rlabel=rlabel)
            # print(cur_reconstructed_loss_orbit.size(), cur_transformed_points.size(), cur_estimated_pose.size(), attn.size())
            tot_loss += cur_reconstructed_loss_orbit
        return tot_loss / nn_inter, attn

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
