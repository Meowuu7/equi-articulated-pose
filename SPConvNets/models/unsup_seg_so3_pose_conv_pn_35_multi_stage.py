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
import SPConvNets.utils.base_so3conv as Mso3
import vgtk.so3conv.functional as L
import vgtk.so3conv as sptk
from SPConvNets.utils.slot_attention import SlotAttention
from SPConvNets.utils.slot_attention_spec import SlotAttention
from SPConvNets.utils.slot_attention_spec_v2 import SlotAttention
# from SPConvNets.utils.slot_attention_orbit import SlotAttention
# import vgtk.spconv as zptk
import vgtk.spconv as zpconv
from SPConvNets.utils.loss_util import batched_index_select
from extensions.chamfer_dist import ChamferDistance
# from chamfer_distance import ChamferDistance
from DGCNN import PrimitiveNet
from SPConvNets.models.common_utils import *
from SPConvNets.models.model_utils import *

from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
# from model_util import farthest_point_sampling, DecoderFC, DecoderFCAtlas, DecoderConstantCommon
from SPConvNets.models.model_util import *


class ClsSO3ConvModel(nn.Module):  # SO(3) equi-conv-network #
    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()
        ''' EPN for global rotation factorization '''
        self.glb_backbone = nn.ModuleList()
        # use equi as the backbone for global pose factorization
        for block_param in params['backbone']:
            self.glb_backbone.append(M.BasicSO3PoseConvBlock(block_param))

        self.backbone = nn.ModuleList()
        for block_param in params['kpconv_backbone']:
        # for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3PoseConvBlock(block_param))

        # todo: try different `pooling method`

        ''' Set useful arguments '''
        ### Specific model-related parameter settings ####
        self.encoded_feat_dim = params['outblock']['dim_in']
        # self.encoded_feat_dim = 128
        self.inv_out_dim = params['outblock']['mlp'][-1]
        self.kanchor = params['outblock']['kanchor']
        self.kpconv_kanchor = params['general']['kpconv_kanchor']
        self.num_slots = params['outblock']['k']

        #### General parameter settings ####
        self.num_iters = params['general']['num_iters']
        self.global_rot = params['general']['global_rot']
        self.npoints = params['general']['npoints']
        self.batch_size = params['general']['batch_size']
        self.init_lr = params['general']['init_lr']
        self.part_pred_npoints = params['general']['part_pred_npoints']
        self.use_equi = params['general']['use_equi']
        self.model_type = params['general']['model_type']
        self.decoder_type = params['general']['decoder_type']
        self.inv_attn = params['general']['inv_attn']
        self.topk = params['general']['topk']
        self.orbit_attn = params['general']['orbit_attn']
        self.slot_iters = params['general']['slot_iters']
        self.rot_factor = params['general']['rot_factor']
        self.translation = params['general']['translation']
        self.gt_oracle_seg = params['general']['gt_oracle_seg']
        self.gt_oracle_trans = params['general']['gt_trans']
        self.feat_pooling = params['general']['feat_pooling']
        self.cent_trans = params['general']['cent_trans']
        self.soft_attn = params['general']['soft_attn']
        self.recon_prior = params['general']['recon_prior']
        self.shape_type = params['general']['shape_type']
        self.factor = params['general']['factor']
        self.queue_len = params['general']['queue_len']
        self.glb_recon_factor = params['general']['glb_recon_factor']
        self.slot_recon_factor = params['general']['slot_recon_factor']
        self.use_sigmoid = params['general']['use_sigmoid']
        self.use_flow_reg = params['general']['use_flow_reg']
        self.use_axis_queue = params['general']['use_axis_queue']
        self.run_mode = params['general']['run_mode']
        self.exp_indicator = params['general']['exp_indicator']
        self.sel_mode = params['general']['sel_mode']
        self.sel_mode_trans = params['general']['sel_mode_trans']
        self.slot_single_mode = params['general']['slot_single_mode']
        self.rot_angle_factor = params['general']['rot_angle_factor']
        self.pred_axis = params['general']['pred_axis']
        self.pred_pv_equiv = params['general']['pred_pv_equiv']
        self.mtx_based_axis_regression = params['general']['mtx_based_axis_regression']
        self.with_part_proposal = params['general']['with_part_proposal']
        self.glb_single_cd = params['general']['glb_single_cd']
        self.slot_single_cd = params['general']['slot_single_cd']

        self.sel_mode_trans = None if self.sel_mode_trans == -1 else self.sel_mode_trans

        self.local_rank = int(os.environ['LOCAL_RANK'])

        #### Set parameter alias ####
        self.recon_part_M = self.part_pred_npoints  # 128, 256, 512, 1024
        self.transformation_dim = 7

        self.stage = params['general']['cur_stage']

        self.log_fn = f"{self.exp_indicator}_{self.shape_type}_out_feats_weq_wrot_{self.global_rot}_rel_rot_factor_{self.rot_factor}_equi_{self.use_equi}_model_{self.model_type}_decoder_{self.decoder_type}_inv_attn_{self.inv_attn}_orbit_attn_{self.orbit_attn}_reconp_{self.recon_prior}_topk_{self.topk}_num_iters_{self.num_iters}_npts_{self.npoints}_perpart_npts_{self.part_pred_npoints}_bsz_{self.batch_size}_init_lr_{self.init_lr}"
        # self.log_fn = os.path.join("/share/xueyi/", self.log_fn) # file name for logging
        # self.log_fn

        ''' Set chamfer distance '''
        self.chamfer_dist = ChamferDistance()

        ''' Get anchors '''
        self.anchors = torch.from_numpy(L.get_anchors(params['outblock']['kanchor'])).cuda()
        # self.kpconv_anchors = torch.from_numpy(L.get_anchors(1)).cuda()
        if self.kpconv_kanchor == 1:
            self.kpconv_anchors = torch.eye(3, dtype=torch.float32).cuda().unsqueeze(0)
        else:
            self.kpconv_anchors = torch.from_numpy(L.get_anchors(self.kpconv_kanchor)).cuda()

        ''' Use slot attention for points grouping '''
        if self.gt_oracle_seg == 0: # not a GT seg...
            ''' Construct slot-attention module '''
            #### ??
            orbit_attn_three_in_dim = self.inv_out_dim + 3
            # orbit_attn_three_in_dim = self.inv_out_dim + 5
            # orbit_attn_three_in_dim = 3
            self.attn_in_dim = (self.inv_out_dim + self.kanchor) if self.orbit_attn == 1 else (
                self.kanchor) if self.orbit_attn == 2 else (orbit_attn_three_in_dim) if self.orbit_attn == 3 else (
                self.inv_out_dim)
            # inv_pooling_method = 'max' if self.recon_prior not in [0, 2] else 'attention'
            inv_pooling_method = 'attention'

            self.inv_pooling_method = inv_pooling_method
            self.sel_mode = None if self.sel_mode == -1 else self.sel_mode
            self.inv_pooling_method = self.inv_pooling_method if self.sel_mode is None else 'sel_mode'
            self.ppint_outblk = Mso3.InvPPOutBlockOurs(params['outblock'], norm=1, pooling_method=inv_pooling_method,  sel_mode=self.sel_mode)
            # whether to use slot attention module
            # slot attention;
            self.slot_attention = SlotAttention(num_slots=params['outblock']['k'],
                                                dim=self.attn_in_dim, hidden_dim=self.inv_out_dim,
                                                iters=self.slot_iters)

        ''' Construct whole shape output block '''
        use_abs_pos = False
        use_abs_pos = True
        # use_abs_pos = False
        # use_abs_pos = False
        ''' Construct inv-feat output block for slots '''
        # modify this process to incorperate masks to the pooling and other calculation process
        # slot_outblock: invariant feature output block for each slot
        self.slot_outblock = nn.ModuleList()
        for i_s in range(self.num_slots):
            return_point_pooling_feature = True
            # whether to return point pooling features
            self.slot_outblock.append(
                Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention', use_pointnet=True, use_abs_pos=use_abs_pos, return_point_pooling_feature=return_point_pooling_feature) # whether to use abs pos
            )



        ''' Construct inv-feat output block for the whole shape '''
        ### the difference is that we should set `mask` to None for each forward pass ###
        ''' Output block for global invariant feature '''
        self.glb_outblock = Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention',
                                                         use_pointnet=True)

        ''' Construct reconstruction branches for slots, input features should be those inv-feats output from inv-feats extraction branches '''
        self.slot_shp_recon_net = nn.ModuleList()
        for i_s in range(self.num_slots):
            if self.recon_prior == 4:
                self.slot_shp_recon_net.append(
                    DecoderFCAtlas([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None, prior_dim=3)
                )
            elif self.recon_prior == 2:
                self.slot_shp_recon_net.append(
                    DecoderConstantCommon([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None)
                )
            elif self.recon_prior == 5:
                # Use cuboic for points reguralization
                self.slot_shp_recon_net.append(
                    DecoderFCWithCuboic([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None,
                                        # pred_rot=True
                                        pred_rot=False
                                        )
                )
            elif self.recon_prior == 6:
                # slot shape reconstruction network
                # pivot point related prediction...
                self.slot_shp_recon_net.append(
                    DecoderFCWithPVP([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None)
                )
            elif self.recon_prior == 7:
                # slot shape reconstruction network
                self.slot_shp_recon_net.append(
                    DecoderFCWithPVPCuboic([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None, pred_rot=False)
                )
            else:
                self.slot_shp_recon_net.append(
                    DecoderFC([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None)
                )


        self.nn_slot_pairs = (self.num_slots - 1) * self.num_slots // 2
        self.slot_pairs_list = []
        for i_s_a in range(self.num_slots - 1):
            for i_s_b in range(i_s_a + 1, self.num_slots):
                self.slot_pairs_list.append((i_s_a, i_s_b))
        assert len(self.slot_pairs_list) == self.nn_slot_pairs

        ''' Construct inv-feat output block for slots '''
        use_abs_pos_pair_out = True
        # use_abs_pos_pair_out = False
        self.zz_pred_pv_equiv = True
        return_point_pooling_feature_pair_out = True if self.zz_pred_pv_equiv else False
        self.pair_slot_outblock = nn.ModuleList()
        for i_s in range(self.nn_slot_pairs):
            self.pair_slot_outblock.append(
                Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention', use_pointnet=True,
                                             use_abs_pos=use_abs_pos_pair_out,
                                             return_point_pooling_feature=return_point_pooling_feature_pair_out)
            )

        self.pred_pair_conf = True
        self.pair_slot_shp_recon_net = nn.ModuleList()
        for i_s in range(self.nn_slot_pairs):
            self.pair_slot_shp_recon_net.append(
                DecoderFCWithPVP([256, 256], params['outblock']['mlp'][-1], 2, None, with_conf=self.pred_pair_conf)
            )

        ''' Construct reconstruction branch for the whole shape '''
        self.glb_recon_npoints = self.npoints
        # self.glb_recon_npoints = 512
        #### global reconstruction net ####
        self.glb_shp_recon_net = DecoderFC(
            [256, 256], params['outblock']['mlp'][-1], self.glb_recon_npoints, None
        )
        #### axis prediction net ####
        # todo: reguralizations for canonical shape axis prediction?
        # axis
        self.glb_axis_pred_net = DecoderFCAxis(
            [256, 256], params['outblock']['mlp'][-1], None
        )

        ''' Construct transformation branches for slots '''
        # todo: how translations are predicted?
        self.slot_trans_outblk_RT = nn.ModuleList()
        self.pair_slot_trans_outblk_RT = nn.ModuleList()
        # self.pred_t = False
        self.pred_t = True
        self.r_representation = 'quat'
        self.r_representation = 'angle'
        for i_s in range(self.num_slots):
            pred_pv_points = True if self.pred_pv_equiv else False
            pred_pv_points_in_dim = params['outblock']['mlp'][-1]
            pred_central_points = True # in this version --> we predict central points and pv points from equiv features
            pred_central_points_in_dim = params['outblock']['mlp'][-1]

            cur_r_representation = self.r_representation if i_s > 0 else 'quat'
            cur_r_representation = self.r_representation if i_s > 0 else 'angle'
            # glboal scalar = True & use_anchors = False ---> t_method_type = 1
            if not self.pred_t:
                self.slot_trans_outblk_RT.append(
                    SO3OutBlockRWithMask(params['outblock'], norm=1, pooling_method='max', pred_t=self.pred_t,
                                         feat_mode_num=self.kanchor, representation=cur_r_representation)
                )
                self.pair_slot_trans_outblk_RT.append(
                    SO3OutBlockRWithMask(params['outblock'], norm=1, pooling_method='max', pred_t=self.pred_t,
                                         feat_mode_num=self.kanchor, representation=cur_r_representation)
                )
            else:
                # cur_slot_trans_outblk = SO3OutBlockRTWithMask if self.shape_type != 'drawer' else SO3OutBlockRTWithAxisWithMask
                cur_slot_trans_outblk = SO3OutBlockRTWithMaskSep
                # c_in_rot = self.inv_out_dim #### If we use inv features for angle decoding
                c_in_rot = self.encoded_feat_dim
                c_in_trans = self.encoded_feat_dim
                # matrix based axis regression
                self.slot_trans_outblk_RT.append(
                    cur_slot_trans_outblk(params['outblock'], norm=1, pooling_method='max',
                                          global_scalar=True,
                                          # global scalar?
                                          use_anchors=False,
                                          feat_mode_num=self.kanchor, num_heads=1, representation=cur_r_representation,
                                          c_in_rot=c_in_rot, c_in_trans=c_in_trans, pred_axis=self.pred_axis,
                                          pred_pv_points=pred_pv_points, pv_points_in_dim=pred_pv_points_in_dim,
                                          pred_central_points=pred_central_points,
                                          central_points_in_dim=pred_central_points_in_dim,
                                          mtx_based_axis_regression=self.mtx_based_axis_regression)
                )
                self.pair_slot_trans_outblk_RT.append(
                    cur_slot_trans_outblk(params['outblock'], norm=1, pooling_method='max',
                                          global_scalar=True, # whether to use
                                          # global scalar?
                                          use_anchors=False,
                                          feat_mode_num=self.kanchor, num_heads=1, representation=cur_r_representation,
                                          c_in_rot=c_in_rot, c_in_trans=c_in_trans, pred_axis=self.pred_axis,
                                          pred_pv_points=pred_pv_points, pv_points_in_dim=pred_pv_points_in_dim,
                                          pred_central_points=pred_central_points,
                                          central_points_in_dim=pred_central_points_in_dim)
                )


        ''' Construct transformation branch for the whole shape '''
        # self.glb_pred_t = False
        ##### Whether to predict a global translation vector #####
        self.glb_pred_t = True
        if not self.glb_pred_t:
            self.glb_trans_outblock_RT = SO3OutBlockRWithMask(params['outblock'], norm=1, pooling_method='max',
                                                              pred_t=self.glb_pred_t, feat_mode_num=self.kanchor)
        else:
            self.glb_trans_outblock_RT = SO3OutBlockRTWithMask(params['outblock'], norm=1, pooling_method='max',
                                                               global_scalar=True,
                                                               use_anchors=False,
                                                               feat_mode_num=self.kanchor, num_heads=1)



    # Utils
    def compute_axis_angle_from_rotation_matrix(self, Rs):
        cos_theta = (Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2] - 1.) / 2.
        cos_theta = torch.clamp(cos_theta, min=-1., max=1.)
        theta = torch.acos(cos_theta)  # from 0 -> pi
        sin_theta = torch.sin(theta)
        kx, ky, kz = Rs[..., 2, 1] - Rs[..., 1, 2], Rs[..., 0, 2] - Rs[..., 2, 0], Rs[..., 1, 0] - Rs[..., 0, 1]
        # print()
        # kx, ky, kz = 0.5 * (1. / sin_theta) * kx, 0.5 * (1. / sin_theta) * ky, 0.5 * (1. / sin_theta) * kz,
        # ... x 3
        computed_axis = torch.cat([kx.unsqueeze(-1), ky.unsqueeze(-1), kz.unsqueeze(-1)], dim=-1)
        # computed_axis: ... x 3; theta -- rad representation
        return computed_axis, theta

    # get slot relative
    def get_slot_rel_Rs_constraint_loss(self, selected_slot_R, slot_weights):
        # selected_slot_R: bz x n_s x 3 x 3
        if self.buf_n == 0:
            cur_bz = selected_slot_R.size(0)

            cur_rel_rot_Rs = []
            for i_s_a in range(self.num_slots - 1):
                cur_a_slot_Rs = selected_slot_R[:, i_s_a, ...]
                for i_s_b in range(i_s_a + 1, self.num_slots):
                    cur_b_slot_Rs = selected_slot_R[:, i_s_b, ...]
                    rel_rot_slot_a_b = torch.matmul(cur_a_slot_Rs, safe_transpose(cur_b_slot_Rs, -1, -2))

                    cur_rel_rot_Rs.append(rel_rot_slot_a_b.unsqueeze(1))
            cur_rel_rot_Rs = torch.cat(cur_rel_rot_Rs, dim=1)

            if self.buf_st + cur_bz >= self.buf_max_n:
                self.buffer_slot_rel_Rs[self.buf_st:] = cur_rel_rot_Rs[: self.buf_max_n - self.buf_st].detach()
                self.buffer_slot_rel_Rs[: cur_bz - self.buf_max_n + self.buf_st] = cur_rel_rot_Rs[
                                                                                   self.buf_max_n - self.buf_st:].detach()
            else:
                self.buffer_slot_rel_Rs[self.buf_st: self.buf_st + cur_bz] = cur_rel_rot_Rs[:].detach()

            self.buf_st = (self.buf_st + cur_bz) % self.buf_max_n
            self.buf_n = min(self.buf_max_n, self.buf_n + cur_bz)
            return None

        cur_rel_rot_Rs = []
        idxx = 0
        dot_axises_loss = 0.
        for i_s_a in range(self.num_slots - 1):
            cur_a_slot_Rs = selected_slot_R[:, i_s_a, ...]
            for i_s_b in range(i_s_a + 1, self.num_slots):

                cur_b_slot_Rs = selected_slot_R[:, i_s_b, ...]
                rel_rot_slot_a_b = torch.matmul(cur_a_slot_Rs, safe_transpose(cur_b_slot_Rs, -1, -2))
                if self.buf_n == self.buf_max_n:
                    buf_slot_rel_rot_a_b = self.buffer_slot_rel_Rs[:, idxx, ...]
                else:
                    # buf_slot_rel_rot_a_b = self.buffer_slot_rel_Rs[self.buf_st: self.buf_st + self.buf_n, idxx, ...]
                    buf_slot_rel_rot_a_b = self.buffer_slot_rel_Rs[: self.buf_st, idxx, ...]
                # buf_slot_rel_rot_a_b: n_buf x 3 x 3
                if len(buf_slot_rel_rot_a_b.size()) == 2:
                    buf_slot_rel_rot_a_b = buf_slot_rel_rot_a_b.unsqueeze(0)
                # rel_rot_slot_a_b: bz x 3 x 3
                # rel_rot_Rs: bz x n_buf x 3 x 3
                rel_rot_Rs = torch.matmul(rel_rot_slot_a_b.unsqueeze(1),
                                          safe_transpose(buf_slot_rel_rot_a_b.unsqueeze(0), -1, -2))
                axis, theta = self.compute_axis_angle_from_rotation_matrix(rel_rot_Rs)
                # axis: bz x n_buf x 3
                axis = axis
                # theta: bz x n_buf x 1
                theta = theta
                # normalize axis
                # get axis
                axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True, p=2), min=1e-8)
                # dot_axises: bz x n_buf x n_buf
                # want to max
                dot_axises = torch.sum(axis.unsqueeze(2) * axis.unsqueeze(1), dim=-1)
                # cross_axies = torch.cross()
                # todo: use eye to mask out self-self-product?
                # dot_axises_loss_cur_slot_pair = -dot_axises.mean(dim=-1).mean(dim=-1).mean()
                dot_axises_loss_cur_slot_pair = -dot_axises.mean(dim=-1).mean(dim=-1) * slot_weights[:,
                                                                                        i_s_a] * slot_weights[:, i_s_b]
                dot_axises_loss += dot_axises_loss_cur_slot_pair.mean()
                # print(f"slot_a: {i_s_a}, slot_b: {i_s_b}, loss: {dot_axises_loss_cur_slot_pair.mean().item()}")
                # todo: theta loss
                cur_rel_rot_Rs.append(rel_rot_slot_a_b.unsqueeze(1))
                idxx += 1
        # ba x idxess x 3 x 3
        cur_rel_rot_Rs = torch.cat(cur_rel_rot_Rs, dim=1)
        cur_bz = cur_rel_rot_Rs.size(0)

        ''' Update buffer '''
        if self.buf_st + cur_bz >= self.buf_max_n:
            self.buffer_slot_rel_Rs[self.buf_st:] = cur_rel_rot_Rs[: self.buf_max_n - self.buf_st].detach()
            self.buffer_slot_rel_Rs[: cur_bz - self.buf_max_n + self.buf_st] = cur_rel_rot_Rs[
                                                                               self.buf_max_n - self.buf_st:].detach()
        else:
            self.buffer_slot_rel_Rs[self.buf_st: self.buf_st + cur_bz] = cur_rel_rot_Rs[:].detach()

        self.buf_st = (self.buf_st + cur_bz) % self.buf_max_n
        self.buf_n = min(self.buf_max_n, self.buf_n + cur_bz)
        return dot_axises_loss * self.num_slots * 2

    def apply_part_reconstruction_net(self, feats):
        recon_pts = []
        bz = feats.size(0)
        dim = feats.size(1)
        # bz x dim x n_s x na
        for i_s, mod in enumerate(self.part_reconstruction_net):
            cur_slot_inv_feats = feats[:, :, i_s, :].view(bz, dim, 1, 1)  # .unsqueeze(-2)
            cur_slot_inv_feats = mod(cur_slot_inv_feats)
            cur_slot_inv_feats = cur_slot_inv_feats.squeeze(-1)
            # recon_slot_points: bz x n_s x M x 3
            cur_slot_inv_feats = cur_slot_inv_feats.contiguous().transpose(1, 2).contiguous().view(bz, 1,
                                                                                                   self.recon_part_M,
                                                                                                   -1)
            recon_pts.append(cur_slot_inv_feats - 0.5)
        recon_pts = torch.cat(recon_pts, dim=1)  # .cuda()
        # print(recon_pts.size())
        return recon_pts

    def apply_part_reconstruction_net_v2(self, feats):
        recon_pts = []
        bz = feats.size(0)
        dim = feats.size(1)
        # bz x dim x n_s x na
        for i_s, mod in enumerate(self.part_reconstruction_net):
            #### Use values sampled from a distribution as the prior ####
            cur_mu = self.mu_params[i_s]
            cur_log_sigma = self.log_sigma_params[i_s]
            cur_sigma = torch.exp(cur_log_sigma)
            cur_dist = torch.distributions.Normal(loc=cur_mu, scale=cur_sigma)
            # sampled_value:  bz x encoded_feat_dim
            sampled_value = cur_dist.sample((bz,))
            cur_slot_inv_feats = sampled_value.view(bz, dim, 1, 1)

            # cur_slot_inv_feats =  feats[:, :, i_s, :].view(bz, dim, 1, 1) # .unsqueeze(-2)
            cur_slot_inv_feats = mod(cur_slot_inv_feats)
            cur_slot_inv_feats = cur_slot_inv_feats.squeeze(-1)
            # recon_slot_points: bz x n_s x M x 3
            cur_slot_inv_feats = cur_slot_inv_feats.contiguous().transpose(1, 2).contiguous().view(bz, 1,
                                                                                                   self.recon_part_M,
                                                                                                   -1)
            recon_pts.append(cur_slot_inv_feats - 0.5)
        recon_pts = torch.cat(recon_pts, dim=1)  # .cuda()
        # print(recon_pts.size())
        return recon_pts

    def apply_part_reconstruction_net_category_common(self, feats):
        recon_pts = []
        bz = feats.size(0)
        dim = feats.size(1)
        # n_s x dim

        ''' Use which kind of category-common prior feature '''
        ## Running mean fashion feature ##
        cur_slot_prior = self.get_slot_prior_rep(feats)

        ## Running queue fashion feature ##
        # cur_slot_prior = self.get_slot_prior_rep_queue(feats)

        ## Constant fashion feature ##
        # cur_slot_prior = torch.ones_like(cur_slot_prior, )
        cur_slot_prior = torch.ones((self.num_slots, self.encoded_feat_dim * 2), dtype=torch.float32).cuda()
        # cur_slot_prior = torch.ones((self.num_slots, self.encoded_feat_dim), dtype=torch.float32).cuda()

        for i_s, mod in enumerate(self.category_part_reconstruction_net):
            #
            #### Use variable part prior as input ####
            # cur_slot_inv_feats = self.category_part_prior_params[i_s].unsqueeze(0).repeat(bz, 1).unsqueeze(-1).unsqueeze(-1)
            cur_slot_inv_feats = cur_slot_prior[i_s].unsqueeze(0).repeat(bz, 1).unsqueeze(-1).unsqueeze(-1)

            # cur_slot_inv_feats = self.category_part_prior_params[i_s].unsqueeze(0).repeat(bz, 1).unsqueeze(-1).unsqueeze(-1)

            # cur_slot_inv_feats =  feats[:, :, i_s, :].view(bz, dim, 1, 1) # .unsqueeze(-2)
            # print(i_s, self.category_part_prior_params[i_s][:10])
            cur_slot_inv_feats = mod(cur_slot_inv_feats)
            cur_slot_inv_feats = cur_slot_inv_feats.squeeze(-1)
            # recon_slot_points: bz x n_s x M x 3
            cur_slot_inv_feats = cur_slot_inv_feats.contiguous().transpose(1, 2).contiguous().view(bz, 1,
                                                                                                   self.recon_part_M,
                                                                                                   -1)
            if self.use_sigmoid == 1:
                # recon_pts.append((cur_slot_inv_feats - 0.5) * 0.7)
                if self.shape_type == 'eyeglasses':
                    if i_s == 0:
                        # rest = torch.tensor([0.1, 1., 0.3], dtype=torch.float32).cuda().view(1, 1, 1, 3)
                        rest = torch.tensor([0.1, 1., 0.1], dtype=torch.float32).cuda().view(1, 1, 1, 3)
                    elif i_s == 1 or i_s == 2:
                        rest = torch.tensor([1., 0.1, 0.1], dtype=torch.float32).cuda().view(1, 1, 1, 3)
                    # else:
                    #     raise ValueError("")
                    else:
                        rest = torch.tensor([1., 0.1, 0.1], dtype=torch.float32).cuda().view(1, 1, 1, 3)
                    cur_slot_inv_feats = (cur_slot_inv_feats - 0.5) * rest
                else:
                    cur_slot_inv_feats = cur_slot_inv_feats
                recon_pts.append(cur_slot_inv_feats)
            else:
                if self.shape_type == 'eyeglasses':
                    # centralize predicted points
                    minn_coor, _ = torch.min(cur_slot_inv_feats, dim=-2, keepdim=True)
                    maxx_coor, _ = torch.max(cur_slot_inv_feats, dim=-2, keepdim=True)
                    cent_off = (maxx_coor + minn_coor) / 2.
                    length_bb = torch.norm(maxx_coor - minn_coor, dim=-1, keepdim=True)
                    #
                    cur_slot_inv_feats = cur_slot_inv_feats - cent_off  # centralize points
                    # the orientation does not change
                    cur_slot_inv_feats = cur_slot_inv_feats / length_bb
                    if i_s == 0:
                        rest = torch.tensor([0.1, 1., 0.1], dtype=torch.float32).cuda().view(1, 1, 1, 3)
                    elif i_s == 1 or i_s == 2:
                        rest = torch.tensor([1., 0.1, 0.1], dtype=torch.float32).cuda().view(1, 1, 1, 3)
                    # else:
                    #     raise ValueError("")
                    else:
                        rest = torch.tensor([1., 0.1, 0.1], dtype=torch.float32).cuda().view(1, 1, 1, 3)
                    cur_slot_inv_feats = cur_slot_inv_feats * rest
                else:
                    cur_slot_inv_feats = cur_slot_inv_feats
                recon_pts.append(cur_slot_inv_feats)

        ## Update running mean ##
        self.update_slot_prior_rep(feats=feats)
        recon_pts = torch.cat(recon_pts, dim=1)  # .cuda()
        # print(recon_pts.size())
        # bz x n_s x M x 3
        return recon_pts

    def apply_part_reconstruction_net_category_common_v2(self, feats):
        recon_pts = []
        bz = feats.size(0)
        dim = feats.size(1)
        # n_s x dim

        ''' Use which kind of category-common prior feature '''
        ## Running mean fashion feature ##
        # cur_slot_prior = self.get_slot_prior_rep(feats)

        ## Running queue fashion feature ##
        # cur_slot_prior = self.get_slot_prior_rep_queue(feats)

        ## Constant fashion feature ##
        # cur_slot_prior = torch.ones_like(cur_slot_prior, )
        cur_slot_prior = torch.ones((self.num_slots, self.encoded_feat_dim * 2), dtype=torch.float32).cuda()

        #
        for i_s, mod in enumerate(self.category_part_reconstruction_net_v2):
            #### Use variable part prior as input ####
            # cur_slot_inv_feats = self.category_part_prior_params[i_s].unsqueeze(0).repeat(bz, 1).unsqueeze(-1).unsqueeze(-1)
            # bz x (dim) x 1 x 1
            cur_slot_inv_feats = cur_slot_prior[i_s].unsqueeze(0).unsqueeze(-1).repeat(bz, 1, self.recon_part_M)
            cur_slot_inv_feats = cur_slot_inv_feats.unsqueeze(-1)
            # ss_pts = torch.from_numpy(self.sphere_pts).float().cuda() / 3.
            if self.recon_prior == 3:
                ss_pts = torch.from_numpy(self.sphere_pts).float().cuda() / 3.  # 3 or 4, which is better?
            elif self.recon_prior == 4:
                ss_pts = safe_transpose(self.grid[i_s], 0, 1)
            else:
                raise ValueError(
                    f"In apply_part_reconstruction_net_category_common_v2 function: unrecognized parameter recon_prior: {self.recon_prior}")
            ss_pts = safe_transpose(ss_pts, 0, 1).unsqueeze(0).unsqueeze(-1).repeat(bz, 1, 1, 1)
            cur_slot_inv_feats = torch.cat([cur_slot_inv_feats, ss_pts], dim=1)
            cur_slot_inv_feats = ss_pts

            # print(i_s, self.category_part_prior_params[i_s][:10])
            cur_slot_inv_feats = mod(cur_slot_inv_feats)
            # bz x 3 x M x 1 ---> bz x
            cur_slot_inv_feats = cur_slot_inv_feats.squeeze(-1)
            # recon_slot_points: bz x n_s x M x 3
            cur_slot_inv_feats = cur_slot_inv_feats.contiguous().transpose(1, 2).contiguous().view(bz, 1,
                                                                                                   self.recon_part_M,
                                                                                                   -1)
            if self.use_sigmoid == 1:
                recon_pts.append(cur_slot_inv_feats - 0.5)
            else:
                recon_pts.append(cur_slot_inv_feats)

        ## Update running mean ##
        # self.update_slot_prior_rep(feats=feats)

        recon_pts = torch.cat(recon_pts, dim=1)  # .cuda()
        # print(recon_pts.size())
        # bz x n_s x M x 3
        return recon_pts

    def apply_part_reconstruction_net_category_common_sphere(self, feats):
        recon_pts = []

        bz = feats.size(0)
        dim = feats.size(1)

        n_s = feats.size(2)
        sphere_pts = torch.from_numpy(self.sphere_pts).float().cuda()
        sphere_pts = sphere_pts.unsqueeze(0).unsqueeze(0).repeat(bz, n_s, 1, 1).contiguous()
        # recon_pts = sphere_pts / 2.
        recon_pts = sphere_pts / 4.

        return recon_pts

    def apply_part_reconstruction_net_category_common_atlas(self, feats):
        recon_pts = []

        bz = feats.size(0)
        dim = feats.size(1)

        n_s = feats.size(2)

        gg = []
        for i_s in range(self.num_slots):
            cur_grid = self.grid[i_s]
            gg.append(cur_grid.unsqueeze(0))
        gg = torch.cat(gg, dim=0)

        # self.grid: n_s x 3 x M
        grid_pts = safe_transpose(gg, 1, 2).unsqueeze(0).repeat(bz, 1, 1, 1).contiguous()
        return grid_pts

    def apply_part_deformation_flow_net_instance(self, category_pts, feats):
        # feats: bz x dim x n_s x 1
        deformed_pts = []
        bz = feats.size(0)
        flow_reg_losses = []
        for i_s, mod in enumerate(self.deformation_flow_predict_per_part_net):
            cur_slot_category_pts = category_pts[:, i_s, :, :]
            cur_slot_category_pts = safe_transpose(cur_slot_category_pts, 1, 2)
            expaned_feats = feats[:, :, i_s, :].repeat(1, 1, self.recon_part_M)
            cat_feats = torch.cat([expaned_feats, cur_slot_category_pts], dim=1)
            # bz x 3 x M x 1
            cur_slot_predicted_flow = mod(cat_feats.unsqueeze(-1)).squeeze(-1)
            # cur_slot_predicted_flow: bz x M x 3
            cur_slot_predicted_flow = cur_slot_predicted_flow.contiguous().transpose(1, 2).contiguous() - 0.5
            if self.use_flow_reg == 0:
                # cur_slot_predicted_flow = cur_slot_predicted_flow * 0.2 # 0.2 for oven
                cur_slot_predicted_flow = cur_slot_predicted_flow * 0.10  # 0.2 for oven

            cur_flow_reg_loss = torch.sum(torch.sum(cur_slot_predicted_flow ** 2, dim=-1), dim=-1)
            flow_reg_losses.append(cur_flow_reg_loss.unsqueeze(-1))
            cur_slot_category_pts = safe_transpose(cur_slot_category_pts, 1, 2)

            if self.use_flow_reg == 2:
                cur_slot_deformed_pts = cur_slot_predicted_flow
            else:
                cur_slot_deformed_pts = cur_slot_category_pts + cur_slot_predicted_flow

            # bz x M x 3
            # cur_slot_deformed_pts = safe_transpose(cur_slot_deformed_pts, 1, 2)
            deformed_pts.append(cur_slot_deformed_pts.unsqueeze(1))
        deformed_pts = torch.cat(deformed_pts, dim=1)
        flow_reg_losses = torch.cat(flow_reg_losses, dim=-1).mean(dim=-1)
        return deformed_pts, flow_reg_losses

    ''' Apply part deformation flow net to transform a shape from a sphere to a concrete shape '''
    def apply_part_deformation_flow_net_instance_sphere(self, category_pts, feats):
        # feats: bz x dim x n_s x 1
        deformed_pts = []
        bz = feats.size(0)
        for i_s, mod in enumerate(self.deformation_flow_predict_per_part_net):
            cur_slot_category_pts = category_pts[:, i_s, :, :]
            cur_slot_category_pts = safe_transpose(cur_slot_category_pts, 1, 2)
            expaned_feats = feats[:, :, i_s, :].repeat(1, 1, self.recon_part_M)
            cat_feats = torch.cat([expaned_feats, cur_slot_category_pts], dim=1)
            # bz x 3 x M x 1
            cur_slot_predicted_flow = mod(cat_feats.unsqueeze(-1)).squeeze(-1)
            # cur_slot_predicted_flow: bz x M x 3
            cur_slot_predicted_flow = cur_slot_predicted_flow.contiguous().transpose(1, 2).contiguous() - 0.5

            ## Use predicted as points ##
            cur_slot_deformed_pts = cur_slot_predicted_flow
            ## Use predicted as points ##

            # bz x M x 3
            # cur_slot_deformed_pts = safe_transpose(cur_slot_deformed_pts, 1, 2)
            deformed_pts.append(cur_slot_deformed_pts.unsqueeze(1))
        deformed_pts = torch.cat(deformed_pts, dim=1)
        return deformed_pts

    # slot prior representation
    def get_slot_prior_rep(self, feats):
        # feats: bz x dim x n_s x 1
        if self.updated == False:
            updated_feats = torch.mean(feats, dim=0).squeeze(-1)
            updated_feats = safe_transpose(updated_feats, 0, 1)
            self.slot_prior_rep = updated_feats.detach()
            # self.slot_prior_rep.requires_grad = False
            self.updated = True
        else:
            updated_feats = torch.mean(feats, dim=0).squeeze(-1)
            updated_feats = safe_transpose(updated_feats, 0, 1)
            indis = torch.sum(self.slot_prior_rep ** 2, dim=-1).unsqueeze(-1).repeat(1, self.encoded_feat_dim)
            indis = (indis < 0.01).long()
            self.slot_prior_rep[indis] = updated_feats.detach()[indis]
        return self.slot_prior_rep.detach()

    def get_slot_prior_rep_queue(self, feats):
        # feats: bz x dim x n_s x 1
        bz = feats.size(0)
        # updated_feats: bz x n_s x dim
        updated_feats = safe_transpose(feats, 1, 2).squeeze(-1)
        cur_st, cur_ed = self.queue_st, self.queue_st + bz
        if cur_ed <= self.queue_len:
            # if bz == 1:
            #     self.slot_prior_rep_queue[cur_st: cur_ed] = updated_feats
            #### prior rep queue ####

            self.slot_prior_rep_queue[cur_st: cur_ed] = updated_feats
        else:
            # print("???")
            self.slot_prior_rep_queue[cur_st:] = updated_feats[: self.queue_len - cur_st]
            self.slot_prior_rep_queue[: cur_ed - self.queue_len] = updated_feats[self.queue_len - cur_st:]
        self.queue_st = (self.queue_st + bz) % self.queue_len
        self.queue_tot_len = min(self.queue_len, self.queue_tot_len + bz)
        # print(f"queue_st: {self.queue_st}, queue_tot_len: {self.queue_tot_len}")
        avg_rep = torch.sum(self.slot_prior_rep_queue[:self.queue_tot_len], dim=0) / self.queue_tot_len
        return avg_rep.detach()

    def update_slot_prior_rep(self, feats, factor=0.99):
        factor = self.factor
        updated_feats = torch.mean(feats, dim=0).squeeze(-1)
        updated_feats = safe_transpose(updated_feats, 0, 1)
        self.slot_prior_rep = factor * self.slot_prior_rep + (1. - factor) * updated_feats.detach()

    def apply_transformation_prediction_net(self, feats):
        recon_pts = []
        bz = feats.size(0)
        dim = feats.size(1)
        # bz x dim x n_s x na
        for i_s, mod in enumerate(self.transformation_prediction):
            cur_slot_inv_feats = feats[:, :, i_s, :].view(bz, dim, 1, -1)  # .unsqueeze(-2)
            cur_slot_inv_feats = mod(cur_slot_inv_feats)
            recon_pts.append(cur_slot_inv_feats)
        recon_pts = torch.cat(recon_pts, dim=-2)  # .cuda()

        return recon_pts

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

    def get_orbit_mask(self, slot_pred_rots, selected_orbit):
        # slot_pred_rots: bz x n_s x na x 3 x 3
        # selected_orbit: bz x --> selected orbit of the first slot
        # selected_rots: bz x n_s x 1 x 3 x 3
        selected_rots = batched_index_select(values=slot_pred_rots[:, 0, ...], indices=selected_orbit.unsqueeze(1),
                                             dim=1)
        selected_rots = selected_rots.squeeze(1)
        # mult_rots: bz x n_s x na x 3 x 3
        mult_rots = torch.matmul(selected_rots.unsqueeze(1).unsqueeze(1), safe_transpose(slot_pred_rots, -1, -2))
        # dx: bz x n_s x na
        dx, dy, dz = mult_rots[..., 2, 1] - mult_rots[..., 1, 2], mult_rots[..., 0, 2] - mult_rots[..., 2, 0], \
                     mult_rots[..., 1, 0] - mult_rots[..., 0, 1]
        # axises: bz x n_s x na x 3
        axises = torch.cat([dx.unsqueeze(-1), dy.unsqueeze(-1), dz.unsqueeze(-1)], dim=-1)
        axises = axises / torch.clamp(torch.norm(axises, dim=-1, keepdim=True, p=2), min=1e-8)
        # dot_product: bz x n_s x na
        # print(f"axis: {axises.size()}, self.axis_prior_slot_pairs: {self.axis_prior_slot_pairs.size()}")
        # axises = axises.squeeze(1)

        # axis_prior_slot_pairs: n_s x 3 --> 1 x n_s x 1 x 3
        dot_product = torch.sum(axises * self.axis_prior_slot_pairs.data.unsqueeze(0).unsqueeze(2), dim=-1)
        orbit_mask = torch.zeros_like(dot_product)
        orbit_mask[dot_product < 0.3] = 1.
        # if self.local_rank == 0:
        # print(f"current axis prior: {self.axis_prior_slot_pairs.data}")
        # print(f"mean of oribt mask: {torch.mean(orbit_mask).item()}")
        return orbit_mask, axises

    # slot pair axis prior
    def update_slot_pair_axis_prior(self, axises, factor=0.9):
        # axises: bz x n_s x 3
        dot_axises_with_prior = torch.sum(axises * self.axis_prior_slot_pairs.unsqueeze(0), dim=-1)
        # axises_prior_consistent_indicator = (dot_axises_with_prior > 0.).float()
        # try this method first...
        axises[dot_axises_with_prior.unsqueeze(-1).repeat(1, 1, 3).contiguous() < 0.] = axises[
                                                                                            dot_axises_with_prior.unsqueeze(
                                                                                                -1).repeat(1, 1,
                                                                                                           3).contiguous() < 0.] * (
                                                                                            -1.0)
        avg_axises = axises.mean(dim=0)
        avg_axises = avg_axises / torch.clamp(torch.norm(avg_axises, dim=-1, keepdim=True, p=2), min=1e-8)
        avg_axises = avg_axises.detach()
        self.axis_prior_slot_pairs.data = self.axis_prior_slot_pairs.data * factor + (1. - factor) * avg_axises
        self.axis_prior_slot_pairs.data = self.axis_prior_slot_pairs.data / torch.clamp(
            torch.norm(self.axis_prior_slot_pairs.data, dim=-1, keepdim=True, p=2), min=1e-8)

    def select_anchors_via_previous_consistency(self, dot_anchor_part_rots_anchors):
        # dot_anchor_part_rots_anchors: bz x n_s x na x na x 3 x 3
        # queue: queue_len x n_s x 3 x 3
        # only can be called when queue_len > 1
        if self.queue_len == self.queue_tot_len:
            cur_queue = self.slot_pair_mult_R_queue.data[:]
        else:
            cur_queue = self.slot_pair_mult_R_queue.data[: self.queue_st]
        # cur_queue: queue_len x n_s x 3 x 3
        # dot_rots_cur_queue: bz x queue_len x n_s x na x na x 3 x 3
        # print(
        #     f"Current queue size: {cur_queue.size()}, queue len: {self.queue_len}, queue total len: {self.queue_tot_len}")
        dot_rots_cur_queue = torch.matmul(dot_anchor_part_rots_anchors.unsqueeze(1),
                                          cur_queue.view(1, self.queue_len, self.num_slots, 1, 1, 3, 3).contiguous())
        dot_ax_x, dot_ax_y, dot_ax_z = dot_rots_cur_queue[..., 2, 1] - dot_rots_cur_queue[..., 1, 2], \
                                       dot_rots_cur_queue[..., 0, 2] - dot_rots_cur_queue[..., 2, 0], \
                                       dot_rots_cur_queue[..., 1, 0] - dot_rots_cur_queue[..., 0, 1]
        # dot_axis: bz x queue_len x n_s x na x na x 3
        dot_axis = torch.cat([dot_ax_x.unsqueeze(-1), dot_ax_y.unsqueeze(-1), dot_ax_z.unsqueeze(-1)], dim=-1)
        dot_axis = dot_axis / torch.clamp(torch.norm(dot_axis, dim=-1, p=2, keepdim=True), min=1e-8)
        # dot_axis_between_queue: bz x q_l x q_l x n_s x na x na

        dot_axis_between_queue = torch.sum(dot_axis.unsqueeze(2) * dot_axis.unsqueeze(1), dim=-1)

        # todo: whether to add abs?
        dot_axis_between_queue = torch.abs(dot_axis_between_queue)

        # dot_axis_between_queue_aggr: bz x n_s x na x na
        # dot_axis_between_queue_aggr = dot_axis_between_queue.sum(1).sum(1)
        dot_axis_between_queue_aggr = dot_axis_between_queue.mean(1).mean(1)

        # select topk anchors that achieve highest dot product values
        # selected_dots: bz x n_s x na x 5
        selected_dots, selected_anchors = torch.topk(dot_axis_between_queue_aggr, k=5, largest=True, dim=-1)
        return selected_dots, selected_anchors

    ''' Update slot pair mult rotation matrix queue '''
    def update_slot_pair_mult_R_queue(self, selected_mult_R):
        # selected_mult_R: bz x n_s x 3 x 3
        bz = selected_mult_R.size(0)
        if self.queue_st + bz > self.queue_tot_len:
            self.slot_pair_mult_R_queue.data[self.queue_st:] = selected_mult_R[
                                                               : self.queue_tot_len - self.queue_st].detach()
            self.slot_pair_mult_R_queue.data[: bz - self.queue_tot_len + self.queue_st] = selected_mult_R[
                                                                                          self.queue_tot_len - self.queue_st:].detach()
        else:
            self.slot_pair_mult_R_queue.data[self.queue_st: self.queue_st + bz] = selected_mult_R[:].detach()
        self.queue_st = (self.queue_st + bz) % self.queue_tot_len
        self.queue_len = min(self.queue_tot_len, self.queue_len + bz)

    def select_slot_orbits_bak(self, slot_recon_loss, slot_pred_rots):
        # slot_recon_loss: bz x n_s x na # the reconstruction loss for each (slot, anchor) pair
        # slot_pred_rots: bz x n_s x na x 3 x 3
        # selected_orbit: bz
        # if len(slot_recon_loss.size())
        # print(slot_recon_loss.size())

        ''' If using queue to involve other shapes '''
        if self.use_axis_queue == 1:
            if self.queue_len < 2:
                dist_chamfer_recon_slot_ori, selected_slot_oribt = torch.min(slot_recon_loss, dim=2)
                selected_slot_pred_rots = batched_index_select(values=slot_pred_rots,
                                                               indices=selected_slot_oribt.unsqueeze(-1), dim=2)
                # selected_slot_pred_rots: bz x n_s x 3 x 3
                selected_slot_pred_rots = selected_slot_pred_rots.squeeze(2)
                # anchor_part_others_rel_rots: bz x n_s x 3 x 3
                anchor_part_others_rel_rots = torch.matmul(
                    selected_slot_pred_rots[:, 0, :, :].unsqueeze(1).contiguous().transpose(-1, -2).contiguous(),
                    selected_slot_pred_rots)
                self.update_slot_pair_mult_R_queue(anchor_part_others_rel_rots)
                return dist_chamfer_recon_slot_ori, selected_slot_oribt

        bz = slot_recon_loss.size(0)
        # if os.path.exists("axis_prior_0.npy"):
        #     self.axis_prior_slot_pairs.data = torch.from_numpy(np.load("axis_prior_0.npy", allow_pickle=True)).cuda()
        # anchor_part_rots: bz x na x 3 x 3; self.anchors: na x 3 x 3 -> 1 x 1 x na x 3 x 3
        anchor_part_rots = slot_pred_rots[:, 0, :, :]
        # dot_anchor_part_rots_anchors: bz x n_s x na x na x 3 x 3

        dot_anchor_part_rots_anchors = torch.matmul(anchor_part_rots.unsqueeze(1).unsqueeze(3),
                                                    self.anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0).transpose(-1,
                                                                                                                  -2).contiguous())
        # R_1^T R_2
        # bz x n_s x na x
        # dot_anchor_part_rots_anchors = torch.matmul(anchor_part_rots.unsqueeze(1).unsqueeze(3).transpose(-1, -2).contiguous(), self.anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0))

        # dot_anchor_part_rots_anchors = torch.matmul(anchor_part_rots.unsqueeze(1).unsqueeze(3).transpose(-1, -2).contiguous(), slot_pred_rots.unsqueeze(2))

        ax_x, ax_y, ax_z = dot_anchor_part_rots_anchors[..., 2, 1] - dot_anchor_part_rots_anchors[..., 1, 2], \
                           dot_anchor_part_rots_anchors[..., 0, 2] - dot_anchor_part_rots_anchors[..., 2, 0], \
                           dot_anchor_part_rots_anchors[..., 1, 0] - dot_anchor_part_rots_anchors[..., 0, 1]
        # anchor_part_anchors_axises: bz x n_s x na x na x 3
        anchor_part_anchors_axises = torch.cat([ax_x.unsqueeze(-1), ax_y.unsqueeze(-1), ax_z.unsqueeze(-1)], dim=-1)
        # anchor_part_anchors_axises: get the direction of the axis

        anchor_part_anchors_axises = anchor_part_anchors_axises / torch.clamp(
            torch.norm(anchor_part_anchors_axises, dim=-1, keepdim=True, p=2), min=1e-8)

        # dot_anchor_part_anchors_axises_prior: bz x n_s x na x na
        dot_anchor_part_anchors_axises_prior = torch.sum(
            anchor_part_anchors_axises * self.axis_prior_slot_pairs.view(1, self.num_slots, 1, 1, 3).contiguous(),
            dim=-1)

        # not using axis will be better
        ''' Whether to use abs dot product value? '''
        # dot_anchor_part_anchors_axises_prior = torch.abs(dot_anchor_part_anchors_axises_prior)

        # selected_anchors: bz x n_s x na x 5
        ''' Select topk anchors for each combination by their consistency with the maintained axises '''

        if self.use_axis_queue == 1:
            ''' Select via axis consistency --- multi shape selection '''
            # selected_dots: bz x n_s x na
            selected_dots, selected_anchors = self.select_anchors_via_previous_consistency(
                dot_anchor_part_rots_anchors=dot_anchor_part_rots_anchors)
        else:
            ''' Single shape selection '''
            # use queue for selection
            selected_dots, selected_anchors = torch.topk(dot_anchor_part_anchors_axises_prior, k=5, largest=True,
                                                         dim=-1)

        selected_mean_dots = torch.mean(torch.mean(selected_dots, dim=-1), dim=-1)
        # selected_dot_anchor_part_rots_anchors: bz x n_s x na x 5 x 3 x 3
        # print(dot_anchor_part_rots_anchors.size())
        if dot_anchor_part_rots_anchors.size(1) == 1:
            dot_anchor_part_rots_anchors = dot_anchor_part_rots_anchors.repeat(1, self.num_slots, 1, 1, 1, 1)
        selected_dot_anchor_part_rots_anchors = batched_index_select(values=dot_anchor_part_rots_anchors,
                                                                     indices=selected_anchors.long(), dim=3)

        # select angles
        selected_dot_anchor_part_rots_anchors_angles = selected_dot_anchor_part_rots_anchors[..., 0, 0] + \
                                                       selected_dot_anchor_part_rots_anchors[..., 1, 1] + \
                                                       selected_dot_anchor_part_rots_anchors[..., 2, 2]
        # selected_dot_anchor_part_rots_anchors_angles: bz x n_s x na x 3
        selected_dot_anchor_part_rots_anchors_angles = (selected_dot_anchor_part_rots_anchors_angles - 1.) / 2.
        selected_angles, selected_anchors_angles = torch.topk(selected_dot_anchor_part_rots_anchors_angles, k=3,
                                                              largest=True, dim=-1)
        # selected_anchors: bz x n_s x na x 3
        # print(f'selected_anchors: {selected_anchors.size()}, selected_anchors_angles: {selected_anchors_angles.size()}')
        selected_anchors = batched_index_select(values=selected_anchors, indices=selected_anchors_angles.long(), dim=3)
        # selected_dots: bz x n_s x na x 3
        selected_dots = batched_index_select(values=selected_dots, indices=selected_anchors_angles.long(), dim=3)
        #

        res_selected_anchors = []
        for i_bz in range(bz):
            tot_orbit_cmb = []
            tot_minn_loss = 1e8
            # cur_bz_mean_dot = selected_mean_dots[i_bz].item()
            for i_a in range(self.kanchor):
                cur_bz_anchor_part_recon_loss = slot_recon_loss[i_bz, 0, i_a].item()
                nn_permute = 3 ** (self.num_slots - 1)  #
                minn_recon_other_parts = 1e8
                # minn_recon_i_perm = 0
                minn_orbit_cmb = []
                for i_perm in range(nn_permute):
                    zz_i_perm = i_perm + 0
                    recon_loss_other_parts = 0.
                    orbit_cmb = []
                    for i_other_part in range(self.num_slots - 1):
                        curr_orbit_idx = zz_i_perm % (self.num_slots - 1)
                        real_orbit_idx = int(selected_anchors[i_bz, i_other_part + 1, i_a, curr_orbit_idx].item())
                        recon_loss_other_parts += slot_recon_loss[i_bz, 1 + i_other_part, real_orbit_idx].item()
                        zz_i_perm = zz_i_perm // (self.num_slots - 1)
                        orbit_cmb.append(real_orbit_idx)
                    if minn_recon_other_parts > recon_loss_other_parts:
                        minn_recon_other_parts = recon_loss_other_parts
                        # minn_recon_i_perm = zz_i_perm # zz perm
                        minn_orbit_cmb = orbit_cmb
                if tot_minn_loss > minn_recon_other_parts + cur_bz_anchor_part_recon_loss:
                    tot_minn_loss = minn_recon_other_parts + cur_bz_anchor_part_recon_loss
                    tot_orbit_cmb = [i_a] + minn_orbit_cmb
            res_selected_anchors.append(tot_orbit_cmb)

        # res_selected_anchors: bz x n_s
        res_selected_anchors = torch.tensor(res_selected_anchors, dtype=torch.long).cuda()
        # res_selected_loss: bz x n_s # select anchors
        res_selected_loss = batched_index_select(values=slot_recon_loss, indices=res_selected_anchors.unsqueeze(-1),
                                                 dim=2).squeeze(2)
        anchor_part_selected_anchor = res_selected_anchors[:, 0]
        # anchor_part_anchors_axises: bz x n_s x na x 3
        anchor_part_anchors_axises = batched_index_select(safe_transpose(anchor_part_anchors_axises, 1, 2),
                                                          indices=anchor_part_selected_anchor.unsqueeze(-1),
                                                          dim=1).squeeze(1)
        # anchor_part_anchors_dot_rotations: bz x n_s x na x 3 x 3
        anchor_part_anchors_dot_rotations = batched_index_select(safe_transpose(dot_anchor_part_rots_anchors, 1, 2),
                                                                 indices=anchor_part_selected_anchor.unsqueeze(-1),
                                                                 dim=1).squeeze(1)
        # bz x n_s x na x 3 --> bz x na x n_s x 3 --> bz x n_s x 3
        # dot product between the first slot and remaining slots
        selected_dots = batched_index_select(values=safe_transpose(selected_dots, 1, 2),
                                             indices=anchor_part_selected_anchor.unsqueeze(-1), dim=1).squeeze(1)
        avg_selected_dots = selected_dots.mean(dim=-1)

        self.avg_selected_dots = avg_selected_dots
        # print(f"avg_selected_dots: {avg_selected_dots}")

        # anchor_part_anchors_axises: bz x n_s x 3
        # print(f"anchor_part_anchors_axises: {anchor_part_anchors_axises.size()}, res_selected_anchors: {res_selected_anchors.size()}")
        if anchor_part_anchors_axises.size(1) == 1:
            anchor_part_anchors_axises = anchor_part_anchors_axises.repeat(1, self.num_slots, 1, 1)
            anchor_part_anchors_dot_rotations = anchor_part_anchors_dot_rotations.repeat(1, self.num_slots, 1, 1, 1)
        anchor_part_anchors_axises = batched_index_select(values=anchor_part_anchors_axises,
                                                          indices=res_selected_anchors.unsqueeze(-1), dim=2).squeeze(2)
        # anchor_part_anchors_dot_rotations: bz x n_s x 3 x 3
        anchor_part_anchors_dot_rotations = batched_index_select(values=anchor_part_anchors_dot_rotations,
                                                                 indices=res_selected_anchors.unsqueeze(-1),
                                                                 dim=2).squeeze(2)

        if self.run_mode == "train":
            ''' Update queue or axis prior and save '''
            if self.use_axis_queue == 1:
                ''' Save slot prior mult R queue --- for multi shape selection '''
                self.update_slot_pair_mult_R_queue(anchor_part_anchors_dot_rotations)
                np.save(f"slot_pair_mult_R_queue_{self.local_rank}.npy",
                        self.slot_pair_mult_R_queue.data.detach().cpu().numpy())
                ''' Save slot prior mult R queue --- for multi shape selection '''
                pass
            else:
                ''' Save slot pair axis prior --- for single shape selection '''
                self.update_slot_pair_axis_prior(anchor_part_anchors_axises)
                # print(self.axis_prior_slot_pairs.data)
                # # if self.local_rank == 0:
                np.save(f"axis_prior_{self.local_rank}.npy", self.axis_prior_slot_pairs.data.detach().cpu().numpy())
                pass
                ''' Save slot pair axis prior --- for single shape selection '''

        return res_selected_loss, res_selected_anchors

    ''' Select orbits via soem other strategies '''
    def select_slot_orbits(self, slot_recon_loss, slot_pred_rots):

        ''' If using queue to involve other shapes '''
        if self.use_axis_queue == 1:
            if self.queue_len < 2:
                dist_chamfer_recon_slot_ori, selected_slot_oribt = torch.min(slot_recon_loss, dim=2)
                selected_slot_pred_rots = batched_index_select(values=slot_pred_rots,
                                                               indices=selected_slot_oribt.unsqueeze(-1), dim=2)
                # selected_slot_pred_rots: bz x n_s x 3 x 3
                selected_slot_pred_rots = selected_slot_pred_rots.squeeze(2)
                # anchor_part_others_rel_rots: bz x n_s x 3 x 3
                anchor_part_others_rel_rots = torch.matmul(
                    selected_slot_pred_rots[:, 0, :, :].unsqueeze(1).contiguous().transpose(-1, -2).contiguous(),
                    selected_slot_pred_rots)
                self.update_slot_pair_mult_R_queue(anchor_part_others_rel_rots)
                return dist_chamfer_recon_slot_ori, selected_slot_oribt

        bz = slot_recon_loss.size(0)
        # if os.path.exists("axis_prior_0.npy"):
        #     self.axis_prior_slot_pairs.data = torch.from_numpy(np.load("axis_prior_0.npy", allow_pickle=True)).cuda()
        # anchor_part_rots: bz x na x 3 x 3; self.anchors: na x 3 x 3 -> 1 x 1 x na x 3 x 3
        anchor_part_rots = slot_pred_rots[:, 0, :, :]
        # dot_anchor_part_rots_anchors: bz x n_s x na x na x 3 x 3

        # dot_anchor_part_rots_anchors = torch.matmul(anchor_part_rots.unsqueeze(1).unsqueeze(3), self.anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0).transpose(-1, -2).contiguous())
        # R_1^T R_2
        # bz x n_s x na x
        # dot_anchor_part_rots_anchors = torch.matmul(anchor_part_rots.unsqueeze(1).unsqueeze(3).transpose(-1, -2).contiguous(), self.anchors.unsqueeze(0).unsqueeze(0).unsqueeze(0))

        dot_anchor_part_rots_anchors = torch.matmul(
            anchor_part_rots.unsqueeze(1).unsqueeze(3).transpose(-1, -2).contiguous(), slot_pred_rots.unsqueeze(2))

        ax_x, ax_y, ax_z = dot_anchor_part_rots_anchors[..., 2, 1] - dot_anchor_part_rots_anchors[..., 1, 2], \
                           dot_anchor_part_rots_anchors[..., 0, 2] - dot_anchor_part_rots_anchors[..., 2, 0], \
                           dot_anchor_part_rots_anchors[..., 1, 0] - dot_anchor_part_rots_anchors[..., 0, 1]
        # anchor_part_anchors_axises: bz x n_s x na x na x 3
        anchor_part_anchors_axises = torch.cat([ax_x.unsqueeze(-1), ax_y.unsqueeze(-1), ax_z.unsqueeze(-1)], dim=-1)
        # dot_anchor_part_anchors_axises_prior: bz x n_s x na x na
        dot_anchor_part_anchors_axises_prior = torch.sum(
            anchor_part_anchors_axises * self.axis_prior_slot_pairs.view(1, self.num_slots, 1, 1, 3).contiguous(),
            dim=-1)

        # not using axis will be better
        ''' Whether to use abs dot product value? '''
        # dot_anchor_part_anchors_axises_prior = torch.abs(dot_anchor_part_anchors_axises_prior)

        # selected_anchors: bz x n_s x na x 5
        ''' Select topk anchors for each combination by their consistency with the maintained axises '''

        if self.use_axis_queue == 1:
            ''' Select via axis consistency --- multi shape selection '''
            # selected_dots: bz x n_s x na
            selected_dots, selected_anchors = self.select_anchors_via_previous_consistency(
                dot_anchor_part_rots_anchors=dot_anchor_part_rots_anchors)
        else:
            ''' Single shape selection '''
            # use queue for selection
            selected_dots, selected_anchors = torch.topk(dot_anchor_part_anchors_axises_prior, k=5, largest=True,
                                                         dim=-1)

        # selected_dot_anchor_part_rots_anchors: bz x n_s x na x 5 x 3 x 3
        # print(dot_anchor_part_rots_anchors.size())
        if dot_anchor_part_rots_anchors.size(1) == 1:
            dot_anchor_part_rots_anchors = dot_anchor_part_rots_anchors.repeat(1, self.num_slots, 1, 1, 1, 1)
        selected_dot_anchor_part_rots_anchors = batched_index_select(values=dot_anchor_part_rots_anchors,
                                                                     indices=selected_anchors.long(), dim=3)

        # select angles
        selected_dot_anchor_part_rots_anchors_angles = selected_dot_anchor_part_rots_anchors[..., 0, 0] + \
                                                       selected_dot_anchor_part_rots_anchors[..., 1, 1] + \
                                                       selected_dot_anchor_part_rots_anchors[..., 2, 2]
        # selected_dot_anchor_part_rots_anchors_angles: bz x n_s x na x 3
        selected_dot_anchor_part_rots_anchors_angles = (selected_dot_anchor_part_rots_anchors_angles - 1.) / 2.
        selected_angles, selected_anchors_angles = torch.topk(selected_dot_anchor_part_rots_anchors_angles, k=3,
                                                              largest=True, dim=-1)
        # selected_anchors: bz x n_s x na x 3
        # print(f'selected_anchors: {selected_anchors.size()}, selected_anchors_angles: {selected_anchors_angles.size()}')
        selected_anchors = batched_index_select(values=selected_anchors, indices=selected_anchors_angles.long(), dim=3)
        # selected_dots: bz x n_s x na x 3
        selected_dots = batched_index_select(values=selected_dots, indices=selected_anchors_angles.long(), dim=3)
        #

        res_selected_anchors = []
        for i_bz in range(bz):
            tot_orbit_cmb = []
            # tot_minn_loss = 1e8
            tot_maxx_cosine_sum = -9999.0
            for i_a in range(self.kanchor):
                cur_bz_anchor_part_recon_loss = slot_recon_loss[i_bz, 0, i_a].item()
                nn_permute = 3 ** (self.num_slots - 1)  #

                maxx_cosine_sum = -9999.0

                # minn_recon_other_parts = 1e8

                # minn_recon_i_perm = 0
                minn_orbit_cmb = []
                for i_perm in range(nn_permute):
                    zz_i_perm = i_perm + 0
                    recon_loss_other_parts = 0.
                    cosine_sum_parts = 0.
                    orbit_cmb = []
                    for i_other_part in range(self.num_slots - 1):
                        curr_orbit_idx = zz_i_perm % (self.num_slots - 1)
                        real_orbit_idx = int(selected_anchors[i_bz, i_other_part + 1, i_a, curr_orbit_idx].item())
                        # recon_loss_other_parts += slot_recon_loss[i_bz, 1 + i_other_part, real_orbit_idx].item()

                        cosine_sum_parts += dot_anchor_part_anchors_axises_prior[
                                                i_bz, 1 + i_other_part, i_a, real_orbit_idx].item() * self.slot_weights[
                                                i_bz, 1 + i_other_part].item()
                        zz_i_perm = zz_i_perm // (self.num_slots - 1)
                        orbit_cmb.append(real_orbit_idx)
                    if maxx_cosine_sum < cosine_sum_parts:
                        maxx_cosine_sum = cosine_sum_parts
                        # minn_recon_i_perm = zz_i_perm
                        minn_orbit_cmb = orbit_cmb
                if tot_maxx_cosine_sum < maxx_cosine_sum:
                    tot_maxx_cosine_sum = maxx_cosine_sum
                    tot_orbit_cmb = [i_a] + minn_orbit_cmb
            res_selected_anchors.append(tot_orbit_cmb)

        # res_selected_anchors: bz x n_s
        res_selected_anchors = torch.tensor(res_selected_anchors, dtype=torch.long).cuda()
        # res_selected_loss: bz x n_s # select anchors
        res_selected_loss = batched_index_select(values=slot_recon_loss, indices=res_selected_anchors.unsqueeze(-1),
                                                 dim=2).squeeze(2)
        anchor_part_selected_anchor = res_selected_anchors[:, 0]
        # anchor_part_anchors_axises: bz x n_s x na x 3
        anchor_part_anchors_axises = batched_index_select(safe_transpose(anchor_part_anchors_axises, 1, 2),
                                                          indices=anchor_part_selected_anchor.unsqueeze(-1),
                                                          dim=1).squeeze(1)
        # anchor_part_anchors_dot_rotations: bz x n_s x na x 3 x 3
        anchor_part_anchors_dot_rotations = batched_index_select(safe_transpose(dot_anchor_part_rots_anchors, 1, 2),
                                                                 indices=anchor_part_selected_anchor.unsqueeze(-1),
                                                                 dim=1).squeeze(1)
        # bz x n_s x na x 3 --> bz x na x n_s x 3 --> bz x n_s x 3
        # dot product between the first slot and remaining slots
        selected_dots = batched_index_select(values=safe_transpose(selected_dots, 1, 2),
                                             indices=anchor_part_selected_anchor.unsqueeze(-1), dim=1).squeeze(1)
        avg_selected_dots = selected_dots.mean(dim=-1)

        self.avg_selected_dots = avg_selected_dots
        # print(f"avg_selected_dots: {avg_selected_dots}")

        # anchor_part_anchors_axises: bz x n_s x 3
        # print(f"anchor_part_anchors_axises: {anchor_part_anchors_axises.size()}, res_selected_anchors: {res_selected_anchors.size()}")
        if anchor_part_anchors_axises.size(1) == 1:
            anchor_part_anchors_axises = anchor_part_anchors_axises.repeat(1, self.num_slots, 1, 1)
            anchor_part_anchors_dot_rotations = anchor_part_anchors_dot_rotations.repeat(1, self.num_slots, 1, 1, 1)
        anchor_part_anchors_axises = batched_index_select(values=anchor_part_anchors_axises,
                                                          indices=res_selected_anchors.unsqueeze(-1), dim=2).squeeze(2)
        # anchor_part_anchors_dot_rotations: bz x n_s x 3 x 3
        anchor_part_anchors_dot_rotations = batched_index_select(values=anchor_part_anchors_dot_rotations,
                                                                 indices=res_selected_anchors.unsqueeze(-1),
                                                                 dim=2).squeeze(2)

        if self.run_mode == "train":
            if self.use_axis_queue == 1:
                ''' Save slot prior mult R queue --- for multi shape selection '''
                self.update_slot_pair_mult_R_queue(anchor_part_anchors_dot_rotations)
                np.save(f"slot_pair_mult_R_queue_{self.local_rank}.npy",
                        self.slot_pair_mult_R_queue.data.detach().cpu().numpy())
                ''' Save slot prior mult R queue --- for multi shape selection '''
            else:
                ''' Save slot pair axis prior --- for single shape selection '''
                self.update_slot_pair_axis_prior(anchor_part_anchors_axises)
                # print(self.axis_prior_slot_pairs.data)
                # if self.local_rank == 0:
                np.save(f"axis_prior_{self.local_rank}.npy", self.axis_prior_slot_pairs.data.detach().cpu().numpy())
                ''' Save slot pair axis prior --- for single shape selection '''

        return res_selected_loss, res_selected_anchors

    ''' Calcuate axises for rotation matrices '''
    def from_rotation_mtx_to_axis(self, rots):
        ori_rots_size = rots.size()[:-1] # number of rotation matrices
        exp_rots = rots.contiguous().view(-1, 3, 3).contiguous()
        exp_axises = []

        for i in range(exp_rots.size(0)):
            cur_rot = exp_rots[i]
            trace = cur_rot[0, 0] + cur_rot[1, 1] + cur_rot[2, 2] # current rotation
            trace = (trace - 1.) / 2.
            trace = float(trace.item())
            trace = min(1.0, max(trace, -1.0))
            if abs(trace - 1.0) < 1e-8:
                if i == 29:
                    axes = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float).cuda()
                else:
                    axes = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float).cuda()
            elif abs(trace + 1.0) < 1e-8:
                ax_x = torch.sqrt((cur_rot[0, 0] + 1.) / 2.)
                if float(ax_x.item()) > 1e-8:
                    ax_y = cur_rot[0, 1] + cur_rot[1, 0]
                    ax_y = ax_y / (4. * ax_x)
                    ax_z = cur_rot[0, 2] + cur_rot[2, 0]
                    ax_z = ax_z / (4. * ax_x)
                    axes = torch.tensor([ax_x, ax_y, ax_z], dtype=torch.float).cuda()
                else:
                    ax_y = torch.sqrt((cur_rot[1, 1] + 1) / 2.0)
                    if float(ax_y.item()) > 1e-8:
                        ax_z = cur_rot[1, 2] + cur_rot[2, 1]
                        ax_z = ax_z / (4. * ax_y)
                        axes = torch.tensor([0, ax_y, ax_z], dtype=torch.float).cuda()
                    else:
                        ax_z = torch.sqrt((cur_rot[2, 2] + 1) / 2.0)
                        axes = torch.tensor([0, 0, ax_z], dtype=torch.float).cuda()
            else:
                angle = math.acos(trace)  # acos
                # print(angle / np.pi * 180.)
                sine = math.sin(angle)  # angle
                ax_x, ax_y, ax_z = cur_rot[2, 1] - cur_rot[1, 2], cur_rot[0, 2] - cur_rot[2, 0], cur_rot[1, 0] - \
                                   cur_rot[0, 1]
                axes = torch.tensor([ax_x, ax_y, ax_z], dtype=torch.float).cuda()
                axes = axes / (2.0 * sine)
            exp_axises.append(axes.unsqueeze(0))
        # exp_dim x 3
        exp_axises = torch.cat(exp_axises, dim=0)
        # exp_dim x 3; axises
        exp_axises = exp_axises.contiguous().view(*ori_rots_size).contiguous()
        return exp_axises

    def forward_one_iter(self, x, pose, ori_pc=None, rlabel=None, cur_iter=0, gt_pose=None, gt_pose_segs=None,
                         canon_pc=None, selected_pts_orbit=None, normals=None, canon_normals=None):  # rotation label
        '''
            gt_pose_segs: bz x n_parts x 4 x 4
            # Get x, pose
        '''
        torch.cuda.empty_cache()
        if self.stage == 0:
            #### Get original points ####
            ori_pts = x.clone()
            bz, npoints = x.size(0), x.size(2)

            # should transpose the input if using Mso3.preprocess_input...
            # cur_kanchor = 60 if cur_iter == 0 else 1
            # cur_kanchor = 60
            cur_kanchor = self.kpconv_kanchor
            x = M.preprocess_input(x, cur_kanchor, pose, False)
            cur_backbone = self.glb_backbone if cur_iter == 0 else self.backbone
            # cur_backbone = self.glb_backbone # if cur_iter == 0 else self.backbone
            # if we use a different backbone?
            for block_i, block in enumerate(cur_backbone):
                x = block(x)

            torch.cuda.empty_cache()

            # cur_anchors = self.anchors if cur_iter == 0 else self.kpconv_anchors
            cur_anchors = self.anchors # if cur_iter == 0 else self.kpconv_anchors

            # if cur_iter == 0:
            # would use global reconstruction for each iteration
            # glb_inv_feats: bz x dim; glb_orbit..: bz x na
            glb_inv_feats, glb_orbit_confidence = self.glb_outblock(x, mask=None)
            glb_output_RT = self.glb_trans_outblock_RT(x, mask=None, anchors=cur_anchors.unsqueeze(0).repeat(bz, 1, 1, 1).contiguous())
            # glb_output_R: bz x 4 x na --> output RT
            glb_R = glb_output_RT['R']
            # glb_output_T: bz x 3 x na
            glb_T = glb_output_RT['T']

            # print(torch.mean(glb_T, dim=-1))

            if glb_T is None:
                glb_T = torch.zeros((bz, 3, cur_kanchor), dtype=torch.float32).cuda()

            glb_T = None

            # glb_recon_canon_pts: bz x 3 x npoints;
            glb_recon_canon_pts = self.glb_shp_recon_net(glb_inv_feats)
            glb_recon_canon_pts = glb_recon_canon_pts - 0.5

            # get global predicted axis direction
            # glb_pred_axis: 3 --> predicted unit vector for the axis in the canonical frame
            # glb_pred_axis = self.glb_axis_pred_net(glb_inv_feats)

            ''' Get glb_R and glb_T '''
            glb_R = compute_rotation_matrix_from_quaternion(
                safe_transpose(glb_R, -1, -2).view(bz * cur_kanchor, -1)).contiguous().view(bz, cur_kanchor, 3, 3)
            # todo: when to add quat constraints on R and how such constraints would influence following process and further the results?
            # slot_R: bz x n_s x na x 3 x 3; slot_T: bz x n_s x na x 3
            glb_R = torch.matmul(cur_anchors.unsqueeze(0), glb_R)

            if isinstance(self.glb_trans_outblock_RT, SO3OutBlockRTWithMask) and glb_T is not None:
                # T has been matmuled with anchors already
                glb_T = safe_transpose(glb_T, -1, -2)
                # print("here1")
            else:
                if glb_T is None:
                    glb_T = torch.zeros((bz, 3, cur_kanchor), dtype=torch.float32).cuda()
                glb_T = torch.matmul(glb_R, safe_transpose(glb_T, -1, -2).unsqueeze(-1)).squeeze(-1)
                # avg_offset: bz x 3
                avg_offset = torch.mean(x.xyz, dim=-1)
                # add xyzs as offset
                glb_T = glb_T + avg_offset.unsqueeze(-2)
                # print("here2")
            ''' Get glb_R and glb_T '''

            # bz x na x npoints x 3
            transformed_glb_recon_pts = safe_transpose(torch.matmul(glb_R, glb_recon_canon_pts.unsqueeze(1)), -1,
                                                       -2) + glb_T.unsqueeze(-2)
            expanded_ori_pts = safe_transpose(ori_pts, -1, -2).unsqueeze(1).contiguous().repeat(1, cur_kanchor, 1, 1)
            # chamfer_recon_to_ori: (bz x na) x npoints; chamfer_ori_to_recon: bz x npoints
            chamfer_recon_to_ori, chamfer_ori_to_recon = safe_chamfer_dist_call(
                transformed_glb_recon_pts.contiguous().view(bz * cur_kanchor, transformed_glb_recon_pts.size(-2), 3),
                expanded_ori_pts.contiguous().view(bz * cur_kanchor, npoints, 3), self.chamfer_dist)
            # chamfer_recon_to_ori: bz x na
            chamfer_recon_to_ori = chamfer_recon_to_ori.contiguous().view(bz, cur_kanchor, -1).mean(dim=-1)
            # chamfer_ori_to_recon: bz x na
            chamfer_ori_to_recon = chamfer_ori_to_recon.contiguous().view(bz, cur_kanchor, -1).mean(dim=-1)
            glb_chamfer = chamfer_recon_to_ori + chamfer_ori_to_recon

            # glb_orbit: bz; minn_glb_chamfer: bz
            minn_glb_chamfer, glb_orbit = torch.min(glb_chamfer, dim=-1)
            # print(minn_glb_chamfer)
            # self.glb_recon_ori_dist = (torch.sqrt(minn_glb_chamfer)).mean() * 0.5

            minn_chamfer_recon_to_ori = batched_index_select(chamfer_recon_to_ori, indices=glb_orbit.unsqueeze(-1), dim=1).squeeze(-1)
            minn_chamfer_ori_to_recon = batched_index_select(chamfer_ori_to_recon, indices=glb_orbit.unsqueeze(-1), dim=1).squeeze(-1)
            self.glb_recon_ori_dist = (torch.sqrt(minn_chamfer_recon_to_ori) + torch.sqrt(minn_chamfer_ori_to_recon)).mean() * 0.5
            # print(self.glb_recon_ori_dist)
            # minn_glb_chamfer for optimization. & glb_orbit for global orbit selection/pose transformed?
            # selected global chamfer distance and global orbit...

            self.glb_ori_to_recon_dist = torch.sqrt(minn_chamfer_ori_to_recon).mean()  # minn chamfer ori to recon...
            # print(f"glb_ori_to_recon, L1: {float(self.glb_ori_to_recon_dist.item())}")

            # should minimize the selected global reconstruction chamfer distance
            # selected_glb_R: bz x 3 x 3
            selected_glb_R = batched_index_select(glb_R, glb_orbit.unsqueeze(-1).long(), dim=1).squeeze(1)
            selected_glb_T = batched_index_select(glb_T, glb_orbit.unsqueeze(-1).long(), dim=1).squeeze(1)

            selected_transformed_glb_recon_pts = batched_index_select(transformed_glb_recon_pts,
                                                                      indices=glb_orbit.unsqueeze(-1).long(),
                                                                      dim=1).squeeze(1)
            inv_trans_ori_pts = torch.matmul(safe_transpose(selected_glb_R, -1, -2), ori_pts - selected_glb_T.unsqueeze(-1))
            inv_trans_ori_pts = safe_transpose(inv_trans_ori_pts, -1, -2)

            self.inv_trans_ori_pts = safe_transpose(inv_trans_ori_pts, -1, -2).detach()
            self.glb_R = selected_glb_R.detach()
            self.glb_T = selected_glb_T.detach()

            out_feats = {}
            out_feats['recon_pts'] = selected_transformed_glb_recon_pts.detach().cpu().numpy()
            out_feats['inv_trans_pts'] = inv_trans_ori_pts.detach().cpu().numpy()
            out_feats['ori_pts'] = safe_transpose(ori_pts, -1, -2).detach().cpu().numpy()
            out_feats['canon_recon'] = safe_transpose(glb_recon_canon_pts, -1, -2).detach().cpu().numpy()

            out_save_fn = self.log_fn + "_stage_0.npy"
            self.out_feats = out_feats
            #### Save output crucial features ####
            np.save(out_save_fn, out_feats)
            return minn_glb_chamfer
        else:
            #### Get original points ####
            # x = x - torch.mean(x, dim=-1, keepdim=True)
            ori_pts = x.clone()
            bz, npoints = x.size(0), x.size(2)

            cur_kanchor = self.kpconv_kanchor
            cur_anchors = self.kpconv_anchors
            x = M.preprocess_input(x, cur_kanchor, pose, False)

            # pose: bz x N x 4 x 4; backbone... current backbone...
            cur_backbone = self.backbone
            # x.feats: bz x N x dim x 1 --> na = 1 now for kpconv net # cai
            for block_i, block in enumerate(cur_backbone):
                x = block(x)

            torch.cuda.empty_cache()

            # Get per-point invariant feature # is not None
            if self.sel_mode is not None and cur_iter > 0: # sel mode in the invariant otuput block
                sel_mode_new = self.sel_mode_new
            else:
                sel_mode_new = None
            if self.inv_pooling_method == 'attention':
                # confidence: bz x N x na; ppinvout: bz x dim x N
                ppinv_out, confidence = self.ppint_outblk(x, sel_mode_new=sel_mode_new)
                if self.orbit_attn == 1:
                    ppinv_out = torch.cat([ppinv_out, safe_transpose(confidence, -1, -2)], dim=1)
            else:
                ppinv_out = self.ppint_outblk(x, sel_mode_new=sel_mode_new)

            # ppinv_out: bz x dim x N
            if self.orbit_attn == 3: # coordinates involved in to the slot-attention process; Get ppinv_out by concatenating ...
                ppinv_out = torch.cat([ppinv_out, ori_pts], dim=1)

            ''' Point grouping '''
            # slot attention # Get attention values from each point to each slot; slot attention
            rep_slots, attn_ori = self.slot_attention(safe_transpose(ppinv_out, -1, -2))
            ''' Point grouping '''

            # rep_slots; attn_ori...
            # hard_labels: bz x n2; hard_labels...
            hard_labels = torch.argmax(attn_ori, dim=1)
            # hard_one_hot_labels: bz x n2 x ns
            hard_one_hot_labels = torch.eye(self.num_slots, dtype=torch.float32).cuda()[hard_labels]

            #### get tot_seg_to_idxes arr for each shape ####
            tot_seg_to_idxes = []
            tot_minn_seg_label = []
            for i_bz in range(hard_labels.size(0)):
                curr_minn_seg_label = 999.0
                curr_maxx_seg_nn = 0
                cur_seg_to_idxes = {}
                for i_pts in range(hard_labels.size(1)):
                    cur_pts_label = int(hard_labels[i_bz, i_pts].item())
                    if cur_pts_label in cur_seg_to_idxes:
                        cur_seg_to_idxes[cur_pts_label].append(i_pts)
                    else:
                        cur_seg_to_idxes[cur_pts_label] = [i_pts]
                for seg_label in cur_seg_to_idxes:
                    curr_seg_pts_nn = len(cur_seg_to_idxes[seg_label])
                    cur_seg_to_idxes[seg_label] = torch.tensor(cur_seg_to_idxes[seg_label], dtype=torch.long).cuda()
                    if curr_seg_pts_nn > curr_maxx_seg_nn: # maxx_seg_nn --> maximum number of points in an existing segmentation
                        curr_maxx_seg_nn = curr_seg_pts_nn
                        curr_minn_seg_label = seg_label
                    # curr_minn_seg_label = min(curr_minn_seg_label, seg_label)
                tot_seg_to_idxes.append(cur_seg_to_idxes) # we should use
                tot_minn_seg_label.append(curr_minn_seg_label) #


            # generate features -> points & transformations
            # todo: how to consider `mask` in a better way?
            slot_canon_pts = []
            slot_R, slot_T = [], []
            slot_axis = []
            # slot_oribts = []
            # tot_nll_orbit_selection_loss = None
            slot_recon_cuboic_constraint_loss = []
            slot_cuboic_recon_pts = []
            slot_cuboic_R = []
            slot_recon_pivot_points = []
            slot_recon_central_points = []

            slot_recon_pivot_points_equiv = []
            slot_recon_central_points_equiv = []

            tot_minn_seg_labels = []

            pv_points = []
            central_points = []

            slot_pv_canon_cd_loss = 0.0
            # other_slots_trans_pv = [[] for _ in range(bz)]
            # base_slot_canon_pred = [[] for _ in range(bz)]
            # current pair pivot points...
            # pair

            pair_pivot_points, pair_confidences = [], []

            ''' Get pivot point & confidence for each pair of slots '''
            # slot pairs...
            for i_p, i_s_pair in enumerate(self.slot_pairs_list): # pair list...
                i_s_a, i_s_b = i_s_pair[0], i_s_pair[1]
                cur_pair_inv_feats = []
                for i_bz in range(bz): # pair_inv_
                    cur_bz_cur_pair_xyz = []
                    cur_bz_cur_pair_feats = [] # cur
                    for i_s in [i_s_a, i_s_b]:
                        if i_s in tot_seg_to_idxes[i_bz]:
                            if self.with_part_proposal:
                                ''' xyz, feats '''  # current slot xyz... #
                                cur_bz_cur_slot_xyz = safe_transpose(x.xyz, -1, -2)[
                                    i_bz, tot_seg_to_idxes[i_bz][i_s]].unsqueeze(0)
                                cur_bz_cur_slot_feats = safe_transpose(x.feats, 1, 2)[
                                    i_bz, tot_seg_to_idxes[i_bz][i_s]].unsqueeze(
                                    0)
                                ''' xyz, feats '''
                            else:
                                cur_bz_cur_slot_xyz = safe_transpose(x.xyz, -1, -2)[i_bz].unsqueeze(0)
                                cur_bz_cur_slot_feats = safe_transpose(x.feats, 1, 2)[i_bz].unsqueeze(0)
                            cur_bz_cur_pair_xyz.append(cur_bz_cur_slot_xyz)
                            cur_bz_cur_pair_feats.append(cur_bz_cur_slot_feats)
                    if len(cur_bz_cur_pair_xyz) == 0:
                        cur_bz_cur_pair_xyz = torch.zeros((1, 2, 3), dtype=torch.float32).cuda()
                        cur_bz_cur_pair_feats = torch.zeros((1, 2, x.feats.size(1), x.feats.size(-1)),
                                                            dtype=torch.float32).cuda()
                    else:
                        cur_bz_cur_pair_xyz = torch.cat(cur_bz_cur_pair_xyz, dim=1)
                        cur_bz_cur_pair_feats = torch.cat(cur_bz_cur_pair_feats, dim=1)

                    cur_bz_cur_pair_x = sptk.SphericalPointCloud(safe_transpose(cur_bz_cur_pair_xyz, -1, -2),
                                                                 safe_transpose(cur_bz_cur_pair_feats, 1, 2), x.anchors)
                    cur_bz_cur_pair_equiv_feats, cur_bz_cur_pair_inv_feats, _ = self.pair_slot_outblock[i_p](cur_bz_cur_pair_x, mask=None)
                    if self.sel_mode is not None: # get points under this mode? central point, inv feautre...
                        cur_bz_cur_pair_inv_feats = cur_bz_cur_pair_equiv_feats[..., self.sel_mode]
                    cur_pair_inv_feats.append(cur_bz_cur_pair_inv_feats)
                cur_pair_inv_feats = torch.cat(cur_pair_inv_feats, dim=0)
                # pair pivot points? pair conf?
                _, cur_pair_pivot_points, _, cur_pair_conf  = self.pair_slot_shp_recon_net[i_p](cur_pair_inv_feats)
                # pair pivot points
                cur_pair_pivot_points = cur_pair_pivot_points - 0.5
                pair_pivot_points.append(cur_pair_pivot_points.unsqueeze(1))
                pair_confidences.append(cur_pair_conf.unsqueeze(1))
            pair_pivot_points = torch.cat(pair_pivot_points, dim=1) # pair pivot points
            pair_confidences = torch.cat(pair_confidences, dim=1)
            # pair_confidences: bz x n_p x 1
            pair_confidences = pair_confidences / torch.clamp(torch.sum(pair_confidences, dim=-2, keepdim=True), min=1e-6)

            # pair_pivot_points: bz x n_p x 3 --> pivot point for each slot pair
            # pair_confidence: bz x n_p x 1 --> confidence value...
            # pair_pivot_points =
            ''' Get minimum spanning tree '''
            tot_ips = []
            for i_bz in range(bz):
                cur_bz_confidences = pair_confidences[i_bz, :, 0].tolist()
                cur_bz_confidences = [(i_p, p_c) for i_p, p_c in enumerate(cur_bz_confidences)]
                sorted_confs = sorted(cur_bz_confidences, key=lambda ii:ii[1], reverse=True) # bz confidence
                slot_idx_to_exist = {}
                cur_bz_ips = []
                for ii_p, conf in enumerate(sorted_confs):
                    i_p = conf[0]
                    i_s_a, i_s_b = self.slot_pairs_list[i_p][0], self.slot_pairs_list[i_p][1]
                    if i_s_a in slot_idx_to_exist and i_s_b in slot_idx_to_exist: # the edge; reached slots
                        continue
                    cur_bz_ips.append(i_p)
                    slot_idx_to_exist[i_s_a] = 1
                    slot_idx_to_exist[i_s_b] = 1
                ### if we just fix the pair index? ###;
                # cur_bz_ips = [0, 2]
                # indices of pairs...
                cur_bz_ips = [0, 1]  # pair index list for the current batch # fix the;
                cur_bz_ips = torch.tensor(cur_bz_ips, dtype=torch.long).cuda()
                # batch selected slot pair's index...
                tot_ips.append(cur_bz_ips.unsqueeze(0))
            # bz x (n_s - 1); total pair indices...
            tot_ips = torch.cat(tot_ips, dim=0)
            #### Select pivot points via selected pair indices ####
            # pair pivot points;
            pair_pivot_points = batched_index_select(values=pair_pivot_points, indices=tot_ips.long(), dim=1)
            #### Select pair confidence via selected pair indices ####
            # selected pair indices
            # bz x (n_s - 1) x 1; bz x (n_s - 1)
            pair_confidences = batched_index_select(values=pair_confidences, indices=tot_ips.long(), dim=1)

            # spanning tree
            # it is a category-level order
            slot_orders = []
            slot_inv_orders = []
            for i_bz in range(bz):
                cur_bz_ips = tot_ips[i_bz].tolist()
                ip_a, ip_b = cur_bz_ips[0], cur_bz_ips[1]
                p_a, p_b = self.slot_pairs_list[ip_a], self.slot_pairs_list[ip_b]
                s_a = p_a[0] if p_a[0] != p_b[0] and p_a[0] != p_b[1] else p_a[1]
                s_b = p_a[1] if p_a[0] == s_a else p_a[0]
                s_c = p_b[0] if p_b[0] != s_a and p_b[0] != s_b else p_b[1]
                slot_orders.append([s_a, s_b, s_c])
                slot_idx_to_order = {ss: ii for ii, ss in enumerate([s_a, s_b, s_c])}
                cur_inv_order = [slot_idx_to_order[ss] for ss in range(self.num_slots)] # num
                slot_inv_orders.append(cur_inv_order)
            # Get slot orders...
            slot_orders = torch.tensor(slot_orders, dtype=torch.long).cuda()
            slot_inv_orders = torch.tensor(slot_inv_orders, dtype=torch.long).cuda() # slot_inv_orders: bz x ns
            ### now we have pair order tot_ips and slot orders slot_orders ###

            # print(f"slot_orders: {slot_orders}")

            ''' Get transformations for each slot '''
            # todo: change it to chain order #
            for i_s in range(self.num_slots):
                cur_slot_inv_feats = []
                # cur_pair_slot_inv_feats = [] # pair slot inv features...
                # cur_slot_orbit_confidence = []
                cur_slot_R = []
                cur_slot_T = []
                cur_slot_axis = []
                # cur_slot_defined_axises = []
                cur_slot_pivot_points_equiv = []
                cur_slot_central_points_equiv = []
                for i_bz in range(bz):
                    curr_minn_seg_label = tot_minn_seg_label[i_bz]
                    # curr_minn_seg_label = 0
                    tot_minn_seg_labels.append(curr_minn_seg_label)
                    # print(f"check x, x.xyz: {x.xyz.size()}, x.feats: {x.feats.size()}, x.anchors: {x.anchors.size()}")
                    if i_s in tot_seg_to_idxes[i_bz]:
                        # sptk.SphericalPointCloud(x_xyz, out_feat, x.anchors) # x.xyz: bz x N
                        ''' xyz, feats ''' # current slot xyz...
                        cur_bz_cur_slot_xyz = safe_transpose(x.xyz, -1, -2)[i_bz, tot_seg_to_idxes[i_bz][i_s]].unsqueeze(0)
                        cur_bz_cur_slot_feats = safe_transpose(x.feats, 1, 2)[i_bz, tot_seg_to_idxes[i_bz][i_s]].unsqueeze(
                            0)
                    else:
                        cur_bz_cur_slot_xyz = torch.zeros((1, 2, 3), dtype=torch.float32).cuda()
                        cur_bz_cur_slot_feats = torch.zeros((1, 2, x.feats.size(1), x.feats.size(-1)),
                                                            dtype=torch.float32).cuda()
                    cur_bz_cur_slot_x = sptk.SphericalPointCloud(safe_transpose(cur_bz_cur_slot_xyz, -1, -2),
                                                                 safe_transpose(cur_bz_cur_slot_feats, 1, 2), x.anchors)

                    # invariant features
                    cur_bz_cur_slot_equiv_feats, cur_bz_cur_slot_inv_feats, _ = self.slot_outblock[i_s](cur_bz_cur_slot_x, mask=None)
                    if self.sel_mode is not None: # get points under this mode? central point, inv feautre...
                        cur_bz_cur_slot_inv_feats = cur_bz_cur_slot_equiv_feats[..., self.sel_mode]
                    cur_slot_inv_feats.append(cur_bz_cur_slot_inv_feats) # slot_outblock

                    pre_feats = None

                    if self.shape_type == 'drawer':
                        ''' Use no pre-defined axis for further translation decoding '''
                        defined_proj_axis = None
                        ''' Use pre-defined axis (z-axis) for further translation decoding '''
                        defined_proj_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda().unsqueeze(0)
                        use_offset = False #
                        # use_offset = True
                        cur_bz_slot_output_RT = self.slot_trans_outblk_RT[i_s](cur_bz_cur_slot_x, mask=None,
                                                                               anchors=self.anchors.unsqueeze(0).repeat(
                                                                                   bz, 1, 1, 1).contiguous(), proj_axis=defined_proj_axis, pre_feats=None, use_offset=use_offset)
                    else:
                        ''' If we use equivariant features for rotation decoding '''
                        # we can use the predicted rotation in the following process, while the translation could not be of good use
                        # Use equiv feature for transformation & pv-points prediction?
                        # pred_pv_poitns_in_feats = cur_bz_cur_slot_equiv_feats if self.pred_pv_equiv else None #
                        pred_pv_poitns_in_feats = None # pred_pv_points_in_feats
                        pred_central_points_in_feats = None # pred_central_points
                        pred_axis_in_feats =  None
                        # ''' Use pair-wise inv feature for axis prediction? '''
                        # pred_axis_in_feats = pair_cur_bz_cur_slot_inv_feats.unsqueeze(-1).contiguous().repeat(1, 1, self.kanchor).contiguous()
                        # ''' Use pair-wise equiv feature for axis prediction? '''
                        # pred_axis_in_feats = pair_cur_bz_cur_slot_equiv_feats

                        cur_bz_slot_output_RT = self.slot_trans_outblk_RT[i_s](cur_bz_cur_slot_x, mask=None,
                                                                               trans_feats=safe_transpose(
                                                                                   cur_bz_cur_slot_feats, 1, 2),
                                                                               trans_xyz=safe_transpose(
                                                                                   cur_bz_cur_slot_xyz, -1, -2),
                                                                               anchors=self.anchors.unsqueeze(0).repeat(
                                                                                   bz, 1, 1, 1).contiguous(),
                                                                               pre_feats=pre_feats,
                                                                               pred_pv_poitns_in_feats=pred_pv_poitns_in_feats,
                                                                               pred_central_points_in_feats=pred_central_points_in_feats,
                                                                               pred_axis_in_feats=pred_axis_in_feats)

                    # Get current slot's rotation; slot_R
                    cur_bz_cur_slot_R = cur_bz_slot_output_RT['R']
                    cur_bz_cur_slot_axis = cur_bz_slot_output_RT['axis'] # 1 x 3 x na; get axis... ### Get axis for each slot...
                    # cur_bz_cur_slot_axis = pair_cur_bz_slot_output_RT['axis']
                    # Get current slot's translation
                    if self.pred_t:
                        cur_bz_cur_slot_T = cur_bz_slot_output_RT['T']
                    else:
                        cur_bz_cur_slot_T = torch.zeros((1, 3, cur_kanchor), dtype=torch.float).cuda()
                    if self.pred_pv_equiv:
                        cur_bz_cur_slot_pv_points_equiv = cur_bz_slot_output_RT['pv_points']
                        cur_bz_cur_slot_central_points_equiv = cur_bz_slot_output_RT['central_points']
                        cur_bz_cur_slot_pv_points_equiv = cur_bz_cur_slot_pv_points_equiv - 0.5 # quite
                        cur_bz_cur_slot_central_points_equiv = cur_bz_cur_slot_central_points_equiv - 0.5
                        cur_slot_pivot_points_equiv.append(cur_bz_cur_slot_pv_points_equiv)
                        cur_slot_central_points_equiv.append(cur_bz_cur_slot_central_points_equiv)
                    # !!! Important values: rotation, translation, and rotation axis !!! #
                    cur_slot_R.append(cur_bz_cur_slot_R)
                    cur_slot_T.append(cur_bz_cur_slot_T)
                    cur_slot_axis.append(cur_bz_cur_slot_axis)

                # invariant features #
                cur_slot_inv_feats = torch.cat(cur_slot_inv_feats, dim=0)
                # cur_pair_slot_inv_feats = torch.cat(cur_pair_slot_inv_feats, dim=0)
                # cur_slot_orbit_confidence = torch.cat(cur_slot_orbit_confidence, dim=0)
                if self.pred_pv_equiv:
                    cur_slot_pivot_points_equiv = torch.cat(cur_slot_pivot_points_equiv, dim=0)
                    cur_slot_central_points_equiv = torch.cat(cur_slot_central_points_equiv, dim=0)

                if self.recon_prior == 5:
                    # cur_slot_cuboic_R: bz x 3 x 3
                    cur_slot_canon_pts, cur_slot_cuboic_constraint_loss, cur_slot_cuboic_x, cur_slot_cuboic_R = \
                    self.slot_shp_recon_net[i_s](cur_slot_inv_feats)
                    slot_recon_cuboic_constraint_loss.append(cur_slot_cuboic_constraint_loss.unsqueeze(-1))
                    slot_cuboic_recon_pts.append(cur_slot_cuboic_x.unsqueeze(1))
                    slot_cuboic_R.append(cur_slot_cuboic_R.unsqueeze(1))
                elif self.recon_prior == 6:
                    # cur_slot_cuboic_R: bz x 3 x 3
                    cur_slot_canon_pts, cur_slot_pivot_points, cur_slot_central_points = \
                        self.slot_shp_recon_net[i_s](cur_slot_inv_feats) # get canonical points and central points...
                    # cur_slot_canon_pts = cur_slot_canon_pts + curslo
                    cur_slot_pivot_points = cur_slot_pivot_points - 0.5
                    cur_slot_central_points = cur_slot_central_points - 0.5

                    slot_recon_pivot_points.append(cur_slot_pivot_points.unsqueeze(1))
                    slot_recon_central_points.append(cur_slot_central_points.unsqueeze(1)) # central points

                    pv_points.append(cur_slot_pivot_points.unsqueeze(1))
                    central_points.append(cur_slot_central_points.unsqueeze(1))

                    if self.pred_pv_equiv:
                        # cur_slot_pivot_points_equiv, cur_slot_central_points_equiv = cur_slot_pivot_points_equiv[:, :3, ...], cur_slot_pivot_points_equiv[:, 3:, ...]
                        slot_recon_pivot_points_equiv.append(cur_slot_pivot_points_equiv.unsqueeze(1))
                        slot_recon_central_points_equiv.append(cur_slot_central_points_equiv.unsqueeze(1))
                elif self.recon_prior == 7:
                    # cur_slot_cuboic_R: bz x 3 x 3
                    # get
                    cur_slot_canon_pts, cur_slot_pivot_points, cur_slot_central_points, cur_slot_cuboic_x, cur_slot_cuboic_R = \
                        self.slot_shp_recon_net[i_s](cur_slot_inv_feats)
                    # cur_slot_canon_pts = cur_slot_canon_pts + curslo
                    cur_slot_pivot_points = cur_slot_pivot_points - 0.5
                    cur_slot_central_points = cur_slot_central_points - 0.5

                    slot_recon_pivot_points.append(cur_slot_pivot_points.unsqueeze(1))
                    slot_recon_central_points.append(cur_slot_central_points.unsqueeze(1))
                    #### Save pivot points and central points ####
                    slot_cuboic_recon_pts.append(cur_slot_cuboic_x.unsqueeze(1)) # cuboid reconstruction points
                    slot_cuboic_R.append(cur_slot_cuboic_R.unsqueeze(1)) # cuboid rotation xxx
                else:
                    cur_slot_canon_pts = self.slot_shp_recon_net[i_s](cur_slot_inv_feats)

                # Get slot canonical points
                cur_slot_canon_pts = cur_slot_canon_pts - 0.5 # get canonical points

                # How to use pv poitns? #
                ''' Saperated prediction version '''
                cur_slot_R = torch.cat(cur_slot_R, dim=0)
                cur_slot_T = torch.cat(cur_slot_T, dim=0)
                cur_slot_axis = torch.cat(cur_slot_axis, dim=0) # cat slot axis over all shapes in a single batch...
                ''' Saperated prediction version '''

                # cur_slot_canon_pts
                slot_canon_pts.append(cur_slot_canon_pts.unsqueeze(1))
                slot_R.append(cur_slot_R.unsqueeze(1))
                slot_T.append(cur_slot_T.unsqueeze(1))
                slot_axis.append(cur_slot_axis.unsqueeze(1))


            # slot_canon_pts: bz x n_s x M x 3
            # slot_R: bz x n_s x 4 x na
            # slot_T: bz x n_s x 3 x na;
            # reconstruction for points...
            slot_canon_pts = torch.cat(slot_canon_pts, dim=1)
            slot_canon_pts = safe_transpose(slot_canon_pts, -1, -2)

            # slot_axis: bz x ns x 3 x na
            slot_axis = torch.cat(slot_axis, dim=1) # slot axis
            slot_axis = safe_transpose(slot_axis, -1, -2)

            # slot_cuboic_recon_pts: bz x n_s x 3
            # slot_cuboic_recon_pts = torch.cat(slot_cuboic_recon_pts, dim=1)
            # slot_cuboic_R: bz x n_s x 3 x 3
            # slot_cuboic_R = torch.cat(slot_cuboic_R, dim=1)

            # pv_points: bz x n_s x 3; pv_points
            pv_points = torch.cat(pv_points, dim=1) # your predicted pv_points #
            central_points = torch.cat(central_points, dim=1) # get central point

            if self.pred_pv_equiv:
                slot_recon_pivot_points_equiv = torch.cat(slot_recon_pivot_points_equiv, dim=1) # pivot points predicted by equiv features
                slot_recon_central_points_equiv = torch.cat(slot_recon_central_points_equiv, dim=1)
                slot_recon_pivot_points_equiv = safe_transpose(slot_recon_pivot_points_equiv, -1, -2)
                slot_recon_central_points_equiv = safe_transpose(slot_recon_central_points_equiv, -1, -2) # bz x ns x na x 3 --> central points for each anchor

            # Get pivot points
            slot_pivot_points = torch.cat(slot_recon_pivot_points, dim=1)  # Concatenate slot pivot points
            # Get central points
            slot_central_points = torch.cat(slot_recon_central_points, dim=1) # Central points

            # pair_pivot_points: bz x n_p x 3
            # slot_orders: bz x 3

            # slot_R: bz x n_s x na x 1
            ''' From predicted rotation angles to rotation matrices '''
            # todo: it should be an transformation chain from the left most part to the center part & the one from the right most part to the center part; and for three parts we can achieve the goal by the following strategy...

            slot_R_raw = torch.cat(slot_R, dim=1).detach()
            slot_R = torch.cat(slot_R, dim=1) # slot_R
            mtx_slot_R = []
            slot_T = []


            ''' An inter-chain-like transformation modeling --- transformation chain modeling version 1 '''
            # # R: bz
            defined_axises = []
            for i_s in range(self.num_slots): # get slots' rotations
                cur_slot_idxes = slot_orders[:, i_s]
                # cur_slot_central_points: bz x 3
                cur_slot_central_points = batched_index_select(values=central_points, indices=cur_slot_idxes.unsqueeze(1), dim=1).squeeze(1)
                defined_axis = slot_axis[:, 0, :]  # defined_axis: bz x 3 # todo: is it a proper shape for axis as input for computing rotation matrices?

                cur_slot_defined_axises = defined_axis # .unsqueeze(0)
                defined_axises.append(cur_slot_defined_axises)
                if i_s == self.num_slots // 2: # and still use the central part as the
                    cur_slot_R_mtx = torch.eye(3, dtype=torch.float32).cuda().contiguous().unsqueeze(0).unsqueeze(
                            0).unsqueeze(0).repeat(bz, 1, cur_kanchor, 1, 1).contiguous()
                    cur_slot_trans = cur_slot_central_points.unsqueeze(1).unsqueeze(-2).contiguous().repeat(1, 1, cur_kanchor, 1).contiguous()
                else:
                    # cur_slot_idxes: bz defined axis; defined axis...

                    # defined
                    ''' If we inverse the axis when passing the non-moving part '''
                    defined_axis = -1.0 * defined_axis if i_s < self.num_slots // 2 else defined_axis
                    ''' If we do not inverse the axis when passing the non-moving part '''
                    # defined_axis = 1.0 * defined_axis if i_s < self.num_slots // 2 else defined_axis
                    # cur_slot_angles: bz x 1 x na
                    cur_slot_angles = batched_index_select(values=slot_R, indices=cur_slot_idxes.long().unsqueeze(-1), dim=1).squeeze(1)
                    cur_slot_R_mtx = torch.sigmoid(cur_slot_angles) * math.pi * self.rot_angle_factor # get slot indices
                    if self.shape_type == 'drawer':
                        cur_slot_R_mtx = cur_slot_R_mtx * 0.0
                    cur_slot_R_mtx = compute_rotation_matrix_from_angle(
                        cur_anchors,
                        safe_transpose(cur_slot_R_mtx, -1, -2).view(1 * 1, cur_kanchor, -1),
                        defined_axis=defined_axis
                    ).contiguous().view(bz, 1, cur_kanchor, 3, 3).contiguous()

                    cur_slot_pv_point_idx = i_s if i_s < self.num_slots // 2 else i_s - self.num_slots #

                    # you need a batch selection if ...
                    cur_slot_pv_points = pair_pivot_points[:, cur_slot_pv_point_idx, :] # bz x 3 for the pivot point of this slot...
                    # cur_slot_R_mtx: bz x 1 x na x 3 x 3; central_points: bz x 3 --> bz x 1 x na x 3;
                    cur_slot_trans = 1.0 * torch.matmul(cur_slot_R_mtx, # slot_R_mtx slot_R_trans
                                                        (cur_slot_central_points - cur_slot_pv_points).unsqueeze(1).unsqueeze(1).unsqueeze(-1)) + (cur_slot_pv_points).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
                    cur_slot_trans = cur_slot_trans.squeeze(-1) # cur_slot_trans

                mtx_slot_R.append(cur_slot_R_mtx)
                slot_T.append(cur_slot_trans)
            ''' An inter-chain-like transformation modeling --- transformation chain modeling version 1 '''

            slot_R = torch.cat(mtx_slot_R, dim=1)
            slot_T = torch.cat(slot_T, dim=1)
            defined_axises = defined_axises[0]

            #
            defined_pivot_points = pv_points[:, 0, :]  # bz x 3
            # defined_axises: bz x 3
            offset_pivot_points = defined_pivot_points - torch.sum(defined_pivot_points * defined_axises, dim=-1,
                                                                   keepdim=True) * defined_axises
            offset_pivot_points = torch.norm(defined_pivot_points, p=2, dim=-1)

            # slot_R: bz x 3 x na x 3 x 3 --> inverse transform via inv orders;
            # slot_T: bz x 3 x na x 3 x 3 --> inverse transform via inv orders...
            slot_R = batched_index_select(values=slot_R, indices=slot_inv_orders.long(), dim=1)
            slot_T = batched_index_select(values=slot_T, indices=slot_inv_orders.long(), dim=1)

            if self.recon_prior == 5:
                slot_cuboic_R = torch.cat(slot_cuboic_R, dim=1)

            # slot_R = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), slot_R)
            slot_R_ori = slot_R.clone().detach() # rotation matrix # rotation matrix...
            # slot_R: bz x ns x na x 3 x 3

            slot_T_ori = slot_T.clone()
            # transformed_pv_points: bz x n_s x 3
            # pv_points: bz x n_s x 3
            ''' Get pv points '''
            # ... but how about add a translation for pv
            # central_transformed_pv_points = pv_points + central_points.detach()
            # central_transformed_pv_points = pv_points
            # central_transformed_pv_points_equiv = slot_recon_pivot_points_equiv
            # + slot_recon_central_points_equiv.detach()

            # R_anchor(R(P) + T)
            ''' Get slots' rotations and translations --- if we use pivot points... '''
            slot_R = torch.matmul(cur_anchors.unsqueeze(0).unsqueeze(0), slot_R) #
            slot_T = torch.matmul(cur_anchors.unsqueeze(0).unsqueeze(0), slot_T.unsqueeze(-1)).squeeze(-1)

            #### Shape type --> drawer ####
            if self.shape_type == 'drawer': # drawer
                # slot_T[:, 0] = 0.0 # set the translation of the first slot to zero... set the other to xxx...; the
                slot_T[:, 0] = slot_T[:, 0] * 0.0 # fix points of the first slot

            # kpconv-kanchor
            k = self.kpconv_kanchor if self.sel_mode_trans is None else 1
            if self.sel_mode_trans is not None: # select a specific mode for transformation
                topk_anchor_idxes = torch.tensor([self.sel_mode_trans], dtype=torch.long).cuda().unsqueeze(0).unsqueeze(
                    0).repeat(bz, self.num_slots, 1).contiguous()

            # transformed_pts: bz x n_s x na x M x 3
            # slot_canon_pts: bz x n_s x M x 3
            # slot_R: bz x n_s x na x 3 x 3 @ slot_recon_pts: bz x n_s x 1 x 3 x M
            # transformed points
            transformed_pts = safe_transpose(torch.matmul(slot_R, safe_transpose(slot_canon_pts.unsqueeze(2), -1, -2)), -1,
                                             -2) + slot_T.unsqueeze(-2)

            if self.recon_prior == 6 or self.recon_prior == 7:
                # transformed_pts_ori: bz x n_s x na x M x 3
                # slot_canon_pts: bz x n_s x M x 3;
                transformed_pts_ori = safe_transpose(
                    torch.matmul(slot_R_ori, safe_transpose(slot_canon_pts.unsqueeze(2), -1, -2)), -1,
                    -2) + slot_T_ori.unsqueeze(-2)
                # pv_points: bz x n_s x 3; slot_R_ori: bz x n_s x na x 3 x 3 --> bz x n_s x na x 3
                transformed_pv_points_ori = torch.matmul(slot_R_ori, pv_points.unsqueeze(2).unsqueeze(-1)).squeeze(-1) #
                transformed_pv_points = torch.matmul(slot_R, pv_points.unsqueeze(2).unsqueeze(-1)).squeeze(-1)
                # transformed_central_pv_points_ori = torch.matmul(slot_R_ori, central_transformed_pv_points.unsqueeze(2).unsqueeze(-1)).squeeze(-1)


            # selected_anchors = cur_anchors.clone() # na x 3 x 3
            if k < cur_kanchor: # cur_kanchor
                # transformed_pts: bz x n_s x na x M x 3 --> bz x n_s x k x M x 3
                # selected_anchors: bz x ns x k_a x 3 x 3
                # selected_anchors = batched_index_select(values=selected_anchors, indices=topk_anchor_idxes, dim=0)
                transformed_pts = batched_index_select(values=transformed_pts, indices=topk_anchor_idxes, dim=2)
                if self.recon_prior == 6 or self.recon_prior == 7:
                    transformed_pts_ori = batched_index_select(values=transformed_pts_ori, indices=topk_anchor_idxes, dim=2)
                    transformed_pv_points_ori = batched_index_select(values=transformed_pv_points_ori, indices=topk_anchor_idxes, dim=2)
                    transformed_pv_points = batched_index_select(values=transformed_pv_points, indices=topk_anchor_idxes, dim=2)
                    # transformed_central_pv_points = batched_index_select(values=transformed_central_pv_points,
                    # indices=topk_anchor_idxes, dim=2)
                    # transformed pv points
                    slot_axis = batched_index_select(values=slot_axis, indices=topk_anchor_idxes, dim=2)
                    defined_axises = batched_index_select(values=defined_axises, indices=topk_anchor_idxes, dim=1)

                    if self.pred_pv_equiv:
                        slot_recon_pivot_points_equiv = batched_index_select(values=slot_recon_pivot_points_equiv, indices=topk_anchor_idxes, dim=2)
                        slot_recon_central_points_equiv = batched_index_select(values=slot_recon_central_points_equiv, indices=topk_anchor_idxes, dim=2)

            # transformed_pts: bz x n_s x na x M x 3 --> bz x n_s x k x M x 3
            # transformed_pts = batched_index_select(values=transformed_pts, indices=topk_anchor_idxes, dim=2)

            # hard_one_hot_labels: bz x N x ns
            # dist_recon_ori: bz x n_s x na x M x N

            dist_recon_ori = torch.sum((transformed_pts.unsqueeze(-2) - safe_transpose(ori_pts, -1, -2).unsqueeze(
                1).unsqueeze(1).unsqueeze(1)) ** 2, dim=-1)
            expanded_hard_one_hot_labels = safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(2).unsqueeze(2).repeat(1, 1, k,  self.recon_part_M, 1)

            minn_dist_ori_to_recon_all_pts, _ = torch.min(dist_recon_ori, dim=-2)
            # slot reconstruction points to original points
            minn_dist_recon_to_ori_all_pts, _ = torch.min(dist_recon_ori, dim=-1)
            minn_dist_recon_to_ori_all_pts = minn_dist_recon_to_ori_all_pts.mean(dim=-1)
            # restrict the range of min
            dist_recon_ori[expanded_hard_one_hot_labels < 0.5] = 99999.0
            minn_dist_recon_to_ori, _ = torch.min(dist_recon_ori, dim=-1)
            # minn_dist_recon_to_ori: bz x n_s x na x M --> bz x n_s x na
            minn_dist_recon_to_ori = minn_dist_recon_to_ori.mean(-1)
            # dist_recon_ori[expanded_hard_one_hot_labels < 0.5] = 99999.0
            # minn_dist_ori_to_recon: bz x n_s x na x N --> the distance from point in ori shape to an orbit...
            minn_dist_ori_to_recon, _ = torch.min(dist_recon_ori, dim=-2)
            # minn_dist_ori_to_recon = minn_dist_ori_to_recon *
            # minn_dist_ori_to_reco: bz x n_s x na
            # ori to recon ---> for each slot an each orbit
            #
            #### If we use soft weights only for points in the cluster ####
            soft_weights = safe_transpose(hard_one_hot_labels, -1, -2) * attn_ori
            #### If we use soft weights for all points #### --- if we add a parameter for it?
            # soft_weights = attn_ori
            # minn_dist_ori_to_recon = torch.sum(minn_dist_ori_to_recon * safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(2), dim=-1) / torch.clamp(torch.sum(safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(2), dim=-1), min=1e-8)
            minn_dist_ori_to_recon_hard = torch.sum(
                minn_dist_ori_to_recon * safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(2), dim=-1) / torch.clamp(
                torch.sum(safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(2), dim=-1), min=1e-8)
            minn_dist_ori_to_recon = torch.sum(minn_dist_ori_to_recon * soft_weights.unsqueeze(2), dim=-1) / torch.clamp(
                torch.sum(soft_weights.unsqueeze(2), dim=-1), min=1e-8)

            #### use soft weights of all points for ori_to_recon soft loss aggregation ####
            minn_dist_ori_to_recon_all_pts = torch.sum(minn_dist_ori_to_recon_all_pts * attn_ori.unsqueeze(2),
                                                       dim=-1) / torch.clamp(torch.sum(attn_ori.unsqueeze(2), dim=-1),
                                                                             min=1e-8)

            orbit_slot_dist_ori_recon = minn_dist_ori_to_recon + minn_dist_recon_to_ori
            # orbit_slot_dist_ori_recon = minn_dist_ori_to_recon_hard + minn_dist_recon_to_ori

            ''' Distance for slot orbit selection '''
            if self.slot_single_cd == 1:
                orbit_slot_dist_ori_recon = minn_dist_ori_to_recon  # single direction chamfer distance
            else:
                orbit_slot_dist_ori_recon = minn_dist_ori_to_recon + minn_dist_recon_to_ori  # ori_to_recon, recon_to_ori?
            ''' Distance for slot orbit selection '''

            if self.slot_single_mode == 1:
                # orbit_slot_dist_ori_recon_all_slots: bz x na
                orbit_slot_dist_ori_recon_all_slots = torch.sum(orbit_slot_dist_ori_recon, dim=1)
                slot_dist_ori_recon_all_slots, slot_orbits = torch.min(orbit_slot_dist_ori_recon_all_slots, dim=-1)
                # slot_orbits:
                slot_orbits = slot_orbits.unsqueeze(-1).contiguous().repeat(1, self.num_slots).contiguous()
            else:
                # orbit_slot_dist_ori_recon = minn_dist_ori_to_recon_hard + minn_dist_recon_to_ori
                # slot_dist_ori_recon: bz x n_s; slot_orbits: bz x n_s
                slot_dist_ori_recon, slot_orbits = torch.min(orbit_slot_dist_ori_recon, dim=-1)


            # slot_dist_ori_recon: bz x n_s; slot_orbits: bz x n_s
            # slot_dist_ori_recon, slot_orbits = torch.min(orbit_slot_dist_ori_recon, dim=-1)

            hard_slot_indicators = (hard_one_hot_labels.sum(1) > 0.5).float() # Get hard slot indicators
            # mult by slot indicators.... whether it should be restricted by slot indicators? --- Yes!!!
            # slot_dist_ori_recon = (slot_dist_ori_recon * (hard_one_hot_labels.sum(1) > 0.5).float()).sum(-1)

            # use (1) in-cluster points ori_to_recon loss + (2) recon_to_all_ori_pts recon loss for optimization
            # orbit_slot_dist_ori_recon_all_pts = minn_dist_ori_to_recon + minn_dist_recon_to_ori_all_pts

            #### slot distance recon all pts ####
            orbit_slot_dist_ori_recon_all_pts = minn_dist_ori_to_recon + minn_dist_recon_to_ori

            ''' Distance for further optimization '''
            #### slot distance recon all pts ####
            if self.slot_single_cd == 1:
                orbit_slot_dist_ori_recon_all_pts = minn_dist_ori_to_recon
            else:
                orbit_slot_dist_ori_recon_all_pts = minn_dist_ori_to_recon + minn_dist_recon_to_ori
            ''' Distance for further optimization '''

            orbit_slot_dist_ori_recon_all_pts = batched_index_select(values=orbit_slot_dist_ori_recon_all_pts,
                                                                     indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(-1)
            slot_dist_ori_recon = (orbit_slot_dist_ori_recon_all_pts * hard_slot_indicators).float().sum(-1)

            # print(f"After transformed: {transformed_pts.size()}, slot_orbits: {slot_orbits.size()}")
            # transformed_pts: bz x n_s x M x 3
            transformed_pts = batched_index_select(transformed_pts, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)

            ''' Add loss for transformed pv points, ori transformed points, and predicted axises '''
            if self.recon_prior == 6 or self.recon_prior == 7:
                # Canonical reconstruction and
                # original transformed points
                # transformed_pts_ori: bz x n_s x M x 3
                transformed_pts_ori = batched_index_select(transformed_pts_ori, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                # transformed pv points...
                transformed_pv_points_ori = batched_index_select(transformed_pv_points_ori, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                # transformed_pv_points = batched_index_select(transformed_pv_points, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                # selected_anchors: bz x n_s x k_a x 3 x 3 --> bz x n_s x 3 x 3
                # print(f"selected_anchors: {selected_anchors.size()}, slot_orbits: {slot_orbits.size()}, max: {torch.max(slot_orbits)}, min: {torch.min(slot_orbits)}")
                # selected_anchors = batched_index_select(values=selected_anchors, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)

                # seg_label_to_inv_transformed_pts = {}
                # tot_seg_to_idxes: [bz][i_s] --> for further points indexes in this segmentation extraction...

                # the distance between pv points and adjacent parts; as well as those shifted pv points along the axis...
                if self.pred_pv_equiv:
                    # slot_recon_pivot_points_equiv: bz x n_s x 3
                    slot_recon_pivot_points_equiv = batched_index_select(slot_recon_pivot_points_equiv, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                    # slot_recon_central_points_equiv: bz x n_s x 3
                    slot_recon_central_points_equiv = batched_index_select(slot_recon_central_points_equiv, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                    slot_pivot_points = slot_recon_pivot_points_equiv
                    central_points = slot_recon_central_points_equiv # get slot recon central points
                # transformed_central_pv_points = batched_index_select(transformed_central_pv_points, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)

                # todo: make predicted pivot points close to central points transformed slot canonical points
                # slot_canon_pts: bz x n_s x M x 3 --> central_transformed_canon_points: bz x n_s x M x 3
                # with central points --> no rotation is involved in the process #
                central_transformed_canon_points = slot_canon_pts + central_points.unsqueeze(-2)  # central point transformed part shape reconstruction
                # rotation is involved...
                # central_transformed_canon_points = transformed_pts_ori.detach() if self.pred_axis else transformed_pts_ori
                canon_transformed_points = transformed_pts_ori  # transformed points in the canonical space

                if self.pred_axis:
                    central_transformed_canon_points = central_transformed_canon_points.detach()
                    canon_transformed_points = canon_transformed_points.detach()

                # todo: we can just use y-axis here for rotation matrix calculation
                ''' Get predicted axises for the selected mode '''
                # print("slot_orbits_now", slot_orbits)
                # print(
                #     f"before orbit selection...slot_axis: {slot_axis.size()}, slot_orbits: {slot_orbits.size()}, defined_axises: {defined_axises.size()}")
                # print("slot_orbits_now", slot_orbits)
                # slot_axis: bz x n_s x 3
                slot_axis = batched_index_select(values=slot_axis, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)

                defined_axises = batched_index_select(values=defined_axises, indices=slot_orbits[:, 0].unsqueeze(-1),
                                                      dim=1).squeeze(1)

                # print(f"max of slot-orbits: {torch.max(slot_orbits)}, min of slot-orbits: {torch.min(slot_orbits)}")
                # print(f"slot_axis: {slot_axis.size()}, slot_orbits: {slot_orbits.size()}, defined_axises: {defined_axises.size()}")

                avg_slot_axis = slot_axis[:, 0, :]
                avg_slot_axis = avg_slot_axis.detach()
                avg_slot_axis = defined_axises

                nn = 7
                dists = [0.02 * _ for _ in range(1, nn + 1)]

                # joint_len = 0.20
                # joint_len = 0.10 # 0.15 # for washing machine?
                joint_len = 0.05 # 0.15 # for washing machine?
                # assume the length of the joint is 0.3;
                # then we randomly gneerate 10 points in the 0.3 range
                randomized_dists = torch.randint(1, int(joint_len * 100) + 1, (nn,)) # get randomized distances
                randomized_dists = randomized_dists.float() / 100. # - joint_len / 2.
                dists = randomized_dists.tolist() # dist...

                # dists = []

                ''' For confidence realted pivot points and part shape constraints... '''
                minn_other_slots_base_canon_dist = 0.0

                # pair_confidences: bz x n_p x 1
                # original transformed points and central transformed points ---> both of them should be included for distance calculation
                # tot_ips: bz x (n_s - 1); pair_pv_points: bz x (n_s - 1) x 3
                slot_pair_idxes = torch.tensor(self.slot_pairs_list, dtype=torch.long).cuda() # (n_p) x 2
                # print(f"tot_pair_confidence: {pair_confidences}") # print out pair confidences...

                for i_p in range(tot_ips.size(1)):
                    cur_ip = tot_ips[:, i_p]
                    cur_p_confidence = pair_confidences[:, i_p, 0] # bz
                    # slot_pair_idxes
                    # cur_p_slots: bz x 1 x 2 --> bz x 2
                    cur_p_slots = batched_index_select(values=slot_pair_idxes, indices=cur_ip.unsqueeze(-1), dim=0).squeeze(1)
                    # slot_a_idxes: bz;
                    slot_a_idxes, slot_b_idxes = cur_p_slots[:, 0], cur_p_slots[:, 1]
                    cur_pv_point = pair_pivot_points[:, i_p, :] # cur_pv_point: bz x 3

                    # cur_pv_point = cur_pv_point.detach()

                    cur_slot_a_central_transformed_points = batched_index_select(values=central_transformed_canon_points, indices=slot_a_idxes.unsqueeze(-1), dim=1).squeeze(1) # bz x M x 3
                    cur_slot_b_central_transformed_points = batched_index_select(
                        values=central_transformed_canon_points, indices=slot_b_idxes.unsqueeze(-1), dim=1).squeeze(
                        1)  # bz x M x 3
                    cur_slot_a_canon_transformed_points = batched_index_select(values=canon_transformed_points, indices=slot_a_idxes.unsqueeze(-1), dim=1).squeeze(1)
                    cur_slot_b_canon_transformed_points = batched_index_select(values=canon_transformed_points, indices=slot_b_idxes.unsqueeze(-1), dim=1).squeeze(1)

                    dist_pv_central_transformed_a = torch.sum((cur_pv_point.unsqueeze(1) - cur_slot_a_central_transformed_points) ** 2, dim=-1)
                    minn_dist_pv_central_transformed_a, _ = torch.min(dist_pv_central_transformed_a, dim=-1)
                    dist_pv_central_transformed_b = torch.sum(
                        (cur_pv_point.unsqueeze(1) - cur_slot_b_central_transformed_points) ** 2, dim=-1)
                    minn_dist_pv_central_transformed_b, _ = torch.min(dist_pv_central_transformed_b, dim=-1)

                    minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + (minn_dist_pv_central_transformed_a + minn_dist_pv_central_transformed_b).mean() / 2.

                    dist_pv_canon_transformed_a = torch.sum(
                        (cur_pv_point.unsqueeze(1) - cur_slot_a_canon_transformed_points) ** 2, dim=-1)
                    minn_dist_pv_canon_transformed_a, _ = torch.min(dist_pv_canon_transformed_a, dim=-1)
                    dist_pv_canon_transformed_b = torch.sum(
                        (cur_pv_point.unsqueeze(1) - cur_slot_b_canon_transformed_points) ** 2, dim=-1)
                    minn_dist_pv_canon_transformed_b, _ = torch.min(dist_pv_canon_transformed_b, dim=-1)

                    # minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist +
                    minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + (minn_dist_pv_canon_transformed_a + minn_dist_pv_canon_transformed_b).mean() / 2. # * cur_p_confidence

                    # cur_pv_point = cur_pv_point.detach()
                    # no backward for pv point...
                    for dis in dists:
                        ''' First point -- the predicted pivot point & central transformed canon points '''
                        # dist_pv_points_central_transformed_canon_pts: bz x n_s x M;
                        shift_slot_recon_pv_points = cur_pv_point - dis * avg_slot_axis


                        dist_pv_central_transformed_a = torch.sum(
                            (shift_slot_recon_pv_points.unsqueeze(1) - cur_slot_a_central_transformed_points) ** 2, dim=-1)
                        minn_dist_pv_central_transformed_a, _ = torch.min(dist_pv_central_transformed_a, dim=-1)
                        dist_pv_central_transformed_b = torch.sum(
                            (shift_slot_recon_pv_points.unsqueeze(1) - cur_slot_b_central_transformed_points) ** 2, dim=-1)
                        minn_dist_pv_central_transformed_b, _ = torch.min(dist_pv_central_transformed_b, dim=-1)

                        minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + (
                                    minn_dist_pv_central_transformed_a + minn_dist_pv_central_transformed_b).mean() / 2.

                        dist_pv_canon_transformed_a = torch.sum(
                            (shift_slot_recon_pv_points.unsqueeze(1) - cur_slot_a_canon_transformed_points) ** 2, dim=-1)
                        minn_dist_pv_canon_transformed_a, _ = torch.min(dist_pv_canon_transformed_a, dim=-1)
                        dist_pv_canon_transformed_b = torch.sum(
                            (shift_slot_recon_pv_points.unsqueeze(1) - cur_slot_b_canon_transformed_points) ** 2, dim=-1)
                        minn_dist_pv_canon_transformed_b, _ = torch.min(dist_pv_canon_transformed_b, dim=-1)

                        # minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist +
                        minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + (
                                    minn_dist_pv_canon_transformed_a + minn_dist_pv_canon_transformed_b).mean() / 2. # * cur_p_confidence
                minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist / (nn / 2.)

                ''' For points projection  '''
                # along the axis offset direction...vote for axis? or add constraints for axis? then how to add points central based constraints for axis?
                # pv point and the vector from pv point to other points we would like that the direction
                # for both of central transformed and rotated...
                clutter_loss = 0.0
                for i_p in range(tot_ips.size(1)):
                    cur_ip = tot_ips[:, i_p] # total pair indices...
                    cur_p_confidence = pair_confidences[:, i_p, 0]  # bz # confidence....
                    # cur_p_slots: bz x 1 x 2 --> bz x 2
                    cur_p_slots = batched_index_select(values=slot_pair_idxes, indices=cur_ip.unsqueeze(-1),
                                                       dim=0).squeeze(1)
                    # slot_a_idxes: bz;
                    slot_a_idxes, slot_b_idxes = cur_p_slots[:, 0], cur_p_slots[:, 1]
                    cur_pv_point = pair_pivot_points[:, i_p, :]  # cur_pv_point: bz x 3 # for
                    cur_slot_a_central_transformed_points = batched_index_select(
                        values=central_transformed_canon_points, indices=slot_a_idxes.unsqueeze(-1), dim=1).squeeze(
                        1)  # bz x M x 3
                    cur_slot_b_central_transformed_points = batched_index_select(
                        values=central_transformed_canon_points, indices=slot_b_idxes.unsqueeze(-1), dim=1).squeeze(
                        1)  # bz x M x 3
                    cur_slot_a_canon_transformed_points = batched_index_select(values=canon_transformed_points,
                                                                               indices=slot_a_idxes.unsqueeze(-1),
                                                                               dim=1).squeeze(1)
                    cur_slot_b_canon_transformed_points = batched_index_select(values=canon_transformed_points,
                                                                               indices=slot_b_idxes.unsqueeze(-1),
                                                                               dim=1).squeeze(1)
                    # pv_to_central_transformed_a: bz x M x 3
                    pv_to_central_transformed_a = cur_slot_a_central_transformed_points - cur_pv_point.unsqueeze(-2)
                    # avg_slot_axis: bz x 3; dot: bz x M
                    dot_avg_axis_pv_to_central_a = torch.sum(avg_slot_axis.unsqueeze(-2) * pv_to_central_transformed_a, dim=-1)
                    # re_pv_to_central_transformed_a: bz x M x 3
                    re_pv_to_central_transformed_a = pv_to_central_transformed_a - dot_avg_axis_pv_to_central_a.unsqueeze(-1) * avg_slot_axis.unsqueeze(-2)

                    # pv_to_central_transformed_a: bz x M x 3
                    pv_to_central_transformed_b = cur_slot_b_central_transformed_points - cur_pv_point.unsqueeze(-2)
                    # avg_slot_axis: bz x 3; dot: bz x M
                    dot_avg_axis_pv_to_central_b = torch.sum(avg_slot_axis.unsqueeze(-2) * pv_to_central_transformed_b,
                                                             dim=-1)
                    # re_pv_to_central_transformed_a: bz x M x 3
                    re_pv_to_central_transformed_b = pv_to_central_transformed_b - dot_avg_axis_pv_to_central_b.unsqueeze(
                        -1) * avg_slot_axis.unsqueeze(-2)
                    # bz x M x 3
                    re_pv_to_central_transformed_a = re_pv_to_central_transformed_a / torch.clamp(torch.norm(re_pv_to_central_transformed_a, dim=2, keepdim=True), min=1e-6)
                    re_pv_to_central_transformed_b = re_pv_to_central_transformed_b / torch.clamp(
                        torch.norm(re_pv_to_central_transformed_b, dim=2, keepdim=True), min=1e-6)
                    # dot_a_a: bz x M x M
                    dot_a_a = torch.sum(re_pv_to_central_transformed_a.unsqueeze(-2) * re_pv_to_central_transformed_a.unsqueeze(-3), dim=-1)
                    dot_b_b = torch.sum(re_pv_to_central_transformed_b.unsqueeze(-2) * re_pv_to_central_transformed_b.unsqueeze(-3), dim=-1)
                    loss_a_a = (1.0 - dot_a_a).mean(dim=-1).mean(dim=-1)
                    loss_b_b = (1.0 - dot_b_b).mean(dim=-1).mean(dim=-1)
                    clutter_loss = clutter_loss + (loss_a_a + loss_b_b).mean() / 2.0

                    # pv_to_central_transformed_a: bz x M x 3
                    pv_to_canon_transformed_a = cur_slot_a_canon_transformed_points - cur_pv_point.unsqueeze(-2)
                    # avg_slot_axis: bz x 3; dot: bz x M
                    dot_avg_axis_pv_to_canon_a = torch.sum(avg_slot_axis.unsqueeze(-2) * pv_to_canon_transformed_a,
                                                             dim=-1)
                    # re_pv_to_central_transformed_a: bz x M x 3
                    re_pv_to_canon_transformed_a = pv_to_canon_transformed_a - dot_avg_axis_pv_to_canon_a.unsqueeze(
                        -1) * avg_slot_axis.unsqueeze(-2)

                    # pv_to_central_transformed_a: bz x M x 3
                    pv_to_canon_transformed_b = cur_slot_b_canon_transformed_points - cur_pv_point.unsqueeze(-2)
                    # avg_slot_axis: bz x 3; dot: bz x M
                    dot_avg_axis_pv_to_canon_b = torch.sum(avg_slot_axis.unsqueeze(-2) * pv_to_canon_transformed_b,
                                                             dim=-1)
                    # re_pv_to_central_transformed_a: bz x M x 3
                    re_pv_to_canon_transformed_b = pv_to_canon_transformed_b - dot_avg_axis_pv_to_canon_b.unsqueeze(
                        -1) * avg_slot_axis.unsqueeze(-2)
                    # bz x M x 3
                    re_pv_to_canon_transformed_a = re_pv_to_canon_transformed_a / torch.clamp(
                        torch.norm(re_pv_to_canon_transformed_a, dim=2, keepdim=True), min=1e-6)
                    re_pv_to_canon_transformed_b = re_pv_to_canon_transformed_b / torch.clamp(
                        torch.norm(re_pv_to_canon_transformed_b, dim=2, keepdim=True), min=1e-6)
                    # dot_a_a: bz x M x M
                    dot_a_a = torch.sum(
                        re_pv_to_canon_transformed_a.unsqueeze(-2) * re_pv_to_canon_transformed_a.unsqueeze(-3),
                        dim=-1)
                    dot_b_b = torch.sum(
                        re_pv_to_canon_transformed_b.unsqueeze(-2) * re_pv_to_canon_transformed_b.unsqueeze(-3),
                        dim=-1)
                    loss_a_a = (1.0 - dot_a_a).mean(dim=-1).mean(dim=-1)
                    loss_b_b = (1.0 - dot_b_b).mean(dim=-1).mean(dim=-1)
                    clutter_loss = clutter_loss + (loss_a_a + loss_b_b).mean() / 2.0

                # print(f"current clutter_loss: {clutter_loss.item()}")
                # minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + clutter_loss

                # minn_other_slots_base_canon_dist = 0.0

            # slot_canon_pts: bz x n_s x 3 x M --> bz x n_s x M x 3; slot_canon_pts --> bz x n_s x M x 3
            slot_canon_pts = safe_transpose(slot_canon_pts, -1, -2)

            # slot_orbits --> bz x n_s
            # minn_dist_ori_to_recon: bz x n_s x na x N ---> bz x n_s x N
            selected_minn_dist_ori_to_recon = batched_index_select(values=minn_dist_ori_to_recon,
                                                                   indices=slot_orbits.long().unsqueeze(-1),
                                                                   dim=2).squeeze(2)
            # selected_minn_dist_ori_to_recon: bz x N
            selected_minn_dist_ori_to_recon, _ = torch.min(selected_minn_dist_ori_to_recon, dim=1)
            ori_to_recon = torch.sqrt(selected_minn_dist_ori_to_recon).mean(dim=-1).mean()
            # print(f"ori_to_recon, uni L1: {float(ori_to_recon.item())}")
            self.ori_to_recon = ori_to_recon

            if k < cur_kanchor:
                # slot_orbits: bz x n_s
                slot_orbits = batched_index_select(values=topk_anchor_idxes, indices=slot_orbits.unsqueeze(-1),
                                                   dim=2).squeeze(-1)
            # register slot_orbits...
            # print("slot_orbits", slot_orbits) # register slot_orbits...

            selected_anchors = batched_index_select(values=cur_anchors, indices=slot_orbits, dim=0)

            if self.slot_single_mode == 1:
                # self.sel_mode_new = slot_orbits[:, 0] #### Get slot mode new!
                self.sel_mode_new = None # slot_orbits[:, 0] #### Get slot mode new!

            # print(f"check slot_R: slot_R: {slot_R.size()}, slot_T: {slot_T.size()}")
            slot_R = batched_index_select(values=slot_R, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
            slot_T = batched_index_select(values=slot_T, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)

            # print("slot_R_raw", slot_R_raw.size(), "slot_orbits", slot_orbits.size())
            slot_R_raw = batched_index_select(values=slot_R_raw, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2).detach()

            filtered_transformed_pts = transformed_pts * hard_slot_indicators.unsqueeze(-1).unsqueeze(-1)
            # filtered_transformed_pts = transformed_pts  # * hard_slot_indicators.unsqueeze(-1).unsqueeze(-1)

            # expanded_recon_slot_pts: bz x (n_s x M) x 3
            # print("filter:", filtered_transformed_pts.size())
            expanded_recon_slot_pts = filtered_transformed_pts.contiguous().view(bz, self.num_slots * self.recon_part_M, 3).contiguous()
            # down sample
            fps_idx = farthest_point_sampling(expanded_recon_slot_pts, npoints)
            expanded_recon_slot_pts = expanded_recon_slot_pts.contiguous().view(bz * self.num_slots * self.recon_part_M, 3)[
                                      fps_idx, :].contiguous().view(bz, npoints, 3)
            # recon_to_ori_dist: bz x (n_s x M)
            # ori_to_recon_dist: bz x N
            recon_to_ori_dist, ori_to_recon_dist = safe_chamfer_dist_call(expanded_recon_slot_pts,
                                                                          safe_transpose(ori_pts, -1, -2),
                                                                          self.chamfer_dist)
            recon_to_ori_dist = recon_to_ori_dist.mean(-1)
            ori_to_recon_dist = ori_to_recon_dist.mean(-1)

            self.glb_recon_ori_dist = (torch.sqrt(recon_to_ori_dist) + torch.sqrt(ori_to_recon_dist)).mean() * 0.5

            ''' Global reconstruction distance for optimization '''
            if self.slot_single_cd == 1:
                glb_recon_ori_dist = ori_to_recon_dist
            else:
                glb_recon_ori_dist = recon_to_ori_dist + ori_to_recon_dist
            ''' Global reconstruction distance for optimization '''

            # tot_recon_loss = (recon_to_ori_dist + ori_to_recon_dist) * self.glb_recon_factor + (
            #     slot_dist_ori_recon) * self.slot_recon_factor + slot_pv_canon_cd_loss # add slot_pv_canon loss to the tot_recon_loss term

            tot_recon_loss = glb_recon_ori_dist * self.glb_recon_factor + (
                slot_dist_ori_recon) * self.slot_recon_factor + slot_pv_canon_cd_loss  # add slot_pv_canon loss to the tot_recon_loss term
            if self.recon_prior == 6 or self.recon_prior == 7:
                tot_recon_loss = tot_recon_loss + minn_other_slots_base_canon_dist * 0.2

            ''' Add cuboic constraint loss '''
            if self.recon_prior == 5 or self.recon_prior == 7:
                # slot_recon_cuboic_constraint_loss = torch.sum(slot_recon_cuboic_constraint_loss * hard_slot_indicators, dim=-1) / torch.sum(hard_slot_indicators, dim=-1)
                # tot_recon_loss = tot_recon_loss + slot_recon_cuboic_constraint_loss
                # Get cuboid reconstruction points
                # slot_cuboic_recon_pts: bz x n_s x 3
                # recon_prior...
                slot_cuboic_recon_pts = torch.cat(slot_cuboic_recon_pts, dim=1)

                slot_cuboic_R = torch.cat(slot_cuboic_R, dim=1)

                # slot_cuboic_recon_pts[:, 1:, 1][slot_cuboic_recon_pts[:, 1:, 1] > 0.30] = 0.30
                ''' Get cuboid reconstruction: whether to consider normals in the calculation process or not '''
                # if normals is not None:
                #     slot_recon_cuboic_constraint_loss = get_cuboic_constraint_loss_with_normals(
                #         slot_R, slot_T, ori_pts, normals, slot_cuboic_recon_pts, slot_cuboic_R, hard_one_hot_labels, attn_ori
                #     )
                # else:
                #     slot_recon_cuboic_constraint_loss = get_cuboic_constraint_loss(
                #         slot_R, slot_T, ori_pts, slot_cuboic_recon_pts, slot_cuboic_R, hard_one_hot_labels, attn_ori
                #     )
                ''' Get cuboid reconstruction: whether to consider normals in the calculation process or not '''

                forb_slot_idx = None #
                # forb_slot_idx = 0 # Add cuboid constraint for predicted points coordinates
                slot_recon_cuboic_constraint_loss = get_cuboic_constraint_loss(
                    slot_R, slot_T, ori_pts, slot_cuboic_recon_pts, slot_cuboic_R, hard_one_hot_labels, attn_ori, forb_slot_idx=forb_slot_idx
                )

                ''' Get cuboid reconstruction loss: chamfer distance based loss '''
                # slot_recon_cuboic_constraint_loss = get_cuboic_constraint_loss_cd_based(
                #     slot_R, slot_T, ori_pts, slot_cuboic_recon_pts, slot_cuboic_R, hard_one_hot_labels, attn_ori
                # )
                ''' Get cuboid reconstruction loss: chamfer distance based loss '''

                # Get total reconstruction loss + cuboid constraint loss
                # tot_recon_loss = tot_recon_loss + 10.0 * slot_recon_cuboic_constraint_loss
                tot_recon_loss = tot_recon_loss + 10.0 * slot_recon_cuboic_constraint_loss
                # tot_recon_loss = tot_recon_loss + 5.0 * slot_recon_cuboic_constraint_loss

            tot_recon_loss = tot_recon_loss.mean()

            out_feats = {}

            labels = torch.argmax(attn_ori, dim=1)

            selected_point_labels = labels

            selected_recon_slot_pts = transformed_pts

            ori_selected_recon_slot_pts = slot_canon_pts

            selected_expanded_sampled_recon_pts = expanded_recon_slot_pts
            selected_attn = hard_one_hot_labels  # one-hot labels actually
            pure_ori_x = safe_transpose(ori_pts, -1, -2)

            out_feats['vis_pts_hard'] = pure_ori_x.detach().cpu().numpy()
            out_feats['vis_labels_hard'] = selected_point_labels.detach().cpu().numpy()
            out_feats['ori_recon_slot_pts_hard'] = ori_selected_recon_slot_pts.detach().cpu().numpy()

            out_feats['recon_slot_pts_hard'] = selected_recon_slot_pts.detach().cpu().numpy()
            # out_feats['category_common_slot'] = category_pts.detach().cpu().numpy()
            out_feats['sampled_recon_pts_hard'] = selected_expanded_sampled_recon_pts.detach().cpu().numpy()

            if self.recon_prior == 6 or self.recon_prior == 7:
                out_feats['recon_slot_pts_hard_wo_glb_rot'] = transformed_pts_ori.detach().cpu().numpy()
                out_feats['pv_pts_hard_wo_glb_rot'] = transformed_pv_points_ori.detach().cpu().numpy()
                out_feats['selected_anchors'] = selected_anchors.detach().cpu().numpy()


            if self.recon_prior == 5 or self.recon_prior == 7:
                ##### Register predicted cuboid boundary points for slots #####
                out_feats['slot_cuboic_recon_pts'] = slot_cuboic_recon_pts.detach().cpu().numpy()
                ##### Register predicted cuboid rotation matrix for slots #####
                out_feats['slot_cuboic_R'] = slot_cuboic_R.detach().cpu().numpy()
            elif self.recon_prior == 6 or self.recon_prior == 7: # slot pivot points --- for pivot points and others
                out_feats['slot_pivot_points'] = slot_pivot_points.detach().cpu().numpy()
                out_feats['pair_pivot_points'] = pair_pivot_points.detach().cpu().numpy()
                # central_points: bz x n_s x 3
                out_feats['slot_central_points'] = central_points.detach().cpu().numpy()
                out_feats['slot_axis'] = slot_axis.detach().cpu().numpy()
                real_defined_axises = torch.matmul(selected_anchors, defined_axises.unsqueeze(-1)).squeeze(-1)
                out_feats['defined_axises'] = defined_axises.detach().cpu().numpy()  # defined_axises:
                real_defined_axises = real_defined_axises[:, 0, :]
                out_feats['real_defined_axises'] = real_defined_axises.detach().cpu().numpy()  # defined_axises:
                self.real_defined_axises = real_defined_axises.clone()

            # print("here afater defined axises...")

            out_feats['attn'] = hard_one_hot_labels.detach().cpu().numpy()

            if cur_iter == 0:
                self.attn_iter_0 = safe_transpose(hard_one_hot_labels, -1, -2)  # .cpu().numpy()
                # self.attn_saved = attn_ori # safe_transpose(hard_one_hot_labels, -1,
                #     -2)  # .contiguous().transpose(1, 2).contiguous().detach()
                self.attn_saved = attn_ori
            elif cur_iter == 1:
                self.attn_iter_1 = safe_transpose(hard_one_hot_labels, -1,
                                                  -2)  # .contiguous().transpose(1, 2).contiguous().detach()  # .cpu().numpy()
                # self.attn_saved_1 = safe_transpose(hard_one_hot_labels, -1,
                #                                    -2)  # .contiguous().transpose(1, 2).contiguous().detach()
                # self.attn_saved_1 = attn_ori
                self.attn_saved_1 = attn_ori

            # self.attn_saved = attn_ori
            # self.attn_saved = safe_transpose(hard_one_hot_labels, -1, -2)
            # self.attn_saved_1 = attn_ori

            # bz x 3 x 4
            # glb_pose = self.glb_pose
            # glb_rot, glb_trans = glb_pose[:, :3, :3], glb_pose[:, :3, 3]
            ''' Predict rotations '''
            # selected_labels: bz x N
            selected_labels = torch.argmax(selected_attn, dim=-1)

            selected_pred_R = slot_R

            selected_pred_R_saved = selected_pred_R.detach().clone()
            # selected_pred_R: bz x N x 3 x 3
            selected_pred_R = batched_index_select(values=selected_pred_R, indices=selected_labels, dim=1)
            # selected_pred_R = selected_pred_R.squeeze(2).squeeze(2)

            # bz x n_s x 3 x 3
            out_feats['pred_slot_Rs'] = selected_pred_R.detach().cpu().numpy()
            out_feats['slot_R_raw'] = slot_R_raw.detach().cpu().numpy()

            # print("selected_slot_oribt", selected_slot_oribt.size())

            selected_pred_T = slot_T
            selected_pred_T_saved = selected_pred_T.detach().clone()
            # selected_pred_R: bz x N x 3 x 3
            selected_pred_T = batched_index_select(values=selected_pred_T, indices=selected_labels, dim=1)

            # selected_pred_pose: bz x N x 3 x 4
            selected_pred_pose = torch.cat([selected_pred_R, selected_pred_T.unsqueeze(-1)], dim=-1)
            # selected_pred_pose: bz x N x 4 x 4
            selected_pred_pose = torch.cat([selected_pred_pose, torch.zeros((bz, npoints, 1, 4), dtype=torch.float32).cuda()], dim=2)

            selected_pred_pose = pose

            # selected_pts_orbit = batched_index_select(values=selected_slot_oribt, indices=selected_labels, dim=1)

            out_feats['pred_slot_Ts'] = selected_pred_T.detach().cpu().numpy()

            # selected_inv_pred_R: bz x N x 3 x 3; ori_pts: bz x N x 3
            selected_inv_pred_R = selected_pred_R.contiguous().transpose(-1, -2).contiguous()
            # rotated_ori_pts = torch.matmul(selected_pred_R, ori_pts.contiguous().unsqueeze(-1).contiguous()).squeeze(-1).contiguous()
            # From transformed points to original canonical points
            transformed_ori_pts = torch.matmul(selected_inv_pred_R,
                                               (safe_transpose(ori_pts, -1, -2) - selected_pred_T).unsqueeze(-1)).squeeze(-1)
            # transformed_ori_pts = torch.matmul(selected_inv_pred_R, (rotated_ori_pts - selected_pred_T).unsqueeze(-1)).squeeze(-1)

            # transformed ori pts
            out_feats['transformed_ori_pts'] = transformed_ori_pts.detach().cpu().numpy()

            out_feats['pv_points'] = pv_points.detach().cpu().numpy()

            if gt_pose is not None:
                gt_R = gt_pose[..., :3, :3]
                gt_T = gt_pose[..., :3, 3]
                gt_inv_R = gt_R.contiguous().transpose(-1, -2).contiguous()
                gt_transformed_ori_pts = torch.matmul(gt_inv_R, (safe_transpose(ori_pts, -1, -2) - gt_T).unsqueeze(-1)).squeeze(-1)
                out_feats['gt_transformed_ori_pts'] = gt_transformed_ori_pts.detach().cpu().numpy()

            #### Remember global reconstruction related information ####
            # out_feats['glb_canon_pred'] = self.glb_canon_pred
            # out_feats['glb_transformed_pred'] = self.glb_transformed_pred
            # out_feats['glb_ori_x'] = self.glb_ori_x

            np.save(self.log_fn + f"_n_stage_{self.stage}_iter_{cur_iter}.npy", out_feats)

            self.pred_R = selected_pred_R_saved
            self.pred_T = selected_pred_T_saved
            self.defined_axises = defined_axises.clone()
            self.offset_pivot_points = offset_pivot_points.clone()

            # if cur_iter > 0:
            #     selected_pred_R_saved = torch.matmul(pred_glb_R.unsqueeze(1), selected_pred_R).detach()
            #     selected_pred_T_saved = (torch.matmul(pred_glb_R.unsqueeze(1), selected_pred_T.unsqueeze(-1)).squeeze(-1) + pred_glb_T).detach()

            out_feats['pred_R_slots'] = selected_pred_R_saved.cpu().numpy()
            out_feats['pred_T_slots'] = selected_pred_T_saved.cpu().numpy()
            out_feats['pv_points'] = pv_points.detach().cpu().numpy()

            self.out_feats = out_feats

            tot_loss = tot_recon_loss  # + (pts_ov_max_percent_loss) * 4.0 # encourage entropy

            return tot_loss, selected_pred_pose

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
        # print("Similartiy with gt_rot", torch.mean(traces).item())
        return torch.mean(traces).item()

    def forward(self, x, pose, ori_pc=None, rlabel=None, nn_inter=2, pose_segs=None, canon_pc=None, normals=None, canon_normals=None):

        # loss, attn = self.forward_one_iter(x, pose, rlabel=rlabel)
        # return loss, attn
        bz, np = x.size(0), x.size(2)
        init_pose = torch.zeros([bz, np, 4, 4], dtype=torch.float32).cuda()
        init_pose[..., 0, 0] = 1.;
        init_pose[..., 1, 1] = 1.;
        init_pose[..., 2, 2] = 1.
        # init_pose = pose
        tot_loss = 0.0
        loss = 0.0
        cur_transformed_points = x
        cur_estimated_pose = init_pose
        # nn_inter = 1
        nn_inter = self.num_iters
        # cur_estimated_pose = pose
        out_feats_all_iters = {}
        cur_selected_pts_orbit = None

        # tot_loss, selected_attn, pred_pose, out_feats, selected_pts_orbit = self.forward_one_iter(cur_transformed_points, cur_estimated_pose, ori_pc=ori_pc, rlabel=rlabel, cur_iter=0, gt_pose=pose, gt_pose_segs=pose_segs, canon_pc=canon_pc, selected_pts_orbit=cur_selected_pts_orbit)

        torch.cuda.empty_cache()

        cur_gt_pose = pose

        if self.stage == 0:
            loss = self.forward_one_iter(
                    cur_transformed_points, cur_estimated_pose, ori_pc=ori_pc, rlabel=rlabel, cur_iter=0, gt_pose=cur_gt_pose, gt_pose_segs=pose_segs, canon_pc=canon_pc, selected_pts_orbit=cur_selected_pts_orbit, normals=normals, canon_normals=canon_normals)
        else:
            for i_iter in range(self.num_iters):
                cur_loss, cur_estimated_pose = self.forward_one_iter(
                    cur_transformed_points, cur_estimated_pose, ori_pc=ori_pc, rlabel=rlabel, cur_iter=i_iter, gt_pose=pose, gt_pose_segs=pose_segs, canon_pc=canon_pc, selected_pts_orbit=cur_selected_pts_orbit)
                loss += cur_loss
            loss = loss / self.num_iters

        torch.cuda.empty_cache()
        return loss

    def get_anchor(self):
        # return self.backbone[-1].get_anchor()
        return self.glb_backbone[-1].get_anchor()


# Full Version
def build_model(opt,
                mlps=[[64, 64], [128, 128], [256, 256], [256]],
                out_mlps=[256],
                strides=[2, 2, 2, 2],
                initial_radius_ratio=0.2,
                sampling_ratio=0.4,
                sampling_density=0.5,
                kernel_density=1,
                kernel_multiplier=2,
                input_radius=1.0,
                sigma_ratio=0.5,  # 0.1
                xyz_pooling=None,
                so3_pooling="max",
                to_file=None):
    # mlps[-1][-1] = 256 # 512
    # out_mlps[-1] = 256 # 512

    if opt.model.kanchor < 60:
        mlps = [[64, 128], [256], [512], [1024]]
        # mlps = [[64,128], [256], [512]]
        # mlps = [[64], [128], [512]]
        # mlps = [[1024]]
        out_mlps = [1024]
        # out_mlps = [512]
    else:
        mlps = [[64], [128], [512]]
        # mlps = [[64], [128], [256]] # you need to
        # out_mlps = [512]
        out_mlps = [256]
        # mlps = [[32, 32], [64, 64], [128, 128], [256, 256]]
        # out_mlps = [128, 128]

    # initial_radius_ratio = 0.05
    # initial_radius_ratio = 0.15
    # initial_radius_ratio = 0.20
    # initial_radius_ratio = 0.20
    initial_radius_ratio = opt.equi_settings.init_radius
    input_radius = 0.4
    print(f"Using initial radius: {initial_radius_ratio}")
    device = opt.device
    # print(f"opt.device: {device}")
    input_num = opt.model.input_num  # 1024
    dropout_rate = opt.model.dropout_rate  # default setting: 0.0
    # temperature
    temperature = opt.train_loss.temperature  # set temperature
    so3_pooling = 'attention'  # opt.model.flag # model flag
    # opt.model.kpconv = 1
    na = 1 if opt.model.kpconv else opt.model.kanchor  # how to represent rotation possibilities? --- sampling from the sphere ---- points!
    na = opt.model.kanchor  # how to represent rotation possibilities? --- sampling from the sphere ---- points!
    # nmasks = opt.train_lr.nmasks
    nmasks = opt.nmasks
    num_iters = opt.equi_settings.num_iters
    kpconv_kanchor = opt.equi_settings.kpconv_kanchor
    permute_modes = opt.equi_settings.permute_modes
    # nmasks = 2
    sampling_ratio = 0.8
    sampling_density = 0.5
    kernel_multiplier = 2
    sigma_ratio = 0.5
    xyz_pooling = None

    strides = [2, 2, 2, 2]

    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    params = {'name': 'Invariant ZPConv Model',
              'backbone': [],
              'kpconv_backbone': [],
              'na': na
              }

    dim_in = 1
    # dim_in = 3

    # process args
    n_layer = len(mlps)
    stride_current = 1  # stride_current_--
    stride_multipliers = [stride_current]
    # for i in range(n_layer):
    #     stride_current *= 2 # strides[i]
    #     stride_multipliers += [stride_current]
    # todo: use ohter strides? --- possible other choices?
    for i in range(n_layer):
        stride_current *= strides[i]
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]
    # radius ratio should increase as the stride increases to sample more reasonable points
    radius_ratio = [initial_radius_ratio * multiplier ** sampling_density for multiplier in stride_multipliers]
    # radius_ratio = [0.25, 0.5]
    # set radius for each layer
    radii = [r * input_radius for r in radius_ratio]
    # Compute sigma
    # weighted_sigma = [sigma_ratio * radii[i]**2 * stride_multipliers[i] for i in range(n_layer + 1)]
    # sigma for radius and points
    weighted_sigma = [sigma_ratio * radii[0] ** 2]

    for idx, s in enumerate(strides):
        # weighted_sigma.append(weighted_sigma[idx] * 2)
        weighted_sigma.append(weighted_sigma[idx] * s)  #

    for i, block in enumerate(mlps):
        block_param = []
        kpconv_block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0
            stride_conv = i == 0 or xyz_pooling != 'stride'
            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.

            # neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
            neighbor = 32  # int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
            # neighbor = 16 # int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
            # if i==0 and j==0:
            #    neighbor *= int(input_num/1024)

            kernel_size = 1
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i]
                nidx = i if i == 0 else i + 1
                if stride_conv:
                    neighbor *= 2  # = 2 * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                    kernel_size = 1  # if inter_stride < 4 else 3
            else:
                inter_stride = 1
                nidx = i + 1

            inter_stride = 1

            print(f"At block {i}, layer {j}!")
            print(f'neighbor: {neighbor}')
            print(f'stride: {inter_stride}')
            sigma_to_print = weighted_sigma[nidx] ** 2 / 3
            print(f'sigma: {sigma_to_print}')
            print(f'radius ratio: {radius_ratio[nidx]}')

            print(f"current nei: {neighbor}, radius: {radii[nidx]}, stride: {inter_stride}")

            # one-inter one-intra policy
            block_type = 'inter_block' if na < 60 else 'separable_block'  # point-conv and group-conv separable conv
            # block_type = 'inter_block' if na<2 else 'separable_block' # point-conv and group-conv separable conv
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
                    'pooling': xyz_pooling,
                    'kanchor': na,
                    'norm': 'BatchNorm2d',
                    # 'norm': None,
                }
            }

            kpconv_conv_param = {
                'type': 'inter_block',
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
                    'pooling': xyz_pooling,
                    'kanchor': kpconv_kanchor, ### set kanchor to 1!!
                    'norm': 'BatchNorm2d',
                    'permute_modes': permute_modes
                    # 'norm': None,
                }
            }
            block_param.append(conv_param)
            kpconv_block_param.append(kpconv_conv_param)
            dim_in = dim_out

        params['backbone'].append(block_param)
        params['kpconv_backbone'].append(kpconv_block_param)

    # kernels here are defined as kernel points --- explicit kernels --- each with a [dim_in, dim_out] weight matrix
    params['outblock'] = {
        'dim_in': dim_in,
        'mlp': out_mlps,
        'fc': [64],  #
        'k': nmasks,  # 40,
        'pooling': so3_pooling,
        'temperature': temperature,
        'kanchor': na,
    }

    params['general'] = {
        'num_iters': num_iters,
        'global_rot': opt.equi_settings.global_rot,
        'npoints': opt.model.input_num,
        'batch_size': opt.batch_size,
        'init_lr': opt.train_lr.init_lr,
        'part_pred_npoints': opt.equi_settings.part_pred_npoints,
        'use_equi': opt.equi_settings.use_equi,
        'model_type': opt.equi_settings.model_type,
        'decoder_type': opt.equi_settings.decoder_type,
        'inv_attn': opt.equi_settings.inv_attn,
        'topk': opt.equi_settings.topk,
        'orbit_attn': opt.equi_settings.orbit_attn,
        'slot_iters': opt.equi_settings.slot_iters,
        'rot_factor': opt.equi_settings.rot_factor,
        'translation': opt.equi_settings.translation,
        'gt_oracle_seg': opt.equi_settings.gt_oracle_seg,
        'gt_trans': opt.equi_settings.gt_oracle_trans,
        'feat_pooling': opt.equi_settings.feat_pooling,
        'cent_trans': opt.equi_settings.cent_trans,
        'soft_attn': opt.equi_settings.soft_attn,
        'recon_prior': opt.equi_settings.recon_prior,
        'shape_type': opt.equi_settings.shape_type,
        'factor': opt.equi_settings.factor,
        'queue_len': opt.equi_settings.queue_len,
        'glb_recon_factor': opt.equi_settings.glb_recon_factor,
        'slot_recon_factor': opt.equi_settings.slot_recon_factor,
        'use_sigmoid': opt.equi_settings.use_sigmoid,
        'use_flow_reg': opt.equi_settings.use_flow_reg,
        'use_axis_queue': opt.equi_settings.use_axis_queue,
        'run_mode': opt.mode,
        'exp_indicator': opt.equi_settings.exp_indicator,
        'cur_stage': opt.equi_settings.cur_stage,
        'kpconv_kanchor': opt.equi_settings.kpconv_kanchor,
        'sel_mode': opt.equi_settings.sel_mode,
        'sel_mode_trans': opt.equi_settings.sel_mode_trans,
        'slot_single_mode': opt.equi_settings.slot_single_mode,
        'rot_angle_factor': opt.equi_settings.rot_angle_factor,
        'pred_axis': opt.equi_settings.pred_axis,
        'pred_pv_equiv': opt.equi_settings.pred_pv_equiv,
        'mtx_based_axis_regression': opt.equi_settings.mtx_based_axis_regression,
        'with_part_proposal': opt.equi_settings.with_part_proposal,
        'glb_single_cd': opt.equi_settings.glb_single_cd,
        'slot_single_cd': opt.equi_settings.slot_single_cd,
        # 'opt': opt

    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model


def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
