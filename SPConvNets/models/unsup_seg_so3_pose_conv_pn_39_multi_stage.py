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
# from SPConvNets.utils.slot_attention import SlotAttention
# from SPConvNets.utils.slot_attention_spec import SlotAttention
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

# for drawers only


class ClsSO3ConvModel(nn.Module):  # SO(3) equi-conv-network #
    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()

        ''' Construct backbone model '''
        # get backbone model
        # self.backbone = nn.ModuleList()
        # for block_param in params['backbone']:  # backbone
        #     self.backbone.append(M.BasicSO3PoseConvBlock(block_param))
        # print(f"number of convs in the backbone: {len(self.backbone)}")
        # self.outblock = M.ClsOutBlockR(params['outblock'])
        # output classification block

        ''' EPN for global rotation factorization '''
        self.glb_backbone = nn.ModuleList()
        # use equi as the backbone for global pose factorization
        for block_param in params['backbone']:
            self.glb_backbone.append(M.BasicSO3PoseConvBlock(block_param))
        # for block_param in params['backbone']:
        #     self.glb_backbone.append(Mso3.BasicSO3ConvBlock(block_param))
        ''' PointNet for feature aggregation '''  # global features
        # self.glb_outblock = Mso3.ClsOutBlockPointnet(params['outblock'])
        # self.glb_outblock = Mso3.InvOutBlockOurs(params['outblock'], norm=1, pooling_method='max')

        self.backbone = nn.ModuleList()
        for block_param in params['kpconv_backbone']:
            # for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3PoseConvBlock(block_param))

        self.backbone_sec = nn.ModuleList()
        for block_param in params['kpconv_backbone']:
            # for block_param in params['backbone']:
            self.backbone_sec.append(M.BasicSO3PoseConvBlock(block_param))

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
        self.axis_reg_stra = params['general']['axis_reg_stra']
        self.glb_single_cd = params['general']['glb_single_cd']
        self.slot_single_cd = params['general']['slot_single_cd']
        self.rel_for_points = params['general']['rel_for_points']  # --rel-for-points
        self.use_art_mode = params['general']['use_art_mode']  # --rel-for-points

        self.sel_mode_trans = None if self.sel_mode_trans == -1 else self.sel_mode_trans

        self.local_rank = int(os.environ['LOCAL_RANK'])

        #### Set parameter alias ####
        self.recon_part_M = self.part_pred_npoints  # 128, 256, 512, 1024
        self.transformation_dim = 7

        self.stage = params['general']['cur_stage']

        # pts_to_real_pts = {128: 146, 256: 258, 512: 578}

        # if self.recon_prior in [2, 3]:
        #     self.sphere_recon_M = pts_to_real_pts[self.recon_part_M]
        #     self.recon_part_M = self.sphere_recon_M

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
        if self.gt_oracle_seg == 0:  # not a GT seg...
            ''' Construct slot-attention module '''
            #### ??
            orbit_attn_three_in_dim = self.inv_out_dim + 3
            # orbit_attn_three_in_dim = self.inv_out_dim + 5
            # orbit_attn_three_in_dim = 3
            # attention in feature dim...
            self.attn_in_dim = (self.inv_out_dim + self.kanchor) if self.orbit_attn == 1 else (
                self.kanchor) if self.orbit_attn == 2 else (orbit_attn_three_in_dim) if self.orbit_attn == 3 else (
                self.inv_out_dim)
            # inv_pooling_method = 'max' if self.recon_prior not in [0, 2] else 'attention'
            inv_pooling_method = 'attention'

            self.inv_pooling_method = inv_pooling_method
            self.sel_mode = None if self.sel_mode == -1 else self.sel_mode
            self.inv_pooling_method = self.inv_pooling_method if self.sel_mode is None else 'sel_mode'
            self.ppint_outblk = Mso3.InvPPOutBlockOurs(params['outblock'], norm=1, pooling_method=inv_pooling_method,
                                                       sel_mode=self.sel_mode)
            # whether to use slot attention module
            # slot attention;
            self.slot_attention = SlotAttention(num_slots=params['outblock']['k'],
                                                dim=self.attn_in_dim, hidden_dim=self.inv_out_dim,
                                                iters=self.slot_iters)

        ''' Construct whole shape output block '''
        use_abs_pos = False
        # use_abs_pos = True
        # whole shape output block; for pv-points decoding...
        # whole shape outblock
        self.whole_shp_outblock = Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention',
                                                               use_pointnet=True, use_abs_pos=use_abs_pos)
        ''' Construct pv points decoding block '''  # decode pv points
        self.pv_points_decoding_blk = DecoderFC([256, 256], params['outblock']['mlp'][-1], self.num_slots - 1, None)

        use_abs_pos = True  # if self.shape_type not in ['washing_machine'] else False
        # use_abs_pos = False
        ''' Construct inv-feat output block for slots '''
        # modify this process to incorperate masks to the pooling and other calculation process
        # slot_outblock: invariant feature output block for each slot
        self.slot_outblock = nn.ModuleList()
        self.abs_slot_outblock = nn.ModuleList()
        for i_s in range(self.num_slots):
            # we should not use the pooled features directly since weights for different orbits should be determined by all points in the slot
            # slot invariant feature output block
            # use_abs_pos: whether to use absolute points to decode positions
            # we should set it to False ----
            ''' Use absolute coordinates for both shape decoding and pv-point/central point decoding '''
            return_point_pooling_feature = True if self.pred_pv_equiv else False
            use_abs_pos = use_abs_pos
            ''' Use relative coordinates for shape decoding '''
            # return_point_pooling_feature = False
            use_abs_pos = False if self.rel_for_points == 1 else True  # set to 1 if we use relative coordinates for canonical points decoding
            # whether to return point pooling features
            self.slot_outblock.append(
                # invariant feature
                Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention', use_pointnet=True,
                                             use_abs_pos=use_abs_pos,
                                             return_point_pooling_feature=return_point_pooling_feature)
                # whether to use abs pos
            )

            self.abs_slot_outblock.append(
                # invariant feature
                Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention', use_pointnet=True,
                                             use_abs_pos=True,
                                             return_point_pooling_feature=return_point_pooling_feature)
                # whether to use abs pos
            )

        ''' Construct inv-feat output block for slots '''
        # For pv-point prediction
        use_abs_pos_pair_out = True
        # use_abs_pos_pair_out = False
        self.zz_pred_pv_equiv = True
        return_point_pooling_feature_pair_out = True if self.zz_pred_pv_equiv else False
        self.pair_slot_outblock = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.pair_slot_outblock.append(
                Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention', use_pointnet=True,
                                             use_abs_pos=use_abs_pos_pair_out,
                                             return_point_pooling_feature=return_point_pooling_feature_pair_out)
            )

        ''' Construct inv-feat output block for the whole shape '''
        ### the difference is that we should set `mask` to None for each forward pass ###
        #
        ''' Output block for global invariant feature '''  #
        self.glb_outblock = Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention',
                                                         use_pointnet=True)

        ''' Construct reconstruction branches for slots, input features should be those inv-feats output from inv-feats extraction branches '''
        self.slot_shp_recon_net = nn.ModuleList()
        for i_s in range(self.num_slots):
            if self.recon_prior == 4:
                # we use 3-dim atlas-prior-dim for the category encoder's prior
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
                                        pred_rot=False)
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
                    DecoderFCWithPVPCuboic([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None,
                                           pred_rot=False)
                )
            elif self.recon_prior == 8:
                self.slot_shp_recon_net.append(
                    DecoderFCWithPVPAtlas([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None,
                                          prior_dim=3)
                    # DecoderFCWithPVPConstantCommon([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None, prior_dim=3)
                )
            elif self.recon_prior == 9:
                self.slot_shp_recon_net.append(
                    DecoderFCWithPVPConstantCommon([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None,
                                                   prior_dim=3)
                )
            else:
                self.slot_shp_recon_net.append(
                    DecoderFC([256, 256], params['outblock']['mlp'][-1], self.recon_part_M, None)
                )
        self.pair_slot_shp_recon_net = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.pair_slot_shp_recon_net.append(
                DecoderFCWithPVP([256, 256], params['outblock']['mlp'][-1], 2, None)
            )

        self.glb_recon_npoints = 512
        # self.glb_recon_npoints = 1024
        self.glb_recon_npoints = self.npoints

        ''' Construct reconstruction branch for the whole shape '''
        #### global reconstruction net ####
        # global
        self.glb_shp_recon_net = DecoderFC(
            [256, 256], params['outblock']['mlp'][-1], self.glb_recon_npoints, None
        )
        #### axis prediction net ####
        # todo: reguralizations for canonical shape axis prediction?
        # # axis
        # self.glb_axis_pred_net = DecoderFCAxis(
        #     [256, 256], params['outblock']['mlp'][-1], None
        # )

        ''' Construct transformation branches for slots '''
        # todo: how translations are predicted?
        self.slot_trans_outblk_RT = nn.ModuleList()
        self.pair_slot_trans_outblk_RT = nn.ModuleList()
        # self.pred_t = False
        self.pred_t = True
        self.r_representation = 'quat'  # aaa
        self.r_representation = 'angle'
        # if we decompose a central point from a part shape's invariant feature?
        # a part shape's invariant
        print(f"self.shape_type: {self.shape_type}", "1")
        for i_s in range(self.num_slots):
            pred_pv_points = True if self.pred_pv_equiv else False
            pred_pv_points_in_dim = params['outblock']['mlp'][-1]
            pred_central_points = True  # in this version --> we predict central points and pv points from equiv features
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
                cur_slot_trans_outblk = SO3OutBlockRTWithMask if self.shape_type != 'drawer' else SO3OutBlockRTWithAxisWithMask
                print(f"self.shape_type: {self.shape_type}")
                cur_slot_trans_outblk = SO3OutBlockRTWithAxisWithMask
                # cur_slot_trans_outblk = SO3OutBlockRTWithMaskSep  # output block for transformation...
                # c_in_rot = self.inv_out_dim #### If we use inv features for angle decoding
                c_in_rot = self.encoded_feat_dim
                c_in_trans = self.encoded_feat_dim
                self.slot_trans_outblk_RT.append(
                    cur_slot_trans_outblk(params['outblock'], norm=1, pooling_method='max',
                                          global_scalar=True,
                                          # global scalar?
                                          use_anchors=False,
                                          feat_mode_num=self.kanchor, num_heads=1, representation=cur_r_representation)
                )
                # self.slot_trans_outblk_RT.append(
                #     cur_slot_trans_outblk(params['outblock'], norm=1, pooling_method='max',
                #                           global_scalar=True,
                #                           # global scalar?
                #                           use_anchors=False,
                #                           feat_mode_num=self.kanchor, num_heads=1, representation=cur_r_representation,
                #                           c_in_rot=c_in_rot, c_in_trans=c_in_trans, pred_axis=self.pred_axis,
                #                           # whehter to predict axis
                #                           pred_pv_points=pred_pv_points, pv_points_in_dim=pred_pv_points_in_dim,
                #                           # whether to predict pv point --> only useful when we use equiv feature for pivot point prediction
                #                           pred_central_points=pred_central_points,
                #                           # whether to predict central points --> only useful when we use equiv feature for central point prediction
                #                           central_points_in_dim=pred_central_points_in_dim,  #
                #                           mtx_based_axis_regression=self.mtx_based_axis_regression)
                #     # how to predict axis...
                # )
                # self.pair_slot_trans_outblk_RT.append(
                #     cur_slot_trans_outblk(params['outblock'], norm=1, pooling_method='max',
                #                           global_scalar=True,  # whether to use
                #                           # global scalar?
                #                           use_anchors=False,
                #                           feat_mode_num=self.kanchor, num_heads=1, representation=cur_r_representation,
                #                           c_in_rot=c_in_rot, c_in_trans=c_in_trans, pred_axis=self.pred_axis,
                #                           pred_pv_points=pred_pv_points, pv_points_in_dim=pred_pv_points_in_dim,
                #                           pred_central_points=pred_central_points,
                #                           central_points_in_dim=pred_central_points_in_dim)
                # )

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

        ''' For different axis-reg-stra '''
        if self.axis_reg_stra == 1:  # running mean
            n_joints = self.num_slots - 1  # we use the same direction for all joints in one shape
            self.joint_dir_running_mean = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
            self.momentum = 0.99
            self.pv_in_count_nn = 0
            # avg_pv_point_distance? --> update by (pv_in_count_nn * avg_distance + sum_of_valid_distance) / (in_count_nn + valid_nn)
            self.avg_pv_point_distance = torch.zeros((n_joints,), dtype=torch.float32).cuda()

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
                print(f"slot_a: {i_s_a}, slot_b: {i_s_b}, loss: {dot_axises_loss_cur_slot_pair.mean().item()}")
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

            ''''''
            ## Use predicted as deformation flow ##
            # cur_slot_predicted_flow = cur_slot_predicted_flow * 2.0 # * 0.2
            # cur_slot_category_pts = safe_transpose(cur_slot_category_pts, 1, 2)
            # cur_slot_deformed_pts = cur_slot_category_pts + cur_slot_predicted_flow
            ## Use predicted as deformation flow ##

            ''''''
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
            print("???")
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
        print(f"current axis prior: {self.axis_prior_slot_pairs.data}")
        print(f"mean of oribt mask: {torch.mean(orbit_mask).item()}")
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
        print(
            f"Current queue size: {cur_queue.size()}, queue len: {self.queue_len}, queue total len: {self.queue_tot_len}")
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
        print(f"avg_selected_dots: {avg_selected_dots}")

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
                print(self.axis_prior_slot_pairs.data)
                # # if self.local_rank == 0:
                np.save(f"axis_prior_{self.local_rank}.npy", self.axis_prior_slot_pairs.data.detach().cpu().numpy())
                pass
                ''' Save slot pair axis prior --- for single shape selection '''

        return res_selected_loss, res_selected_anchors

    ''' Select orbits via soem other strategies '''

    def select_slot_orbits(self, slot_recon_loss, slot_pred_rots):
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
        print(f"avg_selected_dots: {avg_selected_dots}")

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
                print(self.axis_prior_slot_pairs.data)
                # if self.local_rank == 0:
                np.save(f"axis_prior_{self.local_rank}.npy", self.axis_prior_slot_pairs.data.detach().cpu().numpy())
                ''' Save slot pair axis prior --- for single shape selection '''

        return res_selected_loss, res_selected_anchors

    ''' Calcuate axises for rotation matrices '''

    def from_rotation_mtx_to_axis(self, rots):
        ori_rots_size = rots.size()[:-1]  # number of rotation matrices
        exp_rots = rots.contiguous().view(-1, 3, 3).contiguous()
        exp_axises = []

        for i in range(exp_rots.size(0)):
            cur_rot = exp_rots[i]
            trace = cur_rot[0, 0] + cur_rot[1, 1] + cur_rot[2, 2]  # current rotation
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

    def forward_one_iter(self, x, pose, x_list=None, hard_label=None, ori_pc=None, rlabel=None, cur_iter=0, gt_pose=None, gt_pose_segs=None, canon_pc=None, selected_pts_orbit=None, normals=None, canon_normals=None):  # rotation label
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
            cur_anchors = self.anchors  # if cur_iter == 0 else self.kpconv_anchors

            # if x_list is None:
            #     x = M.preprocess_input(x, cur_kanchor, pose, False)
            #     # pose: bz x N x 4 x 4
            #     cur_backbone = self.backbone # get backbone...
            #     # x.feats: bz x N x dim x 1 --> na = 1 now for kpconv net; process input point coordinates
            #     for block_i, block in enumerate(cur_backbone):
            #         x = block(x)
            #     torch.cuda.empty_cache()
            # else:
            #     # here, `pose` should be the identity pose matrix --- we have projected the relative pose change onto the input coordinates
            #     processed_x_list = []
            #     # feats: bz x dim x N x na
            #     processed_feats = []
            #     for cur_x in x:
            #         # get preprocessed input from point coordinates, kanchor,
            #         cur_preprocessed_x = M.preprocess_input(cur_x, cur_kanchor, pose, False)
            #         cur_backbone = self.backbone
            #         for block_i, block in enumerate(cur_backbone):
            #             cur_preprocessed_x = block(cur_preprocessed_x)
            #         torch.cuda.empty_cache()
            #         processed_x_list.append(cur_preprocessed_x)
            #         # cur_processed_feats: bz x dim x N x na -> bz x N x dim x na -> bz x N x 1 x dim x na
            #         cur_processed_feats = safe_transpose(cur_preprocessed_x.feats, 1, 2).unsqueeze(2)
            #         processed_feats.append(cur_processed_feats)
            #     torch.cuda.empty_cache()
            #     # processed_feats: bz x N x n_s x dim x na
            #     processed_feats = torch.cat(processed_feats, dim=2)
            #     # hard_label: bz x N; processed_feats: bz x N x n_s x dim x na --> bz x N x dim x na
            #     processed_feats = batched_index_select(values=processed_feats, indices=hard_label.long().unsqueeze(-1), dim=2).squeeze(2).contiguous()
            #     # get processed x...
            #     x = sptk.SphericalPointCloud(x, safe_transpose(processed_feats, 1, 2), cur_anchors)



            # if cur_iter == 0:
            # would use global reconstruction for each iteration
            # glb_inv_feats: bz x dim; glb_orbit..: bz x na
            glb_inv_feats, glb_orbit_confidence = self.glb_outblock(x, mask=None)
            glb_output_RT = self.glb_trans_outblock_RT(x, mask=None, anchors=cur_anchors.unsqueeze(0).repeat(bz, 1, 1,
                                                                                                             1).contiguous())
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

            # glb_chamfer = chamfer_recon_to_ori + chamfer_ori_to_recon

            if self.glb_single_cd == 1:
                glb_chamfer = chamfer_ori_to_recon  #
            else:
                glb_chamfer = chamfer_recon_to_ori + chamfer_ori_to_recon

            # glb_orbit: bz; minn_glb_chamfer: bz
            minn_glb_chamfer, glb_orbit = torch.min(glb_chamfer, dim=-1)

            minn_chamfer_recon_to_ori = batched_index_select(chamfer_recon_to_ori, indices=glb_orbit.unsqueeze(-1),
                                                             dim=1).squeeze(-1)
            minn_chamfer_ori_to_recon = batched_index_select(chamfer_ori_to_recon, indices=glb_orbit.unsqueeze(-1),
                                                             dim=1).squeeze(-1)
            self.glb_recon_ori_dist = (torch.sqrt(minn_chamfer_recon_to_ori) + torch.sqrt(
                minn_chamfer_ori_to_recon)).mean() * 0.5
            print(self.glb_recon_ori_dist)


            self.glb_ori_to_recon_dist = torch.sqrt(minn_chamfer_ori_to_recon).mean()  # minn chamfer ori to recon...
            print(f"glb_ori_to_recon, L1: {float(self.glb_ori_to_recon_dist.item())}")
            # minn_glb_chamfer for optimization. & glb_orbit for global orbit selection/pose transformed?
            # selected global chamfer distance and global orbit...

            # should minimize the selected global reconstruction chamfer distance
            # selected_glb_R: bz x 3 x 3
            selected_glb_R = batched_index_select(glb_R, glb_orbit.unsqueeze(-1).long(), dim=1).squeeze(1)
            selected_glb_T = batched_index_select(glb_T, glb_orbit.unsqueeze(-1).long(), dim=1).squeeze(1)

            selected_transformed_glb_recon_pts = batched_index_select(transformed_glb_recon_pts,
                                                                      indices=glb_orbit.unsqueeze(-1).long(),
                                                                      dim=1).squeeze(1)
            inv_trans_ori_pts = torch.matmul(safe_transpose(selected_glb_R, -1, -2),
                                             ori_pts - selected_glb_T.unsqueeze(-1))
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


            # cur_kanchor = 1 # use kpcovn
            cur_kanchor = self.kpconv_kanchor  # use kpcovn
            cur_anchors = self.kpconv_anchors

            x_seg = None
            if x_list is None:
                if not self.use_art_mode:
                    x = M.preprocess_input(x, cur_kanchor, pose, False)
                    # pose: bz x N x 4 x 4
                    cur_backbone = self.backbone # get backbone...
                    # x.feats: bz x N x dim x 1 --> na = 1 now for kpconv net; process input point coordinates
                    for block_i, block in enumerate(cur_backbone):
                        x = block(x)
                    torch.cuda.empty_cache()

                    x_seg = M.preprocess_input(ori_pts, cur_kanchor, pose, False)
                    cur_backbone = self.backbone_sec
                    for block_i, block in enumerate(cur_backbone):
                        x_seg = block(x_seg)
                    torch.cuda.empty_cache()
                else:
                    ''' Process points via chaning convolution process '''
                    x_w_art_mode = x.unsqueeze(1)
                    # x_w_art_mode: bz x ns x 3 x np
                    # x_w_art_mode = torch.cat(x_w_art_mode, dim=1)
                    x_w_art_mode = sptk.SphericalPointCloudPose(x_w_art_mode,
                                                                sptk.get_occupancy_features(safe_transpose(x, 1, 2), self.kanchor, False),
                                                                None, pose)
                    cur_backbone = self.backbone  # if cur_iter == 0 else self.backbone_sec
                    hard_label = torch.zeros((bz, npoints), dtype=torch.long).cuda()
                    for block_i, block in enumerate(cur_backbone):
                        # print(f"current block {block_i}")
                        x_w_art_mode = block(x_w_art_mode, seg=hard_label)
                        # print(f"after current block: {block_i}")
                    x = sptk.SphericalPointCloud(x, x_w_art_mode.feats, cur_anchors)
                    torch.cuda.empty_cache()
                    ''' Process points via chaning convolution process '''
            else:
                # here, `pose` should be the identity pose matrix --- we have projected the relative pose change onto the input coordinates
                if not self.use_art_mode:
                    ''' Process points via changing input points '''
                    processed_x_list = []
                    # feats: bz x dim x N x na
                    processed_feats = []
                    # print()

                    for cur_x in x_list:
                        # todo: if we just detach the r and t and feed it into the next backbone? or the current backbone?
                        # get preprocessed input from point coordinates, kanchor
                        cur_preprocessed_x = M.preprocess_input(cur_x, cur_kanchor, pose, False)
                        cur_backbone = self.backbone # if cur_iter == 0 else self.backbone_sec
                        for block_i, block in enumerate(cur_backbone):
                            cur_preprocessed_x = block(cur_preprocessed_x)
                        torch.cuda.empty_cache()
                        processed_x_list.append(cur_preprocessed_x)
                        # cur_processed_feats: bz x dim x N x na -> bz x N x dim x na -> bz x N x 1 x dim x na
                        cur_processed_feats = safe_transpose(cur_preprocessed_x.feats, 1, 2).unsqueeze(2)
                        processed_feats.append(cur_processed_feats)
                        torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    # processed_feats: bz x N x n_s x dim x na
                    processed_feats = torch.cat(processed_feats, dim=2)
                    # hard_label: bz x N; processed_feats: bz x N x n_s x dim x na --> bz x N x dim x na
                    processed_feats = batched_index_select(values=processed_feats, indices=hard_label.long().unsqueeze(-1), dim=2).squeeze(2).contiguous()
                    # get processed x...; and processed features
                    x = sptk.SphericalPointCloud(x, safe_transpose(processed_feats, 1, 2), cur_anchors)
                    ''' Process points via changing input points '''

                    torch.cuda.empty_cache()

                    x_seg = M.preprocess_input(ori_pts, cur_kanchor, pose, False)
                    cur_backbone = self.backbone_sec
                    for block_i, block in enumerate(cur_backbone):
                        x_seg = block(x_seg)
                    torch.cuda.empty_cache()

                    ''' Process points via changing input points '''
                    # processed_x_list_seg = []
                    # # feats: bz x dim x N x na
                    # processed_feats_seg = []
                    # # print()
                    # for cur_x in x_list:
                    #     # todo: if we just detach the r and t and feed it into the next backbone? or the current backbone?
                    #     # get preprocessed input from point coordinates, kanchor
                    #     cur_preprocessed_x_seg = M.preprocess_input(cur_x, cur_kanchor, pose, False)
                    #     cur_backbone = self.backbone_sec  # if cur_iter == 0 else self.backbone_sec
                    #     for block_i, block in enumerate(cur_backbone):
                    #         cur_preprocessed_x_seg = block(cur_preprocessed_x_seg)
                    #     torch.cuda.empty_cache()
                    #     processed_x_list_seg.append(cur_preprocessed_x_seg)
                    #     # cur_processed_feats: bz x dim x N x na -> bz x N x dim x na -> bz x N x 1 x dim x na
                    #     cur_processed_feats_seg = safe_transpose(cur_preprocessed_x_seg.feats, 1, 2).unsqueeze(2)
                    #     processed_feats_seg.append(cur_processed_feats_seg)
                    #     torch.cuda.empty_cache()
                    # torch.cuda.empty_cache()
                    # # processed_feats: bz x N x n_s x dim x na
                    # processed_feats_seg = torch.cat(processed_feats_seg, dim=2)
                    # # hard_label: bz x N; processed_feats: bz x N x n_s x dim x na --> bz x N x dim x na
                    # processed_feats_seg = batched_index_select(values=processed_feats_seg,
                    #                                        indices=hard_label.long().unsqueeze(-1), dim=2).squeeze(
                    #     2).contiguous()
                    # # get processed x...; and processed features
                    # x_seg = sptk.SphericalPointCloud(x, safe_transpose(processed_feats_seg, 1, 2), cur_anchors)
                    ''' Process points via changing input points '''
                else:
                    ''' Process points via chaning convolution process '''
                    x_w_art_mode = []
                    for cur_x in x_list:
                        x_w_art_mode.append(cur_x.unsqueeze(1))
                    # x_w_art_mode: bz x ns x 3 x np
                    x_w_art_mode = torch.cat(x_w_art_mode, dim=1)
                    x_w_art_mode = sptk.SphericalPointCloudPose(x_w_art_mode,  # permute x
                                                 sptk.get_occupancy_features(safe_transpose(x, 1, 2), self.kanchor, False),  # add feature
                                                 None, pose)  #
                    cur_backbone = self.backbone  # if cur_iter == 0 else self.backbone_sec
                    for block_i, block in enumerate(cur_backbone):
                        x_w_art_mode = block(x_w_art_mode, seg=hard_label)
                    x = sptk.SphericalPointCloud(x, x_w_art_mode.feats, cur_anchors)
                    ''' Process points via chaning convolution process '''

            torch.cuda.empty_cache()

            # Get per-point invariant feature # is not None
            if self.sel_mode is not None and cur_iter > 0:  # sel mode in the invariant otuput block
                sel_mode_new = self.sel_mode_new
            else:
                sel_mode_new = None

            # x_seg = None
            x_seg = x if x_seg is None else x_seg
            if self.inv_pooling_method == 'attention':
                # confidence: bz x N x na; ppinvout: bz x dim x N
                # ppinv_out, confidence = self.ppint_outblk(x, sel_mode_new=sel_mode_new) # ppinv_out; confidence...
                ppinv_out, confidence = self.ppint_outblk(x_seg, sel_mode_new=sel_mode_new)
                if self.orbit_attn == 1:
                    ppinv_out = torch.cat([ppinv_out, safe_transpose(confidence, -1, -2)], dim=1)
            else:
                # ppinv_out = self.ppint_outblk(x, sel_mode_new=sel_mode_new)
                ppinv_out = self.ppint_outblk(x_seg, sel_mode_new=sel_mode_new)

            # ppinv_out: bz x dim x N
            if self.orbit_attn == 3:  # coordinates involved in to the slot-attention process; Get ppinv_out by concatenating ...
                ppinv_out = torch.cat([ppinv_out, ori_pts], dim=1)

            ''' Point grouping '''
            # slot attention # Get attention values from each point to each slot
            rep_slots, attn_ori = self.slot_attention(safe_transpose(ppinv_out, -1, -2))
            ''' Point grouping '''

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
                    if curr_seg_pts_nn > curr_maxx_seg_nn:  # maxx_seg_nn --> maximum number of points in an existing segmentation
                        curr_maxx_seg_nn = curr_seg_pts_nn
                        curr_minn_seg_label = seg_label
                    # curr_minn_seg_label = min(curr_minn_seg_label, seg_label)
                tot_seg_to_idxes.append(cur_seg_to_idxes)  # we should use
                tot_minn_seg_label.append(curr_minn_seg_label)  #

            # whl_shp_inv_feats, whl_shp_orbit_confidence = self.whole_shp_outblock(x, mask=None) # Get whole shape invariant feature
            # pv_points: bz x 3 x (self.num_slots - 1); pv_points; pv_points decoding
            # pv_points decoding...; decode pv-points
            # use_whl_shp_inv_feats = False
            use_whl_shp_inv_feats = True
            # whl_shp_offset = torch.mean(x.xyz, dim=-1, keepdim=True)
            # only sigmoid without centralizing points

            ''' Use whole shape inv-feature for pivot point decoding '''
            # pv_points = self.pv_points_decoding_blk(whl_shp_inv_feats) - 0.5; # pv_points
            # # pv-point ---> bz x 3 x n_s
            # pv_points = pv_points + torch.mean(x.xyz, dim=-1, keepdim=True)
            ''' Use whole shape inv-feature for pivot point decoding '''

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
            pv_points_moving_parts = []
            central_points = []

            slot_pv_canon_cd_loss = 0.0
            other_slots_trans_pv = [[] for _ in range(bz)]
            base_slot_canon_pred = [[] for _ in range(bz)]

            slot_avg_offset = []

            for i_s in range(self.num_slots):
                cur_slot_inv_feats = []
                cur_pair_slot_inv_feats = []
                cur_slot_orbit_confidence = []
                cur_slot_R = []
                cur_slot_T = []
                cur_slot_axis = []
                cur_slot_pivot_points_equiv = []
                cur_slot_central_points_equiv = []
                cur_slot_avg_offset = []
                cur_slot_abs_inv_feats = []
                for i_bz in range(bz):
                    curr_minn_seg_label = tot_minn_seg_label[i_bz]
                    # curr_minn_seg_label = 0
                    tot_minn_seg_labels.append(curr_minn_seg_label)
                    # print(f"check x, x.xyz: {x.xyz.size()}, x.feats: {x.feats.size()}, x.anchors: {x.anchors.size()}")
                    if i_s in tot_seg_to_idxes[i_bz]:
                        # sptk.SphericalPointCloud(x_xyz, out_feat, x.anchors) # x.xyz: bz x N
                        ''' xyz, feats '''  # current slot xyz... # cur_bz_cur_slot_xyz: 1 x n_pts x 3
                        cur_bz_cur_slot_xyz = safe_transpose(x.xyz, -1, -2)[
                            i_bz, tot_seg_to_idxes[i_bz][i_s]].unsqueeze(0)
                        cur_bz_cur_slot_feats = safe_transpose(x.feats, 1, 2)[
                            i_bz, tot_seg_to_idxes[i_bz][i_s]].unsqueeze(
                            0)
                        ''' xyz, feats '''
                        # cur_bz_cur_slot_xyz = safe_transpose(x.xyz, -1, -2)[i_bz].unsqueeze(0)
                        # cur_bz_cur_slot_feats = safe_transpose(x.feats, 1, 2)[i_bz].unsqueeze(0)
                        if i_s != curr_minn_seg_label:
                            curr_bz_minn_slot_xyz = safe_transpose(x.xyz, -1, -2)[
                                i_bz, tot_seg_to_idxes[i_bz][curr_minn_seg_label]].unsqueeze(0)
                            curr_bz_minn_slot_feats = safe_transpose(x.feats, 1, 2)[
                                i_bz, tot_seg_to_idxes[i_bz][curr_minn_seg_label]].unsqueeze(0)
                            pair_cur_bz_cur_slot_xyz = torch.cat([curr_bz_minn_slot_xyz, cur_bz_cur_slot_xyz], dim=1)
                            pair_cur_bz_cur_slot_feats = torch.cat([curr_bz_minn_slot_feats, cur_bz_cur_slot_feats],
                                                                   dim=1)
                        else:
                            # pair_cur_bz_cur_slot --> slot xyz and slot features
                            pair_cur_bz_cur_slot_xyz = cur_bz_cur_slot_xyz
                            pair_cur_bz_cur_slot_feats = cur_bz_cur_slot_feats
                        pair_cur_bz_cur_slot_xyz = safe_transpose(x.xyz, -1, -2)[i_bz].unsqueeze(0)  # xyz and feats
                        pair_cur_bz_cur_slot_feats = safe_transpose(x.feats, 1, 2)[i_bz].unsqueeze(0)
                    else:
                        cur_bz_cur_slot_xyz = torch.zeros((1, 2, 3), dtype=torch.float32).cuda()
                        cur_bz_cur_slot_feats = torch.zeros((1, 2, x.feats.size(1), x.feats.size(-1)),
                                                            dtype=torch.float32).cuda()
                        # pair_cur_bz_cur_slot_xyz = cur_bz_cur_slot_xyz
                        # pair_cur_bz_cur_slot_feats = cur_bz_cur_slot_feats
                        pair_cur_bz_cur_slot_xyz = safe_transpose(x.xyz, -1, -2)[i_bz].unsqueeze(0)  #
                        pair_cur_bz_cur_slot_feats = safe_transpose(x.feats, 1, 2)[i_bz].unsqueeze(0)

                    cur_bz_cur_slot_avg_offset = torch.mean(cur_bz_cur_slot_xyz, dim=-2)
                    cur_slot_avg_offset.append(cur_bz_cur_slot_avg_offset)

                    # cur_bz_cur_slot_soft_mask = None
                    # Get spherical point cloud
                    cur_bz_cur_slot_x = sptk.SphericalPointCloud(safe_transpose(cur_bz_cur_slot_xyz, -1, -2),
                                                                 safe_transpose(cur_bz_cur_slot_feats, 1, 2), x.anchors)

                    pair_cur_bz_cur_slot_x = sptk.SphericalPointCloud(safe_transpose(pair_cur_bz_cur_slot_xyz, -1, -2),
                                                                      safe_transpose(pair_cur_bz_cur_slot_feats, 1, 2),
                                                                      x.anchors)

                    # Get output invariant feature... # inv_feats
                    if self.pred_pv_equiv:
                        cur_bz_cur_slot_equiv_feats, cur_bz_cur_slot_inv_feats, cur_bz_cur_slot_orbit_confidence = \
                        self.slot_outblock[i_s](cur_bz_cur_slot_x, mask=None)
                    else:
                        cur_bz_cur_slot_inv_feats, cur_bz_cur_slot_orbit_confidence = self.slot_outblock[i_s](
                            cur_bz_cur_slot_x, mask=None)
                        cur_bz_cur_slot_abs_inv_feats, _ = self.abs_slot_outblock[i_s](cur_bz_cur_slot_x, mask=None)

                    if self.zz_pred_pv_equiv:
                        # pair slot outblock...
                        pair_cur_bz_cur_slot_equiv_feats, pair_cur_bz_cur_slot_inv_feats, pair_cur_bz_cur_slot_orbit_confidence = \
                        self.pair_slot_outblock[i_s](pair_cur_bz_cur_slot_x, mask=None)
                    else:
                        pair_cur_bz_cur_slot_inv_feats, pair_cur_bz_cur_slot_orbit_confidence = \
                            self.pair_slot_outblock[i_s](pair_cur_bz_cur_slot_x, mask=None)

                    # # pair_cur_bz_cur_slot_inv_feats: 1 x dim
                    # pair_cur_bz_cur_slot_inv_feats, pair_cur_bz_cur_slot_orbit_confidence = self.pair_slot_outblock[i_s](
                    #     pair_cur_bz_cur_slot_x, # pair
                    #     mask=None)

                    #### expanded pair current bz current slot invariant features
                    expanded_pair_cur_bz_cur_slot_inv_feats = pair_cur_bz_cur_slot_inv_feats.unsqueeze(-1).unsqueeze(
                        -1).contiguous().repeat(1, 1, cur_bz_cur_slot_xyz.size(0), cur_bz_cur_slot_feats.size(-1))
                    # cur_bz_cur_slot_x.feats = expanded_pair_cur_bz_cur_slot_inv_feats
                    expanded_cur_bz_cur_slot_inv_feats = cur_bz_cur_slot_inv_feats.unsqueeze(-1).unsqueeze(
                        -1).contiguous().repeat(1, 1, cur_bz_cur_slot_xyz.size(0), cur_bz_cur_slot_feats.size(-1))

                    ''' inv feats '''
                    if use_whl_shp_inv_feats:
                        ''' If each slot uses the same invariant feature '''
                        # cur_slot_inv_feats.append(whl_shp_inv_feats[i_bz].unsqueeze(0))
                        ''' If we use different invariant features for those slots '''
                        cur_slot_inv_feats.append(cur_bz_cur_slot_inv_feats)
                        cur_pair_slot_inv_feats.append(pair_cur_bz_cur_slot_inv_feats)
                        cur_slot_abs_inv_feats.append(cur_bz_cur_slot_abs_inv_feats)
                    else:
                        cur_slot_inv_feats.append(cur_bz_cur_slot_inv_feats)  # inv feat;
                        cur_pair_slot_inv_feats.append(pair_cur_bz_cur_slot_inv_feats)
                        cur_slot_abs_inv_feats.append(cur_bz_cur_slot_abs_inv_feats)
                    # invariant features for the whole shape...
                    # cur_slot_inv_feats.append(whl_shp_inv_feats[i_bz].unsqueeze(0)) # confidence values...
                    cur_slot_orbit_confidence.append(cur_bz_cur_slot_orbit_confidence)

                    #
                    pre_feats = cur_bz_cur_slot_inv_feats if i_s > 0 else None
                    # pre_feats = None
                    pre_feats = None

                    defined_proj_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda().unsqueeze(0)
                    if self.shape_type == 'drawer':
                        ''' Use no pre-defined axis for further translation decoding '''
                        defined_proj_axis = None
                        ''' Use pre-defined axis (z-axis) for further translation decoding '''
                        defined_proj_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda().unsqueeze(0)
                        use_offset = False
                        # use_offset = True
                        # decode transformation from grouped points
                        cur_bz_slot_output_RT = self.slot_trans_outblk_RT[i_s](cur_bz_cur_slot_x, mask=None,
                                                                               anchors=self.anchors.unsqueeze(0).repeat(
                                                                                   bz, 1, 1, 1).contiguous(),
                                                                               proj_axis=defined_proj_axis,  use_offset=use_offset)
                    else:
                        ''' If we use pair invariant features for rotation decoding '''
                        # cur_bz_cur_slot_x = sptk.SphericalPointCloud(safe_transpose(cur_bz_cur_slot_xyz, -1, -2),
                        #                                              expanded_pair_cur_bz_cur_slot_inv_feats, x.anchors)
                        # cur_bz_slot_output_RT = self.slot_trans_outblk_RT[i_s](cur_bz_cur_slot_x, mask=None, trans_feats=safe_transpose(cur_bz_cur_slot_feats, 1, 2), trans_xyz=safe_transpose(cur_bz_cur_slot_xyz, -1, -2), anchors=self.anchors.unsqueeze(0).repeat(bz, 1, 1, 1).contiguous(), pre_feats=pre_feats)
                        ''' If we use pair equivariant features for rotation decoding '''
                        # cur_bz_slot_output_RT = self.slot_trans_outblk_RT[i_s](pair_cur_bz_cur_slot_x, mask=None,
                        #                                                        trans_feats=safe_transpose(
                        #                                                            cur_bz_cur_slot_feats, 1, 2),
                        #                                                        trans_xyz=safe_transpose(
                        #                                                            cur_bz_cur_slot_xyz, -1, -2),
                        #                                                        anchors=self.anchors.unsqueeze(0).repeat(
                        #                                                            bz, 1, 1, 1).contiguous(),
                        #                                                        pre_feats=pre_feats)
                        ''' If we use equivariant features for rotation decoding '''
                        # we can use the predicted rotation in the following process, while the translation could not be of good use
                        # Use equiv feature for transformation & pv-points prediction?
                        # pred_pv_poitns_in_feats = cur_bz_cur_slot_equiv_feats if self.pred_pv_equiv else None #
                        pred_pv_poitns_in_feats = pair_cur_bz_cur_slot_equiv_feats if self.pred_pv_equiv else None  # pred_pv_points_in_feats
                        pred_central_points_in_feats = cur_bz_cur_slot_equiv_feats if self.pred_pv_equiv else None  # pred_central_points

                        pred_axis_in_feats = None

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
                        # pair_cur_bz_slot_output_RT = self.pair_slot_trans_outblk_RT[i_s](pair_cur_bz_cur_slot_x, mask=None,
                        #                                                        trans_feats=safe_transpose(pair_cur_bz_cur_slot_feats, 1, 2),
                        #                                                        trans_xyz=safe_transpose(pair_cur_bz_cur_slot_xyz, -1, -2),
                        #                                                        anchors=self.anchors.unsqueeze(0).repeat(bz, 1, 1, 1).contiguous(),
                        #                                                        pre_feats=pre_feats,
                        #                                                        pred_pv_poitns_in_feats=pred_pv_poitns_in_feats,
                        #                                                        pred_central_points_in_feats=pred_central_points_in_feats)
                        ''' If we use invariant features for rotation decoding '''
                        # cur_bz_cur_slot_x = sptk.SphericalPointCloud(safe_transpose(cur_bz_cur_slot_xyz, -1, -2),
                        #                                              expanded_cur_bz_cur_slot_inv_feats,
                        #                                              x.anchors)
                        # cur_bz_slot_output_RT = self.slot_trans_outblk_RT[i_s](cur_bz_cur_slot_x, mask=None,
                        #                                                        trans_feats=safe_transpose(
                        #                                                            cur_bz_cur_slot_feats, 1, 2),
                        #                                                        trans_xyz=safe_transpose(
                        #                                                            cur_bz_cur_slot_xyz, -1, -2),
                        #                                                        anchors=self.anchors.unsqueeze(0).repeat(
                        #                                                            bz, 1, 1, 1).contiguous(),
                        #                                                        pre_feats=pre_feats)

                    # Get current slot's rotation; slot_R
                    cur_bz_cur_slot_R = cur_bz_slot_output_RT['R']
                    # cur_bz_cur_slot_axis = cur_bz_slot_output_RT[
                    #     'axis']  # 1 x 3 x na; get axis... ### Get axis for each slot...
                    # cur_bz_cur_slot_axis = pair_cur_bz_slot_output_RT['axis']
                    # Get current slot's translation
                    # if self.pred_t:
                    #     cur_bz_cur_slot_T = cur_bz_slot_output_RT['T']
                    # else:
                    #     cur_bz_cur_slot_T = torch.zeros((1, 3, cur_kanchor), dtype=torch.float).cuda()
                    # if self.pred_pv_equiv:
                    #     cur_bz_cur_slot_pv_points_equiv = cur_bz_slot_output_RT['pv_points']
                    #     cur_bz_cur_slot_central_points_equiv = cur_bz_slot_output_RT['central_points']
                    #     cur_bz_cur_slot_pv_points_equiv = cur_bz_cur_slot_pv_points_equiv - 0.5  # quite
                    #     cur_bz_cur_slot_central_points_equiv = cur_bz_cur_slot_central_points_equiv - 0.5
                    #     cur_slot_pivot_points_equiv.append(cur_bz_cur_slot_pv_points_equiv)
                    #     cur_slot_central_points_equiv.append(cur_bz_cur_slot_central_points_equiv)

                    cur_bz_cur_slot_T = cur_bz_slot_output_RT['T']
                    cur_slot_R.append(cur_bz_cur_slot_R)
                    cur_slot_T.append(cur_bz_cur_slot_T)
                    cur_slot_axis.append(defined_proj_axis.unsqueeze(-1).contiguous().repeat(bz, 1, self.kanchor))

                cur_slot_inv_feats = torch.cat(cur_slot_inv_feats, dim=0)  # invariant features
                cur_pair_slot_inv_feats = torch.cat(cur_pair_slot_inv_feats, dim=0)
                cur_slot_abs_inv_feats = torch.cat(cur_slot_abs_inv_feats,
                                                   dim=0)  # invariant features encoded using absolute coordinates
                # cur_slot_orbit_confidence = torch.cat(cur_slot_orbit_confidence, dim=0) #
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
                elif self.recon_prior in [6, 8, 9]:
                    # cur_slot_cuboic_R: bz x 3 x 3
                    if self.rel_for_points == 0:  # still use inv-features encoded by absolute coordinates for canonical shape decoding
                        # cur_slot_abs_inv_feats = None
                        cur_slot_abs_inv_feats = None
                    if self.recon_prior in [6, 8]:  # reconstruction prior in 6 or 8...
                        cur_slot_canon_pts, cur_slot_pivot_points, cur_slot_central_points = \
                            self.slot_shp_recon_net[i_s](cur_slot_inv_feats, pv_point_inv_feat=cur_slot_abs_inv_feats,
                                                         central_point_inv_feat=cur_slot_abs_inv_feats)
                    else:
                        cur_slot_canon_pts, cur_slot_pivot_points, cur_slot_central_points = \
                            self.slot_shp_recon_net[i_s](cur_slot_inv_feats)
                    # cur_slot_canon_pts = cur_slot_canon_pts + curslo
                    cur_slot_pivot_points = cur_slot_pivot_points - 0.5
                    cur_slot_central_points = cur_slot_central_points - 0.5
                    # avg_offset = torch.sum(cur_bz_cur_slot_xyz, dim=)
                    # cur_slot_avg_offset = torch.cat(cur_slot_avg_offset, dim=0)

                    # if self.shape_type in ['washing_machine']:
                    #     cur_slot_central_points = cur_slot_central_points + cur_slot_avg_offset

                    # x.xyz: bz x 3 x N
                    pair_cur_slot_canon_pts, cur_slot_pivot_points, _ = self.pair_slot_shp_recon_net[i_s](
                        cur_pair_slot_inv_feats)
                    # mean of input pc's coordinates
                    ''' If we use relative pos to predict pivot points '''
                    # cur_slot_pivot_points = cur_slot_pivot_points - 0.5 + torch.mean(x.xyz, dim=-1)
                    ''' If we use absolute pos to predict pivot points '''
                    cur_slot_pivot_points = cur_slot_pivot_points - 0.5

                    slot_recon_pivot_points.append(cur_slot_pivot_points.unsqueeze(1))
                    slot_recon_central_points.append(cur_slot_central_points.unsqueeze(1))

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

                    # x.xyz: bz x 3 x N
                    pair_cur_slot_canon_pts, cur_slot_pivot_points, _ = self.pair_slot_shp_recon_net[i_s](
                        cur_pair_slot_inv_feats)
                    cur_slot_pivot_points = cur_slot_pivot_points - 0.5
                    #### Save pivot points and central points ####
                    slot_recon_pivot_points.append(cur_slot_pivot_points.unsqueeze(1))
                    slot_recon_central_points.append(cur_slot_central_points.unsqueeze(1))
                    #### Save pivot points and central points ####
                    slot_cuboic_recon_pts.append(cur_slot_cuboic_x.unsqueeze(1))  # cuboid reconstruction points
                    slot_cuboic_R.append(cur_slot_cuboic_R.unsqueeze(1))  # cuboid rotation xxx
                else:
                    cur_slot_canon_pts = self.slot_shp_recon_net[i_s](cur_slot_inv_feats)

                # cur_slot_canon_pts: bz x M x 3 --> predicted canonicalized points?

                # Get slot canonical points
                ''' Scale slot reconstruction's coordinates '''
                if i_s >= self.num_slots - 3:
                    # scale_factor = torch.tensor([1.0, 0.3, 1.0], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(-1)
                    scale_factor = torch.tensor([1.0, 0.3, 1.0], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(-1)

                    # print(f"canon recon: {cur_slot_canon_pts.size()}, scale_factor: {scale_factor.size()}")
                    minn_pts_coord, _ = torch.min(cur_slot_canon_pts, dim=-1, keepdim=True)
                    maxx_pts_coord, _ = torch.max(cur_slot_canon_pts, dim=-1, keepdim=True)

                    maxx_minn_scale = maxx_pts_coord - minn_pts_coord

                    # maxx_minn_scale[:, 0, :] = 1.0

                    maxx_minn_scale = maxx_minn_scale.detach()
                    zeross_pts_coord = (cur_slot_canon_pts - minn_pts_coord) / torch.clamp(maxx_minn_scale, min=1e-6)

                    maxx_minn_scale[:, 1, :][maxx_minn_scale[:, 1, :] > 0.3] = 0.3

                    maxx_ori_pts_x_coord, _ = torch.max(ori_pts[:, 0, :], dim=-1, keepdim=True)
                    minn_ori_pts_x_coord, _ = torch.min(ori_pts[:, 0, :], dim=-1, keepdim=True)
                    # maxx_ori_pts_x_coord =
                    maxx_minn_scale[:, 0, :] = maxx_ori_pts_x_coord - minn_ori_pts_x_coord # x_scale...
                    scale_factor = maxx_minn_scale.detach()

                    maxx_ori_coord_y, _ = torch.max(ori_pts[:, 1, :], dim=-1, keepdim=True)
                    minn_ori_coord_y, _ = torch.min(ori_pts[:, 1, :], dim=-1, keepdim=True)

                    minn_pts_coord[:, 0, :] = minn_ori_pts_x_coord + 0.5

                    if i_s == self.num_slots - 3:
                        minn_pts_coord[:, 1, :] = (1.0 - (maxx_ori_coord_y - minn_ori_coord_y)) / 2.0 + 0.1

                    elif i_s == self.num_slots - 2:
                        # minn_pts_coord[:, 1, :] = 0.3
                        minn_pts_coord[:, 1, :] = (maxx_ori_coord_y - minn_ori_coord_y) / 3. + (1.0 - (maxx_ori_coord_y - minn_ori_coord_y)) / 2.0
                    elif i_s == self.num_slots - 1:
                        # minn_pts_coord[:, 1, :] = 0.6
                        minn_pts_coord[:, 1, :] = (maxx_ori_coord_y - minn_ori_coord_y) / 3. * 2. + (1.0 - (maxx_ori_coord_y - minn_ori_coord_y)) / 2.0

                    cur_slot_canon_pts = zeross_pts_coord * scale_factor + minn_pts_coord
                    print("max and min", torch.max(cur_slot_canon_pts, dim=-1)[0], torch.min(cur_slot_canon_pts, dim=-1)[0])

                cur_slot_canon_pts = cur_slot_canon_pts - 0.5
                # cur_slot_canon_pts[:, 0, ]



                # How to use pv points?
                ''' Saperated prediction version '''
                cur_slot_R = torch.cat(cur_slot_R, dim=0)
                cur_slot_T = torch.cat(cur_slot_T, dim=0)
                cur_slot_axis = torch.cat(cur_slot_axis, dim=0)
                cur_slot_avg_offset = torch.cat(cur_slot_avg_offset, dim=0)
                ''' Saperated prediction version '''

                ''' Add x-z offset... '''
                # cur_bz_cur_slot_avg_offset_xz = cur_slot_avg_offset.clone().detach()
                # cur_bz_cur_slot_avg_offset_xz[:, 2] = 0.0
                # cur_slot_canon_pts = cur_bz_cur_slot_avg_offset_xz.unsqueeze(-1) + cur_slot_canon_pts
                ''' Add x-z offset... '''

                # cur_slot_canon_pts
                slot_canon_pts.append(cur_slot_canon_pts.unsqueeze(1))
                slot_R.append(cur_slot_R.unsqueeze(1))
                slot_T.append(cur_slot_T.unsqueeze(1))
                slot_axis.append(cur_slot_axis.unsqueeze(1))  # current slot axis
                slot_avg_offset.append(cur_slot_avg_offset.unsqueeze(1))

            # slot_canon_pts: bz x n_s x M x 3
            # slot_R: bz x n_s x 4 x na
            # slot_T: bz x n_s x 3 x na;
            slot_canon_pts = torch.cat(slot_canon_pts, dim=1)
            slot_canon_pts = safe_transpose(slot_canon_pts, -1, -2)

            # slot_axis: bz x ns x 3 x na
            slot_axis = torch.cat(slot_axis, dim=1)
            slot_axis = safe_transpose(slot_axis, -1, -2)

            slot_avg_offset = torch.cat(slot_avg_offset, dim=1)


            # slot_R = torch.cat(mtx_slot_R, dim=1)
            slot_R = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).contiguous().repeat(bz, self.num_slots, self.kanchor, 1, 1).contiguous().cuda()
            # slot_R: bz x n_s x na x 3 x 3
            slot_T = torch.cat(slot_T, dim=1)
            slot_T = safe_transpose(slot_T, -1, -2)

            if self.shape_type == 'drawer':
                # slot_T[:, 0] = slot_T[:, 0] * 0.0 # set to none moving...
                slot_T[:, :self.num_slots - 3] = slot_T[:, :self.num_slots - 3] * 0.0

            # defined_axises = defined_axises[0]
            defined_axises = slot_axis[:, 0, :, :]


            ''' Add distance restriction for slot canonical reconstruction '''
            # moving_parts_canon_recon: bz x (n_s - 1) x M x 3; -->
            moving_parts_canon_recon = slot_canon_pts[:, self.num_slots - 3:] # get moving parts
            # print(f"moving_parts_canon_recon: {moving_parts_canon_recon.size()}")
            moving_parts_y_coord = moving_parts_canon_recon[..., 1] # get y_coordinates of moving parts reconstruction
            # minn_moving_parts_y_coord: bz x (n_s - 1)
            # print(f"moving_parts_y_coord: {moving_parts_y_coord.size()}")

            minn_moving_parts_y_coord, _ = torch.min(moving_parts_y_coord, dim=2) # minn_moving_parts_y_coord -->
            maxx_moving_parts_y_coord, _ = torch.max(moving_parts_y_coord, dim=2)
            # print(f"minn_moving_parts_y_coord: {minn_moving_parts_y_coord.size()}, maxx_moving_parts_y_coord: {maxx_moving_parts_y_coord.size()}")

            # maximize the distance --->
            dist_minn_to_maxx_y_coord = (minn_moving_parts_y_coord.unsqueeze(-1) - maxx_moving_parts_y_coord.unsqueeze(-2)) ** 2
            # print(f"dist_minn_to_maxx_y_coord, {dist_minn_to_maxx_y_coord.size()}")
            # summ_dist_minn_to_maxx = torch.sum(torch.sum(dist_minn_to_maxx_y_coord, dim=-1), dim=-1) - (dist_minn_to_maxx_y_coord[..., 0, 0] + dist_minn_to_maxx_y_coord[..., 1, 1] + dist_minn_to_maxx_y_coord[..., 2, 2])
            summ_dist_minn_to_maxx = (dist_minn_to_maxx_y_coord[..., 0, 0] + dist_minn_to_maxx_y_coord[..., 1, 1] +
                        dist_minn_to_maxx_y_coord[..., 2, 2])

            # maximize the distance --->
            dist_minn_to_minn_y_coord = (minn_moving_parts_y_coord.unsqueeze(-1) - minn_moving_parts_y_coord.unsqueeze(
                -2)) ** 2
            # print(f"dist_minn_to_maxx_y_coord, {dist_minn_to_maxx_y_coord.size()}")
            summ_dist_minn_to_minn = torch.sum(torch.sum(dist_minn_to_minn_y_coord, dim=-1), dim=-1) - (
                        dist_minn_to_minn_y_coord[..., 0, 0] + dist_minn_to_minn_y_coord[..., 1, 1] +
                        dist_minn_to_minn_y_coord[..., 2, 2])

            # maximize the distance --->
            dist_maxx_to_maxx_y_coord = (maxx_moving_parts_y_coord.unsqueeze(-1) - maxx_moving_parts_y_coord.unsqueeze(
                -2)) ** 2
            # print(f"dist_minn_to_maxx_y_coord, {dist_minn_to_maxx_y_coord.size()}")
            summ_dist_maxx_to_maxx = torch.sum(torch.sum(dist_maxx_to_maxx_y_coord, dim=-1), dim=-1) - (
                    dist_maxx_to_maxx_y_coord[..., 0, 0] + dist_maxx_to_maxx_y_coord[..., 1, 1] +
                    dist_maxx_to_maxx_y_coord[..., 2, 2])


            avg_summ_dist_minn_to_maxx = torch.mean(summ_dist_minn_to_maxx)
            avg_summ_dist_minn_to_minn = torch.mean(summ_dist_minn_to_minn)
            avg_summ_dist_maxx_to_maxx = torch.mean(summ_dist_maxx_to_maxx)
            # avg_summ_dist_minn_to_maxx = 0.1 * avg_summ_dist_minn_to_maxx + avg_summ_dist_minn_to_minn + avg_summ_dist_maxx_to_maxx
            avg_summ_dist_minn_to_maxx = avg_summ_dist_minn_to_minn + avg_summ_dist_maxx_to_maxx
            avg_summ_dist_minn_to_maxx = 0.1 * avg_summ_dist_minn_to_maxx
            print(f"avg_summ_dist_minn_to_maxx: {avg_summ_dist_minn_to_maxx.item()}")

            # scale_minn_maxx_y_coord = maxx_moving_parts_y_coord - minn_moving_parts_y_coord
            # scale_minn_maxx_y_coord[scale_minn_maxx_y_coord > 0.4] = 0.4
            # scale_minn_maxx_y_coord = scale_minn_maxx_y_coord / torch.clamp(maxx_moving_parts_y_coord - minn_moving_parts_y_coord, min=1e-6)
            # moving_parts_y_coord = (moving_parts_y_coord - minn_moving_parts_y_coord.unsqueeze(-1)) * scale_minn_maxx_y_coord.unsqueeze(-1) + minn_moving_parts_y_coord.unsqueeze(-1)
            # slot_canon_pts[:, 1:, :, 1] = moving_parts_y_coord



            # defined_axises = safe_transpose(defined_axises, -1, -2)

            #
            # defined_pivot_points = pv_points[:, 0, :] # bz x 3
            pv_points = torch.zeros((bz, self.num_slots, 3), dtype=torch.float32).cuda()
            defined_pivot_points = torch.zeros((bz, 3), dtype=torch.float32).cuda()
            # defined_axises: bz x 3
            # offset_pivot_points = defined_pivot_points - torch.sum(defined_pivot_points * defined_axises, dim=-1, keepdim=True) * defined_axises
            offset_pivot_points = torch.norm(defined_pivot_points, p=2, dim=-1)

            # if self.recon_prior == 5:
            #     slot_cuboic_R = torch.cat(slot_cuboic_R, dim=1)

            # slot_R = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), slot_R)
            slot_R_ori = slot_R.clone().detach()  # rotation matrix # rotation matrix...
            slot_R_ori_nod = slot_R.clone()
            # slot_R: bz x ns x na x 3 x 3


            slot_T_ori = slot_T.clone()
            # slot_T_joint...
            slot_T_joint = slot_T.clone()
            # transformed_pv_points: bz x n_s x 3
            # pv_points: bz x n_s x 3
            ''' Get pv points '''
            # ... but how about add a translation for pv
            # central_transformed_pv_points = pv_points + central_points.detach()
            central_transformed_pv_points = pv_points # pivot points...
            central_transformed_pv_points_equiv = slot_recon_pivot_points_equiv  # + slot_recon_central_points_equiv.detach()

            # R_anchor(R(P) + T)
            ''' Get slots' rotations and translations --- if we use pivot points... '''
            slot_R = torch.matmul(cur_anchors.unsqueeze(0).unsqueeze(0), slot_R)  #
            slot_T = torch.matmul(cur_anchors.unsqueeze(0).unsqueeze(0), slot_T.unsqueeze(-1)).squeeze(-1)
            slot_T_joint = torch.matmul(cur_anchors.unsqueeze(0).unsqueeze(0), slot_T_joint.unsqueeze(-1)).squeeze(-1)
            #
            # #### Shape type --> drawer #####
            # if self.shape_type == 'drawer':  # drawer; drawer...
            #     # slot_T[:, 0] = 0.0 # set the translation of the first slot to zero... set the other to xxx...; the
            #     slot_T[:, 0] = slot_T[:, 0] * 0.0  # fix points of the first slot
            #     slot_T_joint[:, 0] = slot_T_joint[:, 0] * 0.0  # fix points of the first slot

            k = self.kpconv_kanchor if self.sel_mode_trans is None else 1
            if self.sel_mode_trans is not None:  # select a specific mode for transformation
                topk_anchor_idxes = torch.tensor([self.sel_mode_trans], dtype=torch.long).cuda().unsqueeze(0).unsqueeze(
                    0).repeat(bz, self.num_slots, 1).contiguous()

            # transformed_pts: bz x n_s x na x M x 3
            # slot_canon_pts: bz x n_s x M x 3
            # slot_R: bz x n_s x na x 3 x 3 @ slot_recon_pts: bz x n_s x 1 x 3 x M
            # transformed points
            transformed_pts = safe_transpose(torch.matmul(slot_R, safe_transpose(slot_canon_pts.unsqueeze(2), -1, -2)),
                                             -1,
                                             -2) + slot_T.unsqueeze(-2)

            if self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:
                # transformed_pts_ori: bz x n_s x na x M x 3
                # slot_canon_pts: bz x n_s x M x 3;
                transformed_pts_ori = safe_transpose(
                    torch.matmul(slot_R_ori, safe_transpose(slot_canon_pts.unsqueeze(2), -1, -2)), -1,
                    -2) + slot_T_ori.unsqueeze(-2)
                # pv_points: bz x n_s x 3; slot_R_ori: bz x n_s x na x 3 x 3 --> bz x n_s x na x 3
                transformed_pv_points_ori = torch.matmul(slot_R_ori, pv_points.unsqueeze(2).unsqueeze(-1)).squeeze(
                    -1)  #
                transformed_pv_points = torch.matmul(slot_R, pv_points.unsqueeze(2).unsqueeze(-1)).squeeze(-1)
                # transformed_central_pv_points_ori = torch.matmul(slot_R_ori, central_transformed_pv_points.unsqueeze(2).unsqueeze(-1)).squeeze(-1)

            # selected_anchors = cur_anchors.clone() # na x 3 x 3
            if k < cur_kanchor:  # cur_kanchor
                # transformed_pts: bz x n_s x na x M x 3 --> bz x n_s x k x M x 3
                # selected_anchors: bz x ns x k_a x 3 x 3
                # selected_anchors = batched_index_select(values=selected_anchors, indices=topk_anchor_idxes, dim=0)
                transformed_pts = batched_index_select(values=transformed_pts, indices=topk_anchor_idxes, dim=2)
                if self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:
                    transformed_pts_ori = batched_index_select(values=transformed_pts_ori, indices=topk_anchor_idxes,
                                                               dim=2)
                    transformed_pv_points_ori = batched_index_select(values=transformed_pv_points_ori,
                                                                     indices=topk_anchor_idxes, dim=2)
                    transformed_pv_points = batched_index_select(values=transformed_pv_points,
                                                                 indices=topk_anchor_idxes, dim=2)
                    # transformed_central_pv_points = batched_index_select(values=transformed_central_pv_points, indices=topk_anchor_idxes, dim=2)
                    slot_axis = batched_index_select(values=slot_axis, indices=topk_anchor_idxes, dim=2)
                    # defined_axises = batched_index_select(values=defined_axises, indices=topk_anchor_idxes, dim=1)

                    if self.pred_pv_equiv:
                        central_transformed_pv_points_equiv = batched_index_select(
                            values=central_transformed_pv_points_equiv, indices=topk_anchor_idxes, dim=2)
                        slot_recon_pivot_points_equiv = batched_index_select(values=slot_recon_pivot_points_equiv,
                                                                             indices=topk_anchor_idxes, dim=2)
                        slot_recon_central_points_equiv = batched_index_select(values=slot_recon_central_points_equiv,
                                                                               indices=topk_anchor_idxes, dim=2)

            # transformed_pts: bz x n_s x na x M x 3 --> bz x n_s x k x M x 3
            # transformed_pts = batched_index_select(values=transformed_pts, indices=topk_anchor_idxes, dim=2)

            # hard_one_hot_labels: bz x N x ns
            # dist_recon_ori: bz x n_s x na x M x N

            dist_recon_ori = torch.sum((transformed_pts.unsqueeze(-2) - safe_transpose(ori_pts, -1, -2).unsqueeze(
                1).unsqueeze(1).unsqueeze(1)) ** 2, dim=-1)
            expanded_hard_one_hot_labels = safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(2).unsqueeze(2).repeat(
                1, 1, k, self.recon_part_M, 1)

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
                minn_dist_ori_to_recon * safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(2),
                dim=-1) / torch.clamp(
                torch.sum(safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(2), dim=-1), min=1e-8)
            minn_dist_ori_to_recon = torch.sum(minn_dist_ori_to_recon * soft_weights.unsqueeze(2),
                                               dim=-1) / torch.clamp(
                torch.sum(soft_weights.unsqueeze(2), dim=-1), min=1e-8)

            #### use soft weights of all points for ori_to_recon soft loss aggregation ####
            minn_dist_ori_to_recon_all_pts = torch.sum(minn_dist_ori_to_recon_all_pts * attn_ori.unsqueeze(2),
                                                       dim=-1) / torch.clamp(torch.sum(attn_ori.unsqueeze(2), dim=-1),
                                                                             min=1e-8)

            # orbit_slot_dist_ori_recon = minn_dist_ori_to_recon + minn_dist_recon_to_ori
            # # orbit_slot_dist_ori_recon = minn_dist_ori_to_recon_hard + minn_dist_recon_to_ori
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



            hard_slot_indicators = (hard_one_hot_labels.sum(1) > 0.5).float()  # Get hard slot indicators


            ''' Distance for further optimization '''
            #### slot distance recon all pts ####
            if self.slot_single_cd == 1:
                orbit_slot_dist_ori_recon_all_pts = minn_dist_ori_to_recon
            else:
                orbit_slot_dist_ori_recon_all_pts = minn_dist_ori_to_recon + minn_dist_recon_to_ori
            ''' Distance for further optimization '''

            orbit_slot_dist_ori_recon_all_pts = batched_index_select(values=orbit_slot_dist_ori_recon_all_pts,
                                                                     indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(
                -1)
            slot_dist_ori_recon = (orbit_slot_dist_ori_recon_all_pts * hard_slot_indicators).float().sum(-1)

            # print(f"After transformed: {transformed_pts.size()}, slot_orbits: {slot_orbits.size()}")
            # transformed_pts: bz x n_s x M x 3
            transformed_pts = batched_index_select(transformed_pts, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)

            # slot_canon_pts = batched_index_select(safe_transpose(slot_canon_pts, -1, -2), indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
            # slot_canon_pts: bz x n_s x 3 x M --> bz x n_s x M x 3

            slot_canon_pts = safe_transpose(slot_canon_pts, -1, -2)

            defined_axises = batched_index_select(values=defined_axises, indices=slot_orbits[:, 0].unsqueeze(-1),
                                                  dim=1).squeeze(1)

            # slot_orbits --> bz x n_s
            # minn_dist_ori_to_recon: bz x n_s x na x N ---> bz x n_s x N
            selected_minn_dist_ori_to_recon = batched_index_select(values=minn_dist_ori_to_recon,
                                                                   indices=slot_orbits.long().unsqueeze(-1),
                                                                   dim=2).squeeze(2)
            # selected_minn_dist_ori_to_recon: bz x N
            selected_minn_dist_ori_to_recon, _ = torch.min(selected_minn_dist_ori_to_recon, dim=1)
            ori_to_recon = torch.sqrt(selected_minn_dist_ori_to_recon).mean(dim=-1).mean()
            print(f"ori_to_recon, uni L1: {float(ori_to_recon.item())}")
            self.ori_to_recon = ori_to_recon

            
            if k < cur_kanchor:
                # slot_orbits: bz x n_s
                slot_orbits = batched_index_select(values=topk_anchor_idxes, indices=slot_orbits.unsqueeze(-1),
                                                   dim=2).squeeze(-1)


            print("slot_orbits", slot_orbits)  # register slot_orbits...



            # selected_anchors: bz x n_s x 3 x 3
            selected_anchors = batched_index_select(values=cur_anchors, indices=slot_orbits, dim=0)

            if self.slot_single_mode == 1:
                # self.sel_mode_new = slot_orbits[:, 0] #### Get slot mode new!
                self.sel_mode_new = None  # slot_orbits[:, 0] #### Get slot mode new!

            # print(f"check slot_R: slot_R: {slot_R.size()}, slot_T: {slot_T.size()}")
            slot_R_raw = slot_R.clone()
            slot_R = batched_index_select(values=slot_R, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
            slot_T = batched_index_select(values=slot_T, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
            # slot_T_joint --> select from slot_T_joint...
            slot_T_joint = batched_index_select(values=slot_T_joint, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)


            slot_R_raw = batched_index_select(values=slot_R_raw, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(
                2).detach()

            filtered_transformed_pts = transformed_pts * hard_slot_indicators.unsqueeze(-1).unsqueeze(-1)
            # filtered_transformed_pts = transformed_pts  # * hard_slot_indicators.unsqueeze(-1).unsqueeze(-1)

            # expanded_recon_slot_pts: bz x (n_s x M) x 3
            # print("filter:", filtered_transformed_pts.size())
            expanded_recon_slot_pts = filtered_transformed_pts.contiguous().view(bz, self.num_slots * self.recon_part_M,
                                                                                 3).contiguous()
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

            print(f"self.glb_recon_ori_dist: {self.glb_recon_ori_dist}")

            ''' Global reconstruction distance for optimization '''
            if self.slot_single_cd == 1:
                glb_recon_ori_dist = ori_to_recon_dist
            else:
                glb_recon_ori_dist = recon_to_ori_dist + ori_to_recon_dist
            ''' Global reconstruction distance for optimization '''

            tot_recon_loss = glb_recon_ori_dist * self.glb_recon_factor + (
                slot_dist_ori_recon) * self.slot_recon_factor # + slot_pv_canon_cd_loss  # add slot_pv_canon loss to the tot_recon_loss term

            minn_to_maxx_loss_factor = 0.01
            tot_recon_loss = tot_recon_loss.mean() + (-1.0 * avg_summ_dist_minn_to_maxx * minn_to_maxx_loss_factor)

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

            print("anchors,", selected_anchors.size(), defined_axises.size())
            real_defined_axises = torch.matmul(selected_anchors, defined_axises.unsqueeze(-1)).squeeze(-1)
            out_feats['defined_axises'] = defined_axises.detach().cpu().numpy()  # defined_axises:
            out_feats['real_defined_axises'] = real_defined_axises.detach().cpu().numpy()  # defined_axises:
            real_defined_axises = real_defined_axises[:, 0, :]
            self.real_defined_axises = real_defined_axises.clone()
            # print()

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
            selected_pred_pose = torch.cat(
                [selected_pred_pose, torch.zeros((bz, npoints, 1, 4), dtype=torch.float32).cuda()], dim=2)

            if not self.use_art_mode:
                selected_pred_pose = pose ## orz...

            # selected_pts_orbit = batched_index_select(values=selected_slot_oribt, indices=selected_labels, dim=1)

            out_feats['pred_slot_Ts'] = selected_pred_T.detach().cpu().numpy()

            # selected_inv_pred_R: bz x N x 3 x 3; ori_pts: bz x N x 3
            selected_inv_pred_R = selected_pred_R.contiguous().transpose(-1, -2).contiguous()
            # rotated_ori_pts = torch.matmul(selected_pred_R, ori_pts.contiguous().unsqueeze(-1).contiguous()).squeeze(-1).contiguous()
            # From transformed points to original canonical points
            transformed_ori_pts = torch.matmul(selected_inv_pred_R,
                                               (safe_transpose(ori_pts, -1, -2) - selected_pred_T).unsqueeze(
                                                   -1)).squeeze(-1)
            # transformed_ori_pts = torch.matmul(selected_inv_pred_R, (rotated_ori_pts - selected_pred_T).unsqueeze(-1)).squeeze(-1)

            # transformed ori pts
            out_feats['transformed_ori_pts'] = transformed_ori_pts.detach().cpu().numpy()

            out_feats['pv_points'] = pv_points.detach().cpu().numpy()

            if gt_pose is not None:
                gt_R = gt_pose[..., :3, :3]
                gt_T = gt_pose[..., :3, 3]
                gt_inv_R = gt_R.contiguous().transpose(-1, -2).contiguous()
                gt_transformed_ori_pts = torch.matmul(gt_inv_R,
                                                      (safe_transpose(ori_pts, -1, -2) - gt_T).unsqueeze(-1)).squeeze(
                    -1)
                out_feats['gt_transformed_ori_pts'] = gt_transformed_ori_pts.detach().cpu().numpy()


            np.save(self.log_fn + f"_n_stage_{self.stage}_iter_{cur_iter}.npy", out_feats)

            # if cur_iter == 0:
            # tmp_R = torch.matmul(safe_transpose(selected_glb_R.detach(), -1, -2).unsqueeze(1), gt_pose[:, :, :3, :3])
            # tmp_T = torch.matmul(safe_transpose(selected_glb_R.detach(), -1, -2).unsqueeze(1),
            #                      (gt_pose[:, :, :3, 3] - selected_glb_T.detach().unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
            # # Get GT Pose for each point
            # gt_pose = torch.cat([tmp_R, tmp_T.unsqueeze(-1)], dim=-1)
            # # Get GT Pose for
            # gt_pose = torch.cat([gt_pose, torch.zeros((tmp_R.size(0), tmp_R.size(1), 1, 4), dtype=torch.float32).cuda()],
            #                     dim=-2)

            # out_feats['inv_feats'] = invariant_feats_npy
            # np.save(self.log_fn + f"_n_iter_{cur_iter}_with_feats.npy", out_feats)

            # print("selected_pred_R", selected_pred_R.size())
            # we just use zero translations for the predicted pose? (dummy translations for pose)

            ''' If use pose for iteration '''
            # selected_pred_R = torch.matmul(safe_transpose(selected_glb_R.unsqueeze(1), -1, -2), selected_pred_R)
            # pred_pose = torch.cat([selected_pred_R, torch.zeros((bz, npoints, 3, 1), dtype=torch.float32).cuda()],
            #                       dim=-1)
            # pred_pose = torch.cat([pred_pose, torch.zeros((bz, npoints, 1, 4), dtype=torch.float32).cuda()], dim=-2)
            ''' If use pose for iteration '''

            ''' If not to use pose for iteration '''
            # pred_pose = pose
            ''' If not to use pose for iteration '''

            # if cur_iter == 0:
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

            # if cur_iter == 0:
            self.out_feats = out_feats

            # ori pc;
            # selected_glb_anchor: bz x n_s x 3 x 3
            # selected_glb_anchor = batched_index_select(self.anchors, indices=slot_orbits.long(), dim=0)[:, 0]
            # # ori_pts: bz x 3 x N --> bz x 3 x N
            # inv_trans_ori_pts = torch.matmul(safe_transpose(selected_glb_anchor, -1, -2), ori_pts)

            ''' Get inv-sel-mode-new '''
            if self.sel_mode is not None and self.slot_single_mode:
                # selected_glb_anchor: bz x n_s x 3 x 3
                selected_glb_anchor = batched_index_select(self.anchors, indices=slot_orbits.long(), dim=0)[:, 0]
                inv_selected_glb_anchor = safe_transpose(selected_glb_anchor, -1, -2)
                dot_product_inv_glb_anchor_anchors = torch.matmul(inv_selected_glb_anchor.unsqueeze(1),
                                                                  self.anchors.unsqueeze(0))
                # traces: bz x na
                traces = dot_product_inv_glb_anchor_anchors[..., 0, 0] + dot_product_inv_glb_anchor_anchors[..., 1, 1] + \
                         dot_product_inv_glb_anchor_anchors[..., 2, 2]
                inv_slot_orbits = torch.argmax(traces, dim=-1)
                ##### self.sel_mode_new: Get sel_mode_new from inv_slot_orbits
                #### get inv lsot orbits ####
                self.sel_mode_new = inv_slot_orbits

            # get inv transformed points
            # selected_labels: bz x N; slot_R: bz x ns x 3 x 3; slot_T: bz x ns x 3
            # selected_pts_R: bz x N x 3 x 3

            # slot_R = gt_pose_segs[:, :, :3, :3]
            # slot_T = gt_pose_segs[:, :, :3, 3]

            selected_pts_R = batched_index_select(values=slot_R, indices=selected_labels.long(), dim=1)
            # selected_pts_T_joint: bz x N x 3
            # selected_pts_T_joint = batched_index_select(values=slot_T, indices=selected_labels.long(), dim=1)
            selected_pts_T_joint = batched_index_select(values=slot_T_joint, indices=selected_labels.long(), dim=1)



            slot_R = slot_R.detach()
            slot_T_joint = slot_T_joint.detach()
            selected_pts_R = selected_pts_R.detach()
            selected_pts_T_joint = selected_pts_T_joint.detach()

            inv_transformed_x_list = []
            inv_transformed_x_dict = {}
            for i_s in range(self.num_slots):
                # cur_slot_inv_transformed_x = []

                cur_bz_cur_slot_R = slot_R[:, i_s] # bz x 3 x 3 matrix
                cur_bz_cur_slot_T_joint = slot_T_joint[:, i_s] # selected_pts_T_joint
                # to other slots.... # need to use relative rotation and relative translation to transform...

                # p1 = R1(p1') + t1; p2 = R2(p2') + t2
                # p2'' = R1(R2^{-1}(p2 - t2)) + t1 =
                # cur_bz_cur_slot_inv_R: bz x N x 3 x 3; cur_bz_cur_slot_inv_T_joint: bz x N x 3
                # cur_bz_cur_slot_inv_R: bz x N x 3 x 3; cur_bz_cur_slot_R: bz x ; selected_pts_R:
                cur_bz_cur_slot_inv_R = torch.matmul(cur_bz_cur_slot_R.unsqueeze(1), safe_transpose(selected_pts_R, -1, -2))
                # cur_bz_cur_slot_inv_T_joint: bz x N x 3; cur_bz_cur_slot_T_joint: bz x N x 3
                cur_bz_cur_slot_inv_T_joint = cur_bz_cur_slot_T_joint.unsqueeze(1) - torch.matmul(cur_bz_cur_slot_inv_R, selected_pts_T_joint.unsqueeze(-1)).squeeze(-1)
                # ori_pts: bz x 3 x N
                # cur_bz_inv_trans_x: bz x N x 3
                # cur_bz_inv_trans_x = torch.matmul(cur_bz_cur_slot_inv_R, (safe_transpose(ori_pts, -1, -2) - cur_bz_cur_slot_inv_T_joint).unsqueeze(-1)).squeeze(-1)
                cur_bz_inv_trans_x = torch.matmul(cur_bz_cur_slot_inv_R, (
                            safe_transpose(ori_pts, -1, -2)).unsqueeze(-1)).squeeze(-1) + cur_bz_cur_slot_inv_T_joint
                inv_transformed_x_list.append(safe_transpose(cur_bz_inv_trans_x, -1, -2))

                inv_transformed_x_dict[i_s] = cur_bz_inv_trans_x.detach().cpu().numpy()

            # if cur_iter == 0:
            #     # pred_glb_pose: bz x 3 x 4
            #     # pred_glb_pose = torch.cat([selected_glb_R, selected_glb_T.unsqueeze(-1)], dim=-1)
            # else:
            #     pred_glb_pose = None

            # self.pred_glb_pose = pred_glb_pose

            if cur_iter == 0 and self.num_iters > 1:
                inv_transformed_x_dict['hard_labels'] = hard_labels.detach().cpu().numpy()
                np.save(f"inv_tras_x_{cur_iter}.npy", inv_transformed_x_dict)

            tot_loss = tot_recon_loss  # + (pts_ov_max_percent_loss) * 4.0 # encourage entropy

            return tot_loss, selected_pred_pose, inv_transformed_x_list, selected_labels

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

    def forward(self, x, pose, ori_pc=None, rlabel=None, nn_inter=2, pose_segs=None, canon_pc=None, normals=None,
                canon_normals=None):

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
        cur_inv_transformed_x_list = None
        cur_selected_labels = None

        if self.stage == 0:
            loss = self.forward_one_iter(
                cur_transformed_points, cur_estimated_pose, ori_pc=ori_pc, rlabel=rlabel, cur_iter=0,
                gt_pose=cur_gt_pose, gt_pose_segs=pose_segs, canon_pc=canon_pc,
                selected_pts_orbit=cur_selected_pts_orbit, normals=normals, canon_normals=canon_normals)
            torch.cuda.empty_cache()
        else:
            for i_iter in range(self.num_iters):
                cur_loss, cur_estimated_pose, cur_inv_transformed_x_list, cur_selected_labels = self.forward_one_iter(
                    cur_transformed_points, cur_estimated_pose, x_list=cur_inv_transformed_x_list, hard_label=cur_selected_labels, ori_pc=ori_pc, rlabel=rlabel, cur_iter=i_iter, gt_pose=pose, gt_pose_segs=pose_segs, canon_pc=canon_pc, selected_pts_orbit=cur_selected_pts_orbit)
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
                    'kanchor': kpconv_kanchor,  ### set kanchor to 1!!
                    'norm': 'BatchNorm2d',
                    'permute_modes': permute_modes,
                    'use_art_mode': opt.equi_settings.use_art_mode,
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
        'axis_reg_stra': opt.equi_settings.axis_reg_stra,
        'glb_single_cd': opt.equi_settings.glb_single_cd,
        'slot_single_cd': opt.equi_settings.slot_single_cd,
        'rel_for_points': opt.equi_settings.rel_for_points,
        'use_art_mode': opt.equi_settings.use_art_mode
        # 'opt': opt

    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model


def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
