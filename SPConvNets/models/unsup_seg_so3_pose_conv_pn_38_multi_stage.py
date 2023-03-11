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
from SPConvNets.utils.slot_attention_spec_v2 import SlotAttention
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


class ClsSO3ConvModel(nn.Module):
    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()
        ''' Global pose factorization '''
        self.glb_backbone = nn.ModuleList()
        
        for block_param in params['backbone']:
            self.glb_backbone.append(M.BasicSO3PoseConvBlock(block_param))


        self.backbone = nn.ModuleList()
        for block_param in params['kpconv_backbone']:
            self.backbone.append(M.BasicSO3PoseConvBlock(block_param))

        self.backbone_sec = nn.ModuleList()
        for block_param in params['kpconv_backbone']:
            self.backbone_sec.append(M.BasicSO3PoseConvBlock(block_param))

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
        self.with_part_proposal = params['general']['with_part_proposal']  # --rel-for-points

        self.sel_mode_trans = None if self.sel_mode_trans == -1 else self.sel_mode_trans

        self.local_rank = int(os.environ['LOCAL_RANK'])

        #### Set parameter alias ####
        self.recon_part_M = self.part_pred_npoints  # 128, 256, 512, 1024
        self.transformation_dim = 7

        self.stage = params['general']['cur_stage']

        self.log_fn = f"{self.exp_indicator}_{self.shape_type}_reconp_{self.recon_prior}_num_iters_{self.num_iters}"

        ''' Set chamfer distance '''
        self.chamfer_dist = ChamferDistance()

        ''' Get anchors '''
        self.anchors = torch.from_numpy(L.get_anchors(params['outblock']['kanchor'])).cuda()
        # self.kpconv_anchors = torch.from_numpy(L.get_anchors(1)).cuda()
        if self.kpconv_kanchor == 1:
            self.kpconv_anchors = torch.eye(3, dtype=torch.float32).cuda().unsqueeze(0)
        else:
            self.kpconv_anchors = torch.from_numpy(L.get_anchors(self.kpconv_kanchor)).cuda()

        ''' Construct slot-attention module '''
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
        self.whole_shp_outblock = Mso3.InvOutBlockOursWithMask(params['outblock'], norm=1, pooling_method='attention',
                                                               use_pointnet=True, use_abs_pos=use_abs_pos)
        ''' Construct pv points decoding block '''  # decode pv points
        self.pv_points_decoding_blk = DecoderFC([256, 256], params['outblock']['mlp'][-1], self.num_slots - 1, None)

        use_abs_pos = True
        ''' Construct inv-feat output block for slots '''
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
        self.glb_recon_npoints = 1024 # partial motion laptop, motion laptop
        self.glb_recon_npoints = 512 # complete oven, washing machine, safe, real laptop
        # self.glb_recon_npoints = self.npoints

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
        self.slot_trans_outblk_RT = nn.ModuleList()
        self.pair_slot_trans_outblk_RT = nn.ModuleList()
        # self.pred_t = False
        self.pred_t = True
        self.r_representation = 'quat'  # aaa
        self.r_representation = 'angle'
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
                # cur_slot_trans_outblk = SO3OutBlockRTWithMask if self.shape_type != 'drawer' else SO3OutBlockRTWithAxisWithMask
                cur_slot_trans_outblk = SO3OutBlockRTWithMaskSep  # output block for transformation...
                # c_in_rot = self.inv_out_dim #### If we use inv features for angle decoding
                c_in_rot = self.encoded_feat_dim
                c_in_trans = self.encoded_feat_dim
                self.slot_trans_outblk_RT.append(
                    cur_slot_trans_outblk(params['outblock'], norm=1, pooling_method='max',
                                          global_scalar=True,
                                          # global scalar?
                                          use_anchors=False,
                                          feat_mode_num=self.kanchor, num_heads=1, representation=cur_r_representation,
                                          c_in_rot=c_in_rot, c_in_trans=c_in_trans, pred_axis=self.pred_axis,
                                          # whehter to predict axis
                                          pred_pv_points=pred_pv_points, pv_points_in_dim=pred_pv_points_in_dim,
                                          # whether to predict pv point --> only useful when we use equiv feature for pivot point prediction
                                          pred_central_points=pred_central_points,
                                          # whether to predict central points --> only useful when we use equiv feature for central point prediction
                                          central_points_in_dim=pred_central_points_in_dim,  #
                                          mtx_based_axis_regression=self.mtx_based_axis_regression)
                    # how to predict axis...
                )
                self.pair_slot_trans_outblk_RT.append(
                    cur_slot_trans_outblk(params['outblock'], norm=1, pooling_method='max',
                                          global_scalar=True,  # whether to use
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

        ''' For different axis-reg-stra '''
        if self.axis_reg_stra == 1:  # running mean
            n_joints = self.num_slots - 1  # we use the same direction for all joints in one shape
            self.joint_dir_running_mean = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
            self.momentum = 0.99
            self.pv_in_count_nn = 0
            # avg_pv_point_distance? --> update by (pv_in_count_nn * avg_distance + sum_of_valid_distance) / (in_count_nn + valid_nn)
            self.avg_pv_point_distance = torch.zeros((n_joints,), dtype=torch.float32).cuda()


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
            # print(minn_glb_chamfer)

            minn_chamfer_recon_to_ori = batched_index_select(chamfer_recon_to_ori, indices=glb_orbit.unsqueeze(-1),
                                                             dim=1).squeeze(-1)
            minn_chamfer_ori_to_recon = batched_index_select(chamfer_ori_to_recon, indices=glb_orbit.unsqueeze(-1),
                                                             dim=1).squeeze(-1)
            self.glb_recon_ori_dist = (torch.sqrt(minn_chamfer_recon_to_ori) + torch.sqrt(
                minn_chamfer_ori_to_recon)).mean() * 0.5
            # print(self.glb_recon_ori_dist)

            self.glb_ori_to_recon_dist = torch.sqrt(minn_chamfer_ori_to_recon).mean() # minn chamfer ori to recon...
            # print(f"glb_ori_to_recon, L1: {float(self.glb_ori_to_recon_dist.item())}")

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

                    processed_feats = x.feats # processed_feats: bz x dim x N x na

                    processed_feats_ori = processed_feats.clone()

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
                    # print("here1111")
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
                    processed_feats_ori = processed_feats.clone()
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

                # ppinv_out, confidence = self.ppint_outblk(x, sel_mode_new=sel_mode_new)
                ppinv_out, confidence = self.ppint_outblk(x_seg, sel_mode_new=sel_mode_new)
                if self.orbit_attn == 1:
                    ppinv_out = torch.cat([ppinv_out, safe_transpose(confidence, -1, -2)], dim=1)
            else:
                # ppinv_out = self.ppint_outblk(x, sel_mode_new=sel_mode_new)
                ppinv_out = self.ppint_outblk(x_seg, sel_mode_new=sel_mode_new)

            # ppinv_out: bz x dim x N
            if self.orbit_attn == 3:
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

                        if not self.with_part_proposal:
                            cur_bz_cur_slot_xyz = safe_transpose(x.xyz, -1, -2)[i_bz].unsqueeze(0)
                            cur_bz_cur_slot_feats = safe_transpose(x.feats, 1, 2)[i_bz].unsqueeze(0)

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

                    if self.shape_type == 'drawer':
                        ''' Use no pre-defined axis for further translation decoding '''
                        defined_proj_axis = None
                        ''' Use pre-defined axis (z-axis) for further translation decoding '''
                        defined_proj_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda().unsqueeze(0)
                        use_offset = False
                        # use_offset = True
                        cur_bz_slot_output_RT = self.slot_trans_outblk_RT[i_s](cur_bz_cur_slot_x, mask=None,
                                                                               anchors=self.anchors.unsqueeze(0).repeat(
                                                                                   bz, 1, 1, 1).contiguous(),
                                                                               proj_axis=defined_proj_axis,
                                                                               pre_feats=None, use_offset=use_offset)
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


                    # Get current slot's rotation; slot_R
                    cur_bz_cur_slot_R = cur_bz_slot_output_RT['R']
                    cur_bz_cur_slot_axis = cur_bz_slot_output_RT['axis']
                    # cur_bz_cur_slot_axis = pair_cur_bz_slot_output_RT['axis']
                    # Get current slot's translation
                    if self.pred_t:
                        cur_bz_cur_slot_T = cur_bz_slot_output_RT['T']
                    else:
                        cur_bz_cur_slot_T = torch.zeros((1, 3, cur_kanchor), dtype=torch.float).cuda()
                    if self.pred_pv_equiv:
                        cur_bz_cur_slot_pv_points_equiv = cur_bz_slot_output_RT['pv_points']
                        cur_bz_cur_slot_central_points_equiv = cur_bz_slot_output_RT['central_points']
                        cur_bz_cur_slot_pv_points_equiv = cur_bz_cur_slot_pv_points_equiv - 0.5  # quite
                        cur_bz_cur_slot_central_points_equiv = cur_bz_cur_slot_central_points_equiv - 0.5
                        cur_slot_pivot_points_equiv.append(cur_bz_cur_slot_pv_points_equiv)
                        cur_slot_central_points_equiv.append(cur_bz_cur_slot_central_points_equiv)
                    cur_slot_R.append(cur_bz_cur_slot_R)
                    cur_slot_T.append(cur_bz_cur_slot_T)
                    cur_slot_axis.append(cur_bz_cur_slot_axis)

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
                cur_slot_canon_pts = cur_slot_canon_pts - 0.5
                if use_whl_shp_inv_feats:  # whl_shp_offset: bz x 3 x 1; cur_slot_canon_pts: bz x M x 3
                    # print("whl_shp_offset", whl_shp_offset.size(), "cur_slot_canon_pts", cur_slot_canon_pts.size())
                    cur_slot_canon_pts = cur_slot_canon_pts  # + whl_shp_offset # .squeeze(-1).unsqueeze(-2)
                    if self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:
                        # print("cur_slot_central_points", cur_slot_central_points.size(), "whl_shp_offset", whl_shp_offset.size())
                        cur_slot_central_points = cur_slot_central_points  # + whl_shp_offset.squeeze(-1)

                        ''' If we use the predicted pivot point directly... '''
                        cur_slot_pivot_points = cur_slot_pivot_points  # + whl_shp_offset.squeeze(-1)
                        # cur_slot_pivot_points: bz x 3
                        # cur_slot_canon_pts: bz x M x 3
                        dist_slot_pv_canon = torch.sum(
                            (cur_slot_pivot_points.unsqueeze(-1) - cur_slot_canon_pts.detach()) ** 2, dim=1)
                        minn_dist_slot_pv_canon, _ = torch.min(dist_slot_pv_canon, dim=-1)
                        slot_pv_canon_cd_loss = slot_pv_canon_cd_loss + minn_dist_slot_pv_canon.mean()
                        ''' If we use the predicted pivot point directly... '''

                        ''' If take the first point of the slot as the pivot point directly... '''
                        # # print("cur_slot_pivot_points", cur_slot_pivot_points.size())
                        # cur_slot_pivot_points = cur_slot_canon_pts[:, :, 0] # bz x 3 --> the shape of the canonicalized points
                        # # print("cur_slot_pivot_points", cur_slot_pivot_points.size(), "cur_slot_canon_pts", cur_slot_canon_pts.size())
                        # # cur_slot_pivot_points
                        ''' If take the first point of the slot as the pivot point directly... '''

                        cur_slot_trans_pivot_points = cur_slot_pivot_points + cur_slot_central_points.detach()
                        for i_bz in range(bz):
                            if i_s == tot_minn_seg_labels[i_bz]:
                                base_slot_canon_pred[i_bz].append(cur_slot_canon_pts[i_bz].detach())
                            else:  # base_slot_canon_pred[i_bz]
                                other_slots_trans_pv[i_bz].append(cur_slot_trans_pivot_points[i_bz].unsqueeze(0))

                        # if curr_minn_seg_label != i_s: # pv_points #
                        pv_points.append(cur_slot_pivot_points.unsqueeze(1))
                        if curr_minn_seg_label != i_s:
                            pv_points_moving_parts.append(cur_slot_pivot_points.unsqueeze(1))
                        central_points.append(cur_slot_central_points.unsqueeze(1))

                # How to use pv poitns?
                ''' Saperated prediction version '''
                cur_slot_R = torch.cat(cur_slot_R, dim=0)
                cur_slot_T = torch.cat(cur_slot_T, dim=0)
                cur_slot_axis = torch.cat(cur_slot_axis, dim=0)
                cur_slot_avg_offset = torch.cat(cur_slot_avg_offset, dim=0)
                ''' Saperated prediction version '''

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

            # slot_cuboic_recon_pts: bz x n_s x 3
            # slot_cuboic_recon_pts = torch.cat(slot_cuboic_recon_pts, dim=1)
            # slot_cuboic_R: bz x n_s x 3 x 3
            # slot_cuboic_R = torch.cat(slot_cuboic_R, dim=1)

            # pv_points: bz x n_s x 3
            pv_points = torch.cat(pv_points, dim=1)  # your predicted pv_points;
            pv_points_moving_parts = torch.cat(pv_points_moving_parts, dim=1)
            ''' Set pv-point's y-axis to  '''
            # pv_points[:, :, 1] = torch.mean(slot_canon_pts.detach(), dim=-2)[:, :, 1]
            ''' Set pv-point's y-axis to  '''
            central_points = torch.cat(central_points, dim=1)  # get central point

            # get transformed points in the shape canonical space... bz x ns x M x 3 ---> slot recon points
            canon_transformed_pts = slot_canon_pts + central_points.unsqueeze(-2)

            if self.pred_pv_equiv:
                slot_recon_pivot_points_equiv = torch.cat(slot_recon_pivot_points_equiv,
                                                          dim=1)  # pivot points predicted by equiv features
                slot_recon_central_points_equiv = torch.cat(slot_recon_central_points_equiv, dim=1)
                slot_recon_pivot_points_equiv = safe_transpose(slot_recon_pivot_points_equiv, -1, -2)
                slot_recon_central_points_equiv = safe_transpose(slot_recon_central_points_equiv, -1,
                                                                 -2)  # bz x ns x na x 3 --> central points for each anchor

            # Get pivot points
            slot_pivot_points = torch.cat(slot_recon_pivot_points, dim=1)
            # Get central points
            slot_central_points = torch.cat(slot_recon_central_points, dim=1)

            ''' From predicted rotation angles to rotation matrices '''
            slot_R_raw = torch.cat(slot_R, dim=1).detach()
            mtx_slot_R = []
            defined_axises = []
            for i_s in range(len(slot_R)):  # get slots' rotations
                tot_cur_slot_mtx_R = []
                cur_slot_defined_axises = []
                for i_bz in range(bz):  # seg label for the base part?
                    cur_bz_cur_minn_seg_label = tot_minn_seg_labels[i_bz]

                    ##### Avg direction of the axis #####
                    # defined_axis = torch.mean(slot_axis, dim=1)[i_bz]  # use the mean of predicted axis...
                    # defined_axis = defined_axis / torch.clamp(torch.norm(defined_axis, dim=-1, keepdim=True, p=2),
                    #                                           min=1e-8)
                    ##### Avg direction of the axis #####

                    ##### Slot 1's predicted axis #####
                    # defined_axis = slot_axis[i_bz, 1]  ### the second slot's aixs...
                    defined_axis = slot_axis[i_bz, 0]
                    ##### Slot 1's predicted axis #####
                    # predicted axis 1KDTree
                    cur_slot_defined_axises.append(defined_axis.unsqueeze(0))

                    # cur_slot_defined_axises.append(slot_axis[i_bz, cur_bz_cur_minn_seg_label].unsqueeze(0))

                    if i_s == cur_bz_cur_minn_seg_label and i_s != cur_bz_cur_minn_seg_label:
                    # if i_s == cur_bz_cur_minn_seg_label:
                        cur_slot_R_mtx = torch.eye(3, dtype=torch.float32).cuda().contiguous().unsqueeze(0).unsqueeze(
                            0).unsqueeze(0).repeat(1, 1, cur_kanchor, 1, 1).contiguous()
                    else:
                        ''' Each slot should have its rotation matrix... '''
                        if not self.pred_axis:  # whether to predict axis
                            defined_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda().unsqueeze(0)
                        else:
                            
                            ''' The same axis for all slots --- version 3 for axis usage... '''
                            defined_axis = slot_axis[i_bz, 0]
                            ''' The same axis for all slots --- version 3 for axis usage...  '''

                        # From current slot's rotation value to its rotation angle
                        ''' Previous -- fix the maximum angle to 0.5 * pi '''
                        # cur_slot_R_mtx = torch.sigmoid(slot_R[i_s][i_bz].unsqueeze(0)) * math.pi * 0.5
                        ''' Previous -- maximum angle equals to rot_angle_factor '''
                        cur_slot_R_mtx = torch.sigmoid(slot_R[i_s][i_bz].unsqueeze(
                            0)) * math.pi * self.rot_angle_factor  # get rotation matrix for the slot
                        # if i_s == 0 or self.shape_type == 'drawer':
                        #     cur_slot_R_mtx = cur_slot_R_mtx * 0.0
                        if self.shape_type == 'drawer':
                            cur_slot_R_mtx = cur_slot_R_mtx * 0.0
                        cur_slot_R_mtx = compute_rotation_matrix_from_angle(
                            cur_anchors,
                            safe_transpose(cur_slot_R_mtx, -1, -2).view(1 * 1, cur_kanchor, -1),
                            defined_axis=defined_axis
                        ).contiguous().view(1, 1, cur_kanchor, 3, 3).contiguous()
                    tot_cur_slot_mtx_R.append(cur_slot_R_mtx)
                tot_cur_slot_mtx_R = torch.cat(tot_cur_slot_mtx_R, dim=0)
                # slot R for the matrix...
                mtx_slot_R.append(tot_cur_slot_mtx_R)
                cur_slot_defined_axises = torch.cat(cur_slot_defined_axises, dim=0)
                defined_axises.append(cur_slot_defined_axises)
                ''' From predicted rotation angles to rotation matrices '''

            slot_R = torch.cat(mtx_slot_R, dim=1)
            slot_T = torch.cat(slot_T, dim=1)
            defined_axises = defined_axises[0]

            #
            defined_pivot_points = pv_points[:, 0, :] # bz x 3
            # defined_axises: bz x 3
            offset_pivot_points = defined_pivot_points - torch.sum(defined_pivot_points * defined_axises, dim=-1, keepdim=True) * defined_axises
            offset_pivot_points = torch.norm(defined_pivot_points, p=2, dim=-1)

            if self.recon_prior == 5:
                slot_cuboic_R = torch.cat(slot_cuboic_R, dim=1)

            # slot_R = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), slot_R)
            slot_R_ori = slot_R.clone().detach()  # rotation matrix # rotation matrix...
            slot_R_ori_nod = slot_R.clone()
            # slot_R: bz x ns x na x 3 x 3

            ''' Use pv points to get slots' translations '''
            consumed_pv_idx = [0 for _ in range(bz)]  # consumed_pv_idx
            slot_T = []
            slot_T_joint = []
            for i_s in range(self.num_slots):
                tot_cur_slot_T = []
                tot_cur_slot_T_joint = []
                for i_bz in range(bz):
                    cur_bz_cur_minn_seg_label = tot_minn_seg_labels[i_bz]
                    if i_s == cur_bz_cur_minn_seg_label:  #
                        # cur_slot_trans: 3;
                        # cur_slot_trans = torch.zeros((3,), dtype=torch.float32).cuda().contiguous().unsqueeze(0).unsqueeze(0).unsqueeze(0).contiguous().repeat(1, 1, cur_kanchor, 1).contiguous()
                        ### central points realted...
                        ''' V1 -- For base part: only use central points in the translation calculation process '''
                        # cur_slot_trans = central_points[i_bz, i_s, :].unsqueeze(0).unsqueeze(0).unsqueeze(0).contiguous().repeat(1, 1, cur_kanchor, 1).contiguous()
                        ''' V1 -- For base part: only use central points in the translation calculation process '''

                        ''' V2 -- For base part: only use the pivot of another part for translation prediction '''
                        # cur_slot_pv_point = pv_points_moving_parts[i_bz, 0, :]
                        # cur_bz_cur_slot_R = slot_R[i_bz, i_s]  # na x 3 x 3
                        # # cur_slot_pv_point: how to build connections between
                        # cur_slot_trans = -1.0 * torch.matmul(cur_bz_cur_slot_R, cur_slot_pv_point.unsqueeze(0).unsqueeze(-1)) + cur_slot_pv_point.unsqueeze(0).unsqueeze(-1)
                        # cur_slot_trans = cur_slot_trans.squeeze(-1).unsqueeze(0).unsqueeze(0)
                        ''' V2 -- For base part: only use the pivot of another part for translation prediction '''
                        ''' V3 -- For base part: Use both pvp and central p for translation prediction '''
                        cur_slot_pv_point = pv_points[i_bz, 0, :]  # use the same pv-point for all slots...
                        cur_slot_central_point = central_points[i_bz, i_s, :]  # for central point
                        if self.pred_pv_equiv:
                            # cur_slot_pv_point = slot_recon_pivot_points_equiv[i_bz, i_s]
                            # cur_slot_central_point = slot_recon_central_points_equiv[i_bz, i_s]
                            cur_slot_pv_point = slot_recon_pivot_points_equiv[i_bz, 0]
                            cur_slot_central_point = slot_recon_central_points_equiv[i_bz, i_s]
                        else:
                            cur_slot_pv_point = cur_slot_pv_point.unsqueeze(0)
                            cur_slot_central_point = cur_slot_central_point.unsqueeze(0)

                            ''' Central point related... '''
                            ##### if use the offset as a term for central points in each anchor? #####
                            # if self.shape_type in ['washing_machine']:
                            #     cur_bz_cur_slot_avg_offset = slot_avg_offset[i_bz, i_s, :].unsqueeze(0)
                            #     # cur_bz_cur_slot_avg_offset: na x 3
                            #     cur_bz_cur_slot_avg_offset = torch.matmul(safe_transpose(cur_anchors, -1, -2), cur_bz_cur_slot_avg_offset.unsqueeze(-1)).squeeze(-1)
                            #     cur_slot_central_point = cur_slot_central_point + cur_bz_cur_slot_avg_offset
                            ''' Central point related... '''

                        cur_bz_cur_slot_R = slot_R[i_bz, i_s]  # na x 3 x 3
                        # Get translations
                        # R(P + c_p - p_v) + p_v; R(P + p_c - p_v) + p_v; R(P + p_c - p_v) + p_v
                        cur_slot_trans = 1.0 * torch.matmul(cur_bz_cur_slot_R,
                                                            (cur_slot_central_point - cur_slot_pv_point).unsqueeze(
                                                                -1)) + (cur_slot_pv_point).unsqueeze(-1)
                        cur_slot_trans = cur_slot_trans.squeeze(-1).unsqueeze(0).unsqueeze(0)

                        cur_slot_trans_joint = 1.0 * torch.matmul(cur_bz_cur_slot_R,
                                                            (- cur_slot_pv_point).unsqueeze(
                                                                -1)) + (cur_slot_pv_point).unsqueeze(-1)
                        cur_slot_trans_joint = cur_slot_trans_joint.squeeze(-1).unsqueeze(0).unsqueeze(0)
                        ''' V3 -- For base part: Use both pvp and central p for translation prediction '''
                    else:
                        if self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:
                            # use the self's predicted pv_points
                            cur_slot_pv_point = pv_points[i_bz, 0, :]
                        else:
                            cur_slot_pv_point = pv_points[i_bz, :, consumed_pv_idx[i_bz]]  # 3 --> for pv points
                        if self.pred_pv_equiv:
                            # cur_slot_pv_point = slot_recon_pivot_points_equiv[i_bz, i_s]
                            cur_slot_pv_point = slot_recon_pivot_points_equiv[i_bz, 0]
                        else:
                            cur_slot_pv_point = cur_slot_pv_point.unsqueeze(0)

                        consumed_pv_idx[i_bz] = consumed_pv_idx[i_bz] + 1
                        # trans = -R * pv_point + pv_point
                        cur_bz_cur_slot_R = slot_R[i_bz, i_s]  # na x 3 x 3 # translations
                        # cur_slot_trans: na x 3
                        # anchor_R * (-ori_R * pv + pv)
                        if self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:
                            # predicted central point
                            cur_slot_central_point = central_points[i_bz, i_s, :]  # (3,) --> central points
                            if self.pred_pv_equiv:
                                cur_slot_central_point = slot_recon_central_points_equiv[i_bz, i_s]
                            else:
                                cur_slot_central_point = cur_slot_central_point.unsqueeze(0)

                                if self.shape_type in ['washing_machine']:
                                    cur_bz_cur_slot_avg_offset = slot_avg_offset[i_bz, i_s, :].unsqueeze(0)
                                    # cur_bz_cur_slot_avg_offset: na x 3
                                    cur_bz_cur_slot_avg_offset = torch.matmul(safe_transpose(cur_anchors, -1, -2),
                                                                              cur_bz_cur_slot_avg_offset.unsqueeze(
                                                                                  -1)).squeeze(-1)
                                    cur_slot_central_point = cur_slot_central_point + cur_bz_cur_slot_avg_offset

                            # R(P - p_v) + p_v + c_p = -Rp_cv + p_v + c_p
                            # predicted slot translation...

                            # R(P + p_c - p_v) + p_v; translation for slots
                            cur_slot_trans = 1.0 * torch.matmul(cur_bz_cur_slot_R,
                                                                (cur_slot_central_point - cur_slot_pv_point).unsqueeze(
                                                                    -1)) + (cur_slot_pv_point).unsqueeze(-1)  #

                            cur_slot_trans_joint = 1.0 * torch.matmul(cur_bz_cur_slot_R,
                                                                      (- cur_slot_pv_point).unsqueeze(
                                                                          -1)) + (cur_slot_pv_point).unsqueeze(-1)
                            # cur_slot_trans_joint = cur_slot_trans_joint.squeeze(-1).unsqueeze(0).unsqueeze(0)
                        else:
                            # pv_point related translation vector;
                            cur_slot_trans = -1.0 * torch.matmul(cur_bz_cur_slot_R, cur_slot_pv_point.unsqueeze(
                                -1)) + cur_slot_pv_point.unsqueeze(-1)
                            cur_slot_trans_joint = 1.0 * torch.matmul(cur_bz_cur_slot_R,
                                                                      (- cur_slot_pv_point).unsqueeze(
                                                                          -1)) + (cur_slot_pv_point).unsqueeze(-1)
                            # cur_slot_trans_joint = cur_slot_trans_joint.squeeze(-1).unsqueeze(0).unsqueeze(0)
                            # pv_point related translation vector
                        cur_slot_trans = cur_slot_trans.squeeze(-1)  # translation for slots
                        cur_slot_trans = cur_slot_trans.unsqueeze(0).unsqueeze(0)
                        cur_slot_trans_joint = cur_slot_trans_joint.squeeze(-1).unsqueeze(0).unsqueeze(0)
                    tot_cur_slot_T.append(cur_slot_trans)  #
                    tot_cur_slot_T_joint.append(cur_slot_trans_joint)
                tot_cur_slot_T = torch.cat(tot_cur_slot_T, dim=0)
                tot_cur_slot_T_joint = torch.cat(tot_cur_slot_T_joint, dim=0)
                slot_T.append(tot_cur_slot_T)
                slot_T_joint.append(tot_cur_slot_T_joint)
            # slot_T: bz x ns x na x 3
            slot_T = torch.cat(slot_T, dim=1)
            # print("slot_T", slot_T.size())
            slot_T_ori = slot_T.clone()
            # slot_T_joint...
            slot_T_joint = torch.cat(slot_T_joint, dim=1)
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

            #### Shape type --> drawer #####
            if self.shape_type == 'drawer':  # drawer; drawer...
                # slot_T[:, 0] = 0.0 # set the translation of the first slot to zero... set the other to xxx...; the
                slot_T[:, 0] = slot_T[:, 0] * 0.0  # fix points of the first slot
                slot_T_joint[:, 0] = slot_T_joint[:, 0] * 0.0  # fix points of the first slot

            k = self.kpconv_kanchor if self.sel_mode_trans is None else 1
            if self.sel_mode_trans is not None:  # select a specific mode for transformation
                topk_anchor_idxes = torch.tensor([self.sel_mode_trans], dtype=torch.long).cuda().unsqueeze(0).unsqueeze(
                    0).repeat(bz, self.num_slots, 1).contiguous()

            # transformed_pts: bz x n_s x na x M x 3
            # slot_canon_pts: bz x n_s x M x 3
            # slot_R: bz x n_s x na x 3 x 3 @ slot_recon_pts: bz x n_s x 1 x 3 x M
            # transformed points
            transformed_pts = safe_transpose(torch.matmul(slot_R, safe_transpose(slot_canon_pts.unsqueeze(2), -1, -2)), -1, -2) + slot_T.unsqueeze(-2)

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
                    defined_axises = batched_index_select(values=defined_axises, indices=topk_anchor_idxes, dim=1)

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
            # ori_to_recon: dist_recon_ori: bz x n_s x na x M x N --> bz x n_s x na x N
            minn_dist_ori_to_recon, _ = torch.min(dist_recon_ori, dim=-2)
            # minn_dist_ori_to_recon = minn_dist_ori_to_recon
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
                orbit_slot_dist_ori_recon = minn_dist_ori_to_recon  # single direction chamfer distance # minn distance ori_to_recon ---
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
            # mult by slot indicators.... whether it should be restricted by slot indicators? --- Yes!!!
            # slot_dist_ori_recon = (slot_dist_ori_recon * (hard_one_hot_labels.sum(1) > 0.5).float()).sum(-1)

            # use (1) in-cluster points ori_to_recon loss + (2) recon_to_all_ori_pts recon loss for optimization
            # orbit_slot_dist_ori_recon_all_pts = minn_dist_ori_to_recon + minn_dist_recon_to_ori_all_pts

            #### slot distance recon all pts ####
            # orbit_slot_dist_ori_recon_all_pts = minn_dist_ori_to_recon + minn_dist_recon_to_ori

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

            ''' Add loss for transformed pv points, ori transformed points, and predicted axises '''
            if self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:
                # Canonical reconstruction and
                # original transformed points
                transformed_pts_ori = batched_index_select(transformed_pts_ori, indices=slot_orbits.unsqueeze(-1),
                                                           dim=2).squeeze(2)
                # transformed pv points...
                transformed_pv_points_ori = batched_index_select(transformed_pv_points_ori,
                                                                 indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                # transformed_pv_points = batched_index_select(transformed_pv_points, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                # selected_anchors: bz x n_s x k_a x 3 x 3 --> bz x n_s x 3 x 3
                # selected_anchors = batched_index_select(values=selected_anchors, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)

                seg_label_to_inv_transformed_pts = {}
                # tot_seg_to_idxes: [bz][i_s] --> for further points indexes in this segmentation extraction...

                if self.pred_pv_equiv:
                    # print("Using equiv!")
                    central_transformed_pv_points_equiv = batched_index_select(central_transformed_pv_points_equiv,
                                                                               indices=slot_orbits.unsqueeze(-1),
                                                                               dim=2).squeeze(2)
                    central_transformed_pv_points = central_transformed_pv_points_equiv

                    # slot_recon_pivot_points_equiv: bz x n_s x 3
                    slot_recon_pivot_points_equiv = batched_index_select(slot_recon_pivot_points_equiv,
                                                                         indices=slot_orbits.unsqueeze(-1),
                                                                         dim=2).squeeze(2)
                    # slot_recon_central_points_equiv: bz x n_s x 3
                    slot_recon_central_points_equiv = batched_index_select(slot_recon_central_points_equiv,
                                                                           indices=slot_orbits.unsqueeze(-1),
                                                                           dim=2).squeeze(2)
                    slot_pivot_points = slot_recon_pivot_points_equiv
                    central_points = slot_recon_central_points_equiv  # get slot recon central points
                # transformed_central_pv_points = batched_index_select(transformed_central_pv_points, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                # todo: we can just use y-axis here for rotation matrix calculation
                ''' Get predicted axises for the selected mode '''
                # slot_axis: bz x n_s x 3
                slot_axis = batched_index_select(values=slot_axis, indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
                defined_axises = batched_index_select(values=defined_axises, indices=slot_orbits[:, 0].unsqueeze(-1),
                                                      dim=1).squeeze(1)

                ''' Mean of the axis for further projection and other usages '''
                # avg_slot_axis = torch.mean(slot_axis, dim=1, keepdim=True)
                ''' Mean of the axis for further projection and other usages '''

                ''' The first pair-slot's axis for further projection and other usages '''  #
                # slot_pivot_points (here the first predicted pv point); central point; cuboid x; cuboid r; avg slot axis (direction of the predicted axis); normals... we can define them...
                # from axis to cuboid constraints? # pv point -> (inv R, central point) -> inv-transformed pv point; the nearest face (via points' inv-transformed coordinate); inv transformed axis (by applying inv-R to the predicted rotation axis) -> dot product with the normal vector of the nearest face...
                ### use axis of the first slot as defined avg slot ###
                avg_slot_axis = slot_axis[:, 0]  # avg_slot_axis...
                # detach the defined axis? ---- # detach defined axis...
                ''' Detach the axis... '''

                # avg_slot_axis = defined_axises.detach()  # not optimize the loss in the
                ''' Not detach the axis... '''
                avg_slot_axis = defined_axises # .detach() # not optimize the loss in the # whether to use detach is not
                ''' The first pair-slot's axis for further projection and other usages '''

                # slot_cuboic_recon_pts = torch.cat(slot_cuboic_recon_pts, dim=1)
                #                 slot_cuboic_R = torch.cat(slot_cuboic_R, dim=1)

                minn_other_slots_base_canon_dist = 0.0  # other slots'
                if self.recon_prior == 7:
                    # dot_normal_axis:
                    dot_normal_axis = get_cuboic_constraint_loss_with_axis_cuboid(slot_pivot_points,
                                                                                  slot_central_points,
                                                                                  torch.cat(slot_cuboic_recon_pts,
                                                                                            dim=1),
                                                                                  torch.cat(slot_cuboic_R, dim=1),
                                                                                  avg_slot_axis)
                    minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + dot_normal_axis.mean()

                # todo: make predicted pivot points close to central points transformed slot canonical points
                # slot_canon_pts: bz x n_s x M x 3 --> central_transformed_canon_points: bz x n_s x M x 3
                # with central points --> no rotation is involved in the process #

                central_transformed_canon_points = slot_canon_pts + central_points.unsqueeze(-2)  # central point transformed part shape reconstruction
                # print(
                # f"slot_canon_pts: {slot_canon_pts.size()}, central_points: {central_points.size()}, central_transformed_canon_points: {central_transformed_canon_points.size()}, defined_axises: {defined_axises.size()}")
                # rotation is involved...
                # central_transformed_canon_points = transformed_pts_ori.detach() if self.pred_axis else transformed_pts_ori
                canon_transformed_points = transformed_pts_ori  # transformed points in the canonical space

                if self.pred_axis:
                    central_transformed_canon_points = central_transformed_canon_points.detach()
                    canon_transformed_points = canon_transformed_points.detach()

                # bz x n_s x 3 --> bz x 3;
                slot_recon_pv_points = slot_pivot_points[:, 0, :]  # pivot points...
                ### try to detach them ###
                # slot_recon_pv_points = slot_recon_pv_points.detach()

                ''' First point -- the predicted pivot point & central transformed canon points '''
                # dist_pv_points_central_transformed_canon_pts: bz x n_s x M;
                # dist_pv_points_central_transformed_canon_pts = torch.sum((central_transformed_canon_points.detach() - slot_recon_pv_points.unsqueeze(1).unsqueeze(-2)) ** 2, dim=-1)
                #### Distance between pv point and central point transformed points #####
                dist_pv_points_central_transformed_canon_pts = torch.sum(
                    (central_transformed_canon_points - slot_recon_pv_points.unsqueeze(1).unsqueeze(-2)) ** 2,
                    dim=-1)
                # minn_dist_pv_points_central_transformed_canon_pts: bz x n_s
                minn_dist_pv_points_central_transformed_canon_pts, _ = torch.min(
                    dist_pv_points_central_transformed_canon_pts, dim=-1)

                ''' Get and update valid distance... '''
                if self.axis_reg_stra == 1:
                    if self.pv_in_count_nn > 0:
                        # valid_pv_points
                        cur_in_count_pv_point_indicators = (
                                minn_dist_pv_points_central_transformed_canon_pts <= self.avg_pv_point_distance.unsqueeze(
                            0)).float()
                        cur_valid_pv_nn = torch.sum(cur_in_count_pv_point_indicators).item()
                        minn_dist_pv_points_central_transformed_canon_pts[cur_in_count_pv_point_indicators < 0.5] = 0.0
                        self.avg_pv_point_distance = (self.avg_pv_point_distance * self.pv_in_count_nn + torch.sum(
                            minn_dist_pv_points_central_transformed_canon_pts)) / (
                                                                 self.pv_in_count_nn + cur_valid_pv_nn)
                        self.pv_in_count_nn = self.pv_in_count_nn + cur_valid_pv_nn
                    else:
                        cur_valid_pv_nn = minn_dist_pv_points_central_transformed_canon_pts.size(
                            0) * minn_dist_pv_points_central_transformed_canon_pts.size(1)
                        self.avg_pv_point_distance = torch.sum(
                            minn_dist_pv_points_central_transformed_canon_pts) / cur_valid_pv_nn
                        self.pv_in_count_nn = self.pv_in_count_nn + cur_valid_pv_nn
                ''' Get and update valid distance... '''

                minn_dist_pv_points_central_transformed_canon_pts = minn_dist_pv_points_central_transformed_canon_pts.sum(
                    dim=-1).mean()
                minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + minn_dist_pv_points_central_transformed_canon_pts
                ##### Distance between pv point and canon transformed points #####

                ##### Distance between pv point and canon transformed points #####
                dist_pv_points_canon_transformed_canon_pts = torch.sum(
                    (canon_transformed_points - slot_recon_pv_points.unsqueeze(1).unsqueeze(-2)) ** 2,
                    dim=-1)
                # minn_dist_pv_points_central_transformed_canon_pts: bz x n_s
                minn_dist_pv_points_canon_transformed_canon_pts, _ = torch.min(
                    dist_pv_points_canon_transformed_canon_pts, dim=-1)

                ''' Get and update valid distance... '''
                if self.axis_reg_stra == 1:
                    if self.pv_in_count_nn > 0:
                        # valid_pv_points
                        cur_in_count_pv_point_indicators = (
                                    minn_dist_pv_points_canon_transformed_canon_pts <= self.avg_pv_point_distance.unsqueeze(
                                0)).float()
                        cur_valid_pv_nn = torch.sum(cur_in_count_pv_point_indicators).item()
                        minn_dist_pv_points_canon_transformed_canon_pts[cur_in_count_pv_point_indicators < 0.5] = 0.0
                        self.avg_pv_point_distance = (self.avg_pv_point_distance * self.pv_in_count_nn + torch.sum(
                            minn_dist_pv_points_canon_transformed_canon_pts)) / (self.pv_in_count_nn + cur_valid_pv_nn)
                        self.pv_in_count_nn = self.pv_in_count_nn + cur_valid_pv_nn
                    else:
                        cur_valid_pv_nn = minn_dist_pv_points_canon_transformed_canon_pts.size(
                            0) * minn_dist_pv_points_canon_transformed_canon_pts.size(1)
                        self.avg_pv_point_distance = torch.sum(
                            minn_dist_pv_points_canon_transformed_canon_pts) / cur_valid_pv_nn
                        self.pv_in_count_nn = self.pv_in_count_nn + cur_valid_pv_nn
                ''' Get and update valid distance... '''

                minn_dist_pv_points_canon_transformed_canon_pts = minn_dist_pv_points_canon_transformed_canon_pts.sum(
                    dim=-1).mean()
                minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + minn_dist_pv_points_canon_transformed_canon_pts
                ##### Distance between pv point and canon transformed points #####

                ''' First point -- the predicted pivot point & central transformed canon points '''
                nn = 10  # if self.shape_type not in ['washing_machine'] else 5
                dists = [0.02 * _ for _ in range(1, nn + 1)]

                # joint_len = 0.3 if self.shape_type not in ['washing_machine'] else 0.10
                joint_len = 0.30  # 0.15 # for washing machine?
                ''' For partial point clouds? '''
                # joint_len = 0.10 # 0.15 # for washing machine? #
                # assume the length of the joint is 0.3
                # then we randomly gneerate 10 points in the 0.3 range
                randomized_dists = torch.randint(1, int(joint_len * 100) + 1, (nn,))
                randomized_dists = randomized_dists.float() / 100.

                if self.shape_type in ['washing_machine']:
                    randomized_dists = randomized_dists - joint_len / 2.0

                dists = randomized_dists.tolist()

                ''' Detach the pivot point for further computing '''
                # slot_recon_pv_points = slot_recon_pv_points.detach() # detach

                for dis in dists:
                    ''' First point -- the predicted pivot point & central transformed canon points '''
                    # dist_pv_points_central_transformed_canon_pts: bz x n_s x M; # avg_slot_axis?
                    shift_slot_recon_pv_points = slot_recon_pv_points - dis * avg_slot_axis.squeeze(1)  #
                    # dist_pv_points_central_transformed_canon_pts = torch.sum(
                    #     (central_transformed_canon_points.detach() - shift_slot_recon_pv_points.unsqueeze(1).unsqueeze(-2)) ** 2,
                    #     dim=-1)
                    ##### Distance between shifted pv point and central point transformed points #####
                    dist_pv_points_central_transformed_canon_pts = torch.sum(
                        (central_transformed_canon_points - shift_slot_recon_pv_points.unsqueeze(1).unsqueeze(
                            -2)) ** 2,
                        dim=-1)
                    # minn_dist_pv_points_central_transformed_canon_pts: bz x n_s
                    secon_minn_dist_pv_points_central_transformed_canon_pts, _ = torch.min(
                        dist_pv_points_central_transformed_canon_pts, dim=-1)

                    if self.axis_reg_stra == 1:
                        if self.pv_in_count_nn > 0:
                            # valid_pv_points
                            cur_in_count_pv_point_indicators = (
                                    secon_minn_dist_pv_points_central_transformed_canon_pts <= self.avg_pv_point_distance.unsqueeze(
                                0)).float()
                            cur_valid_pv_nn = torch.sum(cur_in_count_pv_point_indicators).item()
                            secon_minn_dist_pv_points_central_transformed_canon_pts[
                                cur_in_count_pv_point_indicators < 0.5] = 0.0
                            self.avg_pv_point_distance = (self.avg_pv_point_distance * self.pv_in_count_nn + torch.sum(
                                secon_minn_dist_pv_points_central_transformed_canon_pts)) / (
                                                                 self.pv_in_count_nn + cur_valid_pv_nn)
                            self.pv_in_count_nn = self.pv_in_count_nn + cur_valid_pv_nn
                        else:
                            cur_valid_pv_nn = secon_minn_dist_pv_points_central_transformed_canon_pts.size(
                                0) * secon_minn_dist_pv_points_central_transformed_canon_pts.size(1)
                            self.avg_pv_point_distance = torch.sum(
                                secon_minn_dist_pv_points_central_transformed_canon_pts) / cur_valid_pv_nn
                            self.pv_in_count_nn = self.pv_in_count_nn + cur_valid_pv_nn

                    secon_minn_dist_pv_points_central_transformed_canon_pts = secon_minn_dist_pv_points_central_transformed_canon_pts.sum(
                        dim=-1).mean()
                    minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + secon_minn_dist_pv_points_central_transformed_canon_pts
                    ##### Distance between shifted pv point and central point transformed points #####

                    ''' Distance between shifted pivot points and canonical transformed points '''
                    dist_pv_points_canon_transformed_canon_pts = torch.sum(
                        (canon_transformed_points - shift_slot_recon_pv_points.unsqueeze(1).unsqueeze(-2)) ** 2,
                        dim=-1)
                    # minn_dist_pv_points_central_transformed_canon_pts: bz x n_s
                    minn_dist_pv_points_canon_transformed_canon_pts, _ = torch.min(
                        dist_pv_points_canon_transformed_canon_pts, dim=-1)
                    ''' Distance between shifted pivot points and canonical transformed points '''

                    if self.axis_reg_stra == 1:
                        if self.pv_in_count_nn > 0:
                            # valid_pv_points
                            cur_in_count_pv_point_indicators = (
                                    minn_dist_pv_points_canon_transformed_canon_pts <= self.avg_pv_point_distance.unsqueeze(
                                0)).float()
                            cur_valid_pv_nn = torch.sum(cur_in_count_pv_point_indicators).item()
                            minn_dist_pv_points_canon_transformed_canon_pts[
                                cur_in_count_pv_point_indicators < 0.5] = 0.0
                            self.avg_pv_point_distance = (self.avg_pv_point_distance * self.pv_in_count_nn + torch.sum(
                                minn_dist_pv_points_canon_transformed_canon_pts)) / (
                                                                 self.pv_in_count_nn + cur_valid_pv_nn)
                            self.pv_in_count_nn = self.pv_in_count_nn + cur_valid_pv_nn
                        else:
                            cur_valid_pv_nn = minn_dist_pv_points_canon_transformed_canon_pts.size(
                                0) * minn_dist_pv_points_canon_transformed_canon_pts.size(1)
                            self.avg_pv_point_distance = torch.sum(
                                minn_dist_pv_points_canon_transformed_canon_pts) / cur_valid_pv_nn
                            self.pv_in_count_nn = self.pv_in_count_nn + cur_valid_pv_nn

                    minn_dist_pv_points_canon_transformed_canon_pts = minn_dist_pv_points_canon_transformed_canon_pts.sum(
                        dim=-1).mean()
                    minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist + minn_dist_pv_points_canon_transformed_canon_pts
                    ''' First point -- the predicted pivot point & central transformed canon points '''
                minn_other_slots_base_canon_dist = minn_other_slots_base_canon_dist / (float(nn) / 4.)

            # slot_canon_pts = batched_index_select(safe_transpose(slot_canon_pts, -1, -2), indices=slot_orbits.unsqueeze(-1), dim=2).squeeze(2)
            # slot_canon_pts: bz x n_s x 3 x M --> bz x n_s x M x 3

            slot_canon_pts = safe_transpose(slot_canon_pts, -1, -2)



            # slot_orbits --> bz x n_s
            # minn_dist_ori_to_recon: bz x n_s x na x N ---> bz x n_s x N
            selected_minn_dist_ori_to_recon = batched_index_select(values=minn_dist_ori_to_recon, indices=slot_orbits.long().unsqueeze(-1), dim=2).squeeze(2)
            # selected_minn_dist_ori_to_recon: bz x N
            selected_minn_dist_ori_to_recon, _ = torch.min(selected_minn_dist_ori_to_recon, dim=1)
            # selected_minn_dist_ori_to_recon: bz x N
            ori_to_recon = torch.sqrt(selected_minn_dist_ori_to_recon).mean(dim=-1).mean()
            # print(f"ori_to_recon, uni L1: {float(ori_to_recon.item())}")
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

            ''' Global reconstruction distance for optimization '''
            if self.slot_single_cd == 1:
                glb_recon_ori_dist = ori_to_recon_dist
            else:
                glb_recon_ori_dist = recon_to_ori_dist + ori_to_recon_dist
            ''' Global reconstruction distance for optimization '''

            tot_recon_loss = glb_recon_ori_dist * self.glb_recon_factor + (
                slot_dist_ori_recon) * self.slot_recon_factor + slot_pv_canon_cd_loss  # add slot_pv_canon loss to the tot_recon_loss term
            if self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:
                tot_recon_loss = tot_recon_loss + minn_other_slots_base_canon_dist

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
                # cuboid reconstruction
                # if normals is not None:
                #     slot_recon_cuboic_constraint_loss = get_cuboic_constraint_loss_with_normals(
                #         slot_R, slot_T, ori_pts, normals, slot_cuboic_recon_pts, slot_cuboic_R, hard_one_hot_labels, attn_ori
                #     )
                # else:
                #     slot_recon_cuboic_constraint_loss = get_cuboic_constraint_loss(
                #         slot_R, slot_T, ori_pts, slot_cuboic_recon_pts, slot_cuboic_R, hard_one_hot_labels, attn_ori
                #     )
                ''' Get cuboid reconstruction: whether to consider normals in the calculation process or not '''

                forb_slot_idx = None  #
                # forb_slot_idx = 0 # Add cuboid constraint for predicted points coordinates
                slot_recon_cuboic_constraint_loss = get_cuboic_constraint_loss(
                    slot_R, slot_T, ori_pts, slot_cuboic_recon_pts, slot_cuboic_R, hard_one_hot_labels, attn_ori,
                    forb_slot_idx=forb_slot_idx
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

            if self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:
                out_feats['recon_slot_pts_hard_wo_glb_rot'] = transformed_pts_ori.detach().cpu().numpy()
                out_feats['pv_pts_hard_wo_glb_rot'] = transformed_pv_points_ori.detach().cpu().numpy()
                # Get selected_anchors!
                out_feats['selected_anchors'] = selected_anchors.detach().cpu().numpy()

            if self.recon_prior == 5 or self.recon_prior == 7:
                ##### Register predicted cuboid boundary points for slots #####
                out_feats['slot_cuboic_recon_pts'] = slot_cuboic_recon_pts.detach().cpu().numpy()
                ##### Register predicted cuboid rotation matrix for slots #####
                out_feats['slot_cuboic_R'] = slot_cuboic_R.detach().cpu().numpy()
            elif self.recon_prior == 6 or self.recon_prior == 7 or self.recon_prior == 8 or self.recon_prior == 9:  # slot pivot points --- for pivot points and others
                out_feats['slot_pivot_points'] = slot_pivot_points.detach().cpu().numpy()
                # central_points: bz x n_s x 3
                out_feats['slot_central_points'] = central_points.detach().cpu().numpy()
                out_feats['slot_axis'] = slot_axis.detach().cpu().numpy()
                real_defined_axises = torch.matmul(selected_anchors, defined_axises.unsqueeze(-1)).squeeze(-1)
                out_feats['defined_axises'] = defined_axises.detach().cpu().numpy() # defined_axises:
                out_feats['real_defined_axises'] = real_defined_axises.detach().cpu().numpy() # defined_axises:
                self.real_defined_axises = real_defined_axises.clone()

                # processed_feats: bz x dim x N x na
                selected_processed_feats = processed_feats.contiguous().permute(0, 3, 1, 2).contiguous()
                selected_processed_feats = batched_index_select(selected_processed_feats, indices=slot_orbits[:, 0].long().unsqueeze(-1), dim=1).contiguous().squeeze(1)
                out_feats[f'selected_processed_feats_iter_{cur_iter}'] = selected_processed_feats.detach().cpu().numpy()

                # out_feats[f"processed_feats_ori_iter_{cur_iter}"] = processed_feats_ori.detach().cpu().numpy()

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
            # canonicalized in the space...

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

            self.pred_R = selected_pred_R_saved
            self.pred_T = selected_pred_T_saved
            self.defined_axises = defined_axises.clone()
            self.offset_pivot_points = offset_pivot_points.clone()

            out_feats['pred_R_slots'] = selected_pred_R_saved.cpu().numpy()
            out_feats['pred_T_slots'] = selected_pred_T_saved.cpu().numpy()
            out_feats['pv_points'] = pv_points.detach().cpu().numpy()

            self.out_feats = out_feats

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

    def forward(self, x, pose, ori_pc=None, rlabel=None, nn_inter=2, pose_segs=None, canon_pc=None, normals=None,
                canon_normals=None):

        ''' Set initial input per-point pose '''
        bz, N = x.size(0), x.size(2)
        init_pose = torch.zeros([bz, N, 4, 4], dtype=torch.float32).cuda()
        init_pose[..., 0, 0] = 1.
        init_pose[..., 1, 1] = 1.
        init_pose[..., 2, 2] = 1.


        loss = 0.0
        cur_transformed_points = x
        cur_estimated_pose = init_pose
        out_feats_all_iters = {}
        cur_selected_pts_orbit = None

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
                # print("cur_transformed_points.avg", torch.mean(cur_transformed_points, dim=-1))
                cur_loss, cur_estimated_pose, cur_inv_transformed_x_list, cur_selected_labels = self.forward_one_iter(
                    cur_transformed_points, cur_estimated_pose, x_list=cur_inv_transformed_x_list, hard_label=cur_selected_labels, ori_pc=ori_pc, rlabel=rlabel, cur_iter=i_iter, gt_pose=pose, gt_pose_segs=pose_segs, canon_pc=canon_pc, selected_pts_orbit=cur_selected_pts_orbit)
                loss += cur_loss
                out_feats_all_iters[i_iter] = self.out_feats
            loss = loss / self.num_iters

        # np.save(self.log_fn + f"_n_stage_{self.stage}_all_iters.npy", out_feats_all_iters)
        # self.out_feats_all_iters = out_feats_all_iters
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
        out_mlps = [1024]
        # out_mlps = [512]
    else:
        mlps = [[64], [128], [512]]
        # mlps = [[64], [128], [256]] # you need to
        # out_mlps = [512]
        out_mlps = [256]
        # mlps = [[32, 32], [64, 64], [128, 128], [256, 256]]
        # out_mlps = [128, 128]

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
        'use_art_mode': opt.equi_settings.use_art_mode,
        'with_part_proposal': opt.equi_settings.with_part_proposal,
        # 'opt': opt

    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model


def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
