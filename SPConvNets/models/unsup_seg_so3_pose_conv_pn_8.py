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

from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
from model_util import farthest_point_sampling, DecoderFC
from SPConvNets.models.PointNet2 import PointnetPP


class PartDecoder(nn.Module):

    def __init__(self, feat_len, recon_M=128):
        super(PartDecoder, self).__init__()

        self.mlp1 = nn.Linear(feat_len + 3, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, 3)

        self.recon_M = recon_M

    def forward(self, feat, pc):
        num_point = pc.shape[0]
        batch_size = feat.shape[0]

        net = torch.cat([feat.unsqueeze(dim=1).repeat(1, num_point, 1), \
                         pc.unsqueeze(dim=0).repeat(batch_size, 1, 1)], dim=-1)

        net = F.leaky_relu(self.mlp1(net))
        net = F.leaky_relu(self.mlp2(net))
        net = self.mlp3(net)

        if self.recon_M < num_point:
            fps_idx = farthest_point_sampling(net, self.recon_M)
            net = net.contiguous().view(
                batch_size * num_point, -1)[fps_idx, :].contiguous().view(batch_size, self.recon_M, -1)

        return net


class ClsSO3ConvModel(nn.Module): # SO(3) equi-conv-network #
    def __init__(self, params):
        super(ClsSO3ConvModel, self).__init__()

        ''' Construct backbone model '''
        # # get backbone model
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']: # backbone
            self.backbone.append(M.BasicSO3PoseConvBlock(block_param))
        print(f"number of convs in the backbone: {len(self.backbone)}")
        # self.outblock = M.ClsOutBlockR(params['outblock'])
        self.outblock = M.ClsOutBlockPointnet(params['outblock'], down_task=False)
        self.glb_outblock = Mso3.ClsOutBlockPointnet(params['outblock'])
        # output classification block

        ''' Set useful arguments '''
        ### Specific model-related parameter settings ####
        self.encoded_feat_dim = params['outblock']['dim_in']
        self.kanchor = params['outblock']['kanchor']
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

        #### Set parameter alias ####
        self.recon_part_M = self.part_pred_npoints # 128, 256, 512, 1024
        self.transformation_dim = 7

        pts_to_real_pts = {128: 146, 256: 258, 512: 578}

        self.log_fn = f"{self.shape_type}_out_feats_weq_wrot_{self.global_rot}_rel_rot_factor_{self.rot_factor}_equi_{self.use_equi}_model_{self.model_type}_decoder_{self.decoder_type}_inv_attn_{self.inv_attn}_orbit_attn_{self.orbit_attn}_slot_iters_{self.slot_iters}_topk_{self.topk}_num_iters_{self.num_iters}_npts_{self.npoints}_perpart_npts_{self.part_pred_npoints}_bsz_{self.batch_size}_init_lr_{self.init_lr}"
        # self.log_fn = os.path.join("/share/xueyi/", self.log_fn)

        ''' Set PointNet model '''
        # self.model = PointnetPP(in_feat_dim=3)
        # self.backbone = PointnetPP(in_feat_dim=3)
        # output: bz x N x feat_dim
        # self.encoded_feat_dim = 128

        ''' Construct segmentation prediction module '''
        # After that, apply softmax to the otuput feature
        # Number of slot: the number of segmentation for prediction
        self.seg_net_aa = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim // 2, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 2),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(in_channels=self.encoded_feat_dim // 2, out_channels=self.num_slots,
            #           kernel_size=(1, 1), stride=(1, 1), bias=True)
        nn.Conv2d(in_channels=self.encoded_feat_dim // 2, out_channels=self.num_slots,
                  kernel_size=(1, 1), stride=(1, 1), bias=True)
        )

        ''' Construct NPCS prediction module '''
        # Need to minus 0.5 for predicting centralized coordinates
        self.npcs_net = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim // 2, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 2, out_channels=self.num_slots * 3,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.Sigmoid()
        )

        ''' Construct NPCS prediction module '''
        # Need to minus 0.5 for predicting centralized coordinates
        self.axis_net = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim // 4, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 4, out_channels=3,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            # nn.Sigmoid()
        )

        self.pvp_net = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim // 4, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 4, out_channels=3 * (self.num_slots - 1),
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            # nn.Sigmoid()
        )

        self.anchors = torch.from_numpy(L.get_anchors(params['outblock']['kanchor'])).cuda()

    def forward_one_iter(self, x, npcs_pc, pose, real_pose): # rotation label

        bz = x.size(0)
        npoints = x.size(2)

        ori_pts = x.clone()
        ''' Preprocess input to get input for the network '''
        # pose:
        x = M.preprocess_input(x, self.kanchor, pose, False)
        for block_i, block in enumerate(self.backbone):
            # x = block(x, seg=seg)
            x = block(x)

        glb_feats, out_feats, _ = self.glb_outblock(x)
        if len(out_feats.size()) == 1:
            out_feats = out_feats.unsqueeze(1)

        # we should use global

        ''' Use per-point aggregation '''
        # feats, out_feats = self.outblock(x)
        # if len(feats.size()) == 4:
        #     feats = feats.squeeze(-1)
        ''' Use per-point aggregation '''

        # psoe
        rots = real_pose[:, :, :3, :3]
        # self.anchors: na x 3 x 3
        # rots: bz x N x 3 x 3
        # dot_rots_anchors: bz x N x na x 3 x 3
        dot_rots_anchors = torch.matmul(rots.unsqueeze(2), safe_transpose(self.anchors, -1, -2).unsqueeze(0).unsqueeze(0))
        dot_traces = dot_rots_anchors[..., 0, 0] + dot_rots_anchors[..., 1, 1] + dot_rots_anchors[..., 2, 2]
        # gt_orbit: bz x N # we can use
        gt_orbit = torch.argmax(dot_traces, dim=-1)
        # gt_orbit: bz
        gt_orbit = gt_orbit[:, 0]


        # out_feats = safe_transpose(out_feats, 1, 2)

        # pred_orbits: bz
        pred_orbits = torch.argmax(out_feats, dim=1)
        # x.feats:

        if self.kanchor > 1:
            # out_feats: bz x N x na
            # print(f"out_feats.size: {out_feats.size()}, gt_orbit.size: {gt_orbit.size()}")
            oribt_selection_loss = nn.functional.nll_loss(nn.functional.log_softmax(out_feats, dim=1), gt_orbit)
            oribt_selection_loss = oribt_selection_loss.mean()

        # x.feats: bz x dim x N x na
        # print(f"x.feats: {x.feats.size()}")
        feats = batched_index_select(x.feats.contiguous().permute(0, 3, 1, 2).contiguous(), indices=pred_orbits.unsqueeze(1), dim=1)
        feats = feats.squeeze(1)
        # print(f"feats: {feats.size()}")

        # sel_anchor: bz x 3 x 3
        sel_anchor = batched_index_select(values=self.anchors, indices=pred_orbits, dim=0)

        ''' If using the SPFN backbone '''
        # x: bz x 3 x npoints --> feats: bz x N x dim
        # feats, pos = self.backbone(None, safe_transpose(x, 1, 2))
        # feats = safe_transpose(feats, 1, 2)
        ''' If using the SPFN backbone '''

        glb_featts, _ = torch.max(feats, dim=-1)
        # bz x 3

        pred_axis = self.axis_net(glb_featts.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        pred_axis = pred_axis / torch.clamp(torch.norm(pred_axis, dim=-1, keepdim=True, p=2), min=1e-6)
        pred_axis = torch.matmul(sel_anchor, pred_axis.unsqueeze(-1)).squeeze(-1)
        self.pred_axis = pred_axis


        pred_pv = self.pvp_net(glb_featts.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        pred_pv = pred_pv.contiguous().view(pred_pv.size(0), -1, 3).contiguous()
        pred_pv = torch.matmul(sel_anchor.unsqueeze(1), pred_pv.unsqueeze(-1)).squeeze(-1)

        self.pred_pv = pred_pv




        # pred_seg: bz x n_s x N
        pred_seg = self.seg_net_aa(feats.unsqueeze(-1)).squeeze(-1)
        # pred_seg: bz x n_s x N
        pred_seg = torch.softmax(pred_seg, dim=1)

        # pred_labels: bz x N
        pred_labels = torch.argmax(pred_seg, dim=1)

        self.attn = pred_seg




        # pred_npcs: bz x (3 * n_s) x N
        pred_npcs = self.npcs_net(feats.unsqueeze(-1)).squeeze(-1)
        pred_npcs = pred_npcs - 0.5

        pred_npcs = safe_transpose(pred_npcs, 1, 2).contiguous().view(bz, npoints, self.num_slots, 3)
        pred_npcs = batched_index_select(values=pred_npcs, indices=pred_labels.unsqueeze(-1), dim=2).squeeze(2)
        pred_npcs = safe_transpose(pred_npcs, 1, 2)

        self.pred_npcs = pred_npcs


        # calculate NPCS prediction loss # predicted npcs
        loss_npcs = torch.sum((pred_npcs - safe_transpose(npcs_pc, 1, 2)) ** 2, dim=1).mean()
        # npcs multiplier is 10.0
        if self.kanchor > 1:
            print(f"current oribt_selection_loss: {oribt_selection_loss.mean().item()}")
            print(f"current loss_npcs: {loss_npcs.mean().item()}")
            loss_npcs = loss_npcs * 10.0 + oribt_selection_loss
        else:
            loss_npcs = loss_npcs * 10.0

        out_feats = {
            "pred_labels": pred_labels.detach().cpu().numpy(),
            "pred_npcs": pred_npcs.detach().cpu().numpy(),
            "real_npcs": npcs_pc.detach().cpu().numpy(),
            "pos": ori_pts.detach().cpu().numpy(),
        }
        np.save(self.log_fn + "_npcs.npy", out_feats)

        return loss_npcs, pred_seg

    def forward(self, x, pose, ori_pc=None, rlabel=None, nn_inter=2, pose_segs=None, canon_pc=None):

        # loss, attn = self.forward_one_iter(x, pose, rlabel=rlabel)
        # return loss, attn
        bz, np = x.size(0), x.size(2)
        init_pose = torch.zeros([bz, np, 4, 4], dtype=torch.float32).cuda()
        init_pose[..., 0, 0] = 1.;
        init_pose[..., 1, 1] = 1.;
        init_pose[..., 2, 2] = 1.
        # self.pose_segs = pose_segs
        loss_npcs, attn = self.forward_one_iter(x, npcs_pc=canon_pc, pose=init_pose, real_pose=pose)
        # self.attn = attn

        return loss_npcs

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
    # mlps[-1][-1] = 256 # 512
    # out_mlps[-1] = 256 # 512

    if opt.model.kanchor < 60:
        mlps = [[64,128], [256], [512], [1024]]
        # mlps = [[64,128], [256], [512]]
        # mlps = [[64], [128], [512]]
        # mlps = [[1024]]
        out_mlps = [1024]
        # out_mlps = [512]
    else:
        mlps = [[64], [128], [512]]
        out_mlps = [512]

    # initial_radius_ratio = 0.05
    # initial_radius_ratio = 0.15
    initial_radius_ratio = 0.20
    initial_radius_ratio = 0.20
    initial_radius_ratio = opt.equi_settings.init_radius
    print(f"Using initial radius: {initial_radius_ratio}")
    device = opt.device
    # print(f"opt.device: {device}")
    input_num = opt.model.input_num # 1024
    dropout_rate = opt.model.dropout_rate # default setting: 0.0
    # temperature
    temperature = opt.train_loss.temperature # set temperature
    so3_pooling = 'attention' #  opt.model.flag # model flag
    # opt.model.kpconv = 1
    na = 1 if opt.model.kpconv else opt.model.kanchor # how to represent rotation possibilities? --- sampling from the sphere ---- points!
    na = opt.model.kanchor # how to represent rotation possibilities? --- sampling from the sphere ---- points!
    # nmasks = opt.train_lr.nmasks
    nmasks = opt.nmasks
    num_iters = opt.equi_settings.num_iters
    # nmasks = 2

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
    # dim_in = 3

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
            # neighbor = 16 # int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
            # if i==0 and j==0:
            #    neighbor *= int(input_num/1024)
            kernel_size = 1
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
                    'pooling':  xyz_pooling,
                    'kanchor': na,
                    'norm': 'BatchNorm2d',
                    # 'norm': None,
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
        # 'opt': opt

    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
