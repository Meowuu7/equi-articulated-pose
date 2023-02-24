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
# from SPConvNets.utils.slot_attention_orbit import SlotAttention
import vgtk.spconv as zptk
from SPConvNets.utils.loss_util import batched_index_select
from extensions.chamfer_dist import ChamferDistance
# from chamfer_distance import ChamferDistance
from DGCNN import PrimitiveNet
from SPConvNets.models.common_utils import *

from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
from model_util import farthest_point_sampling, DecoderFC


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
        # get backbone model
        self.backbone = nn.ModuleList()
        for block_param in params['backbone']: # backbone
            self.backbone.append(M.BasicSO3PoseConvBlock(block_param))
        print(f"number of convs in the backbone: {len(self.backbone)}")
        # self.outblock = M.ClsOutBlockR(params['outblock'])
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

        #### Set parameter alias ####
        self.recon_part_M = self.part_pred_npoints # 128, 256, 512, 1024
        self.transformation_dim = 7

        self.log_fn = f"out_feats_weq_wrot_{self.global_rot}_equi_{self.use_equi}_model_{self.model_type}_decoder_{self.decoder_type}_inv_attn_{self.inv_attn}_topk_{self.topk}_num_iters_{self.num_iters}_npts_{self.npoints}_perpart_npts_{self.part_pred_npoints}_bsz_{self.batch_size}_init_lr_{self.init_lr}.npy"

        ''' Set decoder base for PT2PC's Decoder '''
        if self.decoder_type == DECODER_PT2PC:
            cubes = load_pts('cube.pts')
            print(f"Cubes' points loaded: {cubes.shape}")
            self.register_buffer('cubes', torch.from_numpy(cubes))
            #### Set reconstruction network for PT2PC's Decoder ####
            self.part_reconstruction_net = PartDecoder(self.encoded_feat_dim, recon_M=self.recon_part_M)

        # todo: we should set arguments for dgcnn in the option file
        ''' Construct DGCNN backbone '''
        if self.model_type == MODEL_DGCNN:
            class Args():
                def __init__(self):
                    pass

            args = Args()
            args.dgcnn_out_dim = 512;
            args.dgcnn_in_feat_dim = 3;
            args.dgcnn_layers = 3;
            args.nn_nb = 80;
            args.input_normal = False
            args.dgcnn_out_dim = 256;
            args.backbone = 'DGCNN'
            self.dgcnn = PrimitiveNet(args)

        ''' Set chamfer distance '''
        self.chamfer_dist = ChamferDistance()

        ''' Get anchors '''
        self.anchors = torch.from_numpy(L.get_anchors(params['outblock']['kanchor'])).cuda()
        self.outblock = M.ClsOutBlockPointnet(params['outblock'], down_task=False) # clsoutblockpointnet?
        # PointNet Encoder
        # self.pointnetenc = sptk.PointnetSO3Conv(dim_in=256, dim_out=1024, kanchor=60)
        # Need a decoder for position and latent variant features --- but it is what makes it tricky --- we need implicit shape decoded from invariant features as well as each point's variant implicit features, we should factorize positiona and pose to a canonical frame with position and pose from the equivariant features --- position & variant features

        ''' Construct slot-attention module now.. '''
        ### eps is set to default; we may need to tune `dim` and `hidden_dim` ###
        ### output feature shape: bz x num_slots x dim ###
        # self.encoded_feat_dim = 1024;
        self.slot_attention = SlotAttention(num_slots=params['outblock']['k'], dim=self.encoded_feat_dim, hidden_dim=self.encoded_feat_dim)
        ''' Construct per-slot rotation and translation prediction MLP ''' # rotation and translation MLP
        # todo: whether to use BNs in transformation prediction network?
        self.transformation_prediction = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim // 2, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 2),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 2, out_channels=self.encoded_feat_dim // 4,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim // 4),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=self.encoded_feat_dim // 4, out_channels=self.transformation_dim,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
        )

        ''' Construct part point construction network '''
        self.part_reconstruction_net = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=1024, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=1024),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=1024),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=self.recon_part_M * 3,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            # nn.Sigmoid()
        )
        ''' Initialize the part reconstruction network '''
        for zz in self.part_reconstruction_net:
            if isinstance(zz, nn.Conv2d):
                torch.nn.init.xavier_uniform_(zz.weight)
                if zz.bias is not None:
                    torch.nn.init.zeros_(zz.bias)
        # n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False
        # self.decoder = DecoderFC(n_features=(512,1024), latent_dim=256, output_pts=512, bn=True)

        ''' Construct encoder '''
        # self.encoded_feat_dim = 1024
        # # self.encoded_feat_dim = 512
        # self.fc_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 3), stride=(1, 1), bias=True),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.LeakyReLU(inplace=True),
        #     # nn.ReLU(),
        #
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=True),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.LeakyReLU(inplace=True),
        #     # nn.ReLU(),
        #
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True),
        #     nn.BatchNorm2d(num_features=128),
        #     nn.LeakyReLU(inplace=True),
        #     # nn.ReLU(),
        #
        #     nn.Conv2d(in_channels=128, out_channels=self.encoded_feat_dim, kernel_size=(1, 1), stride=(1, 1), bias=True),
        #     nn.BatchNorm2d(num_features=self.encoded_feat_dim),
        #     nn.LeakyReLU(inplace=True),
        #     # nn.ReLU(),
        #
        # )

        ''' Construct decoder '''
        self.fc_decoder = nn.Sequential(
            # first layer
            nn.Conv2d(in_channels=1024, out_channels=1024, stride=(1, 1), kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(),

            # second layer
            nn.Conv2d(in_channels=1024, out_channels=1024, stride=(1, 1), kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(),

            # third layer: reconstruction layer
            nn.Conv2d(in_channels=1024, out_channels=self.recon_part_M * 3, stride=(1, 1), kernel_size=(1, 1), bias=True)
        )

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

        ''' Reconstruction with slot-attention '''
        # centralize points
        x = x - torch.mean(x, dim=-1, keepdim=True)
        ori_pts = x.clone()
        bz, npoints = x.size(0), x.size(2)
        # input_x = x # preprocess input and equivariant

        ''' Preprocess input to get input for the network '''
        x = M.preprocess_input(x, self.kanchor, pose, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        x_xyz = x.xyz; x_anchors = x.anchors; x_pose = x.pose

        # x.feats: bz x dim x N x na
        ''' Get invariant features '''
        # invariant_feats: bz x dim x N
        if self.inv_attn == 0:
            invariant_feats, _ = torch.max(x.feats, dim=-1)
        else:
            invariant_feats, _ = self.outblock(x)
            invariant_feats = invariant_feats.squeeze(-1)

        ''' Get slots' representations and attentions '''
        # rep_slots: bz x n_s x dim; attn_ori: bz x n_s x N
        rep_slots, attn_ori = self.slot_attention(invariant_feats.contiguous().permute(0, 2, 1).contiguous())

        ''' Attention from each point to each cluster '''
        # attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True); bz x N x num_slots
        # point_label: bz x N x na
        point_label = torch.argmax(attn_ori, dim=1)
        # hard_one_hot_labels: bz x N x n_s
        hard_one_hot_labels = torch.eye(self.num_slots, dtype=torch.float32).cuda()[point_label]
        # hard_one_hot_labels: bz x n_s x N
        hard_one_hot_labels = safe_transpose(hard_one_hot_labels, 1, 2)
        nns_slot_labels = torch.sum(hard_one_hot_labels, dim=-1, keepdim=False)

        # hard_one_hot_labels_slot: bz x n_s x N ---- weights from each slot to each point
        hard_one_hot_labels_slot = hard_one_hot_labels / torch.clamp(hard_one_hot_labels.sum(dim=-2, keepdim=True), min=1e-9)

        # attn: bz x N x n_s
        attn = attn_ori.contiguous().transpose(1, 2).contiguous()
        # slot_weights: bz x n_s
        slot_weights = attn.sum(dim=1)
        # slot_weights: bz x n_s
        slot_weights = slot_weights / torch.sum(slot_weights, dim=-2, keepdim=True)

        ''' Get variant features for each slot '''
        # transformed_variant_feats = self.variant_feat_trans(x.feats) # variant feature transformation
        try:
            transformed_variant_feats = x.feats
        except:
            transformed_variant_feats = x
        ''' Aggregate variant feats for each slot '''
        # transformed_variant_feats: bz x c_out x N x na; attn_ori: bz x num_slots x N
        # variant_feats_slot: bz x c_out x num_slots x N x na -> bz x c_out x num_slots x na
        ''' Use soft attention and soft feature aggregation '''
        # variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * attn_slot.unsqueeze(1).unsqueeze(-1),
        #                                dim=3)
        ''' Use hard attention and hard feature aggregation '''
        # bz x num_dim_feat x num_slot
        # transformed_variant_feats: bz x dim x 1 x N x na xxxx bz x 1 x n_s x N x 1 --> bz x dim x n_s x na
        variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * hard_one_hot_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
        # invariant_feats_slot: bz x dim x n_s x 1
        invariant_feats_slot = torch.sum(invariant_feats.unsqueeze(-1).unsqueeze(2) * hard_one_hot_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
        # bz x 3 x num_slot
        # print(ori_pts.size(), hard_one_hot_labels_slot.size())
        # ori_pts: bz x 3 x 1 x N xxxx bz x 1 x n_s x N -> bz x 3 x n_s x N -> bz x 3 x n_s
        # pts_slot = torch.sum(ori_pts.unsqueeze(2) * hard_one_hot_labels_slot.unsqueeze(1),
        #                                dim=3)
        # pts_slot = safe_transpose(pts_slot, 1, 2)

        ''' From aggregated cluster features to reconstructed points for different slots '''
        # recon_slot_points: bz x dim x n_s x 1
        recon_slot_points = self.part_reconstruction_net(invariant_feats_slot) # points recon
        recon_slot_points = recon_slot_points.squeeze(-1)
        # recon_slot_points: bz x n_s x M x 3
        recon_slot_points = recon_slot_points.contiguous().transpose(1, 2).contiguous().view(bz, self.num_slots, self.recon_part_M, -1)
        # ori_recon_slot_points: bz x n_s x M x 3
        ori_recon_slot_points = recon_slot_points.clone()

        per_slot_transformation = self.transformation_prediction(variant_feats_slot)
        # bz x n_feats x n_slots x na
        pred_R, pred_T = per_slot_transformation[:, :4, ...], per_slot_transformation[:, 4:, ...]
        pred_R = compute_rotation_matrix_from_quaternion(
            pred_R.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, 4)).contiguous().view(bz, self.num_slots, self.kanchor,
                                                                                                 3, 3)
        # self.anchors: na x 3 x 3
        # pred_R: bz x n_s x na x 3 x 3
        pred_R = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), pred_R)

        # pred_pose = torch.cat([pred_R, torch.zeros((bz, ))])

        ''' From predicted T to the real translation vector --- part rotation modeling '''
        # pred_res_T: bz x num_slots x na x 3
        pred_T = pred_T.contiguous().permute(0, 2, 3, 1).contiguous() # .squeeze(-2)
        pred_T = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), pred_T.unsqueeze(-1)).squeeze(-1)
        # pred_T: bz x num_slots x na x 3
        # todo: other transformation strategy, like those used in equi-pose?
        # out_feats['ori_recon_slot_pts_hard'] = recon_slot_points.detach().cpu().numpy()
        ''' From predicted rotation matrix and translation matrix to transformed points  '''
        # transformed_slot_pts: bz x num_slots x na x M x 3; bz x n_s x na x 3 x 3    xxxx    bz x n_s x 1 x M x 3 ---> bz x n_s x na x M x 3
        transformed_slot_pts = torch.matmul(pred_R, recon_slot_points.unsqueeze(2).contiguous().transpose(-1, -2)).contiguous().transpose(-1, -2) #  + pred_T.unsqueeze(-2)
        # transformed_slot_pts = torch.matmul(pred_R, recon_slot_points.contiguous().transpose(-1, -2)).contiguous().transpose(-1, -2)  + pred_T.unsqueeze(-2)
        recon_slot_points = transformed_slot_pts

        # purity_loss = get_purity_loss(recon_slot_points)

        # recon_slot_points: bz x n_s x M x 3
        ''' If we only reconstruct centralized points '''
        # recon_slot_points = recon_slot_points + pts_slot.unsqueeze(-2)
        # ori_pts: bz x 3 x N --> bz x N x 3
        ori_pts = ori_pts.contiguous().transpose(1, 2).contiguous()
        # slot_weights: bz x n_s x na
        hard_slot_indicator = (slot_weights > 1e-4).float()
        # recon_slot_points: bz x n_s x na x M x 3
        # hard_slot_indicator: bz x n_s x na
        if self.topk == 1:

            k = (npoints // self.recon_part_M) + (1 if npoints % self.recon_part_M > 0 else 0)
            # print(f"using took... k = {k}")
            topk_slot_values, topk_slot_indicators = torch.topk(slot_weights, dim=-1, k=k)
            hard_slot_indicator = torch.zeros_like(slot_weights)
            # hard_slot_indicator[torch.arange(bz).cuda(), topk_slot_indicators] = 1.

            hard_slot_indicator = (nns_slot_labels >= 20.).float()

        expanded_recon_slot_points = (recon_slot_points * hard_slot_indicator.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).contiguous() # .view(bz, self.recon_part_M * self.num_slots, -1).contiguous()
        expanded_recon_slot_points = expanded_recon_slot_points.contiguous().permute(0, 2, 1, 3, 4).contiguous()
        expanded_recon_slot_points = save_view(expanded_recon_slot_points, (bz * self.kanchor, self.recon_part_M * self.num_slots, -1))

        ''' Calculate chamfer distance for each slot between its reconstructed points and its original points '''
        # hard_one_hot_labels_slot: bz x ns x npoints
        # recon_slot_points: bz x ns x recon_M x 3 & bz x npoints x 3 -> bz x ns x recon_M x npoints
        hard_slot_indicator = (hard_one_hot_labels.sum(-1) > 0.5).float()

        # recon_slot_points: bz x n_s x M x 3
        # dist_recon_between_slots: bz x n_s x n_s x M x M


        # dist_recon_between_slots = torch.sum((recon_slot_points.unsqueeze(2).unsqueeze(-2) - recon_slot_points.unsqueeze(1).unsqueeze(3)) ** 2,  dim=-1)
        # # dist_recon_between_slots: bz x n_s x n_s
        # # dist_recon_between_slots = torch.min(dist_recon_between_slots, dim=-1)[0].mean(dim=-1)
        # dist_recon_between_slots = torch.mean(dist_recon_between_slots, dim=-1).mean(dim=-1)
        # slot_slot_indicator = torch.eye(self.num_slots, ).cuda().float()
        # dist_recon_between_slots = (dist_recon_between_slots * (1. - slot_slot_indicator.unsqueeze(0)))
        # dist_recon_between_slots = dist_recon_between_slots.sum(dim=-1).sum(dim=-1) / (float(self.num_slots * (self.num_slots - 1)) / 2.)
        # dist_recon_between_slots = -0.1 * dist_recon_between_slots.mean()
        # print(f"dist_recon_between_slots: {dist_recon_between_slots.item()}")



        # recon_slot_points: bz x n_s x na x M x 3; ori_pts: bz x N x 3 ---> bz x n_s x na x M x N
        dist_recon_ori_slot = torch.sum((recon_slot_points.unsqueeze(4) - ori_pts.unsqueeze(1).unsqueeze(1).unsqueeze(1)) ** 2, dim=-1)
        dist_recon_ori_slot = dist_recon_ori_slot.contiguous().permute(0, 1, 3, 4, 2).contiguous()
        #  bz x n_s x M x N x na; hard_one_hot_labels: bz x n_s x N ---> bz x ns x M x N x na
        dist_recon_ori_slot = dist_recon_ori_slot * hard_one_hot_labels.unsqueeze(-1).unsqueeze(2) + (1. - hard_one_hot_labels.unsqueeze(-1).unsqueeze(2)) * 1e8 * dist_recon_ori_slot
        # bz x n_s x recon_M x na
        dist_chamfer_recon_slot, _ = torch.min(dist_recon_ori_slot, dim=-2)
        # bz x n_s x na
        dist_chamfer_recon_slot = dist_chamfer_recon_slot.mean(dim=-2) # .mean(dim=-1)
        # dist_chamfer_recon_ori:  bz x ns x N x na; bz x n_s x M x N x na -> bz x n_s x N x na
        # get the chamfer distance
        dist_chamfer_recon_ori, _ = torch.min(dist_recon_ori_slot, dim=2)
        # dist_chamfer_recon_ori_to_slot: bz x n_s x na
        dist_chamfer_recon_ori_to_slot = torch.sum(dist_chamfer_recon_ori * hard_one_hot_labels.unsqueeze(-1), dim=2) / torch.clamp(torch.sum(hard_one_hot_labels.unsqueeze(-1), dim=2), min=1e-9)
        # dist_chamfer_ori.size = bz x N x na
        dist_chamfer_recon_ori, _ = torch.min(dist_chamfer_recon_ori, dim=1)

        dist_chamfer_recon_slot[dist_chamfer_recon_slot > 10.] = 0.
        dist_chamfer_recon_ori[dist_chamfer_recon_ori > 10.] = 0.
        # bz x na
        avg_slots_pts_dist_to_shp = torch.sum(dist_chamfer_recon_slot * slot_weights.unsqueeze(-1), dim=-2) # .mean()
        # bz x na
        shp_pts_dist_to_avg_slot = dist_chamfer_recon_ori.mean(dim=-2) # .mean()

        # glb_dist_slot_to_shp, glb_dist_shp_to_slot = self.chamfer_dist(
        #     expanded_recon_slot_points, ori_pts, return_raw=True
        # )
        # glb_dist_slot_to_shp = save_view(glb_dist_slot_to_shp, (bz, self.num_slots, self.recon_part_M))
        # # glb_dist_slot_to_shp = glb_dist_slot_to_shp.contiguous().view(bz, self.num_slots, self.recon_part_M).contiguous()
        # glb_dist_shp_to_slot = save_view(glb_dist_shp_to_slot, (bz, npoints))

        # slot_to_pts = {}
        # for i_slot in range(self.num_slots):
            # pts_nn = [ii for ii in range(npoints) if point_label]

        ''' If use hard attention and reconstruction '''
        # tot_slots_pts_dist_to_shp = []
        # shp_pts_dist_to_tot_slot = []
        #
        # for i_slot in range(self.num_slots):
        #     ''' If use soft-attention and reconstruction '''
        #     cur_slot_recon_pts = recon_slot_points[:, i_slot, :, :]
        #     # cur_slot_
        #     # cur_slot_pts_dist_to_shp.size = bz x per_slot_recon_M; shp_pts_dist_to_cur_slot: bz x npoints
        #     cur_slot_pts_dist_to_shp, shp_pts_dist_to_cur_slot = safe_chamfer_dist_call(
        #         cur_slot_recon_pts, ori_pts, self.chamfer_dist
        #     )
        #     tot_slots_pts_dist_to_shp.append(torch.mean(cur_slot_pts_dist_to_shp, dim=-1, keepdim=True))
        #     shp_pts_dist_to_tot_slot.append(shp_pts_dist_to_cur_slot.unsqueeze(1))
        #
        # # avg_slots_pts_dist_to_shp: a distance value that can be then used as part of the final loss
        # tot_slots_pts_dist_to_shp = torch.cat(tot_slots_pts_dist_to_shp, dim=-1)
        # avg_slots_pts_dist_to_shp = torch.sum(tot_slots_pts_dist_to_shp * slot_weights, dim=-1).mean()

        # shp_pts_dist_to_tot_slot = torch.cat(shp_pts_dist_to_tot_slot, dim=1)
        # # shp_pts_dist_to_avg_slot: bz x npoints
        # # print(attn_ori.size(), shp_pts_dist_to_tot_slot.size())
        # shp_pts_dist_to_avg_slot = torch.sum(attn_ori.contiguous().transpose(1, 2).contiguous() * shp_pts_dist_to_tot_slot, dim=1)
        # # shp_pts_dist_to_avg_slot: a distance value that can be then used as part of the final loss
        # shp_pts_dist_to_avg_slot = shp_pts_dist_to_avg_slot.mean(dim=-1).mean()
        ''' If use hard attention and reconstruction '''

        # sampled_recon_pts: bz x npoints x 3
        fps_idx = farthest_point_sampling(expanded_recon_slot_points, npoints)
        sampled_recon_pts = expanded_recon_slot_points.contiguous().view(bz * self.kanchor * (self.recon_part_M) * (self.num_slots), -1)[fps_idx, :].contiguous().view(bz * self.kanchor, npoints, -1)
        expanded_sampled_recon_pts = save_view(sampled_recon_pts, (bz, self.kanchor, npoints, -1))
        # glb_dist_recon_to_ori: (bz * na) x npoints; glb_dist_ori_to_recon: bz x npoints
        glb_dist_recon_to_ori, glb_dist_ori_to_recon = safe_chamfer_dist_call(
            sampled_recon_pts, ori_pts.unsqueeze(1).repeat(1, self.kanchor, 1, 1).view(bz * self.kanchor, npoints, 3), self.chamfer_dist )
        glb_dist_recon_to_ori = save_view(glb_dist_recon_to_ori, (bz, self.kanchor, npoints))
        glb_dist_ori_to_recon = save_view(glb_dist_ori_to_recon, (bz, self.kanchor, npoints))
        # bz x na
        glb_dist_recon_to_ori = glb_dist_recon_to_ori.mean(dim=-1) # .mean()
        # bz x na
        glb_dist_ori_to_recon = glb_dist_ori_to_recon.mean(dim=-1) # .mean()

        tot_recon_loss = glb_dist_ori_to_recon + glb_dist_recon_to_ori + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        # tot_recon_idx: bz x 1

        tot_recon_loss, minn_orbit_idx = torch.min(tot_recon_loss, dim=-1)
        tot_recon_loss = tot_recon_loss.mean()

        out_feats = {}

        # bz x na x N
        # print(f"point_labels.size: {point_label.size()}, minn_orbit_idx: {minn_orbit_idx.size()}")

        minn_orbit_idx = minn_orbit_idx.unsqueeze(-1)

        selected_point_labels = point_label

        # print(f"recon_slot_points.size: {recon_slot_points.size()}, ori_recon_slot_points.size: {ori_recon_slot_points.size()}, expanded_sampled_recon_pts.size: {expanded_sampled_recon_pts.size()}")
        selected_recon_slot_pts = safe_transpose(recon_slot_points, 1, 2)
        selected_recon_slot_pts = batched_index_select(values=selected_recon_slot_pts, indices=minn_orbit_idx, dim=1)
        selected_recon_slot_pts = selected_recon_slot_pts.squeeze(1)

        ori_selected_recon_slot_pts = ori_recon_slot_points
        selected_expanded_sampled_recon_pts = batched_index_select(values=expanded_sampled_recon_pts, indices=minn_orbit_idx, dim=1)
        selected_expanded_sampled_recon_pts = selected_expanded_sampled_recon_pts.squeeze(1)
        #
        # print("attn.size:", attn.size())
        selected_attn = attn
        # selected_attn = expanded_attn[:, 0, ...]
        # print("selected_attn.size:", selected_attn.size())
        #
        out_feats['vis_pts_hard'] = ori_pts.detach().cpu().numpy()
        out_feats['vis_labels_hard'] = selected_point_labels.detach().cpu().numpy()
        out_feats['ori_recon_slot_pts_hard'] = ori_selected_recon_slot_pts.detach().cpu().numpy()

        out_feats['recon_slot_pts_hard'] = selected_recon_slot_pts.detach().cpu().numpy()
        out_feats['sampled_recon_pts_hard'] = selected_expanded_sampled_recon_pts.detach().cpu().numpy()

        np.save(self.log_fn, out_feats)

        # selected_labels: bz x N
        selected_labels = torch.argmax(selected_attn, dim=-1)
        # pred_R: bz x n_s x na x 3 x 3
        selected_pred_R = pred_R.contiguous().transpose(1, 2).contiguous()
        # selected_pred_R: bz x n_s x 3 x 3
        selected_pred_R = batched_index_select(values=selected_pred_R, indices=minn_orbit_idx, dim=1)
        selected_pred_R = selected_pred_R.squeeze(1)
        selected_pred_R = batched_index_select(values=selected_pred_R, indices=selected_labels, dim=1)

        pred_pose = torch.cat([selected_pred_R, torch.zeros((bz, npoints, 3, 1), dtype=torch.float32).cuda()],
                              dim=-1)
        pred_pose = torch.cat([pred_pose, torch.zeros((bz, npoints, 1, 4), dtype=torch.float32).cuda()], dim=-2)


        # out_feats['x_features_hard'] = .squeeze(-1).detach().cpu().numpy()
        # np.save("out_feats_with_features_10.npy", out_feats)

        # get tot loss
        ''' If use soft attention and reconstruction '''
        # tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot + purity_loss
        tot_loss = tot_recon_loss
        # tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        ''' If use hard attention and reconstruction '''
        # tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon #  + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        # tot_loss = avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        # return tot_loss, attn
        return tot_loss, selected_attn, pred_pose

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
        # nn_inter = 1
        nn_inter = self.num_iters
        # cur_estimated_pose = pose
        for i in range(nn_inter):
            cur_reconstructed_loss_orbit, attn, cur_estimated_pose = self.forward_one_iter(cur_transformed_points, cur_estimated_pose, rlabel=rlabel)
            tot_loss += cur_reconstructed_loss_orbit
            # cur_gt_rot_dis = self.get_rotation_sims(pose, cur_estimated_pose)
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
    # mlps[-1][-1] = 256 # 512
    # out_mlps[-1] = 256 # 512

    mlps = [[64,128], [256], [512], [1024]]
    out_mlps = [1024]

    # initial_radius_ratio = 0.05
    # initial_radius_ratio = 0.15
    initial_radius_ratio = 0.20
    device = opt.device
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
        'topk': opt.equi_settings.topk
    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
