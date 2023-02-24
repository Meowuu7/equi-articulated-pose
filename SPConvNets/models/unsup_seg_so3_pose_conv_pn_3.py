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
import SPConvNets.utils.base_so3conv as M_so3
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
        self.glb_recon_factor = params['general']['glb_recon_factor']
        self.slot_recon_factor = params['general']['slot_recon_factor']

        #### Set parameter alias ####
        self.recon_part_M = self.part_pred_npoints # 128, 256, 512, 1024
        self.transformation_dim = 7

        self.log_fn = f"{self.shape_type}_out_feats_weq_wrot_{self.global_rot}_rel_rot_factor_{self.rot_factor}_equi_{self.use_equi}_model_{self.model_type}_decoder_{self.decoder_type}_inv_attn_{self.inv_attn}_orbit_attn_{self.orbit_attn}_slot_iters_{self.slot_iters}_topk_{self.topk}_num_iters_{self.num_iters}_npts_{self.npoints}_perpart_npts_{self.part_pred_npoints}_bsz_{self.batch_size}_init_lr_{self.init_lr}"

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
        if self.inv_attn == 1:
            self.outblock = M.ClsOutBlockPointnet(params['outblock'], down_task=False) # clsoutblockpointnet?
            self.outblock_glb = M_so3.ClsOutBlockPointnet(params['outblock'])
        # PointNet Encoder
        # self.pointnetenc = sptk.PointnetSO3Conv(dim_in=256, dim_out=1024, kanchor=60)
        # Need a decoder for position and latent variant features --- but it is what makes it tricky --- we need implicit shape decoded from invariant features as well as each point's variant implicit features, we should factorize positiona and pose to a canonical frame with position and pose from the equivariant features --- position & variant features

        ''' Construct slot-attention module now.. '''
        ### eps is set to default; we may need to tune `dim` and `hidden_dim` ###
        ### output feature shape: bz x num_slots x dim ###
        # self.encoded_feat_dim = 1024;
        self.slot_attention = SlotAttention(num_slots=params['outblock']['k'], dim=(self.encoded_feat_dim + self.kanchor) if self.orbit_attn == 1 else (self.kanchor) if self.orbit_attn == 2 else (self.encoded_feat_dim), hidden_dim=self.encoded_feat_dim,
                                            iters=self.slot_iters)
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

        self.transformation_prediction = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.transformation_prediction.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim // 2,
                              kernel_size=(1, 1),
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
            )

        ''' Construct part point construction network '''
        # self.part_reconstruction_net = nn.Sequential(
        #     nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=1024, kernel_size=(1, 1),
        #               stride=(1, 1), bias=True),
        #     nn.BatchNorm2d(num_features=1024),
        #     # nn.ReLU(),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=1024, out_channels=1024,
        #               kernel_size=(1, 1), stride=(1, 1), bias=True),
        #     nn.BatchNorm2d(num_features=1024),
        #     # nn.ReLU(),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=1024, out_channels=self.recon_part_M * 3,
        #               kernel_size=(1, 1), stride=(1, 1), bias=True),
        #     # nn.Sigmoid()
        # )
        self.part_reconstruction_net = nn.ModuleList()
        for i_s in range(self.num_slots):
            self.part_reconstruction_net.append(
                nn.Sequential(
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
                    nn.Sigmoid()
                )
            )
        ''' Initialize the part reconstruction network '''
        for zz in self.part_reconstruction_net:
            if isinstance(zz, nn.Sequential):
                for zzz in zz:
                    if isinstance(zz, nn.Conv2d):
                        torch.nn.init.xavier_uniform_(zz.weight)
                        if zz.bias is not None:
                            torch.nn.init.zeros_(zz.bias)
            elif isinstance(zz, nn.Conv2d):
                    torch.nn.init.xavier_uniform_(zz.weight)
                    if zz.bias is not None:
                         torch.nn.init.zeros_(zz.bias)

        ''' Prior based reconstruction '''
        if self.recon_prior == 1:
            self.mu_params = nn.ParameterList()
            self.log_sigma_params = nn.ParameterList()
            for ii in range(len(self.part_reconstruction_net)):
                cur_mu = nn.Parameter(torch.zeros((self.encoded_feat_dim, ), dtype=torch.float))
                cur_log_sigma = nn.Parameter(torch.zeros((self.encoded_feat_dim, ), dtype=torch.float))
                self.mu_params.append(cur_mu)
                self.log_sigma_params.append(cur_log_sigma)
        ''' Prior based reconstruction '''

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
        # self.fc_decoder = nn.Sequential(
        #     # first layer
        #     nn.Conv2d(in_channels=1024, out_channels=1024, stride=(1, 1), kernel_size=(1, 1), bias=True),
        #     nn.BatchNorm2d(num_features=1024),
        #     nn.LeakyReLU(inplace=True),
        #     # nn.ReLU(),
        #
        #     # second layer
        #     nn.Conv2d(in_channels=1024, out_channels=1024, stride=(1, 1), kernel_size=(1, 1), bias=True),
        #     nn.BatchNorm2d(num_features=1024),
        #     nn.LeakyReLU(inplace=True),
        #     # nn.ReLU(),
        #
        #     # third layer: reconstruction layer
        #     nn.Conv2d(in_channels=1024, out_channels=self.recon_part_M * 3, stride=(1, 1), kernel_size=(1, 1), bias=True)
        # )

    def apply_part_reconstruction_net(self, feats):
        recon_pts = []
        bz = feats.size(0)
        dim = feats.size(1)
        # bz x dim x n_s x na
        for i_s, mod in enumerate(self.part_reconstruction_net):

            cur_slot_inv_feats =  feats[:, :, i_s, :].view(bz, dim, 1, 1) # .unsqueeze(-2)
            cur_slot_inv_feats = mod(cur_slot_inv_feats)
            cur_slot_inv_feats = cur_slot_inv_feats.squeeze(-1)
            # recon_slot_points: bz x n_s x M x 3
            cur_slot_inv_feats = cur_slot_inv_feats.contiguous().transpose(1, 2).contiguous().view(bz, 1,
                                                                                                 self.recon_part_M, -1)
            recon_pts.append(cur_slot_inv_feats - 0.5)
        recon_pts = torch.cat(recon_pts, dim=1) # .cuda()
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
                                                                                                 self.recon_part_M, -1)
            recon_pts.append(cur_slot_inv_feats - 0.5)
        recon_pts = torch.cat(recon_pts, dim=1) # .cuda()
        # print(recon_pts.size())
        return recon_pts

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

    # def get_gt_slot_rotation_translations(self, label, pose):
    #     bz = label.size(0); N = label.size(1)
    #     gt_rots = torch.zeros((bz, self.num_slots, 3, 3), dtype=torch.float)
    #     gt_trans = torch.zeros((bz, self.num_slots, 3), dtype=torch.float)

        # slot_idx_to_pts_idx = torch.zeros((bz, self.num_slots), dtype=torch.long)
        # for i_bz in range(bz):
        #     for i_n in range(N):
        #         cur_label = int(label[i_bz, i_n].item())
        #         slot_idx_to_pts_idx[]

    def forward_one_iter(self, x, pose, ori_pc=None, rlabel=None, cur_iter=0, gt_pose=None, gt_pose_segs=None, canon_pc=None): # rotation label

        output = {}

        ''' Reconstruction with slot-attention '''
        # centralize points
        # x = x - torch.mean(x, dim=-1, keepdim=True)
        ori_pts = x.clone()
        bz, npoints = x.size(0), x.size(2)
        # input_x = x # preprocess input and equivariant

        ''' Preprocess input to get input for the network '''
        x = M.preprocess_input(x, self.kanchor, pose, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        x_xyz = x.xyz; x_anchors = x.anchors; x_pose = x.pose

        torch.cuda.empty_cache()

        # x.feats: bz x dim x N x na
        ''' Get invariant features '''
        # invariant_feats: bz x dim x N
        if self.inv_attn == 0:
            invariant_feats, _ = torch.max(x.feats, dim=-1)
        else:
            # orbit_attn_weights: bz x N x na
            invariant_feats, orbit_attn_weights = self.outblock(x)
            invariant_feats = invariant_feats.squeeze(-1)

            # invariant_feats_glb, _ = self.outblock_glb(x)

        # #### Just for test ####
        # print(self.anchors.size(), ori_pts.size())
        # trans_x = torch.matmul(self.anchors[7].unsqueeze(0).unsqueeze(0), ori_pts.contiguous().transpose(1, 2).contiguous().unsqueeze(-1)).squeeze(-1)
        # trans_x = trans_x.contiguous().transpose(1, 2).contiguous()
        ''' Preprocess input to get input for the network '''
        # trans_x = M.preprocess_input(trans_x, self.kanchor, pose, False)
        # for block_i, block in enumerate(self.backbone):
        #     trans_x = block(trans_x)
        #     print(f"block: {block_i}, trans_x.size: {trans_x.feats.size()}")
        #
        # # x.feats: bz x dim x N x na
        # ''' Get invariant features '''
        # # invariant_feats: bz x dim x N
        # if self.inv_attn == 0:
        #     now_invariant_feats, _ = torch.max(trans_x.feats, dim=-1)
        # else:
        #     # orbit_attn_weights: bz x N x na
        #     now_invariant_feats, orbit_attn_weights = self.outblock(trans_x)
        #     now_invariant_feats = now_invariant_feats.squeeze(-1)
        # disstt = torch.sum((now_invariant_feats - invariant_feats) ** 2, dim=1).mean()
        # print(disstt)
        # #### Just for test ####

        invariant_feats_npy = invariant_feats.contiguous().permute(0, 2, 1).contiguous().detach().cpu().numpy()

        ''' Get slots' representations and attentions '''
        if self.orbit_attn > 0:
            if bz == 1:
                orbit_attn_weights = orbit_attn_weights.unsqueeze(0)
            # print(f"orbit_attn_weights.size: {orbit_attn_weights.size()}")
            if self.orbit_attn == 1:
                inv_orbit_feats = torch.cat([invariant_feats.contiguous().permute(0, 2, 1).contiguous(), orbit_attn_weights], dim=-1)
            else:
                inv_orbit_feats = orbit_attn_weights
            rep_slots, attn_ori = self.slot_attention(inv_orbit_feats)
        else:
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
        # nns_slot_labels = torch.sum(hard_one_hot_labels, dim=-1, keepdim=False)

        # hard_one_hot_labels_slot: bz x n_s x N ---- weights from each slot to each point
        hard_one_hot_labels_slot = hard_one_hot_labels / torch.clamp(hard_one_hot_labels.sum(dim=-1, keepdim=True),
                                                                     min=1e-9)

        soft_labels_slot = attn_ori / torch.clamp(attn_ori.sum(dim=-1, keepdim=True),
                                                                     min=1e-9)

        # attn: bz x N x n_s
        attn = attn_ori.contiguous().transpose(1, 2).contiguous()
        # slot_weights: bz x n_s
        slot_weights = attn.sum(dim=1)
        # slot_weights: bz x n_s
        slot_weights = slot_weights / torch.clamp(torch.sum(slot_weights, dim=-2, keepdim=True), min=1e-9)
        # print(f"slotweights: {slot_weights}")
        ''' Attention from each point to each cluster '''

        ''' Add cross entropy between hard labels and soft weights '''
        cr_loss = torch.nn.functional.cross_entropy(input=attn_ori, target=point_label)
        ''' Add cross entropy between hard labels and soft weights '''

        # rlabel: bz x N x n_s
        ''' If using GT-Seg '''
        if self.gt_oracle_seg == 1: # oracle seg
            # attn_ori = rlabel.float().contiguous().transpose(1, 2).contiguous()
            _, pred_labels = torch.max(attn_ori, dim=1)
            pred_labels_one_hot = torch.eye(self.num_slots, dtype=torch.float32).cuda()[pred_labels].float()
            pred_labels_one_hot = safe_transpose(pred_labels_one_hot, 1, 2)
            attn_ori = pred_labels_one_hot * attn_ori
            gt_attn = rlabel.float().contiguous().transpose(1, 2).contiguous()
            # attn_pred_gt_rel: bz x n_s x n_s
            attn_pred_gt_rel_pts = attn_ori.unsqueeze(2) * gt_attn.unsqueeze(1)
            attn_pred_gt_rel = torch.sum(attn_pred_gt_rel_pts, dim=-1)
            _, attn_gt_selected_idx = torch.max(attn_pred_gt_rel, dim=-1)
            attn_ori = batched_index_select(values=attn_pred_gt_rel_pts, indices=attn_gt_selected_idx.unsqueeze(-1), dim=2)
            attn_ori = attn_ori.squeeze(2)
            attn_ori = gt_attn

            ''' Attention from each point to each cluster '''
            # attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True); bz x N x num_slots
            # point_label: bz x N x na
            point_label = torch.argmax(attn_ori, dim=1)
            # hard_one_hot_labels: bz x N x n_s
            hard_one_hot_labels = torch.eye(self.num_slots, dtype=torch.float32).cuda()[point_label]
            # hard_one_hot_labels: bz x n_s x N
            hard_one_hot_labels = safe_transpose(hard_one_hot_labels, 1, 2)
            # nns_slot_labels = torch.sum(hard_one_hot_labels, dim=-1, keepdim=False)

            # hard_one_hot_labels_slot: bz x n_s x N ---- weights from each slot to each point
            hard_one_hot_labels_slot = hard_one_hot_labels / torch.clamp(hard_one_hot_labels.sum(dim=-1, keepdim=True),
                                                                         min=1e-9)

            soft_labels_slot = attn_ori / torch.clamp(attn_ori.sum(dim=-1, keepdim=True),
                                                      min=1e-9)

            # attn: bz x N x n_s
            attn = attn_ori.contiguous().transpose(1, 2).contiguous()
            # slot_weights: bz x n_s
            slot_weights = attn.sum(dim=1)
            # slot_weights: bz x n_s
            slot_weights = slot_weights / torch.clamp(torch.sum(slot_weights, dim=-2, keepdim=True), min=1e-9)
            # print(f"slotweights: {slot_weights}")
            ''' Attention from each point to each cluster '''
        elif self.gt_oracle_seg == 2:
            # attn_ori = rlabel.float().contiguous().transpose(1, 2).contiguous()
            _, pred_labels = torch.max(attn_ori, dim=1)
            pred_labels_one_hot = torch.eye(self.num_slots, dtype=torch.float32).cuda()[pred_labels].float()
            pred_labels_one_hot = safe_transpose(pred_labels_one_hot, 1, 2)
            attn_ori = pred_labels_one_hot * attn_ori
        elif self.gt_oracle_seg == 3:
            purity_r = 0.30
            # ball_idx: bz x N x nn; grouped_xyz: bz x 3 x N x nn
            ball_idx, grouped_xyz = zpconv.ball_query(ori_pts, ori_pts, purity_r, 32)
            ball_idx = ball_idx.long()
            _, pred_labels = torch.max(attn_ori, dim=1)
            # pred_labels: bz x N
            # grouped_labels: bz x N x nn
            grouped_labels = batched_index_select(values=pred_labels, indices=ball_idx, dim=1)
            # sim_labels_indicator: bz x N x nn
            sim_labels_indicator = (pred_labels.unsqueeze(-1) == grouped_labels).float()
            thresh_purity = 0.90
            pts_purity = torch.sum(sim_labels_indicator, dim=-1) / float(sim_labels_indicator.size(-1))
            # purity_indicator: bz x N
            purity_indicator = (pts_purity > thresh_purity).float()
            attn_ori = attn_ori * purity_indicator.unsqueeze(1)

            hard_one_hot_labels_slot_2 = hard_one_hot_labels_slot * purity_indicator.unsqueeze(1)
            _, pred_labels_2 = torch.max(attn_ori, dim=1)


        ''' If using soft one-hot labels '''
        # soft_one_hot_labels_slot = attn_ori / torch.clamp(attn_ori.sum(dim=-2, keepdim=True), min=1e-9)

        ''' Attention from each point to each cluster '''
        # # attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True); bz x N x num_slots
        # # point_label: bz x N x na
        # point_label = torch.argmax(attn_ori, dim=1)
        # # hard_one_hot_labels: bz x N x n_s
        # hard_one_hot_labels = torch.eye(self.num_slots, dtype=torch.float32).cuda()[point_label]
        # # hard_one_hot_labels: bz x n_s x N
        # hard_one_hot_labels = safe_transpose(hard_one_hot_labels, 1, 2)
        # # nns_slot_labels = torch.sum(hard_one_hot_labels, dim=-1, keepdim=False)
        #
        # # hard_one_hot_labels_slot: bz x n_s x N ---- weights from each slot to each point
        # hard_one_hot_labels_slot = hard_one_hot_labels / torch.clamp(hard_one_hot_labels.sum(dim=-2, keepdim=True), min=1e-9)

        # # attn: bz x N x n_s
        # attn = attn_ori.contiguous().transpose(1, 2).contiguous()
        # # slot_weights: bz x n_s
        # slot_weights = attn.sum(dim=1)
        # # slot_weights: bz x n_s
        # slot_weights = slot_weights / torch.clamp(torch.sum(slot_weights, dim=-2, keepdim=True), min=1e-9)
        # # print(f"slotweights: {slot_weights}")

        # hard_slot_weights = hard_one_hot_labels.sum(dim=-1)
        # hard_slot_weights = hard_slot_weights / torch.sum(hard_slot_weights, dim=-1, keepdim=True)
        # hard_slot_indicator = (hard_one_hot_labels.sum(dim=-1) > 0.5).float()
        ''' Attention from each point to each cluster '''

        ''' Get variant features for each slot '''
        # transformed_variant_feats = self.variant_feat_trans(x.feats) # variant feature transformation
        try:
            transformed_variant_feats = x.feats
        except:
            transformed_variant_feats = x

        # point_label_2 = torch.argmax(attn_ori, dim=1)
        # # hard_one_hot_labels: bz x N x n_s
        # hard_one_hot_labels_2 = torch.eye(self.num_slots, dtype=torch.float32).cuda()[point_label_2]
        # # hard_one_hot_labels: bz x n_s x N
        # hard_one_hot_labels = safe_transpose(hard_one_hot_labels, 1, 2)

        ''' Aggregate variant feats for each slot '''
        # transformed_variant_feats: bz x c_out x N x na; attn_ori: bz x num_slots x N
        # variant_feats_slot: bz x c_out x num_slots x N x na -> bz x c_out x num_slots x na
        ''' Use soft attention and soft feature aggregation '''
        # variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * attn_slot.unsqueeze(1).unsqueeze(-1),
        #                                dim=3)
        ''' Use hard attention and hard feature aggregation '''
        expaned_hard_one_hot_label = hard_one_hot_labels.unsqueeze(1).unsqueeze(-1).contiguous().repeat(1, transformed_variant_feats.size(1), 1, 1, self.kanchor)
        # bz x num_dim_feat x num_slot
        # transformed_variant_feats: bz x dim x 1 x N x na xxxx bz x 1 x n_s x N x 1 --> bz x dim x n_s x na
        ''' Use hard labels for feature aggregation '''
        if self.feat_pooling == "mean":
            variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * hard_one_hot_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
            variant_feats_slot = torch.mean(transformed_variant_feats, dim=2).unsqueeze(2).repeat(1, 1, self.num_slots, 1)
            variant_feats_slot = torch.sum(
                transformed_variant_feats.unsqueeze(2) * soft_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
        elif self.feat_pooling == "max":
            variant_feats_slot = transformed_variant_feats.unsqueeze(2) * hard_one_hot_labels.unsqueeze(1).unsqueeze(-1)
            variant_feats_slot[expaned_hard_one_hot_label < 0.5] = -9999999.0
            variant_feats_slot, _ = torch.max(variant_feats_slot, dim=3)
            variant_feats_slot = torch.max(transformed_variant_feats, dim=2)[0].unsqueeze(2).repeat(1, 1, self.num_slots,
                                                                                                  1)
        else:
            raise ValueError(f"Unrecognized feature pooling matrix: {self.feat_pooling}.")
        # variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * hard_one_hot_labels_slot_2.unsqueeze(1).unsqueeze(-1), dim=3)
        ''' Use soft labels for feature aggregation '''
        # variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * soft_one_hot_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
        # invariant_feats_slot: bz x dim x n_s x 1
        ''' Use hard labels for feature aggregation '''

        # invariant_feats expanded: bz x dim x 1 x N x 1
        # hard_one_hot_labels_slot/hard_one_hot_labels: bz x n_s x N;
        expaned_hard_one_hot_label =  hard_one_hot_labels.unsqueeze(1).contiguous().repeat(1, transformed_variant_feats.size(1), 1, 1)
        if self.feat_pooling == "mean":
            invariant_feats_slot = torch.sum(invariant_feats.unsqueeze(-1).unsqueeze(2) * hard_one_hot_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
            invariant_feats_slot = torch.mean(invariant_feats, dim=-1).unsqueeze(-1).repeat(1, 1, self.num_slots).unsqueeze(-1)
            invariant_feats_slot = torch.sum(invariant_feats.unsqueeze(-1).unsqueeze(2) * soft_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
        elif self.feat_pooling == "max":
            invariant_feats_slot = invariant_feats.unsqueeze(-1).unsqueeze(2) * hard_one_hot_labels.unsqueeze(1).unsqueeze(-1)
            invariant_feats_slot[expaned_hard_one_hot_label < 0.5] = -9999999.0
            invariant_feats_slot, _ = torch.max(invariant_feats_slot, dim=3)
            invariant_feats_slot = torch.max(invariant_feats, dim=-1)[0].unsqueeze(-1).repeat(1, 1, self.num_slots).unsqueeze(-1)
        else:
            raise ValueError(f"Unrecognized feature pooling matrix: {self.feat_pooling}.")
        # invariant_feats_slot = torch.sum(invariant_feats.unsqueeze(-1).unsqueeze(2) * hard_one_hot_labels_slot_2.unsqueeze(1).unsqueeze(-1), dim=3)
        ''' Use soft labels for feature aggregation '''
        # invariant_feats_slot = torch.sum(invariant_feats.unsqueeze(-1).unsqueeze(2) * soft_one_hot_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
        # bz x 3 x num_slot
        # print(ori_pts.size(), hard_one_hot_labels_slot.size())
        # ori_pts: bz x 3 x 1 x N xxxx bz x 1 x n_s x N -> bz x 3 x n_s x N -> bz x 3 x n_s
        # pts_slot = torch.sum(ori_pts.unsqueeze(2) * hard_one_hot_labels_slot.unsqueeze(1),
        #                                dim=3)
        # pts_slot = safe_transpose(pts_slot, 1, 2)

        if isinstance(self.part_reconstruction_net, nn.Sequential):
            ''' From aggregated cluster features to reconstructed points for different slots '''
            # recon_slot_points: bz x dim x n_s x 1
            recon_slot_points = self.part_reconstruction_net(invariant_feats_slot) # points recon
            recon_slot_points = recon_slot_points.squeeze(-1)
            # recon_slot_points: bz x n_s x M x 3
            recon_slot_points = recon_slot_points.contiguous().transpose(1, 2).contiguous().view(bz, self.num_slots, self.recon_part_M, -1)
            # ori_recon_slot_points: bz x n_s x M x 3
            # ori_recon_slot_points = recon_slot_points.clone()
            ori_recon_slot_points = recon_slot_points
            ori_recon_slot_points = recon_slot_points - 0.5
        else:
            if self.recon_prior == 0:
                recon_slot_points = self.apply_part_reconstruction_net(invariant_feats_slot)
            else:
                recon_slot_points = self.apply_part_reconstruction_net_v2(invariant_feats_slot)
            ori_recon_slot_points = recon_slot_points

        # ''' Calculate loss for reference frame '''
        # # tot_ref_shp_recon_loss = 0.0
        # losses = []
        # # print("self.ref_shape_slot", self.ref_shape_slot)
        # # self.ref_shape_slot = self.ref_shape_slot.cuda()
        # ### num_slots is set to number of segs in the shape ####
        # for j_s in range(min(self.ref_shape_slot.size(0), self.num_slots)):
        #     # bz x M x 3
        #     ori_recon_cur_slot_points = ori_recon_slot_points[:, j_s, :, :]
        #     ref_shp_cur_slot_points = self.ref_shape_slot[j_s, :, :]
        #     # print(ref_shp_cur_slot_points.size(), ori_recon_cur_slot_points.size())
        #     # dist_recon_to_ori_ref_shap: bz x M; dist_ori_to_shp_ref_sho: bz x M'
        #     dist_recon_to_ori_ref_shap, dist_ori_to_shp_ref_sho = safe_chamfer_dist_call(
        #         ori_recon_cur_slot_points, ref_shp_cur_slot_points.unsqueeze(0).repeat(bz, 1, 1), self.chamfer_dist
        #     )
        #     # print("disss", dist_recon_to_ori_ref_shap, dist_recon_to_ori_ref_shap)
        #     # tot_ref_shp_recon_loss += dist_recon_to_ori_ref_shap.mean(dim=-1) + dist_ori_to_shp_ref_sho.mean(dim=-1)
        #     losses.append((dist_recon_to_ori_ref_shap.mean(dim=-1) + dist_ori_to_shp_ref_sho.mean(dim=-1)).unsqueeze(0))
        #     # print("losses: ", losses[-1])
        #
        # # print(f"tot len ref recon loss: {len(losses)}")
        # losses = torch.cat(losses, dim=0)
        # losses = torch.mean(losses, dim=0) #
        # tot_ref_shp_recon_loss = (losses / float(self.num_slots)).mean()
        # tot_ref_shp_recon_loss = (tot_ref_shp_recon_loss / float(self.num_slots)).mean()

        # print(f"tot_ref_shp_recon_loss: {tot_ref_shp_recon_loss}")

        if isinstance(self.transformation_prediction, nn.Sequential):
            ''' Predict transformations based on variant features '''
            per_slot_transformation = self.transformation_prediction(variant_feats_slot)
        else:
            ''' Predict transformations based on variant features '''
            per_slot_transformation = self.apply_transformation_prediction_net(variant_feats_slot)
        # bz x n_feats x n_slots x na
        pred_R, pred_T = per_slot_transformation[:, :4, ...], per_slot_transformation[:, 4:, ...]
        pred_R = compute_rotation_matrix_from_quaternion(
            pred_R.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, 4)).contiguous().view(bz, self.num_slots, self.kanchor, 3, 3)
        # self.anchors: na x 3 x 3
        # pred_R: bz x n_s x na x 3 x 3

        # pred_pose = torch.cat([pred_R, torch.zeros((bz, ))])

        ''' From predicted T to the real translation vector --- part rotation modeling '''
        # pred_res_T: bz x num_slots x na x 3
        pred_T = pred_T.contiguous().permute(0, 2, 3, 1).contiguous() # .squeeze(-2)

        if self.gt_oracle_trans == 1:
            pred_R = torch.zeros((bz, self.num_slots, self.kanchor, 3, 3), dtype=torch.float).cuda()
            pred_R[:, :, :, 0, 0] = 1.
            pred_R[:, :, :, 1, 1] = 1.
            pred_R[:, :, :, 2, 2] = 1.
            pred_T = torch.zeros((bz, self.num_slots, self.kanchor, 3), dtype=torch.float).cuda()
            # hard_one_hot_label_slot: bz x n_s x N; ori_pts: bz x 3 x N; bz x n_s x 1 x N
            # pcc: bz x n_s x 3
            pcc = torch.sum(ori_pts.unsqueeze(1) * hard_one_hot_labels_slot.unsqueeze(2), dim=-1)
            # pred_T: bz x n_s x kanchor x 3
            pred_T = pcc.unsqueeze(2).repeat(1, 1, self.kanchor, 1)
            # print(pred_T)
            # 1 x 1 x kanchor x 3 x 3 xxxxxx bz x n_s x kanchor x 3 x 1 ---> bz x n_s x kanchor x 3
            # print(self.anchors.size(), pred_T.size())
            # print(self.anchors)
            # pred_T = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), pred_T.unsqueeze(-1)).squeeze(-1)

            pred_R = gt_pose_segs[:, :, :3, :3].unsqueeze(2)
            pred_T = gt_pose_segs[:, :, :3, 3].unsqueeze(2)
            gt_segs_nn = pred_R.size(1)
            if gt_segs_nn < self.num_slots:
                pred_R = torch.cat(
                    [pred_R, torch.eye(3, dtype=torch.float).cuda().view(1, 1, 1, 3, 3).repeat(bz, self.num_slots - gt_segs_nn, self.kanchor, 1, 1)], dim=1
                )
                pred_T = torch.cat(
                    [pred_T, torch.zeros((bz, self.num_slots - gt_segs_nn, self.kanchor, 3), dtype=torch.float).cuda()], dim=1
                )
        elif self.gt_oracle_trans == 2:
            pred_R = gt_pose_segs[:, :, :3, :3].unsqueeze(2)
            # pred_T = gt_pose_segs[:, :, :3, 3].unsqueeze(2)
            gt_segs_nn = pred_R.size(1)
            if gt_segs_nn < self.num_slots:
                pred_R = torch.cat(
                    [pred_R,
                     torch.eye(3, dtype=torch.float).cuda().view(1, 1, 1, 3, 3).repeat(bz, self.num_slots - gt_segs_nn,
                                                                                       self.kanchor, 1, 1)], dim=1
                )
                # pred_T = torch.cat(
                #     [pred_T, torch.zeros((bz, self.num_slots - gt_segs_nn, self.kanchor, 3), dtype=torch.float).cuda()],
                #     dim=1
                # )

        pred_R = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), pred_R)

        if self.translation == EQUI_TRANSLATION_RESIDULE:
            #### Assume that we predict redidue transition ####
            pred_T = torch.matmul(self.anchors.unsqueeze(0).unsqueeze(0), pred_T.unsqueeze(-1)).squeeze(-1)
        elif self.translation == EQUI_TRANSLATION_ROVOLUTE:
            #### Assume that we predict the center and then from the transition center to predicted T ####
            pred_T = pred_T - torch.matmul(pred_R, pred_T.unsqueeze(-1)).squeeze(-1)
        elif self.translation == EQUI_TRANSLATION_ORI:
            pred_T = pred_T
        else:
            raise ValueError(f"Unrecognized translation setting: {self.translation}.")

        if self.cent_trans == 1:
            # pcc = torch.sum(ori_pts.unsqueeze(1) * hard_one_hot_labels_slot.unsqueeze(2), dim=-1)
            # # pred_T: bz x n_s x kanchor x 3
            # pcc = pcc.unsqueeze(2).repeat(1, 1, self.kanchor, 1)
            # pred_T = pred_T + pcc

            # bz x 3 x N ---> bz x N x 3
            ori_pc = safe_transpose(ori_pc, 1, 2)
            # ori_pc: bz x N x 3, hard_one_hot_labels_slot: bz x n_s x N ---> bz x 1 x N x 3 xxxx bz x n_s x N x 3 ----> bz x n_s x N x 3 ----> bz x n_s x 3
            pcc = torch.sum(ori_pc.unsqueeze(1) * hard_one_hot_labels_slot.unsqueeze(-1), dim=-2)
            pcc = pcc.unsqueeze(2).repeat(1, 1, self.kanchor, 1)
            # print(pcc.size(), pred_R.size())
            pcc = torch.matmul(pred_R, pcc.unsqueeze(-1)).squeeze(-1)

            pred_T = pred_T + pcc
        elif self.cent_trans == 2:
            #
            ori_pc = safe_transpose(ori_pc, 1, 2)
            # ori_pc: bz x N x 3, hard_one_hot_labels_slot: bz x n_s x N ---> bz x 1 x N x 3 xxxx bz x n_s x N x 3 ----> bz x n_s x N x 3 ----> bz x n_s x 3
            pcc = torch.sum(ori_pc.unsqueeze(1) * hard_one_hot_labels_slot.unsqueeze(-1), dim=-2)
            pcc = pcc.unsqueeze(2).repeat(1, 1, self.kanchor, 1)
            pred_T = pcc
        elif self.cent_trans == 3:
            # ori_pts: bz x 3 x N --> bz x N x 3
            ori_pt = safe_transpose(ori_pts, 1, 2)
            # ori_pc: bz x N x 3, hard_one_hot_labels_slot: bz x n_s x N ---> bz x 1 x N x 3 xxxx bz x n_s x N x 3 ----> bz x n_s x N x 3 ----> bz x n_s x 3
            if self.soft_attn == 0:
                pcc = torch.sum(ori_pt.unsqueeze(1) * hard_one_hot_labels_slot.unsqueeze(-1), dim=-2)
            else:
                zz = hard_one_hot_labels_slot.unsqueeze(-1) * soft_labels_slot.unsqueeze(-1)
                pcc = torch.sum(ori_pt.unsqueeze(1) * zz, dim=-2) / torch.clamp(torch.sum(zz, dim=-2), min=1e-9)
                # pcc = torch
            pcc = pcc.unsqueeze(2).repeat(1, 1, self.kanchor, 1)
            # print(pcc.size(), pred_R.size())
            # pcc = torch.matmul(pred_R, pcc.unsqueeze(-1)).squeeze(-1)

            pred_T = pred_T + pcc

        # if self.gt_oracle_trans == 1:
        #     pred_R =

        # pred_R: bz x n_s x na x 3 x 3

        # pred_T: bz x num_slots x na x 3
        # todo: other transformation strategy, like those used in equi-pose?
        # out_feats['ori_recon_slot_pts_hard'] = recon_slot_points.detach().cpu().numpy()
        ''' From predicted rotation matrix and translation matrix to transformed points '''
        # transformed_slot_pts: bz x num_slots x na x M x 3; bz x n_s x na x 3 x 3    xxxx    bz x n_s x 1 x M x 3 ---> bz x n_s x na x M x 3
        # transformed_slot_pts = torch.matmul(pred_R, recon_slot_points.unsqueeze(2).contiguous().transpose(-1, -2)).contiguous().transpose(-1, -2) #  + pred_T.unsqueeze(-2)
        # print(transformed_slot_pts.size(), pred_T.size())
        transformed_slot_pts = torch.matmul(pred_R, recon_slot_points.unsqueeze(2).contiguous().transpose(-1, -2)).contiguous().transpose(-1, -2)  + pred_T.unsqueeze(-2)
        # recon_slot_points = transformed_slot_pts

        # purity_loss = get_purity_loss(transformed_slot_pts)

        # transformed_slot_pts: bz x n_s x M x 3
        ''' If we only reconstruct centralized points '''
        # transformed_slot_pts = transformed_slot_pts + pts_slot.unsqueeze(-2)
        # ori_pts: bz x 3 x N --> bz x N x 3
        ori_pts = ori_pts.contiguous().transpose(1, 2).contiguous()
        # slot_weights: bz x n_s x na
        # hard_slot_indicator = (slot_weights > 1e-4).float()
        # # transformed_slot_pts: bz x n_s x na x M x 3
        # # hard_slot_indicator: bz x n_s x na
        # if self.topk == 1:
        #
        #     k = (npoints // self.recon_part_M) + (1 if npoints % self.recon_part_M > 0 else 0)
        #     # print(f"using took... k = {k}")
        #     topk_slot_values, topk_slot_indicators = torch.topk(slot_weights, dim=-1, k=k)
        #     hard_slot_indicator = torch.zeros_like(slot_weights)
        #     # hard_slot_indicator[torch.arange(bz).cuda(), topk_slot_indicators] = 1.
        #
        #     hard_slot_indicator = (nns_slot_labels >= 20.).float()

        # expanded_recon_slot_points = (transformed_slot_pts * hard_slot_indicator.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).contiguous() # .view(bz, self.recon_part_M * self.num_slots, -1).contiguous()
        # expanded_recon_slot_points = expanded_recon_slot_points.contiguous().permute(0, 2, 1, 3, 4).contiguous()
        # expanded_recon_slot_points = save_view(expanded_recon_slot_points, (bz * self.kanchor, self.recon_part_M * self.num_slots, -1))

        ''' Calculate chamfer distance for each slot between its reconstructed points and its original points '''
        # hard_one_hot_labels_slot: bz x ns x npoints
        # transformed_slot_pts: bz x ns x recon_M x 3 & bz x npoints x 3 -> bz x ns x recon_M x npoints
        hard_slot_indicator = (hard_one_hot_labels.sum(-1) > 0.5).float()

        # transformed_slot_pts: bz x n_s x M x 3
        # dist_recon_between_slots: bz x n_s x n_s x M x M


        # dist_recon_between_slots = torch.sum((transformed_slot_pts.unsqueeze(2).unsqueeze(-2) - transformed_slot_pts.unsqueeze(1).unsqueeze(3)) ** 2,  dim=-1)
        # # dist_recon_between_slots: bz x n_s x n_s
        # # dist_recon_between_slots = torch.min(dist_recon_between_slots, dim=-1)[0].mean(dim=-1)
        # dist_recon_between_slots = torch.mean(dist_recon_between_slots, dim=-1).mean(dim=-1)
        # slot_slot_indicator = torch.eye(self.num_slots, ).cuda().float()
        # dist_recon_between_slots = (dist_recon_between_slots * (1. - slot_slot_indicator.unsqueeze(0)))
        # dist_recon_between_slots = dist_recon_between_slots.sum(dim=-1).sum(dim=-1) / (float(self.num_slots * (self.num_slots - 1)) / 2.)
        # dist_recon_between_slots = -0.1 * dist_recon_between_slots.mean()
        # print(f"dist_recon_between_slots: {dist_recon_between_slots.item()}")

        ''' Calculate reconstruction loss for each part --- pervious version '''
        # transformed_slot_pts: bz x n_s x na x M x 3; ori_pts: bz x N x 3 ---> bz x n_s x na x M x N
        dist_recon_ori_slot = torch.sum((transformed_slot_pts.unsqueeze(4) - ori_pts.unsqueeze(1).unsqueeze(1).unsqueeze(1)) ** 2, dim=-1)
        # dist_recon_ori_slot = torch.sqrt(dist_recon_ori_slot)
        dist_recon_ori_slot = dist_recon_ori_slot.contiguous().permute(0, 1, 3, 4, 2).contiguous()

        dist_recon_ori_slot = dist_recon_ori_slot * hard_one_hot_labels.unsqueeze(-1).unsqueeze(2) + (
                    1. - hard_one_hot_labels.unsqueeze(-1).unsqueeze(2)) * 1e8 * dist_recon_ori_slot
        # dist_recon_ori_slot[dist_recon_ori_slot > 100.] = 0.

        # if self.soft_attn == 0:
        #     #  bz x n_s x M x N x na; hard_one_hot_labels: bz x n_s x N ---> bz x ns x M x N x na
        #     dist_recon_ori_slot = dist_recon_ori_slot * hard_one_hot_labels.unsqueeze(-1).unsqueeze(2) + (1. - hard_one_hot_labels.unsqueeze(-1).unsqueeze(2)) * 1e8 * dist_recon_ori_slot
        #     dist_recon_ori_slot[dist_recon_ori_slot > 100.] = 0.
        # else:
        #     dist_recon_ori_slot = dist_recon_ori_slot * soft_labels_slot.unsqueeze(-1).unsqueeze(2) # + (1. - hard_one_hot_labels.unsqueeze(-1).unsqueeze(2)) * 1e8 * dist_recon_ori_slot
        #     dist_recon_ori_slot[dist_recon_ori_slot > 100.] = 0.



        # bz x n_s x M x na
        dist_chamfer_recon_slot, orbit_minn_idx = torch.min(dist_recon_ori_slot, dim=-2)

        dist_chamfer_recon_slot[dist_chamfer_recon_slot > 100.] = 0.
        # bz x n_s x na
        dist_chamfer_recon_slot = dist_chamfer_recon_slot.mean(dim=-2) # .mean(dim=-1)
        # dist_chamfer_recon_ori:  bz x ns x N x na; bz x n_s x M x N x na -> bz x n_s x N x na
        # get the chamfer distance
        dist_chamfer_recon_ori, _ = torch.min(dist_recon_ori_slot, dim=2)
        dist_chamfer_recon_ori[dist_chamfer_recon_ori > 100.] = 0.
        if self.soft_attn == 0:
            # dist_chamfer_recon_ori_to_slot: bz x n_s x na
            dist_chamfer_recon_ori_to_slot = torch.sum(dist_chamfer_recon_ori * hard_one_hot_labels.unsqueeze(-1), dim=2) / torch.clamp(torch.sum(hard_one_hot_labels.unsqueeze(-1), dim=2), min=1e-9)
        else:
            # dist_chamfer_recon_ori_to_slot = torch.sum(dist_chamfer_recon_ori * hard_one_hot_labels.unsqueeze(-1), dim=2) / torch.clamp(torch.sum(hard_one_hot_labels.unsqueeze(-1), dim=2), min=1e-9)
            # dist_chamfer_recon_ori_to_slot = torch.sum(dist_chamfer_recon_ori * soft_labels_slot.unsqueeze(-1), dim=2)
            # tmp_weights = (hard_one_hot_labels.unsqueeze(-1) * soft_labels_slot.unsqueeze(-1)) / torch.clamp(torch.sum(soft_labels_slot.unsqueeze(-1), dim=2, keepdim=True), min=1e-9)
            # dist_chamfer_recon_ori_to_slot = torch.sum(dist_chamfer_recon_ori * tmp_weights,
                                                       # dim=2) / torch.clamp(
                # torch.sum(tmp_weights, dim=2), min=1e-9)
            dist_chamfer_recon_ori_to_slot = torch.sum(dist_chamfer_recon_ori * hard_one_hot_labels.unsqueeze(-1), dim=2) / torch.clamp(torch.sum(hard_one_hot_labels.unsqueeze(-1), dim=2), min=1e-9)

        # bz x n_s x na
        dist_chamfer_recon_slot_ori = dist_chamfer_recon_ori_to_slot + dist_chamfer_recon_slot
        # bz x n_s; bz x n_s
        dist_chamfer_recon_slot_ori, selected_slot_oribt = torch.min(dist_chamfer_recon_slot_ori, dim=2)
        ''' Calculate reconstruction loss for each part --- pervious version '''

        ''' Calculate reconstruction loss for each part --- current version '''
        # seg_label_to_pts = []
        # seg_losses = []
        # for i_bz in range(bz):
        #     cur_bz_point_labels = point_label[i_bz, :]
        #     cur_bz_losses = []
        #     cur_bz_label_to_pts = {}
        #     for i_n in range(cur_bz_point_labels.shape[0]):
        #         cur_bz_pts_label = int(cur_bz_point_labels[i_n].item())
        #         if cur_bz_pts_label not in cur_bz_label_to_pts:
        #             cur_bz_label_to_pts[cur_bz_pts_label] = [i_n]
        #         else:
        #             cur_bz_label_to_pts[cur_bz_pts_label].append(i_n)
        #     for i_s in range(self.num_slots):
        #         if i_s not in cur_bz_label_to_pts:
        #             cur_bz_seg_loss = torch.zeros((self.kanchor, ), dtype=torch.float32).cuda()
        #             cur_bz_losses.append(cur_bz_seg_loss.unsqueeze(0))
        #             continue
        #         # n_seg_pts x 3
        #         cur_bz_label_pts = torch.from_numpy(np.array(cur_bz_label_to_pts[i_s])).long() # .cuda().float()
        #         cur_bz_label_pts = ori_pts[i_bz, cur_bz_label_pts, :]
        #         cur_bz_label_pred_pts = transformed_slot_pts[i_bz, i_s, :, :, :]
        #         # print(cur_bz_label_pred_pts.size())
        #         cur_bz_seg_pred_to_ori, cur_bz_seg_ori_to_pred = safe_chamfer_dist_call(cur_bz_label_pred_pts, cur_bz_label_pts.unsqueeze(0).repeat(self.kanchor, 1, 1), self.chamfer_dist)
        #         # cur_bz_seg_loss: na
        #         # cur_bz_seg_loss = (cur_bz_seg_pred_to_ori.mean(dim=-1) + cur_bz_seg_ori_to_pred.mean(dim=-1))
        #         cur_bz_seg_loss = cur_bz_seg_pred_to_ori.mean(dim=-1)
        #         cur_bz_losses.append(cur_bz_seg_loss.unsqueeze(0))
        #     cur_bz_losses = torch.cat(cur_bz_losses, dim=0)
        #     seg_losses.append(cur_bz_losses.unsqueeze(0))
        #
        # # bz x n_s x na
        #
        # seg_losses = torch.cat(seg_losses, dim=0)
        # # print(f"seg_losses.size: {seg_losses.size()}")
        # dist_recon_ori_slot = seg_losses
        # dist_chamfer_recon_slot_ori, selected_slot_oribt = torch.min(dist_recon_ori_slot, dim=-1)
        ''' Calculate reconstruction loss for each part --- current version '''
        # for i_s in range(self.num_slots):
            # point_label: bz x N
        # print(dist_chamfer_recon_slot_ori)


        # bz
        #### Use soft slot weights to aggregate slots' losses ####
        # print(dist_chamfer_recon_slot_ori)
        # slot_weights_loss = torch.ones_like(slot_weights)
        # slot_weights_loss[:, 1] = 2.
        avg_dist_chamfer_recon_slot_ori = torch.sum(slot_weights * dist_chamfer_recon_slot_ori, dim=-1)
        # avg_dist_chamfer_recon_slot_ori = torch.sum(slot_weights_loss * dist_chamfer_recon_slot_ori, dim=-1)

        #### Use hard slot average weights to aggregate slots' losses ####
        # avg_dist_chamfer_recon_slot_ori = torch.mean(dist_chamfer_recon_slot_ori, dim=-1)
        # avg_dist_chamfer_recon_slot_ori = torch.sum(dist_chamfer_recon_slot_ori, dim=-1)
        #### Use hard slot weights to aggregate slots' losses ####
        # avg_dist_chamfer_recon_slot_ori = torch.sum(hard_slot_weights * dist_chamfer_recon_slot_ori, dim=-1)
        # avg_dist_chamfer_recon_slot_ori = torch.sum(hard_slot_indicator * dist_chamfer_recon_slot_ori, dim=-1)  / torch.clamp(torch.sum(hard_slot_indicator, dim=-1), min=1e-9)

        # transformed_slot_pts: bz x n_s x na x M x 3;
        # selected_recon_slot_points: bz x n_s x M x 3
        selected_recon_slot_points = batched_index_select(values=transformed_slot_pts, indices=selected_slot_oribt.unsqueeze(-1), dim=2).squeeze(2)
        # recon_slot_points: bz x n_s x na x M x 3
        # selected_ori_recon_slot_points: bz x n_s x M x 3
        # selected_ori_recon_slot_points = batched_index_select(values=recon_slot_points, indices=selected_slot_oribt.unsqueeze(-1), dim=2).squeeze(2)
        # selected_ori_recon_slot_points = recon_slot_points
        # hard_slot_indicator: bz x n_s

        # purity_loss = get_purity_loss(selected_ori_recon_slot_points)

        ''' if we eliminate slots via the number of points? '''
        # hard_slot_indicator = (hard_one_hot_labels.sum(-1) > 10.).float()

        # print(f"selected_recon_slot_points: {selected_recon_slot_points.size()}")

        # bz x (n_s * M) x 3
        #
        expanded_recon_slot_points = (selected_recon_slot_points * hard_slot_indicator.unsqueeze(-1).unsqueeze(-1)).contiguous().view(bz, self.num_slots * self.recon_part_M, 3)
        # print(selected_ori_recon_slot_points.size())

        ''' expanded_ori_recon_slot_points '''
        # expanded_ori_recon_slot_points = (selected_ori_recon_slot_points * hard_slot_indicator.unsqueeze(-1).unsqueeze(-1)).contiguous().view(bz, self.num_slots * self.recon_part_M, 3)

        ''' Sample points from ori_recon_slot_points '''
        # fps_idx_ori = farthest_point_sampling(expanded_ori_recon_slot_points, npoints)
        # sampled_ori_recon_pts = expanded_ori_recon_slot_points.contiguous().view(bz * (self.recon_part_M) * (self.num_slots), -1)[fps_idx_ori, :].contiguous().view(bz, npoints, -1)
        # ''' Get loss between reconstructed whole shape and reference whole shape '''
        # expanded_ref_pts = self.ref_whole_shape.unsqueeze(0).repeat(bz, 1, 1)
        # ori_glb_dist_recon_to_ori, ori_glb_dist_ori_to_recon = safe_chamfer_dist_call(
        #     sampled_ori_recon_pts, expanded_ref_pts, self.chamfer_dist)
        # ref_whole_shape_recon_loss = (ori_glb_dist_recon_to_ori.mean(dim=-1) + ori_glb_dist_ori_to_recon.mean(dim=-1)).mean()

        # sampled_recon_pts: bz x npoints x 3
        fps_idx = farthest_point_sampling(expanded_recon_slot_points, npoints)
        # sampled_recon_pts: bz x N x 3
        sampled_recon_pts = expanded_recon_slot_points.contiguous().view(bz * (self.recon_part_M) * (self.num_slots),
                                                                         -1)[fps_idx, :].contiguous().view(bz, npoints,
                                                                                                           -1)
        ''' Recon to Ori dist --- v1 '''
        ''' Sample points from transformed reconstructed slot points '''

        # ''' Get loss between transformed reconstructed whole shape and input original shape '''
        # # glb_dist_recon_to_ori: (bz * na) x npoints; glb_dist_ori_to_recon: bz x npoints
        # glb_dist_recon_to_ori, glb_dist_ori_to_recon = safe_chamfer_dist_call(
        #     sampled_recon_pts, ori_pts, self.chamfer_dist )
        ''' Recon to Ori dist --- v1 '''

        ''' Recon to Ori dist --- v2 '''
        # bz x npoints x npoints
        glb_dist_recon_ww_ori = torch.sum((sampled_recon_pts.unsqueeze(2) - ori_pts.unsqueeze(1)) ** 2, dim=-1)
        # glb_dist_recon_ww_ori = torch.sqrt(glb_dist_recon_ww_ori)
        glb_dist_recon_to_ori, _ = torch.min(glb_dist_recon_ww_ori, dim=-1)
        glb_dist_ori_to_recon, _ = torch.min(glb_dist_recon_ww_ori, dim=-2)
        ''' Recon to Ori dist --- v2 '''

        # bz
        # glb_dist_recon_to_ori = glb_dist_recon_to_ori.mean(dim=-1) # .mean()
        glb_dist_recon_to_ori = glb_dist_recon_to_ori.sum(dim=-1) # .mean()
        # bz
        # glb_dist_ori_to_recon = glb_dist_ori_to_recon.mean(dim=-1) # .mean()
        glb_dist_ori_to_recon = glb_dist_ori_to_recon.sum(dim=-1) # .mean()

        ''' Get canonical reconstruction loss '''
        # expanded_oir_recon_slot_points = (recon_slot_points * hard_slot_indicator.unsqueeze(-1).unsqueeze(-1)).contiguous().view(bz, self.num_slots * self.recon_part_M, 3)
        # ori_fps_idx = farthest_point_sampling(expanded_oir_recon_slot_points, npoints)
        # # sampled_recon_pts: bz x N x 3
        # sampled_ori_recon_pts = expanded_oir_recon_slot_points.contiguous().view(bz * (self.recon_part_M) * (self.num_slots), -1)[ori_fps_idx, :].contiguous().view(bz, npoints, -1)
        # glb_dist_recon_ww_ori_canon = torch.sum((sampled_ori_recon_pts.unsqueeze(2) - canon_pc.unsqueeze(1)) ** 2, dim=-1)
        # glb_dist_recon_to_ori_canon, _ = torch.min(glb_dist_recon_ww_ori_canon, dim=-1)
        # glb_dist_ori_to_recon_canon, _ = torch.min(glb_dist_recon_ww_ori_canon, dim=-2)
        # glb_dist_recon_to_ori_canon = glb_dist_recon_to_ori_canon.mean(dim=-1)
        # glb_dist_ori_to_recon_canon = glb_dist_ori_to_recon_canon.mean(dim=-1)
        # recon_canon = glb_dist_recon_to_ori_canon + glb_dist_ori_to_recon_canon
        ''' Get canonical reconstruction loss '''

        glb_recon_factor = 2.0
        glb_recon_factor = self.glb_recon_factor
        slot_recon_factor = 3.0  # .0
        slot_recon_factor = self.slot_recon_factor
        summ_factor = glb_recon_factor + slot_recon_factor
        # glb_recon_factor = glb_recon_factor / summ_factor
        # slot_recon_factor = slot_recon_factor / summ_factor

        # tot_recon_loss = glb_dist_ori_to_recon + glb_dist_recon_to_ori + 2.0 * avg_dist_chamfer_recon_slot_ori # + 2.0 * recon_canon
        tot_recon_loss = glb_recon_factor * (
                    glb_dist_ori_to_recon + glb_dist_recon_to_ori) + slot_recon_factor * avg_dist_chamfer_recon_slot_ori
        # tot_recon_loss = 2.0 * avg_dist_chamfer_recon_slot_ori  # + 2.0 * recon_canon
        # tot_recon_loss = avg_dist_chamfer_recon_slot_ori
        # tot_recon_loss = 2.0 * avg_dist_chamfer_recon_slot_ori

        tot_recon_loss = tot_recon_loss.mean()

        out_feats = {}

        # bz x na x N
        # print(f"point_labels.size: {point_label.size()}, minn_orbit_idx: {minn_orbit_idx.size()}")

        # minn_orbit_idx = minn_orbit_idx.unsqueeze(-1)

        selected_point_labels = point_label

        selected_recon_slot_pts = selected_recon_slot_points

        ori_selected_recon_slot_pts = ori_recon_slot_points

        selected_expanded_sampled_recon_pts = sampled_recon_pts
        #
        # print("attn.size:", attn.size())
        selected_attn = attn
        # selected_attn = expanded_attn[:, 0, ...]
        # print("selected_attn.size:", selected_attn.size())
        #
        out_feats['vis_pts_hard'] = ori_pts.detach().cpu().numpy()
        out_feats['vis_labels_hard'] = selected_point_labels.detach().cpu().numpy()
        if self.gt_oracle_seg == 3:
            out_feats['vis_labels_hard_2'] = pred_labels_2.detach().cpu().numpy()
        out_feats['ori_recon_slot_pts_hard'] = ori_selected_recon_slot_pts.detach().cpu().numpy()

        out_feats['recon_slot_pts_hard'] = selected_recon_slot_pts.detach().cpu().numpy()
        out_feats['sampled_recon_pts_hard'] = selected_expanded_sampled_recon_pts.detach().cpu().numpy()

        if self.inv_attn == 1:
            out_feats['orbit_attn_weights'] = orbit_attn_weights.detach().cpu().numpy()

        out_feats['attn'] = attn.detach().cpu().numpy()

        if cur_iter == 0:
            self.attn_iter_0 = attn.contiguous().transpose(1, 2).contiguous().detach() # .cpu().numpy()
        elif cur_iter == 1:
            self.attn_iter_1 = attn.contiguous().transpose(1, 2).contiguous().detach() # .cpu().numpy()
        elif cur_iter == 2:
            self.attn_iter_2 = attn.contiguous().transpose(1, 2).contiguous().detach() # .cpu().numpy()
        # out_feats['orbit_minn_idx'] = orbit_minn_idx.detach().cpu().numpy()

        ''' Predict rotations '''
        # selected_labels: bz x N
        selected_labels = torch.argmax(selected_attn, dim=-1)
        # pred_R: bz x n_s x na x 3 x 3 --> selected_pred_R: bz x n_s x 3 x 3
        selected_pred_R = batched_index_select(values=pred_R, indices=selected_slot_oribt.unsqueeze(-1), dim=2)
        selected_pred_R = selected_pred_R.squeeze(2)
        selected_pred_R_saved = selected_pred_R.detach().clone()
        # selected_pred_R: bz x N x 3 x 3
        selected_pred_R = batched_index_select(values=selected_pred_R, indices=selected_labels, dim=1)

        # bz x
        out_feats['pred_slot_Rs'] = selected_pred_R.detach().cpu().numpy()

        selected_pred_T = batched_index_select(values=pred_T, indices=selected_slot_oribt.unsqueeze(-1), dim=2)
        selected_pred_T = selected_pred_T.squeeze(2)
        # selected_pred_R: bz x N x 3 x 3
        selected_pred_T = batched_index_select(values=selected_pred_T, indices=selected_labels, dim=1)

        out_feats['pred_slot_Ts'] = selected_pred_T.detach().cpu().numpy()

        # selected_inv_pred_R: bz x N x 3 x 3; ori_pts: bz x N x 3
        selected_inv_pred_R = selected_pred_R.contiguous().transpose(-1, -2).contiguous()
        #
        # rotated_ori_pts = torch.matmul(selected_pred_R, ori_pts.contiguous().unsqueeze(-1).contiguous()).squeeze(-1).contiguous()
        # From transformed points to original canonical points
        transformed_ori_pts = torch.matmul(selected_inv_pred_R, (ori_pts - selected_pred_T).unsqueeze(-1)).squeeze(-1)
        # transformed_ori_pts = torch.matmul(selected_inv_pred_R, (rotated_ori_pts - selected_pred_T).unsqueeze(-1)).squeeze(-1)

        out_feats['transformed_ori_pts'] = transformed_ori_pts.detach().cpu().numpy()

        if gt_pose is not None:
            gt_R = gt_pose[..., :3, :3]
            gt_T = gt_pose[..., :3, 3]
            gt_inv_R = gt_R.contiguous().transpose(-1, -2).contiguous()
            gt_transformed_ori_pts = torch.matmul(gt_inv_R, (ori_pts - gt_T).unsqueeze(-1)).squeeze(-1)
            out_feats['gt_transformed_ori_pts'] = gt_transformed_ori_pts.detach().cpu().numpy()

        np.save(self.log_fn + f"_n_iter_{cur_iter}.npy", out_feats)

        out_feats['inv_feats'] = invariant_feats_npy
        np.save(self.log_fn + f"_n_iter_{cur_iter}_with_feats.npy", out_feats)

        pred_pose = torch.cat([selected_pred_R, torch.zeros((bz, npoints, 3, 1), dtype=torch.float32).cuda()],
                              dim=-1)
        pred_pose = torch.cat([pred_pose, torch.zeros((bz, npoints, 1, 4), dtype=torch.float32).cuda()], dim=-2)

        out_feats['pred_R_slots'] = selected_pred_R_saved.cpu().numpy()

        self.out_feats = out_feats

        if cur_iter == self.num_iters - 1 and gt_pose is not None:
            gt_Rs = gt_pose[..., :3, :3]
            rot_dist = get_dist_two_rots(gt_Rs, selected_pred_R)
            rot_dist = rot_dist.mean()
        else:
            rot_dist = None

        # out_feats['x_features_hard'] = .squeeze(-1).detach().cpu().numpy()
        # np.save("out_feats_with_features_10.npy", out_feats)

        # get tot loss
        ''' If use soft attention and reconstruction '''
        # tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot + purity_loss
        # tot_loss = tot_recon_loss
        # tot_loss = tot_recon_loss + ref_whole_shape_recon_loss

        # if rot_dist is not None:
        #     tot_loss = tot_recon_loss + rot_dist
        # else:
        #     tot_loss = tot_recon_loss

        # tot_loss = tot_recon_loss + 0.05 * cr_loss
        tot_loss = tot_recon_loss
        # print(f"cr_loss: {cr_loss}")
        # tot_loss = tot_recon_loss + purity_loss
        # tot_loss = tot_recon_loss + tot_ref_shp_recon_loss
        # tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        ''' If use hard attention and reconstruction '''
        # tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon #  + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        # tot_loss = avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        # return tot_loss, attn
        return tot_loss, selected_attn, pred_pose, out_feats

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

    def forward(self, x, pose, ori_pc=None, rlabel=None, nn_inter=2, pose_segs=None, canon_pc=None):

        # loss, attn = self.forward_one_iter(x, pose, rlabel=rlabel)
        # return loss, attn
        bz, np = x.size(0), x.size(2)
        init_pose = torch.zeros([bz, np, 4, 4], dtype=torch.float32).cuda()
        init_pose[..., 0, 0] = 1.; init_pose[..., 1, 1] = 1.; init_pose[..., 2, 2] = 1.
        # init_pose = pose
        tot_loss = 0.0
        cur_transformed_points = x
        cur_estimated_pose = init_pose
        # nn_inter = 1
        nn_inter = self.num_iters
        # cur_estimated_pose = pose
        out_feats_all_iters = {}
        for i in range(nn_inter):
            cur_reconstructed_loss_orbit, attn, cur_estimated_pose, cur_out_feats = self.forward_one_iter(cur_transformed_points, cur_estimated_pose, ori_pc=ori_pc, rlabel=rlabel, cur_iter=i, gt_pose=pose, gt_pose_segs=pose_segs, canon_pc=canon_pc)
            torch.cuda.empty_cache()
            tot_loss += cur_reconstructed_loss_orbit
            # cur_gt_rot_dis = self.get_rotation_sims(pose, cur_estimated_pose)
            torch.cuda.empty_cache()

            out_feats_all_iters[i] = cur_out_feats


        # return tot_loss / nn_inter, attn # range(n_iter)
        self.out_feats_all_iters = out_feats_all_iters
        return tot_loss / nn_inter

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
        'glb_recon_factor': opt.equi_settings.glb_recon_factor,
        'slot_recon_factor': opt.equi_settings.slot_recon_factor,
        # 'opt': opt

    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
