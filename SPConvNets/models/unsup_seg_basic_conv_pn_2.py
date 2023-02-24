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
        ''' Construct DGCNN backbone '''
        class Args():
            def __init__(self):
                pass
        args = Args()
        args.dgcnn_out_dim = 512; args.dgcnn_in_feat_dim = 3; args.dgcnn_layers = 3; args.nn_nb = 80; args.input_normal = False
        args.dgcnn_out_dim = 256; args.backbone = 'DGCNN'
        self.dgcnn = PrimitiveNet(args)

        ''' For PT2PC's PartDecoder '''
        cubes = load_pts('cube.pts')
        print(f"Cubes' points loaded: {cubes.shape}")
        self.register_buffer('cubes', torch.from_numpy(cubes))

        self.num_iters = params['general']['num_iters']
        self.global_rot = params['general']['global_rot']
        self.npoints = params['general']['npoints']
        self.batch_size = params['general']['batch_size']
        self.init_lr = params['general']['init_lr']
        self.part_pred_npoints = params['general']['part_pred_npoints']
        self.use_equi = params['general']['use_equi']
        self.model_type = params['general']['model_type']
        self.decoder_type = params['general']['decoder_type']

        self.log_fn = f"out_feats_woeq_wrot_{self.global_rot}_equi_{self.use_equi}_model_{self.model_type}_decoder_{self.decoder_type}_num_iters_{self.num_iters}_npts_{self.npoints}_perpart_npts_{self.part_pred_npoints}_bsz_{self.batch_size}_init_lr_{self.init_lr}"

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
        # self.encoded_feat_dim = 1024;
        self.slot_attention = SlotAttention(num_slots=params['outblock']['k'], dim=self.encoded_feat_dim, hidden_dim=self.encoded_feat_dim)
        ''' Construct per-point variant feature transformation MLP '''
        self.variant_feat_trans = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=self.encoded_feat_dim,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
        )
        ''' Construct per-slot rotation and translation prediction MLP ''' # rotation and translation MLP
        self.transformation_dim = 7
        # will such batchnorms affect the transformation prediction? ---- or is it suitable to use BNs in the transformation prediction net?
        # predict transformations and use transformations to transform predicted canonical points' coordinates
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
            # nn.BatchNorm2d(num_features=self.encoded_feat_dim // 4),
            # nn.ReLU(),
        )
        ''' Construct part point construction network '''
        # todo: better reconstruction process
        self.recon_part_M = 96
        self.recon_part_M = 128
        self.recon_part_M = 1024
        # self.recon_part_M = 1500
        self.encoded_feat_dim = 1024
        self.recon_part_M = 512
        self.recon_part_M = 128
        self.recon_part_M = self.part_pred_npoints
        # self.recon_part_M = 512
        # self.recon_part_M = 256
        self.part_reconstruction_net = nn.Sequential(
            nn.Conv2d(in_channels=self.encoded_feat_dim, out_channels=1024, kernel_size=(1, 1),
                      stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=1024),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            # nn.BatchNorm2d(num_features=1024),
            # nn.ReLU(),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv2d(in_channels=1024, out_channels=1024,
            #           kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=1024),
            # nn.ReLU(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=self.recon_part_M * 3,
                      kernel_size=(1, 1), stride=(1, 1), bias=True),
            # nn.Sigmoid()
        )
        for zz in self.part_reconstruction_net:
            if isinstance(zz, nn.Conv2d):
                torch.nn.init.xavier_uniform_(zz.weight)
                if zz.bias is not None:
                    torch.nn.init.zeros_(zz.bias)
        # n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False
        # self.decoder = DecoderFC(n_features=(512,1024), latent_dim=256, output_pts=512, bn=True)

        ''' If using PT2PC's PartDecoder '''
        # self.part_reconstruction_net = PartDecoder(self.encoded_feat_dim, recon_M=self.recon_part_M)

        ''' Construct encoder '''
        self.encoded_feat_dim = 1024
        # self.encoded_feat_dim = 512
        self.fc_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 3), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=self.encoded_feat_dim, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=self.encoded_feat_dim),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(),

        )

        ''' Construct decoder '''
        self.fc_decoder = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, stride=(1, 1), kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(),

            nn.Conv2d(in_channels=1024, out_channels=1024, stride=(1, 1), kernel_size=(1, 1), bias=True),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(),

            # Set reconstruction layers
            nn.Conv2d(in_channels=1024, out_channels=self.recon_part_M * 3, stride=(1, 1), kernel_size=(1, 1), bias=True)
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
        # print(f"batch_size: {x.size(0)}")
        # print(torch.max(x[:, 0, :]), torch.min(x[:, 0, :]))
        # print(torch.max(x[:, 1, :]), torch.min(x[:, 1, :]))
        # print(torch.max(x[:, 2, :]), torch.min(x[:, 2, :]))
        output = {}
        #
        # should with pose as input; actually an orientation of each point --- not orientation... te rotation has been done? fix the global pose of each input point; relative pose relations --- how to compute? create laptops and put the joint-center to the original point
        ''' A simple pure whole-shape reconstruction '''
        # # # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na];
        # # # centralize points
        # x = x - torch.mean(x, dim=-1, keepdim=True)
        # x_min, _ = torch.min(x, dim=2, keepdim=True)
        # x_max, _ = torch.max(x, dim=2, keepdim=True)
        # # x = ((x - x_min) / (x_max - x_min) - 0.5) * 2.
        # ori_pts = x.clone()
        # # bz and number of points
        # bz, npoints = x.size(0), x.size(2)
        #
        # ''' Reshape the tensor x '''
        # x = x.contiguous().transpose(1, 2).contiguous().unsqueeze(1)
        #
        # ''' Batch size and number of points '''
        # encoded_x = self.fc_encoder(x) # bz x 1 x np x 3 ---> bz x num_features x N x 1
        #
        # ''' Get encoded latent features for each (npoint) shape '''
        # latent, _ = torch.max(encoded_x, dim=2, keepdim=True)
        # # while len(latent.size()) > 2:
        # #     latent = latent.squeeze(-1)
        # ''' Decode latent representation for each code '''
        # ''' Decode latent representations for decoded position information for each shape '''
        # decoded_pos = self.fc_decoder(latent)
        # decoded_pos = decoded_pos.squeeze(-1).squeeze(-1)
        # decoded_pos = decoded_pos.contiguous().view(bz, self.recon_part_M, 3).contiguous()
        #
        # ''' Get reconstruction lsos between original shape and reconstructed shape '''
        # dist1_glb, dist2_glb = self.chamfer_dist(
        #     decoded_pos, ori_pts.contiguous().transpose(1, 2).contiguous(), return_raw=True
        # )
        #
        # ''' Get reconstruction loss for global shape and local part '''
        # glb_recon_loss = dist1_glb.mean(-1) + dist2_glb.mean(-1)
        # # print(slot_weights.size(), dist2.size())
        # recon_loss = glb_recon_loss.mean()
        # attn = torch.zeros((bz, npoints, 200)).cuda()
        # attn[:, : npoints // 2, 0] = 1.
        # attn[:, npoints // 2:, 0] = 0.
        # attn[:, : npoints // 2, 1] = 0.
        # attn[:, npoints // 2:, 1] = 1.
        # # if recon_loss.item() < 0.06:
        # np.save("ori_pts.npy", ori_pts.detach().cpu().numpy())
        # np.save("downsampled_pts.npy", decoded_pos.detach().cpu().numpy())
        # return recon_loss, attn

        ''' A simple pure whole-shape reconstruction '''
        # #
        # # input_x = x # preprocess input and equivariant
        # ''' Using Conv net '''
        # #### Preprocess input and equivariant feature transformation ####
        # x = M.preprocess_input(x, self.na_in, pose, False)
        # for block_i, block in enumerate(self.backbone):
        #     x = block(x)
        # x_feats, _ = torch.max(x.feats, dim=2, keepdim=True)
        # ''' Using Conv net '''
        #
        # # x, _, _ = self.dgcnn(x.transpose(1, 2).contiguous(), normal=x.transpose(1, 2).contiguous())
        # # x_feats = x.contiguous().transpose(1, 2).contiguous().unsqueeze(-1)
        # # x_feats, _ = torch.max(x_feats, dim=2, keepdim=True)
        # # x.feats.size = bz x n_dim x n_p x a
        #
        # # bz x (128 * 3) x 1 x a
        # recons_points = (self.part_reconstruction_net(x_feats) - 0.5) * 2
        # # print(x_feats.size())
        # # recons_points = self.decoder(x_feats.contiguous().transpose(1, 2).contiguous().squeeze(-1)) - 0.5
        # recons_points = recons_points.squeeze(-1).squeeze(-1)
        # recons_points = recons_points.contiguous().view(bz, self.recon_part_M, 3).contiguous()
        #
        # dist1_glb, dist2_glb = self.chamfer_dist(
        #     recons_points, ori_pts.contiguous().transpose(1, 2).contiguous(), return_raw=True
        # )
        #
        # ''' Get reconstruction loss for global shape and local part '''
        # glb_recon_loss = dist1_glb.mean(-1) + dist2_glb.mean(-1)
        # # print(slot_weights.size(), dist2.size())
        # recon_loss = glb_recon_loss.mean()
        # attn = torch.zeros((bz, npoints, 200)).cuda()
        # attn[:, : npoints // 2, 0] = 1.; attn[:, npoints // 2:, 0] = 0.
        # attn[:, : npoints // 2, 1] = 0.
        # attn[:, npoints // 2: , 1] = 1.
        # if recon_loss.item() < 0.06:
        #     np.save("ori_pts.npy", ori_pts.detach().cpu().numpy())
        #     np.save("downsampled_pts.npy", recons_points.detach().cpu().numpy())
        # return recon_loss, attn
        #
        # x: (out_feature: bz x C x N x 1; atten_weight: bz x N x A)
        # get position information
        #
        ''' Only Whole-shape reconstruction '''
        # # # print(x.size())
        # x = x - torch.mean(x, dim=-1, keepdim=True)
        # ori_pts = x.clone()
        # #
        # bz, npoints = x.size(0), x.size(2)
        # # input_x = x # preprocess input and equivariant
        # ''' Preprocess input to get input for the network '''
        # x = M.preprocess_input(x, self.na_in, pose, False)
        # for block_i, block in enumerate(self.backbone):
        #     x = block(x)
        # # x_xyz = x.xyz;
        # # x_anchors = x.anchors;
        # # x_pose = x.pose
        #
        # # x = self.fc_encoder(x.contiguous().transpose(1, 2).contiguous().unsqueeze(1))
        # ''' Cluster points via slot attention; clustering; inv_feat '''
        # inv_feat, atten_score = self.outblock(x, rlabel)
        # # inv_feat = x
        # # inv_feat = inv_feat.squeeze(-1)  # squeeze invariant features
        # inv_feat, _ = torch.max(inv_feat, dim=2, keepdim=True)
        # recon_pts = self.part_reconstruction_net(inv_feat)
        # recon_pts = recon_pts.squeeze(-1).squeeze(-1)
        # recon_pts = save_view(recon_pts, (bz, self.recon_part_M, 3))
        #
        # glb_dist_recon_to_ori, glb_dist_ori_to_recon = safe_chamfer_dist_call(
        #     recon_pts, ori_pts.contiguous().transpose(1, 2).contiguous(), self.chamfer_dist
        # )
        #
        # dist_loss = glb_dist_recon_to_ori.mean(dim=-1).mean() + glb_dist_ori_to_recon.mean(dim=-1).mean()
        #
        # attn = torch.zeros((bz, npoints, 4), dtype=torch.float).cuda()
        # attn[:, :npoints // 2, 0] = 1.
        # attn[:, npoints // 2:, 1] = 1.
        #
        # np.save("ori_pts_only_recon.npy", ori_pts.detach().cpu().numpy())
        # np.save("reconstructed_pts_only_recon.npy", recon_pts.detach().cpu().numpy())
        #
        # return dist_loss, attn

        ''' Reconstruction with slot-attention '''
        # centralize points
        x = x - torch.mean(x, dim=-1, keepdim=True)
        ori_pts = x.clone()
        bz, npoints = x.size(0), x.size(2)
        # input_x = x # preprocess input and equivariant

        ''' Preprocess input to get input for the network '''
        x = M.preprocess_input(x, self.na_in, pose, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        x_xyz = x.xyz; x_anchors = x.anchors; x_pose = x.pose

        ''' Cluster points via slot attention; clustering; inv_feat '''
        inv_feat, atten_score = self.outblock(x, rlabel)

        # np.save("x_features_multi.npy", inv_feat.squeeze(-1).detach().cpu().numpy())
        # np.save("x_features_hard_5.npy", inv_feat.squeeze(-1).detach().cpu().numpy())

        ''' If using fc-encoder to encode the point cloud '''
        # x = x.contiguous().transpose(1, 2).contiguous().unsqueeze(1)
        # x = self.fc_encoder(x)
        # inv_feat = x.squeeze(-1)

        # inv_feat: bz x c_out x N x 1
        # get features, invariant features?
        # inv_feat: bz x c_out x N x 1 -> bz x c_out x N
        inv_feat = inv_feat.squeeze(-1)  # squeeze invariant features
        ''' Cluster '''
        # inv_feat: bz x c_out x N -> bz x N x c_out
        # it seems that it is hard for us to cluster points via part-by-part partitioning?
        rep_slots, attn_ori = self.slot_attention(inv_feat.contiguous().transpose(1, 2).contiguous())
        # attn_ori[:, 0, :npoints // 2] = 1.; attn_ori[:, 0, npoints // 2:] = 0.
        # attn_ori[:, 1, :npoints // 2] = 0.; attn_ori[:, 1, npoints // 2:] = 1.
        ''' Attention from each point to each cluster '''
        # attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True); bz x N x num_slots
        ### for vis ###
        point_label = torch.argmax(attn_ori, dim=1)
        # if not os.path.exists("vis_pts.npy"):

        hard_one_hot_labels = torch.eye(self.num_slots, dtype=torch.float32).cuda()[point_label]
        hard_one_hot_labels = safe_transpose(hard_one_hot_labels, 1, 2)
        # weights from each point to each slot
        hard_one_hot_labels_slot = hard_one_hot_labels / torch.clamp(hard_one_hot_labels.sum(dim=-1, keepdim=True), min=1e-9)

        # np.save("vis_pts_multi.npy", ori_pts.detach().cpu().numpy())
        # np.save("vis_labels_multi.npy", point_label.detach().cpu().numpy())
        out_feats = {}

        out_feats['vis_pts_hard'] = ori_pts.detach().cpu().numpy()
        out_feats['vis_labels_hard'] = point_label.detach().cpu().numpy()

        # np.save("vis_pts_hard_5.npy", ori_pts.detach().cpu().numpy())
        # np.save("vis_labels_hard_5.npy", point_label.detach().cpu().numpy())

        ### for vis ###
        attn = attn_ori.contiguous().transpose(1, 2).contiguous()
        # slot_weights: bz x num_slots
        slot_weights = attn.sum(dim=1)
        slot_weights = slot_weights / torch.sum(slot_weights, dim=-1, keepdim=True)
        # attn_ori: bz x num_slots x N -> bz x N x num_slots
        # attn_slot: bz x num_slots x N
        attn_slot = attn_ori / torch.clamp(attn_ori.sum(dim=-1, keepdim=True), min=1e-9)
        attn_ori = attn_ori.contiguous().transpose(1, 2).contiguous()

        ''' Get variant features for each slot '''
        # transformed_variant_feats = self.variant_feat_trans(x.feats) # variant feature transformation
        try:
            transformed_variant_feats = x.feats
        except:
            transformed_variant_feats = x
        ''' Aggregate variant feats for each slot '''
        # todo: is it the most suitable way to aggregate features?
        # transformed_variant_feats: bz x c_out x N x na; attn_ori: bz x num_slots x N
        # variant_feats_slot: bz x c_out x num_slots x N x na -> bz x c_out x num_slots x na
        ''' Use soft attention and soft feature aggregation '''
        # variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * attn_slot.unsqueeze(1).unsqueeze(-1),
        #                                dim=3)
        ''' Use hard attention and hard feature aggregation '''
        # bz x num_dim_feat x num_slot
        # transformed_variant_feats: bz x n_feat_dim x 1 x N x na xxxx bz x 1 x n_s x N x 1
        variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * hard_one_hot_labels_slot.unsqueeze(1).unsqueeze(-1), dim=3)
        # bz x 3 x num_slot
        # print(ori_pts.size(), hard_one_hot_labels_slot.size())
        # ori_pts: bz x 3 x 1 x N xxxx bz x 1 x n_s x N -> bz x 3 x n_s x N -> bz x 3 x n_s
        pts_slot = torch.sum(ori_pts.unsqueeze(2) * hard_one_hot_labels_slot.unsqueeze(1),
                                       dim=3)
        pts_slot = safe_transpose(pts_slot, 1, 2)

        ''' From aggregated cluster features to reconstructed points for different slots '''
        recon_slot_points = self.part_reconstruction_net(variant_feats_slot) # points recon
        recon_slot_points = recon_slot_points.squeeze(-1)
        recon_slot_points = recon_slot_points.contiguous().transpose(1, 2).contiguous().view(bz, self.num_slots, self.recon_part_M, -1)

        per_slot_transformation = self.transformation_prediction(variant_feats_slot)
        # bz x n_feats x n_slots x 1
        pred_R, pred_T = per_slot_transformation[:, :4, ...], per_slot_transformation[:, 4:, ...]
        # print(pred_R.size())
        pred_R = compute_rotation_matrix_from_quaternion(
            pred_R.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, 4)).contiguous().view(bz, self.num_slots,
                                                                                                 3, 3)

        ''' From predicted T to the real translation vector --- part rotation modeling '''
        # pred_res_T: bz x num_slots x na x 3 -> bz x n_s x 3
        pred_T = pred_T.contiguous().permute(0, 2, 3, 1).contiguous().squeeze(-2)
        # pred_T: bz x num_slots x na x 3
        # todo: other transformation strategy, like those used in equi-pose?
        out_feats['ori_recon_slot_pts_hard'] = recon_slot_points.detach().cpu().numpy()
        ''' From predicted rotation matrix and translation matrix to transformed points  '''
        # transformed_slot_pts: bz x num_slots x M x 3; bz x n_s x 3 x 3    xxxx    bz x n_s x M x 3; bz x
        transformed_slot_pts = torch.matmul(pred_R, recon_slot_points.contiguous().transpose(-1, -2)).contiguous().transpose(-1, -2) #  + pred_T.unsqueeze(-2)
        # transformed_slot_pts = torch.matmul(pred_R, recon_slot_points.contiguous().transpose(-1, -2)).contiguous().transpose(-1, -2)  + pred_T.unsqueeze(-2)
        recon_slot_points = transformed_slot_pts

        # purity_loss = get_purity_loss(recon_slot_points)

        ''' If use PT2PC's PartDecoder '''
        # expaneded_variant_feats_slot = save_view(variant_feats_slot, (bz * self.num_slots, self.encoded_feat_dim))
        # recon_slot_points = self.part_reconstruction_net(expaneded_variant_feats_slot, self.cubes)
        # recon_slot_points = recon_slot_points.contiguous().view(bz, self.num_slots, self.recon_part_M, 3).contiguous()

        # recon_slot_points: bz x n_s x M x 3
        ''' If we only reconstruct centralized points '''
        # recon_slot_points = recon_slot_points + pts_slot.unsqueeze(-2)

        ori_pts = ori_pts.contiguous().transpose(1, 2).contiguous()
        # filter out weak slots
        hard_slot_indicator = (slot_weights > 1e-4).float()
        expanded_recon_slot_points = (recon_slot_points * hard_slot_indicator.unsqueeze(-1).unsqueeze(-1)).contiguous().view(bz, self.recon_part_M * self.num_slots, -1).contiguous()

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



        # print()
        dist_recon_ori_slot = torch.sum((recon_slot_points.unsqueeze(3) - ori_pts.unsqueeze(1).unsqueeze(1)) ** 2, dim=-1)
        # print(recon_slot_points.size(), dist_recon_ori_slot.size())
        # bz x ns x M x N; hard_one_hot_labels: bz x ns x N ---> bz x ns x M x N
        dist_recon_ori_slot = dist_recon_ori_slot * hard_one_hot_labels.unsqueeze(2) + (1. - hard_one_hot_labels.unsqueeze(2)) * 1e8 * dist_recon_ori_slot
        # bz x npoints x recon_M
        dist_chamfer_recon_slot, _ = torch.min(dist_recon_ori_slot, dim=-1)
        dist_chamfer_recon_slot = dist_chamfer_recon_slot.mean(dim=-1) # .mean(dim=-1)
        # dist_chamfer_recon_ori.
        dist_chamfer_recon_ori, _ = torch.min(dist_recon_ori_slot, dim=2)
        # dist_chamfer_ori.size = bz x npoints
        dist_chamfer_recon_ori, _ = torch.min(dist_chamfer_recon_ori, dim=1)
        # avg_slots_pts_dist_to_shp = dist_chamfer_recon_slot.

        # print(dist_chamfer_recon_slot.size(), slot_weights.size())
        dist_chamfer_recon_slot[dist_chamfer_recon_slot > 10.] = 0.
        dist_chamfer_recon_ori[dist_chamfer_recon_ori > 10.] = 0.
        avg_slots_pts_dist_to_shp = torch.sum(dist_chamfer_recon_slot * slot_weights, dim=-1).mean()
        shp_pts_dist_to_avg_slot = dist_chamfer_recon_ori.mean(dim=-1).mean()

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
        sampled_recon_pts = expanded_recon_slot_points.contiguous().view(
            bz * (self.recon_part_M) * (self.num_slots), -1)[fps_idx, :].contiguous().view(bz, npoints, -1)
        # glb_dist_recon_to_ori: bz x npoints; glb_dist_ori_to_recon: bz x npoints
        glb_dist_recon_to_ori, glb_dist_ori_to_recon = safe_chamfer_dist_call(
            sampled_recon_pts, ori_pts, self.chamfer_dist
        )
        glb_dist_recon_to_ori = glb_dist_recon_to_ori.mean(dim=-1).mean()
        glb_dist_ori_to_recon = glb_dist_ori_to_recon.mean(dim=-1).mean()

        # np.save("recon_slot_pts_multi.npy", recon_slot_points.detach().cpu().numpy())
        # np.save("sampled_recon_pts_multi.npy", sampled_recon_pts.detach().cpu().numpy())
        #
        # np.save("recon_slot_pts_hard_5.npy", recon_slot_points.detach().cpu().numpy())
        # np.save("sampled_recon_pts_hard_5.npy", sampled_recon_pts.detach().cpu().numpy())

        out_feats['recon_slot_pts_hard'] = recon_slot_points.detach().cpu().numpy()
        out_feats['sampled_recon_pts_hard'] = sampled_recon_pts.detach().cpu().numpy()

        np.save(self.log_fn + ".npy", out_feats)
        out_feats['x_features_hard'] = inv_feat.squeeze(-1).detach().cpu().numpy()
        np.save(self.log_fn + "_with_features.npy", out_feats)

        # get tot loss
        ''' If use soft attention and reconstruction '''
        # tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot + purity_loss
        tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        ''' If use hard attention and reconstruction '''
        # tot_loss = glb_dist_recon_to_ori + glb_dist_ori_to_recon #  + avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        # tot_loss = avg_slots_pts_dist_to_shp + shp_pts_dist_to_avg_slot
        return tot_loss, attn


        ''' For a relatively complete pipeline '''
        ori_pts = x.clone()
        bz, npoints = x.size(0), x.size(2)
        # input_x = x # preprocess input and equivariant
        ''' Preprocess input to get input for the network '''
        x = M.preprocess_input(x, self.na_in, pose, False)
        for block_i, block in enumerate(self.backbone):
            x = block(x)
        x_xyz = x.xyz; x_anchors = x.anchors; x_pose = x.pose
        ''' Cluster points via slot attention; clustering; inv_feat '''
        inv_feat, atten_score = self.outblock(x, rlabel)
        # inv_feat: bz x c_out x N x 1
        # get features, invariant features?
        # inv_feat: bz x c_out x N x 1 -> bz x c_out x N
        inv_feat = inv_feat.squeeze(-1) # squeeze invariant features
        ''' Cluster '''
        # rep_slots: bz x num_slots x c_out; attn: bz x N x num_slots
        # slot representation vectors and slot attention vectors
        # invariant features for clustering via slot-attention
        # inv_feat: bz x c_out x N -> bz x N x c_out
        rep_slots, attn_ori = self.slot_attention(inv_feat.contiguous().transpose(1, 2).contiguous())
        attn_ori[:, 0, :npoints // 2] = 1.; attn_ori[:, 0, npoints // 2: ] = 0.
        attn_ori[:, 1, :npoints // 2] = 0.; attn_ori[:, 1, npoints // 2: ] = 1.
        ''' Attention from each point to each cluster '''
        # attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True); bz x N x num_slots
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
        attn_slot = attn_ori / torch.clamp(attn_ori.sum(dim=-1, keepdim=True), min=1e-9)
        attn_ori = attn_ori.contiguous().transpose(1, 2).contiguous()

        ''' Get variant features for each slot '''
        # transformed_variant_feats = self.variant_feat_trans(x.feats) # variant feature transformation
        transformed_variant_feats = x.feats
        ''' Aggregate variant feats for each slot '''
        # todo: is it the most suitable way to aggregate features?
        # transformed_variant_feats: bz x c_out x N x na; attn_ori: bz x num_slots x N
        # variant_feats_slot: bz x c_out x num_slots x N x na -> bz x c_out x num_slots x na
        variant_feats_slot = torch.sum(transformed_variant_feats.unsqueeze(2) * attn_slot.unsqueeze(1).unsqueeze(-1), dim=3)

        ''' Predict \delta_q and translation for each rotation state q '''
        per_slot_transformation = self.transformation_prediction(variant_feats_slot)
        # pred_R: bz x 4 x num_slots x na
        pred_R, pred_T = per_slot_transformation[:, :4, ...], per_slot_transformation[:,4:, ...]
        output['pred_R'], output['pred_T'] = pred_R, pred_T
        R_reg_loss = torch.mean(torch.sqrt((pred_R ** 2).sum(dim=1)) - 1.) # reg R
        ''' Predict points for each part '''
        # slot_points = self.part_reconstruction_net(rep_slots.contiguous().transpose(1, 2).contiguous().unsqueeze(-1)).contiguous().squeeze(-1)
        slot_points = self.part_reconstruction_net(variant_feats_slot.contiguous()).contiguous().squeeze(-1)
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
        transformed_slot_pts = slot_points.unsqueeze(2)
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
        transformed_slot_pts = transformed_slot_pts[:, :2, :, :].contiguous().view(bz, -1, 3)
        # bz x n_recon x 3
        print(torch.mean(torch.mean(transformed_slot_pts, dim=0), dim=0))
        print("ori-shape: ", torch.mean(torch.mean(ori_pts, dim=-1), dim=0))

        ''' Sample points '''
        # transformed_slot_pts = transformed_slot_pts[:, :2, :, :]
        fps_idx = farthest_point_sampling(transformed_slot_pts, self.n_reconstructed)
        downsampled_transformed_pts = transformed_slot_pts.contiguous().view(bz * (self.recon_part_M) * (self.num_slots), -1)[fps_idx, :].contiguous().view(bz, self.n_reconstructed, -1)
        # downsampled_transformed_pts = transformed_slot_pts
        # downsampled_R = selected_R_expanded.contiguous().view(bz * self.recon_part_M * self.num_slots, 3, 3)[fps_idx, :, :].contiguous().view(bz, self.n_reconstructed, 3, 3)
        # downsampled_T = selected_T_expanded.contiguous().view(bz * self.recon_part_M * self.num_slots, 3)[fps_idx, :].contiguous().view(bz, self.n_reconstructed, 3)

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
        recon_loss = (glb_recon_loss + lal_recon_loss).mean() # + 0.1 * R_reg_loss
        # recon_loss = (glb_recon_loss).mean() # + 0.1 * R_reg_loss

        ''' Get down-sampled points and pose '''
        # downsampled
        downsampled_transformed_pts = downsampled_transformed_pts.contiguous().transpose(1, 2).contiguous()
        # downsampled_pose = torch.cat([downsampled_R, downsampled_T.unsqueeze(-1)], dim=-1)
        # downsampled_pose = torch.cat([downsampled_pose, torch.zeros((bz, self.n_reconstructed, 1, 4)).cuda()], dim=-2)

        if recon_loss.item() < 0.05:
            np.save("slot_pts.npy", slot_points.detach().cpu().numpy())
            np.save("downsampled_pts.npy", downsampled_transformed_pts.detach().cpu().numpy())

        return recon_loss, attn

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
        # cur_estimated_pose = pose
        for i in range(nn_inter):
            cur_reconstructed_loss_orbit, attn, = self.forward_one_iter(cur_transformed_points, cur_estimated_pose, rlabel=rlabel)
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
    mlps[-1][-1] = 256 # 512
    out_mlps[-1] = 256 # 512

    mlps = [[64,128], [256], [512], [1024]]
    out_mlps = [1024]

    # initial_radius_ratio = 0.05
    initial_radius_ratio = 0.15
    initial_radius_ratio = 0.20
    device = opt.device
    input_num = opt.model.input_num # 1024
    dropout_rate = opt.model.dropout_rate # default setting: 0.0
    # temperature
    temperature = opt.train_loss.temperature # set temperature
    so3_pooling = 'attention' #  opt.model.flag # model flag
    opt.model.kpconv = 1
    na = 1 if opt.model.kpconv else opt.model.kanchor # how to represent rotation possibilities? --- sampling from the sphere ---- points!
    # na =  opt.model.kanchor  # how to represent rotation possibilities? --- sampling from the sphere ---- points!
    # nmasks = opt.train_lr.nmasks
    nmasks = opt.nmasks
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
        'num_iters': opt.equi_settings.num_iters,
        'global_rot': opt.equi_settings.global_rot,
        'npoints': opt.model.input_num,
        'batch_size': opt.batch_size,
        'init_lr': opt.train_lr.init_lr,
        'part_pred_npoints': opt.equi_settings.part_pred_npoints,
        'use_equi': opt.equi_settings.use_equi,
        'model_type': opt.equi_settings.model_type,
        'decoder_type': opt.equi_settings.decoder_type
    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = ClsSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
