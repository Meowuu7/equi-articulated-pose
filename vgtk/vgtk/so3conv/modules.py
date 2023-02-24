import math
import os
import numpy as np
import time
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vgtk.spconv import SphericalPointCloud, SphericalPointCloudPose
import vgtk.pc as pctk
from . import functional as L

# BasicSO3Conv = BasicZPConv

KERNEL_CONDENSE_RATIO = 0.7


# Basic SO3Conv
# [b, c1, k, p, a] -> [b, c2, p, a]
class BasicSO3Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, debug=False):
        super(BasicSO3Conv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size

        # TODO: initialization argument
        # TODO: add bias

        if debug:
            W = torch.zeros(self.dim_out, self.dim_in*self.kernel_size) + 1
            self.register_buffer('W', W)
        else:
            W = torch.empty(self.dim_out, self.dim_in, self.kernel_size)
            # nn.init.xavier_normal_(W, gain=0.001)
            nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
            # nn.init.normal_(W, mean=0.0, std=0.3)
            W = W.view(self.dim_out, self.dim_in*self.kernel_size)

            self.register_parameter('W', nn.Parameter(W))
            # bias = torch.zeros(self.dim_out) + 1e-3
            # bias = bias.view(1,self.dim_out,1)
            # self.register_parameter('bias', nn.Parameter(bias))

        #self.W = nn.Parameter(torch.Tensor(self.dim_out, self.dim_in*self.kernel_size))

    def forward(self, x): # b x c1 x k x p2 x a;
        bs, np, na = x.shape[0], x.shape[3], x.shape[4]
        x = x.view(bs, self.dim_in*self.kernel_size, np*na)
        x = torch.matmul(self.W, x)

        # x = x + self.bias
        x = x.view(bs, self.dim_out, np, na)
        return x

class KernelPropagation(nn.Module):
    def __init__(self, dim_in, dim_out, n_center, kernel_size, radius, sigma, kanchor=60):
        super(KernelPropagation, self).__init__()

        # get kernel points (ksx3)
        kernels = L.get_sphereical_kernel_points_from_ply(KERNEL_CONDENSE_RATIO * radius, kernel_size)

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors(kanchor)
        # if kpconv:
        #     anchors = anchors[29][None]
        kernels = np.transpose(anchors @ kernels.T, (2,0,1))

        self.radius = radius
        self.sigma = sigma
        self.n_center = n_center

        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('kernels', torch.from_numpy(kernels))

        self.basic_conv = BasicSO3Conv(dim_in, dim_out, kernels.shape[0])


    def _subsample(self, clouds):
        '''
            furthest point sampling
            [b, 3, n_sub, 3] -> [b, 3, n_center]
        '''
        idx, sample_xyz = pctk.furthest_sample(clouds, self.n_center, False)
        return sample_xyz

    def forward(self, frag, clouds):
        '''
        frag (m,3), center (b, 3, n_center), kernels(ks, na, 3)
        ->
        anchor weight (b, 1, ks, nc, na)

        '''
        if clouds.shape[2] == self.n_center:
            centers = clouds
        else:
            centers = self._subsample(clouds)

        wts, nnctn = L.initial_anchor_query(frag, centers, self.kernels, self.radius, self.sigma)

        # normalization!
        wts = wts / (nnctn + 1.0)

        ###################################
        # torch.set_printoptions(sci_mode=False)
        # print('----------------wts------------------------------')
        # print(wts[0,:,16,0])
        # print('---------------mean---------------------------')
        # print(wts[0].mean(-2))
        # print('---------------std----------------------------')
        # print(wts[0].std(-2))
        # print('-----------------------------------------------')
        # import ipdb; ipdb.set_trace()
        ####################################

        feats = self.basic_conv(wts.unsqueeze(1))

        return SphericalPointCloud(centers, feats, self.anchors)



# A single Inter SO3Conv
# [b, c1, p1, a] -> [b, c1, k, p2, a] -> [b, c2, p2, a]; bsz x c1 x p1 x a -> kernel and sampled points
class InterSO3Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, sigma, n_neighbor,
                 lazy_sample=True, pooling=None, kanchor=60):
        super(InterSO3Conv, self).__init__()

        # get kernel points # condense ratio * radius; kernel_size
        kernels = L.get_sphereical_kernel_points_from_ply(KERNEL_CONDENSE_RATIO * radius, kernel_size)

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors(kanchor)

        # # debug only
        # if kanchor == 1:
        #     anchors = anchors[29][None]

        # register hyperparameters
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernels.shape[0]
        self.stride = stride
        self.radius = radius
        self.sigma = sigma
        self.n_neighbor = n_neighbor
        self.lazy_sample = lazy_sample
        self.pooling = pooling

        self.basic_conv = BasicSO3Conv(dim_in, dim_out, self.kernel_size)

        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('kernels', torch.from_numpy(kernels))

    def forward(self, x, inter_idx=None, inter_w=None):
        inter_idx, inter_w, xyz, feats, sample_idx = \
            L.inter_so3conv_grouping(x.xyz, x.feats, self.stride, self.n_neighbor,
                                  self.anchors, self.kernels,
                                  self.radius, self.sigma,
                                  inter_idx, inter_w, self.lazy_sample, pooling=self.pooling)


        # torch.set_printoptions(sci_mode=False)
        # print(feats[0,0,:,16])
        # print("-----------mean -----------------")
        # print(feats[0].mean(-2))
        # print("-----------std -----------------")
        # print(feats[0].std(-2))
        # import ipdb; ipdb.set_trace()
        feats = self.basic_conv(feats)

        return inter_idx, inter_w, sample_idx, SphericalPointCloud(xyz, feats, self.anchors)

# SO3
class InterSO3PoseConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, sigma, n_neighbor, # sampling neighbours, weight normalization term and number of neighbours
                 lazy_sample=True, pooling=None, kanchor=60, permute_modes=0, use_2d=False, use_art_mode=False):
        super(InterSO3PoseConv, self).__init__()

        # get kernel points # condense ratio * radius; kernel_size
        kernels = L.get_sphereical_kernel_points_from_ply(KERNEL_CONDENSE_RATIO * radius, kernel_size)
        # if not os.path.exists("kernels.npy"):
        #     np.save("kernels.npy", kernels)
        # get so3 anchors (60x3x3 rotation matrices)
        # anchors
        anchors = L.get_anchors(kanchor)

        # # debug only
        # if kanchor == 1:
        #     anchors = anchors[29][None]

        # register hyperparameters
        # dim_in = 3
        # self.dim_in = 3 # dim_in

        # if dim_in == 1:
        #     dim_in = 3
        #     self.dim_in = 3

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernels.shape[0]
        self.stride = stride
        self.radius = radius
        self.sigma = sigma
        self.n_neighbor = n_neighbor
        self.lazy_sample = lazy_sample
        self.pooling = pooling
        self.permute_modes = permute_modes
        self.use_2d = use_2d
        self.use_art_mode = use_art_mode

        # basic conv ---
        self.basic_conv = BasicSO3Conv(dim_in, dim_out, self.kernel_size)

        self.register_buffer('anchors', torch.from_numpy(anchors)) # anchors
        self.register_buffer('kernels', torch.from_numpy(kernels))

    def forward(self, x, inter_idx=None, inter_w=None, seg=None):

        # Get sampled index, weights from kernels to points, new xyz and aggregated features
        # If stride is not set to 1, then sampled idx and sampled pose are new coordinate and pose information

        # inter_idx, inter_w, xyz, feats, sample_idx, sampled_pose = \
        #     L.inter_so3poseconv_grouping(x.xyz, x.pose, x.feats, self.stride, self.n_neighbor,
        #                           self.anchors, self.kernels,
        #                           self.radius, self.sigma,
        #                           inter_idx, inter_w, self.lazy_sample, pooling=self.pooling)

        #### pose with strided ####
        # if not self.use_2d:
        #     inter_idx, inter_w, xyz, feats, sample_idx, sampled_pose = \
        #         L.inter_so3poseconv_grouping_strided(x.xyz, x.pose, x.feats, self.stride, self.n_neighbor,
        #                                      self.anchors, self.kernels,
        #                                      self.radius, self.sigma,
        #                                      inter_idx, inter_w, self.lazy_sample, pooling=self.pooling, permute_modes=self.permute_modes)
        # else:
        #     inter_idx, inter_w, xyz, feats, sample_idx, sampled_pose = \
        #         L.inter_so3poseconv_grouping_strided_2D(x.xyz, x.pose, x.feats, self.stride, self.n_neighbor,
        #                                              self.anchors, self.kernels,
        #                                              self.radius, self.sigma,
        #                                              inter_idx, inter_w, self.lazy_sample, pooling=self.pooling,
        #                                              permute_modes=self.permute_modes)
        #### pose with strided ####

        if self.use_2d:
            inter_idx, inter_w, xyz, feats, sample_idx, sampled_pose = \
                L.inter_so3poseconv_grouping_strided_2D(x.xyz, x.pose, x.feats, self.stride, self.n_neighbor,
                                                        self.anchors, self.kernels,
                                                        self.radius, self.sigma,
                                                        inter_idx, inter_w, self.lazy_sample, pooling=self.pooling,
                                                        permute_modes=self.permute_modes)
        elif self.use_art_mode:
            inter_idx, inter_w, xyz, feats, sample_idx, sampled_pose = \
                L.inter_so3poseconv_grouping_strided_arti_mode(x.xyz, x.pose, x.feats, self.stride, self.n_neighbor,
                                                        self.anchors, self.kernels,
                                                        self.radius, self.sigma,
                                                        inter_idx=inter_idx, inter_w=inter_w, lazy_sample=self.lazy_sample, seg_labels=seg, pooling=self.pooling,
                                                        permute_modes=self.permute_modes)
        else: # regular strieded so(3) convolution
            inter_idx, inter_w, xyz, feats, sample_idx, sampled_pose = \
                L.inter_so3poseconv_grouping_strided(x.xyz, x.pose, x.feats, self.stride, self.n_neighbor,
                                                     self.anchors, self.kernels,
                                                     self.radius, self.sigma,
                                                     inter_idx, inter_w, self.lazy_sample, pooling=self.pooling,
                                                     permute_modes=self.permute_modes)

        # inter_idx, inter_w, xyz, feats, sample_idx, sampled_pose = \
        #     L.inter_so3poseconv_grouping_seg(x.xyz, x.pose, x.feats, self.stride, self.n_neighbor,
        #                                  self.anchors, self.kernels,
        #                                  self.radius, self.sigma,
        #                                  inter_idx, inter_w, self.lazy_sample, pooling=self.pooling, seg=seg)

        # group features
        # inter_idx, inter_w, xyz, feats, sample_idx = \
        #     L.inter_so3conv_grouping(x.xyz, x.feats, self.stride, self.n_neighbor,
        #                              self.anchors, self.kernels,
        #                              self.radius, self.sigma,
        #                              inter_idx, inter_w, self.lazy_sample, pooling=self.pooling)
        # sampled_pose = x.pose


        # torch.set_printoptions(sci_mode=False)
        # print(feats[0,0,:,16])
        # print("-----------mean -----------------")
        # print(feats[0].mean(-2))
        # print("-----------std -----------------")
        # print(feats[0].std(-2))
        # import ipdb; ipdb.set_trace()
        feats = self.basic_conv(feats)

        # equi_feats = []
        # for i in range(feats.size(0)):
        #     equi_feats.append(feats[i])
        # equi_feat_a, equi_feat_b = equi_feats[0], equi_feats[1]
        # diffs = []
        # coo = []
        # a_b_rot_simss = []
        # for j in range(feats.size(2)):
        #     # c_in x na; c_in x na
        #     a_p_feat, b_p_feat = equi_feat_a[:, j, :], equi_feat_b[:, j, :]
        #     a_rot, b_rot = x.pose[0, j, :3, :3], x.pose[1, j, :3, :3]
        #     a_b_rot_sim = torch.matmul(a_rot, b_rot.contiguous().transpose(0, 1))
        #     a_b_rot_sim = a_b_rot_sim[0, 0] + a_b_rot_sim[1, 1] + a_b_rot_sim[2, 2]
        #     a_b_rot_sim = (a_b_rot_sim - 1.) / 2
        #     a_b_rot_simss.append(a_b_rot_sim)
        #     a_p_b_p_feat = torch.sum(torch.sqrt((a_p_feat.unsqueeze(-1) - b_p_feat.unsqueeze(1)) ** 2), dim=0)  # na x na
        #     # a_p_b_p_feat_min_dist = torch.min(a_p_b_p_feat).item()
        #     b_min_value, b_min_idx = torch.min(a_p_b_p_feat, dim=1)
        #     a_min_value, a_min_idx = torch.min(b_min_value, dim=0)
        #
        #     a_feat_l2_norm = torch.sqrt(torch.sum(a_p_feat[a_min_idx.item()] ** 2))
        #     diffs.append(a_min_value / (a_feat_l2_norm + 1e-8))
        #     coo.append(float(abs(a_min_idx.item() - b_min_idx[a_min_idx.item()].item())))
        # print("zz", sum(diffs), sum(coo) / feats.size(2), sum(a_b_rot_simss) / feats.size(2))

        # print(f"here xyz: {xyz.size()}, feats: {feats.size()}, sampled_pose: {sampled_pose.size()}")

        return inter_idx, inter_w, sample_idx, SphericalPointCloudPose(xyz, feats, self.anchors, sampled_pose)


class IntraSO3Conv(nn.Module):
    '''
    Note: only use intra conv when kanchor=60
    '''
    def __init__(self, dim_in, dim_out):
        super(IntraSO3Conv, self).__init__()

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors()
        # get so3 convolution index (precomputed 60x12 indexing)
        intra_idx = L.get_intra_idx()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = intra_idx.shape[1]
        self.basic_conv = BasicSO3Conv(dim_in, dim_out, self.kernel_size)
        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('intra_idx', torch.from_numpy(intra_idx).long())

    def forward(self, x):
        feats = L.intra_so3conv_grouping(self.intra_idx, x.feats)
        feats = self.basic_conv(feats)
        return SphericalPointCloud(x.xyz, feats, self.anchors)


class IntraSO3Conv2D(nn.Module):
    '''
    Note: only use intra conv when kanchor=60
    '''
    def __init__(self, dim_in, dim_out):
        super(IntraSO3Conv2D, self).__init__()

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors()
        # get so3 convolution index (precomputed 60x12 indexing)
        intra_idx = L.get_intra_idx()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = intra_idx.shape[1]
        self.basic_conv = BasicSO3Conv(dim_in, dim_out, self.kernel_size)
        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('intra_idx', torch.from_numpy(intra_idx).long())

    def forward(self, x):
        feats = L.intra_so3conv_grouping_2D(self.intra_idx, x.feats)
        # The last dimension is na * na2
        feats = self.basic_conv(feats)
        return SphericalPointCloud(x.xyz, feats, self.anchors)


class PointnetSO3Conv(nn.Module):
    '''
    equivariant pointnet architecture for a better aggregation of spatial point features
    f (nb, nc, np, na) x xyz (nb, 3, np, na) -> maxpool(h(nb,nc+3,p0,na),h(nb,nc+3,p1,na),h(nb,nc+3,p2,na),...)
    '''
    def __init__(self, dim_in, dim_out, kanchor=60, return_raw=False):
        super(PointnetSO3Conv, self).__init__()

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors(kanchor)
        self.dim_in = dim_in + 3
        self.dim_out = dim_out
        self.return_raw = return_raw

        self.embed = nn.Conv2d(self.dim_in, self.dim_out,1)
        self.register_buffer('anchors', torch.from_numpy(anchors))

    def forward(self, x, ):
        xyz = x.xyz
        feats = x.feats
        nb, nc, np, na = feats.shape

        # normalize xyz
        xyz = xyz - xyz.mean(2,keepdim=True)

        if na == 1:
            feats = torch.cat([x.feats, xyz[...,None]],1)
        else:
            # let relative xyz involved in the convolution process
            xyzr = torch.einsum('aji,bjn->bina',self.anchors,xyz)
            feats = torch.cat([x.feats, xyzr],1)

        feats = self.embed(feats)
        if self.return_raw:
            return feats
        else:
            feats = torch.max(feats,2)[0]
            return feats # nb, nc, na


class PointnetSO3PoseConv(nn.Module):
    '''
    equivariant pointnet architecture for a better aggregation of spatial point features
    f (nb, nc, np, na) x xyz (nb, 3, np, na) -> maxpool(h(nb,nc+3,p0,na),h(nb,nc+3,p1,na),h(nb,nc+3,p2,na),...)
    '''
    def __init__(self, dim_in, dim_out, kanchor=60):
        super(PointnetSO3PoseConv, self).__init__()

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors(kanchor)
        self.dim_in = dim_in + 3
        self.dim_out = dim_out

        self.embed = nn.Conv2d(self.dim_in, self.dim_out,1)
        self.register_buffer('anchors', torch.from_numpy(anchors)) # get anchors

    def forward(self, x):
        xyz = x.xyz
        feats = x.feats
        nb, nc, np, na = feats.shape

        # normalize xyz
        xyz = xyz - xyz.mean(2,keepdim=True)

        if na == 1:
            feats = torch.cat([x.feats, xyz[...,None]],1)
        else:
            # [na, 3, 3] x [b x n x 3] -> [b x 3 x n x a]
            xyzr = torch.einsum('aji,bjn->bina',self.anchors,xyz)
            feats = torch.cat([x.feats, xyzr],1)

        feats = self.embed(feats)
        feats = torch.max(feats,2)[0]
        return feats # nb, nc, na
