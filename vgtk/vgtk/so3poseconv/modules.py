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

KERNEL_CONDENSE_RATIO_POSE = 0.7


# Basic SO3Conv
# [b, c1, k, p, a] -> [b, c2, p, a]
class BasicSO3ConvPose(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, debug=False):
        super(BasicSO3ConvPose, self).__init__()
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

    def forward(self, x): # b x c1 x k x p2 x a
        bs, np, na = x.shape[0], x.shape[3], x.shape[4]
        x = x.view(bs, self.dim_in*self.kernel_size, np*na)
        x = torch.matmul(self.W, x)

        # x = x + self.bias
        x = x.view(bs, self.dim_out, np, na)
        return x

class KernelPropagationPose(nn.Module):
    def __init__(self, dim_in, dim_out, n_center, kernel_size, radius, sigma, kanchor=60):
        super(KernelPropagationPose, self).__init__()

        # get kernel points (ksx3)
        kernels = L.get_sphereical_kernel_points_from_ply_pose(KERNEL_CONDENSE_RATIO_POSE * radius, kernel_size)

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors_pose(kanchor)
        # if kpconv:
        #     anchors = anchors[29][None]
        kernels = np.transpose(anchors @ kernels.T, (2,0,1))

        self.radius = radius
        self.sigma = sigma
        self.n_center = n_center

        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('kernels', torch.from_numpy(kernels))

        self.basic_conv = BasicSO3ConvPose(dim_in, dim_out, kernels.shape[0])


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

        wts, nnctn = L.initial_anchor_query_pose(frag, centers, self.kernels, self.radius, self.sigma)

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
class InterSO3PoseConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, sigma, n_neighbor,
                 lazy_sample=True, pooling=None, kanchor=60):
        super(InterSO3PoseConv, self).__init__()

        # get kernel points # condense ratio * radius; kernel_size
        kernels = L.get_sphereical_kernel_points_from_ply_pose(KERNEL_CONDENSE_RATIO_POSE * radius, kernel_size)

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors_pose(kanchor)

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

        self.basic_conv = BasicSO3ConvPose(dim_in, dim_out, self.kernel_size)

        self.register_buffer('anchors', torch.from_numpy(anchors)) # anchors
        self.register_buffer('kernels', torch.from_numpy(kernels))

    def forward(self, x, inter_idx=None, inter_w=None):
        inter_idx, inter_w, xyz, feats, sample_idx, sampled_pose = \
            L.inter_so3poseconv_grouping(x.xyz, x.pose, x.feats, self.stride, self.n_neighbor,
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

        return inter_idx, inter_w, sample_idx, SphericalPointCloudPose(xyz, feats, self.anchors, sampled_pose)


class IntraSO3PoseConv(nn.Module):
    '''
    Note: only use intra conv when kanchor=60

    '''
    def __init__(self, dim_in, dim_out):
        super(IntraSO3PoseConv, self).__init__()

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors_pose()
        # get so3 convolution index (precomputed 60x12 indexing)
        intra_idx = L.get_intra_idx_pose()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = intra_idx.shape[1]
        self.basic_conv = BasicSO3ConvPose(dim_in, dim_out, self.kernel_size)
        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('intra_idx', torch.from_numpy(intra_idx).long())

    def forward(self, x):
        feats = L.intra_so3conv_grouping_pose(self.intra_idx, x.feats)
        feats = self.basic_conv(feats)
        return SphericalPointCloudPose(x.xyz, feats, self.anchors, x.pose)


class PointnetSO3PoseConv(nn.Module):
    '''
    equivariant pointnet architecture for a better aggregation of spatial point features
    f (nb, nc, np, na) x xyz (nb, 3, np, na) -> maxpool(h(nb,nc+3,p0,na),h(nb,nc+3,p1,na),h(nb,nc+3,p2,na),...)
    '''
    def __init__(self, dim_in, dim_out, kanchor=60):
        super(PointnetSO3PoseConv, self).__init__()

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors_pose(kanchor)
        self.dim_in = dim_in + 3
        self.dim_out = dim_out

        self.embed = nn.Conv2d(self.dim_in, self.dim_out,1)
        self.register_buffer('anchors', torch.from_numpy(anchors))

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
