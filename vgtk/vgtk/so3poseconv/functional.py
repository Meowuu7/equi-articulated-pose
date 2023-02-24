import math
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm

# from utils_cuda import _neighbor_query, _spherical_conv
import vgtk
import vgtk.pc as pctk
import vgtk.cuda.zpconv as cuda_zpconv
import vgtk.cuda.gathering as gather
import vgtk.cuda.grouping as cuda_nn

import vgtk.spconv as zpconv
from vgtk.spconv import batched_index_select


inter_so3conv_feat_grouping = zpconv.inter_zpconv_grouping_naive
# batched_index_select = zpconv.batched_index_select

# pc: [nb,np,3] -> feature: [nb,1,np,na]
def get_occupancy_features_pose(pc, n_anchor, use_center=False):
    nb, np, nd = pc.shape
    has_normals = nd == 6

    features = torch.zeros(nb, 1, np, n_anchor) + 1
    features = features.float().to(pc.device)

    if has_normals:
        ns = pc[:,:,3:]
        if n_anchor > 1:
            anchors = torch.from_numpy(get_anchors_pose())
            features_n = torch.einsum('bni,aij->bjna',ns.anchors)
        else:
            features_n = ns.transpose(1,2)[...,None].contiguous()
        features = torch.cat([features,features_n],1)

    if use_center:
        features[:,:,0,:] = 0.0

    return features


# (x,y,z) points derived from conic parameterization
def get_kernel_points_np_pose(radius, aperature, kernel_size, multiplier=1):
    assert isinstance(kernel_size, int)
    rrange = np.linspace(0, radius, kernel_size, dtype=np.float32)
    kps = []

    for ridx, ri in enumerate(rrange):
        alpharange = zpconv.get_angular_kernel_points_np(aperature, ridx * multiplier + 1)
        for aidx, alpha in enumerate(alpharange):
            r_r = ri * np.tan(alpha)
            thetarange = np.linspace(0, 2 * np.pi, aidx * 2 + 1, endpoint=False, dtype=np.float32)
            xs = r_r * np.cos(thetarange)
            ys = r_r * np.sin(thetarange)
            zs = np.repeat(ri, aidx * 2 + 1)
            kps.append(np.vstack([xs,ys,zs]).T)

    kps = np.vstack(kps)
    return kps

def get_spherical_kernel_points_np_pose(radius, kernel_size, multiplier=3):
    assert isinstance(kernel_size, int)
    rrange = np.linspace(0, radius, kernel_size, dtype=np.float32)
    kps = []

    for ridx, r_i in enumerate(rrange):
        asize = ridx * multiplier + 1
        bsize = ridx * multiplier + 1
        alpharange = np.linspace(0, 2*np.pi, asize, endpoint=False, dtype=np.float32)
        betarange = np.linspace(0, np.pi, bsize, endpoint=True, dtype=np.float32)

        xs = r_i * np.cos(alpharange[:, None]) * np.sin(betarange[None])
        ys = r_i * np.sin(alpharange[:, None]) * np.sin(betarange[None])

        zs = r_i * np.cos(betarange)[None].repeat(asize, axis=0)
        kps.append(np.vstack([xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)]).T)

    kps = np.vstack(kps)
    return kps

def get_sphereical_kernel_points_from_ply_pose(radius, kernel_size):
    assert kernel_size <= 3 and kernel_size > 0 # x-dim kernel
    mapping = {1:24, 2:30, 3:66}
    root = vgtk.__path__[0]
    anchors_path = os.path.join(root, 'data', 'anchors')
    ply_path = os.path.join(anchors_path, f'kpsphere{mapping[kernel_size]:d}.ply')
    ply = pctk.load_ply(ply_path).astype('float32')
    def normalize(pc, radius):
        r = np.sqrt((pc**2).sum(1).max())
        return pc*radius/r
    return normalize(ply, radius)


# initial_anchor_query(
#     at::Tensor centers, //[b, 3, nc]
#     at::Tensor xyz,  //[m, 3]
#     at::Tensor kernel_points, // [ks, na, 3]
#     const float radius, const float sigma)
def initial_anchor_query_pose(frag, centers, kernels, r, sigma):
    return cuda_nn.initial_anchor_query(centers, frag, kernels, r, sigma)


def inter_so3conv_blurring_pose(xyz, feats, n_neighbor, radius, stride,
                           inter_idx=None, lazy_sample=True, radius_expansion=1.0):
    if inter_idx is None:
        _, inter_idx, sample_idx, sample_xyz = zpconv.inter_zpconv_grouping_ball(xyz, stride, radius * radius_expansion, n_neighbor, lazy_sample)

    if stride == 1:
        return zpconv.inter_blurring_naive(inter_idx, feats), xyz
    else:
        return zpconv.inter_pooling_naive(inter_idx, sample_idx, feats), sample_xyz


def inter_so3poseconv_grouping(xyz, pose, feats, stride, n_neighbor,
                          anchors, kernels, radius, sigma,
                          inter_idx=None, inter_w=None, lazy_sample=True,
                          radius_expansion=1.0, pooling=None):
    '''
        xyz: [nb, 3, p1] coordinates
        feats: [nb, c_in, p1, na] features
        anchors: [na, 3, 3] rotation matrices
        kernels: [ks, 3] kernel points
        inter_idx: [nb, p2, nn] grouped points, where p2 = p1 / stride
        inter_w: [nb, p2, na, ks, nn] kernel weights:
                    Influences of each neighbor points on each kernel points
    '''
    # b = xyz.size(0

    if pooling is not None and stride > 1 and feats.shape[1] > 1:
        # Apply low pass blurring before strided conv
        if pooling == 'stride':
            # NOTE: if meanpool replaces stride, nn and radius needs to be matched with the next conv
            pool_stride = stride
            # TODO: REMOVE HARD CODING
            stride_nn = int(n_neighbor * pool_stride**0.5)
            stride = 1
        elif pooling == 'no-stride':
            pool_stride = 1
            stride_nn = n_neighbor
        else:
            raise NotImplementedError(f"Pooling mode {pooling} is not implemented!")

        feats, xyz = inter_so3conv_blurring_pose(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None

    if inter_idx is None:
        # rel_grouped_pose.size = b x p2 x nn x 3 x 3; anchors.size = ka x 3 x 3
        grouped_xyz, inter_idx, sample_idx, new_xyz, rel_grouped_pose, sampled_pose = zpconv.inter_zpposeconv_grouping_ball(xyz, pose, stride,
                                                                         radius * radius_expansion, n_neighbor, lazy_sample)
        inter_w = inter_so3conv_grouping_anchor_pose(grouped_xyz, anchors, kernels, sigma)

        # dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2))
        # b x p2 x nn x ka
        dists = torch.trace(torch.matmul(anchors.contiguous().transpose(1, 2).contiguous(), rel_grouped_pose[..., None, :, :]) - 1) / 2
        # b x p2 x nn
        nearest_anchor_idx = torch.argmax(dists, dim=-1)
        n_anchors = anchors.size(0)
        b, p2, nn = inter_idx.size()
        # b x p2 x nn x n_anchors
        rotated_anchor_idx = torch.arange(n_anchors).unsqueeze(0).unsqueeze(0).repeat(b, p2, nn, 1)
        # b x p2 x nn x n_anchors (na)
        rotated_anchor_idx = rotated_anchor_idx + nearest_anchor_idx.unsqueeze(-1)
        # feats: [nb, c_in, p1, na]; inter_idx: [nb, p2, npp]
        feats = zpconv.add_shadow_feature(feats)
        # trans_feats: [nb, p1, na, c_in]
        trans_feats = feats.transpose(1, 2).transpose(2, 3)
        # grouped_feats: [nb, p2, npp, na, c_in]
        grouped_feats = batched_index_select(trans_feats, dim=1, index=inter_idx)
        # grouped_feats: [nb, p2, npp, na, c_in]
        grouped_feats = batched_index_select(grouped_feats, dim=3, index=rotated_anchor_idx)
        grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(1, 3).contiguous()
        # maxx
        new_feats = torch.einsum('bcpna,bpakn->bckpa', grouped_feats, inter_w).contiguous()


# the feature dimension needs to be rotated
        #####################DEBUGDEBUGDEBUGDEBUG####################################
        # print(xyz.shape)
        # xyz_sample = (xyz - xyz.mean(2, keepdim=True))[0]
        # gsample1 = xyz_sample[:,inter_idx[0,12].long()]
        # gsample2 = xyz_sample[:,inter_idx[0,25].long()]
        # gsample3 = xyz_sample[:,inter_idx[0,31].long()]
        # pctk.save_ply('data/gsample2.ply', gsample2.T.cpu().numpy(), c='r')
        # pctk.save_ply('data/gsample3.ply', gsample3.T.cpu().numpy(), c='r')
        # pctk.save_ply('data/xyz.ply', xyz_sample.T.cpu().numpy())

        # for bi in range(new_xyz.shape[0]):
        #     pctk.save_ply(f'vis/gsample{bi}.ply', new_xyz[bi].T.cpu().numpy())
        # # import ipdb; ipdb.set_trace()
        #############################################################################
    else:
        # no stride
        sample_idx = None
        new_xyz = xyz
        sampled_pose = pose

        # inter_idx: [nb, p2, nn]
        grouped_pose = batched_index_select(pose, dim=1, index=inter_idx)
        inv_grouped_pose = grouped_pose.transpose(3, 4).contiguous()
        rel_grouped_pose = torch.matmul(pose.unsqueeze(2), inv_grouped_pose)
        dists = torch.trace(
            torch.matmul(anchors.contiguous().transpose(1, 2).contiguous(), rel_grouped_pose[..., None, :, :]) - 1) / 2
        # b x p2 x nn
        nearest_anchor_idx = torch.argmax(dists, dim=-1)
        n_anchors = anchors.size(0)
        b, p2, nn = inter_idx.size()
        # b x p2 x nn x n_anchors
        rotated_anchor_idx = torch.arange(n_anchors).unsqueeze(0).unsqueeze(0).repeat(b, p2, nn, 1)
        # b x p2 x nn x n_anchors (na)
        rotated_anchor_idx = rotated_anchor_idx + nearest_anchor_idx.unsqueeze(-1)
        # feats: [nb, c_in, p1, na]; inter_idx: [nb, p2, npp]
        feats = zpconv.add_shadow_feature(feats)
        # trans_feats: [nb, p1, na, c_in]
        trans_feats = feats.transpose(1, 2).transpose(2, 3)
        # grouped_feats: [nb, p2, npp, na, c_in]
        grouped_feats = batched_index_select(trans_feats, dim=1, index=inter_idx)
        # grouped_feats: [nb, p2, npp, na, c_in]
        grouped_feats = batched_index_select(grouped_feats, dim=3, index=rotated_anchor_idx)
        grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
            1, 3).contiguous()
        # maxx
        new_feats = torch.einsum('bcpna,bpakn->bckpa', grouped_feats, inter_w).contiguous()

        # feats = zpconv.add_shadow_feature(feats)
        #
        # new_feats = inter_so3conv_feat_grouping(inter_idx, inter_w, feats) # [nb, c_in, ks, np, na]

    return inter_idx, inter_w, new_xyz, new_feats, sample_idx, sampled_pose

# inter point grouping
def inter_so3conv_grouping_anchor_pose(grouped_xyz, anchors,
                                  kernels, sigma, interpolate='linear'):
    '''
        grouped_xyz: [b, 3, p2, nn]
        ball_idx: [b, p2, nn]
        anchors: [na, 3, 3]
        sample_idx: [b, p2]
    '''

    # kernel rotations: 3, na, ks; na x 3 x ks; ns x 3 x 3 xxxx 3 x ks --- a kernel is actually R^3 x n_discreted_rotation_group
    rotated_kernels = torch.matmul(anchors, kernels.transpose(0,1)).permute(1,0,2).contiguous()

    # calculate influences: [3, na, ks] x [b, 3, p2, nn] -> [b, p2, na, ks, nn] weights; why kernels? the function of kernels?
    t_rkernels = rotated_kernels[None, :, None, :, :, None] # [1, 3, 1, na, ks, 1]
    t_gxyz = grouped_xyz[...,None,None,:] # kernels --- convolutional c1 and c2? [b, 3, p2, 1, 1, nn]

    if interpolate == 'linear':
        #  b, p2, na, ks, nn
        # distances between grouped xyz and rkernels directly
        dists = torch.sum((t_gxyz - t_rkernels)**2, dim=1)
        # dists = torch.sqrt(torch.sum((t_gxyz - t_rkernels)**2, dim=1))
        inter_w = F.relu(1.0 - dists/sigma, inplace=True) # why this? [b, 3, p2, na, ks, nn] # weights for one point to different kernel points under different rotation states; number rotation states, kernel point numbers;
        # change weights
        # inter_w = F.relu(1.0 - dists / (3*sigma)**0.5, inplace=True)

        # torch.set_printoptions(precision=2, sci_mode=False)
        # print('---sigma----')
        # print(sigma)
        # print('---mean distance---')
        # print(dists.mean())
        # print(dists[0,10,0,6])
        # print('---weights---')
        # print(inter_w[0,10,0,6])
        # print('---summary---')
        # summary = torch.sum(inter_w[0,:,0] > 0.1, -1)
        # print(summary.float().mean(0))
        # import ipdb; ipdb.set_trace()
    else:
        raise NotImplementedError("kernel function %s is not implemented!"%interpolate)

    return inter_w


# intra convolution
def intra_so3conv_grouping_pose(intra_idx, feature):
    '''
        intra_idx: [na,pnn] so3 neighbors
        feature: [nb, c_in, np, na] features # number of batch, number of point, number of rotation group element
    '''

    # group features -> [nb, c_in, pnn, np, na] # nb, c_in, pnn,

    nb, c_in, nq, na = feature.shape
    _, pnn = intra_idx.shape

    feature1 = feature.index_select(3, intra_idx.view(-1)).view(nb, c_in, nq, na, pnn)
    grouped_feat =  feature1.permute([0,1,4,2,3]).contiguous()

    # print(torch.sort(grouped_feat[0,0].mean(0)))
    # print(torch.sort(grouped_feat[1,0].mean(0)))
    # print(torch.sort(grouped_feat[0,0,0]))
    # print(torch.sort(grouped_feat[1,0,0]))
    # print(torch.sort(grouped_feat[0,0,1]))
    # print(torch.sort(grouped_feat[1,0,1]))

    # def find_rel(idx1, idx2):
    #     idx1 = idx1.cpu().numpy().squeeze()
    #     idx2 = idx2.cpu().numpy().squeeze()
    #     idx3 = np.zeros_like(idx1)
    #     for i in range(len(idx1)):
    #         idx3[idx2[i]] = idx1[i]
    #     return idx3

    # def comb_rel(idx1, idx2):
    #     idx1 = idx1.cpu().numpy().squeeze()
    #     idx2 = idx2.cpu().numpy().squeeze()
    #     idx3 = np.zeros_like(idx1)
    #     for i in range(len(idx1)):
    #         idx3[i] = idx1[idx2[i]]
    #     return idx3

    # rel01 = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[0,0,1])[1])
    # rel02 = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[1,0,0])[1])
    # rel13 = find_rel(torch.sort(grouped_feat[0,0,1])[1], torch.sort(grouped_feat[1,0,1])[1])
    # rel23 = find_rel(torch.sort(grouped_feat[1,0,0])[1], torch.sort(grouped_feat[1,0,1])[1])

    # rel_in = find_rel(torch.sort(feature[0,0,0])[1], torch.sort(feature[1,0,0])[1])
    # rel_out = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[1,0,0])[1])

    # import ipdb; ipdb.set_trace()

    return grouped_feat

# initialize so3 sampling
import vgtk.functional as fr
import os

GAMMA_SIZE_pose = 3
ROOT_pose = vgtk.__path__[0]
ANCHOR_PATH_pose = os.path.join(ROOT_pose, 'data', 'anchors/sphere12.ply')

Rs_pose, R_idx_pose, canonical_relative_pose = fr.icosahedron_so3_trimesh(ANCHOR_PATH_pose, GAMMA_SIZE_pose)


def select_anchor_pose(anchors, k):
    if k == 1:
        return anchors[29][None]
    elif k == 20:
        return anchors[::3]
    elif k == 40:
        return anchors.reshape(20,3,3,3)[:,:2].reshape(-1,3,3)
    else:
        return anchors


def get_anchors_pose(k=60):
    return select_anchor_pose(Rs_pose,k)

def get_intra_idx_pose():
    return R_idx_pose

def get_canonical_relative_pose():
    return canonical_relative_pose
