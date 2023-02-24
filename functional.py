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


inter_so3conv_feat_grouping = zpconv.inter_zpconv_grouping_naive
batched_index_select = zpconv.batched_index_select
batched_index_select_other = zpconv.batched_index_select_other

# pc: [nb,np,3] -> feature: [nb,1,np,na]
def get_occupancy_features(pc, n_anchor, use_center=False):
    nb, np, nd = pc.shape
    has_normals = nd == 6

    features = torch.zeros(nb, 1, np, n_anchor) + 1
    features = features.float().to(pc.device)

    if has_normals:
        ns = pc[:,:,3:]
        if n_anchor > 1:
            anchors = torch.from_numpy(get_anchors())
            features_n = torch.einsum('bni,aij->bjna',ns.anchors)
        else:
            features_n = ns.transpose(1,2)[...,None].contiguous()
        features = torch.cat([features,features_n],1)

    if use_center:
        features[:,:,0,:] = 0.0

    return features


# (x,y,z) points derived from conic parameterization
def get_kernel_points_np(radius, aperature, kernel_size, multiplier=1):
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

def get_spherical_kernel_points_np(radius, kernel_size, multiplier=3):
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

def get_sphereical_kernel_points_from_ply(radius, kernel_size):
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
def initial_anchor_query(frag, centers, kernels, r, sigma):
    return cuda_nn.initial_anchor_query(centers, frag, kernels, r, sigma)


def inter_so3conv_blurring(xyz, feats, n_neighbor, radius, stride,
                           inter_idx=None, lazy_sample=True, radius_expansion=1.0):
    if inter_idx is None:
        _, inter_idx, sample_idx, sample_xyz = zpconv.inter_zpconv_grouping_ball(xyz, stride, radius * radius_expansion, n_neighbor, lazy_sample)

    if stride == 1:
        return zpconv.inter_blurring_naive(inter_idx, feats), xyz
    else:
        return zpconv.inter_pooling_naive(inter_idx, sample_idx, feats), sample_xyz


def inter_so3conv_grouping(xyz, feats, stride, n_neighbor,
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

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None

    if inter_idx is None:
        grouped_xyz, inter_idx, sample_idx, new_xyz = zpconv.inter_zpconv_grouping_ball(xyz, stride,
                                                                         radius * radius_expansion, n_neighbor, lazy_sample)
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)


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
        sample_idx = None
        new_xyz = xyz

    feats = zpconv.add_shadow_feature(feats)

    new_feats = inter_so3conv_feat_grouping(inter_idx, inter_w, feats) # [nb, c_in, ks, np, na]

    return inter_idx, inter_w, new_xyz, new_feats, sample_idx


def canonicalize_points(xyz, pose):
    # pose: b x p x 4 x 4, rotation for rigidly moved points
    # xyz: b x p x 3, position information for rigidly moved points
    rotations = pose[:, :, :3, :3]
    translations = pose[:, :, :3, -1]

    # print(rotations.size(), translations.size(), xyz.size())
    cana_xyz = torch.matmul(torch.transpose(rotations, 2, 3), (xyz.contiguous().transpose(1, 2).contiguous() - translations).unsqueeze(-1)).squeeze(-1)
    return cana_xyz.contiguous().transpose(1, 2).contiguous()


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
    # print("stride", stride, pooling)
    if pooling is not None and stride > 1 and feats.shape[1] > 1:
        # Apply low pass blurring before strided conv
        print("balabala... Arrived at an unkown place...")
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

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None
    # print("xyz", xyz.size())
    if inter_idx is None and stride > 1:
        # rel_grouped_pose.size = b x p2 x nn x 3 x 3; anchors.size = ka x 3 x 3
        grouped_xyz, inter_idx, sample_idx, new_xyz, rel_grouped_pose, sampled_pose = zpconv.inter_zpposeconv_grouping_ball(xyz, pose, stride,
                                                                         radius * radius_expansion, n_neighbor, lazy_sample)
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)

        # dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2))
        # b x p2 x nn x ka
        tmp_mtx_rel = torch.matmul(anchors.contiguous().transpose(1, 2).contiguous(), rel_grouped_pose[..., None, :, :]) - 1
        # print("tmp_mtx_rel", tmp_mtx_rel.size())
        def get_rel_mtx_trace(mm):
            trace = mm[...,0,0] + mm[...,1,1] + mm[...,2,2]
            return trace
        dists = get_rel_mtx_trace(tmp_mtx_rel) / 2
        # dists = torch.trace(tmp_mtx_rel) / 2
        # b x p2 x nn
        nearest_anchor_idx = torch.argmax(dists, dim=-1)
        n_anchors = anchors.size(0)
        b, p2, nn = inter_idx.size()
        # b x p2 x nn x n_anchors
        rotated_anchor_idx = torch.arange(n_anchors).unsqueeze(0).unsqueeze(0).repeat(b, p2, nn, 1).cuda()
        # b x p2 x nn x n_anchors (na)
        rotated_anchor_idx = rotated_anchor_idx + nearest_anchor_idx.unsqueeze(-1)
        # na = anchors.size(0)
        rotated_anchor_idx = (rotated_anchor_idx + n_anchors) % n_anchors # get rotated anchor idx
        # feats: [nb, c_in, p1, na]; inter_idx: [nb, p2, npp]
        # print("feats", feats.size())
        # todo: about shadow features --- the functions and how to add them?
        # feats = zpconv.add_shadow_feature(feats)
        # print("added shadow feats", feats.size())
        # trans_feats: [nb, p1, na, c_in]
        trans_feats = feats.transpose(1, 2).transpose(2, 3)
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
        grouped_feats = batched_index_select_other(trans_feats, inter_idx,  dim=1)
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("grouped_feats", grouped_feats.size())
        grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
        # print("rotated feats", grouped_feats.size())
        # todo: change `transpose` to `premute`
        grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(1, 2).contiguous()
        # print("transposed feats", grouped_feats.size())
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

        # group pose
        # grouped_xyz: b x 3 x p x nn; xyz: b x 3 x p
        # we need a canonicalized frame for ball sampling and so on... Poses for points should be aligned

        ''' Previous version: using points not canonicalized '''
        # ball_idx, grouped_xyz = zpconv.ball_query(xyz, xyz, radius, n_neighbor)
        # grouped_xyz = grouped_xyz - xyz.unsqueeze(-1)
        ''' Previous version: using points not canonicalized '''

        ''' Current version: using points canonicalized '''
        cana_xyz = canonicalize_points(xyz, pose)
        ball_idx, grouped_xyz = zpconv.ball_query(cana_xyz, cana_xyz, radius, n_neighbor)
        # bz x 3 x N x nn
        grouped_xyz = grouped_xyz - cana_xyz.unsqueeze(-1)

        # rot: bz x N x 1 x 3 x 3
        nn = grouped_xyz.size(-1)
        rot = pose[:, :, :3, :3].unsqueeze(2).repeat(1, 1, nn, 1, 1)
        # grouped_xyz: bz x N x nn x 3
        grouped_xyz = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        grouped_xyz  = torch.matmul(rot, grouped_xyz.unsqueeze(-1)).squeeze(-1)
        grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # grouped_xyz

        ''' Current version: using points canonicalized '''
        # discretization error?

        ball_idx = ball_idx.long() # ball_idx is actually `inter_idx` in function of the original version
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)
        # Got an orbit of weights

        # print(inter_w.size())
        # equi_feats = []
        # # kernel_weights = torch.arange(0.1, 1.0, 24).cuda()
        # interww = inter_w.contiguous().view(inter_w.size(0), inter_w.size(1), inter_w.size(2), -1).contiguous()
        # interww = interww.contiguous().permute(0, 3, 1, 2).contiguous()
        # for i in range(interww.size(0)):
        #     equi_feats.append(interww[i])
        # equi_feat_a, equi_feat_b = equi_feats[0], equi_feats[1]
        # diffs = []
        # coo = []
        # for j in range(interww.size(2)):
        #     # c_in x na; c_in x na # neighbours 所对应的不同的
        #     a_p_feat, b_p_feat = equi_feat_a[:, j, :], equi_feat_b[:, j, :]
        #     a_p_b_p_feat = torch.sum(torch.sqrt((a_p_feat.unsqueeze(-1) - b_p_feat.unsqueeze(1)) ** 2),
        #                              dim=0)  # na x na
        #     # a_p_b_p_feat_min_dist = torch.min(a_p_b_p_feat).item()
        #     b_min_value, b_min_idx = torch.min(a_p_b_p_feat, dim=1)
        #     a_min_value, a_min_idx = torch.min(b_min_value, dim=0)
        #
        #     a_feat_l2_norm = torch.sqrt(torch.sum(a_p_feat[a_min_idx.item()] ** 2))
        #     diffs.append(a_min_value)
        #     coo.append(float(abs(a_min_idx.item() - b_min_idx[a_min_idx.item()].item())))
        # print("aaa", sum(diffs[64:]) / (64 * 32 * 24), sum(coo) / interww.size(2))


        # get trace
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Groupe pose for relative distance calculation '''
        # # print(pose.size(), ball_idx.size())
        # # grouped_pos: b x p2 x nn x 3 x 3
        # grouped_pose = batched_index_select_other(pose, ball_idx, dim=1, )
        # # grouped_pose = batched_index_select_other(pose, inter_idx, dim=1) # get inverse grouped pose
        # # just by transposing the grouped poses can we get their inversed pose matrices ---- inversed pose matrices
        # inv_grouped_pose = grouped_pose.transpose(3, 4).contiguous()
        # # pos: b x p1 x 3 x 3
        # # then the relative pose matrices indicate the relations between two poses
        # rel_grouped_pose = torch.matmul(pose.unsqueeze(2), inv_grouped_pose)
        #
        # # tmp mtx rel
        # # then the distance between anchors and grouped poses can also be calculated by multiplying the inversed matrices and regular matrices
        # tmp_mtx_rel = torch.matmul(anchors.contiguous().transpose(1, 2).contiguous(),
        #                            rel_grouped_pose[..., None, :, :])  # - 1
        #
        # # print("tmp_mtx_rel", tmp_mtx_rel.size())
        #
        #
        #
        # # then the negative
        # dists = get_rel_mtx_trace(tmp_mtx_rel) / 2  # negative dists, actually
        ''' Groupe pose for relative distance calculation '''


        ''' Group rotation for relative distance calculation '''
        grouped_rotations = batched_index_select_other(pose[:, :, :3, :3], ball_idx, dim=1)
        inv_grouped_rotations = grouped_rotations.transpose(3, 4).contiguous()
        # rel_grouped_rotations: bz x N x nn x 3 x 3;
        rel_grouped_rotations = torch.matmul(pose[:, :, :3, :3].unsqueeze(2), inv_grouped_rotations)


        # # tmp_mtx_rel: bz x N x nn x na x 3 x 3
        # tmp_mtx_rel = torch.matmul(anchors.contiguous().transpose(1, 2).contiguous(),
        #                            rel_grouped_rotations[..., None, :, :])  # - 1
        # dists = get_rel_mtx_trace(tmp_mtx_rel) / 2  # negative dists, actually
        # ''' Group rotation for relative distance calculation '''
        #
        #
        # # b x p2 x nn
        # # and the maximum value indicates the discretized relative pose between center point and neighouring points
        # # but for others? the translation part-level equivariance?
        # nearest_anchor_idx = torch.argmax(dists, dim=-1)
        # n_anchors = anchors.size(0)
        # b, p2, nn = ball_idx.size()
        # # b x p2 x nn x n_anchors
        # rotated_anchor_idx = torch.arange(n_anchors).unsqueeze(0).unsqueeze(0).repeat(b, p2, nn, 1).cuda()
        # # b x p2 x nn x n_anchors (na)
        # rotated_anchor_idx = rotated_anchor_idx + nearest_anchor_idx.unsqueeze(-1)
        # # then we can get related anchor indexes
        # rotated_anchor_idx = (rotated_anchor_idx + n_anchors) % n_anchors # n_anchors # rotated anchor --- how to rotate anchors...



        # bz x N x nn x na x 3 x 3 .contiguous().transpose(-1, -2).contiguous()
        # from context points' rotations to target point's rotation ---- the rotations wishing to use
        rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors) # rotate anchors
        rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4), anchors.unsqueeze(0).contiguous().transpose(2, 3).contiguous())
        dists = get_rel_mtx_trace(rotated_anchors_dists)
        rotated_anchor_idx = torch.argmax(dists, dim=-1)
        # rotated_anchor_idx = rotat

        # feats: [nb, c_in, p1, na]; inter_idx: [nb, p2, npp]
        # todo: the functionality of shadow feature?
        # feats = zpconv.add_shadow_feature(feats)
        # trans_feats: [nb, p1, na, c_in]
        # trans_feats = feats.transpose(1, 2).transpose(2, 3)
        # grouped_feats: [nb, p2, npp, na, c_in]
        # feats: [nb, c_in, p1, na] features
        # trans_feats: [nb, p1, na, c_in]
        # trans_feats = feats.transpose(1, 2).transpose(2, 3)
        trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()

        # grouped_xyz: bz x 3 x N x nn; trans_feats: bz x N x nn x 3
        # trans_feats = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # bz x N x nn x na x 3
        # grouped_feats = torch.matmul(anchors, trans_feats.unsqueeze(3).unsqueeze(-1)).squeeze(-1)

        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
        grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1) # to another shape
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("grouped_feats", grouped_feats.size())

        #### DEBUG ####
        grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
        #### DEBUG ####
        # print("rotated feats", grouped_feats.size())
        # todo: change `transpose` to `premute`
        # grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
            # 1, 2).contiguous()

        #### zzz
        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        # grouped_feats = torch.ones_like(grouped_feats)[:, 0, ...].unsqueeze(1)

        # new feats
        new_feats = torch.einsum('bcpna,bpakn->bckpa', grouped_feats, inter_w).contiguous()
        # print(new_feats.size())

        # feats = zpconv.add_shadow_feature(feats)
        #
        # new_feats = inter_so3conv_feat_grouping(inter_idx, inter_w, feats) # [nb, c_in, ks, np, na]

        # tmp_feat = new_feats.contiguous().permute(0, )
        # equi_feats = []
        # for i in range(new_feats.size(0)):
        #     equi_feats.append(new_feats[i])
        # equi_feat_a, equi_feat_b = equi_feats[0], equi_feats[1]
        # diffs = []
        # coo = []
        # for j in range(new_feats.size(2)):
        #     # c_in x na; c_in x na
        #     a_p_feat, b_p_feat = equi_feat_a[:, j, :], equi_feat_b[:, j, :]
        #     a_p_b_p_feat = torch.sum((a_p_feat.unsqueeze(-1) - b_p_feat.unsqueeze(1)) ** 2, dim=0)  # na x na
        #     # a_p_b_p_feat_min_dist = torch.min(a_p_b_p_feat).item()
        #     b_min_value, b_min_idx = torch.min(a_p_b_p_feat, dim=1)
        #     a_min_value, a_min_idx = torch.min(b_min_value, dim=0)
        #     diffs.append(a_min_value)
        #     coo.append(float(abs(a_min_idx.item() - b_min_idx[a_min_idx.item()].item())))
        # print("zz", sum(diffs), sum(coo) / new_feats.size(2))

    return inter_idx, inter_w, new_xyz, new_feats, sample_idx, sampled_pose

# inter point grouping
def inter_so3conv_grouping_anchor(grouped_xyz, anchors,
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
        inter_w = F.relu(1.0 - dists/sigma, inplace=True) # why this? [b, 3, p2, na, ks, nn] # weights for one point to different kernel points under different rotation states; number rotation states, kernel point numbers; # weights
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
def intra_so3conv_grouping(intra_idx, feature):
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

GAMMA_SIZE = 3
ROOT = vgtk.__path__[0]
ANCHOR_PATH = os.path.join(ROOT, 'data', 'anchors/sphere12.ply')

Rs, R_idx, canonical_relative = fr.icosahedron_so3_trimesh(ANCHOR_PATH, GAMMA_SIZE)


def select_anchor(anchors, k):
    if k == 1:
        return anchors[29][None]
    elif k == 20:
        return anchors[::3]
    elif k == 40:
        return anchors.reshape(20,3,3,3)[:,:2].reshape(-1,3,3)
    else:
        return anchors


def get_anchors(k=60):
    return select_anchor(Rs,k)

def get_intra_idx():
    return R_idx

def get_canonical_relative():
    return canonical_relative
