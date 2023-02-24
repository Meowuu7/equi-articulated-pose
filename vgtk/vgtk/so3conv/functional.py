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

# [cos_, 0, sin_
#  0, 1., 0.
#  -sin_, 0., cos_]
# rot_matrices: 4 x 3 x 3
def get_2D_res_anchors():
    # angles = [_ * (np.pi / 4.) for _ in range(4)]
    angles = [_ * (np.pi / 2.) for _ in range(4)]
    rot_matrices = []
    for i, theta in enumerate(angles):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        n_x, n_y, n_z = 0.0, 1.0, 0.0
        cur_matrix = np.array(
            [[cos_theta, 0., sin_theta],
             [0., 1., 0.],
             [-sin_theta, 0., cos_theta]], dtype=np.float
        )
        rot_matrices.append(torch.from_numpy(cur_matrix).float().unsqueeze(0))
    rot_matrices = torch.cat(rot_matrices, dim=0) # .cuda()
    return rot_matrices

RES_ROT_2D = get_2D_res_anchors()


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
        xyz: [nb, 3, p1] coordinates #
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


def inter_so3poseconv_grouping_bak(xyz, pose, feats, stride, n_neighbor,
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
    # Then need to select inter_idx for convolution
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
        # how to select points not in the canonicalized frame?
        # transfer to canonical?
        # get real equivariant displacement vectors
        # canonicalize points
        cana_xyz = canonicalize_points(xyz, pose)
        # Query neighbours
        # ball_dix: bz x N x nn; grouped_xyz: bz x 3 x N x nn

        ball_idx, grouped_xyz = zpconv.ball_query(cana_xyz, cana_xyz, radius, n_neighbor)
        # counts: bz x N
        # Then we want to get unique number of neighbours
        # _, counts = torch.unique(ball_idx, dim=-1, return_counts=True)
        # avg_unique_counts = torch.mean(counts.float()).item()
        # print(f"ball_idx.size: {ball_idx.size()}, sampled counts: {counts[0, :32]}, avg_unique_counts: {avg_unique_counts}, ball neighbours sample: {ball_idx[0, 0, :]}")

        ''' Current version: using points not canonicalized '''

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''

        # bz x 3 x N x nn
        # group
        grouped_xyz = grouped_xyz - cana_xyz.unsqueeze(-1)
        #
        # # rot: bz x N x 1 x 3 x 3
        # nn = grouped_xyz.size(-1) # grouped_xyz #
        # rot = pose[:, :, :3, :3].unsqueeze(2).repeat(1, 1, nn, 1, 1)
        # # grouped_xyz: bz x N x nn x 3
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # grouped_xyz  = torch.matmul(rot, grouped_xyz.unsqueeze(-1)).squeeze(-1)
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
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

        # get matrix's trace
        # get trace
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Group pose for relative distance calculation '''
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
        # print(f"For grouped_rotations: pose.size: {pose.size()}, ball_idx: {ball_idx.size()}")
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

        ''' Set initial features to relative positional offset '''
        # if feats.size(1) == 1:
        #     feats = grouped_xyz.clone()
        ''' Set initial features to relative positional offset '''
        trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()

        # grouped_xyz: bz x 3 x N x nn; trans_feats: bz x N x nn x 3
        # trans_feats = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # bz x N x nn x na x 3

        # for each part pose...
        # grouped_feats = torch.matmul(anchors, trans_feats.unsqueeze(3).unsqueeze(-1)).squeeze(-1)

        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
        # print(f"trans_feats: {trans_feats.size()}, ball_idx: {ball_idx.size()}")
        grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1) # to another shape
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("grouped_feats", grouped_feats.size())



        #### DEBUG ####
        # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
        grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
        #### DEBUG ####
        # print("rotated feats", grouped_feats.size())
        # todo: change `transpose` to `premute`
        # grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
            # 1, 2).contiguous()

        #### zzz
        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        ''' Set initial features to relative positional offset '''
        # grouped_feats = torch.ones_like(grouped_feats)[:, 0, ...].unsqueeze(1)
        if grouped_feats.size(1) == 1:
            grouped_feats = grouped_xyz.unsqueeze(-1)
        ''' Set initial features to relative positional offset '''

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
            stride_nn = int(n_neighbor * pool_stride ** 0.5)
            stride = 1
        elif pooling == 'no-stride':
            pool_stride = 1
            stride_nn = n_neighbor
        else:
            raise NotImplementedError(f"Pooling mode {pooling} is not implemented!")

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None
    # print("xyz", xyz.size())
    # Then need to select inter_idx for convolution
    if inter_idx is None and stride > 1: # inter
        # rel_grouped_pose.size = b x p2 x nn x 3 x 3; anchors.size = ka x 3 x 3
        grouped_xyz, inter_idx, sample_idx, new_xyz, rel_grouped_pose, sampled_pose = zpconv.inter_zpposeconv_grouping_ball(
            xyz, pose, stride,
            radius * radius_expansion, n_neighbor, lazy_sample)
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)

        # dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2))
        # b x p2 x nn x ka
        tmp_mtx_rel = torch.matmul(anchors.contiguous().transpose(1, 2).contiguous(),
                                   rel_grouped_pose[..., None, :, :]) - 1

        # print("tmp_mtx_rel", tmp_mtx_rel.size())
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
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
        rotated_anchor_idx = (rotated_anchor_idx + n_anchors) % n_anchors  # get rotated anchor idx
        # feats: [nb, c_in, p1, na]; inter_idx: [nb, p2, npp]
        # print("feats", feats.size())
        # todo: about shadow features --- the functions and how to add them?
        # feats = zpconv.add_shadow_feature(feats)
        # print("added shadow feats", feats.size())
        # trans_feats: [nb, p1, na, c_in]
        trans_feats = feats.transpose(1, 2).transpose(2, 3)
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
        grouped_feats = batched_index_select_other(trans_feats, inter_idx, dim=1)
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("grouped_feats", grouped_feats.size())
        grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
        # print("rotated feats", grouped_feats.size())
        # todo: change `transpose` to `premute`
        grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
            1, 2).contiguous()
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

        ''' Previous version v2: using points canonicalized '''
        # # how to select points not in the canonicalized frame?
        # # transfer to canonical?
        # # get real equivariant displacement vectors
        # # canonicalize points
        # cana_xyz = canonicalize_points(xyz, pose)
        # # Query neighbours
        # # ball_dix: bz x N x nn; grouped_xyz: bz x 3 x N x nn
        #
        # ball_idx, grouped_xyz = zpconv.ball_query(cana_xyz, cana_xyz, radius, n_neighbor)
        # # bz x 3 x N x nn
        # # group
        # grouped_xyz = grouped_xyz - cana_xyz.unsqueeze(-1)
        # # counts: bz x N
        # # Then we want to get unique number of neighbours
        # # _, counts = torch.unique(ball_idx, dim=-1, return_counts=True)
        # # avg_unique_counts = torch.mean(counts.float()).item()
        # # print(f"ball_idx.size: {ball_idx.size()}, sampled counts: {counts[0, :32]}, avg_unique_counts: {avg_unique_counts}, ball neighbours sample: {ball_idx[0, 0, :]}")

        ''' Current version: using points not canonicalized '''

        ball_idx, grouped_xyz = zpconv.ball_query(xyz, xyz, radius, n_neighbor)
        ball_idx = ball_idx.long()  # ball_idx is actually `inter_idx` in function of the original version
        ''' Group rotation for relative distance calculation '''
        # print(f"For grouped_rotations: pose.size: {pose.size()}, ball_idx: {ball_idx.size()}")
        grouped_rotations = batched_index_select_other(pose[:, :, :3, :3], ball_idx, dim=1)
        inv_grouped_rotations = grouped_rotations.transpose(3, 4).contiguous()
        # rel_grouped_rotations: bz x N x nn x 3 x 3;
        rel_grouped_rotations = torch.matmul(pose[:, :, :3, :3].unsqueeze(2), inv_grouped_rotations)
        # print(rel_grouped_rotations.size(), grouped_xyz.size())
        grouped_xyz = grouped_xyz - xyz.unsqueeze(-1)
        #### For Debug ####
        # print(rel_grouped_rotations[0, 0, 1])
        grouped_xyz = torch.matmul(rel_grouped_rotations, grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous().unsqueeze(-1)).squeeze(-1)
        grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        #### For Debug ####

        # grouped_xyz = grouped_xyz -

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''


        #
        # # rot: bz x N x 1 x 3 x 3
        # nn = grouped_xyz.size(-1) # grouped_xyz #
        # rot = pose[:, :, :3, :3].unsqueeze(2).repeat(1, 1, nn, 1, 1)
        # # grouped_xyz: bz x N x nn x 3
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # grouped_xyz  = torch.matmul(rot, grouped_xyz.unsqueeze(-1)).squeeze(-1)
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # grouped_xyz

        ''' Current version: using points canonicalized '''
        # discretization error?

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

        # get matrix's trace
        # get trace
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Group pose for relative distance calculation '''
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

        ''' Get rotated anchors '''
        rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors)  # rotate anchors
        rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4),
                                             anchors.unsqueeze(0).contiguous().transpose(2, 3).contiguous())
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

        ''' Set initial features to relative positional offset '''
        # if feats.size(1) == 1:
        #     feats = grouped_xyz.clone()
        ''' Set initial features to relative positional offset '''
        feats = zpconv.add_shadow_feature(feats)
        trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()

        # grouped_xyz: bz x 3 x N x nn; trans_feats: bz x N x nn x 3
        # trans_feats = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # bz x N x nn x na x 3

        # for each part pose...
        # grouped_feats = torch.matmul(anchors, trans_feats.unsqueeze(3).unsqueeze(-1)).squeeze(-1)

        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
        # print(f"trans_feats: {trans_feats.size()}, ball_idx: {ball_idx.size()}")
        grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("grouped_feats", grouped_feats.size())

        #### DEBUG ####
        # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
        grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
        #### DEBUG ####
        # print("rotated feats", grouped_feats.size())
        # todo: change `transpose` to `premute`
        # grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
        # 1, 2).contiguous()

        #### zzz
        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        ''' Set initial features to relative positional offset '''
        # grouped_feats = torch.ones_like(grouped_feats)[:, 0, ...].unsqueeze(1)
        # if grouped_feats.size(1) == 1:
        #     grouped_feats = grouped_xyz.unsqueeze(-1)
        ''' Set initial features to relative positional offset '''

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

def inter_so3poseconv_grouping_strided(xyz, pose, feats, stride, n_neighbor,
                               anchors, kernels, radius, sigma,
                               inter_idx=None, inter_w=None, lazy_sample=True,
                               radius_expansion=1.0, pooling=None, permute_modes=0):
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
        raise ValueError('xyz_pooling is not None?!!')
        if pooling == 'stride':
            # NOTE: if meanpool replaces stride, nn and radius needs to be matched with the next conv
            pool_stride = stride
            # TODO: REMOVE HARD CODING
            stride_nn = int(n_neighbor * pool_stride ** 0.5)
            stride = 1
        elif pooling == 'no-stride':
            pool_stride = 1
            stride_nn = n_neighbor
        else:
            raise NotImplementedError(f"Pooling mode {pooling} is not implemented!")

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None
    # print("xyz", xyz.size())
    # Then need to select inter_idx for convolution
    if inter_idx is None and stride > 1: # inter
        # rel_grouped_pose.size = b x p2 x nn x 3 x 3; anchors.size = ka x 3 x 3
        grouped_xyz, inter_idx, sample_idx, new_xyz, grouped_pose, sampled_pose = zpconv.inter_zpposeconv_grouping_ball(
            xyz, pose, stride, radius * radius_expansion, n_neighbor, lazy_sample)

        # grouped_pose: bz x p2 x nn x 4 x 4
        # sampled_pose: bz x p2 x 4 x 4
        # rel_grouped_rots: bz x p2 x nn x 3 x 3
        # compute relative rotations from neighbouring pts to the center pt
        rel_grouped_rots = torch.matmul(sampled_pose[..., :3, :3].unsqueeze(2), grouped_pose[..., :3, :3].contiguous().transpose(-1, -2).contiguous())
        # rel_grouped_rots: bz x p2 x nn x 3 x 3 @ bz x p2 x nn x 3 x 1
        grouped_xyz = torch.matmul(rel_grouped_rots,
                                   grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous().unsqueeze(-1)).squeeze(-1)
        grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # inter_w: bz x p2 x
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''

        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Get rotated anchors '''
        # whether it is correct?
        ### how to rotate anchors here ? ###
        ### strategy 1 ###
        # rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors)  # rotate anchors
        ### strategy 2 ###
        if permute_modes == 0:
            feats = zpconv.add_shadow_feature(feats)
            # feats: bz x dim x p1 x na
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()
            grouped_feats = batched_index_select_other(trans_feats, inter_idx.long(), dim=1)  # to another shape
        else:
            rotated_anchors = torch.matmul(rel_grouped_rots.contiguous().transpose(-1, -2).contiguous().unsqueeze(3), anchors)  # rotate anchors
            rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4),
                                                 anchors.unsqueeze(0).contiguous().transpose(2, 3).contiguous())
            dists = get_rel_mtx_trace(rotated_anchors_dists)
            # traces can reveal the angle between two rotation matrices
            rotated_anchor_idx = torch.argmax(dists, dim=-1)

            ''' Set initial features to relative positional offset '''
            # if feats.size(1) == 1:
            #     feats = grouped_xyz.clone()
            ''' Set initial features to relative positional offset '''
            feats = zpconv.add_shadow_feature(feats)

            # feats: bz x dim x p1 x na
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()

            grouped_feats = batched_index_select_other(trans_feats, inter_idx.long(), dim=1)  # to another shape

            #### DEBUG ####
            # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
            grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
            #### DEBUG ####

        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        # new feats
        new_feats = torch.einsum('bcpna,bpakn->bckpa', grouped_feats, inter_w).contiguous()

        #### Just for the convinience of following iterations since we have changed some logics... ####
        inter_idx = None
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

        ''' Previous version v2: using points canonicalized '''
        # # how to select points not in the canonicalized frame?
        # # transfer to canonical?
        # # get real equivariant displacement vectors
        # # canonicalize points
        # cana_xyz = canonicalize_points(xyz, pose)
        # # Query neighbours
        # # ball_dix: bz x N x nn; grouped_xyz: bz x 3 x N x nn
        #
        # ball_idx, grouped_xyz = zpconv.ball_query(cana_xyz, cana_xyz, radius, n_neighbor)
        # # bz x 3 x N x nn
        # # group
        # grouped_xyz = grouped_xyz - cana_xyz.unsqueeze(-1)
        # # counts: bz x N
        # # Then we want to get unique number of neighbours
        # # _, counts = torch.unique(ball_idx, dim=-1, return_counts=True)
        # # avg_unique_counts = torch.mean(counts.float()).item()
        # # print(f"ball_idx.size: {ball_idx.size()}, sampled counts: {counts[0, :32]}, avg_unique_counts: {avg_unique_counts}, ball neighbours sample: {ball_idx[0, 0, :]}")

        ''' Current version: using points not canonicalized '''

        ball_idx, grouped_xyz = zpconv.ball_query(xyz, xyz, radius, n_neighbor)
        ball_idx = ball_idx.long()  # ball_idx is actually `inter_idx` in function of the original version
        ''' Group rotation for relative distance calculation '''
        # print(f"For grouped_rotations: pose.size: {pose.size()}, ball_idx: {ball_idx.size()}")
        grouped_rotations = batched_index_select_other(pose[:, :, :3, :3], ball_idx, dim=1)
        inv_grouped_rotations = grouped_rotations.transpose(3, 4).contiguous()
        # rel_grouped_rotations: bz x N x nn x 3 x 3;
        # ivn grouped rotations? --- the inv rotations performed by neighbouring points
        # inv_grouped_rotations: bz x N x nn x 3 x 3
        rel_grouped_rotations = torch.matmul(pose[:, :, :3, :3].unsqueeze(2), inv_grouped_rotations)
        # print(rel_grouped_rotations.size(), grouped_xyz.size())
        grouped_xyz = grouped_xyz - xyz.unsqueeze(-1)
        #### For Debug ####
        # print(rel_grouped_rotations[0, 0, 1])
        # rotate each neighbour
        # grouped_xyz: bz x N x nn x 3
        grouped_xyz = torch.matmul(rel_grouped_rotations, grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous().unsqueeze(-1)).squeeze(-1)
        grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        #### For Debug ####

        # grouped_xyz = grouped_xyz -

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''

        #
        # # rot: bz x N x 1 x 3 x 3
        # nn = grouped_xyz.size(-1) # grouped_xyz #
        # rot = pose[:, :, :3, :3].unsqueeze(2).repeat(1, 1, nn, 1, 1)
        # # grouped_xyz: bz x N x nn x 3
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # grouped_xyz  = torch.matmul(rot, grouped_xyz.unsqueeze(-1)).squeeze(-1)
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # grouped_xyz

        ''' Current version: using points canonicalized '''
        # discretization error?

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

        # get matrix's trace
        # get trace
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Group pose for relative distance calculation '''
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

        ''' Get rotated anchors '''
        # whether it is correct?
        ### how to rotate anchors here ? ###
        ### strategy 1 ###
        # rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors)  # rotate anchors
        ### strategy 2 ###
        rotated_anchors = torch.matmul(rel_grouped_rotations.contiguous().transpose(-1, -2).contiguous().unsqueeze(3), anchors)  # rotate anchors
        rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4),
                                             anchors.unsqueeze(0).contiguous().transpose(2, 3).contiguous())
        dists = get_rel_mtx_trace(rotated_anchors_dists)
        # traces can reveal the angle between two rotation matrices
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

        ''' Set initial features to relative positional offset '''
        # if feats.size(1) == 1:
        #     feats = grouped_xyz.clone()
        ''' Set initial features to relative positional offset '''
        feats = zpconv.add_shadow_feature(feats)

        if permute_modes == 0:
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()
            grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
        else:
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()
            # grouped_xyz: bz x 3 x N x nn; trans_feats: bz x N x nn x 3
            # trans_feats = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
            # bz x N x nn x na x 3

            # for each part pose...
            # grouped_feats = torch.matmul(anchors, trans_feats.unsqueeze(3).unsqueeze(-1)).squeeze(-1)

            # grouped_feats: [nb, p2, npp, na, c_in]
            # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
            # print(f"trans_feats: {trans_feats.size()}, ball_idx: {ball_idx.size()}")
            grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
            # grouped_feats: [nb, p2, npp, na, c_in]
            # print("grouped_feats", grouped_feats.size())

            #### DEBUG ####
            # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
            grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
            #### DEBUG ####
            # print("rotated feats", grouped_feats.size())
            # todo: change `transpose` to `premute`
            # grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
            # 1, 2).contiguous()

        #### zzz
        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        ''' Set initial features to relative positional offset '''
        # grouped_feats = torch.ones_like(grouped_feats)[:, 0, ...].unsqueeze(1)
        # if grouped_feats.size(1) == 1:
        #     grouped_feats = grouped_xyz.unsqueeze(-1)
        ''' Set initial features to relative positional offset '''

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

# coordinates in different modes
def inter_so3poseconv_grouping_strided_arti_mode(xyz, pose, feats, stride, n_neighbor,
                               anchors, kernels, radius, sigma, seg_labels=None,
                               inter_idx=None, inter_w=None, lazy_sample=True,
                               radius_expansion=1.0, pooling=None, permute_modes=0):
    '''
        xyz: [nb, 3, p1] coordinates; xyz: nb x ns x 3 x p1
        feats: [nb, c_in, p1, na] features; still single mode feats, no concept of modes
        anchors: [na, 3, 3] rotation matrices; single mode rotation anchors
        kernels: [ks, 3] kernel points; kernel points
        inter_idx: [nb, p2, nn] grouped points, where p2 = p1 / stride; inter grouped points
        inter_w: [nb, p2, na, ks, nn] kernel weights:
                    Influences of each neighbor points on each kernel points
        seg_labels: bz x p1 --> segmentation label of each point
    '''
    # b = xyz.size(0
    # print("stride", stride, pooling)
    # strided conv with spatial pooling
    if pooling is not None and stride > 1 and feats.shape[1] > 1:
        # Apply low pass blurring before strided conv
        print("balabala... Arrived at an unkown place...")
        raise ValueError('xyz_pooling is not None?!!')
        if pooling == 'stride':
            # NOTE: if meanpool replaces stride, nn and radius needs to be matched with the next conv
            pool_stride = stride
            # TODO: REMOVE HARD CODING
            stride_nn = int(n_neighbor * pool_stride ** 0.5)
            stride = 1
        elif pooling == 'no-stride':
            pool_stride = 1
            stride_nn = n_neighbor
        else:
            raise NotImplementedError(f"Pooling mode {pooling} is not implemented!")

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None
    # print("xyz", xyz.size())
    # Then need to select inter_idx for convolution
    # strided conv with spatial pooling
    if inter_idx is None and stride > 1: # inter
        # rel_grouped_pose.size = b x p2 x nn x 3 x 3; anchors.size = ka x 3 x 3
        grouped_xyz, inter_idx, sample_idx, new_xyz, grouped_pose, sampled_pose = zpconv.inter_zpposeconv_grouping_ball(
            xyz, pose, stride, radius * radius_expansion, n_neighbor, lazy_sample)

        # grouped_pose: bz x p2 x nn x 4 x 4
        # sampled_pose: bz x p2 x 4 x 4
        # rel_grouped_rots: bz x p2 x nn x 3 x 3
        # compute relative rotations from neighbouring pts to the center pt
        rel_grouped_rots = torch.matmul(sampled_pose[..., :3, :3].unsqueeze(2), grouped_pose[..., :3, :3].contiguous().transpose(-1, -2).contiguous())
        # rel_grouped_rots: bz x p2 x nn x 3 x 3 @ bz x p2 x nn x 3 x 1
        grouped_xyz = torch.matmul(rel_grouped_rots,
                                   grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous().unsqueeze(-1)).squeeze(-1)
        grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # inter_w: bz x p2 x
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''

        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Get rotated anchors '''
        # whether it is correct?
        ### how to rotate anchors here ? ###
        ### strategy 1 ###
        # rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors)  # rotate anchors
        ### strategy 2 ###
        if permute_modes == 0:
            feats = zpconv.add_shadow_feature(feats)
            # feats: bz x dim x p1 x na
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()
            grouped_feats = batched_index_select_other(trans_feats, inter_idx.long(), dim=1)  # to another shape
        else:
            rotated_anchors = torch.matmul(rel_grouped_rots.contiguous().transpose(-1, -2).contiguous().unsqueeze(3), anchors)  # rotate anchors
            rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4),
                                                 anchors.unsqueeze(0).contiguous().transpose(2, 3).contiguous())
            dists = get_rel_mtx_trace(rotated_anchors_dists)
            # traces can reveal the angle between two rotation matrices
            rotated_anchor_idx = torch.argmax(dists, dim=-1)

            ''' Set initial features to relative positional offset '''
            # if feats.size(1) == 1:
            #     feats = grouped_xyz.clone()
            ''' Set initial features to relative positional offset '''
            feats = zpconv.add_shadow_feature(feats)

            # feats: bz x dim x p1 x na
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()

            grouped_feats = batched_index_select_other(trans_feats, inter_idx.long(), dim=1)  # to another shape

            #### DEBUG ####
            # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
            grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
            #### DEBUG ####

        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        # new feats
        new_feats = torch.einsum('bcpna,bpakn->bckpa', grouped_feats, inter_w).contiguous()

        #### Just for the convinience of following iterations since we have changed some logics... ####
        inter_idx = None
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
        # no stride and no pooling
        # no stride
        sample_idx = None
        new_xyz = xyz # set the coordinates to input coordinates with articulation modes
        sampled_pose = pose

        # group pose
        # grouped_xyz: b x 3 x p x nn; xyz: b x 3 x p
        # we need a canonicalized frame for ball sampling and so on... Poses for points should be aligned

        ''' Previous version: using points not canonicalized '''
        # ball_idx, grouped_xyz = zpconv.ball_query(xyz, xyz, radius, n_neighbor)
        # grouped_xyz = grouped_xyz - xyz.unsqueeze(-1)
        ''' Previous version: using points not canonicalized '''

        ''' Previous version v2: using points canonicalized '''
        # # how to select points not in the canonicalized frame?
        # # transfer to canonical?
        # # get real equivariant displacement vectors
        # # canonicalize points
        # cana_xyz = canonicalize_points(xyz, pose)
        # # Query neighbours
        # # ball_dix: bz x N x nn; grouped_xyz: bz x 3 x N x nn
        #
        # ball_idx, grouped_xyz = zpconv.ball_query(cana_xyz, cana_xyz, radius, n_neighbor)
        # # bz x 3 x N x nn
        # # group
        # grouped_xyz = grouped_xyz - cana_xyz.unsqueeze(-1)
        # # counts: bz x N
        # # Then we want to get unique number of neighbours
        # # _, counts = torch.unique(ball_idx, dim=-1, return_counts=True)
        # # avg_unique_counts = torch.mean(counts.float()).item()
        # # print(f"ball_idx.size: {ball_idx.size()}, sampled counts: {counts[0, :32]}, avg_unique_counts: {avg_unique_counts}, ball neighbours sample: {ball_idx[0, 0, :]}")

        ''' Current version: using points not canonicalized '''

        bz, ns, np = xyz.size(0), xyz.size(1), xyz.size(-1)
        # xyz: bz x ns x 3 x np
        # for each point, select the neighbour ; you can have ball_idx and grouped_xyz for each point
        #
        expanded_art_mode_xyz = xyz.contiguous().view(xyz.size(0) * xyz.size(1), xyz.size(2), xyz.size(-1)).contiguous()
        # expanded_art_ball_idx: (bz * ns) x np x k;
        # expanded_art_grouped_xyz: (bz * ns) x 3 x np x k
        expanded_art_ball_idx, expanded_art_grouped_xyz = zpconv.ball_query(expanded_art_mode_xyz, expanded_art_mode_xyz, radius, n_neighbor)
        expanded_art_ball_idx = expanded_art_ball_idx.long()
        # expanded_art_grouped_xyz: (bz x ns) x 3 x np x k
        expanded_art_grouped_xyz = expanded_art_grouped_xyz - expanded_art_mode_xyz.unsqueeze(-1)
        #
        expanded_art_grouped_xyz = expanded_art_grouped_xyz.contiguous().view(bz, ns, 3, np, n_neighbor).contiguous()
        expanded_art_grouped_xyz = expanded_art_grouped_xyz.contiguous().transpose(2, 3).contiguous().transpose(1, 2).contiguous()
        # grouped_xyz: bz x np x 3 x k --> bz x 3 x np x k
        grouped_xyz = batched_index_select_other(expanded_art_grouped_xyz, seg_labels.unsqueeze(2), dim=2).squeeze(2).contiguous().transpose(1, 2).contiguous()

        expanded_art_ball_idx = expanded_art_ball_idx.contiguous().view(bz, ns, np, n_neighbor).contiguous().transpose(1, 2).contiguous()
        # print(f"expanded_art_ball_idx: {expanded_art_ball_idx.size()}, seg_labels: {seg_labels.size()}")
        # ball_idx: bz x np x k
        ball_idx = batched_index_select_other(expanded_art_ball_idx, seg_labels.unsqueeze(2), dim=2).squeeze(2)

        ''' Get ball_idx '''
        # ball_idx, grouped_xyz = zpconv.ball_query(xyz, xyz, radius, n_neighbor)
        # ball_idx = ball_idx.long()  # ball_idx is actually `inter_idx` in function of the original version
        # ''' Group rotation for relative distance calculation '''
        # # print(f"For grouped_rotations: pose.size: {pose.size()}, ball_idx: {ball_idx.size()}")
        # print(f"pose: {pose.size()}, ball_idx: {ball_idx.size()}")
        grouped_rotations = batched_index_select_other(pose[:, :, :3, :3], ball_idx, dim=1)
        inv_grouped_rotations = grouped_rotations.transpose(3, 4).contiguous()
        # # rel_grouped_rotations: bz x N x nn x 3 x 3;
        # # ivn grouped rotations? --- the inv rotations performed by neighbouring points
        # # inv_grouped_rotations: bz x N x nn x 3 x 3
        rel_grouped_rotations = torch.matmul(pose[:, :, :3, :3].unsqueeze(2), inv_grouped_rotations)
        # # print(rel_grouped_rotations.size(), grouped_xyz.size())
        # grouped_xyz = grouped_xyz - xyz.unsqueeze(-1)
        # #### For Debug ####
        # # print(rel_grouped_rotations[0, 0, 1])
        # # rotate each neighbour
        # # grouped_xyz: bz x N x nn x 3
        # grouped_xyz = torch.matmul(rel_grouped_rotations, grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous().unsqueeze(-1)).squeeze(-1)
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()

        #### For Debug ####

        # grouped_xyz = grouped_xyz -

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''

        #
        # # rot: bz x N x 1 x 3 x 3
        # nn = grouped_xyz.size(-1) # grouped_xyz #
        # rot = pose[:, :, :3, :3].unsqueeze(2).repeat(1, 1, nn, 1, 1)
        # # grouped_xyz: bz x N x nn x 3
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # grouped_xyz  = torch.matmul(rot, grouped_xyz.unsqueeze(-1)).squeeze(-1)
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # grouped_xyz

        ''' Current version: using points canonicalized '''
        # discretization error?

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

        # get matrix's trace
        # get trace
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Group pose for relative distance calculation '''
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

        ''' Get rotated anchors '''
        # whether it is correct?
        ### how to rotate anchors here ? ###
        ### strategy 1 ###
        # rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors)  # rotate anchors
        ### strategy 2 ###
        rotated_anchors = torch.matmul(rel_grouped_rotations.contiguous().transpose(-1, -2).contiguous().unsqueeze(3), anchors)  # rotate anchors
        rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4),
                                             anchors.unsqueeze(0).contiguous().transpose(2, 3).contiguous())
        dists = get_rel_mtx_trace(rotated_anchors_dists)
        # traces can reveal the angle between two rotation matrices
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

        ''' Set initial features to relative positional offset '''
        # if feats.size(1) == 1:
        #     feats = grouped_xyz.clone()
        ''' Set initial features to relative positional offset '''
        feats = zpconv.add_shadow_feature(feats)

        if permute_modes == 0:
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()
            grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
        else:

            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()
            # grouped_xyz: bz x 3 x N x nn; trans_feats: bz x N x nn x 3
            # trans_feats = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
            # bz x N x nn x na x 3

            # for each part pose...
            # grouped_feats = torch.matmul(anchors, trans_feats.unsqueeze(3).unsqueeze(-1)).squeeze(-1)

            # grouped_feats: [nb, p2, npp, na, c_in]
            # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
            # print(f"trans_feats: {trans_feats.size()}, ball_idx: {ball_idx.size()}")
            # get neighbours's features
            # print(f"trans_feats: {trans_feats.size()}, ball_idx: {ball_idx.size()}")
            grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
            # grouped_feats: [nb, p2, npp, na, c_in]
            # print("grouped_feats", grouped_feats.size())

            #### DEBUG ####
            # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
            # permute features
            # print(f"grouped_feats: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
            grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
            #### DEBUG ####
            # print("rotated feats", grouped_feats.size())
            # todo: change `transpose` to `premute`
            # grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
            # 1, 2).contiguous()

        #### zzz #### permute grouped features to the desired index order
        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        # print(f"new grouped_feats: {grouped_feats.size()}")

        ''' Set initial features to relative positional offset '''
        # grouped_feats = torch.ones_like(grouped_feats)[:, 0, ...].unsqueeze(1)
        # if grouped_feats.size(1) == 1:
        #     grouped_feats = grouped_xyz.unsqueeze(-1)
        ''' Set initial features to relative positional offset '''

        # new feats; new features xxx
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


def inter_so3poseconv_grouping_strided_2D(xyz, pose, feats, stride, n_neighbor,
                               anchors, kernels, radius, sigma,
                               inter_idx=None, inter_w=None, lazy_sample=True,
                               radius_expansion=1.0, pooling=None, permute_modes=0):
    '''
        xyz: [nb, 3, p1] coordinates
        feats: [nb, c_in, p1, na] features  # na -> we can make it to na x 4
        anchors: [na, 3, 3] rotation matrices # anchors -> na x 4 x 3 x 3; set hyper-parameter to determine whether to use 2D convolution...
        kernels: [ks, 3] kernel points # others... just kernels....
        inter_idx: [nb, p2, nn] grouped points, where p2 = p1 / stride
        inter_w: [nb, p2, na, ks, nn] kernel weights:
                    Influences of each neighbor points on each kernel points
    '''
    # b = xyz.size(0
    # print("stride", stride, pooling)
    # strided convolution; we can skip this kind of convolution here
    if pooling is not None and stride > 1 and feats.shape[1] > 1:
        # Apply low pass blurring before strided conv
        print("balabala... Arrived at an unkown place...")
        raise ValueError('xyz_pooling is not None?!!')
        if pooling == 'stride':
            # NOTE: if meanpool replaces stride, nn and radius needs to be matched with the next conv
            pool_stride = stride
            # TODO: REMOVE HARD CODING
            stride_nn = int(n_neighbor * pool_stride ** 0.5)
            stride = 1
        elif pooling == 'no-stride':
            pool_stride = 1
            stride_nn = n_neighbor
        else:
            raise NotImplementedError(f"Pooling mode {pooling} is not implemented!")

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None
    # print("xyz", xyz.size())
    # Then need to select inter_idx for convolution
    if inter_idx is None and stride > 1: # inter
        # rel_grouped_pose.size = b x p2 x nn x 3 x 3; anchors.size = ka x 3 x 3
        grouped_xyz, inter_idx, sample_idx, new_xyz, grouped_pose, sampled_pose = zpconv.inter_zpposeconv_grouping_ball(
            xyz, pose, stride, radius * radius_expansion, n_neighbor, lazy_sample)

        # grouped_pose: bz x p2 x nn x 4 x 4
        # sampled_pose: bz x p2 x 4 x 4
        # rel_grouped_rots: bz x p2 x nn x 3 x 3
        # compute relative rotations from neighbouring pts to the center pt
        rel_grouped_rots = torch.matmul(sampled_pose[..., :3, :3].unsqueeze(2), grouped_pose[..., :3, :3].contiguous().transpose(-1, -2).contiguous())
        # rel_grouped_rots: bz x p2 x nn x 3 x 3 @ bz x p2 x nn x 3 x 1
        grouped_xyz = torch.matmul(rel_grouped_rots,
                                   grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous().unsqueeze(-1)).squeeze(-1)
        grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # inter_w: bz x p2 x
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''

        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Get rotated anchors '''
        # whether it is correct?
        ### how to rotate anchors here ? ###
        ### strategy 1 ###
        # rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors)  # rotate anchors
        ### strategy 2 ###
        if permute_modes == 0:
            feats = zpconv.add_shadow_feature(feats)
            # feats: bz x dim x p1 x na
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()
            grouped_feats = batched_index_select_other(trans_feats, inter_idx.long(), dim=1)  # to another shape
        else:
            rotated_anchors = torch.matmul(rel_grouped_rots.contiguous().transpose(-1, -2).contiguous().unsqueeze(3), anchors)  # rotate anchors
            rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4),
                                                 anchors.unsqueeze(0).contiguous().transpose(2, 3).contiguous())
            dists = get_rel_mtx_trace(rotated_anchors_dists)
            # traces can reveal the angle between two rotation matrices
            rotated_anchor_idx = torch.argmax(dists, dim=-1)

            ''' Set initial features to relative positional offset '''
            # if feats.size(1) == 1:
            #     feats = grouped_xyz.clone()
            ''' Set initial features to relative positional offset '''
            feats = zpconv.add_shadow_feature(feats)

            # feats: bz x dim x p1 x na
            trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()

            grouped_feats = batched_index_select_other(trans_feats, inter_idx.long(), dim=1)  # to another shape

            #### DEBUG ####
            # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
            grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
            #### DEBUG ####

        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        # new feats
        new_feats = torch.einsum('bcpna,bpakn->bckpa', grouped_feats, inter_w).contiguous()

        #### Just for the convinience of following iterations since we have changed some logics... ####
        inter_idx = None
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

        bz, c_in, p1 = feats.size(0), feats.size(1), feats.size(2)
        feats = feats.contiguous().view(bz, c_in, p1, anchors.size(0), 4).contiguous()

        # group pose
        # grouped_xyz: b x 3 x p x nn; xyz: b x 3 x p
        # we need a canonicalized frame for ball sampling and so on... Poses for points should be aligned

        ''' Previous version: using points not canonicalized '''
        # ball_idx, grouped_xyz = zpconv.ball_query(xyz, xyz, radius, n_neighbor)
        # grouped_xyz = grouped_xyz - xyz.unsqueeze(-1)
        ''' Previous version: using points not canonicalized '''

        ''' Previous version v2: using points canonicalized '''
        # # how to select points not in the canonicalized frame?
        # # transfer to canonical?
        # # get real equivariant displacement vectors
        # # canonicalize points
        # cana_xyz = canonicalize_points(xyz, pose)
        # # Query neighbours
        # # ball_dix: bz x N x nn; grouped_xyz: bz x 3 x N x nn
        #
        # ball_idx, grouped_xyz = zpconv.ball_query(cana_xyz, cana_xyz, radius, n_neighbor)
        # # bz x 3 x N x nn
        # # group
        # grouped_xyz = grouped_xyz - cana_xyz.unsqueeze(-1)
        # # counts: bz x N
        # # Then we want to get unique number of neighbours
        # # _, counts = torch.unique(ball_idx, dim=-1, return_counts=True)
        # # avg_unique_counts = torch.mean(counts.float()).item()
        # # print(f"ball_idx.size: {ball_idx.size()}, sampled counts: {counts[0, :32]}, avg_unique_counts: {avg_unique_counts}, ball neighbours sample: {ball_idx[0, 0, :]}")

        ''' Current version: using points not canonicalized '''

        ball_idx, grouped_xyz = zpconv.ball_query(xyz, xyz, radius, n_neighbor)
        ball_idx = ball_idx.long()  # ball_idx is actually `inter_idx` in function of the original version
        ''' Group rotation for relative distance calculation '''
        # print(f"For grouped_rotations: pose.size: {pose.size()}, ball_idx: {ball_idx.size()}")
        grouped_rotations = batched_index_select_other(pose[:, :, :3, :3], ball_idx, dim=1)
        inv_grouped_rotations = grouped_rotations.transpose(3, 4).contiguous()
        # rel_grouped_rotations: bz x N x nn x 3 x 3;
        # ivn grouped rotations? --- the inv rotations performed by neighbouring points
        # inv_grouped_rotations: bz x N x nn x 3 x 3
        rel_grouped_rotations = torch.matmul(pose[:, :, :3, :3].unsqueeze(2), inv_grouped_rotations)
        # print(rel_grouped_rotations.size(), grouped_xyz.size())
        grouped_xyz = grouped_xyz - xyz.unsqueeze(-1)
        #### For Debug ####
        # print(rel_grouped_rotations[0, 0, 1])
        # rotate each neighbour
        # grouped_xyz: bz x N x nn x 3
        grouped_xyz = torch.matmul(rel_grouped_rotations, grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous().unsqueeze(-1)).squeeze(-1)
        grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        #### For Debug ####

        # grouped_xyz = grouped_xyz -

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''
        #
        # # rot: bz x N x 1 x 3 x 3
        # nn = grouped_xyz.size(-1) # grouped_xyz #
        # rot = pose[:, :, :3, :3].unsqueeze(2).repeat(1, 1, nn, 1, 1)
        # # grouped_xyz: bz x N x nn x 3
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # grouped_xyz  = torch.matmul(rot, grouped_xyz.unsqueeze(-1)).squeeze(-1)
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # grouped_xyz

        ''' Current version: using points canonicalized '''
        # discretization error?

        tot_anchors = torch.matmul(anchors.unsqueeze(1), RES_ROT_2D.cuda().unsqueeze(0))
        tot_anchors = tot_anchors.contiguous().view(-1, 3, 3).contiguous()

        # inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, tot_anchors, kernels, sigma)

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

        # get matrix's trace
        # get trace
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Group pose for relative distance calculation '''
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
        ''' Get rotated anchors '''
        # whether it is correct?
        ### how to rotate anchors here ? ###
        ### strategy 1 ###
        # rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors)  # rotate anchors
        ### strategy 2 ###
        # RES_ROT_2D: 4 x 3 x 3; rel_grouped_rotations: bz x N x nn x 4 x 3 x 3
        rotated_anchors = torch.matmul(rel_grouped_rotations.contiguous().transpose(-1, -2).contiguous().unsqueeze(3), RES_ROT_2D.cuda())  # rotate anchors
        rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4),
                                             RES_ROT_2D.cuda().unsqueeze(0).contiguous().transpose(2, 3).contiguous())
        dists = get_rel_mtx_trace(rotated_anchors_dists)
        # traces can reveal the angle between two rotation matrices
        # rotated_anchor_idx: bz x N x nn x 4
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

        ''' Set initial features to relative positional offset '''
        # if feats.size(1) == 1:
        #     feats = grouped_xyz.clone()
        ''' Set initial features to relative positional offset '''
        # feats: [nb, c_in, p1, na]
        nb, c_in, p1, na, na2 = feats.shape
        feats = zpconv.add_shadow_feature(feats.contiguous().view(nb, c_in, p1, na * na2).contiguous())
        feats = feats.contiguous().view(nb, c_in, p1 + 1, na, na2).contiguous()

        if permute_modes == 0:
            trans_feats = feats.contiguous().permute(0, 2, 3, 4, 1).contiguous()
            grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
        else:
            trans_feats = feats.contiguous().permute(0, 2, 3, 4, 1).contiguous()
            # grouped_xyz: bz x 3 x N x nn; trans_feats: bz x N x nn x 3
            # trans_feats = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
            # bz x N x nn x na x 3

            # for each part pose...
            # grouped_feats = torch.matmul(anchors, trans_feats.unsqueeze(3).unsqueeze(-1)).squeeze(-1)

            # grouped_feats: [nb, p2, npp, na, c_in]
            # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
            # print(f"trans_feats: {trans_feats.size()}, ball_idx: {ball_idx.size()}")
            # grouped_feats: bz x p1 x k x na x na2 x c_in
            grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
            # grouped_feats: [nb, p2, npp, na, c_in]
            # print("grouped_feats", grouped_feats.size())

            #### DEBUG ####
            # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
            # print(grouped_feats.size(), rotated_anchor_idx.size())
            # rotated_anchor_idx: bz x N x nn x 4 -> bz x N x nn x 1 x 4; rotated anchor
            # print(rotated_anchor_idx[0, 0, 0, ])

            rotated_anchor_idx = rotated_anchor_idx.unsqueeze(3).contiguous().repeat(1, 1, 1, grouped_feats.size(3), 1).contiguous()
            grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=4)
            #### DEBUG ####
            # print("rotated feats", grouped_feats.size())
            # todo: change `transpose` to `premute`
            # grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
            # 1, 2).contiguous()

        #### zzz; grouped_feats: bz x c x n x k x na x na2
        grouped_feats = grouped_feats.contiguous().permute(0, 5, 1, 2, 3, 4).contiguous()

        ''' Set initial features to relative positional offset '''
        # grouped_feats = torch.ones_like(grouped_feats)[:, 0, ...].unsqueeze(1)
        # if grouped_feats.size(1) == 1:
        #     grouped_feats = grouped_xyz.unsqueeze(-1)
        ''' Set initial features to relative positional offset '''


        # grouped_feats: bz x c x n x k x na x na2;
        # inter_w: bz x n x (na x na2) x k_kernel x k
        inter_w = inter_w.contiguous().view(bz, inter_w.size(1), grouped_feats.size(-2), grouped_feats.size(-1), inter_w.size(-2), inter_w.size(-1)).contiguous()
        # new_feats: bz x c x n x k x na x na2
        new_feats = torch.einsum('bcpnaz,bpazkn->bckpaz', grouped_feats, inter_w).contiguous()

        # new_feats: bz x c x n x k x (na * na2)
        new_feats = new_feats.contiguous().view(new_feats.size(0), new_feats.size(1), new_feats.size(2), new_feats.size(3), -1).contiguous()

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


# inter grouping seg
def inter_so3poseconv_grouping_seg(xyz, pose, feats, stride, n_neighbor,
                               anchors, kernels, radius, sigma,
                               inter_idx=None, inter_w=None, lazy_sample=True,
                               radius_expansion=1.0, pooling=None, seg=None):
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
    # Should not be None for segmentation based grouping
    assert seg is not None
    ''' If using larger stride '''
    if pooling is not None and stride > 1 and feats.shape[1] > 1:
        # Apply low pass blurring before strided conv
        print("balabala... Arrived at an unkown place...")
        if pooling == 'stride':
            # NOTE: if meanpool replaces stride, nn and radius needs to be matched with the next conv
            pool_stride = stride
            # TODO: REMOVE HARD CODING
            stride_nn = int(n_neighbor * pool_stride ** 0.5)
            stride = 1
        elif pooling == 'no-stride':
            pool_stride = 1
            stride_nn = n_neighbor
        else:
            raise NotImplementedError(f"Pooling mode {pooling} is not implemented!")

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None
    # print("xyz", xyz.size())
    # Then need to select inter_idx for convolution
    if inter_idx is None and stride > 1:
        # rel_grouped_pose.size = b x p2 x nn x 3 x 3; anchors.size = ka x 3 x 3
        grouped_xyz, inter_idx, sample_idx, new_xyz, rel_grouped_pose, sampled_pose = zpconv.inter_zpposeconv_grouping_ball(
            xyz, pose, stride,
            radius * radius_expansion, n_neighbor, lazy_sample)
        inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)

        # dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2))
        # b x p2 x nn x ka
        tmp_mtx_rel = torch.matmul(anchors.contiguous().transpose(1, 2).contiguous(),
                                   rel_grouped_pose[..., None, :, :]) - 1

        # print("tmp_mtx_rel", tmp_mtx_rel.size())
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
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
        rotated_anchor_idx = (rotated_anchor_idx + n_anchors) % n_anchors  # get rotated anchor idx
        # feats: [nb, c_in, p1, na]; inter_idx: [nb, p2, npp]
        # print("feats", feats.size())
        # todo: about shadow features --- the functions and how to add them?
        # feats = zpconv.add_shadow_feature(feats)
        # print("added shadow feats", feats.size())
        # trans_feats: [nb, p1, na, c_in]
        trans_feats = feats.transpose(1, 2).transpose(2, 3)
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
        grouped_feats = batched_index_select_other(trans_feats, inter_idx, dim=1)
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("grouped_feats", grouped_feats.size())
        grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
        # print("rotated feats", grouped_feats.size())
        # todo: change `transpose` to `premute`
        grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
            1, 2).contiguous()
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

        ''' Previous version v2: using points canonicalized '''
        # # how to select points not in the canonicalized frame?
        # # transfer to canonical?
        # # get real equivariant displacement vectors
        # # canonicalize points
        # cana_xyz = canonicalize_points(xyz, pose)
        # # Query neighbours
        # # ball_dix: bz x N x nn; grouped_xyz: bz x 3 x N x nn
        #
        # ball_idx, grouped_xyz = zpconv.ball_query(cana_xyz, cana_xyz, radius, n_neighbor)
        # # bz x 3 x N x nn
        # # group
        # grouped_xyz = grouped_xyz - cana_xyz.unsqueeze(-1)
        # # counts: bz x N
        # # Then we want to get unique number of neighbours
        # # _, counts = torch.unique(ball_idx, dim=-1, return_counts=True)
        # # avg_unique_counts = torch.mean(counts.float()).item()
        # # print(f"ball_idx.size: {ball_idx.size()}, sampled counts: {counts[0, :32]}, avg_unique_counts: {avg_unique_counts}, ball neighbours sample: {ball_idx[0, 0, :]}")

        ''' Current version: using points not canonicalized '''

        ball_idx, grouped_xyz = zpconv.ball_query(xyz, xyz, radius, n_neighbor)
        ball_idx = ball_idx.long()  # ball_idx is actually `inter_idx` in function of the original version
        ''' Group rotation for relative distance calculation '''
        # print(f"For grouped_rotations: pose.size: {pose.size()}, ball_idx: {ball_idx.size()}")
        # grouped_rotations = batched_index_select_other(pose[:, :, :3, :3], ball_idx, dim=1)
        # inv_grouped_rotations = grouped_rotations.transpose(3, 4).contiguous()
        # rel_grouped_rotations: bz x N x nn x 3 x 3;
        # rel_grouped_rotations = torch.matmul(pose[:, :, :3, :3].unsqueeze(2), inv_grouped_rotations)
        # print(rel_grouped_rotations.size(), grouped_xyz.size())
        grouped_xyz = grouped_xyz - xyz.unsqueeze(-1)
        # seg: bz x N, which indicates each point's label
        # ball_idx: bz x N x nn ---> grouped_segs: bz x N x nn
        grouped_segs = batched_index_select_other(seg, ball_idx, dim=1)
        #### For Debug ####
        # print(rel_grouped_rotations[0, 0, 1])
        # grouped_xyz = torch.matmul(rel_grouped_rotations, grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous().unsqueeze(-1)).squeeze(-1)
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        #### For Debug ####

        # grouped_xyz = grouped_xyz -

        ''' Get average number of valid neighbours '''
        # print(f"n_neighbours: {n_neighbor}, radius: {radius}")
        # tot_batches_unique_nn = 0.0
        # for j in range(ball_idx.size(0)):
        #     cur_ball_idx = ball_idx[j]
        #     curr_batch_unique_nn = 0
        #     for i_pts in range(cur_ball_idx.size(0)):
        #         curr_ball_curr_pts_neis = cur_ball_idx[i_pts]
        #         curr_ball_curr_pts_nn_unique_neis = torch.unique(curr_ball_curr_pts_neis)
        #         curr_ball_curr_pts_nn_unique = curr_ball_curr_pts_nn_unique_neis.size(0)
        #         curr_batch_unique_nn += curr_ball_curr_pts_nn_unique
        #     tot_batches_unique_nn += float(curr_batch_unique_nn) / float(cur_ball_idx.size(0))
        # tot_batches_unique_nn /= float(ball_idx.size(0))
        # print(f"avg_batches_unique_nn: {tot_batches_unique_nn}")
        ''' Get average number of valid neighbours '''


        #
        # # rot: bz x N x 1 x 3 x 3
        # nn = grouped_xyz.size(-1) # grouped_xyz #
        # rot = pose[:, :, :3, :3].unsqueeze(2).repeat(1, 1, nn, 1, 1)
        # # grouped_xyz: bz x N x nn x 3
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # grouped_xyz  = torch.matmul(rot, grouped_xyz.unsqueeze(-1)).squeeze(-1)
        # grouped_xyz = grouped_xyz.contiguous().permute(0, 3, 1, 2).contiguous()
        # grouped_xyz

        ''' Current version: using points canonicalized '''
        # discretization error?

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

        # get matrix's trace
        # get trace
        def get_rel_mtx_trace(mm):
            trace = mm[..., 0, 0] + mm[..., 1, 1] + mm[..., 2, 2]
            return trace

        ''' Group pose for relative distance calculation '''
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

        ''' Get rotated anchors '''
        # rotated_anchors = torch.matmul(rel_grouped_rotations.unsqueeze(3), anchors)  # rotate anchors
        # rotated_anchors_dists = torch.matmul(rotated_anchors.unsqueeze(4),
        #                                      anchors.unsqueeze(0).contiguous().transpose(2, 3).contiguous())
        # dists = get_rel_mtx_trace(rotated_anchors_dists)
        # rotated_anchor_idx = torch.argmax(dists, dim=-1)

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

        ''' Set initial features to relative positional offset '''
        # if feats.size(1) == 1:
        #     feats = grouped_xyz.clone()
        ''' Set initial features to relative positional offset '''

        feats = zpconv.add_shadow_feature(feats)
        # trans_feats: bz x N x na x dim
        trans_feats = feats.contiguous().permute(0, 2, 3, 1).contiguous()
        na, feat_dim = trans_feats.size(2), trans_feats.size(3)
        # grouped_feats: bz x N x nn x na x dim
        grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
        # Get and expand not-same-seg-indicators
        not_same_segs_indicator = (seg.unsqueeze(-1) != grouped_segs)
        not_same_segs_indicator = not_same_segs_indicator.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, na, feat_dim)
        pooled_grouped_feats, _ = torch.max(grouped_feats, dim=-2, keepdim=True)
        # pooled_grouped_feats = torch.mean(grouped_feats, dim=-2, keepdim=True)
        pooled_grouped_feats = pooled_grouped_feats.contiguous().repeat(1, 1, 1, na, 1)
        # select pooled grouped features
        grouped_feats[not_same_segs_indicator] = pooled_grouped_feats[not_same_segs_indicator]


        # grouped_xyz: bz x 3 x N x nn; trans_feats: bz x N x nn x 3
        # trans_feats = grouped_xyz.contiguous().permute(0, 2, 3, 1).contiguous()
        # bz x N x nn x na x 3

        # for each part pose...
        # grouped_feats = torch.matmul(anchors, trans_feats.unsqueeze(3).unsqueeze(-1)).squeeze(-1)

        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("trans_feats", trans_feats.size(), "inter_idx", inter_idx.size())
        # print(f"trans_feats: {trans_feats.size()}, ball_idx: {ball_idx.size()}")
        # grouped_feats = batched_index_select_other(trans_feats, ball_idx, dim=1)  # to another shape
        # grouped_feats: [nb, p2, npp, na, c_in]
        # print("grouped_feats", grouped_feats.size())

        #### DEBUG ####
        # print(f"grouped_feats.size: {grouped_feats.size()}, rotated_anchor_idx: {rotated_anchor_idx.size()}")
        # grouped_feats = batched_index_select_other(grouped_feats, rotated_anchor_idx, dim=3)
        #### DEBUG ####
        # print("rotated feats", grouped_feats.size())
        # todo: change `transpose` to `premute`
        # grouped_feats = grouped_feats.contiguous().transpose(3, 4).contiguous().transpose(2, 3).contiguous().transpose(
        # 1, 2).contiguous()

        #### zzz
        grouped_feats = grouped_feats.contiguous().permute(0, 4, 1, 2, 3).contiguous()

        ''' Set initial features to relative positional offset '''
        # grouped_feats = torch.ones_like(grouped_feats)[:, 0, ...].unsqueeze(1)
        # if grouped_feats.size(1) == 1:
        #     grouped_feats = grouped_xyz.unsqueeze(-1)
        ''' Set initial features to relative positional offset '''

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
    # rotate kernels
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

    # select 3 dimensions from features' 60 anchors;
    feature1 = feature.index_select(3, intra_idx.view(-1)).view(nb, c_in, nq, na, pnn)
    # grouped_feat: bz x c_in x pnn x nq x na
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


# intra convolution
def intra_so3conv_grouping_2D(intra_idx, feature):
    '''
        intra_idx: [na,pnn] so3 neighbors
        feature: [nb, c_in, np, na] features # number of batch, number of point, number of rotation group element
    '''

    # group features -> [nb, c_in, pnn, np, na] # nb, c_in, pnn,

    nb, c_in, nq, tot_na = feature.shape
    feature = feature.contiguous().view(nb, c_in, nq, -1, 4).contiguous()
    na2 = 4
    na = feature.size(-2)

    _, pnn = intra_idx.shape

    # select 3 dimensions from features' 60 anchors;
    feature1 = feature.index_select(3, intra_idx.view(-1)).view(nb, c_in, nq, na, pnn, na2)
    # grouped_feat: bz x c_in x pnn x nq x na
    grouped_feat = feature1.permute([0, 1, 4, 2, 3, 5]).contiguous()
    grouped_feat = grouped_feat.contiguous().view(nb, c_in, pnn, nq, tot_na).contiguous()

    return grouped_feat

# initialize so3 sampling
import vgtk.functional as fr
import os

GAMMA_SIZE = 3
ROOT = vgtk.__path__[0]
ANCHOR_PATH = os.path.join(ROOT, 'data', 'anchors/sphere12.ply')

print(f"anchor_path: {ANCHOR_PATH}")
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
