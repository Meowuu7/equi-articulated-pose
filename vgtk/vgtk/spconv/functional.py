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


# load anchors -> [na, 3]
def get_anchors(anchor):
    def filter_anchor(pts):
        norms = np.sqrt(np.sum(pts**2, axis=1))
        pts_selected = pts[np.where(norms>0.5)]
        pts_selected /= np.expand_dims(norms[np.where(norms>0.5)],1)
        return pts_selected
    if isinstance(anchor, int):
        root = vgtk.__path__[0]
        anchors_path = os.path.join(root, 'data', 'anchors')
        ply_path = os.path.join(anchors_path, f'sphere{anchor:d}.ply')
    elif isinstance(anchor, str):
        ply_path = anchor
    elif isinstance(anchor, torch.Tensor):
        return anchor.detach().cpu()
    else:
        raise ValueError(f'Not recognized anchor type {type(anchor)}')
    ply = pctk.load_ply(ply_path).astype('float32')
    ply = filter_anchor(ply)
    ply = torch.from_numpy(ply)
    return ply


def get_kernel_rings_np(radius, aperature, kernel_size, multiplier = 1):

    if isinstance(kernel_size, int):
        rrange = np.linspace(0, radius, kernel_size+2, dtype=np.float32)[1:-1]
        kps = []
        for ri in range(kernel_size):
            wrange = get_angular_kernel_points_np(aperature, multiplier * ri + 1)
            for wi in wrange:
                kps.append([rrange[ri],wi])
    else:
        # assume kernel_size is a 2-tuple
        rrange = np.linspace(radius/kernel_size[0],radius, kernel_size[0], dtype=np.float32)
        wrange = get_angular_kernel_points_np(aperature, kernel_size[1])
        rrange = np.expand_dims(np.expand_dims(rrange,1),2)
        wrange = np.expand_dims(np.expand_dims(wrange,0),2)
        rrange = np.tile(rrange, [1, wrange.shape[1],1])
        wrange = np.tile(wrange, [rrange.shape[0],1,1])
        kps = np.concatenate((rrange,wrange),axis=2).reshape(-1,2)
    kps = np.array(kps).astype('float32')
    return kps


def get_angular_kernel_points_np(aperature, kernel_size):
    end = 0.5*aperature
    return np.linspace(0, end, kernel_size+2, dtype=np.float32)[1:-1]

# pc: [nb,np,3] -> feature: [nb,1,np,na]
# def get_occupancy_features(pc, n_anchor, use_center=False):
#     nb, np, nd = pc.shape
#     has_normals = nd == 6

#     features = torch.zeros(nb, 1, np, n_anchor) + 1

#     import ipdb; ipdb.set_trace()

#     if use_center:
#         features[:,:,0,:] = 0.0
#     return features.float().to(pc.device)

# Add shadow xyz (inf)
# [b, c, n] -> [b, c, n+1]
def add_shadow_point(x):
    b, c, _ = x.shape
    shadow_point = torch.ones(b,c,1).float().to(x.device) * 1e4
    x = torch.cat((x,shadow_point), dim=2).contiguous()
    return x

# Add shadow feature (zeros)
# [b, c, n, a] -> [b, c, n+1, a]
# why we should add shadow features?
def add_shadow_feature(x):
    b, c, _, a = x.shape
    shadow_point = torch.zeros(b,c,1,a).float().to(x.device)
    x = torch.cat((x,shadow_point), dim=2).contiguous()
    return x


################################# Gathering ##############################


class Gathering(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, idx):
        '''
        Params:
            points: [nb, c_in, np]
            idx: [nb, m]
        Returns:
            gathered_points: [nb, c_in, m]
        '''
        gathered_points = gather.gather_points_forward(points, idx)
        ctx.save_for_backward(idx, points)
        return gathered_points

    @staticmethod
    def backward(ctx, grad_gathered_points):
        '''
        Params:
            grad_gathered_points: [nb, m, c_in]
            idx: [nb, m]
        Returns:
            grad_points: [nb, np, c_in]
        '''
        idx, points = ctx.saved_tensors
        np = points.size(2)
        grad_points = gather.gather_points_backward(grad_gathered_points.contiguous(), idx, np)
        return grad_points, None


################################# Intra ZPConv ##############################

def get_intra_kernels(aperature, kernel_size):
    kernels = np.linspace(0, 0.5*aperature, kernel_size, dtype=np.float32)
    kernels = torch.from_numpy(kernels)
    return kernels

def acos_safe(x, eps=1e-4):
    sign = torch.sign(x)
    slope = np.arccos(1-eps) / eps
    return torch.where(abs(x) <= 1-eps,
                    torch.acos(x),
                    torch.acos(sign * (1 - eps)) - slope*sign*(abs(x) - 1 + eps))

def anchor_knn(a_src, a_tgt, k=3, metric="spherical"):
    '''
    for each anchor in a_tgt, find k nearest neighbors in a_src
        ax3, ax3 -> axk indices, axk distances
    '''
    a_src = a_src.unsqueeze(0)
    a_tgt = a_tgt.unsqueeze(1)
    # sum(a_tgt x k)
    if metric == 'spherical':
        dists = torch.sum(a_src*a_tgt, dim=2) - 1.0
        val, idx = dists.topk(k=k,dim=1, largest=True)
    elif metric == 'angular':
        dists = acos_safe(torch.sum(a_src*a_tgt, dim=2))
        # dists[dists != dists] = np.pi
        val, idx = dists.topk(k=k,dim=1, largest=False)
    else:
        dists = torch.sum((a_src - a_tgt)**2, dim=2)
        val, idx = dists.topk(k=k,dim=1, largest=False)
    return val, idx


# TOCHECK
def get_intra_kernel_weights(anchor_in, anchor_out, kernels, ann, aperature, sigma=1e-1, use_suppression=False):
    '''
    Anchor weights for intrasphere convolution
    param:
        anchor_in: [a, 3]
        kernels: [k] angular bins
    return:
        idx: [a_out, ann] -> [a_in]
        influence: [a_out, ks, ann]
    '''
    anchor_out = anchor_in if anchor_out is None else anchor_out
    # a_out x ann
    angles, idx = anchor_knn(anchor_in, anchor_out, k=ann, metric='angular')

    # a_out x ks x ann
    if use_suppression:
        suppression = angles.le(0.5*aperature).unsqueeze(1).expand(-1,kernels.size(0),-1).float()

    # idx = idx.unsqueeze(1).expand(-1,kernels.size(0),-1)
    angles = angles.unsqueeze(1)
    kernels = kernels.unsqueeze(0).unsqueeze(-1)

    # a_out x ks x ann
    # influence = torch.cos(torch.abs(angles - kernels)) - 1.0
    ######### gaussian
    # influence = -(angles - kernels).pow(2)
    # influence = torch.exp(influence/sigma) 
    # influence = F.softmax(influence/sigma, dim=2)
    ######### end

    ######### linear
    influence = (angles - kernels).abs() / np.pi
    # import ipdb; ipdb.set_trace()
    influence = F.relu(1.0 - influence / (3*(sigma/2.0)**0.5), inplace=True)
    ######### end

    if use_suppression:
        influence = influence * suppression

    return idx.int().contiguous(), influence.contiguous()


# intra zpconv grouping
class IntraZPConvGrouping(torch.autograd.Function):

    @staticmethod
    def forward(ctx, intra_idx, intra_w, feats):
        '''
        Params:
            intra_idx:  [na_out, ann]
            intra_w:    [na_out, ks, ann]
            feats:      [nb, c_in, np, na_in]
        Returns:
            grouped_feats:  [nb, c_in, ks, np, na_out]
        '''
        grouped_feats = cuda_zpconv.intra_zpconv_forward(intra_idx,
                                                         intra_w,
                                                         feats)
        ctx.save_for_backward(intra_idx, intra_w, feats)
        return grouped_feats

    @staticmethod
    def backward(ctx, grad_grouped_feats):

        intra_idx, intra_w, feats = ctx.saved_tensors
        anchor_in = feats.shape[3]
        grad_feats = cuda_zpconv.intra_zpconv_backward(intra_idx,
                                                       intra_w,
                                                       grad_grouped_feats.contiguous(),
                                                       anchor_in)
        return None, None, grad_feats

def intra_zpconv_grouping(intra_idx, intra_w, feats):
    torch.cuda.synchronize()
    end = time.time()

    new_feats = IntraZPConvGrouping.apply(intra_idx, intra_w, feats)

    # torch.cuda.synchronize()
    # print('timer5:', time.time()-end, new_feats.shape)
    # end = time.time()

    return new_feats

def intra_zpconv_grouping_naive(intra_idx, intra_w, feats):
    a, k, nn = intra_w.shape
    b, c, p, _ = feats.shape

    torch.cuda.synchronize()
    end = time.time()

    # new_feats = feats[..., intra_idx.long()]
    new_feats = torch.index_select(feats, 3, intra_idx.long().view(-1)).view(b, c, p, a, nn)

    # torch.cuda.synchronize()
    # print('timer1:', time.time()-end, new_feats.shape)
    # end = time.time()

    new_feats = torch.einsum('bcpan,akn->bckpa',new_feats, intra_w).contiguous()

    # torch.cuda.synchronize()
    # print('timer2:', time.time()-end, new_feats.shape, intra_w.shape)
    # end = time.time()

    return new_feats


################################# Inter ZPConv ##############################


def get_inter_kernels(radius, aperature, kernel_size, add_zero=True):
    if isinstance(kernel_size, int):
        rrange = np.linspace(radius/kernel_size, radius, kernel_size, dtype=np.float32)
        kernels = []
        for ri in range(kernel_size):
            wrange = get_intra_kernel_weights(aperature, ri+1)
            for wi in wrange:
                kernels.append([rrange[ri],wi])
    else:
        # assume kernel_size is a 2-tuple
        rrange = np.linspace(radius/kernel_size[0],radius, kernel_size[0], dtype=np.float32)
        wrange = get_intra_kernel_weights(aperature, kernel_size[1])
        rrange = np.expand_dims(np.expand_dims(rrange,1),2)
        wrange = np.expand_dims(np.expand_dims(wrange,0),2)
        rrange = np.tile(rrange, [1, wrange.shape[1],1])
        wrange = np.tile(wrange, [rrange.shape[0],1,1])
        kernels = np.concatenate((rrange,wrange),axis=2).reshape(-1,2)

    if add_zero:
        kernels = np.vstack((kernels,np.array([0,0],dtype=np.float32)))
    return kernels

def get_sphere_kernels(radius, kernel_size, add_zero=True):
    assert isinstance(kernel_size, int)

    rrange = np.linspace(radius/kernel_size, radius, kernel_size, dtype=np.float32)
    kernels = []
    for ri in range(kernel_size):
        wrange = get_intra_kernel_weights(aperature, 2^(ri+1))
        for wi in wrange:
            kernels.append([rrange[ri],wi])

    if add_zero:
        kernels = np.vstack((kernels,np.array([0,0],dtype=np.float32)))
    return kernels

class InterZPConvGrouping(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inter_idx, inter_w, feats):
        '''
        Params:
            inter_idx:  [nb, np, na, ks, ann]
            inter_w:    [nb, np, na, ks, ann]
            feats:      [nb, c_in, nq+1, na]
        Returns:
            grouped_feats:  [nb, c_in, ks, np, na]
        '''
        grouped_feats = cuda_zpconv.inter_zpconv_forward(inter_idx, inter_w, feats)
        ctx.save_for_backward(inter_idx, inter_w, feats)
        return grouped_feats

    @staticmethod
    def backward(ctx, grad_grouped_feats):
        inter_idx, inter_w, feats = ctx.saved_tensors
        grad_feats = cuda_zpconv.inter_zpconv_backward(inter_idx, inter_w,
                                                       grad_grouped_feats.contiguous(), feats.size(2))
        return None, None, grad_feats



# [b, 3, n] x [b, 3, m] x r x k x [b, c, m] ->
# [b, n, k] x [b, 3, n, k] x [b, c, n, k]
def ball_query(query_points, support_points, radius, n_sample, support_feats=None):
    # TODO remove add_shadow_point here
    idx = pctk.ball_query_index(query_points, support_points, radius, n_sample)
    support_points = add_shadow_point(support_points)
    # import ipdb; ipdb.set_trace()

    if support_feats is None:
        return idx, pctk.group_nd(support_points, idx)
    else:
        return idx, pctk.group_nd(support_points, idx), pctk.group_nd(support_feats, idx)

# def inter_zpconv_grouping_naive(inter_idx, inter_w, feats):
#     b, p, a, ks, ann = inter_idx.shape
#     _, c, q, _ = feats.shape
#     device = inter_idx.device
#


#     inter_idx = inter_idx[..., None]*a+torch.arange(a)[:, None, None, None].to(device)+torch.arange(c).to(device)*q*a+\
#                                        torch.arange(b)[:, None, None, None, None, None].to(device)*c*q*a
#     new_feats = (torch.take(feats, inter_idx).view(b, p, a, ks, ann, c) * inter_w[..., None]).sum(-2)
#     return new_feats.permute(0, 4, 3, 1, 2).contiguous()

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def inter_zpconv_grouping_naive(inter_idx, inter_w, feats):
    # inter_w: [3, 128, 12, 1, 32]; bz x np x 1 x 12 x 32
    # inter_idx: [3, 128, 32]
    b, p, pnn = inter_idx.shape
    _, c, q, a = feats.shape # bz x c x np x na
    device = inter_idx.device

    # torch.cuda.synchronize()
    # end = time.time()
    # p, pnn --- should select proper a-order for p-pnn item; for each anchor calculate the distance between R and each anchor --- a in total
    # new_feats = batched_index_select(feats, 2, inter_idx.long().view(b, -1)).view(b, -1, p, pnn, a)
    feat_trans = feats.contiguous().permute(0, 2, 3, 1)
    # new_feats: bz x p x pnn x na x c
    # print("feat_trans.size, inter_idx.size", feat_trans.size(), inter_idx.size())
    # feat_trans.size, inter_idx.size torch.Size([3, 129, 1, 1]) torch.Size([3, 128, 32])
    new_feats = batched_index_select_other(values=feat_trans, indices=inter_idx.long(), dim=1)
    new_feats = new_feats.contiguous().permute(0, 4, 1, 2, 3)

    # torch.cuda.synchronize()
    # print('timer3:', time.time()-end, new_feats.shape)
    # end = time.time()
    # print("new_feat.size, inter_w.size", new_feats.size(), inter_w.size())
    # new_feat.size, inter_w.size torch.Size([3, 1, 128, 32, 1]) torch.Size([3, 128, 1, 12, 32])
    new_feats = torch.einsum('bcpna,bpakn->bckpa', new_feats, inter_w).contiguous()
    # print("new_feats.size", new_feats.size())
    # new_feats.size torch.Size([3, 1, 12, 128, 1])

    # torch.cuda.synchronize()
    # print('timer4:', time.time()-end, new_feats.shape, inter_w.shape)
    # end = time.time()

    return new_feats


def inter_pooling_naive(inter_idx, sample_idx, feats, alpha=0.5):
    b, p, pnn = inter_idx.shape
    _, c, q, a = feats.shape
    
    new_feats = batched_index_select(feats, 2, sample_idx.long())
    grouped_feats = batched_index_select(add_shadow_feature(feats), 2, inter_idx.long().view(b, -1)).view(b, -1, p, pnn, a)
    return alpha * new_feats + (1 - alpha) * grouped_feats.mean(3)


def inter_blurring_naive(inter_idx, feats, alpha=0.5):
    b, p, pnn = inter_idx.shape
    _, c, q, a = feats.shape
    assert p == q
    grouped_feats = batched_index_select(add_shadow_feature(feats), 2, inter_idx.long().view(b, -1)).view(b, -1, p, pnn, a)
    return alpha * feats + (1 - alpha) * grouped_feats.mean(3)


# inter zpconv grouping
# [b, 3, p1] x [b, 3, p2, a] -> [b, 3, p2, nn+1]
def inter_zpconv_grouping_ball(xyz, stride, radius, n_neighbor, lazy_sample=True):

    n_sample = math.ceil(xyz.shape[2] / stride) # number of samples
    if stride > 1:
        # [b, 3, p1] x [p2] -> [b,p2] x [b, 3, p2]
        idx, sample_xyz = pctk.furthest_sample(xyz, n_sample, lazy_sample)
    else:
        # print("here1")
        sample_xyz = xyz
        idx = torch.arange(xyz.shape[2], dtype=torch.long).cuda().unsqueeze(0).repeat(xyz.shape[0], 1)
    # [b, p2, nn] # how we aggregate neighbours' information... Ball query for valid neighbours and use kernels for aggregation
    #### For debug ####
    # ball_idx = torch.arange(xyz.shape[2], dtype=torch.long).cuda().unsqueeze(0).unsqueeze(-1).repeat(xyz.shape[0], 1, n_neighbor)
    # ball_idx += torch.arange(n_neighbor, dtype=torch.long).cuda().unsqueeze(0).unsqueeze(0)
    # ball_idx = ball_idx % xyz.shape[2]
    # print("ball_idx:", ball_idx.size, sample_xyz.size())
    # grouped_xyz = batched_index_select_other(values=sample_xyz.contiguous().transpose(1, 2).contiguous(), indices=ball_idx, dim=1).contiguous().permute(0, 3, 1, 2)
    #### For debug ####
    ball_idx, grouped_xyz = ball_query(sample_xyz, xyz, radius, n_neighbor)
    # [b, 3, p1+1] x [b, p2, nn] -> [b, 3, p2, nn]
    grouped_xyz = grouped_xyz - sample_xyz.unsqueeze(3) # relative position
    return grouped_xyz, ball_idx, idx, sample_xyz


def batched_index_select_other(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def inter_zpposeconv_grouping_ball(xyz, pose, stride, radius, n_neighbor, lazy_sample=True):

    n_sample = math.ceil(xyz.shape[2] / stride) # number of samples
    # [b, 3, p1] x [p2] -> [b,p2] x [b, 3, p2]
    # print(xyz.size(), n_sample)
    idx, sample_xyz = pctk.furthest_sample(xyz, n_sample, lazy_sample)
    # pose.size = b x p1 x 3 x 3 -> b x p2 x 3 x 3
    # print(idx.size())

    idx = idx.long()
    sampled_pose = batched_index_select(pose, dim=1, index=idx)
    # [b, p2, nn]
    ball_idx, grouped_xyz = ball_query(sample_xyz, xyz, radius, n_neighbor)
    # b x p1 x 3 x 3 -> b x p2 x nn x 3 x 3; torch tensor
    ball_idx = ball_idx.long()
    grouped_pose = batched_index_select_other(pose, ball_idx, dim=1,)

    #### get relative grouped rotation (from a neighbour to the center point) ####
    # inv_grouped_pos = grouped_pose.transpose(3, 4).contiguous()
    # # [b x p2 x 3 x 3] @ [b x p2 x nn x 3 x 3 ] -> [b x p2 x nn x 3 x 3]
    # rel_grouped_pos = torch.matmul(sampled_pose.unsqueeze(2), inv_grouped_pos)
    #### get relative grouped rotation (from a neighbour to the center point) ####

    # [b, 3, p1+1] x [b, p2, nn] -> [b, 3, p2, nn]
    grouped_xyz = grouped_xyz - sample_xyz.unsqueeze(3) # relative position

    # Transform the relative position
    #### transform via relative rotation difference #### ...but
    # grouped_xyz = torch.matmul(rel_grouped_pos, grouped_xyz.contiguous().transpose(1, 2).contiguous().transpose(2, 3).unsqueeze(-1)).squeeze(-1).contiguous().transpose(2, 3).contiguous().transpose(1, 2).contiguous()
    #### transform via relative rotation difference ####

    # return grouped_xyz, ball_idx, idx, sample_xyz, rel_grouped_pos, sampled_pose
    return grouped_xyz, ball_idx, idx, sample_xyz, grouped_pose, sampled_pose

# [b, 3, p2, nn] x [b, p2, nn] -> [b, p2, a, k, ann] #
def inter_zpconv_grouping_anchor(grouped_xyz: object, ball_idx: object, sample_idx: object, anchors: object, kernels: object,
                                 anchor_nn: object, n_support: object, radius: object, aperture: object, sigma: object) -> object:



    norm = grouped_xyz.pow(2).sum(1).sqrt() + 1e-6 #[b, p, nn]
    # 12 x 3 -> 3 x 12 -> [3, 1, 12, 1]; bz x 3 x np x nn
    # grouped_xyz, anchor, size:  torch.Size([3, 3, 128, 32]) torch.Size([12, 3])
    # [12, 1, 3, 1]; grouped_xyz: [3 x 3 x 128 x 1 x 32]
    # print("grouped_xyz, anchor, size: ", grouped_xyz.size(), anchors.size())
    # print("anchor reshaped: ", anchors.t()[:, None, :, None].size())
    # [3 x 3 x 128 x 1 x 32] xx [3 x 1 x 12 x 1] -> [3 x 3 x 128 x 12 x 32]
    cos_theta = (grouped_xyz.unsqueeze(3) * anchors.t()[:, None, :, None]).sum(1) / norm.unsqueeze(2) #[b, p, a, nn]
    # print("cos_theta.size: ", cos_theta.size())
    # cos_theta.size:  torch.Size([3, 128, 12, 32])
    theta = acos_safe(cos_theta).unsqueeze(3)
    # theta = acos_safe(cos_theta).unsqueeze(2)

    norm2 = norm[:, :, None, None, :]
    knorm2 = kernels[:, :1]
    theta2 = kernels[:, 1:]
    # dist2 = norm2.pow(2) + knorm2.pow(2) - 2.* norm2 * knorm2 * torch.cos(theta - theta2)
    ######## gaussian kernel
    # ratio = 10.0
    # dist2 = (norm2-knorm2).pow(2)*ratio+(norm2*(theta-theta2)).pow(2)/ratio
    # inter_w = torch.exp(-dist2 / sigma)
    ######## end

    ######## linear kernel
    ratio = 3.0
    dist1 = (norm2-knorm2).abs()+(norm2*(theta-theta2)).abs()/ratio
    
    # inter_w = F.relu(1.0 - dist1 / (((sigma)/2.0)**0.5*3), inplace=True)
    inter_w = F.relu(1.0 - dist1 / sigma**0.5, inplace=True)
    
    ######## end
    # inter_w = torch.softmax(-dist2 / sigma, dim=4)
    # import ipdb; ipdb.set_trace()
    inter_idx = ball_idx#[:, :, None, None, :].expand_as(inter_w).contiguous()

    # inter_w = inter_w[:, :, :, :, :anchor_nn].contiguous()
    # inter_idx = inter_idx[:, :, :, :, :anchor_nn].contiguous()

    # torch.cuda.synchronize()
    # print('time: ', time.time() - end)



    # #########################
    # # TODO
    # # grouped_xyz = grouped_xyz.permute(0,2,3,1).contiguous()
    # torch.cuda.synchronize()
    # end = time.time()
    # inter_w = cuda_nn.anchor_query(sample_idx, ball_idx, grouped_xyz,
    #                                anchors, kernels,
    #                                n_support)
    # inter_w = inter_w[0]

    # torch.cuda.synchronize()
    # print('time: ', time.time() - end)
    # # import ipdb; ipdb.set_trace()

    # # import ipdb;ipdb.set_trace()
    # # inter_w = F.softmax(-inter_w / sigma, dim=4)
    # inter_w = torch.exp(-inter_w / sigma)
    # # inter_n = (inter_w != 0).sum(-1)[...,None].float()
    # # inter_w = inter_w / inter_n
    # inter_idx = ball_idx[:, :, None, None, :].expand_as(inter_w).contiguous()
    # #########################

    return inter_idx, inter_w


def inter_zpconv_grouping(xyz, feats, stride, n_neighbor,
                          anchors, kernels, anchor_nn,
                          radius, aperture, sigma,
                          inter_idx=None, inter_w=None, lazy_sample=True,
                          radius_expansion=1.0):

    if inter_idx is None:

        grouped_xyz, ball_idx, idx, new_xyz = inter_zpconv_grouping_ball(xyz, stride, 
                                                                         radius * radius_expansion, n_neighbor, lazy_sample)

        n_support = xyz.shape[2]
        inter_idx, inter_w = inter_zpconv_grouping_anchor(grouped_xyz, ball_idx,
                                                          idx, anchors, kernels,
                                                          anchor_nn, n_support,
                                                          radius, aperture, sigma)
        inter_w = inter_w.contiguous().permute(0, 1, 3, 2, 4).contiguous()
        print("inter_idx and inter_w.size: ", inter_idx.size(), inter_w.size())
        # inter_idx_w.size:  torch.Size([3, 128, 32]) torch.Size([3, 128, 12, 1, 32]) [b, p2, a, k, ann]
    else:
        new_xyz = xyz

    feats = add_shadow_feature(feats)

    # new_feats = InterZPConvGrouping.apply(inter_idx, inter_w, feats) # [nb, nc, ks, np, na]
    new_feats = inter_zpconv_grouping_naive(inter_idx, inter_w, feats) # [nb, nc, ks, np, na]
    # new_feats = new_feats / (kernels[:, 0] * torch.sin(kernels[:, 1]))[:, None, None]

    return inter_idx, inter_w, new_xyz, new_feats


################################# Anchor Prop ##############################


def compute_anchor_weights(anchor_in, anchor_out, k=3, sigma=1e-1, interpolation="inv"):
    '''
    Anchor weights for knn interpolation
    param:
        anchor_in: [a1, 3]
        anchor_out: [a2, 3]
    return:
        idx: [a2, k]
        w: [a2, k]
    '''

    # TODO add knn to pctk
    # val, idx = anchor_knn(anchor_in, anchor_out, k=k, metric=interpolation)


    if interpolation == 'spherical':
        dists = (anchor_in.unsqueeze(0) * anchor_out.unsqueeze(1)).sum(2) - 1.0
        val, idx = dists.topk(k=k, dim=1, largest=True)
        w = F.softmax(val / sigma, dim=1)
    elif interpolation == 'euclidean':
        # TODO code redundant
        dists = (anchor_in.unsqueeze(0) - anchor_out.unsqueeze(1)).pow(2).sum(2)
        val, idx = dists.topk(k=k, dim=1, largest=False)
        w = F.softmax(-val / sigma, dim=1)
    elif interpolation == 'inv':
        dists = (anchor_in.unsqueeze(0) - anchor_out.unsqueeze(1)).pow(2).sum(2)
        val, idx = dists.topk(k=k, dim=1, largest=False)
        # w = F.softmax(1. / ((val + 1e-6) * sigma), dim=1)
        inv_val = 1./(sigma * val+1e-6)
        w = inv_val / inv_val.sum(1, keepdim=True)
        # import ipdb; ipdb.set_trace()
    return idx, w


# [b, c, p, a1] -> [b, c, p, a2]
def anchor_prop(x, idx, w):
    '''
    Propagate signals to new anchor directions (with 3NN)
    param:
        x: [b, c, p, a1]    BxNxa_inxc_in
        idx: [a2, k]        Bxa_outx3
        w: [a2, k]          1x1xa_outx3x1
    return:
        [b, c, p, a2]
    '''
    return (x[:, :, :, idx] * w).sum(-1)


    # bdim = x.size(0)
    # ndim = x.size(1)
    # c_in = x.size(3)
    # a_out = idx.size(1)
    # grouped_indices = idx.view(bdim, -1).int().contiguous()
    # grouped_features = SphereGathering.apply(x, grouped_indices)
    # # BxNxa_outx3xc_in
    # grouped_features = grouped_features.view(bdim, ndim, a_out, 3, c_in).contiguous()
    # return torch.sum(grouped_features * w, dim=3)


# ------------------------- LEGACY CODE ------------------------------------



# -------------------- kernel propagation -----------------------------------

# def compute_anchor_kernel_weights(anchors, anchor_out, kpts, ann, aperature, sigma=1e-1):
#     '''
#     Anchor weights for intrasphere convolution
#     param:
#         anchors_in: ax3
#         kpts: k angular bins
#     return:
#         idx: a_outxksxann -> a_in
#         influence: a_outxksxann
#     '''

#     # axann
#     anchor_out = anchors if anchor_out is None else anchor_out

#     # a_out x ann
#     angles, idx = anchor_knn(anchors, anchor_out, k=ann, metric='angular')
#     # a_out x ks x ann
#     suppression = angles.le(0.5*aperature).unsqueeze(1).expand(-1,kpts.size(0),-1).float()

#     idx = idx.unsqueeze(1).expand(-1,kpts.size(0),-1)
#     angles = angles.unsqueeze(1)
#     kpts = kpts.unsqueeze(0).unsqueeze(-1)

#     # a_out x ks x ann
#     influence = torch.cos(torch.abs(angles - kpts)) - 1.0
#     influence = torch.exp(influence/sigma)
#     influence = influence * suppression

#     return idx.int().contiguous(), influence.contiguous()

