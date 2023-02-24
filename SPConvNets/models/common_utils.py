import torch
import torch.nn as nn
import numpy as np
import vgtk.spconv as zpconv

DECODER_PT2PC = 'pt2pc'
DECODER_REGULAR = 'regular'

MODEL_DGCNN = 'DGCNN'
MODEL_SO3POSE = 'so3pose'
MODEL_SO3 = 'so3'
MODEL_KPCONV = 'kpconv'

EQUI_TRANSLATION_RESIDULE = 0
EQUI_TRANSLATION_ROVOLUTE = 1
EQUI_TRANSLATION_ORI = 2

DATASET_PARTNET = 'partnet'
DATASET_MOTION = 'motion'
DATASET_MOTION_PARTIAL = 'motion_partial'
DATASET_MOTION2 = 'motion2'
DATASET_HOI4D = 'hoi4d'
DATASET_HOI4D_PARTIAL = 'hoi4d_partial'
DATASET_DRAWER = 'drawer'

SHAPE_LAPTOP = 'laptop'

def save_view(x, target_shape):
    x = x.contiguous().view(*target_shape).contiguous()
    return x

def safe_chamfer_dist_call(xa, xb, chamfer_func):
    try:
        dist1, dist2 = chamfer_func(
            xa, xb, return_raw=True
        )
    except:
        dist1, dist2 = chamfer_func(
            xa, xb,
        )
    return dist1, dist2

def safe_transpose(x, dim1, dim2):
    x = x.contiguous().transpose(dim1, dim2).contiguous()
    return x


def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts

# # [b, 3, n] x [b, 3, m] x r x k x [b, c, m] ->
# # [b, n, k] x [b, 3, n, k] x [b, c, n, k]
# def ball_query(query_points, support_points, radius, n_sample, support_feats=None):
#     # TODO remove add_shadow_point here
#     idx = pctk.ball_query_index(query_points, support_points, radius, n_sample)
#     support_points = add_shadow_point(support_points)
#     # import ipdb; ipdb.set_trace()
#
#     if support_feats is None:
#         return idx, pctk.group_nd(support_points, idx)
#     else:
#         return idx, pctk.group_nd(support_points, idx), pctk.group_nd(support_feats, idx)

def get_purity_loss(recon_transformed_slot_points):
    # bz x n_s x M x 3
    bz, n_s, M = recon_transformed_slot_points.size(0), recon_transformed_slot_points.size(1), recon_transformed_slot_points.size(2)
    expanded_recon_slot_points = save_view(recon_transformed_slot_points, (bz, n_s * M, 3))
    expanded_recon_slot_points = safe_transpose(expanded_recon_slot_points, 1, 2)
    # ball_idx: bz x ns * M x k; grouped_xyz: bz x 3 x ns * M x k
    k = 40
    k = 32
    ball_idx, grouped_xyz = zpconv.ball_query(expanded_recon_slot_points, expanded_recon_slot_points, radius=0.20, n_sample=k, )

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

    idx_to_semantic_labels = torch.arange(n_s,).cuda().unsqueeze(-1).repeat(1, M).long()
    idx_to_semantic_labels = save_view(idx_to_semantic_labels, (n_s * M, ))
    # ball_semantic_labels: bz x n_s x M x k
    ball_semantic_labels = idx_to_semantic_labels[ball_idx.long()]
    ball_semantic_labels = save_view(ball_semantic_labels, (bz, n_s, M, k))
    # grouped_xyz = save_view(grouped_xyz, (bz, 3, n_s, M, k))
    # self_semantic_labels: bz x n_s x M x k
    self_semantic_labels = torch.arange(n_s).cuda().view(1, n_s, 1, 1).repeat(bz, 1, M, k)
    # l2_dist: bz x n_s x M x k
    # l2_dist = 2 * torch.clamp(torch.sum((grouped_xyz - recon_transformed_slot_points.contiguous().permute(0, 3, 1, 2).contiguous().unsqueeze(-1)) ** 2, dim=1), min=0.0, max=33)
    count_indicators = (ball_semantic_labels != self_semantic_labels).float()
    count_indicators_ss = torch.sum(count_indicators, dim=-1, keepdim=True)
    count_indicators_ss = count_indicators_ss.repeat(1, 1, 1, k)
    # count_indicators_ss = (count_indicators_ss < k / 2.).long()
    count_indicators[count_indicators_ss < k / 3.] = 0.

    # l2_dist = l2_dist * count_indicators
    # purity_loss = torch.sum(l2_dist) / torch.clamp(torch.sum(count_indicators, ), min=1e-9)
    ### not pure indicator ###
    # purity_loss = 2 * torch.mean(count_indicators)
    purity_loss = torch.mean(count_indicators)
    # print(f"purity_loss: {purity_loss.item()}")
    return purity_loss


def generate_3d(smaller=False):
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)

    rng = 0.25
    offset = 0.15

    if not smaller:
        effi = np.random.uniform(-rng, rng, (1,)).item()
    else:
        rng = 0.15
        offset = 0.05
        effi = np.random.uniform(-rng, rng, (1,)).item()

    # control the range of generated angle
    if effi < 0:
        effi -= offset
    else:
        effi += offset
    # angle
    theta = effi * np.pi


    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    w = np.array([np.cos(2 * np.pi * x2) * np.sqrt(x3),
                  np.sin(2 * np.pi * x2) * np.sqrt(x3),
                  np.sqrt(1 - x3)], dtype=np.float)
    w_matrix = np.array(
        [[0, -float(w[2]), float(w[1])], [float(w[2]), 0, -float(w[0])], [-float(w[1]), float(w[0]), 0]]
    )

    # rotation_matrix = np.eye(3) + w_matrix * sin_theta + (w_matrix ** 2) * (1. - cos_theta)
    rotation_matrix = np.eye(3) + w_matrix * sin_theta + (w_matrix.dot(w_matrix)) * (1. - cos_theta)

    return rotation_matrix

def get_dist_two_rots(Ra, Rb):
    inv_Ra = Ra.contiguous().transpose(-1, -2).contiguous()
    cc_Rab = torch.matmul(inv_Ra, Rb)
    cc_Rab = cc_Rab[..., 0, 0] + cc_Rab[..., 1, 1] + cc_Rab[..., 2, 2]
    cc_Rab = (cc_Rab - 1.) / 2.
    cc_Rab = 1. - cc_Rab
    return cc_Rab
