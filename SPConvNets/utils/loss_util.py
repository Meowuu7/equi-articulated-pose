import torch
try:
    from torch_cluster import fps
except:
    pass

import warnings
from torch.autograd import Function, Variable
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MeanShift

def batched_index_select(values, indices, dim = 1):
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

def hungarian_matching(pred_x, gt_x, curnmasks, include_background=True):
    """ pred_x, gt_x: B x nmask x nsmp
        curnmasks: B
        return matching_idx: B x nmask x 2 """
    batch_size = gt_x.shape[0]
    nmask = gt_x.shape[1]

    matching_score = np.matmul(gt_x,np.transpose(pred_x, axes=[0,2,1])) # B x nmask x nmask
    # matching_score = torch.matmul(gt_x, pred_x.transpose(1, 2))
    matching_score = 1-np.divide(matching_score,
                                 np.expand_dims(np.sum(pred_x, 2), 1) + np.sum(gt_x, 2, keepdims=True) - matching_score+1e-8)
    # matching_score = 1. - torch.divide(matching_score,
    #                                    pred_x.sum(2).unsqueeze(1) + ji)
    matching_idx = np.zeros((batch_size, nmask, 2)).astype('int32')
    curnmasks = curnmasks.astype('int32')
    for i, curnmask in enumerate(curnmasks):
        # print(curnmask.shape)
        curnmask = int(curnmask)
        # curnmask = min(curnmask, pred_x.shape[1])
        assert pred_x.shape[1] >= curnmask, "Should predict no less than n_max_instance segments!"
        # Truncate invalid masks in GT predictions
        row_ind, col_ind = linear_sum_assignment(matching_score[i,:curnmask,:])
        # row_ind, col_ind = linear_sum_assignment(matching_score[i,:,:])
        matching_idx[i,:curnmask,0] = row_ind[:curnmask]
        matching_idx[i,:curnmask,1] = col_ind[:curnmask]
    return torch.from_numpy(matching_idx).long()

def get_hard_pred_res(pred_x):
    with torch.no_grad():

        _, pred_idx = torch.max(pred_x.contiguous().transpose(1, 2).contiguous(), dim=-1)
        maxx_idx = torch.max(pred_idx).detach().item()

        # hard_pred_x = torch.eye(maxx_idx + 1)[pred_idx].float().to(pred_x.device)
        hard_pred_x = batched_index_select(torch.eye(maxx_idx + 1).to(pred_x.device).float(), pred_idx, dim=0)
        hard_pred_x = hard_pred_x.contiguous().transpose(1, 2).contiguous()
        # print(hard_pred_x.size())
        return hard_pred_x


def iou(pred_x, gt_x, gt_conf, nsmp=128, nmask=10, pred_conf=None, tch=True):
    # print(pred_x.size(), gt_x.size(), gt_conf.size())
    device = gt_x.device
    pred_x = pred_x.float()
    gt_x = gt_x.float()
    cur_masks = gt_conf.sum(1).long()
    gt_conf = gt_conf.float()
    # print(f"in iou pred_x.size = {pred_x.size()}, gt_x.size = {gt_x.size()}, gt_conf.size = {gt_conf.size()}")
    with torch.no_grad():
        # print(gt_conf.size())
        matching_idx = hungarian_matching(pred_x.cpu().numpy(), gt_x.cpu().numpy(), gt_conf.sum(1).cpu().numpy())


    matching_idx = matching_idx.to(device)

    matching_idx_row = matching_idx[:, :, 0]
    gt_x_matched = batched_index_select(gt_x, matching_idx_row, dim=1)
    gt_conf = batched_index_select(gt_conf, matching_idx_row, dim=1)

    curnmasks = cur_masks.cpu().numpy().astype('int32')
    for i, curnmask in enumerate(curnmasks):
        # print(curnmask.shape)
        curnmask = int(curnmask)
        gt_conf[i, :curnmask] = 1.
        gt_conf[i, curnmask:] = 0
    # gt_conf[:, cur_masks:] = 0
    gt_conf = gt_conf.float()

    matching_idx_colum = matching_idx[:, :, 1]


    with torch.no_grad():
        # pred_x.size = bz x nmasks x N
        hard_pred_x = get_hard_pred_res(pred_x)
        # hard_pred_x = pred_x
        matching_sc = torch.matmul(gt_x, hard_pred_x.contiguous().transpose(1, 2))
        # matching_sc = 1 - np.divide(matching_sc,
        #
        #                                np.expand_dims(np.sum(pred_x, 2), 1) + np.sum(gt_x, 2,
        #                                                                              keepdims=True) - matching_sc + 1e-8)
        # matching_sc = torch.div(matching_sc,
        #                         torch.sqrt(torch.sum(gt_x ** 2, dim=-1, keepdim=True)) + torch.sqrt(torch.sum(hard_pred_x ** 2, dim=-1, keepdim=False).unsqueeze(-2)) - matching_sc + 1e-8)
        matching_sc = torch.div(matching_sc,
                                torch.sum(gt_x, dim=-1, keepdim=True) +
                                    torch.sum(hard_pred_x, dim=-1, keepdim=False).unsqueeze(
                                        -2) - matching_sc + 1e-8)
        matching_sc = torch.clamp(matching_sc, min=0)
        # bz x nmasks
        max_matching_sc, _ = torch.max(matching_sc, dim=-1)
        thr = 0.5
        avg_tot_recall = 0.0
        tot_step = 0
        # thr_to_recall = {}
        # record_iou = {0.5: 1, 0.7: 1, 0.9: 1}
        while thr < 0.96:
            cur_recalled_indicator = (max_matching_sc >= thr).float()
            cur_recalled_nn = torch.sum(cur_recalled_indicator * gt_conf, dim=-1) / torch.clamp(
                torch.sum(gt_conf, dim=-1), min=1e-9)
            avg_cur_recall = torch.mean(cur_recalled_nn).item()
            avg_tot_recall += avg_cur_recall
            # if abs(thr - 0.5) < 0.025 or abs(thr - 0.7) < 0.025 or abs(thr - 0.9) < 0.025:
            #     thr_to_recall[thr] = avg_cur_recall
            tot_step += 1
            thr += 0.05
        cur_avg_recall = avg_tot_recall / float(tot_step)

        # hard_x_matched = batched_index_select(hard_pred_x, matching_idx_colum, dim=1)
        # hard_gt_x_matched = batched_index_select(gt_x, matching_idx_row, dim=1)
        # hard_matching_score = torch.sum(hard_x_matched * hard_gt_x_matched, dim=2)
        #
        # hard_iou_all = torch.div(
        #     hard_matching_score, hard_gt_x_matched.sum(2) + hard_x_matched.sum(2) - hard_matching_score + 1e-8
        # )
        #
        # hard_meaniou = torch.div(torch.sum(hard_iou_all * gt_conf, 1), gt_conf.sum(1) + 1e-8)
        #
        # cur_avg_recall = hard_meaniou.mean()


    pred_x_matched = batched_index_select(pred_x, matching_idx_colum, dim=1)

    matching_score = torch.sum(gt_x_matched * pred_x_matched, dim=2)

    iou_all = torch.div(
        matching_score, gt_x_matched.sum(2) + pred_x_matched.sum(2) - matching_score + 1e-8
    )

    meaniou = torch.div(torch.sum(iou_all * gt_conf, 1), gt_conf.sum(1) + 1e-8)
    iou_all = iou_all * gt_conf
    # return meaniou, gt_conf, iou_all
    # return meaniou, gt_conf, cur_avg_recall
    return meaniou, matching_idx_row, matching_idx_colum

def get_rel_angles(RRs):
    pred_Rs = RRs
    inv_pred_Rs = pred_Rs.contiguous().transpose(-1, -2).contiguous()
    # rel_pred_Rs: bz x n_s x n_s x 3 x 3
    rel_pred_Rs = torch.matmul(inv_pred_Rs.unsqueeze(1), pred_Rs.unsqueeze(2))
    rel_pred_angles = rel_pred_Rs[..., 0, 0] + rel_pred_Rs[..., 1, 1] + rel_pred_Rs[..., 2, 2]
    rel_pred_angles = (rel_pred_angles - 1.) / 2.
    rel_pred_angles = torch.clamp(rel_pred_angles, max=1.0, min=0.0)
    return rel_pred_angles

def calculate_res_relative_Rs(pred_Rs, gt_Rs):
    # print(pred_Rs.size(), gt_Rs.size())
    # pred_RS: bz x n_s x 3 x 3 (torch array)
    # gt_RS: bz x n_s x 3 x 3 (torch array)
    # inv_pred_Rs = pred_Rs.contiguous().transpose(-1, -2).contiguous()
    # # rel_pred_Rs: bz x n_s x n_s x 3 x 3
    # rel_pred_Rs = torch.matmul(inv_pred_Rs.unsqueeze(1), pred_Rs.unsqueeze(2))
    # rel_pred_angles = rel_pred_Rs[..., 0, 0] + rel_pred_Rs[..., 1, 1] + rel_pred_Rs[..., 2, 2]
    # rel_pred_angles = (rel_pred_angles - 1.) / 2.
    # rel_pred_angles = torch.clamp(rel_pred_angles, max=1.0, min=0.0)
    n_s = pred_Rs.size(1)
    div_values = float((n_s - 1) * n_s) / 2.
    rel_pred_angles = 1. - get_rel_angles(pred_Rs)

    rel_gt_angles = 1. - get_rel_angles(gt_Rs)

    dists = torch.sum(torch.sum((rel_pred_angles - rel_gt_angles) ** 2, dim=-1), dim=-1)
    dists = dists / (div_values)
    return dists.mean() # .item()

