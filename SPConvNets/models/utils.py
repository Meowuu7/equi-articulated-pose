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

def v(var, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(), volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var), volatile=volatile)
    if cuda:
        res = res.cuda()
    return res

def mean_shift(x, bandwidth):
    # x: [N, f]
    b, N, c = x.shape
    IDX = torch.zeros(b, N).to(x.device).long()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=8)
    x_np = x.data.cpu().numpy()
    for i in range(b):
        #print ('Mean shift clustering, might take some time ...')
        #tic = time.time()
        ms.fit(x_np[i])
        #print ('time for clustering', time.time() - tic)
        IDX[i] = v(ms.labels_)
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]
    return IDX

def labels_to_one_hot_labels(labels, max_class=25):
    # bz x N
    bz, N = labels.size(0), labels.size(1)
    labels = labels.long()
    one_hot_labels = torch.eye(max_class, device=labels.device, dtype=torch.long)[torch.clamp(labels, max=max_class - 1)]
    return one_hot_labels

def calculate_acc(pred: torch.LongTensor, gt: torch.LongTensor):
    return (torch.sum(torch.max(pred.detach(), dim=1)[1] == gt).cpu().item())

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


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    feat_dim = pos.size(-1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()

    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.contiguous().view(-1, feat_dim).contiguous() # (bz x N, 3)
    # sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=False)
    # shape of sampled_idx?
    return sampled_idx


def get_knn_idx(pos: torch.FloatTensor, k: int, sampled_idx: torch.LongTensor=None, n_sampling: int=None):
    bz, N = pos.size(0), pos.size(1)
    if sampled_idx is not None:
        assert n_sampling is not None
        pos_exp = pos.view(bz * N, -1)
        pos_sampled = pos_exp[sampled_idx, :]
        rel_pos_sampled = pos_sampled.view(bz, n_sampling, 1, -1) - pos.view(bz, 1, N, -1)
        rel_dist = rel_pos_sampled.norm(dim=-1)
        nearest_k_dist, nearest_k_idx = rel_dist.topk(k, dim=-1, largest=False)
        return nearest_k_idx
    # N = pos.size(0)

    rel_pos = pos.view(bz, N, 1, -1) - pos.view(bz, 1, N, -1)
    rel_dist = rel_pos.norm(dim=-1)
    nearest_k_dist, nearest_k_idx = rel_dist.topk(k, dim=-1, largest=False)
    # [bz, N, k]
    # return the nearest k idx
    return nearest_k_idx


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
    return meaniou, gt_conf, cur_avg_recall

#### Input:
def non_maximum_suppression(ious, scores, thd):
    scores = scores.detach().cpu().numpy()
    ious = ious.detach().cpu().numpy()
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        cur_iou = ious[i, ixs[1:]]
        remove_ixs = np.where(cur_iou > thd)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.long)
