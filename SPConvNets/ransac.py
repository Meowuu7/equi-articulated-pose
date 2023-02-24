
import torch
import numpy as np

import vgtk.functional.rotation as vfr

from SPConvNets.pose_utils import mean_angular_error, rot_diff_degree
# from global_info import global_info

def random_choice_noreplace(l, n_sample, num_draw):
    '''
    l: 1-D array or list -> to choose from, e.g. range(N)
    n_sample: sample size for each draw
    num_draw: number of draws

    Intuition: Randomly generate numbers,
    get the index of the smallest n_sample number for each row.
    '''
    l = np.array(l)
    return l[np.argpartition(np.random.rand(num_draw, len(l)),
                             n_sample - 1,
                             axis=-1)[:, :n_sample]]


def ransac_fit_r(batch_dr, max_iter=100, thres=5.0, chosen_axis=None, flip_axis=False):
    thres = 20.0 # choose to use a relatively large rotation threshold
    # B, 3, 3
    best_score = 0
    chosen_hyp = None
    nb = batch_dr.shape[0] # batch size
    if chosen_axis is not None:
        print('--- we are processing a symmetric object!!!')
        batch_dr = batch_dr.transpose(-1, -2)

    # choosen_nn = 3
    # choosen_nn = 4
    # choosen_nn = 5
    # choosen_nn = 8
    choosen_nn = 10
    choosen_nn = nb
    # choosen_nn = 5
    # choosen_nn = 6

    def proj_rot(rot, num):
        cur_vec = rot[num]  # normalized
        next_vec = rot[(num + 1) % 3]
        proj = (next_vec * cur_vec).sum()
        next_vec = next_vec - proj * cur_vec
        next_vec = next_vec / torch.clamp(torch.sqrt((next_vec ** 2).sum()), min=1e-5)
        # current vector and next vector?
        final_vec = torch.cross(cur_vec, next_vec)
        new_ret = torch.eye(3).to(rot.device)
        new_ret[num] = cur_vec
        new_ret[(num + 1) % 3] = next_vec
        new_ret[(num + 2) % 3] = final_vec
        return new_ret

    def axis_mean(rot, chosen_axis, flip_axis):
        char2num = {'x': 0, 'y': 1, 'z': 2}
        num = char2num[chosen_axis]
        axis = np.eye(3)[num]
        axis = torch.tensor(axis).float().to(rot.device).reshape(1, 3, 1)
        proj = torch.matmul(rot, axis)  # [B, 3, 3] * [B, 3, 1] -> [B, 3, 1]
        if flip_axis:
            proj_reverse = proj[:, 0:1]   # [B, 1, 1]
            factor = torch.ones_like(proj_reverse)
            factor[torch.where(proj_reverse < 0)[0]] = -1
            proj = proj * factor
        avg_axis = torch.mean(proj, dim=0)  # [3, 1]
        avg_axis /= torch.norm(avg_axis, dim=0, keepdim=True)

        ret = torch.eye(3).to(rot.device)
        ret[num] = avg_axis.reshape(-1)
        ret = proj_rot(ret, num)
        ret = ret.transpose(-1, -2)
        return ret

    def compute_r(sample_idx):
        r_samples = batch_dr[sample_idx]
        if chosen_axis is not None:

            r_hyp = axis_mean(r_samples, chosen_axis, flip_axis)
            # error
            err = rot_diff_degree(r_hyp, batch_dr, chosen_axis=chosen_axis, flip_axis=flip_axis)
        else:
            r_hyp = vfr.so3_mean(r_samples.unsqueeze(0))
            # r_hyp = r_samples.unsqueeze(0)
            err = mean_angular_error(r_hyp, batch_dr) * 180 / np.pi
        inliers = (err < thres) * 1.0 # how to
        curr_score = inliers.mean()
        return curr_score, r_hyp, torch.where(inliers)[0]

    best_idx = random_choice_noreplace(torch.tensor(np.arange(nb)), choosen_nn, 1).squeeze()
    with torch.no_grad():
        for i in range(max_iter):
            sample_idx = random_choice_noreplace(torch.tensor(np.arange(nb)), choosen_nn, 1).squeeze()
            curr_score, r_hyp, idx = compute_r(sample_idx)
            if curr_score > best_score or chosen_hyp is None:
                best_score = curr_score
                chosen_hyp = r_hyp
        # r score and the chosen chosen hypothesis
        rec_score, rec_hyp, _ = compute_r(best_idx)
        if rec_score > best_score:
            best_score = rec_score
            chosen_hyp = rec_hyp

    if chosen_axis is not None:
        chosen_hyp = chosen_hyp.transpose(-1, -2)

    return chosen_hyp, best_score


def ransac_fit_t(batch_dt, batch_dr, delta_r, max_iter=100, thres=0.025):
    # B, 3, 3
    best_score = 0
    chosen_hyp = None
    nb = batch_dt.shape[0]
    chosen_nn = min(5, nb)
    # dt_candidates = torch.matmul(-batch_dt, delta_r)
    def compute_t(sample_idx): # compute translations
        t_samples = batch_dt[sample_idx]
        t_hyp = t_samples.mean(dim=0, keepdim=True)
        err = torch.norm(t_hyp - batch_dt, dim=-1)
        inliers = (err < thres) * 1.0 #
        curr_score = inliers.mean()
        return curr_score, t_hyp, torch.where(inliers)[0]

    best_idx = None
    with torch.no_grad():
        for i in range(max_iter):
            sample_idx = random_choice_noreplace(torch.tensor(np.arange(nb)), chosen_nn, 1).squeeze()
            curr_score, t_hyp, idx = compute_t(sample_idx)
            if curr_score > best_score:
                best_score = curr_score
                chosen_hyp = t_hyp
                best_idx = idx
        rec_score, rec_hyp, _ = compute_t(best_idx)
        if rec_score > best_score:
            best_score = rec_score
            chosen_hyp = rec_hyp

    return chosen_hyp, best_score
