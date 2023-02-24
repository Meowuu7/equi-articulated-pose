import numpy as np
import os

def get_rot_diff_dist():
    rot_diff_fodler = os.path.join("data", "rot-diff-data")
    fns = os.listdir(rot_diff_fodler)
    aa = {}
    for fn in fns:
        cur_rot_diff_fn = os.path.join(rot_diff_fodler, fn)
        rot_diffs = np.load(cur_rot_diff_fn, allow_pickle=True, ).item()
        for i_p in rot_diffs:
            # iip = i_p
            if i_p not in aa:
                aa[i_p] = rot_diffs[i_p] # .tolist()
            else:
                aa[i_p] = aa[i_p] + rot_diffs[i_p] # .tolist()


    for i_p in aa:
        print(f"current part: {i_p}")
        dists = {}
        angles = []
        for zz in aa[i_p]:
            # zzz = int(zz.item()) // 10
            zzz = int(zz) // 10
            # zzz = zz // 10
            if zzz not in dists:
                # dists[zzz] = [int(zz.item())]
                dists[zzz] = [int(zz)]
            else:
                # dists[zzz].append(int(zz.item()))
                dists[zzz].append(int(zz))
            # zz_val = float(zz.item())
            zz_val = float(zz)
            zz_val = min(zz_val, min(abs(90. - zz_val), abs(180. - zz_val)))
            angles.append(zz_val)
        sorted_angles = sorted(angles)
        medium_idx = int(len(sorted_angles) // 2)
        print(f"media: {sorted_angles[medium_idx]}")
        print("zzval: ", sum(angles) / len(angles))
        for zzz in dists:
            print(zzz, len(dists[zzz]))


def test_slot_pair_mult_R_queue():
    queue = np.load("slot_pair_mult_R_queue_0.npy", allow_pickle=True)
    print(queue.shape)
    # queue_len x n_s x 3 x 3
    qn, ns = queue.shape[0], queue.shape[1]
    # rel_dots: queue_len x queue_len x n_s x 3 x 3
    rel_dots = np.matmul(np.reshape(queue, (qn, 1, ns, 3, 3)), np.transpose(np.reshape(queue, (1, qn, ns, 3, 3)), [0, 1, 2, 4, 3]))
    print(rel_dots.shape)
    ax_x, ax_y, ax_z = rel_dots[..., 2, 1] - rel_dots[..., 1, 2], rel_dots[..., 0, 2] - rel_dots[..., 2, 0], rel_dots[..., 1, 0] - rel_dots[..., 0, 1]
    ax_x, ax_y, ax_z = np.reshape(ax_x, (qn, qn, ns, 1)), np.reshape(ax_y, (qn, qn, ns, 1)), np.reshape(ax_z, (qn, qn, ns, 1))
    # axes: queue_len x queue_len x n_s x 3
    axes = np.concatenate([ax_x, ax_y, ax_z], axis=-1)
    axes = axes / np.sqrt(np.clip(np.sum(axes ** 2, axis=-1, keepdims=True), a_min=1e-8, a_max=999999.9))
    # axes: queue_len x n_s x 3
    axes = axes[0]
    # dot_axes: queue_len x queue_len x n_s
    dot_axes = np.sum(np.reshape(axes, (qn, 1, ns, 3)) * np.reshape(axes, (1, qn, ns, 3)), axis=-1)
    dot_axes = np.mean(np.mean(dot_axes, axis=0), axis=0)
    print(dot_axes)

def get_res_with_large_rotation_errors(file_folder):
    res_names = {}
    file_names = os.listdir(file_folder)
    for cur_fn in file_names:
        real_fn = os.path.join(file_folder, cur_fn)
        # try:
        out_feats = np.load(real_fn, allow_pickle=True).item()
        if "rot_diff_canon" in out_feats:
            rot_diff_canon = out_feats["rot_diff_canon"][0]
            assert type(rot_diff_canon) == dict, f"type of rot_diff_canon: {type(rot_diff_canon)}, rot_diff_canon: {rot_diff_canon}"
            maxx_part_rot_diff = 0.0
            for part_idx in rot_diff_canon:
                cur_part_rot_diff = rot_diff_canon[part_idx]
                if cur_part_rot_diff > maxx_part_rot_diff:
                    maxx_part_rot_diff = cur_part_rot_diff

            if maxx_part_rot_diff > 50:
                res_names[cur_fn] = maxx_part_rot_diff
                # break
        # except:
        #     print(real_fn)
        #     continue
    print(res_names)

# ['test_out_feats_[250]_rnk_0.npy',
# 'test_out_feats_[283]_rnk_0.npy',
# 'test_out_feats_[236]_rnk_0.npy',
# 'test_out_feats_[253]_rnk_0.npy',
# 'test_out_feats_[2]_rnk_0.npy', 'test_out_feats_[4]_rnk_0.npy', 'test_out_feats_[97]_rnk_0.npy', 'test_out_feats_[101]_rnk_0.npy', 'test_out_feats_[294]_rnk_0.npy', 'test_out_feats_[106]_rnk_0.npy', 'test_out_feats_[204]_rnk_0.npy', 'test_out_feats_[208]_rnk_0.npy', 'test_out_feats_[0]_rnk_0.npy', 'test_out_feats_[185]_rnk_0.npy', 'test_out_feats_[194]_rnk_0.npy', 'test_out_feats_[90]_rnk_0.npy', 'test_out_feats_[90]_rnk_0.npy', 'test_out_feats_[263]_rnk_0.npy', 'test_out_feats_[256]_rnk_0.npy', 'test_out_feats_[210]_rnk_0.npy', 'test_out_feats_[210]_rnk_0.npy', 'test_out_feats_[202]_rnk_0.npy', 'test_out_feats_[202]_rnk_0.npy', 'test_out_feats_[222]_rnk_0.npy', 'test_out_feats_[297]_rnk_0.npy', 'test_out_feats_[213]_rnk_0.npy', 'test_out_feats_[51]_rnk_0.npy', 'test_out_feats_[51]_rnk_0.npy', 'test_out_feats_[89]_rnk_0.npy', 'test_out_feats_[76]_rnk_0.npy', 'test_out_feats_[200]_rnk_0.npy', 'test_out_feats_[291]_rnk_0.npy', 'test_out_feats_[291]_rnk_0.npy', 'test_out_feats_[251]_rnk_0.npy', 'test_out_feats_[102]_rnk_0.npy', 'test_out_feats_[105]_rnk_0.npy', 'test_out_feats_[1]_rnk_0.npy', 'test_out_feats_[1]_rnk_0.npy', 'test_out_feats_[209]_rnk_0.npy', 'test_out_feats_[3]_rnk_0.npy', 'test_out_feats_[288]_rnk_0.npy', 'test_out_feats_[288]_rnk_0.npy', 'test_out_feats_[279]_rnk_0.npy', 'test_out_feats_[279]_rnk_0.npy', 'test_out_feats_[87]_rnk_0.npy', 'test_out_feats_[11]_rnk_0.npy', 'test_out_feats_[11]_rnk_0.npy', 'test_out_feats_[174]_rnk_0.npy', 'test_out_feats_[174]_rnk_0.npy', 'test_out_feats_[77]_rnk_0.npy', 'test_out_feats_[98]_rnk_0.npy', 'test_out_feats_[5]_rnk_0.npy']


if __name__=="__main__":
    #### Get rotation different dictionary ####
    get_rot_diff_dist()
    #
    # test_slot_pair_mult_R_queue()

    file_folder = "./symmetry_reg_learn_axis_pred_T_oven_out_feats_weq_wrot_1_rel_rot_factor_0.5_equi_27_model_so3pose_decoder_regular_inv_attn_1_orbit_attn_0_slot_iters_7_topk_0_num_iters_2_npts_512_perpart_npts_256_bsz_1_init_lr_0.0001"
    # get_res_with_large_rotation_errors(file_folder)

    # current part: 0
    # 11 80
    # 10 58
    # 13 90
    # 9 66
    # 6 36
    # 8 58
    # 12 88
    # 15 93
    # 3 12
    # 17 132
    # 16 105
    # 14 92
    # 18 6
    # 5 18
    # 7 38
    # 4 19
    # 2 6
    # 1 2
    # 0 1
    # current part: 1
    # 17 83
    # 13 41
    # 8 22
    # 14 46
    # 10 32
    # 9 27
    # 15 46
    # 16 50
    # 7 21
    # 4 9
    # 2 2
    # 12 46
    # 5 11
    # 11 39
    # 6 15
    # 18 6
    # 3 2
    # 1 2
