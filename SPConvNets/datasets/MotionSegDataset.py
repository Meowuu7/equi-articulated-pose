'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''

import os
import os.path
import json
import numpy as np
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import provider
from torch.utils import data
import scipy.io as sio
import copy
from SPConvNets.models.common_utils import *
# from model.utils import farthest_point_sampling

# padding 1
def padding_1(pos):
    pad = np.array([1.], dtype=np.float).reshape(1, 1)
    # print(pos.shape, pad.shape)
    return np.concatenate([pos, pad], axis=1)

# normalize point-cloud
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


# decod rotation info
def decode_rotation_info(rotate_info_encoding):
    if rotate_info_encoding == 0:
        return []
    rotate_vec = []
    if rotate_info_encoding <= 3:
        temp_angle = np.reshape(np.array(np.random.rand(3)) * np.pi, (3, 1))
        if rotate_info_encoding == 1:
            line_vec = np.concatenate([
                np.cos(temp_angle), np.zeros_like(temp_angle), np.sin(temp_angle),
            ], axis=-1)
        elif rotate_info_encoding == 2:
            line_vec = np.concatenate([
                np.cos(temp_angle), np.sin(temp_angle), np.zeros_like(temp_angle)
            ], axis=-1)
        else:
            line_vec = np.concatenate([
                np.zeros_like(temp_angle), np.cos(temp_angle), np.sin(temp_angle)
            ], axis=-1)
        return [line_vec[0], line_vec[1], line_vec[2]]
    elif rotate_info_encoding <= 6:
        base_rotate_vec = [np.array([1.0, 0.0, 0.0], dtype=np.float),
                           np.array([0.0, 1.0, 0.0], dtype=np.float),
                           np.array([0.0, 0.0, 1.0], dtype=np.float)]
        if rotate_info_encoding == 4:
            return [base_rotate_vec[0], base_rotate_vec[2]]
        elif rotate_info_encoding == 5:
            return [base_rotate_vec[0], base_rotate_vec[1]]
        else:
            return [base_rotate_vec[1], base_rotate_vec[2]]
    else:
        return []


def rotate_by_vec_pts(un_w, p_x, bf_rotate_pos):

    def get_zero_distance(p, xyz):
        k1 = np.sum(p * xyz).item()
        k2 = np.sum(xyz ** 2).item()
        t = -k1 / (k2 + 1e-10)
        p1 = p + xyz * t
        # dis = np.sum(p1 ** 2).item()
        return np.reshape(p1, (1, 3))

    w = un_w / np.sqrt(np.sum(un_w ** 2, axis=0))
    # w = np.array([0, 0, 1.0])
    w_matrix = np.array(
        [[0, -float(w[2]), float(w[1])], [float(w[2]), 0, -float(w[0])], [-float(w[1]), float(w[0]), 0]]
    )

    rng = 0.25
    offset = 0.1

    effi = np.random.uniform(-rng, rng, (1,)).item()
    # effi = effis[eff_id].item()
    if effi < 0:
        effi -= offset
    else:
        effi += offset
    theta = effi * np.pi
    # rotation_matrix = np.exp(w_matrix * theta)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # rotation_matrix = np.eye(3) + w_matrix * sin_theta + (w_matrix ** 2) * (1. - cos_theta)
    rotation_matrix = np.eye(3) + w_matrix * sin_theta + (w_matrix.dot(w_matrix)) * (1. - cos_theta)

    # bf_rotate_pos = pcd_points[sem_label_to_idxes[rotate_idx][rotate_idx_inst]]

    trans = get_zero_distance(p_x, un_w)

    af_rotate_pos = np.transpose(np.matmul(rotation_matrix, np.transpose(bf_rotate_pos - trans, [1, 0])), [1, 0]) + trans

    # af_rotate_pos = rotation_matrix.dot((bf_rotate_pos - trans).T).T + trans
    return af_rotate_pos, rotation_matrix, np.reshape(trans, (3, 1))


DATASET_ROOT_THU_LAB = ["/mnt/8T/xueyi/part-segmentation/data", "/mnt/sas-raid5-7.2T/xueyi/part-segmentation/data", "./data/part-segmentation/data", "/home/xueyi/inst-segmentation/data/part-segmentation/data"]

class PartSegmentationMetaInfoDataset(data.Dataset):
    def __init__(
            self, root="/data-input/motion_part_split_meta_info", npoints=512, split='train', nmask=10, shape_types=["03642806", "04379243"],  real_test=False,
            part_net_seg=False, partnet_split=False, args=None, split_types=False, global_rot=0
    ):
        super(PartSegmentationMetaInfoDataset, self).__init__()
        self.dataset_path = os.path.join(root, f"seg_{split}")
        self.masks_names = "momasks"
        if "test" in split:
            self.dataset_path = os.path.join(root, split)
            self.masks_names = "seg1"
        self.nmask = nmask
        self.npoints = npoints
        self.split = split
        self.real_test = real_test
        self.part_net_seg = part_net_seg
        self.args = args
        # global_rot = global_rot

        self.lm = dict()
        self.peridx = dict()

        print(f"ROOT_DIR = {ROOT_DIR}")

        if self.split == 'train' or self.split == 'val' or (not self.real_test):
            self.data = {}
            if not self.part_net_seg:
                ''' If use ShapeNetPart dataset '''
                fl_path = os.path.join(root, f"all_type_tot_{self.split}_tot_part_motion_meta_info.npy")
                data = np.load(fl_path, allow_pickle=True).item()
                if split_types:
                    self.data = {}
                    shape_types_dict = {st: 1 for st in shape_types}
                    for k in data:
                        shp_ty_k, shp_idx_k = k.split("_")
                        if shp_ty_k in shape_types_dict:
                            self.data[k] = data[k]
                else:
                    self.data = data
            else:
                ''' If use PartNet dataset '''
                self.shp_types_to_number = dict()
                partnet_root = os.path.join(root, "..", "..")
                partnet_root = "/home/xueyi/inst-segmentation/data/part-segmentation/"
                for shp_name in shape_types:
                    cur_shp_meta_file_pth = os.path.join(partnet_root, "part_net_meta_info_category", shp_name)
                    if not partnet_split:
                        cur_shp_merged_meta_info = np.load(os.path.join(cur_shp_meta_file_pth, "motion_part_meta_info_merged.npy"), allow_pickle=True).item()
                    else:
                        cur_shp_merged_meta_info = np.load(os.path.join(cur_shp_meta_file_pth, f"{split}_motion_part_meta_info_merged.npy"), allow_pickle=True).item()
                    for shp_idx in cur_shp_merged_meta_info:
                        self.data[shp_idx] = cur_shp_merged_meta_info[shp_idx]
                    self.shp_types_to_number[shp_name] = len(cur_shp_merged_meta_info)
            print(f"{self.split} data loaded with total length = {len(self.data)}")
        else:
            self.dataset_path = os.path.join(root, "..", f"sf2f_test.mat")
            self.masks_names = "seg1"
            self.nmask = nmask
            self.npoints = npoints
            self.data = sio.loadmat(self.dataset_path)

        self.new_idx_to_old_idx = {}
        self.reindex_data_index()

        if split == 'train':
            self.whether_mov_p = np.array([0.0, 1.0 ], dtype=np.float)
        else:
            self.whether_mov_p = np.array([0.0, 1.0], dtype=np.float)
        self.get_trans_encoding_to_trans_dir()

        self.num_mov_parts_to_cnt = {}

    def get_trans_encoding_to_trans_dir(self):
        trans_dir_to_trans_mode = {
            (0, 1, 2): 1,
            (0, 2): 2,
            (0, 1): 3,
            (1, 2): 4,
            (0,): 5,
            (1,): 6,
            (2,): 7,
            (): 0
        }
        self.trans_mode_to_trans_dir = {trans_dir_to_trans_mode[k]: k for k in trans_dir_to_trans_mode}
        self.base_transition_vec = [
            np.array([1.0, 0.0, 0.0], dtype=np.float),
            np.array([0.0, 1.0, 0.0], dtype=np.float),
            np.array([0.0, 0.0, 1.0], dtype=np.float),
        ]

    def get_test_data(self):
        return self.test_data

    def reindex_data_index(self):
        old_idx_to_new_idx = {}
        ii = 0
        for old_idx in self.data:
            old_idx_to_new_idx[old_idx] = ii
            ii += 1
        self.new_idx_to_old_idx = {old_idx_to_new_idx[k]: k for k in old_idx_to_new_idx}

    def reindex_shape_seg(self, shape_seg):
        old_seg_to_new_seg = {}
        ii = 0
        for i in range(shape_seg.shape[0]):
            old_seg = int(shape_seg[i].item())
            if old_seg not in old_seg_to_new_seg:
                old_seg_to_new_seg[old_seg] = ii
                ii += 1
            new_seg = old_seg_to_new_seg[old_seg]
            shape_seg[i] = new_seg
            # old_seg_to_new_seg[old_seg] = ii
            # ii += 1
        return shape_seg

    def transit_pos_by_transit_vec(self, trans_pos):
        tdir = np.random.uniform(-1.0, 1.0, (3,))
        tdir = tdir / (np.sqrt(np.sum(tdir ** 2)).item() + 1e-9)
        trans_scale = np.random.uniform(1.0, 2.0, (1,)).item()
        # for flow test...
        trans_pos_af_pos = trans_pos + tdir * 0.1 * trans_scale * 2
        return trans_pos_af_pos, tdir * 0.1 * trans_scale * 2

    def transit_pos_by_transit_vec_dir(self, trans_pos, tdir):
        # tdir = np.zeros((3,), dtype=np.float)
        # axis_dir = np.random.choice(3, 1).item()
        # tdir[int(axis_dir)] = 1.
        trans_scale = np.random.uniform(0.0, 1.0, (1,)).item()
        trans_pos_af_pos = trans_pos + tdir * 0.1 * trans_scale
        return trans_pos_af_pos, tdir * 0.1 * trans_scale

    def get_random_transition_dir_scale(self):
        tdir = np.random.uniform(-1.0, 1.0, (3,))
        tdir = tdir / (np.sqrt(np.sum(tdir ** 2)).item() + 1e-9)
        trans_scale = np.random.uniform(1.0, 2.0, (1,)).item()
        return tdir, trans_scale

    def decode_trans_dir(self, trans_encoding):
        trans_dir = self.trans_mode_to_trans_dir[trans_encoding]
        return [self.base_transition_vec[d] for ii, d in enumerate(trans_dir)]

    def __getitem__(self, index):

        # reindex_idx = self.new_idx_to_old_idx[index]
        reindex_idx = self.new_idx_to_old_idx[index % 300]
        # cur_shape = copy.deepcopy(self.data[reindex_idx])
        cur_shape = self.data[reindex_idx]
        pc1 = copy.deepcopy(cur_shape['pc1'])
        # print(f"pc1.shape: {pc1.shape}")
        # print(np.max(pc1[:, 0]), np.min(pc1[:, 0]),)
        # print(np.max(pc1[:, 1]), np.min(pc1[:, 1]),)
        # print(np.max(pc1[:, 2]), np.min(pc1[:, 2]),)
        inst_part_idx_to_part_idx_to_meta_info = cur_shape['part_meta_info']
        chosen_whether_mov_index = np.random.choice(2, 1, p=self.whether_mov_p).item()
        pc2 = copy.deepcopy(pc1)
        if 'sem_seg' in cur_shape:
            shape_seg = cur_shape['sem_seg']
        else:
            shape_seg = cur_shape['inst_seg']
        # print(shape_seg.shape, pc1.shape, cur_shape.keys())
        # motion_seg = np.zeros_like(shape_seg, dtype=np.long)
        motion_seg = np.zeros((pc1.shape[0], ), dtype=np.long)
        chosen_num_moving_parts = 0
        gt_rotation = np.reshape(np.eye(3, dtype=np.float), (1, 3, 3)) * np.ones((pc1.shape[0], 1, 1), dtype=np.float)
        gt_transition = np.zeros((pc1.shape[0], 3, 1), dtype=np.float)
        if chosen_whether_mov_index == 1:
            # mov_inst_part_idx = list(inst_part_idx_to_part_idx_to_meta_info.keys())
            mov_inst_part_idx = [
                k for k in inst_part_idx_to_part_idx_to_meta_info if ((len(inst_part_idx_to_part_idx_to_meta_info[k]['rot_meta']) > 0
                or len(inst_part_idx_to_part_idx_to_meta_info[k]['trans_meta']) > 0) and
                inst_part_idx_to_part_idx_to_meta_info[k]['points_idx'].shape[0] > 30)]
            # mov
            if len(mov_inst_part_idx) > 0:
                chosen_num_moving_parts = np.random.choice(len(mov_inst_part_idx), 1).item() + 1
                chosen_num_moving_parts = min(chosen_num_moving_parts, 4)
                chosen_mov_inst_new_idx = np.random.choice(len(mov_inst_part_idx), chosen_num_moving_parts, replace=False)

                cur_mov_pts_number = 0
                for jj in range(chosen_mov_inst_new_idx.shape[0]):

                    jjjj = int(chosen_mov_inst_new_idx[jj].item())
                    mov_inst = mov_inst_part_idx[jjjj]
                    mov_inst_meta_info = inst_part_idx_to_part_idx_to_meta_info[mov_inst]
                    #
                    mov_points_idx = mov_inst_meta_info['points_idx']
                    rot_meta_info_list = mov_inst_meta_info['rot_meta']
                    trans_meta_info_list = mov_inst_meta_info['trans_meta']

                    if pc1.shape[0] - (cur_mov_pts_number + mov_points_idx.shape[0]) < 30:
                        break

                    if len(rot_meta_info_list) > 0:
                        for_chosen_rot_meta_idx_list = [ii for ii in range(len(rot_meta_info_list)) if
                                                        int(rot_meta_info_list[ii][0].item()) > 0]
                    else:
                        for_chosen_rot_meta_idx_list = []
                    if len(trans_meta_info_list) > 0:
                        for_chosen_trans_meta_idx_list = [ii for ii in range(len(trans_meta_info_list)) if
                                                          int(trans_meta_info_list[ii][0].item()) > 0]
                    else:
                        for_chosen_trans_meta_idx_list = []

                    if len(for_chosen_trans_meta_idx_list) > 0 and len(for_chosen_rot_meta_idx_list) > 0:
                        choose_trans_rot = int(np.random.choice(2, 1, p=[0.7, 0.3]).item())
                    elif len(for_chosen_trans_meta_idx_list) > 0:
                        choose_trans_rot = 1
                    elif len(for_chosen_rot_meta_idx_list) > 0:
                        choose_trans_rot = 0
                    else:
                        choose_trans_rot = 2

                    if choose_trans_rot == 0:
                        chosen_rot_meta_idx = np.random.choice(len(for_chosen_rot_meta_idx_list), 1).item()
                        chosen_rot_meta_idx = for_chosen_rot_meta_idx_list[chosen_rot_meta_idx]

                        rot_meta_info = rot_meta_info_list[chosen_rot_meta_idx]
                        rot_meta_type = int(rot_meta_info[0].item())
                        rot_base_point = rot_meta_info[1:4]
                        possi_rot_axis_vec = decode_rotation_info(rot_meta_type)
                        #
                        # todo: generate multiple random value at a time and choose from them gradually
                        if len(possi_rot_axis_vec) > 0:
                            chosen_rot_axis_idx = np.random.choice(len(possi_rot_axis_vec), 1).item()
                            rot_vec = possi_rot_axis_vec[chosen_rot_axis_idx]
                            bf_rotate_pos = pc1[mov_points_idx, :]
                            af_rotate_pos, applied_rotation, applied_transition = rotate_by_vec_pts(rot_vec, rot_base_point, bf_rotate_pos)
                            motion_seg[mov_points_idx] = jj + 1
                            cur_mov_pts_number += mov_points_idx.shape[0]
                            pc2[mov_points_idx, :] = af_rotate_pos
                            # multiplication between rotation matrices
                            gt_rotation[mov_points_idx] = np.matmul(np.reshape(applied_rotation, (1, 3, 3)), gt_rotation[mov_points_idx])
                            # gt_rotation[mov_points_idx, :, :] = np.reshape(applied_rotation, (1, 3, 3)).dot(gt_rotation[mov_points_idx, :, :])
                            gt_transition[mov_points_idx] = np.matmul(np.reshape(applied_rotation, (1, 3, 3)), gt_transition[mov_points_idx] - np.reshape(applied_transition, (1, 3, 1))) + np.reshape(applied_transition, (1, 3, 1))
                            # gt_transition[mov_points_idx, :, 0] = np.reshape(np.reshape(applied_rotation, (1, 3, 3)).dot(np.reshape(gt_transition[mov_points_idx, :, :], (mov_points_idx.shape[0], 3, 1))), (mov_points_idx.shape[0], 3))
                        else:
                            chosen_num_moving_parts -= 1
                            if chosen_num_moving_parts == 0:
                                chosen_whether_mov_index = 0
                    elif choose_trans_rot == 1:
                        chosen_trans_meta_idx = np.random.choice(len(for_chosen_trans_meta_idx_list), 1).item()
                        chosen_trans_meta_idx = for_chosen_trans_meta_idx_list[chosen_trans_meta_idx]

                        trans_meta_info = trans_meta_info_list[chosen_trans_meta_idx]
                        trans_meta_type = int(trans_meta_info[0].item())
                        possi_trans_axis_vec = self.decode_trans_dir(trans_meta_type)
                        if len(possi_trans_axis_vec) > 0:
                            chosen_trans_axis_idx = np.random.choice(len(possi_trans_axis_vec), 1).item()
                            trans_vec = possi_trans_axis_vec[chosen_trans_axis_idx]
                            bf_transit_pos = pc1[mov_points_idx, :]
                            af_transit_pos, applied_transition = self.transit_pos_by_transit_vec_dir(trans_pos=bf_transit_pos,
                                                                                 tdir=trans_vec)
                            motion_seg[mov_points_idx] = jj + 1
                            cur_mov_pts_number += mov_points_idx.shape[0]
                            pc2[mov_points_idx, :] = af_transit_pos
                            gt_transition[mov_points_idx] += np.reshape(applied_transition, (1, 3, 1))
                        else:
                            chosen_num_moving_parts -= 1
                            if chosen_num_moving_parts == 0:
                                chosen_whether_mov_index = 0
                    else:
                        chosen_num_moving_parts -= 1
                        if chosen_num_moving_parts == 0:
                            chosen_whether_mov_index = 0
            else:
                chosen_whether_mov_index = 0

        if chosen_whether_mov_index == 0:
            chosen_num_moving_parts = 0

        # pc1_af_rel_trans = pc1

        if self.args.equi_settings.global_rot == 1:
            R1 = generate_3d(smaller=True)
            pc1 = np.transpose(np.matmul(R1, np.transpose(pc1, [1, 0])), [1, 0])

        # pc1_af_rel = pc1_af_rel_trans

        # rel_rot_for_pc2 = R1.T
        # gt_rotation = np.matmul(gt_rotation, np.reshape(rel_rot_for_pc2, (1, 3, 3)))
        # flow12 = pc2 - pc1_af_rel
        # rd_num = np.random.choice(2, 1).item()
        # pc1_af_glb = pc1_af_rel
        pc2_af_glb = pc2 # we only have rel transformation here

        # gt_transform_vec = np.concatenate(
        #     [np.reshape(gt_rotation, (pc1.shape[0], 9)), np.reshape(gt_transition, (pc1.shape[0], 3))], axis=-1
        # )

        gt_rotation = np.zeros((pc1.shape[0], 3, 3), dtype=np.float)
        gt_rotation[:, 0, 0] = 1.; gt_rotation[:, 1, 1] = 0.; gt_rotation[:, 2, 2] = 1.
        gt_transition = np.zeros((pc1.shape[0], 3), dtype=np.float)

        # gt_rotation = np.eye(3, dtype=np.float)
        # gt_transition = np.zeros((3,), dtype=np.float)

        # gt_rotation = np.reshape(gt_rotation, (pc1.shape[0], 3, 3))
        gt_transition = np.reshape(gt_transition, (pc1.shape[0], 3, 1))
        gt_pose = np.concatenate(
            [gt_rotation, gt_transition], axis=-1
        )
        gt_pose = np.concatenate(
            [gt_pose, np.zeros((pc1.shape[0], 1, 4))], axis=1
        )



        assert gt_pose.shape[0] == pc1.shape[0] and gt_pose.shape[1] == 4 and gt_pose.shape[2] == 4

        permidx = np.random.permutation(pc2_af_glb.shape[0])[:self.npoints]
        pc2_af_glb = pc2_af_glb[permidx]
        # print(f"pc1.shape: {pc1.shape}")
        pc1 = pc1[permidx]
        # print(f"pc1.shape after permutation sampling: {pc1.shape}")
        shape_seg = shape_seg[permidx]
        gt_pose = gt_pose[permidx]

        # permidx = np.random.permutation(pc1_af_glb.shape[0])[:self.npoints]
        # pc1_af_glb = pc1_af_glb[permidx, :]
        # gt_transform_vec = gt_transform_vec[permidx, :]
        # pc1_af_glb = np.concatenate([pc1_af_glb, gt_transform_vec], axis=-1)
        # flow12 = flow12[permidx, :]
        # print(shape_seg)

        # gt_sehape_seg_idx
        shape_seg = self.reindex_shape_seg(shape_seg)

        # permidx2 = np.random.permutation(pc2_af_glb.shape[0])[:self.npoints]
        # pc2_af_glb = pc2_af_glb[permidx2, :]

        # permidx = np.random.permutation(pc2_af_glb.shape[0])[:self.npoints]

        rt_dict = {
            'pc': torch.from_numpy(pc1.astype(np.float32).T),
            'af_pc': torch.from_numpy(pc2_af_glb.astype(np.float32).T),
            'label': torch.from_numpy(shape_seg).long(),
            'pose': torch.from_numpy(gt_pose.astype(np.float32))
        }

        return rt_dict
        # return np.array([chosen_whether_mov_index], dtype=np.long), np.array([chosen_num_moving_parts], dtype=np.long), \
        #        pc1, pc2, flow12, shape_seg_masks, motion_seg_masks, pc1, pc1

    def __len__(self):
        if self.split == 'train' or (not self.real_test):
            return len(self.data)
        else:
            return self.data['pc1'].shape[0]
        # return self.data['pc1'].shape[0]

    def get_num_moving_parts_to_cnt(self):
        return self.num_mov_parts_to_cnt

    def reset_num_moving_parts_to_cnt(self):
        self.num_mov_parts_to_cnt = {}


if __name__ == '__main__':
    d = ModelNetDataset(root='../data/modelnet40_normal_resampled', split='test')
    print(d.shuffle)
    print(len(d))
    import time

    tic = time.time()
    for i in range(10):
        ps, cls = d[i]
    print(time.time() - tic)
    print(ps.shape, type(ps), cls)

    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
