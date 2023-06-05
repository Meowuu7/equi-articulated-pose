'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''

import os
import os.path
import json
import numpy as np
import math
import sys
import torch
import vgtk.so3conv.functional as L
from SPConvNets.datasets.part_transform import revoluteTransform
from SPConvNets.models.model_util import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import provider
from torch.utils import data
from SPConvNets.models.common_utils import *
from SPConvNets.datasets.data_utils import *
import scipy.io as sio
import copy
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


class MotionDataset(data.Dataset):
    def __init__(
            self, root="./data/MDV02", npoints=512, split='train', nmask=10, shape_type="laptop", args=None, global_rot=0
    ):
        super(MotionDataset, self).__init__()

        self.root = root
        self.npoints = npoints
        self.shape_type = shape_type
        self.shape_root = os.path.join(self.root, shape_type)
        self.args = args

        self.mesh_fn = "summary.obj"
        self.surface_to_seg_fn = "sfs_idx_to_dof_name_idx.npy"
        self.attribute_fn = "motion_attributes.json"

        self.global_rot = global_rot
        self.split = split
        self.rot_factor = self.args.equi_settings.rot_factor
        self.pre_compute_delta = self.args.equi_settings.pre_compute_delta

        print(f"no_articulation: {self.no_articulation}, equi_settings: {self.args.equi_settings.no_articulation}")

        self.train_ratio = 0.9

        dataset_root_path = "/share/zhangji/data/Motion_aligned_part_sampled_1"
        dataset_root_path = os.path.join(dataset_root_path, self.shape_type, self.split)
        self.dataset_root_path = dataset_root_path

        self.shape_folders = []

        shape_idxes = os.listdir(dataset_root_path)
        shape_idxes = sorted(shape_idxes)
        shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]

        self.pts_files = []
        for cur_idx in shape_idxes:
            cur_shape_folder = os.path.join(self.dataset_root_path, cur_idx)
            for i_shp in range(100):
                cur_shape_cur_idx_file = os.path.join(cur_shape_folder, f"sample_points{i_shp}.npy")
                self.pts_files.append(cur_shape_cur_idx_file)

        self.shape_idxes = self.pts_files

        ## todo: add a val split logic

        # self.shape_idxes = shape_idxes

        self.anchors = L.get_anchors(args.model.kanchor)

        # self.anchors = torch.from_numpy(L.get_anchors(args.model.kanchor)).cuda()
        self.kanchor = args.model.kanchor

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

    def get_rotation_from_anchor(self):
        ii = np.random.randint(0, self.kanchor, (1,)).item()
        ii = int(ii)
        R = self.anchors[ii]
        return R

    def get_whole_shape_by_idx(self, index):
        shp_idx = self.shape_idxes[index + 1]
        cur_folder = os.path.join(self.shape_root, shp_idx)

        cur_mesh_fn = os.path.join(cur_folder, self.mesh_fn)
        cur_surface_to_seg_fn = os.path.join(cur_folder, self.surface_to_seg_fn)
        cur_motion_attributes_fn = os.path.join(cur_folder, self.attribute_fn)

        cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
        cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn)
        cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

        sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles,
                                                                                    cur_triangles_to_seg_idx,
                                                                                    npoints=self.npoints)
        sampled_pcts = torch.from_numpy(sampled_pcts).float()
        return sampled_pcts

    def get_shape_by_idx(self, index):
        shp_idx = self.shape_idxes[index + 1]
        cur_folder = os.path.join(self.shape_root, shp_idx)

        cur_mesh_fn = os.path.join(cur_folder, self.mesh_fn)
        cur_surface_to_seg_fn = os.path.join(cur_folder, self.surface_to_seg_fn)
        cur_motion_attributes_fn = os.path.join(cur_folder, self.attribute_fn)

        cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
        cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn)
        cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

        sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles,
                                                                                    cur_triangles_to_seg_idx,
                                                                                    npoints=self.npoints)

        # get points for each segmentation/part
        tot_transformed_pts = []
        pts_nns = []
        for i_seg in range(len(cur_motion_attributes)):
            cur_seg_motion_info = cur_motion_attributes[i_seg]
            cur_seg_pts_idxes = np.array(seg_idx_to_sampled_pts[i_seg], dtype=np.long)
            cur_seg_pts = sampled_pcts[cur_seg_pts_idxes]
            pts_nns.append(cur_seg_pts.shape[0])

            tot_transformed_pts.append(cur_seg_pts)
        maxx_nn_pt = max(pts_nns)
        res_pts = []
        for i, trans_pts in enumerate(tot_transformed_pts):
            cur_seg_nn_pt = trans_pts.shape[0]
            cur_seg_center_pt = np.mean(trans_pts, axis=0, keepdims=True)
            if cur_seg_nn_pt < maxx_nn_pt:
                cur_seg_trans_pts = np.concatenate(
                    [trans_pts] + [cur_seg_center_pt for _ in range(maxx_nn_pt - cur_seg_nn_pt)], axis=0
                )
                res_pts.append(np.reshape(cur_seg_trans_pts, (1, maxx_nn_pt, 3)))
            else:
                res_pts.append(np.reshape(trans_pts, (1, maxx_nn_pt, 3)))

        res_pts = np.concatenate(res_pts, axis=0)
        res_pts = torch.from_numpy(res_pts).float()
        return res_pts

    def refine_triangle_idxes_by_seg_idx(self, seg_idx_to_triangle_idxes, cur_triangles):
        res_triangles = []
        cur_triangles_to_seg_idx = []
        for seg_idx in seg_idx_to_triangle_idxes:
            # if seg_idx == 0:
            #     continue
            cur_triangle_idxes = np.array(seg_idx_to_triangle_idxes[seg_idx], dtype=np.long)
            cur_seg_triangles = cur_triangles[cur_triangle_idxes]
            res_triangles.append(cur_seg_triangles)
            cur_triangles_to_seg_idx += [seg_idx for _ in range(cur_triangle_idxes.shape[0])]
        res_triangles = np.concatenate(res_triangles, axis=0)
        cur_triangles_to_seg_idx = np.array(cur_triangles_to_seg_idx, dtype=np.long)
        return res_triangles, cur_triangles_to_seg_idx

    def __getitem__(self, index):

        nparts = None
        if self.shape_type == "eyeglasses":
            nparts = 2
            nparts = None

        shp_idx = self.shape_idxes[index]

        cur_pts = np.load(self.shape_idxes[index], allow_pickle=True)
        cur_pts = np.transpose(cur_pts, (1, 0))

        pts_, labels_ = cur_pts[:, :3], cur_pts[:, 3]

        sampled_pcts = pts_
        boundary_pts = [np.min(sampled_pcts, axis=0), np.max(sampled_pcts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # all normalize into 0
        sampled_pcts = (sampled_pcts - center_pt.reshape(1, 3)) / length_bb

        tot_transformed_pts = sampled_pcts
        canon_transformed_pts = sampled_pcts
        tot_transformation_mtx = np.zeros((sampled_pcts.shape[0], 4, 4), dtype=np.float)
        tot_transformation_mtx[:, 0, 0] = 1.
        tot_transformation_mtx[:, 1, 1] = 1.
        tot_transformation_mtx[:, 2, 2] = 1.
        pts_to_seg_idx = labels_.astype(np.long)
        ''' Set pts to label index '''
        # seg_idx = 0
        # for cur_seg_idx in seg_idx_to_sampled_pts:
        #     cur_sampled_pts = np.array(seg_idx_to_sampled_pts[cur_seg_idx], dtype=np.long)
        #     pts_to_seg_idx[cur_sampled_pts] = seg_idx
        #     seg_idx += 1
        ''' Set trivial transformation matrix for segs '''
        # seg_idx = 1 if seg_idx == 0 else seg_idx
        seg_idx_to_exist = {}
        for i_pts in range(pts_.shape[0]):
            cur_pts_seg_label = int(labels_[i_pts].item())
            seg_idx_to_exist[cur_pts_seg_label] = 1
        seg_idx = len(seg_idx_to_exist)
        tot_transformation_mtx_segs = np.zeros((seg_idx, 4, 4), dtype=np.float)
        tot_transformation_mtx_segs[:, 0, 0] = 1.
        tot_transformation_mtx_segs[:, 1, 1] = 1.
        tot_transformation_mtx_segs[:, 2, 2] = 1.

        gt_pose = tot_transformation_mtx

        ''' Add global rotation '''
        if self.global_rot == 1 and (not (self.split == "train" and self.pre_compute_delta == 1)):
            if self.args.equi_settings.rot_anchors == 1:
                # just use a matrix from rotation anchors
                R1 = self.get_rotation_from_anchor()
            else:
                R1 = generate_3d(smaller=True)
            tot_transformed_pts = np.transpose(np.matmul(R1, np.transpose(tot_transformed_pts, [1, 0])), [1, 0])
            gt_pose = np.matmul(np.reshape(R1, (1, 3, 3)), gt_pose[:, :3, :])
            gt_pose = np.concatenate([gt_pose, np.zeros((tot_transformed_pts.shape[0], 1, 4), dtype=np.float)], axis=1)

        # tot_transformed_pts =
        # permidx = /
        # permidx = np.random.permutation(tot_transformed_pts.shape[0])[:self.npoints]
        # tot_transformed_pts = tot_transformed_pts[permidx]
        # shape_seg = pts_to_seg_idx[permidx]
        # gt_pose = np.zeros((self.npoints, ))

        cur_pc = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        cur_label = torch.from_numpy(pts_to_seg_idx).long()
        cur_pose = torch.from_numpy(gt_pose.astype(np.float32))
        cur_pose_segs = torch.from_numpy(tot_transformation_mtx_segs.astype(np.float32))
        cur_ori_pc = torch.from_numpy(sampled_pcts.astype(np.float32)).float()
        cur_canon_transformed_pts = torch.from_numpy(canon_transformed_pts.astype(np.float32)).float()

        fps_idx = farthest_point_sampling(cur_pc.unsqueeze(0), n_sampling=self.npoints)
        cur_pc = cur_pc[fps_idx]
        cur_label = cur_label[fps_idx]
        cur_pose = cur_pose[fps_idx]
        # cur_pose = cur_pose[fps_idx]
        cur_ori_pc = cur_ori_pc[fps_idx]
        cur_canon_transformed_pts = cur_canon_transformed_pts[fps_idx]

        idx_arr = np.array([index], dtype=np.long)
        idx_arr = torch.from_numpy(idx_arr).long()

        rt_dict = {
            'pc': cur_pc.contiguous().transpose(0, 1).contiguous(),
            'af_pc': cur_pc.contiguous().transpose(0, 1).contiguous(),
            'ori_pc': cur_ori_pc.contiguous().transpose(0, 1).contiguous(),
            'canon_pc': cur_canon_transformed_pts, #.contiguous().transpose(0, 1).contiguous(),
            'label': cur_label,
            'pose': cur_pose,
            'pose_segs': cur_pose_segs,
            'idx': idx_arr
        }

        return rt_dict
        # return np.array([chosen_whether_mov_index], dtype=np.long), np.array([chosen_num_moving_parts], dtype=np.long), \
        #        pc1, pc2, flow12, shape_seg_masks, motion_seg_masks, pc1, pc1

    def __len__(self):
        return len(self.shape_idxes)

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
