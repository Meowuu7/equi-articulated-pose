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
from scipy.spatial.transform import Rotation as sciR
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
        self.no_articulation = self.args.equi_settings.no_articulation
        self.pre_compute_delta = self.args.equi_settings.pre_compute_delta
        self.use_multi_sample = self.args.equi_settings.use_multi_sample
        self.n_samples = self.args.equi_settings.n_samples if self.use_multi_sample == 1 else 1

        self.dataset_root = "/share/zhangji/data/Sapien_aligned_part_sampled/drawers"
        self.dataset_root = os.path.join(self.dataset_root, self.split)

        if self.pre_compute_delta == 1 and self.split == 'train':
            # self.use_multi_sample = False
            self.use_multi_sample = 0
            self.n_samples = 1

        print(f"no_articulation: {self.no_articulation}, equi_settings: {self.args.equi_settings.no_articulation}")

        self.train_ratio = 0.9

        self.shape_folders = []

        self.pts_folders = []
        self.cfg_folders = []

        shape_idxes = os.listdir(self.dataset_root)
        for shp_idx in shape_idxes:
            cur_shp_folder = os.path.join(self.dataset_root, shp_idx)
            for i in range(self.n_samples):
                pts_fn = os.path.join(cur_shp_folder, f"sample_points{i}.npy")
                cfg_fn = os.path.join(cur_shp_folder, f"sample_cfg{i}.npy")
                self.pts_folders.append(pts_fn)
                self.cfg_folders.append(cfg_fn)

        ''' Set shape indexes --- the number of shapes but not the number of all samples '''
        self.shape_idxes = shape_idxes

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

        # n_samples_per_instance = 100

        def get_seg_labels_to_pts_idxes(seg_labels):
            seg_label_to_pts_idxes = {}
            for i_pts in range(seg_labels.shape[0]):
                cur_pts_label = int(seg_labels[i_pts].item())
                if cur_pts_label not in seg_label_to_pts_idxes:
                    seg_label_to_pts_idxes[cur_pts_label] = [i_pts]
                else:
                    seg_label_to_pts_idxes[cur_pts_label].append(i_pts)
            for cur_seg_label in seg_label_to_pts_idxes:
                seg_label_to_pts_idxes[cur_seg_label] = np.array(seg_label_to_pts_idxes[cur_seg_label], dtype=np.long)

            return seg_label_to_pts_idxes

        # nparts = None
        # if self.shape_type == "eyeglasses":
        #     nparts = 2
        #     nparts = None

        shape_index = index
        cur_pts_fn = self.pts_folders[shape_index]
        cur_cfg_fn = self.cfg_folders[shape_index]

        sample_idx = index % 100

        cur_pts = np.load(cur_pts_fn, allow_pickle=True)
        # npts x 4
        cur_pts = np.transpose(cur_pts, (1, 0))
        cur_pts, cur_labels = cur_pts[:, :3], cur_pts[:, 3].astype(np.long)
        cur_seg_label_to_pts_idxes = get_seg_labels_to_pts_idxes(cur_labels)

        # boundary_pts = [np.min(cur_pts, axis=0), np.max(cur_pts, axis=0)]
        # center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        #
        # # all normalize into 0
        # cur_pts = (cur_pts - center_pt.reshape(1, 3)) / length_bb

        cur_cfg = np.load(cur_cfg_fn, allow_pickle=True).item()
        cur_seg_label_to_motion_attr = {}
        for seg_name in cur_cfg:
            motion_attrs = cur_cfg[seg_name]
            # cur_seg_label = int(motion_attrs['label'].item())
            cur_seg_label = int(seg_name[-1])
            cur_motion_type = motion_attrs['motion_type']
            if cur_motion_type != 'none_motion':
                cur_dir = motion_attrs['axis']['direction']
                cur_limit = motion_attrs['limit']
                cur_limit_a, cur_limit_b = float(cur_limit['a']), float(cur_limit['b'])
                cur_state = float(motion_attrs['state'])
                cur_seg_label_to_motion_attr[cur_seg_label] = {
                    'dir': cur_dir, 'a': cur_limit_a, 'b': cur_limit_b, 'state': cur_state
                }

        ori_pts = np.zeros_like(cur_pts)
        canon_transformed_pts = np.zeros_like(cur_pts)
        part_npcs = np.zeros_like(cur_pts)
        npcs_trans = np.zeros_like(cur_pts)

        cur_sample_motion_states = DRAWER_COMBINATIONS[sample_idx]
        cur_sample_motion_states = sorted(cur_sample_motion_states, reverse=True)
        # print(f"cur_sample_motion_states: {cur_sample_motion_states}")
        canon_motion_states = DRAWER_COMBINATIONS[len(DRAWER_COMBINATIONS) // 3]
        consume_idx = 0
        part_ref_trans = []
        part_state_trans = []
        part_ref_trans_bbox = []
        part_state_trans_bbox = []
        pose = np.zeros((ori_pts.shape[0], 4, 4), dtype=np.float)
        pose_segs = np.zeros((4, 4, 4), dtype=np.float)
        pose[:, 0, 0] = 1.
        pose[:, 1, 1] = 1.
        pose[:, 2, 2] = 1.
        pose_segs[:, 0, 0] = 1.
        pose_segs[:, 1, 1] = 1.
        pose_segs[:, 2, 2] = 1.
        part_axis = []

        # npcs_trans = []

        # part_npcs = []

        for cur_seg_label in range(4):
            cur_pts_idxes = cur_seg_label_to_pts_idxes[cur_seg_label]
            if cur_seg_label not in cur_seg_label_to_motion_attr:
                cur_seg_ori_pts = cur_pts[cur_pts_idxes]
                ori_pts[cur_pts_idxes] = cur_pts[cur_pts_idxes]
                canon_transformed_pts[cur_pts_idxes] = cur_pts[cur_pts_idxes]
                cur_ref_trans = np.zeros((1, 3), dtype=np.float32)
                part_ref_trans.append(cur_ref_trans)
                cur_state_trans = np.zeros((1, 3), dtype=np.float32)
                part_state_trans.append(cur_state_trans)
                part_ref_trans_bbox.append(cur_ref_trans)
                part_state_trans_bbox.append(cur_state_trans)
            else:
                # print(cur_seg_label)
                cur_state = cur_seg_label_to_motion_attr[cur_seg_label]['state']
                cur_dir = cur_seg_label_to_motion_attr[cur_seg_label]['dir']
                cur_a = cur_seg_label_to_motion_attr[cur_seg_label]['a']
                cur_b = cur_seg_label_to_motion_attr[cur_seg_label]['b']
                cur_seg_ori_pts = cur_pts[cur_pts_idxes] - np.reshape(cur_state * cur_dir, (1, 3))
                ori_pts[cur_pts_idxes] = cur_seg_ori_pts

                cur_seg_motion_state = float(cur_sample_motion_states[consume_idx]) # cur_seg_motion_state
                # print(f"cur_seg_label: {cur_seg_label}, cur_motion_state: {cur_seg_motion_state}")
                canon_seg_motion_state = float(canon_motion_states[consume_idx]) # cur_seg_motion_state
                # cur_transformed_pts = cur_seg_ori_pts + np.reshape(cur_seg_motion_state * cur_dir, (1, 3))
                # cur_transformed_pts = cur_seg_ori_pts + np.reshape((cur_seg_motion_state * (cur_b - cur_a) + cur_a + 0.1) * cur_dir , (1, 3)) #
                # cur_transformed_pts = cur_seg_ori_pts + np.reshape((cur_seg_motion_state * 3.0 + 1.0) * cur_dir , (1, 3))
                cur_transformed_pts = cur_seg_ori_pts + np.reshape((cur_seg_motion_state) * cur_dir , (1, 3))
                # cur_transformed_pts = cur_seg_ori_pts + np.reshape((cur_seg_motion_state * (cur_b - cur_a) + cur_a) * cur_dir , (1, 3))

                cur_part_state_trans = np.reshape((cur_seg_motion_state) * cur_dir , (1, 3))
                part_state_trans.append(cur_part_state_trans)

                part_axis.append(np.reshape(cur_dir, (1, 3)))

                # if cur_seg_label == 0:
                #     m_dir = np.array([0.0, -1.0, 0.0], dtype=np.float)
                # elif cur_seg_label == 1:
                #     m_dir = np.array([0.0, 0.0, 0.0], dtype=np.float)
                # elif cur_seg_label == 2:
                #     m_dir = np.array([0.0, 1.0, 0.0], dtype=np.float)
                # else:
                #     m_dir = np.array([0.0, 0.0, 0.0], dtype=np.float)

                # cur_transformed_pts = cur_transformed_pts + np.reshape(1.0 * m_dir, (1, 3))

                consume_idx += 1
                # current transformed points
                cur_pts[cur_pts_idxes] = cur_transformed_pts # register current transformed points in...

                # cur_part_ref_trans = np.reshape((cur_a + cur_b) / 2. * cur_dir, (1, 3))
                cur_part_ref_trans = np.reshape(canon_seg_motion_state * cur_dir, (1, 3))
                # cur_canon_transformed_pts = cur_seg_ori_pts + np.reshape((cur_a + cur_b) / 2. * cur_dir, (1, 3))
                cur_canon_transformed_pts = cur_seg_ori_pts + cur_part_ref_trans
                canon_transformed_pts[cur_pts_idxes] = cur_canon_transformed_pts
                part_ref_trans.append(cur_part_ref_trans)

                ''' Get points state translations with centralized boudning box '''
                # rot_pts: n_pts_part x 3
                canon_rot_pts_minn = np.min(cur_canon_transformed_pts[:, :3], axis=0)
                canon_rot_pts_maxx = np.max(cur_canon_transformed_pts[:, :3], axis=0)
                canon_rot_pts_bbox_center = (canon_rot_pts_minn + canon_rot_pts_maxx) / 2.
                cur_part_ref_trans_bbox = cur_part_ref_trans[0] - canon_rot_pts_bbox_center
                part_ref_trans_bbox.append(np.reshape(cur_part_ref_trans_bbox, (1, 3)))

                ''' Get points state translations with centralized bouding box '''
                # rot_pts: n_pts_part x 3
                rot_pts_minn = np.min(cur_transformed_pts[:, :3], axis=0)
                rot_pts_maxx = np.max(cur_transformed_pts[:, :3], axis=0)
                rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                cur_part_state_trans_bbox = cur_part_state_trans[0] - rot_pts_bbox_center
                part_state_trans_bbox.append(np.reshape(cur_part_state_trans_bbox, (1, 3)))

                pose[cur_pts_idxes, :3, 3] = cur_part_state_trans[0]
                pose_segs[cur_seg_label, :3, 3]  = cur_part_state_trans[0]

            tight_w = max(cur_seg_ori_pts[:, 0]) - min(cur_seg_ori_pts[:, 0])
            tight_l = max(cur_seg_ori_pts[:, 1]) - min(cur_seg_ori_pts[:, 1])
            tight_h = max(cur_seg_ori_pts[:, 2]) - min(cur_seg_ori_pts[:, 2])
            norm_factor = np.sqrt(1) / np.sqrt(tight_w ** 2 + tight_l ** 2 + tight_h ** 2)
            corner_pt_left = np.amin(cur_seg_ori_pts, axis=0, keepdims=True)
            corner_pt_right = np.amax(cur_seg_ori_pts, axis=0, keepdims=True)
            cur_corner_pts = [corner_pt_left, corner_pt_right]

            cur_npcs = (cur_seg_ori_pts[:, :3] - cur_corner_pts[0]) * norm_factor - 0.5 * (
                    cur_corner_pts[1] - cur_corner_pts[0]) * norm_factor
            # part_npcs.append(cur_npcs)

            part_npcs[cur_pts_idxes] = cur_npcs

            cur_npcs_trans = -cur_corner_pts[0] - 0.5 * (
                    cur_corner_pts[1] - cur_corner_pts[0])

            # npcs_trans.append(cur_npcs_trans)

            npcs_trans[cur_pts_idxes] = cur_npcs_trans

        # npcs_pts = np.concatenate(part_npcs, axis=0)
        npcs_pts = part_npcs
        # canon_transformed_pts = npcs_pts
        # npcs_trans = np.concatenate(npcs_trans, axis=0)

        ''' Get per-point pose and per-part pose '''
        # pose = np.zeros((ori_pts.shape[0], 4, 4), dtype=np.float)
        # pose_segs = np.zeros((4, 4, 4), dtype=np.float)
        part_state_orts = np.zeros((4, 3, 3), dtype=np.float)
        part_ref_rots = np.zeros((4, 3, 3), dtype=np.float)

        part_state_trans = np.concatenate(part_state_trans, axis=0)
        part_ref_trans = np.concatenate(part_ref_trans, axis=0)
        part_state_trans_bbox = np.concatenate(part_state_trans_bbox, axis=0)
        part_ref_trans_bbox = np.concatenate(part_ref_trans_bbox, axis=0)

        part_axis = np.concatenate(part_axis, axis=0)

        # # part_state_orts[:]
        # for cur_seg_label in cur_seg_label_to_pts_idxes:
        #     cur_pts_idxes = cur_seg_label_to_pts_idxes[cur_seg_label]
        #     if cur_seg_label in cur_seg_label_to_motion_attr:
        #         cur_state = cur_seg_label_to_motion_attr[cur_seg_label]['state']
        #         cur_dir = cur_seg_label_to_motion_attr[cur_seg_label]['dir']
        #         cur_a = cur_seg_label_to_motion_attr[cur_seg_label]['a']
        #         cur_b = cur_seg_label_to_motion_attr[cur_seg_label]['b']
        #         cur_trans = cur_state * cur_dir
        #         pose[cur_pts_idxes, :3, 3] = np.reshape(cur_trans, (1, 3))
        #         pose_segs[cur_seg_label, :3, 3] = cur_trans
        #         # part_state_orts[cur_seg_label, :3, 3] = cur_trans
        #
        #         cur_ref_trans = cur_dir * (cur_a + cur_b) / 2.
        #         # part_ref_rots[cur_seg_label, :3, 3] = cur_ref_trans

        #
        ''' Set first three dimension to identity '''
        # pose[:, 0, 0] = 1.;  pose[:, 1, 1] = 1.;  pose[:, 2, 2] = 1.
        # pose_segs[:, 0, 0] = 1.
        # pose_segs[:, 1, 1] = 1.
        # pose_segs[:, 2, 2] = 1.
        part_state_orts[:, 0, 0] = 1.
        part_state_orts[:, 1, 1] = 1.
        part_state_orts[:, 2, 2] = 1.
        part_ref_rots[:, 0, 0] = 1.
        part_ref_rots[:, 1, 1] = 1.
        part_ref_rots[:, 2, 2] = 1.
        # part_ref_trans = np.zeros((part_ref_rots.shape[0], 3), dtype=np.float)

        ''' Add global rotation '''
        if self.global_rot == 1 and (not (self.split == "train" and self.pre_compute_delta == 1)):
            if self.args.equi_settings.rot_anchors == 1:
                # just use a matrix from rotation anchors
                R1 = self.get_rotation_from_anchor()
            else:
                # R1 = generate_3d(smaller=True)
                rotation_angle = sciR.random().as_matrix()
                rotation_matrix = rotation_angle[:3, :3]
                R1 = rotation_matrix
            # rotate transformed points
            cur_pts = np.transpose(np.matmul(R1, np.transpose(cur_pts, [1, 0])), [1, 0])
            pose = np.matmul(np.reshape(R1, (1, 3, 3)), pose[:, :3, :])
            pose = np.concatenate([pose, np.zeros((cur_pts.shape[0], 1, 4), dtype=np.float)], axis=1)

            pose_segs[:, :3, :] = np.matmul(np.reshape(R1, (1, 3, 3)), pose_segs[:, :3, :])
            part_state_orts[:, :3, :] = np.matmul(np.reshape(R1, (1, 3, 3)), part_state_orts[:, :3, :])

            part_axis = np.transpose(np.matmul(R1, np.transpose(part_axis, (1, 0))), (1, 0))

            part_state_trans = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(part_state_trans, (part_state_trans.shape[0], 3, 1)))
            part_state_trans = np.reshape(part_state_trans, (part_state_trans.shape[0], 3))
            part_state_trans_bbox = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(part_state_trans_bbox, (part_state_trans_bbox.shape[0], 3, 1)))
            part_state_trans_bbox = np.reshape(part_state_trans_bbox, (part_state_trans_bbox.shape[0], 3))


        # tot_transformed_pts =
        # permidx = /
        # permidx = np.random.permutation(tot_transformed_pts.shape[0])[:self.npoints]
        # tot_transformed_pts = tot_transformed_pts[permidx]
        # shape_seg = pts_to_seg_idx[permidx]
        # gt_pose = np.zeros((self.npoints, ))

        # # todo: change gt-part-pose!!!! -> np.mean for the mean of all points
        # cur_pts = cur_pts - np.mean(cur_pts, axis=0, keepdims=True) # centralize points?

        ''' Centralize transformed points '''
        # ##### Get boundaries of ori_pts #####
        # boundary_pts = [np.min(ori_pts, axis=0), np.max(ori_pts, axis=0)]
        boundary_pts = [np.min(ori_pts, axis=0), np.max(ori_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # all normalize into 0
        cur_pts = (cur_pts - center_pt.reshape(1, 3)) / length_bb #
        pose[:, :3, 3] = (pose[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb
        pose_segs[:, :3, 3] = (pose_segs[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb
        part_state_trans = (part_state_trans - center_pt.reshape(1, 3)) / length_bb
        part_state_trans_bbox = part_state_trans_bbox / length_bb

        ''' Centralize points in the canonical state '''
        canon_boundary_pts = [np.min(ori_pts, axis=0), np.max(ori_pts, axis=0)]
        canon_center_pt = (canon_boundary_pts[0] + canon_boundary_pts[1]) / 2
        canon_length_bb = np.linalg.norm(canon_boundary_pts[0] - canon_boundary_pts[1])

        # all normalize into 0
        canon_transformed_pts = (canon_transformed_pts - canon_center_pt.reshape(1, 3)) / canon_length_bb  #


        # [:, :3, 3] = (pose[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb
        # pose_segs[:, :3, 3] = (pose_segs[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb
        part_ref_trans = (part_ref_trans - canon_center_pt.reshape(1, 3)) / canon_length_bb
        part_ref_trans_bbox = part_ref_trans_bbox / canon_length_bb


        cur_pc = torch.from_numpy(cur_pts.astype(np.float32)).float()
        tot_transformed_pts = torch.from_numpy(cur_pts.astype(np.float32)).float()
        cur_label = torch.from_numpy(cur_labels).long()
        tot_label = torch.from_numpy(cur_labels).long()
        cur_pose = torch.from_numpy(pose.astype(np.float32))
        cur_pose_segs = torch.from_numpy(pose_segs.astype(np.float32))
        cur_ori_pc = torch.from_numpy(ori_pts.astype(np.float32)).float()
        cur_canon_transformed_pts = torch.from_numpy(canon_transformed_pts.astype(np.float32)).float()
        cur_part_state_rots = torch.from_numpy(part_state_orts.astype(np.float32)).float()
        cur_part_ref_rots = torch.from_numpy(part_ref_rots.astype(np.float32)).float()
        cur_part_ref_trans = torch.from_numpy(part_ref_trans.astype(np.float32)).float()
        part_ref_trans_bbox = torch.from_numpy(part_ref_trans_bbox.astype(np.float32)).float()
        part_state_trans_bbox = torch.from_numpy(part_state_trans_bbox.astype(np.float32)).float()
        cur_part_axis = torch.from_numpy(part_axis.astype(np.float32)).float()

        fps_idx = farthest_point_sampling(cur_pc.unsqueeze(0), n_sampling=self.npoints)
        fps_idx_oorr = farthest_point_sampling(cur_pc.unsqueeze(0), n_sampling=4096)
        tot_transformed_pts = tot_transformed_pts[fps_idx_oorr]
        tot_label = tot_label[fps_idx_oorr]
        cur_pc = cur_pc[fps_idx]
        cur_label = cur_label[fps_idx]
        cur_pose = cur_pose[fps_idx]
        # cur_pose = cur_pose[fps_idx]

        cur_ori_pc = cur_ori_pc[fps_idx]
        cur_oorr_canon_transformed_pts = cur_canon_transformed_pts[fps_idx_oorr]
        cur_canon_transformed_pts = cur_canon_transformed_pts[fps_idx]

        idx_arr = np.array([index], dtype=np.long)
        idx_arr = torch.from_numpy(idx_arr).long()

        # rt_dict = {
        #     'pc': torch.from_numpy(tot_transformed_pts.astype(np.float32).T),
        #     'af_pc': torch.from_numpy(tot_transformed_pts.astype(np.float32).T),
        #     'label': torch.from_numpy(shape_seg).long(),
        #     'pose': torch.from_numpy(gt_pose.astype(np.float32))
        # }

        rt_dict = {
            'pc': cur_pc.contiguous().transpose(0, 1).contiguous(),
            'af_pc': cur_pc.contiguous().transpose(0, 1).contiguous(),
            'ori_pc': cur_ori_pc.contiguous().transpose(0, 1).contiguous(),
            'canon_pc': cur_canon_transformed_pts, #.contiguous().transpose(0, 1).contiguous(),
            'oorr_pc': tot_transformed_pts.contiguous().transpose(0, 1).contiguous(),
            'oorr_canon_pc': cur_oorr_canon_transformed_pts.contiguous(),
            'label': cur_label,
            'oorr_label': tot_label,
            'pose': cur_pose,
            'pose_segs': cur_pose_segs,
            'part_state_rots': cur_part_state_rots,
            'part_ref_rots': cur_part_ref_rots,
            'part_ref_trans': cur_part_ref_trans,
            'idx': idx_arr,
            'part_state_trans_bbox': part_state_trans_bbox,
            'part_ref_trans_bbox': part_ref_trans_bbox,
            'npcs_trans': npcs_trans,
            'part_axis': cur_part_axis
        }

        return rt_dict
        # return np.array([chosen_whether_mov_index], dtype=np.long), np.array([chosen_num_moving_parts], dtype=np.long), \
        #        pc1, pc2, flow12, shape_seg_masks, motion_seg_masks, pc1, pc1

    def __len__(self):
        return len(self.shape_idxes) * self.n_samples

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
