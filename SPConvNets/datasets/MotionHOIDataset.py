

import os
import os.path
import json
import numpy as np
import math
import sys
import torch
import vgtk.so3conv.functional as L
# import vgtk.pc as pctk
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


class MotionDataset(data.Dataset):
    def __init__(
            self, root="./data/MDV02", npoints=512, split='train', nmask=10, shape_type="laptop", args=None, global_rot=0
    ):
        super(MotionDataset, self).__init__()

        # self.root = root #
        self.root = "./data/HOI4D"
        self.npoints = npoints
        self.shape_type = shape_type
        self.shape_root = os.path.join(self.root, shape_type)
        self.args = args

        self.mesh_fn = "summary.obj"
        self.surface_to_seg_fn = "sfs_idx_to_dof_name_idx.npy"
        self.attribute_fn = "motion_attributes.json"

        self.obj_folder_name = "objs"
        self.mobility_fn = "mobility_v2.json"
        self.res_fn = "result.json"

        self.global_rot = global_rot
        self.split = split
        self.rot_factor = self.args.equi_settings.rot_factor
        self.pre_compute_delta = self.args.equi_settings.pre_compute_delta
        self.use_multi_sample = self.args.equi_settings.use_multi_sample
        self.n_samples = self.args.equi_settings.n_samples if self.use_multi_sample == 1 else 1
        self.partial = self.args.equi_settings.partial

        if self.pre_compute_delta == 1 and self.split == 'train':
            # self.use_multi_sample = False
            self.use_multi_sample = 0
            self.n_samples = 1

        print(f"no_articulation: {self.no_articulation}, equi_settings: {self.args.equi_settings.no_articulation}")

        self.train_ratio = 0.9

        self.shape_folders = []

        shape_idxes = os.listdir(self.shape_root)
        shape_idxes = sorted(shape_idxes)
        shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "."]


        if self.shape_type == "safe":
            not_ok_shape_idxes = [46, 41, 48, 40, 38, 36, 37, 39, 42, 45, 44, 35, 50, 34]
            not_ok_shape_idxes = ["%.3d" % iii for iii in not_ok_shape_idxes]
            not_ok_shape_idx_to_va = {str(iii): 1 for iii in not_ok_shape_idxes}
            shape_idxes = [si for si in shape_idxes if si not in not_ok_shape_idx_to_va]

        train_nns = int(len(shape_idxes) * self.train_ratio)

        if self.split == "train":
            shape_idxes = shape_idxes[:train_nns]
        else:
            shape_idxes = shape_idxes[train_nns:]

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


        shape_index, sample_index = index // self.n_samples, index % self.n_samples

        shp_idx = self.shape_idxes[shape_index]

        cur_folder = os.path.join(self.shape_root, shp_idx)
        cur_fn_folder = cur_folder

        mobility = open(os.path.join(cur_fn_folder, self.mobility_fn), "r")
        mobility = json.load(mobility)
        part_idx_to_mobility = {}

        idx_to_objs = {}
        res_json = open(os.path.join(cur_fn_folder, self.res_fn), "r")
        res_json = json.load(res_json)
        part_idx_to_obj_name = {}
        part_idx_to_pcts = {}
        # nn_part_idx_to_old_part_idx = {}
        part_idx_to_nn_part_idx = {}
        sampled_pcts = []
        pts_to_seg_idx = []
        nn_part_idx = 0

        def get_part_idx_to_obj_name(curr, part_idx_to_obj_name):
            if "objs" not in curr:
                children = curr["children"]
                for child_frame in children:
                    part_idx_to_obj_name = get_part_idx_to_obj_name(child_frame, part_idx_to_obj_name)
            else:
                cur_idxx = int(curr["id"])
                part_idx_to_obj_name[cur_idxx] = curr["objs"]
            return part_idx_to_obj_name

        for frame in res_json:
            cur_frame = frame
            part_idx_to_obj_name = get_part_idx_to_obj_name(cur_frame, part_idx_to_obj_name)

        for part_idx in part_idx_to_obj_name:
            # get object file names for this part
            cur_part_obj_names = part_idx_to_obj_name[part_idx]
            cur_part_pcts = []
            for tmp_obj_name in cur_part_obj_names:
                tmp_obj_file_path = os.path.join(cur_fn_folder, "objs", tmp_obj_name + ".obj")
                # cur_vertices, cur_meshes = load_vertices_triangles(tmp_obj_file_path)
                # cur_obj_sampled_pcts = sample_pts_from_mesh_v2(vertices=cur_vertices, triangles=cur_meshes, pts_per_area=1)
                cur_obj_sampled_pcts = np.load(os.path.join(cur_fn_folder, "objs", tmp_obj_name + "_down_pts.npy"), allow_pickle=True)
                # print(f"vert: {cur_vertices.shape}, meshes: {cur_meshes.shape}")
                # cur_obj_sampled_pcts = cur_vertices
                # print(cur_obj_sampled_pcts .shape)
                cur_part_pcts.append(cur_obj_sampled_pcts)
            cur_part_pcts = np.concatenate(cur_part_pcts, axis=0)
            # part_idx_to_pcts[part_idx] = cur_part_pcts
            part_idx_to_pcts[nn_part_idx] = cur_part_pcts
            pts_to_seg_idx += [nn_part_idx for _ in range(cur_part_pcts.shape[0])]
            part_idx_to_nn_part_idx[part_idx] = 0 + nn_part_idx
            nn_part_idx += 1
            sampled_pcts.append(cur_part_pcts)

        # sampled_pcts: npoints x 3; Get sampled points
        sampled_pcts = np.concatenate(sampled_pcts, axis=0)
        # points to segmentation index?
        pts_to_seg_idx = np.array(pts_to_seg_idx, dtype=np.long)
        seg_idx_to_sampled_pts = part_idx_to_pcts

        if self.partial == 1:
            sampled_pcts[:, 2] = 0.

        ''' Add global rotation '''
        if self.global_rot == 1 and (not (self.split == "train" and self.pre_compute_delta == 1)):
            if self.args.equi_settings.rot_anchors == 1:
                # just use a matrix from rotation anchors
                R1 = self.get_rotation_from_anchor()
            else:
                rotation_angle = sciR.random().as_matrix()
                rotation_matrix = rotation_angle[:3, :3]
                R1 = rotation_matrix
        elif self.global_rot == 2:
            R1 = self.common_R
        else:
            R1 = np.eye(3, dtype=np.float32)
        ''' Add global rotation '''

        boundary_pts = [np.min(sampled_pcts, axis=0), np.max(sampled_pcts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # all normalize into 0
        sampled_pcts = (sampled_pcts - center_pt.reshape(1, 3)) / length_bb

        for cur_part_idx in part_idx_to_pcts:
            part_idx_to_pcts[cur_part_idx] = (part_idx_to_pcts[cur_part_idx] - center_pt.reshape(1, 3)) / length_bb

        part_state_rots, part_ref_rots = [], []
        part_ref_trans = []
        part_ref_trans_bbox = []
        part_state_trans_bbox = []

        part_axis = []
        part_pv_offset = []
        part_pv_point = []
        part_angles = []
        
        
        tot_transformed_pts = []
        tot_transformation_mtx = []
        tot_transformation_mtx_segs = []
        canon_transformed_pts = []

        nn_part_idx_to_mob_attrs = {}

        for sub_mob in mobility:
            if "joint" in sub_mob:
                cur_joint_type = sub_mob["joint"]
                cur_joint_data = sub_mob["jointData"]

                if cur_joint_type == "铰链（旋转）":
                    cur_joint_dir = cur_joint_data["axis"]["direction"]
                    cur_origin_point = cur_joint_data["axis"]["origin"]
                    cur_origin_point = np.array(cur_origin_point, dtype=np.float)
                    cur_limit = cur_joint_data["limit"]
                    # cur_a, cur_b = float(cur_limit["a"]), float(cur_limit["b"])
                    cur_origin_point = (cur_origin_point - center_pt) / length_bb
                    if "parts" in sub_mob:
                        cur_mob_parts = sub_mob["parts"]
                        for cur_mob_part in cur_mob_parts:
                            cur_part_idx = cur_mob_part["id"]
                            cur_nn_part_idx = part_idx_to_nn_part_idx[cur_part_idx]

                            cur_nn_part_idx_to_mob_attrs = {
                                "center": cur_origin_point,
                                "axis": cur_joint_dir
                            }
                            nn_part_idx_to_mob_attrs[cur_nn_part_idx] = cur_nn_part_idx_to_mob_attrs


        for i_seg in range(nn_part_idx):
            cur_seg_pts = part_idx_to_pcts[i_seg]
            if i_seg in nn_part_idx_to_mob_attrs:
                cur_part_mob_attrs = nn_part_idx_to_mob_attrs[i_seg]
                center = cur_part_mob_attrs["center"]
                axis = cur_part_mob_attrs["axis"]

                theta = ((0.5 / self.n_samples * sample_index) * np.pi)
                theta = theta - (45. / 180.) * np.pi


                ''' Get joint aixs direction and aix offset '''
                part_axis.append(np.reshape(axis, (1, 3)))
                center_pt_offset = center - np.sum(axis * center, axis=0, keepdims=True) * axis
                center_pt_offset = np.sqrt(np.sum(center_pt_offset ** 2, axis=0))
                part_pv_offset.append(center_pt_offset)
                ''' Get joint aixs direction and axis offset '''

                part_pv_point.append(np.reshape(center, (1, 3)))
                part_angles.append(theta)

                rot_pts, transformation_mtx = revoluteTransform(cur_seg_pts, center, axis, theta)
                transformation_mtx = np.transpose(transformation_mtx, (1, 0))

                rot_pts[:, :3] = np.matmul(np.reshape(R1, (1, 3, 3)),
                                            np.reshape(rot_pts[:, :3], (rot_pts.shape[0], 3, 1)))[:, :3, 0]
                transformation_mtx[:3] = np.matmul(R1, transformation_mtx[:3])
                ''' Transform points via revolute transformation '''

                ''' Get points state translations with centralized boudning box '''
                # rot_pts: n_pts_part x 3
                # rot pts minn; rot pts maxx
                rot_pts_minn = np.min(rot_pts[:, :3], axis=0)
                rot_pts_maxx = np.max(rot_pts[:, :3], axis=0)
                rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                # print(f"transformation_mtx: {transformation_mtx[:3, 3].shape}, rot_pts_bbox_center: {rot_pts_bbox_center.shape}")
                cur_part_state_trans_bbox = transformation_mtx[:3, 3] - rot_pts_bbox_center
                part_state_trans_bbox.append(np.reshape(cur_part_state_trans_bbox, (1, 3)))

                ''' Set canonical angle via shape type '''
                canon_theta = 0.5 * np.pi
                if self.shape_type in ["laptop", "eyeglasses", "safe"]:
                    canon_theta = 0.0
                canon_rot_pts, canon_transformation_mtx = revoluteTransform(cur_seg_pts, center, axis, canon_theta)
                canon_transformation_mtx = np.transpose(canon_transformation_mtx, (1, 0))

                ''' Get points state translations with centralized boudning box '''
                # rot_pts: n_pts_part x 3
                canon_rot_pts_minn = np.min(canon_rot_pts[:, :3], axis=0)
                canon_rot_pts_maxx = np.max(canon_rot_pts[:, :3], axis=0)
                canon_rot_pts_bbox_center = (canon_rot_pts_minn + canon_rot_pts_maxx) / 2.
                cur_part_ref_trans_bbox = canon_transformation_mtx[:3, 3] - canon_rot_pts_bbox_center
                part_ref_trans_bbox.append(np.reshape(cur_part_ref_trans_bbox, (1, 3)))

                # transformation_mtx = np.transpose(transformation_mtx, (1, 0))
                transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))
                tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                tot_transformation_mtx_segs.append(transformation_mtx)

                part_state_rots.append(np.reshape(transformation_mtx[0, :3, :3], (1, 3, 3)))

                # canon_transformation_mtx = np.transpose(canon_transformation_mtx, (1, 0))
                part_ref_rots.append(np.reshape(canon_transformation_mtx[:3, :3], (1, 3, 3)))
                part_ref_trans.append(np.reshape(canon_transformation_mtx[:3, 3], (1, 3)))

                if self.pre_compute_delta == 1 and self.split == "train":
                    tot_transformed_pts.append(canon_rot_pts[:, :3])
                else:
                    tot_transformed_pts.append(rot_pts[:, :3])
                canon_transformed_pts.append(canon_rot_pts[:, :3])
            else:
                # rot_pts: n_pts_part x 3
                cur_seg_pts_minn = np.min(cur_seg_pts, axis=0)
                cur_seg_pts_maxx = np.max(cur_seg_pts, axis=0)
                cur_seg_pts_bbox_center = (cur_seg_pts_minn + cur_seg_pts_maxx) / 2.
                cur_part_ref_trans_bbox = -1. * cur_seg_pts_bbox_center
                canon_transformed_pts.append(cur_seg_pts)

                rot_pts = np.zeros_like(cur_seg_pts)
                rot_pts[:, :] = cur_seg_pts[:, :]
                # rot_pts = cur_seg_pts
                rot_pts[:, :3] = np.matmul(np.reshape(R1, (1, 3, 3)),
                                            np.reshape(rot_pts[:, :3], (rot_pts.shape[0], 3, 1)))[:, :3, 0]
                transformation_mtx = np.zeros((4, 4), dtype=np.float)
                transformation_mtx[0, 0] = 1.;
                transformation_mtx[1, 1] = 1.;
                transformation_mtx[2, 2] = 1.
                tot_transformed_pts.append(rot_pts)
                transformation_mtx[:3] = np.matmul(R1, transformation_mtx[:3])

                transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                tot_transformation_mtx_segs.append(transformation_mtx)

                part_state_rots.append(np.reshape(transformation_mtx[0, :3, :3], (1, 3, 3)))

                canon_transformation_mtx = np.zeros((4, 4), dtype=np.float)
                canon_transformation_mtx[0, 0] = 1.
                canon_transformation_mtx[1, 1] = 1.
                canon_transformation_mtx[2, 2] = 1.
                part_ref_rots.append(np.reshape(canon_transformation_mtx[:3, :3], (1, 3, 3)))
                part_ref_trans.append(np.zeros((1, 3), dtype=np.float))

                ''' Get points state translations with centralized boudning box '''
                # rot_pts: n_pts_part x 3
                rot_pts_minn = np.min(rot_pts, axis=0)
                rot_pts_maxx = np.max(rot_pts, axis=0)
                rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                cur_part_state_trans_bbox = -1. * rot_pts_bbox_center  # state trans bbox
                part_state_trans_bbox.append(np.reshape(cur_part_state_trans_bbox, (1, 3)))

                ''' Ref transformation bbox '''
                part_ref_trans_bbox.append(np.reshape(cur_part_ref_trans_bbox, (1, 3)))

        ''' Concatenate part axis direction and part axis offset '''
        # part_axis: n_part x 3; part_axis: n_part x 3 --> part axis...
        part_axis = np.concatenate(part_axis, axis=0)
        part_axis = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(part_axis, (part_axis.shape[0], 3, 1)))
        part_axis = np.reshape(part_axis, (part_axis.shape[0], 3))

        part_pv_offset = np.array(part_pv_offset)
        ''' Concatenate part axis direction and part axis offset '''

        part_pv_point = np.concatenate(part_pv_point, axis=0)
        part_pv_point = np.matmul(np.reshape(R1, (1, 3, 3)),
                                    np.reshape(part_pv_point, (part_pv_point.shape[0], 3, 1)))
        part_pv_point = np.reshape(part_pv_point, (part_pv_point.shape[0], 3))

        tot_transformed_pts = np.concatenate(tot_transformed_pts, axis=0)
        canon_transformed_pts = np.concatenate(canon_transformed_pts, axis=0)

        ''' Use GT transformation matrix as initial pose ''' #
        tot_transformation_mtx = np.concatenate(tot_transformation_mtx, axis=0)
        tot_transformation_mtx_segs = np.concatenate(tot_transformation_mtx_segs, axis=0)

        part_state_rots = np.concatenate(part_state_rots, axis=0)
        part_ref_rots = np.concatenate(part_ref_rots, axis=0)
        part_ref_trans = np.concatenate(part_ref_trans, axis=0)
        ''' Get part_state_trans_bbox and part_ref_trans_bbox '''
        part_state_trans_bbox = np.concatenate(part_state_trans_bbox, axis=0)
        part_ref_trans_bbox = np.concatenate(part_ref_trans_bbox, axis=0)

        gt_pose = tot_transformation_mtx

        if self.global_rot >= 0:
            af_glb_boundary_pts = [np.min(tot_transformed_pts, axis=0), np.max(tot_transformed_pts, axis=0)]
            af_glb_center_pt = (af_glb_boundary_pts[0] + af_glb_boundary_pts[1]) / 2
            # length_bb = np.linalg.norm(af_glb_boundary_pts[0] - af_glb_boundary_pts[1])

            af_glb_center_pt = np.mean(tot_transformed_pts, axis=0)

            # latest work? aiaiaia...

            # all normalize into 0
            # sampled_pcts = (sampled_pcts - center_pt.reshape(1, 3)) / length_bb
            tot_transformed_pts = (tot_transformed_pts - af_glb_center_pt.reshape(1, 3))
            # tot_transformed_pts = (tot_transformed_pts - af_glb_center_pt.reshape(1, 3)) / length_bb

            part_pv_point = part_pv_point - np.reshape(af_glb_center_pt, (1, 3))
            gt_pose[:, :3, 3] = gt_pose[:, :3, 3] - af_glb_center_pt
            tot_transformation_mtx_segs[:, :3, 3] = tot_transformation_mtx_segs[:, :3, 3] - af_glb_center_pt

        cur_pc = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        tot_transformed_pts = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        cur_label = torch.from_numpy(pts_to_seg_idx).long()
        tot_label = torch.from_numpy(pts_to_seg_idx).long()
        cur_pose = torch.from_numpy(gt_pose.astype(np.float32))
        cur_pose_segs = torch.from_numpy(tot_transformation_mtx_segs.astype(np.float32))
        cur_ori_pc = torch.from_numpy(sampled_pcts.astype(np.float32)).float()
        cur_canon_transformed_pts = torch.from_numpy(canon_transformed_pts.astype(np.float32)).float()
        cur_part_state_rots = torch.from_numpy(part_state_rots.astype(np.float32)).float()
        cur_part_ref_rots = torch.from_numpy(part_ref_rots.astype(np.float32)).float()
        cur_part_ref_trans = torch.from_numpy(part_ref_trans.astype(np.float32)).float()
        # part_state_trans_bbox part_ref_trans_bbox
        cur_part_state_trans_bbox = torch.from_numpy(part_state_trans_bbox.astype(np.float32)).float()
        cur_part_ref_trans_bbox = torch.from_numpy(part_ref_trans_bbox.astype(np.float32)).float()

        cur_part_axis = torch.from_numpy(part_axis.astype(np.float32)).float()
        cur_part_pv_offset = torch.from_numpy(part_pv_offset.astype(np.float32)).float()

        cur_part_pv_point = torch.from_numpy(part_pv_point.astype(np.float32)).float()
        part_angles = np.array(part_angles, dtype=np.float32)  # .float()
        part_angles = torch.from_numpy(part_angles.astype(np.float32)).float()


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
            'part_axis': cur_part_axis,  # get ground-truth axis
            'idx': idx_arr,
            'part_state_trans_bbox': cur_part_state_trans_bbox,
            'part_ref_trans_bbox': cur_part_ref_trans_bbox,
            'part_pv_offset': cur_part_pv_offset,
            'part_pv_point': cur_part_pv_point,
            'part_angles': part_angles,
        }

        return rt_dict

    def __len__(self):
        return len(self.shape_idxes) * self.n_samples

    def get_num_moving_parts_to_cnt(self):
        return self.num_mov_parts_to_cnt

    def reset_num_moving_parts_to_cnt(self):
        self.num_mov_parts_to_cnt = {}


if __name__ == '__main__':
    pass
