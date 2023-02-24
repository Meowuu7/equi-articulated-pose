import os
from dt_data_utils import *
import scipy.io as sio
import torch
from dt_part_transform import *
try:
    from SPConvNets.models.model_util import *
except:
    pass

from scipy.spatial.transform import Rotation as sciR

try:
    import polyscope as ps

    from sklearn.manifold import TSNE

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.look_at((0., 0.0, 1.5), (0., 0., 1.))
    ps.set_screenshot_extension(".png")

    color = [
        (136/255.0,224/255.0,239/255.0),
        (180/255.0,254/255.0,152/255.0),
        (184/255.0,59/255.0,94/255.0),
        (106/255.0,44/255.0,112/255.0),
        (39/255.0,53/255.0,135/255.0),
    (0,173/255.0,181/255.0), (170/255.0,150/255.0,218/255.0), (82/255.0,18/255.0,98/255.0), (234/255.0,84/255.0,85/255.0), (234/255.0,255/255.0,208/255.0),(162/255.0,210/255.0,255/255.0),
        (187/255.0,225/255.0,250/255.0), (240/255.0,138/255.0,93/255.0), (184/255.0,59/255.0,94/255.0),(106/255.0,44/255.0,112/255.0),(39/255.0,53/255.0,135/255.0),
    ]
except:
    pass



class dummy:
    def __init__(self, split='train'):
        self.mesh_fn = "summary.obj"
        self.surface_to_seg_fn = "sfs_idx_to_dof_name_idx.npy"
        self.attribute_fn = "motion_attributes.json"
        self.shape_root = "/Users/meow/Study/_2021_spring/part-segmentation/MDV02/eyeglasses"
        self.split = split

        self.shape_folders = []

        shape_idxes = os.listdir(self.shape_root)
        shape_idxes = sorted(shape_idxes)
        shape_idxes = [tmpp for tmpp in shape_idxes if tmpp[0] != "." and tmpp != "train" and tmpp != "test" and tmpp != "testR" and tmpp != "Archive.zip"]

        self.train_ratio = 0.9

        train_nns = int(len(shape_idxes) * self.train_ratio)

        if self.split == "train":
            shape_idxes = shape_idxes[:train_nns]
            self.base_idx = 0
        else:
            shape_idxes = shape_idxes[train_nns:]
            self.base_idx = train_nns

        self.shape_idxes = shape_idxes

        if self.split == "train":
            if not os.path.exists(os.path.join(self.shape_root, "train")):
                os.mkdir(os.path.join(self.shape_root, "train"))
                self.save_root = os.path.join(self.shape_root, "train")
            else:
                self.save_root = os.path.join(self.shape_root, "train")
        else:
            if not os.path.exists(os.path.join(self.shape_root, "test")):
                os.mkdir(os.path.join(self.shape_root, "test"))
                self.save_root = os.path.join(self.shape_root, "test")
            else:
                self.save_root = os.path.join(self.shape_root, "test")
            if not os.path.exists(os.path.join(self.shape_root, "testR")):
                os.mkdir(os.path.join(self.shape_root, "testR"))
                self.save_root_R = os.path.join(self.shape_root, "testR")
            else:
                self.save_root_R = os.path.join(self.shape_root, "testR")


    def refine_triangle_idxes_by_seg_idx(self, seg_idx_to_triangle_idxes, cur_triangles):
        res_triangles = []
        cur_triangles_to_seg_idx = []
        for seg_idx in seg_idx_to_triangle_idxes:
            cur_triangle_idxes = np.array(seg_idx_to_triangle_idxes[seg_idx], dtype=np.long)
            cur_seg_triangles = cur_triangles[cur_triangle_idxes]
            res_triangles.append(cur_seg_triangles)
            cur_triangles_to_seg_idx += [seg_idx for _ in range(cur_triangle_idxes.shape[0])]
        res_triangles = np.concatenate(res_triangles, axis=0)
        cur_triangles_to_seg_idx = np.array(cur_triangles_to_seg_idx, dtype=np.long)
        return res_triangles, cur_triangles_to_seg_idx

    def sample_pts(self, n_pts):
        for idx in range(len(self.shape_idxes)):
            shp_idx = self.shape_idxes[idx]
            cur_folder = os.path.join(self.shape_root, shp_idx)

            cur_mesh_fn = os.path.join(cur_folder, self.mesh_fn)
            cur_surface_to_seg_fn = os.path.join(cur_folder, self.surface_to_seg_fn)
            cur_motion_attributes_fn = os.path.join(cur_folder, self.attribute_fn)

            cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
            cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn,
                                                                                            nparts=1)
            cur_triangles, cur_triangles_to_seg_idx = self.refine_triangle_idxes_by_seg_idx(seg_idx_to_triangle_idxes,
                                                                                            cur_triangles)
            cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

            sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles,
                                                                                        cur_triangles_to_seg_idx,
                                                                                        npoints=n_pts)
            # smapled_pts = torch.from_numpy(sampled_pcts).float()
            # fps_idx = farthest_point_sampling(sampled_pcts.unsqueeze(0), n_sampling=n_pts)
            # cur_pc = cur_pc[fps_idx]
            # cur_label = cur_label[fps_idx]
            # cur_pose = cur_pose[fps_idx]
            #
            label = 0
            name = f"eyeglasses_{idx}"
            R = np.eye(3, dtype=np.float)
            cat = 'eyeglasses'
            data = {'pc': sampled_pcts, 'name': name, 'label': label, 'R': R, 'cat': cat}
            if self.split == "train":
                sio.savemat(os.path.join(self.save_root, f"eyeglasses_{idx + self.base_idx}.mat"), data)
            else:
                sio.savemat(os.path.join(self.save_root, f"eyeglasses_{idx + self.base_idx}.mat"), data)
                sio.savemat(os.path.join(self.save_root_R, f"eyeglasses_{idx + self.base_idx}.mat"), data)


def sampled_pts(fn, n_pts=512):
    shape_files = os.listdir(fn)
    for file_n in shape_files:
        cur_shape = sio.loadmat(os.path.join(fn, file_n))
        cur_pc = cur_shape['pc']
        cur_pc = torch.from_numpy(cur_pc).float()
        smapled_pts = torch.from_numpy(cur_pc).float()
        fps_idx = farthest_point_sampling(smapled_pts.unsqueeze(0), n_sampling=n_pts)
        # cur_pc = cur_pc[fps_idx]
        # cur_label = cur_label[fps_idx]
        # cur_pose = cur_pose[fps_idx]


# filter oven motion types...
def filter_oven_motion_types(root, shape_type="oven"):

    nparts = None
    if shape_type == "eyeglasses":
        nparts = 2
        nparts = None

    file_names = os.listdir(root)
    file_names = [fnn for fnn in file_names if fnn != ".DS_Store"]
    mesh_fn = "summary.obj"
    surface_to_seg_fn = "sfs_idx_to_dof_name_idx.npy"
    attribute_fn = "motion_attributes.json"
    npoints = 1024
    rot_factor = 0.5

    # angle_y =

    # file_names = ["0034", "0037", "0039"]
    # file_names = ['0027', '0034']
    file_names = ['0025', '0034'] # test file names for oven?
    #
    # file_names = ['0057', '0059', '0060', '0061', '0062']
    # file_names = ['0039', '0040', '0041', '0043']
    # file_names = ['0057', '0059', '0060', '0061']
    # file_names = ['0056', '0057', '0060']
    # file_names = ['0076', '0077', '0078', '0079', '0080', '0081', '0082', '0084', '0085']
    # 77, 80, 81

    for fn in file_names:
        print(fn)
        try:
            cur_folder = os.path.join(root, fn)
            cur_mesh_fn = os.path.join(cur_folder, mesh_fn)
            cur_surface_to_seg_fn = os.path.join(cur_folder, surface_to_seg_fn)
            cur_motion_attributes_fn = os.path.join(cur_folder, attribute_fn)

            cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
            # cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn, nparts=2)
            cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn)
            cur_triangles, cur_triangles_to_seg_idx = refine_triangle_idxes_by_seg_idx(seg_idx_to_triangle_idxes, cur_triangles)
            cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

            sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles, cur_triangles_to_seg_idx, npoints=npoints)

            tot_transformed_pts = []
            tot_transformation_mtx = []
            tot_transformation_mtx_segs = []
            rot_1 = False
            seg_label_to_transformed_pts = {}
            for i_seg in range(len(cur_motion_attributes)):
                # if i_seg >= 2:
                #     continue
                if nparts is not None and i_seg >= nparts:
                    continue
                cur_seg_motion_info = cur_motion_attributes[i_seg]
                if shape_type == "eyeglasses" and i_seg == 1:
                    cur_seg_motion_info = cur_motion_attributes[i_seg + 1]
                if shape_type == "eyeglasses" and i_seg == 2:
                    cur_seg_motion_info = cur_motion_attributes[i_seg - 1]
                cur_seg_pts_idxes = np.array(seg_idx_to_sampled_pts[i_seg], dtype=np.long)
                cur_seg_pts = sampled_pcts[cur_seg_pts_idxes]
                if cur_seg_motion_info["motion_type"] == "rotation":
                    center = cur_seg_motion_info["center"]
                    axis = cur_seg_motion_info["direction"]

                    if shape_type in ['laptop']:
                        # theta = (np.random.uniform(0., 1., (1,)).item() * np.pi - np.pi / 2.) * rot_factor
                        # mult_factor = 0.25
                        theta = -0.30 * np.pi
                    elif shape_type in ['eyeglasses']:
                        theta = (np.random.uniform(0., 0.5, (1,)).item() * np.pi)  # * rot_factor
                        # Use mult-factor to get consistenty part pose changes...
                        # mult_factor = 0.15
                        # theta = mult_factor * np.pi
                    elif shape_type in ['oven', 'washing_machine']:
                        theta = (np.random.uniform(0.5, 1., (1,)).item() * np.pi) * rot_factor
                    else:
                        theta = (np.random.uniform(0., 1., (1,)).item() * np.pi) * rot_factor
                    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
                    y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                    y_theta = -0.5 * np.pi
                    # cur_seg_pts, y_trans_mtx = revoluteTransform(cur_seg_pts, y_center, y_axis, y_theta)
                    # cur_seg_pts = cur_seg_pts[:, :3]
                    # y_trans_mtx = np.transpose(y_trans_mtx, (1, 0))
                    # y_rot_mtx = y_trans_mtx[:3, :3]
                    # print(theta)
                    rot_pts, transformation_mtx = revoluteTransform(cur_seg_pts, center, axis, theta)
                    transformation_mtx = np.transpose(transformation_mtx, (1, 0))


                    rot_pts, y_transformation_mtx = revoluteTransform(rot_pts[:, :3], y_center, y_axis, y_theta)
                    y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                    transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], transformation_mtx[:3, :])

                    # y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
                    # y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                    # y_theta = -0.5 * np.pi
                    #
                    # rot_pts, y_transformation_mtx = revoluteTransform(rot_pts[:, :3], y_center, y_axis, y_theta)
                    # y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                    # transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], transformation_mtx[:3, :])

                    transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                    cur_rot_mtx = transformation_mtx[0, :3, :3]
                    dot_product = np.dot(cur_rot_mtx, np.transpose(cur_rot_mtx, [1, 0]))
                    traces = dot_product[0, 0].item() + dot_product[1, 1].item() + dot_product[2, 2].item()
                    traces = float(traces) - 1.0;
                    traces = traces / 2.
                    print(f"current traces: {traces}")

                    tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                    tot_transformation_mtx_segs.append(transformation_mtx)
                    # rot_pts = cur_seg_pts
                    # print(rot_pts.shape)
                    tot_transformed_pts.append(rot_pts[:, :3])
                    rot_1 = True
                    seg_label_to_transformed_pts[i_seg] = rot_pts[:, :3]
                else:

                    transformation_mtx = np.zeros((4, 4), dtype=np.float)
                    transformation_mtx[0, 0] = 1.
                    transformation_mtx[1, 1] = 1.
                    transformation_mtx[2, 2] = 1.

                    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
                    y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                    y_theta = -0.5 * np.pi

                    rot_pts, y_transformation_mtx = revoluteTransform(cur_seg_pts[:, :3], y_center, y_axis, y_theta)
                    y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                    transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], transformation_mtx[:3, :])

                    # y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
                    # y_center = np.array([0.0, 0.0, 0.0], dtype=np.float)
                    # y_theta = -0.5 * np.pi
                    #
                    # rot_pts, y_transformation_mtx = revoluteTransform(rot_pts[:, :3], y_center, y_axis, y_theta)
                    # y_transformation_mtx = np.transpose(y_transformation_mtx, (1, 0))
                    # transformation_mtx[:3, :] = np.matmul(y_transformation_mtx[:3, :3], transformation_mtx[:3, :])

                    cur_seg_pts = rot_pts[:, :3]
                    tot_transformed_pts.append(cur_seg_pts)
                    seg_label_to_transformed_pts[i_seg] = cur_seg_pts

                    # transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                    transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))
                    tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                    tot_transformation_mtx_segs.append(transformation_mtx)

            tot_transformed_pts = np.concatenate(tot_transformed_pts, axis=0)
            # print(tot_transformed_pts.shape)

            ''' Use fake initial pose '''
            # gt_pose = np.zeros((self.npoints, 4, 4), dtype=np.float)
            # gt_pose[:, 0, 0] = 1.; gt_pose[:, 1, 1] = 1.; gt_pose[:, 2, 2] = 1.

            ''' Use GT transformation matrix as initial pose '''
            # tot_transformation_mtx = np.concatenate(tot_transformation_mtx, axis=0)
            # tot_transformation_mtx_segs = np.concatenate(tot_transformation_mtx_segs, axis=0)

            ps.register_point_cloud(f"ori_pts", sampled_pcts, radius=0.012, color=color[0])
            ps.show()
            ps.remove_all_structures()

            ps.register_point_cloud(f"seg_transformed_pts_", tot_transformed_pts, radius=0.012, color=color[4])
            ps.show()
            ps.remove_all_structures()

            for seg_label in seg_label_to_transformed_pts:
                cur_seg_transformed_pts = seg_label_to_transformed_pts[seg_label]
                ps.register_point_cloud(f"seg_transformed_pts_{seg_label}", cur_seg_transformed_pts, radius=0.012, color=color[seg_label])
            # ps.register_point_cloud(f"transformed_pts", tot_transformed_pts, radius=0.012, color=color[0])
            ps.show()
            ps.remove_all_structures()

            for seg_label in seg_label_to_transformed_pts:
                cur_seg_transformed_pts = seg_label_to_transformed_pts[seg_label]
                if seg_label == 0:
                    ps.register_point_cloud(f"seg_transformed_pts_{seg_label}", cur_seg_transformed_pts, radius=0.012, color=color[seg_label])
                else:
                    ps.register_point_cloud(f"seg_transformed_pts_{seg_label}_a", cur_seg_transformed_pts[:cur_seg_transformed_pts.shape[0] // 10], radius=0.012,
                                            color=color[seg_label])
                    ps.register_point_cloud(f"seg_transformed_pts_{seg_label}_b",
                                            cur_seg_transformed_pts[cur_seg_transformed_pts.shape[0] // 10: ],
                                            radius=0.012,
                                            color=color[0])
            # ps.register_point_cloud(f"transformed_pts", tot_transformed_pts, radius=0.012, color=color[0])
            ps.show()
            ps.remove_all_structures()
        except:
            center


def get_sphere_pts(lll=256):
    # !/usr/bin/python
    # -*- coding: utf-8 -*-
    import math
    import numpy as np

    class Spherical(object):

        '''球坐标系'''

        def __init__(self, radial=1.0, polar=0.0, azimuthal=0.0):

            self.radial = radial

            self.polar = polar

            self.azimuthal = azimuthal

        def toCartesian(self):

            '''转直角坐标系'''

            r = math.sin(self.azimuthal) * self.radial

            x = math.cos(self.polar) * r

            y = math.sin(self.polar) * r

            z = math.cos(self.azimuthal) * self.radial

            return x, y, z

    def splot(limit):

        s = Spherical()

        n = int(math.ceil(math.sqrt((limit - 2) / 4)))
        # n = 256

        azimuthal = 0.5 * math.pi / n

        for a in range(-n, n + 1):

            s.polar = 0

            size = (n - abs(a)) * 4 or 1

            polar = 2 * math.pi / size

            for i in range(size):

                yield s.toCartesian()

                s.polar += polar

            s.azimuthal += azimuthal

    # points = splot(limit=lll)
    # points = np.array(points, dtype=np.float)

    pts = []
    for point in splot(limit=lll):
        pts.append(point)
        print("%f %f %f" % point)
    points = np.array(pts, dtype=np.float)
    print(points.shape)

    np.save(f"./data/{points.shape[0]}_sphere.npy", points)

    # ps.register_point_cloud(f"ori_pts", points, radius=0.012, color=color[0])
    # ps.show()
    # ps.remove_all_structures()

def get_arbitrary_rot_pts():
    rotation_angle = sciR.random().as_matrix()
    rotation_matrix = rotation_angle[:3, :3]
    R1 = rotation_matrix
    return R1

def filter_oven_motion_types_state_changes(root, shape_type="oven"):

    nparts = None
    if shape_type == "eyeglasses":
        nparts = 2
        nparts = None

    file_names = os.listdir(root)
    file_names = [fnn for fnn in file_names if fnn != ".DS_Store"]
    mesh_fn = "summary.obj"
    surface_to_seg_fn = "sfs_idx_to_dof_name_idx.npy"
    attribute_fn = "motion_attributes.json"
    npoints = 1024
    rot_factor = 0.5

    # file_names = ["0034", "0037", "0039"]
    # file_names = ['0027', '0034']
    # file_names = ['0025', '0034']
    #
    # file_names = ['0057', '0059', '0060', '0061', '0062']
    file_names = ['0039', '0040', '0041', '0043']
    # file_names = ['0057', '0059', '0060', '0061']
    # file_names = ['0056', '0057', '0060']
    # file_names = ['0076', '0077', '0078', '0079', '0080', '0081', '0082', '0084', '0085']
    # 77, 80, 81

    tot_ori_pts = []
    tot_aligned_transformed_pts = []
    tot_rotated_transformed_pts = []
    # tot_seg_label_to_transformed_pts = []
    tot_seg_label_to_sampled_pts = []

    for fn in file_names:
        print(fn)
        try:
            cur_folder = os.path.join(root, fn)
            cur_mesh_fn = os.path.join(cur_folder, mesh_fn)
            cur_surface_to_seg_fn = os.path.join(cur_folder, surface_to_seg_fn)
            cur_motion_attributes_fn = os.path.join(cur_folder, attribute_fn)

            cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
            # cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn, nparts=2)
            cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn)
            cur_triangles, cur_triangles_to_seg_idx = refine_triangle_idxes_by_seg_idx(seg_idx_to_triangle_idxes, cur_triangles)
            cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

            sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles, cur_triangles_to_seg_idx, npoints=npoints)




            tot_transformed_pts = []
            tot_transformation_mtx = []
            tot_transformation_mtx_segs = []
            rot_1 = False
            seg_label_to_transformed_pts = {}
            for i_seg in range(len(cur_motion_attributes)):
                # if i_seg >= 2:
                #     continue
                if nparts is not None and i_seg >= nparts:
                    continue
                cur_seg_motion_info = cur_motion_attributes[i_seg]
                if shape_type == "eyeglasses" and i_seg == 1:
                    cur_seg_motion_info = cur_motion_attributes[i_seg + 1]
                if shape_type == "eyeglasses" and i_seg == 2:
                    cur_seg_motion_info = cur_motion_attributes[i_seg - 1]
                cur_seg_pts_idxes = np.array(seg_idx_to_sampled_pts[i_seg], dtype=np.long)
                cur_seg_pts = sampled_pcts[cur_seg_pts_idxes]
                if cur_seg_motion_info["motion_type"] == "rotation":
                    center = cur_seg_motion_info["center"]
                    axis = cur_seg_motion_info["direction"]

                    if shape_type in ['laptop']:
                        # theta = (np.random.uniform(0., 1., (1,)).item() * np.pi - np.pi / 2.) * rot_factor
                        # mult_factor = 0.25
                        theta = -0.30 * np.pi
                    elif shape_type in ['eyeglasses']:
                        theta = (np.random.uniform(0., 0.5, (1,)).item() * np.pi)  # * rot_factor
                        # Use mult-factor to get consistenty part pose changes...
                        # mult_factor = 0.15
                        # theta = mult_factor * np.pi
                    elif shape_type in ['oven', 'washing_machine']:
                        theta = (np.random.uniform(0.5, 1., (1,)).item() * np.pi) * rot_factor
                    else:
                        theta = (np.random.uniform(0., 1., (1,)).item() * np.pi) * rot_factor
                    print(theta)
                    rot_pts, transformation_mtx = revoluteTransform(cur_seg_pts, center, axis, theta)
                    transformation_mtx = np.transpose(transformation_mtx, (1, 0))
                    transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                    cur_rot_mtx = transformation_mtx[0, :3, :3]
                    dot_product = np.dot(cur_rot_mtx, np.transpose(cur_rot_mtx, [1, 0]))
                    traces = dot_product[0, 0].item() + dot_product[1, 1].item() + dot_product[2, 2].item()
                    traces = float(traces) - 1.0;
                    traces = traces / 2.
                    print(f"current traces: {traces}")

                    tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                    tot_transformation_mtx_segs.append(transformation_mtx)
                    # rot_pts = cur_seg_pts
                    # print(rot_pts.shape)
                    tot_transformed_pts.append(rot_pts[:, :3])
                    rot_1 = True
                    seg_label_to_transformed_pts[i_seg] = rot_pts[:, :3]
                else:
                    tot_transformed_pts.append(cur_seg_pts)
                    seg_label_to_transformed_pts[i_seg] = cur_seg_pts
                    transformation_mtx = np.zeros((4, 4), dtype=np.float)
                    transformation_mtx[0, 0] = 1.
                    transformation_mtx[1, 1] = 1.
                    transformation_mtx[2, 2] = 1.
                    transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))
                    tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                    tot_transformation_mtx_segs.append(transformation_mtx)

            tot_transformed_pts = np.concatenate(tot_transformed_pts, axis=0)
            # print(tot_transformed_pts.shape)

            ''' Use fake initial pose '''
            # gt_pose = np.zeros((self.npoints, 4, 4), dtype=np.float)
            # gt_pose[:, 0, 0] = 1.; gt_pose[:, 1, 1] = 1.; gt_pose[:, 2, 2] = 1.

            ''' Use GT transformation matrix as initial pose '''
            # tot_transformation_mtx = np.concatenate(tot_transformation_mtx, axis=0)
            # tot_transformation_mtx_segs = np.concatenate(tot_transformation_mtx_segs, axis=0)

            tot_seg_label_to_sampled_pts.append(seg_idx_to_sampled_pts)
            tot_ori_pts.append(sampled_pcts)
            tot_aligned_transformed_pts.append(tot_transformed_pts)
            # R1: 3 x 3
            R1 = get_arbitrary_rot_pts()
            rotated_transformed_pts = np.matmul(R1, np.transpose(tot_transformed_pts, (1, 0)))
            rotated_transformed_pts = np.transpose(rotated_transformed_pts, (1, 0))
            tot_rotated_transformed_pts.append(rotated_transformed_pts)

            ''' Ori -- Draw original points without segs, transformed points with segs, transformed points without segs '''
            # ps.register_point_cloud(f"ori_pts", sampled_pcts, radius=0.012, color=color[0])
            # ps.show()
            # ps.remove_all_structures()
            #
            # ps.register_point_cloud(f"seg_transformed_pts_", tot_transformed_pts, radius=0.012, color=color[4])
            # ps.show()
            # ps.remove_all_structures()
            #
            # for seg_label in seg_label_to_transformed_pts:
            #     cur_seg_transformed_pts = seg_label_to_transformed_pts[seg_label]
            #     ps.register_point_cloud(f"seg_transformed_pts_{seg_label}", cur_seg_transformed_pts, radius=0.012, color=color[seg_label])
            # # ps.register_point_cloud(f"transformed_pts", tot_transformed_pts, radius=0.012, color=color[0])
            # ps.show()
            # ps.remove_all_structures()

        except:
            continue

    # tot_ori_pts = []
    # tot_aligned_transformed_pts = []
    # tot_rotated_transformed_pts = []
    # # tot_seg_label_to_transformed_pts = []
    # tot_seg_label_to_sampled_pts = []

    #### Draw original points #### ->
    #### Draw transformed points ####
    # with/with-out seg #
    #### Draw rotated transformed points ####
    # with/with-out seg #

    n_segs = 3

    # #### Draw original points ####
    # for i_shp in range(len(tot_ori_pts)):
    #     cur_shp_ori_pts = tot_ori_pts[i_shp]
    #     cur_shp_ori_pcd = ps.register_point_cloud(f"seg_transformed_pts_{i_shp}", cur_shp_ori_pts, radius=0.012,
    #                             color=color[i_shp])
    # ps.show()
    # ps.remove_all_structures()
    #
    #
    # #### Draw aligned segs ####
    # for i_seg in range(n_segs):
    #     for i_shp in range(len(tot_ori_pts)):
    #         cur_shp_cur_seg_pts_idxes = tot_seg_label_to_sampled_pts[i_shp][i_seg]
    #         cur_shp_cur_seg_pts = tot_ori_pts[i_shp][cur_shp_cur_seg_pts_idxes]
    #         cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_transformed_pts_{i_shp}", cur_shp_cur_seg_pts, radius=0.012,
    #                             color=color[i_shp])
    #     ps.show()
    #     ps.remove_all_structures()
    #
    #
    # #### Draw aligned transformed points ####
    # for i_shp in range(len(tot_aligned_transformed_pts)):
    #     cur_shp_ori_pts = tot_aligned_transformed_pts[i_shp]
    #     cur_shp_ori_pcd = ps.register_point_cloud(f"aligned_transformed_{i_shp}", cur_shp_ori_pts, radius=0.012,
    #                             color=color[i_shp])
    # ps.show()
    # ps.remove_all_structures()
    #
    #
    # #### Draw aligned segs ####
    # for i_seg in range(n_segs):
    #     for i_shp in range(len(tot_aligned_transformed_pts)):
    #         cur_shp_cur_seg_pts_idxes = tot_seg_label_to_sampled_pts[i_shp][i_seg]
    #         cur_shp_cur_seg_pts = tot_aligned_transformed_pts[i_shp][cur_shp_cur_seg_pts_idxes]
    #         cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_aligned_transformed_pts_{i_shp}", cur_shp_cur_seg_pts,
    #                                                       radius=0.012,
    #                                                       color=color[i_shp])
    #     ps.show()
    #     ps.remove_all_structures()
    #
    #
    #### Draw rotated transformed points ####
    for i_shp in range(len(tot_rotated_transformed_pts)):
        cur_shp_ori_pts = tot_rotated_transformed_pts[i_shp]
        cur_shp_ori_pcd = ps.register_point_cloud(f"rotated_transformed_{i_shp}", cur_shp_ori_pts, radius=0.012,
                                                  color=color[i_shp])
    ps.show()
    ps.remove_all_structures()
    #
    #
    # #### Draw aligned segs ####
    # for i_seg in range(n_segs):
    #     for i_shp in range(len(tot_rotated_transformed_pts)):
    #         cur_shp_cur_seg_pts_idxes = tot_seg_label_to_sampled_pts[i_shp][i_seg]
    #         cur_shp_cur_seg_pts = tot_rotated_transformed_pts[i_shp][cur_shp_cur_seg_pts_idxes]
    #         cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_rotated_transformed_pts_{i_shp}", cur_shp_cur_seg_pts,
    #                                                       radius=0.012,
    #                                                       color=color[i_shp])
    #     ps.show()
    #     ps.remove_all_structures()


    for i_shp in range(len(tot_rotated_transformed_pts)):
        cur_rotated_transformed_pts = tot_rotated_transformed_pts[i_shp]
        for i_seg in range(n_segs):
            cur_shp_cur_seg_pts_idxes = tot_seg_label_to_sampled_pts[i_shp][i_seg]
            cur_shp_cur_seg_pts = tot_rotated_transformed_pts[i_shp][cur_shp_cur_seg_pts_idxes]
            if i_seg == 0:
                cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_rotated_transformed_pts_{i_seg}", cur_shp_cur_seg_pts,
                                                              radius=0.012,
                                                              color=color[i_seg])
            else:
                cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_rotated_transformed_pts_{i_seg}_ori",
                                                              cur_shp_cur_seg_pts[cur_shp_cur_seg_pts.shape[0] // 2:, :],
                                                              radius=0.012,
                                                              color=color[i_seg])
                cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_rotated_transformed_pts_{i_seg}_near",
                                                              cur_shp_cur_seg_pts[: cur_shp_cur_seg_pts.shape[0] // 2,
                                                              :],
                                                              radius=0.012,
                                                              color=color[0])
        ps.show()
        ps.remove_all_structures()

    for i_shp in range(len(tot_rotated_transformed_pts)):
        cur_rotated_transformed_pts = tot_rotated_transformed_pts[i_shp]
        for i_seg in range(n_segs):
            cur_shp_cur_seg_pts_idxes = tot_seg_label_to_sampled_pts[i_shp][i_seg]
            cur_shp_cur_seg_pts = tot_rotated_transformed_pts[i_shp][cur_shp_cur_seg_pts_idxes]
            if i_seg >= 0:
                cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_rotated_transformed_pts_{i_seg}", cur_shp_cur_seg_pts,
                                                              radius=0.012,
                                                              color=color[i_seg])
            else:
                cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_rotated_transformed_pts_{i_seg}_ori",
                                                              cur_shp_cur_seg_pts[cur_shp_cur_seg_pts.shape[0] // 2:, :],
                                                              radius=0.012,
                                                              color=color[i_seg])
                cur_shp_cur_seg_pcd = ps.register_point_cloud(f"seg_rotated_transformed_pts_{i_seg}_near",
                                                              cur_shp_cur_seg_pts[: cur_shp_cur_seg_pts.shape[0] // 2,
                                                              :],
                                                              radius=0.012,
                                                              color=color[0])
        ps.show()
        ps.remove_all_structures()





def sample_pts_from_mesh_v2(vertices, triangles, pts_per_area=500):

    sampled_pcts = []
    for i in range(triangles.shape[0]):

        v_a, v_b, v_c = int(triangles[i, 0].item()), int(triangles[i, 1].item()), int(
            triangles[i, 2].item())
        v_a, v_b, v_c = vertices[v_a], vertices[v_b], vertices[v_c]
        ab, ac = v_b - v_a, v_c - v_a
        cos_ab_ac = (np.sum(ab * ac) / np.clip(np.sqrt(np.sum(ab ** 2)) * np.sqrt(np.sum(ac ** 2)), a_min=1e-9,
                                               a_max=9999999.)).item()
        sin_ab_ac = math.sqrt(min(max(0., 1. - cos_ab_ac ** 2), 1.))
        cur_area = 0.5 * sin_ab_ac * np.sqrt(np.sum(ab ** 2)).item() * np.sqrt(np.sum(ac ** 2)).item()

        # cur_tri_seg = int(triangles_to_seg_idx[i].item())

        cur_sampled_pts = int(cur_area * pts_per_area)
        cur_sampled_pts = 1 if cur_sampled_pts == 0 else cur_sampled_pts
        # if cur_sampled_pts == 0:

        tmp_x, tmp_y = np.random.uniform(0, 1., (cur_sampled_pts,)).tolist(), np.random.uniform(0., 1., (
        cur_sampled_pts,)).tolist()

        for xx, yy in zip(tmp_x, tmp_y):
            sqrt_xx, sqrt_yy = math.sqrt(xx), math.sqrt(yy)
            aa = 1. - sqrt_xx
            bb = sqrt_xx * (1. - yy)
            cc = yy * sqrt_xx
            cur_pos = v_a * aa + v_b * bb + v_c * cc
            sampled_pcts.append(cur_pos)
            # pts_to_seg_idx.append(cur_tri_seg)

    sampled_pcts = np.array(sampled_pcts, dtype=np.float)
    return sampled_pcts


def from_meshes_to_pts(root_fn):
    shape_idxes = os.listdir(root_fn)
    for shp_idx in shape_idxes:
        obj_file_path = os.path.join(root_fn, shp_idx, "objs")
        obj_files = os.listdir(obj_file_path)
        for obj_fn in obj_files:
            try:
                obj_pure_fn = obj_fn.split(".")[0]
                if len(obj_pure_fn) == 5:
                    cur_obj_full_fn = os.path.join(obj_file_path, obj_fn)
                    vertices, meshes = load_vertices_triangles(cur_obj_full_fn)
                    sampled_pts = sample_pts_from_mesh_v2(vertices=vertices, triangles=meshes, pts_per_area=1)
                    downs_pts_idxes = np.random.permutation(sampled_pts.shape[0])[:8096]
                    downs_pts = sampled_pts[downs_pts_idxes]
                    # np.save(os.path.join(obj_file_path, obj_pure_fn + "_pts.npy"), sampled_pts)
                    np.save(os.path.join(obj_file_path, obj_pure_fn + "_down_pts.npy"), downs_pts)
            except:
                print(obj_fn, shp_idx)


def test_mesh_to_depth():
    import numpy as np
    import mesh_to_depth as m2d

    cur_obj_full_fn = os.path.join("./data/MDV02", "eyeglasses", "0010", "summary.obj")

    vertices, meshes = load_vertices_triangles(cur_obj_full_fn)

    params = []

    params.append({
        # 'cam_pos': [1, 1, 1], 'cam_lookat': [0, 0, 0], 'cam_up': [0, 1, 0],
        'cam_pos': [2, 2, 2], 'cam_lookat': [0, 0, 0], 'cam_up': [0, 1, 0],
        'x_fov': 0.349,  # End-to-end field of view in radians
        'near': 0.1, 'far': 10,
        'height': 480, 'width': 640,
        'is_depth': True,  # If false, output a ray displacement map, i.e. from the mesh surface to the camera center.
    })
    # Append more camera parameters if you want batch processing.

    # Load triangle mesh data. See python/resources/airplane/models/model_normalized.obj
    vertices = vertices.astype(np.float32)  # An array of shape (num_vertices, 3) and type np.float32.
    faces = meshes.astype(np.uint32)  # An array of shape (num_faces, 3) and type np.uint32.

    depth_maps = m2d.mesh2depth(vertices, faces, params, empty_pixel_value=np.nan)
    for depth in depth_maps:
        print(depth.shape)
    np.save("cur_depth.npy", depth_maps[0])


if __name__=="__main__":
    # dm = dummy(split='test')
    # dm.sample_pts(n_pts=512)
    #### oven data ####
    # 1 - right; 2 - down; 3 - right (not ok); 4 - left (not ok); 5 - right; 6 - right (not ok); 7 - right (ok);
    # 8 - right ok; 9 - right (not ok); 10 - right (ok); 11 - right ok; 12 - right ok; 13 - right ok; 14 - right ok;
    # 15 - right (not ok --- too dense?); 16 - right ok; 17 - right ok; 18 - right (not ok); 19 - right ok; 20 - r ok;
    # 21 - r ok; 22 - r ok; 23 - d notok; 24 - r not ok; 25 - r ok; 26 - r ok; 27 - r ok; 28 - r not ok; 29 - r ok;
    # 30 - r ok; 31 - d ok; 32 - r ok; 33 - r not ok; 34 - r not ok; 35 - r ok; 36 - r ok; 37 - r ok; 38 - r not ok;
    # 39 - r ok; 40 - r ok; 41 - r ok; 42 - r ok
    # not ok idxes: [2, 3, 4, 6, 9, 15, 18, 23, 24, 28, 33, 34, 38]
    #### oven data ####

    # 1 -- down (too dense); 2 -- down (too dense); 3 -- left (not ok!!);  4 -- down (too dense); 5 -- down; 6 - down; 7 - down; 8 - down;
    # 9 - down (not even); 10 - down; 11 - down (not ok!!); 12 - dosn (too dense); 13 - down; 14 - down; 15 - down
    # 16 - left; 17 - down; 18 - down; 19 - down; 20 - down; 21 - down; 22 - down; 23 - down; 24 - down; 25 - down; 26 - down;
    # 27 - down; 28 - down; 29 - down; 30 - down; 31 - down; 32 - left (not ok!!); 33 - up (not ok!!); 33 - down; 34 - up (...);
    # 35 - left (not ok!!); 36 - down (too dense and not even); 37 - down; 38 - down (not even); 39 - down (not ok); 40 - down;
    # 41 - down; 42 - left (not ok!!); 43 - down; 44 - left (not ok!!); 45 - up (not ok!!); 46 - down (not even); 47 - up;
    # 48 - up; 49 - down; 50 - down; 51 - down; 52 - down; 53 - down; 54 - down; 55 - left (not ok!!); 56 - down; 57 - down;
    # 58 - down; 59 - down; 60 -
    # not oks --- 61, 59, 50, 57, 56, 58, 60, 2, 7, 38, 9, 31, 62, 53, 30, 1, 39, 52, 55, 46, 41, 40,
    # oven
    # not oks ---
    dataset_root = "/Users/meow/Study/_2021_spring/part-segmentation/MDV02/oven"
    # dataset_root = "/Users/meow/Study/_2021_spring/part-segmentation/MDV02/eyeglasses"
    # dataset_root = "/Users/meow/Study/_2021_spring/part-segmentation/MDV02/laptop"
    # dataset_root = "/Users/meow/Study/_2021_spring/part-segmentation/MDV02/washing_machine"
    # dataset_root = "/Users/meow/Study/_2021_spring/part-segmentation/MDV02/plane"
    shape_type = "oven"
    # shape_type = "plane"
    # shape_type = "eyeglasses"
    # shape_type = "laptop"
    # shape_type = "washing_machine"
    # 32, 35, 33, 42, 38, 36, 6, 41,
    # 32,
    # 4, 35, 38, 6, 15,
    # 42,
    # 58, 18, 31, 62, 30,
    # 83, 23
    # 50, 58, 2, 18, 29, 10, 7, 38, 9, 31, 30, 1, 39, 52, 46, 41, 40, 13
    # 86, 83, 23, 73,
    # mode 1: 61, 50, 35, 34, 29, 72, 54, 8, 55, 79, 77, 85, 76
    # mode 2: 59, 66, 68, 4, 32, 69, 56, 58, 67, 5, 2, 20, 18, 27,
    # mode 3: 57, 3, 51, 60, 33, 45, 17, 81, 86, 7, 31, 30, 63, 15, 71, 49, 40, 14,
    # 42, 20, 10, 3, 2, 9, 1
    # 32, 35, 33, 18, 10, 21, 36, 31, 6, 24, 15, 40, 13, 42, 30, 41, 22
    # 3, 33, 5, 2, 20, 11, 29, 16, 42, 28, 10, 19, 7, 38, 9, 30, 8, 1, 6, 41, 23, 15, 12, 13, 14, 4, 32, 28, 17,
    # 50, 58, 34, 2, 18, 27, 11, 29, 45, 28, 10, 21, 7, 38, 9, 31, 62, 30, 37, 8, 1, 6, 39, 52, 46, 41, 24, 12, 40, 13, 61, 59, 32, 35

    filter_oven_motion_types(dataset_root, shape_type)
    # filter_oven_motion_types_state_changes(dataset_root, shape_type)

    nn_pts = 256
    nn_pts = 128
    nn_pts = 512
    # get_sphere_pts(lll=nn_pts)

    # root_fn = "./data/HOI4D/laptop"
    # root_fn = "./data/HOI4D/safe"
    # from_meshes_to_pts(root_fn)

    # test_mesh_to_depth()

    # print("%.4d" % 2)
