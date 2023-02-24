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
# from SPConvNets.models.common_utils import *
from SPConvNets.datasets.data_utils import *
import scipy.io as sio
import copy
# from model.utils import farthest_point_sampling

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
from scipy.spatial.transform import Rotation as sciR
from SPConvNets.datasets.part_transform import revoluteTransform
from SPConvNets.models.model_util import *
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import provider
from torch.utils import data
from SPConvNets.models.common_utils import *
from SPConvNets.datasets.data_utils import *
import scipy.io as sio
import copy
import trimesh
import pyrender

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


def reindex_triangles_vertices(vertices, triangles):
    vert_idx_to_new_idx = {}
    new_idx = 0
    for i_tri in range(triangles.shape[0]):
        cur_tri = triangles[i_tri]
        v1, v2, v3 = int(cur_tri[0].item()), int(cur_tri[1].item()), int(cur_tri[2].item())
        for v in [v1, v2, v3]:
            if v not in vert_idx_to_new_idx:
                vert_idx_to_new_idx[v] = new_idx
                new_idx = new_idx + 1
    new_vertices = np.zeros((new_idx, 3), dtype=np.float)
    for vert_idx in vert_idx_to_new_idx:
        new_vert_idx = vert_idx_to_new_idx[vert_idx]
        new_vertices[new_vert_idx] = vertices[vert_idx]
    new_triangles = np.zeros_like(triangles)
    for i_tri in range(triangles.shape[0]):
        cur_tri = triangles[i_tri]
        v1, v2, v3 = int(cur_tri[0].item()), int(cur_tri[1].item()), int(cur_tri[2].item())
        new_v1, new_v2, new_v3 = vert_idx_to_new_idx[v1], vert_idx_to_new_idx[v2], vert_idx_to_new_idx[v3]
        cur_new_tri = np.array([new_v1, new_v2, new_v3], dtype=np.long)
        new_triangles[i_tri] = cur_new_tri
    # new_triangles: tot_n_tri x 3
    # new_vertices: tot_n_vert x 3
    return new_vertices, new_triangles

def ndc_depth_to_buffer(z, near, far):  # z in [-1, 1]
    return 2 * near * far / (near + far - z * (far - near))


def buffer_depth_to_ndc(d, near, far):  # d in (0, +
    return ((near + far) - 2 * near * far / np.clip(d, a_min=1e-6, a_max=1e6)) / (far - near)






def create_partial_pts(mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
                   yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=False, vertices=None, triangles=None, seg_label_to_triangles=None, seg_transformation_mtx=None, render_img=True, no_transformation=False):
    # seg_transformation_mtx: n_seg x 4 x 4
    # if vertices is not None and triangles is not None:
    #     assert seg_label_to_triangles is not None and seg_transformation_mtx is not None
    #     m = trimesh.base.Trimesh(vertices=vertices, faces=triangles)
    # else:
    #     m = trimesh.load(read_path)

    # centralization
    # c = np.mean(m.vertices, axis=0)
    # trans = np.eye(4)
    # trans[:3, 3] = -c
    # m.apply_transform(trans) # pose of each part --- should be remembered and transformed for the further use
    # for further
    # centralize
    # seg_transformation_mtx[:, :3, 3] = seg_transformation_mtx[:, :3, 3] + np.reshape(trans, (1, 3))
    seg_label_to_depth_buffer = {}
    seg_label_to_new_pose = {}
    seg_label_to_idxes = {}
    tot_depth_buffer = []
    tot_n = 0
    seg_label_to_pts = {}
    seg_label_to_no_rot_pts = {}

    if not no_transformation:

        axis_angle = np.random.randint(0, 50, (3,))

        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float)

        ''' Random angle strategy 1 '''
        # x_angle = -1.0 * float(
        #     axis_angle[0].item()) / 100. * np.pi  # [-0.5, 0] -> sampled rotation angle around the x-axis
        # # y_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # # z_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # # ''' Strategy 1 -- '''
        # y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi  # - 0.5 * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  # - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi
        ''' Random angle strategy 1 '''

        ''' Random angle strategy 2 '''
        x_angle = 0.0 # -1.0 * float(axis_angle[0].item()) / 100. * np.pi  # [-0.5, 0] -> sampled rotation angle
        # around the x-axis
        ''' Washing machine '''
        x_angle = -0.2 * np.pi # washing...
        # y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi - 0.25 * np.pi  # - 0.5 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 300. * np.pi # - 0.25 * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  # - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi - 0.25 * np.pi
        z_angle = 1.0 * float(axis_angle[2].item()) / 300. * np.pi # - 0.25 * np.pi
        ''' Washing machine '''

        ''' Oven --- v1 '''
        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 0.0 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi - 0.5 * np.pi
        z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi - 0.5 * np.pi
        ''' Oven --- v1 '''

        ''' Oven --- v2 (small range of view change) --- oven --- no axis/pv p '''
        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 0.0
        x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # y_angle = 0.0
        # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi
        ''' Oven --- v2 (small range of view change) --- oven --- no axis/pv p '''

        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p -- ok '''
        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 0.0
        # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # y_angle = 0.0
        # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        z_angle = 0.0  # - 0.5 * np.pi
        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p -- ok'''

        ''' Washing machine --- v1 --- no axis/pv p '''
        # # with sel-mode = 29 --> iou ~ 52+... global alignment: /share/xueyi/ckpt/playground/model_20220430_10:56:55/ckpt/playground_net_Iter1600.pth
        # axis_angle = np.random.randint(0, 100, (3,))
        # # x_angle = 0.0
        # x_angle = 1.0 / 8.0 * np.pi
        # # x_angle = -1.0 / 8.0 * np.pi
        # # x_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # # x_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi
        # # x_angle = 1.0 * float(axis_angle[1].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi
        #
        # # x_angle = -1.0 * float(axis_angle[1].item()) / 1600. * np.pi - 1.0 / 16.0 * np.pi # if we want to change x-angle...
        #
        # # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        #
        # # y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi # for global alignment
        # # y_angle = 1.0 * float(axis_angle[1].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi
        #
        # # y_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi
        # # y_angle = 1.0 * float(axis_angle[1].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi
        #
        # y_angle = 1.0 / 8.0 * np.pi
        # # y_angle = 0.0
        # # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi # for global alignment
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi  # - 0.5 * np.pi # for global alignment
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # # z_angle = 0.0  # - 0.5 * np.pi
        # z_angle = 0.0
        ''' Washing machine --- v1 --- no axis/pv p '''

        ''' eyeglasses '''
        # axis_angle = np.random.randint(0, 100, (3,))
        # x_angle = 1.0 * float(axis_angle[0].item()) / 100. * np.pi  - 0.5 * np.pi # [-0.5, 0] -> sampled rotation angle around the x-axis
        # # y_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # # z_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # # ''' Strategy 1 -- '''
        # y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi  - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  - 0.5 * np.pi
        # # z_angle = 0.0  # - 0.5 * np.pi
        ''' eyeglasses '''


        # y_angle = 0.0
        # z_angle = 0.0
        ''' Random angle strategy 2 '''

        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''
        # axis_angle = np.random.randint(0, 100, (3,))
        # x_angle =  1.0 * float(axis_angle[0].item()) / 200. * np.pi - 1.0 / 4.0 * np.pi
        # # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 200. * np.pi - 1.0 / 4.0 * np.pi
        # # y_angle = 0.0
        # # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle =  1.0 * float(axis_angle[2].item())  / 200. * np.pi - 1.0 / 4.0 * np.pi  # - 0.5 * np.pi
        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''

        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''
        axis_angle = np.random.randint(0, 100, (3,))
        # x_angle = 1.0 * float(axis_angle[0].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # y_angle = 0.0
        # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        ''' Oven --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''

        ''' Washing_machine --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''
        axis_angle = np.random.randint(0, 100, (3,))
        # x_angle = 1.0 * float(axis_angle[0].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        x_angle = 1.0 * float(axis_angle[0].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi
        # x_angle = 0.0
        # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi
        y_angle = 0.0
        # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        z_angle = 0.0
        ''' Washing_machine --- v2 (small range of view change) --- oven --- axis/pv p for vis only '''

        # axis_angle = np.random.randint(0, 100, (3,))
        # x_angle = 0.0
        # # x_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # # y_angle = -1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 6.0 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi
        # # y_angle = 0.0
        # # z_angle = -1.0 * float(axis_angle[2].item()) / 400. * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi  # - 0.5 * np.pi
        # # z_angle = 1.0 * float(axis_angle[2].item()) / 800. * np.pi  # - 0.5 * np.pi
        # z_angle = 0.0  # - 0.5 * np.pi
        # rotation = np.eye(3)
        #

        axis_angle = np.random.randint(0, 100, (3,))
        x_angle = 0.0 # [-0.5, 0] -> sampled rotation angle around the x-axis
        # y_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # z_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
        # ''' Strategy 1 -- '''
        # y_angle = 1.0 * float(axis_angle[1].item()) / 400. * np.pi - 1.0 / 8.0 * np.pi - 0.5 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi # - 0.5 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi # - 0.5 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi + 1.0 / 16.0 * np.pi # - 0.5 * np.pi
        # y_angle = 1.0 * float(axis_angle[1].item()) / 800. * np.pi - 1.0 / 16.0 * np.pi # - 0.5 * np.pi
        y_angle = 0.0
        # y_angle = 1.0 * float(axis_angle[1].item()) / 1600. * np.pi + 1.0 / 16.0 * np.pi # - 0.5 * np.pi
        # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  - 0.5 * np.pi
        z_angle = 0.0  # - 0.5 * np.pi

        # y_angle = 0.0

        x_mtx = compute_rotation_matrix_from_axis_angle(x_axis, x_angle)
        y_mtx = compute_rotation_matrix_from_axis_angle(y_axis, y_angle)
        z_mtx = compute_rotation_matrix_from_axis_angle(z_axis, z_angle)

        rotation = np.matmul(z_mtx, np.matmul(y_mtx, x_mtx))

        # rotation_angle = sciR.random().as_matrix()
        # rotation_matrix = rotation_angle[:3, :3]
        # R1 = rotation_matrix
        # rotation = R1

        # rotation = np.eye(3)

        # rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        # while upper_hemi and (rotation[1, 2] < 0 or rotation[2, 2] < 0):
        #     rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]

        # rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        # while upper_hemi and (rotation[1, 1] < 0 or rotation[2, 1] < 0):
        #     rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]

        randd = np.random.randn(3)
        pose_trans = mean_pose # + randd * std_pose
    else:
        rotation = np.eye(3)
        # pose_trans = np.zeros((3,))
        pose_trans = mean_pose
    # pose_trans = pose_trans * 0.0
    glb_pose = {'rotation': rotation, 'trans': pose_trans}
    for seg_label in seg_label_to_triangles:
        # cur_seg_triangles: n_tri
        cur_seg_triangles = seg_label_to_triangles[seg_label]
        cur_seg_triangles = triangles[cur_seg_triangles]
        # n_tri --> indexes of triangles...
        cur_seg_vertices, cur_seg_triangles = reindex_triangles_vertices(vertices, cur_seg_triangles)
        m = trimesh.base.Trimesh(vertices=cur_seg_vertices, faces=cur_seg_triangles)

        ''' Setup scene '''
        scene = pyrender.Scene()

        ''' Add mesh to the scene '''
        # add mesh to the node
        mesh = pyrender.Mesh.from_trimesh(m)
        node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        # add node to the scene
        scene.add_node(node)

        camera_pose = np.eye(4)
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=pw / ph, znear=near, zfar=far)
        projection = camera.get_projection_matrix()
        scene.add(camera, camera_pose)
        r = pyrender.OffscreenRenderer(pw, ph)

        # print(f"cur_seg_label: {seg_label}, projection matrix: {projection}")

        ''' Set pose '''
        pose = np.eye(4)  # init pose
        # pose[:3, 3] = mean_pose + randd * std_pose  # translation?
        pose[:3, 3] = pose_trans  # translation?
        # rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        # while upper_hemi and (rotation[1, 2] < 0 or rotation[2, 2] < 0):
        #     rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        pose[:3, :3] = rotation # rotation matrix

        cur_seg_art_rot = seg_transformation_mtx[seg_label, :3, :3]
        cur_seg_art_trans = seg_transformation_mtx[seg_label, :3, 3]
        cur_seg_rot = np.matmul(rotation, cur_seg_art_rot)
        cur_seg_trans = np.matmul(rotation, np.reshape(cur_seg_art_trans, (3, 1))) + np.reshape(pose[:3, 3], (3, 1))
        cur_seg_pose = np.eye(4)
        cur_seg_pose[:3, :3] = cur_seg_rot
        cur_seg_pose[:3, 3] = cur_seg_trans[:, 0]

        ''' Get new pose for this seg... '''
        seg_label_to_new_pose[seg_label] = cur_seg_pose

        scene.set_pose(node, pose)  # set pose for the node # the rotation?  --> rotate

        depth_buffer = r.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)  # render the depth buffer
        # pts = backproject(depth_buffer, projection, near, far, from_image=False)
        mask = depth_buffer > 0  # get the mask for valied area

        seg_label_to_depth_buffer[seg_label] = depth_buffer

        cur_seg_pts_idxes = np.array([ii for ii in range(tot_n, tot_n + depth_buffer.shape[0])], dtype=np.long)
        tot_n = tot_n + depth_buffer.shape[0]
        tot_depth_buffer.append(depth_buffer)
        seg_label_to_idxes[seg_label] = cur_seg_pts_idxes

        # from total depth buffer to depth matrix...
        # depth = np.concatenate(tot_depth_buffer, axis=0) #

    # tot_idxs =
    # height, width = depth.shape
    # non_zero_mask = (depth > 0)
    # idxs = np.where(non_zero_mask)
    # depth_selected = depth[idxs[0], idxs[1]].astype(np.float32).reshape((1, -1))

    # for seg_label in seg_label_to_depth_buffer:
    #     depth_buffer = seg_label_to_depth_buffer[seg_label]
    #     cur_seg_pose = seg_label_to_new_pose[seg_label]
    #     cur_seg_rot = cur_seg_pose[:3, :3]
    #     cur_seg_trans = np.reshape(cur_seg_pose[:3, 3], (3, 1))

        depth = depth_buffer

        proj_inv = np.linalg.inv(projection)
        height, width = depth.shape
        non_zero_mask = (depth > 0)
        idxs = np.where(non_zero_mask)
        depth_selected = depth[idxs[0], idxs[1]].astype(np.float32).reshape((1, -1))

        d = depth_selected
        z = buffer_depth_to_ndc(d, near, far)

        grid = np.array([idxs[1] / width * 2 - 1, 1 - idxs[0] / height * 2])  # ndc [-1, 1]

        ones = np.ones_like(z)
        # depth
        pts = np.concatenate((grid, z, ones), axis=0) * d  # before dividing by w, w = -z_world = d

        pts = proj_inv @ pts  # back project points
        pts = np.transpose(pts)  # transpose points

        pts = pts[:, :3]

        # inv transform points
        # no_rot_pts = np.matmul(np.transpose(cur_seg_rot, (1, 0)), np.transpose(pts - np.transpose(cur_seg_trans, (1, 0)), (1, 0)))
        # no_rot_pts = np.transpose(no_rot_pts, (1, 0))
        # # pts = pts - np.mean(pts, axis=0, keepdims=True)
        seg_label_to_pts[seg_label] = pts
        # seg_label_to_no_rot_pts[seg_label] = no_rot_pts
    return seg_label_to_pts, seg_label_to_new_pose, glb_pose



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
        self.shape_indexes = []

        shape_idxes = os.listdir(self.dataset_root)
        for shp_idx in shape_idxes:
            cur_shp_folder = os.path.join(self.dataset_root, shp_idx)
            self.shape_indexes.append(shp_idx)
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

        ''' nparts = None... '''
        # nparts = None
        # if self.shape_type == "eyeglasses":
        #     nparts = 2
        #     nparts = None

        shape_index = index
        cur_pts_fn = self.pts_folders[shape_index]
        cur_cfg_fn = self.cfg_folders[shape_index]

        sample_idx = index % 100
        shape_idx = index // 100

        cur_pts = np.load(cur_pts_fn, allow_pickle=True) # load current points... # load points
        # npts x 4
        cur_pts = np.transpose(cur_pts, (1, 0)) #
        cur_pts, cur_labels = cur_pts[:, :3], cur_pts[:, 3].astype(np.long) # labels
        cur_seg_label_to_pts_idxes = get_seg_labels_to_pts_idxes(cur_labels)

        # boundary_pts = [np.min(cur_pts, axis=0), np.max(cur_pts, axis=0)]
        # center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        #
        # # all normalize into 0
        # cur_pts = (cur_pts - center_pt.reshape(1, 3)) / length_bb

        cur_cfg = np.load(cur_cfg_fn, allow_pickle=True).item()
        cur_seg_label_to_motion_attr = {}
        cur_seg_label_to_vertices = {} # vertices idxes
        cur_seg_label_to_triangles = {}
        tot_vertices = []
        tot_triangels = []
        tot_n_triangles = 0
        tot_n_vertices = 0
        cur_n_part = 0
        print(f"cfg.keys: {cur_cfg.keys()}")
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
            cur_part_sampled_triangles_fn = os.path.join(self.dataset_root, self.shape_idxes[shape_idx], f"sample_mesh{sample_idx}_{cur_n_part}.obj")
            # cur_part_sampled_triangles_fn = os.path.join(self.dataset_root, self.shape_idxes[shape_idx], f"sample_mesh{sample_idx}_{cur_seg_label}.obj")
            cur_part_vertices, cur_part_triangles =  load_vertices_triangles(cur_part_sampled_triangles_fn)
            cur_part_vertices_idxes = np.array([tot_n_vertices + _ for _ in range(cur_part_vertices.shape[0])], dtype=np.long)
            cur_seg_label_to_vertices[cur_seg_label] = cur_part_vertices_idxes
            cur_part_triangles_idxes = np.array([tot_n_triangles + _ for _ in range(cur_part_triangles.shape[0])], dtype=np.long)
            cur_seg_label_to_triangles[cur_seg_label] = cur_part_triangles_idxes

            tot_vertices.append(cur_part_vertices)
            tot_triangels.append(cur_part_triangles + tot_n_vertices)
            tot_n_vertices += cur_part_vertices.shape[0]
            tot_n_triangles += cur_part_triangles.shape[0]
            cur_n_part += 1

        tot_vertices = np.concatenate(tot_vertices, axis=0)
        tot_triangels = np.concatenate(tot_triangels, axis=0)

        ori_pts = np.zeros_like(cur_pts)
        canon_transformed_pts = np.zeros_like(cur_pts)

        cur_sample_motion_states = DRAWER_COMBINATIONS[sample_idx]
        # cur_sample_motion_states = sorted(cur_sample_motion_states, reverse=True)
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

        ''' The following code aims to get articulation changed shape '''
        transformed_vertices = np.zeros((tot_vertices.shape[0], 3), dtype=np.float)
        canon_transformed_vertices = np.zeros((tot_vertices.shape[0], 3), dtype=np.float)

        tot_transformation_mtx_segs = []
        canon_tot_transformation_mtx_segs = []

        # and then you use the seg_idx_to_pts_idxes to get the triangles_idxes...
        #
        for cur_seg_label in range(4):
        # for cur_seg_label in [1, 0, 2, 3]: # seg_label to sampled_mesh and sampled_traiangles?
            cur_pts_idxes = cur_seg_label_to_pts_idxes[cur_seg_label]
            cur_seg_tri_vertices_idxes = cur_seg_label_to_vertices[cur_seg_label]

            if cur_seg_label not in cur_seg_label_to_motion_attr:
                ori_pts[cur_pts_idxes] = cur_pts[cur_pts_idxes]
                canon_transformed_pts[cur_pts_idxes] = cur_pts[cur_pts_idxes]
                cur_ref_trans = np.zeros((1, 3), dtype=np.float32)
                part_ref_trans.append(cur_ref_trans)
                cur_state_trans = np.zeros((1, 3), dtype=np.float32)
                part_state_trans.append(cur_state_trans)
                part_ref_trans_bbox.append(cur_ref_trans)
                part_state_trans_bbox.append(cur_state_trans)

                transformed_vertices[cur_seg_tri_vertices_idxes] = tot_vertices[cur_seg_tri_vertices_idxes]
                canon_transformed_vertices[cur_seg_tri_vertices_idxes] = canon_transformed_vertices[cur_seg_tri_vertices_idxes]
                transformed_rot = np.eye(3, dtype=np.float)
                transformed_trans = np.zeros((3, 1), dtype=np.float)

                canon_transformed_rot = np.eye(3, dtype=np.float)
                canon_transformed_trans = np.zeros((3, 1), dtype=np.float)

            else:
                # print(cur_seg_label)
                cur_state = cur_seg_label_to_motion_attr[cur_seg_label]['state']
                cur_dir = cur_seg_label_to_motion_attr[cur_seg_label]['dir']
                cur_a = cur_seg_label_to_motion_attr[cur_seg_label]['a']
                cur_b = cur_seg_label_to_motion_attr[cur_seg_label]['b']
                cur_seg_ori_pts = cur_pts[cur_pts_idxes] - np.reshape(cur_state * cur_dir, (1, 3))
                ori_pts[cur_pts_idxes] = cur_seg_ori_pts

                # and then you can also use motion state to transform triangles vertices of the segmentation
                cur_seg_motion_state = float(cur_sample_motion_states[consume_idx]) # cur_seg_motion_state
                # print(f"cur_seg_label: {cur_seg_label}, cur_motion_state: {cur_seg_motion_state}")
                canon_seg_motion_state = float(canon_motion_states[consume_idx]) # cur_seg_motion_state
                # cur_transformed_pts = cur_seg_ori_pts + np.reshape(cur_seg_motion_state * cur_dir, (1, 3))
                # cur_transformed_pts = cur_seg_ori_pts + np.reshape((cur_seg_motion_state * (cur_b - cur_a) + cur_a + 0.1) * cur_dir , (1, 3)) #
                # cur_transformed_pts = cur_seg_ori_pts + np.reshape((cur_seg_motion_state * 3.0 + 1.0) * cur_dir , (1, 3))
                cur_transformed_pts = cur_seg_ori_pts + np.reshape((cur_seg_motion_state) * cur_dir , (1, 3)) # motion state of segmentatioon
                # cur_transformed_pts = cur_seg_ori_pts + np.reshape((cur_seg_motion_state * (cur_b - cur_a) + cur_a) * cur_dir , (1, 3))

                cur_part_state_trans = np.reshape((cur_seg_motion_state) * cur_dir , (1, 3))
                part_state_trans.append(cur_part_state_trans)

                cur_part_state_trans_vertices = tot_vertices[cur_seg_tri_vertices_idxes] + cur_part_state_trans
                transformed_vertices[cur_seg_tri_vertices_idxes] = cur_part_state_trans_vertices

                part_axis.append(np.reshape(cur_dir, (1, 3)))

                transformed_rot = np.eye(3, dtype=np.float)
                transformed_trans = np.reshape(cur_part_state_trans, (3, 1))

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

                cur_part_ref_trans_vertices = tot_vertices[cur_seg_tri_vertices_idxes] + cur_part_ref_trans
                canon_transformed_vertices[cur_seg_tri_vertices_idxes] = cur_part_ref_trans_vertices

                ''' Get points state translations with centralized boudning box '''
                # rot_pts: n_pts_part x 3
                canon_rot_pts_minn = np.min(cur_canon_transformed_pts[:, :3], axis=0)
                canon_rot_pts_maxx = np.max(cur_canon_transformed_pts[:, :3], axis=0)
                canon_rot_pts_bbox_center = (canon_rot_pts_minn + canon_rot_pts_maxx) / 2.
                cur_part_ref_trans_bbox = cur_part_ref_trans[0] - canon_rot_pts_bbox_center
                part_ref_trans_bbox.append(np.reshape(cur_part_ref_trans_bbox, (1, 3)))

                canon_transformed_rot = np.eye(3, dtype=np.float)
                canon_transformed_trans = np.reshape(cur_part_ref_trans, (3, 1))

                ''' Get points state translations with centralized bouding box '''
                # rot_pts: n_pts_part x 3
                rot_pts_minn = np.min(cur_transformed_pts[:, :3], axis=0)
                rot_pts_maxx = np.max(cur_transformed_pts[:, :3], axis=0)
                rot_pts_bbox_center = (rot_pts_minn + rot_pts_maxx) / 2.
                cur_part_state_trans_bbox = cur_part_state_trans[0] - rot_pts_bbox_center
                part_state_trans_bbox.append(np.reshape(cur_part_state_trans_bbox, (1, 3)))

                pose[cur_pts_idxes, :3, 3] = cur_part_state_trans[0]
                pose_segs[cur_seg_label, :3, 3]  = cur_part_state_trans[0]

            cur_seg_transformation_mtx = np.concatenate([transformed_rot, transformed_trans], axis=1)
            cur_seg_transformation_mtx = np.concatenate([cur_seg_transformation_mtx, np.zeros((1, 4), dtype=np.float)], axis=0)
            tot_transformation_mtx_segs.append(cur_seg_transformation_mtx)

            cur_seg_canon_transformation_mtx = np.concatenate([canon_transformed_rot, canon_transformed_trans], axis=1)
            cur_seg_canon_transformation_mtx = np.concatenate([cur_seg_canon_transformation_mtx, np.zeros((1, 4), dtype=np.float)],
                                                    axis=0)
            canon_tot_transformation_mtx_segs.append(cur_seg_canon_transformation_mtx)

        tot_transformation_mtx_segs = np.array(tot_transformation_mtx_segs, dtype=np.float)
        canon_tot_transformation_mtx_segs = np.array(canon_tot_transformation_mtx_segs, dtype=np.float)
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

        # ins_num - # create partial points --> with global rotation and translation from one viewpoint
        seg_label_to_pts, seg_label_to_new_pose, glb_pose = create_partial_pts(mean_pose=np.array([0, 0, -1.8]),
                                                                               std_pose=np.array([0.2, 0.2, 0.15]),
                                                                               yfov=np.deg2rad(60), pw=640, ph=480,
                                                                               near=0.1, far=10, upper_hemi=True,
                                                                               vertices=transformed_vertices,
                                                                               triangles=tot_triangels,
                                                                               seg_label_to_triangles=cur_seg_label_to_triangles,
                                                                               seg_transformation_mtx=tot_transformation_mtx_segs,
                                                                               render_img=False)

        ''' Transform and render partial pc '''
        canon_seg_label_to_pts, canon_seg_label_to_new_pose, canon_glb_pose = create_partial_pts(
            mean_pose=np.array([0, 0, -1.8]),
            std_pose=np.array([0.2, 0.2, 0.15]),
            yfov=np.deg2rad(60), pw=640, ph=480, near=0.1,
            far=10, upper_hemi=True,
            vertices=canon_transformed_vertices,
            triangles=tot_triangels,
            seg_label_to_triangles=cur_seg_label_to_triangles,
            seg_transformation_mtx=canon_tot_transformation_mtx_segs,
            render_img=False, no_transformation=True)

        tot_transformed_full_vertices = []
        canon_tot_transformed_full_vertices = []
        tot_transformed_pts = []
        pts_to_seg_idx = []
        tot_transformation_mtx = []
        # seg label to
        canon_transformed_pts = []
        part_state_trans_bbox = np.zeros_like(part_ref_trans_bbox)
        glb_rotation, glb_trans = glb_pose['rotation'], glb_pose['trans']
        part_axis = np.matmul(glb_rotation, np.transpose(part_axis, (1, 0)))
        part_axis = np.transpose(part_axis, (1, 0))

        rotated_ori_pts = np.transpose(np.matmul(glb_rotation, np.transpose(ori_pts, (1, 0))), (1, 0)) + np.reshape(glb_trans, (1, 3))

        # global rotation: [3, 3] xx [
        # part_pv_point = np.matmul(glb_rotation, np.transpose(part_pv_point, (1, 0)))
        # get transformed part pv point
        # part_pv_point = np.transpose(part_pv_point, (1, 0)) + np.reshape(glb_trans, (1, 3))
        # part_pv_offset = part_pv_point - np.sum(part_pv_point * part_axis, axis=-1, keepdims=True) * part_axis
        # part_pv_offset = np.sqrt(np.sum(part_pv_offset ** 2, axis=-1))

        # print(f"seg_label_to_pts: {len(seg_label_to_pts)}, seg_label_to_new_pose: {len(seg_label_to_new_pose)}")

        # todo: set parT_state_trans_bbox, transformation_mtx_segs, part_state_rots, tot_transformed_pts, tot_full_transformed_vertices
        # todo: so what about the canon_transformed_vertices? --- should we also render it to partial observed? Oh yeah, otherwise where can you get points for the canonically transformed shape from?
        for seg_label in seg_label_to_pts:  #

            cur_seg_trans_pts = seg_label_to_pts[seg_label]
            cur_seg_pose = seg_label_to_new_pose[seg_label]
            canon_cur_seg_pose = canon_seg_label_to_new_pose[seg_label]
            tot_transformed_pts.append(cur_seg_trans_pts)  # partial transformed points
            pts_to_seg_idx += [seg_label for _ in range(cur_seg_trans_pts.shape[0])]

            canon_transformed_pts.append(canon_seg_label_to_pts[seg_label])

            ''' Get triangle idxes for this segmentation '''
            cur_seg_tri_idxes = cur_seg_label_to_triangles[seg_label]
            cur_seg_tri = tot_triangels[cur_seg_tri_idxes]
            cur_seg_tri_v1, cur_seg_tri_v2, cur_seg_tri_v3 = tot_vertices[cur_seg_tri[:, 0]], tot_vertices[
                cur_seg_tri[:, 1]], tot_vertices[cur_seg_tri[:, 2]]
            cur_seg_tri_vertices = np.concatenate([cur_seg_tri_v1, cur_seg_tri_v2, cur_seg_tri_v3], axis=0)
            cur_seg_tri_vertices_idxes = np.concatenate([cur_seg_tri[:, 0], cur_seg_tri[:, 1], cur_seg_tri[:, 2]],
                                                        axis=0)

            ''' Get transformed full vertices '''
            cur_seg_rot, cur_seg_trans = cur_seg_pose[:3, :3], cur_seg_pose[:3, 3]
            rot_cur_seg_tri_vertices = np.matmul(cur_seg_rot, np.transpose(cur_seg_tri_vertices, (1, 0))) + np.reshape(
                cur_seg_trans, (3, 1))
            rot_cur_seg_tri_vertices = np.transpose(rot_cur_seg_tri_vertices, (1, 0))

            tot_transformed_full_vertices.append(rot_cur_seg_tri_vertices)
            ''' Get transformed full vertices '''

            ''' Get  '''
            canon_cur_seg_rot, canon_cur_seg_trans = canon_cur_seg_pose[:3, :3], canon_cur_seg_pose[:3, 3]
            canon_rot_cur_seg_tri_vertices = np.matmul(canon_cur_seg_rot, np.transpose(cur_seg_tri_vertices, (1, 0))) + np.reshape(
                canon_cur_seg_trans, (3, 1))
            canon_rot_cur_seg_tri_vertices = np.transpose(canon_rot_cur_seg_tri_vertices, (1, 0))

            canon_tot_transformed_full_vertices.append(canon_rot_cur_seg_tri_vertices)

            # cur_seg_pose: 3 x 3; 3

            cur_seg_pts_minn = np.min(rot_cur_seg_tri_vertices, axis=0)
            cur_seg_pts_maxx = np.max(rot_cur_seg_tri_vertices, axis=0)
            cur_seg_pts_bbox_center = (cur_seg_pts_minn + cur_seg_pts_maxx) / 2.
            cur_seg_trans_bbox = cur_seg_trans - cur_seg_pts_bbox_center

            part_state_trans_bbox[seg_label] = cur_seg_trans_bbox
            part_state_orts[seg_label] = cur_seg_rot
            tot_transformation_mtx_segs[seg_label] = cur_seg_pose
            # tot_transformation_mtx.append([cur_seg_pose for _ in range(cur_seg_trans_pts.shape[0])])
            tot_transformation_mtx += [np.reshape(cur_seg_pose, (1, 4, 4)) for _ in range(cur_seg_trans_pts.shape[0])]

        tot_transformed_pts = np.concatenate(tot_transformed_pts, axis=0)
        tot_transformed_full_vertices = np.concatenate(tot_transformed_full_vertices, axis=0)
        canon_tot_transformed_full_vertices = np.concatenate(canon_tot_transformed_full_vertices, axis=0)
        # center_transformed_full_vertices = np.mean(tot_transformed_full_vertices, axis=0, keepdims=True)
        tot_transformation_mtx = np.concatenate(tot_transformation_mtx, axis=0)
        canon_transformed_pts = np.concatenate(canon_transformed_pts, axis=0)
        pts_to_seg_idx = np.array(pts_to_seg_idx, dtype=np.long)

        gt_pose = tot_transformation_mtx

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
        # part_state_orts[:, 0, 0] = 1.
        # part_state_orts[:, 1, 1] = 1.
        # part_state_orts[:, 2, 2] = 1.
        # part_ref_rots[:, 0, 0] = 1.
        # part_ref_rots[:, 1, 1] = 1.
        # part_ref_rots[:, 2, 2] = 1.
        # # part_ref_trans = np.zeros((part_ref_rots.shape[0], 3), dtype=np.float)
        #
        # ''' Add global rotation '''
        # if self.global_rot == 1 and (not (self.split == "train" and self.pre_compute_delta == 1)):
        #     if self.args.equi_settings.rot_anchors == 1:
        #         # just use a matrix from rotation anchors
        #         R1 = self.get_rotation_from_anchor()
        #     else:
        #         # R1 = generate_3d(smaller=True)
        #         rotation_angle = sciR.random().as_matrix()
        #         rotation_matrix = rotation_angle[:3, :3]
        #         R1 = rotation_matrix
        #     # rotate transformed points
        #     cur_pts = np.transpose(np.matmul(R1, np.transpose(cur_pts, [1, 0])), [1, 0])
        #     pose = np.matmul(np.reshape(R1, (1, 3, 3)), pose[:, :3, :])
        #     pose = np.concatenate([pose, np.zeros((cur_pts.shape[0], 1, 4), dtype=np.float)], axis=1)
        #
        #     pose_segs[:, :3, :] = np.matmul(np.reshape(R1, (1, 3, 3)), pose_segs[:, :3, :])
        #     part_state_orts[:, :3, :] = np.matmul(np.reshape(R1, (1, 3, 3)), part_state_orts[:, :3, :])
        #
        #     part_axis = np.transpose(np.matmul(R1, np.transpose(part_axis, (1, 0))), (1, 0))
        #
        #     part_state_trans = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(part_state_trans, (part_state_trans.shape[0], 3, 1)))
        #     part_state_trans = np.reshape(part_state_trans, (part_state_trans.shape[0], 3))
        #     part_state_trans_bbox = np.matmul(np.reshape(R1, (1, 3, 3)), np.reshape(part_state_trans_bbox, (part_state_trans_bbox.shape[0], 3, 1)))
        #     part_state_trans_bbox = np.reshape(part_state_trans_bbox, (part_state_trans_bbox.shape[0], 3))

        # glb_R, glb_T =

        # af_glb_center_pt = np.mean(tot_transformed_full_vertices, axis=0)
        # af_glb_center_pt = np.mean(canon_tot_transformed_full_vertices, axis=0)
        af_glb_center_pt = np.mean(rotated_ori_pts, axis=0)
        tot_transformed_pts = (tot_transformed_pts - af_glb_center_pt.reshape(1, 3))
        ''' Point normalization via centralization '''

        # latest work? aiaiaia...

        # all normalize into 0
        # sampled_pcts = (sampled_pcts - center_pt.reshape(1, 3)) / length_bb

        gt_pose[:, :3, 3] = gt_pose[:, :3, 3] - af_glb_center_pt  # transformation matrix...
        tot_transformation_mtx_segs[:, :3, 3] = tot_transformation_mtx_segs[:, :3, 3] - af_glb_center_pt
        # part_pv_point = part_pv_point - np.reshape(af_glb_center_pt, (1, 3))
        # part_pv_offset = part_pv_point - np.sum(part_pv_point * part_axis, axis=-1, keepdims=True) * part_axis
        # part_pv_offset = np.sqrt(np.sum(part_pv_offset ** 2, axis=-1))

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
        # boundary_pts = [np.min(ori_pts, axis=0), np.max(ori_pts, axis=0)]
        # center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # all normalize into 0
        # cur_pts = (cur_pts - center_pt.reshape(1, 3)) / length_bb #
        # gt_pose[:, :3, 3] = (gt_pose[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb
        # tot_transformation_mtx_segs[:, :3, 3] = (tot_transformation_mtx_segs[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb
        # part_state_trans = (part_state_trans - center_pt.reshape(1, 3)) / length_bb
        # part_state_trans_bbox = part_state_trans_bbox / length_bb

        ''' Centralize points in the canonical state '''
        # canon_boundary_pts = [np.min(ori_pts, axis=0), np.max(ori_pts, axis=0)]
        # canon_center_pt = (canon_boundary_pts[0] + canon_boundary_pts[1]) / 2
        # canon_length_bb = np.linalg.norm(canon_boundary_pts[0] - canon_boundary_pts[1])

        # all normalize into 0
        # canon_transformed_pts = (canon_transformed_pts - canon_center_pt.reshape(1, 3)) / canon_length_bb  #
        # # [:, :3, 3] = (pose[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb
        # # pose_segs[:, :3, 3] = (pose_segs[:, :3, 3] - center_pt.reshape(1, 3)) / length_bb
        # part_ref_trans = (part_ref_trans - canon_center_pt.reshape(1, 3)) / canon_length_bb
        # part_ref_trans_bbox = part_ref_trans_bbox / canon_length_bb


        # cur_pc = torch.from_numpy(cur_pts.astype(np.float32)).float()
        cur_pc = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        tot_transformed_pts = torch.from_numpy(tot_transformed_pts.astype(np.float32)).float()
        # cur_label = torch.from_numpy(cur_labels).long()
        # tot_label = torch.from_numpy(cur_labels).long()
        cur_label = torch.from_numpy(pts_to_seg_idx).long()
        tot_label = torch.from_numpy(pts_to_seg_idx).long()
        cur_pose = torch.from_numpy(gt_pose.astype(np.float32))
        cur_pose_segs = torch.from_numpy(tot_transformation_mtx_segs.astype(np.float32))
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

        # cur_ori_pc = cur_ori_pc[fps_idx]
        canon_fps_idx = farthest_point_sampling(cur_canon_transformed_pts.unsqueeze(0), n_sampling=self.npoints)
        canon_fps_idx = canon_fps_idx[:self.npoints]
        canon_fps_idx_oorr = farthest_point_sampling(cur_canon_transformed_pts.unsqueeze(0), n_sampling=4096)
        cur_oorr_canon_transformed_pts = cur_canon_transformed_pts[canon_fps_idx_oorr]
        cur_canon_transformed_pts = cur_canon_transformed_pts[canon_fps_idx]

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
            'ori_pc': tot_transformed_pts.contiguous().transpose(0, 1).contiguous(),
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
