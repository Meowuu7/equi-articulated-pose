import trimesh
import numpy as np
import os
import multiprocessing as mp
import gc
import argparse
from tqdm import tqdm
# import pyrender
# import matplotlib.pyplot as plt
import cv2
from os.path import join as pjoin
import pickle
from dt_part_transform import *
from dt_data_utils import *

# from OpenGL import platform, _configflags
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender

try:
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.look_at((0., 0.0, 1.5), (0., 0., 1.))
    ps.set_screenshot_extension(".png")
except:
    pass

print("here1")

color = [
    (136/255.0,224/255.0,239/255.0),
    (180/255.0,254/255.0,152/255.0),
    (184/255.0,59/255.0,94/255.0),
    (106/255.0,44/255.0,112/255.0),
    (39/255.0,53/255.0,135/255.0),
(0,173/255.0,181/255.0), (170/255.0,150/255.0,218/255.0), (82/255.0,18/255.0,98/255.0), (234/255.0,84/255.0,85/255.0), (234/255.0,255/255.0,208/255.0),(162/255.0,210/255.0,255/255.0),
    (187/255.0,225/255.0,250/255.0), (240/255.0,138/255.0,93/255.0), (184/255.0,59/255.0,94/255.0),(106/255.0,44/255.0,112/255.0),(39/255.0,53/255.0,135/255.0),
]

def bp():
    import pdb;pdb.set_trace()

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

def compute_rotation_matrix_from_axis_angle(axis, angle):

    u, v, w = axis[0].item(), axis[1].item(), axis[2].item()
    costheta = np.cos(angle)
    sintheta = np.sin(angle)

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    m = np.zeros((3, 3), dtype=np.float)


    # m = torch.zeros((n_angles, na, 3, 3), dtype=torch.float32)  # .cuda()
    # print(uu.size(), costheta.size())
    m[0, 0] = uu + (vv + ww) * costheta
    m[1, 0] = uv * (1 - costheta) + w * sintheta
    m[2, 0] = uw * (1 - costheta) - v * sintheta

    m[0, 1] = uv * (1 - costheta) - w * sintheta
    m[1, 1] = vv + (uu + ww) * costheta
    m[2, 1] = vw * (1 - costheta) + u * sintheta

    m[0, 2] = uw * (1 - costheta) + v * sintheta
    m[1, 2] = vw * (1 - costheta) - u * sintheta
    m[2, 2] = ww + (uu + vv) * costheta
    return m

def create_partial_pts(read_path, save_folder, ins_num, render_num,
                   mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
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


    axis_angle = np.random.randint(0, 50, (3,))

    ''' Angle strategy 1 '''
    # x_angle = -1.0 * float(axis_angle[0].item()) / 100. * np.pi # [-0.5, 0] -> sampled rotation angle around the x-axis
    # # y_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
    # # z_angle = float(axis_angle[0].item()) / 50. * np.pi - 0.5 * np.pi
    # y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi # - 0.5 * np.pi # + is better than -
    # z_angle = 1.0 * float(axis_angle[2].item()) / 150. * np.pi # - 0.5 * np.pi
    # z_angle = 0.0 # - 0.5 * np.pi
    # it seems that two positive is a good choice
    ''' Angle strategy 1 '''

    ''' Angle strategy 2 '''
    x_angle = 0.0  # -1.0 * float(axis_angle[0].item()) / 100. * np.pi  # [-0.5, 0] -> sampled rotation angle around the x-axis
    y_angle = 1.0 * float(axis_angle[1].item()) / 100. * np.pi - 0.25 * np.pi  # - 0.5 * np.pi
    # y_angle = 1.0 * float(axis_angle[1].item()) / 300. * np.pi # - 0.25 * np.pi  # - 0.5 * np.pi
    # z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi  # - 0.5 * np.pi
    # z_angle = 0.0  # - 0.5 * np.pi
    z_angle = 1.0 * float(axis_angle[2].item()) / 100. * np.pi - 0.25 * np.pi
    ''' Angle strategy 2 '''

    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float)

    x_mtx = compute_rotation_matrix_from_axis_angle(x_axis, x_angle)
    y_mtx = compute_rotation_matrix_from_axis_angle(y_axis, y_angle)
    z_mtx = compute_rotation_matrix_from_axis_angle(z_axis, z_angle)

    if not no_transformation:
        rotation = np.matmul(z_mtx, np.matmul(y_mtx, x_mtx))
        randd = np.random.randn(3)
        pose_trans = mean_pose + randd * std_pose
    else:
        rotation = np.eye(3)
        pose_trans = mean_pose

    # rotation = np.eye(3)
    ''' Upper hemisphere transformation '''
    # rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
    # while upper_hemi and ((rotation[1, 2] < 0 or rotation[2, 2] < 0)):
    #     rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
    ''' Upper hemisphere transformation '''

    # pose_trans = pose_trans * 0.0
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
        no_rot_pts = np.matmul(np.transpose(cur_seg_rot, (1, 0)), np.transpose(pts - np.transpose(cur_seg_trans, (1, 0)), (1, 0)))
        no_rot_pts = np.transpose(no_rot_pts, (1, 0))
        # pts = pts - np.mean(pts, axis=0, keepdims=True)
        seg_label_to_pts[seg_label] = pts
        seg_label_to_no_rot_pts[seg_label] = no_rot_pts
    return seg_label_to_pts, seg_label_to_no_rot_pts, seg_label_to_new_pose




def create_partial(read_path, save_folder, ins_num, render_num,
                   mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
                   yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=False, vertices=None, triangles=None, render_img=True):
    if vertices is not None and triangles is not None:
        m = trimesh.base.Trimesh(vertices=vertices, faces=triangles)
    else:
        m = trimesh.load(read_path)
    # centralization
    c = np.mean(m.vertices, axis=0)
    trans = np.eye(4)
    trans[:3, 3] = -c
    m.apply_transform(trans) # pose of each part --- should be remembered and transformed for the further use
    # for further

    # scale = np.max(np.sqrt(np.sum(m.vertices ** 2, axis=1)))
    # trans = np.eye(4)
    # trans[:3, :3] = np.eye(3) / scale # sacle #
    # m.apply_transform(trans)

    scene = pyrender.Scene()

    mesh = pyrender.Mesh.from_trimesh(m)
    node = pyrender.Node(mesh=mesh, matrix=np.eye(4))

    scene.add_node(node)

    camera_pose = np.eye(4)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=pw / ph, znear=near, zfar=far)
    projection = camera.get_projection_matrix()
    scene.add(camera, camera_pose)
    r = pyrender.OffscreenRenderer(pw, ph)

    if render_img:
        depth_path = pjoin(save_folder, ins_num, 'depth')
        os.makedirs(depth_path, exist_ok=True)
        gt_path = pjoin(save_folder, ins_num, 'gt')
        os.makedirs(gt_path, exist_ok=True)
    # else:

    depths_zs = []

    for i in range(render_num):
        pose = np.eye(4) # init pose
        pose[:3, 3] = mean_pose + np.random.randn(3) * std_pose # translation?
        rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        while upper_hemi and (rotation[1, 2] < 0 or rotation[2, 2] < 0):
            rotation = trimesh.transformations.random_rotation_matrix()[:3, :3]
        pose[:3, :3] = rotation # rotation matrix
        scene.set_pose(node, pose) # set pose for the node # the rotation?  --> rotate
        depth_buffer = r.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY) # render the depth buffer
        # pts = backproject(depth_buffer, projection, near, far, from_image=False)
        mask = depth_buffer > 0 # get the mask for valied area

        if render_img:
            depth_z = buffer_depth_to_ndc(depth_buffer, near, far)  # [-1, 1] # ??? ndc? --- just for value transformation...
            depth_image = depth_z * 0.5 + 0.5  # [0, 1] #
            depth_image = linearize_img(depth_image, near, far)  # [0, 1]
            depth_image = np.uint16((depth_image * mask) * ((1 << 16) - 1)) # image's pixel values
            cv2.imwrite(pjoin(depth_path, f'{i:03}.png'), depth_image) # write the depth image
            np.save(pjoin(gt_path, f'{i:03}.npy'), pose) # we can use this to add global pose
            # backproject(depth_image, projection, near, far, from_image=True, vis=True)
        depths_zs.append(depth_buffer)

    return projection, near, far, depths_zs


def proc_render(first, path_list, save_folder, render_num,
                mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
                yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=False):
    for read_path in tqdm(path_list):
        ins_num = read_path.split('/')[-1].split('.')[-2].split('_')[-1]
        projection, near, far, depth_zs = create_partial(read_path, save_folder, ins_num, render_num,
                                               mean_pose, std_pose,
                                               yfov, pw, ph, near, far, upper_hemi)
    if first:
        meta_path = pjoin(save_folder, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({'near': near, 'far': far, 'projection': projection}, f)


def ndc_depth_to_buffer(z, near, far):  # z in [-1, 1]
    return 2 * near * far / (near + far - z * (far - near))


def buffer_depth_to_ndc(d, near, far):  # d in (0, +
    return ((near + far) - 2 * near * far / np.clip(d, a_min=1e-6, a_max=1e6)) / (far - near)


def linearize_img(d, near, far):  # for visualization only
    return 2 * near / (near + far - d * (far - near))


def inv_linearize_img(d, near, far):  # for visualziation only
    return (near + far - 2 * near / d) / (far - near)


# for each globally rotated pc, create its K depth map from K different view... and then backproject it to the canon view, right?
def backproject(depth, projection, near, far, from_image=False, vis=False):
    proj_inv = np.linalg.inv(projection)
    height, width = depth.shape
    non_zero_mask = (depth > 0)
    idxs = np.where(non_zero_mask)
    depth_selected = depth[idxs[0], idxs[1]].astype(np.float32).reshape((1, -1))
    if from_image:
        z = depth_selected / ((1 << 16) - 1)  # [0, 1]
        z = inv_linearize_img(z, near, far)  # [0, 1]
        z = z * 2 - 1.0  # [-1, 1]
        d = ndc_depth_to_buffer(z, near, far)
    else:
        d = depth_selected
        z = buffer_depth_to_ndc(d, near, far)

    grid = np.array([idxs[1] / width * 2 - 1, 1 - idxs[0] / height * 2])  # ndc [-1, 1]

    ones = np.ones_like(z)
    # depth
    pts = np.concatenate((grid, z, ones), axis=0) * d  # before dividing by w, w = -z_world = d

    pts = proj_inv @ pts # back project points
    pts = np.transpose(pts) # transpose points

    pts = pts[:, :3]

    if vis:
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=1)
        ax.set_xlim3d([center[0] - lim, center[0] + lim])
        ax.set_ylim3d([center[1] - lim, center[1] + lim])
        ax.set_zlim3d([center[2] - lim, center[2] + lim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    return pts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate_num', type=int, default=10)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--category', type=str, default='airplane')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_proc', type=int, default=8)
    parser.add_argument('--upper_hemi', action='store_true')

    args = parser.parse_args()
    return args

# ori mesh -> rot and articulation change -> current mesh -> rendered depth map -> back-propagated points;

def inv_triangles_to_seg_idx(triangles_to_seg_idx):
    # triangles_to_seg_idx: N_tri
    seg_idx_to_triangles_idxes = {}
    for i_t in range(triangles_to_seg_idx.shape[0]):
        cur_tri_seg_idx = int(triangles_to_seg_idx[i_t].item())
        if cur_tri_seg_idx not in seg_idx_to_triangles_idxes:
            seg_idx_to_triangles_idxes[cur_tri_seg_idx] = [i_t]
        else:
            seg_idx_to_triangles_idxes[cur_tri_seg_idx].append(i_t)
    for seg_idx in seg_idx_to_triangles_idxes:
        seg_idx_to_triangles_idxes[seg_idx] = np.array(seg_idx_to_triangles_idxes[seg_idx], dtype=np.long) # N_seg_tri
    return seg_idx_to_triangles_idxes


def test_depth_render(shape_type='oven', root=None, split='train'):
    dataset_root = f"/Users/meow/Study/_2021_spring/part-segmentation/MDV02/{shape_type}"
    root = dataset_root if root is None else root
    nparts = None
    if shape_type == "eyeglasses": #
        nparts = 2
        nparts = None

    file_names = os.listdir(root)
    file_names = [fnn for fnn in file_names if fnn != ".DS_Store"]
    mesh_fn = "summary.obj"
    surface_to_seg_fn = "sfs_idx_to_dof_name_idx.npy"
    attribute_fn = "motion_attributes.json"
    npoints = 1024
    rot_factor = 0.5

    ''' Get file names '''
    cur_cat_not_ok_idxes = MOTION_CATEGORY_TO_NOT_OK_IDX_LIST[shape_type]
    cur_cat_test_idxes = MOTION_CATEGORY_TO_TEST_IDX_LIST[shape_type]

    cur_cat_not_ok_idxes = ["%.4d" % iii for iii in cur_cat_not_ok_idxes]
    cur_cat_test_idxes = ["%.4d" % iii for iii in cur_cat_test_idxes]
    cur_cat_not_ok_train_idxes = cur_cat_not_ok_idxes + cur_cat_test_idxes
    if split == 'train':
        file_names = [cur_fn for cur_fn in file_names if cur_fn not in cur_cat_not_ok_train_idxes]
    else:
        file_names = cur_cat_test_idxes
    ''' Get file names '''

    ''' Set and make directionaries '''
    save_folder = "/share/xueyi/proj_data/Motion_Depth/"
    save_folder = os.path.join(save_folder, shape_type)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_folder = os.path.join(save_folder, split)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    ''' Set and make directionaries '''


    n_samples_per_instance = 100
    n_depth_samples_per_instance = 5

    ''' Set dummy axis and dummy center '''
    dummy_axis = np.array([0.0, 1.0, 0.0], dtype=np.float)
    dummy_center = np.array([0.0, 0.0, 0.0], dtype=np.float)

    # file_names = ["0034", "0037", "0039"]
    # file_names = ['0027', '0034']

    # file_names = ['0025', '0034', '0025', '0034', '0025', '0034']

    #
    # file_names = ['0057', '0059', '0060', '0061', '0062']
    # file_names = ['0039', '0040', '0041', '0043']
    # file_names = ['0057', '0059', '0060', '0061']
    # file_names = ['0056', '0057', '0060']
    # file_names = ['0076', '0077', '0078', '0079', '0080', '0081', '0082', '0084', '0085']
    # 77, 80, 81

    for fn in file_names:
        cur_ins_save_folder = os.path.join(save_folder, fn)
        if not os.path.exists(cur_ins_save_folder):
            os.mkdir(cur_ins_save_folder)

        print(fn)
        # try:
        # Set current folder, mesh file name, triangle index to seg index, and motion attributes
        cur_folder = os.path.join(root, fn)
        cur_mesh_fn = os.path.join(cur_folder, mesh_fn)
        cur_surface_to_seg_fn = os.path.join(cur_folder, surface_to_seg_fn)
        cur_motion_attributes_fn = os.path.join(cur_folder, attribute_fn)

        # vertices and triangles
        cur_vertices, cur_triangles = load_vertices_triangles(cur_mesh_fn)
        # cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn, nparts=2)
        # a triangle: [v1, v2, v3]; rotate vertices --> transformed vertices;
        cur_triangles_to_seg_idx, seg_idx_to_triangle_idxes = load_triangles_to_seg_idx(cur_surface_to_seg_fn)
        cur_triangles, cur_triangles_to_seg_idx = refine_triangle_idxes_by_seg_idx(seg_idx_to_triangle_idxes, cur_triangles)
        seg_idx_to_triangle_idxes = inv_triangles_to_seg_idx(cur_triangles_to_seg_idx)
        cur_motion_attributes = load_motion_attributes(cur_motion_attributes_fn)

        # sampled_pcts, pts_to_seg_idx, seg_idx_to_sampled_pts = sample_pts_from_mesh(cur_vertices, cur_triangles,
        #                                                                             cur_triangles_to_seg_idx,
        #                                                                                npoints=npoints)

        # From vertices to bounding box (points and diagonal length?)
        ''' Normalize vertices of the shape '''
        boundary_pts = [np.min(cur_vertices, axis=0), np.max(cur_vertices, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        # all normalize into 0
        cur_vertices = (cur_vertices - center_pt.reshape(1, 3)) / length_bb
        ''' Normalize vertices of the shape '''

        for sample_idx in range(n_samples_per_instance): # sample_idx -->
            cur_art_sample_save_folder = os.path.join(cur_ins_save_folder, "%.3d" % sample_idx)
            if not os.path.exists(cur_art_sample_save_folder):
                os.mkdir(cur_art_sample_save_folder)
            # sample_idx, instance_idx...
            # transformed_vertices: n_vertices x 3
            transformed_vertices = np.zeros((cur_vertices.shape[0], 3), dtype=np.float)
            canon_transformed_vertices = np.zeros((cur_vertices.shape[0], 3), dtype=np.float)

            tot_transformed_pts = []
            tot_transformation_mtx = []
            tot_transformation_mtx_segs = []
            tot_canon_transformation_mtx_segs = []

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
                cur_seg_tri_idxes = seg_idx_to_triangle_idxes[i_seg]
                cur_seg_tri = cur_triangles[cur_seg_tri_idxes]
                cur_seg_tri_v1, cur_seg_tri_v2, cur_seg_tri_v3 = cur_vertices[cur_seg_tri[:, 0]], cur_vertices[cur_seg_tri[:, 1]], cur_vertices[cur_seg_tri[:, 2]]
                cur_seg_tri_vertices = np.concatenate([cur_seg_tri_v1, cur_seg_tri_v2, cur_seg_tri_v3], axis=0)
                cur_seg_tri_vertices_idxes = np.concatenate([cur_seg_tri[:, 0], cur_seg_tri[:, 1], cur_seg_tri[:, 2]], axis=0)
                # cur_seg_pts_idxes = np.array(seg_idx_to_sampled_pts[i_seg], dtype=np.long)
                # cur_seg_pts = sampled_pcts[cur_seg_pts_idxes]
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
                        # theta = (np.random.uniform(0.5, 1., (1,)).item() * np.pi) * rot_factor
                        # Get rotation degree of this sample;
                        theta = (((80. / 180.) / n_samples_per_instance) * sample_idx + 45. / 180.) * np.pi
                    else:
                        theta = (np.random.uniform(0., 1., (1,)).item() * np.pi) * rot_factor
                    # print(theta)
                    center = (center - center_pt) / length_bb
                    ''' Get transformed triangle vertices and its corresponding transformation matrix '''
                    rot_cur_seg_tri_vertices, transformation_mtx = revolute_transformation(cur_seg_tri_vertices, center, axis, theta, mode=1)
                    ''' Reshape the transformation matrix '''
                    transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                    ''' Get transformed triangle vertices in the canonical space '''
                    canon_theta = 0.5 * np.pi
                    canon_rot_cur_seg_tri_vertices, canon_transformation_mtx = revolute_transformation(cur_seg_tri_vertices, center, axis, canon_theta, mode=1)
                    ''' Reshape '''
                    canon_transformation_mtx = np.reshape(canon_transformation_mtx, (1, 4, 4))

                    ''' Add to list '''
                    # tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                    tot_transformation_mtx_segs.append(transformation_mtx)
                    tot_canon_transformation_mtx_segs.append(canon_transformation_mtx)

                    ''' Set vertices/points '''
                    transformed_vertices[cur_seg_tri_vertices_idxes] = rot_cur_seg_tri_vertices[:, :3]
                    canon_transformed_vertices[cur_seg_tri_vertices_idxes] = canon_rot_cur_seg_tri_vertices[:, :3]

                else:
                    ''' Transform ponts '''
                    theta = 0.0 * np.pi
                    cur_seg_tri_vertices, transformation_mtx = revolute_transformation(cur_seg_tri_vertices, dummy_center, dummy_axis, theta, mode=1)
                    ''' Reshape the transformation matrix '''
                    transformation_mtx = np.reshape(transformation_mtx, (1, 4, 4))

                    ''' Set vertices/points '''
                    transformed_vertices[cur_seg_tri_vertices_idxes] = cur_seg_tri_vertices[:, :3]
                    canon_transformed_vertices[cur_seg_tri_vertices_idxes] = cur_seg_tri_vertices[:, :3]

                    # tot_transformation_mtx += [transformation_mtx for _ in range(cur_seg_pts.shape[0])]
                    tot_transformation_mtx_segs.append(transformation_mtx)
                    tot_canon_transformation_mtx_segs.append(transformation_mtx)
            # transformed points...
            # tot_transformed_pts = np.concatenate(tot_transformed_pts, axis=0)
            # print(tot_transformed_pts.shape)
            # transformed_vertices; triangles; triangles_to_seg_label; --> then we can use them for furhter rendering..
            '''  '''
            tot_transformation_mtx_segs = np.concatenate(tot_transformation_mtx_segs, axis=0)
            tot_canon_transformation_mtx_segs = np.concatenate(tot_canon_transformation_mtx_segs, axis=0)

            # center = np.mean(transformed_vertices, axis=0, keepdims=True)
            # transformed_vertices = transformed_vertices - center
            # tot_transformation_mtx_segs[:, :3, 3] = tot_transformation_mtx_segs[:, :3, 3] - center

            render_num = 5
            ins_num = int(fn)

            ''' Render depth point clouds and save them to the art-folder '''
            for i_depth in range(n_depth_samples_per_instance):
                ''' Render the shape with the current articulation change: 1) Transformed vertices and 2) Total transformed matrix segs '''
                # todo: perhaps we can tune the `mean pose` value for better rendering
                seg_label_to_pts, seg_label_to_no_rot_pts, seg_label_to_new_pose = create_partial_pts(None, None, ins_num, render_num,
                                   mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
                                   yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=True, vertices=transformed_vertices, triangles=cur_triangles, seg_label_to_triangles=seg_idx_to_triangle_idxes, seg_transformation_mtx=tot_transformation_mtx_segs, render_img=False)


                cur_depth_instance = {
                    'seg_label_to_pts': seg_label_to_pts,
                    'seg_label_to_pose': seg_label_to_new_pose,
                    # 'seg_label_to_canon_pts': seg_label_to_canon_pts,
                    # 'seg_label_to_canon_pose': seg_label_to_canon_new_pose,
                    'category': shape_type,
                    'shape_idx': int(fn),
                    'sample_idx': sample_idx,
                    'depth_idx': i_depth,
                }
                cur_depth_instance_save_fn = os.path.join(cur_art_sample_save_folder, f"depth_{i_depth}.npy")
                np.save(cur_depth_instance_save_fn, cur_depth_instance)


            ''' Render the shape with the canonical articulation change: 1) Transformed vertices and 2) Total transformed matrix segs '''
            # todo: perhaps we can tune the `mean pose` value for better rendering
            seg_label_to_canon_pts, seg_label_to_canon_no_rot_pts, seg_label_to_canon_new_pose = create_partial_pts(
                None, None, ins_num, render_num, mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
                yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=True,
                vertices=canon_transformed_vertices,
                triangles=cur_triangles,
                seg_label_to_triangles=seg_idx_to_triangle_idxes,
                seg_transformation_mtx=tot_canon_transformation_mtx_segs,
                render_img=False,
                no_transformation=True
            )

            cur_canon_depth_instance = {
                'seg_label_to_pts': seg_label_to_canon_pts,
                'seg_label_to_pose': seg_label_to_canon_new_pose,
                'category': shape_type,
                'shape_idx': int(fn),
                'sample_idx': sample_idx,
            }
            cur_canon_depth_instance_save_fn = os.path.join(cur_art_sample_save_folder, f"canon_depth.npy")
            np.save(cur_canon_depth_instance_save_fn, cur_canon_depth_instance)

            # tot_pts = []
            #
            #
            # for seg_label in seg_label_to_pts:
            #     cur_seg_pts = seg_label_to_pts[seg_label]
            #     ps.register_point_cloud(f"depth_pts_{seg_label}", cur_seg_pts, radius=0.012, color=color[seg_label + 1])
            #     tot_pts.append(cur_seg_pts)
            #
            # ps.show()
            # ps.remove_all_structures()
            #
            # tot_pts = np.concatenate(tot_pts, axis=0)
            # tot_pts_center = np.mean(tot_pts, axis=0, keepdims=True)
            #
            # for seg_label in seg_label_to_pts:
            #     cur_seg_pts = seg_label_to_pts[seg_label] - tot_pts_center
            #     ps.register_point_cloud(f"depth_pts_{seg_label}", cur_seg_pts, radius=0.012, color=color[seg_label + 1])
            #     # tot_pts.append(cur_seg_pts)
            #
            #
            # ps.show()
            # ps.remove_all_structures()
            #
            # # whether
            #
            # for seg_label in seg_label_to_no_rot_pts:
            #     cur_seg_pts = seg_label_to_no_rot_pts[seg_label]
            #     ps.register_point_cloud(f"depth_pts_no_rot_{seg_label}", cur_seg_pts, radius=0.012, color=color[seg_label + 1])
            # ps.register_point_cloud(f"ori_pts", sampled_pcts, radius=0.012, color=color[0])
            # ps.show()
            # ps.remove_all_structures()

            # projection, near, far, depth_zs = create_partial(None, None, ins_num, render_num,
            #                    mean_pose=np.array([0, 0, -1.8]), std_pose=np.array([0.2, 0.2, 0.15]),
            #                    yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10, upper_hemi=True, vertices=transformed_vertices, triangles=cur_triangles, render_img=False)
            #
            # # ps.register_point_cloud(f"ori_pts", sampled_pcts, radius=0.012, color=color[0])
            # for i_depth, depth_z in enumerate(depth_zs):
            #     cur_depth_pts = backproject(depth_z, projection, near, far, from_image=False, vis=False)
            #     ps.register_point_cloud(f"depth_pts_{i_depth}", cur_depth_pts, radius=0.012, color=color[i_depth + 1])
            #     ps.show()
            #     ps.remove_all_structures()

            # gt_pose = np.zeros((self.npoints, 4, 4), dtype=np.float)
            # gt_pose[:, 0, 0] = 1.; gt_pose[:, 1, 1] = 1.; gt_pose[:, 2, 2] = 1.

            ''' Use GT transformation matrix as initial pose '''
                # tot_transformation_mtx = np.concatenate(tot_transformation_mtx, axis=0)
                # tot_transformation_mtx_segs = np.concatenate(tot_transformation_mtx_segs, axis=0)

                # ps.register_point_cloud(f"ori_pts", sampled_pcts, radius=0.012, color=color[0])
                # ps.show()
                # ps.remove_all_structures()

                # ps.register_point_cloud(f"seg_transformed_pts_", tot_transformed_pts, radius=0.012, color=color[4])
                # ps.show()
                # ps.remove_all_structures()

                # for seg_label in seg_label_to_transformed_pts:
                #     cur_seg_transformed_pts = seg_label_to_transformed_pts[seg_label]
                #     ps.register_point_cloud(f"seg_transformed_pts_{seg_label}", cur_seg_transformed_pts, radius=0.012,
                #                             color=color[seg_label])
                # # ps.register_point_cloud(f"transformed_pts", tot_transformed_pts, radius=0.012, color=color[0])
                # ps.show()
                # ps.remove_all_structures()

                # for seg_label in seg_label_to_transformed_pts:
                #     cur_seg_transformed_pts = seg_label_to_transformed_pts[seg_label]
                #     if seg_label == 0:
                #         ps.register_point_cloud(f"seg_transformed_pts_{seg_label}", cur_seg_transformed_pts, radius=0.012,
                #                                 color=color[seg_label])
                #     else:
                #         ps.register_point_cloud(f"seg_transformed_pts_{seg_label}_a",
                #                                 cur_seg_transformed_pts[:cur_seg_transformed_pts.shape[0] // 10],
                #                                 radius=0.012,
                #                                 color=color[seg_label])
                #         ps.register_point_cloud(f"seg_transformed_pts_{seg_label}_b",
                #                                 cur_seg_transformed_pts[cur_seg_transformed_pts.shape[0] // 10:],
                #                                 radius=0.012,
                #                                 color=color[0])
                # # ps.register_point_cloud(f"transformed_pts", tot_transformed_pts, radius=0.012, color=color[0])
                # ps.show()
                # ps.remove_all_structures()
            # except:
            #     continue

if __name__ == "__main__":

    test_depth_render(shape_type='oven', root="/home/xueyi/EPN2/data/MDV02/oven")
    exit(0)

    args = parse_args()

    # oven: 5, 2, 20, 18, 27, 11, 42, 28, 17, 10, 36, 9, 30, 8, 1, 6, 24, 15, 14

    read_folder = pjoin(args.input, args.category, args.split)
    save_folder = pjoin(args.output, args.category, args.split)
    path_list = [pjoin(read_folder, i) for i in os.listdir(read_folder) if i.endswith('off')]

    os.makedirs(save_folder, exist_ok=True)

    # Mean pose for each category
    mean_pose_dict = {
        'airplane': np.array([0, 0, -1.8]), # pose for the cetegory
        'car': np.array([0, 0, -2.0]), #
        'bottle': np.array([0, 0, -2.0]),
        'bowl': np.array([0, 0, -2.3]),
        'sofa': np.array([0, 0, -2.3]),
        'chair': np.array([0, 0, -2.3])
    }
    #
    mean_pose = mean_pose_dict[args.category] # mean pose
    std_pose = np.array([0.2, 0.2, 0.15])

    mp.set_start_method('spawn')
    num_per_ins = (len(path_list) - 1) // args.num_proc + 1
    processes = []
    for i in range(args.num_proc):
        st = num_per_ins * i
        ed = min(st + num_per_ins, len(path_list))
        p = mp.Process(target=proc_render, args=(i == 0,
                                                 path_list[st: ed], save_folder, args.rotate_num,
                                                 mean_pose, std_pose,
                                                 np.deg2rad(60), 640, 480, 0.01, 10,
                                                 args.upper_hemi))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
