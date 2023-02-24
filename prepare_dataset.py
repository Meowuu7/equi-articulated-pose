import numpy as np
import open3d as o3d
from open3d import *


def centralize_np(pc, batch=False):
    axis = 2 if batch else 1
    return pc - pc.mean(axis=axis, keepdims=True)


def normalize_np(pc, batch=False):
    pc = centralize_np(pc, batch)
    axis = 1 if batch else 0
    var = np.sqrt((pc**2).sum(axis=axis, keepdims=True))
    return pc / var.max(axis=axis+1, keepdims=True)


def create_laptop_dataset(N=100000, totp=1024):
    # z dim; set rotation axis to x-axis

    z_len = np.random.uniform(50, 100, size=(1,)).item()
    z_len2 = np.random.uniform(30, 100, size=(1,)).item()

    x_len = np.random.uniform(40, 200, size=(1,)).item()
    y_len = np.random.uniform(3, 10, size=(1,)).item()
    y_len2 = np.random.uniform(5, 10, size=(1,)).item()

    # sample points
    up_p_ratio = np.random.uniform(0.3, 0.7, (1,)).item()
    up_p_n = int(totp * up_p_ratio)
    down_p_n = totp - up_p_n
    up_ps_x = np.random.uniform(-x_len / 2, x_len / 2, size=(up_p_n,))
    up_ps_y = np.random.uniform(-y_len / 2, y_len / 2, size=(up_p_n,))
    up_ps_z = np.random.uniform(0, z_len, size=(up_p_n,))

    down_ps_x = np.random.uniform(-x_len / 2, x_len / 2, size=(down_p_n,))
    down_ps_y = np.random.uniform(-y_len2 / 2, y_len2 / 2, size=(down_p_n,))
    down_ps_z = np.random.uniform(-z_len2, 0, size=(down_p_n,))

    up_ps = np.array([up_ps_x, up_ps_y, up_ps_z], dtype=np.float)
    if up_ps.shape[0] == 3:
        up_ps = np.transpose(up_ps, (1, 0))
    down_ps = np.array([down_ps_x, down_ps_y, down_ps_z], dtype=np.float)
    if down_ps.shape[0] == 3:
        down_ps = np.transpose(down_ps, (1, 0))

    pos = np.concatenate([up_ps, down_ps], axis=0)

    #### Normalize points ####
    pos = normalize_np(pos.T)
    #### Normalize points ####

    pos = pos.T
    up_ps, down_ps = pos[:up_p_n], pos[up_p_n:]

    angle = np.random.uniform(-0.5, 0.5, size=(1,)).item()
    cos_ = np.cos(angle * np.pi)
    sin_ = np.sin(angle * np.pi)
    rot_w = np.array(
        [[1, 0., 0.],
         [0., cos_, -sin_],
         [0., sin_, cos_]], dtype=np.float
    )

    trans = np.mean(down_ps, axis=0, keepdims=True)

    af_rotate_pos = np.transpose(np.matmul(rot_w, np.transpose(down_ps - trans, [1, 0])),
                                 [1, 0]) + trans

    pos = np.concatenate([up_ps, af_rotate_pos], axis=0)
    labels = np.zeros((pos.shape[0], ), dtype=np.long)
    labels[up_ps.shape[0]:] = 1
    rot_w_up = np.reshape(np.eye(3, dtype=np.float), (1, 3, 3))
    rot_ws = np.concatenate(
        [rot_w_up, np.reshape(rot_w, (1, 3, 3))], axis=0
    )
    # we can just generate them on the fly...

    cc1, cc2 = [1.0, 1.0, 0], [1.0, 0.0, 0]

    pcd_a = PointCloud()
    pcd_a.points = Vector3dVector(up_ps)
    pcd_a.paint_uniform_color(cc1)

    pcd_b = PointCloud()
    pcd_b.points = Vector3dVector(af_rotate_pos)
    pcd_b.paint_uniform_color(cc2)

    draw_geometries([pcd_a, pcd_b])

if __name__=='__main__':
    create_laptop_dataset()


