import numpy as np
import trimesh
import os
import glob
import scipy.io as sio
import torch
import torch.utils.data as data
import vgtk.pc as pctk
import vgtk.point3d as p3dtk
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np, label_relative_rotation_np
from scipy.spatial.transform import Rotation as sciR


class Dataloader_ToySeg(data.Dataset):
    def __init__(self, opt, len, mode=None):
        super(Dataloader_ToySeg, self).__init__()
        self.opt = opt

        # 'train' or 'eval'
        self.mode = opt.mode if mode is None else mode

        # attention method: 'attention | rotation'
        self.flag = opt.model.flag

        # get anchors
        self.anchors = L.get_anchors()

        # register parameters
        self.z_len_min = opt.z_len_min
        self.z_len_max = opt.z_len_max
        self.z_len2_min = opt.z_len2_min
        self.z_len2_max = opt.z_len2_max
        self.x_len_min = opt.x_len_min
        self.x_len_max = opt.x_len_max
        self.y_len_min = opt.y_len_min
        self.y_len_max = opt.y_len_max
        self.y_len2_min = opt.y_len2_min
        self.y_len2_max = opt.y_len2_max
        self.num_points = opt.num_points
        self.up_p_ratio_min = opt.up_p_ratio_min
        self.up_p_ratio_max = opt.up_p_ratio_max

        self.len = len

        # if self.flag == 'rotation':
        #     cats = ['airplane']
        #     print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")
        # else:
        #     cats = os.listdir(opt.dataset_path)

        # laod data
        # self.dataset_path = opt.dataset_path
        # self.all_data = []
        # for cat in cats:
        #     for fn in glob.glob(os.path.join(opt.dataset_path, cat, self.mode, "*.mat")):
        #         self.all_data.append(fn)
        #
        # print("[Dataloader] : Training dataset size:", len(self.all_data))

        if self.opt.no_augmentation:
            print("[Dataloader]: USING ALIGNED MODELNET LOADER!")
        else:
            print("[Dataloader]: USING ROTATED MODELNET LOADER!")

    def get_canonical_rotation(self):
        cos_ = np.cos(0.5 * np.pi)
        sin_ = np.sin(0.5 * np.pi)
        cana_rot_w = np.array(
            [[1, 0., 0.],
             [0., cos_, -sin_],
             [0., sin_, cos_]], dtype=np.float
        )
        return cana_rot_w

    def __len__(self):
        return self.len

    def __getitem_xx__(self, index):
        totp = self.num_points

        z_len = np.random.uniform(self.z_len_min, self.z_len_max, size=(1,)).item()
        z_len2 = np.random.uniform(self.z_len2_min, self.z_len2_max, size=(1,)).item()

        x_len = np.random.uniform(self.x_len_min, self.x_len_max, size=(1,)).item()
        y_len = np.random.uniform(self.y_len_min, self.y_len_max, size=(1,),).item()
        y_len2 = np.random.uniform(self.y_len2_min, self.y_len2_max, size=(1,)).item()

        # sample points
        up_p_ratio = np.random.uniform(self.up_p_ratio_min, self.up_p_ratio_max, (1,)).item()
        up_p_n = int(totp * up_p_ratio)
        down_p_n = totp - up_p_n
        up_ps_x = np.random.uniform(-x_len / 2, x_len / 2, size=(up_p_n,))
        up_ps_y = np.random.uniform(-y_len / 2, y_len / 2, size=(up_p_n,))
        up_ps_z = np.random.uniform(0.0, z_len, size=(up_p_n,))

        down_ps_x = np.random.uniform(-x_len / 2, x_len / 2, size=(down_p_n,))
        down_ps_y = np.random.uniform(-y_len2 / 2, y_len2 / 2, size=(down_p_n,))
        down_ps_z = np.random.uniform(-z_len2, 0.0, size=(down_p_n,))

        up_ps = np.array([up_ps_x, up_ps_y, up_ps_z], dtype=np.float)
        if up_ps.shape[0] == 3: # up-part points
            up_ps = np.transpose(up_ps, (1, 0))
        down_ps = np.array([down_ps_x, down_ps_y, down_ps_z], dtype=np.float)
        if down_ps.shape[0] == 3: # down-part points
            down_ps = np.transpose(down_ps, (1, 0))

        pos = np.concatenate([up_ps, down_ps], axis=0)

        #### Normalize points ####
        pos = p3dtk.normalize_np(pos.T)
        #### Normalize points ####

        pos = pos.T
        up_ps, down_ps = pos[:up_p_n], pos[up_p_n:]

        # todo: larger angle range?
        angle = np.random.uniform(-0.5, 0.5, size=(1,)).item()
        cos_ = np.cos(angle * np.pi)
        sin_ = np.sin(angle * np.pi)
        rot_w = np.array(
            [[1, 0., 0.],
             [0., cos_, -sin_],
             [0., sin_, cos_]], dtype=np.float
        )

        trans = np.mean(down_ps, axis=0, keepdims=True) # 1 x 3

        # rotate down-part along the x-axis
        af_rotate_pos = np.transpose(np.matmul(rot_w, np.transpose(down_ps - trans, [1, 0])),
                                     [1, 0]) + trans

        ''' Get rotation matrix and translation vector in canonical frame '''
        cana_rot_w = self.get_canonical_rotation()
        rel_rot_w = np.dot(rot_w, np.transpose(cana_rot_w, (1, 0)))
        rel_trans = np.reshape(trans, (3, 1))
        rel_trans = rel_trans - np.dot(rel_rot_w, rel_trans)
        ''' Get rotation matrix and translation vector in canonical frame '''

        pose_w_up = np.reshape(np.eye(4, dtype=np.float32), (1, 4, 4))
        pose_w_up[0, -1, -1] = 0.0
        pose_w_down = np.zeros(shape=(4, 4), dtype=np.float32)
        pose_w_down[:3, :3] = rel_rot_w[:, :]
        pose_w_down[:3, -1] = rel_trans[:, 0]

        # concatenate pos information
        pos = np.concatenate([up_ps, af_rotate_pos], axis=0)
        labels = np.zeros((pos.shape[0],), dtype=np.long)
        labels[up_ps.shape[0]:] = 1
        # rot_w_up = np.reshape(np.eye(3, dtype=np.float), (1, 3, 3))
        # rot_ws = np.concatenate(
        #     [rot_w_up for _ in range(up_p_n)] + [ np.reshape(rot_w, (1, 3, 3)) for _ in range(down_p_n)], axis=0
        # )
        pos = pos.T

        pose_ws = np.concatenate(
            [pose_w_up for _ in range(up_p_n)] + [np.reshape(pose_w_down, (1, 4, 4)) for _ in range(down_p_n)], axis=0
        )

        return {'pc': torch.from_numpy(pos.astype(np.float32)),
                'label': torch.from_numpy(labels).long(),
                # 'pose': torch.from_numpy(rot_ws.astype(np.float32))
                'pose': torch.from_numpy(pose_ws.astype(np.float32))
                }

    def __getitem__(self, index):
        totp = self.num_points

        z_len = np.random.uniform(self.z_len_min, self.z_len_max, size=(1,)).item()
        z_len2 = np.random.uniform(self.z_len2_min, self.z_len2_max, size=(1,)).item()

        x_len = np.random.uniform(self.x_len_min, self.x_len_max, size=(1,)).item()
        y_len = np.random.uniform(self.y_len_min, self.y_len_max, size=(1,), ).item()
        y_len2 = np.random.uniform(self.y_len2_min, self.y_len2_max, size=(1,)).item()



        # sample points
        up_p_ratio = np.random.uniform(self.up_p_ratio_min, self.up_p_ratio_max, (1,)).item()
        up_p_n = int(totp * up_p_ratio)
        down_p_n = totp - up_p_n
        up_ps_x = np.random.uniform(-x_len / 2, x_len / 2, size=(up_p_n,))
        up_ps_y = np.random.uniform(-y_len / 2, y_len / 2, size=(up_p_n,))
        up_ps_z = np.random.uniform(0.0, z_len, size=(up_p_n,))

        down_ps_x = np.random.uniform(-x_len / 2, x_len / 2, size=(down_p_n,))
        down_ps_y = np.random.uniform(-y_len2 / 2, y_len2 / 2, size=(down_p_n,))
        down_ps_z = np.random.uniform(-z_len2, 0.0, size=(down_p_n,))

        # 0.2
        up_p_n, down_p_n = self.num_points // 2, self.num_points // 2
        # we have 128 points
        # delta_x = 1.0 / 8; delta_z = delta_x
        delta_x = 1.0 / 16; delta_z = delta_x / 2.
        up_xs = [((delta_x * ii) - 0.5) for ii in range(16)] # up_xs
        xs = up_xs
        # up_xs = np.array(up_xs, dtype=np.float32) - 0.5
        # xs = up_xs
        up_zs = [(delta_z * ii) + (delta_z / 2) for ii in range(16)]
        # zs = up_zs
        # up_zs = np.array(up_zs, dtype=np.float32)
        # ys = np.array(, dtype=)
        ys = [-0.08, 0.08]
        ys = [0.0]

        up_ps = []
        for i_x, ux in enumerate(xs):
            for i_y, uy in enumerate(ys):
                for i_z, uz in enumerate(up_zs):
                    up_ps.append([ux, uy, uz])
        up_ps = np.array(up_ps, dtype=np.float32) # n_ps x 3

        down_ps = []
        down_zs = [-1 * zz for zz in up_zs]
        for i_x, dx in enumerate(xs):
            for i_y, dy in enumerate(ys):
                for i_z, dz in enumerate(down_zs):
                    down_ps.append([dx, dy, dz])
        down_ps = np.array(down_ps, dtype=np.float32)


        # up_ps = np.array([up_ps_x, up_ps_y, up_ps_z], dtype=np.float)
        # if up_ps.shape[0] == 3:  # up-part points
        #     up_ps = np.transpose(up_ps, (1, 0))
        # down_ps = np.array([down_ps_x, down_ps_y, down_ps_z], dtype=np.float)
        # if down_ps.shape[0] == 3:  # down-part points
        #     down_ps = np.transpose(down_ps, (1, 0))

        pos = np.concatenate([up_ps, down_ps], axis=0)

        #### Normalize points ####
        # pos = p3dtk.normalize_np(pos.T)
        #### Normalize points ####

        # pos = pos.T
        # up_ps, down_ps = pos[:up_p_n], pos[up_p_n:]

        # todo: larger angle range?
        angle = np.random.uniform(-0.5, 0.5, size=(1,)).item()
        # angle = 0.5
        # angle = 0.0
        cos_ = np.cos(angle * np.pi)
        sin_ = np.sin(angle * np.pi)
        rot_w = np.array(
            [[1, 0., 0.],
             [0., cos_, -sin_],
             [0., sin_, cos_]], dtype=np.float
        )

        # rot_w = np.array(
        #     [[1, 0., 0.],
        #      [0., 1., 0.],
        #      [0., 0., 1.]], dtype=np.float
        # )

        # trans = np.mean(down_ps, axis=0, keepdims=True)  # 1 x 3
        trans = np.zeros((1, 3), dtype=np.float32)

        # rotate down-part along the x-axis
        # af_rotate_pos = np.transpose(np.matmul(rot_w, np.transpose(down_ps - trans, [1, 0])),
        #                              [1, 0]) + trans
        af_rotate_pos = np.transpose(np.matmul(rot_w, np.transpose(down_ps, [1, 0])),
                                     [1, 0])

        ''' Get rotation matrix and translation vector in canonical frame '''
        cana_rot_w = self.get_canonical_rotation()
        rel_rot_w = np.dot(rot_w, np.transpose(cana_rot_w, (1, 0)))

        rel_trans = np.reshape(trans, (3, 1))
        rel_trans = rel_trans - np.dot(rel_rot_w, rel_trans)
        ''' Get rotation matrix and translation vector in canonical frame '''

        pose_w_up = np.reshape(np.eye(4, dtype=np.float32), (1, 4, 4))
        pose_w_up[0, -1, -1] = 0.0
        pose_w_down = np.zeros(shape=(4, 4), dtype=np.float32)
        pose_w_down[:3, :3] = rel_rot_w[:, :]
        pose_w_down[:3, -1] = rel_trans[:, 0]

        # concatenate pos information
        #
        pos = np.concatenate([up_ps, af_rotate_pos], axis=0)
        labels = np.zeros((pos.shape[0],), dtype=np.long)
        labels[up_ps.shape[0]:] = 1
        # rot_w_up = np.reshape(np.eye(3, dtype=np.float), (1, 3, 3))
        # rot_ws = np.concatenate(
        #     [rot_w_up for _ in range(up_p_n)] + [ np.reshape(rot_w, (1, 3, 3)) for _ in range(down_p_n)], axis=0
        # )
        pos = pos.T

        pose_ws = np.concatenate(
            [pose_w_up for _ in range(up_p_n)] + [np.reshape(pose_w_down, (1, 4, 4)) for _ in range(down_p_n)], axis=0
        )

        return {'pc': torch.from_numpy(pos.astype(np.float32)),
                'label': torch.from_numpy(labels).long(),
                # 'pose': torch.from_numpy(rot_ws.astype(np.float32))
                'pose': torch.from_numpy(pose_ws.astype(np.float32))
                }
