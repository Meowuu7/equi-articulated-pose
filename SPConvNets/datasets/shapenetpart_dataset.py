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
# from model.utils import farthest_point_sampling


DATASET_ROOT_THU_LAB = ["/mnt/8T/xueyi/part-segmentation/data", "/mnt/sas-raid5-7.2T/xueyi/part-segmentation/data", "./data/part-segmentation/data", "/home/xueyi/inst-segmentation/data/part-segmentation/data"]


# 03001627 --- chair
class ShapeNetPartDataset(data.Dataset):
    def __init__(
            self, root="/data-input/motion_part_split_meta_info", npoints=512, split='train', nmask=10, shape_types=["03642806", "04379243"],  real_test=False,
            part_net_seg=False, partnet_split=False, args=None, split_types=False
    ):
        super(ShapeNetPartDataset, self).__init__()
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

        self.lm = dict()
        self.peridx = dict()

        print(f"ROOT_DIR = {ROOT_DIR}")

        # root_path = os.path.join(root, "shapenetcore_partanno_segmentation_benchmark_v0", "points")
        # root_path_labels = os.path.join(root, "shapenetcore_partanno_segmentation_benchmark_v0", "points_label")

        self.file_names = []
        self.point_labels_file_names = []

        for cur_cat_name in shape_types:
            root_path = os.path.join(root, "shapenetcore_partanno_segmentation_benchmark_v0", cur_cat_name, "points")
            # cur_folder_name = os.path.join(root_path, cur_cat_name)
            cur_folder_name = root_path
            cur_shape_names = os.listdir(cur_folder_name)

            cur_shape_file_names = [os.path.join(cur_folder_name, ssn) for ssn in cur_shape_names]
            self.file_names += cur_shape_file_names

        if self.split == 'train':
            train_nn = int(0.9 * len(self.file_names))
            self.file_names = self.file_names[:train_nn]
        else:
            train_nn = int(0.9 * len(self.file_names))
            self.file_names = self.file_names[train_nn:]

    def __getitem__(self, index):
        # cur_file_name = self.file_names[index % 10]
        cur_file_name = self.file_names[4]
        pts = []
        with open(cur_file_name, "r") as rf:
            for line in rf:
                pps = line.strip().split(" ")
                pps = [float(pp) for pp in pps]
                pts.append(pps)
            rf.close()
        pts = np.array(pts, dtype=np.float)

        permidx = np.random.permutation(pts.shape[0])[:self.npoints] # permutation index
        pts = pts[permidx] # points

        shape_seg = np.zeros((pts.shape[0], ), dtype=np.long) # fake shape_seg
        gt_pose = np.zeros((pts.shape[0], 4, 4), dtype=np.float) # fake gt_pose


        rt_dict = {
            'pc': torch.from_numpy(pts.astype(np.float32).T), # input points
            'label': torch.from_numpy(shape_seg).long(), # get label
            'pose': torch.from_numpy(gt_pose.astype(np.float32)) # get gt_pose information from np pose information
        }

        return rt_dict
        # return np.array([chosen_whether_mov_index], dtype=np.long), np.array([chosen_num_moving_parts], dtype=np.long), \
        #        pc1, pc2, flow12, shape_seg_masks, motion_seg_masks, pc1, pc1

    def __len__(self):
        return len(self.file_names)

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
