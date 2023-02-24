from importlib import import_module
# from SPConvNets import Dataloader_ModelNet40
# from SPConvNets.datasets.ToySegDataset import Dataloader_ToySeg
# from SPConvNets.datasets.MotionSegDataset import PartSegmentationMetaInfoDataset
# from SPConvNets.datasets.shapenetpart_dataset import ShapeNetPartDataset
from SPConvNets.datasets.MotionDataset import MotionDataset
from SPConvNets.datasets.MotionDataset2 import MotionDataset as MotionDataset2
from SPConvNets.datasets.MotionSAPIENDataset import MotionDataset as MotionSAPIENDataset
# from SPConvNets.datasets.MotionHOIDatasetNPCS import MotionDataset as MotionHOIDataset
from SPConvNets.datasets.MotionHOIDataset import MotionDataset as MotionHOIDataset
try:
    from SPConvNets.datasets.MotionDatasetPartial import MotionDataset as MotionDatasetPartial
except:
    pass
# from SPConvNets.datasets.MotionDatasetPartial import MotionDataset as MotionDatasetPartial
try:
    from SPConvNets.datasets.MotionHOIDatasetPartial import MotionDataset as MotionHOIDatasetPartial
except:
    pass

try:
    from SPConvNets.datasets.MotionSAPIENDatasetPartial import MotionDataset as MotionSAPIENDatasetPartial
except:
    pass



from tqdm import tqdm
import time
import torch.optim as optim

import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import torch.nn.functional as F
from sklearn.neighbors import KDTree
# from .utils.loss_util import iou, calculate_res_relative_Rs, batched_index_select
from .utils.loss_util import *
import torch.nn as nn
# from datas
import math

import torch.backends.cudnn as cudnn
import torch.distributed as dist

from SPConvNets.pose_utils import *

from SPConvNets.models.common_utils import *
from SPConvNets.ransac import *


class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        ''' dummy '''
        ''' Set device '''
        # if torch.cuda.device_count() > 1:
        #     torch.distributed.init_process_group(backend='nccl')
        #     tmpp_local_rnk = int(os.environ['LOCAL_RANK'])
        #     print("os_environed:", tmpp_local_rnk)
        #     self.local_rank = tmpp_local_rnk
        #     torch.cuda.set_device(self.local_rank)
        # else:
        #     self.local_rank = 0

        torch.distributed.init_process_group(backend='nccl')
        tmpp_local_rnk = int(os.environ['LOCAL_RANK'])
        print("os_environed:", tmpp_local_rnk)
        self.local_rank = tmpp_local_rnk
        torch.cuda.set_device(self.local_rank)

        opt.device = self.local_rank

        self.n_step = 0
        self.n_dec_steps = opt.equi_settings.n_dec_steps
        self.lr_adjust = opt.equi_settings.lr_adjust
        self.lr_decay_factor = opt.equi_settings.lr_decay_factor
        self.pre_compute_delta = opt.equi_settings.pre_compute_delta
        self.num_slots = opt.nmasks
        self.use_equi = opt.equi_settings.use_equi
        # torch.cuda.set_device(opt.parallel.local_rank)
        # self.local_rank = opt.parallel.local_rank
        self.gt_oracle_seg = opt.equi_settings.gt_oracle_seg
        self.slot_recon_factor = opt.equi_settings.slot_recon_factor
        self.est_normals = opt.equi_settings.est_normals
        self.glb_resume_path = opt.resume_path_glb
        self.resume_path = opt.resume_path
        self.global_rot = opt.equi_settings.global_rot

        self.run_mode = opt.run_mode

        # self.stage = 0
        self.stage = opt.equi_settings.cur_stage

        print(f"local_rank: {self.local_rank}")
        ''' Set shape type '''
        self.opt_shape_type = opt.equi_settings.shape_type

        ''' Set number of procs '''
        opt.nprocs = torch.cuda.device_count()
        print("device count:", opt.nprocs)
        nprocs = opt.nprocs
        self.nprocs = nprocs
        #
        # if opt.nprocs > 1:
        #     self._use_multi_gpu = True
        # else:
        #     self._use_multi_gpu = False
        self._use_multi_gpu = True

        cudnn.benchmark = True

        self.dataset_type = opt.equi_settings.dataset_type
        print(f"current dataset_type: {self.dataset_type}")
        self.attention_model = opt.model.flag.startswith('attention') and opt.debug_mode != 'knownatt'

        super(Trainer, self).__init__(opt)

        # if self.attention_model:
        #     self.summary.register(['Loss', 'Acc', 'R_Loss', 'R_Acc'])
        # else:
        self.n_iters = opt.equi_settings.num_iters
        reg_strs = []
        reg_strs.append('Loss')
        reg_strs.append('dot_axis_pred')
        for i_iter in range(self.n_iters):
            reg_strs.append(f'Acc_{i_iter}')
            reg_strs.append(f'Acc_2_{i_iter}')
        reg_strs.append('Avg_R_dist')
        # self.summary.register(['Loss', 'Acc'])
        self.summary.register(reg_strs)
        self.epoch_counter = 0 # epoch counter
        self.iter_counter = 0 # inter counter
        self.test_accs = [] # test metrics
        # self.
        self.best_acc_ever_reached = 0.0
        self.best_loss = 9999.0
        self.not_increased_epoch = 0
        self.last_loss = 9999.0

        model_id = f'model_{time.strftime("%Y%m%d_%H:%M:%S")}'
        self.loss_log_saved_path = "loss_" + model_id + ".txt"

        ''' Setup predicted features save folder '''
        predicted_info_folder = "predicted_info"
        if not os.path.exists(predicted_info_folder):
            # os.mkdir(predicted_info_folder)
            os.makedirs(predicted_info_folder, exist_ok=True)
        cur_shape_info_folder = os.path.join(predicted_info_folder, self.shape_type)
        if not os.path.exists(cur_shape_info_folder):
            # os.mkdir(cur_shape_info_folder, exists)
            os.makedirs(cur_shape_info_folder, exist_ok=True)
        # cur_setting_info_folder =  self.model.module.log_fn
        # cur_setting_info_folder = os.path.join(cur_shape_info_folder, cur_setting_info_folder)
        # if not os.path.exists(cur_setting_info_folder):
        #     os.mkdir(cur_setting_info_folder)
        # self.predicted_info_save_folder = cur_setting_info_folder
        self.opt = opt
        if self.local_rank == 0:
            if not os.path.exists(self.model.module.log_fn):
                # os.mkdir(self.model.module.log_fn)
                os.makedirs(self.model.module.log_fn, exist_ok=True)

    def _setup_optim(self):
        self.logger.log('Setup', 'Setup optimizer!')
        # torch.autograd.set_detect_anomaly(True)
        # print(type(self.model.parameters()))
        # print(self.model.parameters())

        nm_params = []
        for param_nm, param in self.model.named_parameters():
            if "glb_backbone" not in param_nm:
                nm_params.append(param)

        #### set up optimizer ####
        # self.optimizer = optim.Adam(nm_params,
        #                             lr=self.opt.train_lr.init_lr)
        #### set up optimizer ####

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.opt.train_lr.init_lr)
        self.lr_schedule = vgtk.LearningRateScheduler(self.optimizer,
                                                      **vars(self.opt.train_lr))
        self.logger.log('Setup', 'Optimizer all-set!')


    def reduce_mean(self, tensor, nprocs):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt = rt / nprocs
        return rt

    ''' Setup datsets '''
    def _setup_datasets(self):
        # if self.opt.mode == 'train':
        #     dataset = Dataloader_ToySeg(self.opt, len=self.opt.train_len)
        #     self.dataset = torch.utils.data.DataLoader(dataset,
        #                                                 batch_size=self.opt.batch_size,
        #                                                 shuffle=True,
        #                                                 num_workers=self.opt.num_thread)
        #     self.dataset_iter = iter(self.dataset)
        #
        # #
        # dataset_test = Dataloader_ToySeg(self.opt, len=self.opt.test_len, mode='testR')
        # self.dataset_test = torch.utils.data.DataLoader(dataset_test,
        #                                                 batch_size=self.opt.batch_size,
        #                                                 shuffle=False,
        #                                                 num_workers=self.opt.num_thread)

        npoints = 512
        npoints = self.opt.model.input_num
        global_rot = self.opt.equi_settings.global_rot
        # npoints = 256
        # npoints = 1024
        # if self.opt.mode == 'train':
        #
        #     dataset = PartSegmentationMetaInfoDataset(
        #         root="/home/xueyi/inst-segmentation/data/part-segmentation/data/motion_part_split_meta_info",
        #         npoints=npoints, split='train', nmask=10,
        #         shape_types=["03642806", "03636649", "02691156", "03001627", "02773838", "02954340", "03467517",
        #                      "03790512",
        #                      "04099429", "04225987", "03624134", "02958343", "03797390", "03948459", "03261776",
        #                      "04379243"],
        #         real_test=False, part_net_seg=False, partnet_split=False, args=self.opt)
        #     self.dataset = torch.utils.data.DataLoader(dataset,
        #                                                 batch_size=self.opt.batch_size,
        #                                                 shuffle=True,
        #                                                 num_workers=self.opt.num_thread)
        #     self.dataset_iter = iter(self.dataset)
        #
        # dataset_test = PartSegmentationMetaInfoDataset(
        #         root="/home/xueyi/inst-segmentation/data/part-segmentation/data/motion_part_split_meta_info",
        #         npoints=npoints, split='val', nmask=10,
        #         shape_types=["03642806", "03636649", "02691156", "03001627", "02773838", "02954340", "03467517",
        #                      "03790512",
        #                      "04099429", "04225987", "03624134", "02958343", "03797390", "03948459", "03261776",
        #                      "04379243"],
        #         real_test=False, part_net_seg=False, partnet_split=False, args=self.opt)
        # self.dataset_test = torch.utils.data.DataLoader(dataset_test,
        #                                                 batch_size=self.opt.batch_size,
        #                                                 shuffle=False,
        #                                                 num_workers=self.opt.num_thread)

        if self.dataset_type == DATASET_PARTNET:
            ## If using partnet dataset ##

            ''' Shapes from motion segmentation dataset '''
            if self.opt.mode == 'train':

                dataset = PartSegmentationMetaInfoDataset(
                    root="/home/xueyi/inst-segmentation/data/part-segmentation/data/motion_part_split_meta_info",
                    npoints=npoints, split='train', nmask=10,
                    # shape_types=["Sitting Furniture"],
                    shape_types=["Laptop"],
                    real_test=False, part_net_seg=True, partnet_split=False, args=self.opt)
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
                self.dataset = torch.utils.data.DataLoader(dataset,
                                                            batch_size=self.opt.batch_size,
                                                            sampler=train_sampler,
                                                            num_workers=self.opt.num_thread)
                self.dataset_iter = iter(self.dataset)

            dataset_test = PartSegmentationMetaInfoDataset(
                    root="/home/xueyi/inst-segmentation/data/part-segmentation/data/motion_part_split_meta_info",
                    npoints=npoints, split='val', nmask=10,
                    # shape_types=["Sitting Furniture"],
                    shape_types=["Laptop"],
                    real_test=False, part_net_seg=True, partnet_split=False, args=self.opt)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)
            # Test dataloader
            self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                            batch_size=self.opt.batch_size,
                                                            sampler=test_sampler,
                                                            num_workers=self.opt.num_thread)

        elif self.dataset_type == DATASET_MOTION or self.dataset_type == DATASET_MOTION_PARTIAL:
            ## If using motion dataset ##
            shape_type = 'laptop'
            shape_type = 'eyeglasses'
            shape_type = 'oven'
            # shape_type = 'bucket'
            # shape_type = 'washing_machine'
            shape_type = self.opt_shape_type
            self.shape_type = shape_type

            self.shape_type = self.opt_shape_type

            global_rot = self.opt.equi_settings.global_rot # whether using global rotation

            if self.dataset_type == DATASET_MOTION:
                cur_dataset = MotionDataset if self.shape_type != DATASET_DRAWER else MotionSAPIENDataset
            else:
                cur_dataset = MotionDatasetPartial if self.shape_type != DATASET_DRAWER else MotionSAPIENDatasetPartial
            val_str = "val" if self.shape_type != DATASET_DRAWER else "test"

            ''' Shapes from motion segmentation dataset '''
            if self.opt.mode == 'train' or self.opt.mode == "eval":
                dataset = cur_dataset(
                    root="./data/MDV02", #
                    npoints=npoints, split='train', nmask=10,
                    shape_type=shape_type, args=self.opt, global_rot=global_rot)
                if self._use_multi_gpu:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset)
                    self.dataset = torch.utils.data.DataLoader(dataset,
                                                               batch_size=self.opt.batch_size,
                                                               sampler=train_sampler,
                                                               num_workers=self.opt.num_thread)
                else:
                    self.dataset = torch.utils.data.DataLoader(dataset,
                                                               batch_size=self.opt.batch_size,
                                                               shuffle=True,
                                                               num_workers=self.opt.num_thread)
                self.dataset_iter = iter(self.dataset)

            dataset_test = cur_dataset(
                root="./data/MDV02",
                npoints=npoints, split=val_str, nmask=10,
                shape_type=shape_type, args=self.opt, global_rot=global_rot)
            if self._use_multi_gpu:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset_test)
                self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=self.opt.batch_size,
                                                                sampler=test_sampler,
                                                                num_workers=self.opt.num_thread)
            else:
                self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=self.opt.batch_size,
                                                                shuffle=False,
                                                                num_workers=self.opt.num_thread)
        elif self.dataset_type == DATASET_MOTION2:
            ## If using motion dataset ##
            shape_type = 'laptop'
            shape_type = 'eyeglasses'
            shape_type = 'oven'
            # shape_type = 'bucket'
            # shape_type = 'washing_machine'
            shape_type = self.opt_shape_type
            self.shape_type = shape_type

            self.shape_type = self.opt_shape_type

            global_rot = self.opt.equi_settings.global_rot  # whether using global rotation
            ''' Shapes from motion segmentation dataset '''
            if self.opt.mode == 'train' or self.opt.mode == "eval":
                dataset = MotionDataset2(
                    root="./data/MDV02",  #
                    npoints=npoints, split='train', nmask=10,
                    shape_type=shape_type, args=self.opt, global_rot=global_rot)
                if self._use_multi_gpu:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset, shuffle=True)
                    self.dataset = torch.utils.data.DataLoader(dataset,
                                                               batch_size=self.opt.batch_size,
                                                               sampler=train_sampler,
                                                               num_workers=self.opt.num_thread)
                else:
                    self.dataset = torch.utils.data.DataLoader(dataset,
                                                               batch_size=self.opt.batch_size,
                                                               shuffle=True,
                                                               num_workers=self.opt.num_thread)
                self.dataset_iter = iter(self.dataset)

            dataset_test = MotionDataset2(
                root="./data/MDV02",
                npoints=npoints, split='test', nmask=10,
                shape_type=shape_type, args=self.opt, global_rot=global_rot)
            if self._use_multi_gpu:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset_test)
                self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=self.opt.batch_size,
                                                                sampler=test_sampler,
                                                                num_workers=self.opt.num_thread)
            else:
                self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=self.opt.batch_size,
                                                                shuffle=True,
                                                                num_workers=self.opt.num_thread)
        elif self.dataset_type == DATASET_HOI4D or self.dataset_type == DATASET_HOI4D_PARTIAL:
            ## If using motion dataset ##
            # shape_type = 'laptop'
            # shape_type = 'eyeglasses'
            # shape_type = 'oven'
            # shape_type = 'bucket'
            # shape_type = 'washing_machine'
            shape_type = self.opt_shape_type
            self.shape_type = shape_type

            self.shape_type = self.opt_shape_type

            global_rot = self.opt.equi_settings.global_rot  # whether using global rotation

            # Get current dataset
            cur_dataset = MotionHOIDataset if self.dataset_type == DATASET_HOI4D else MotionHOIDatasetPartial

            val_str = "val" if self.shape_type != DATASET_DRAWER else "test"

            ''' Shapes from motion segmentation dataset ''' # motion-seg...
            if self.opt.mode == 'train' or self.opt.mode == "eval":
                dataset = cur_dataset(
                    root="./data/HOI4D",  #
                    npoints=npoints, split='train', nmask=10,
                    shape_type=shape_type, args=self.opt, global_rot=global_rot)
                if self._use_multi_gpu:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset)
                    self.dataset = torch.utils.data.DataLoader(dataset,
                                                               batch_size=self.opt.batch_size,
                                                               sampler=train_sampler,
                                                               num_workers=self.opt.num_thread)
                else:
                    self.dataset = torch.utils.data.DataLoader(dataset,
                                                               batch_size=self.opt.batch_size,
                                                               shuffle=True,
                                                               num_workers=self.opt.num_thread)
                self.dataset_iter = iter(self.dataset)

            dataset_test = cur_dataset(
                root="./data/HOI4D",
                npoints=npoints, split=val_str, nmask=10,
                shape_type=shape_type, args=self.opt, global_rot=global_rot)
            if self._use_multi_gpu:
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset_test)
                self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=self.opt.batch_size,
                                                                sampler=test_sampler,
                                                                num_workers=self.opt.num_thread)
            else:
                self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=self.opt.batch_size,
                                                                shuffle=False,
                                                                num_workers=self.opt.num_thread)
        else:
            raise ValueError(f"Unrecognized dataset type: {self.dataset_type}.")

        ''' Shapes from ShapeNetPart dataset '''
        # npoints = 1024
        # # npoints = 2048
        # # npoints = 1500
        # if self.opt.mode == 'train':
        #     dataset = ShapeNetPartDataset(
        #         root="/home/xueyi/EPN_PointCloud/data",
        #         npoints=npoints, split='train', nmask=10,
        #         shape_types=["03001627"],
        #         real_test=False, part_net_seg=True, partnet_split=False, args=self.opt)
        #     self.dataset = torch.utils.data.DataLoader(dataset,
        #                                                 batch_size=self.opt.batch_size,
        #                                                 shuffle=True,
        #                                                 num_workers=self.opt.num_thread)
        #     self.dataset_iter = iter(self.dataset)
        #
        # dataset_test = ShapeNetPartDataset(
        #         root="/home/xueyi/EPN_PointCloud/data",
        #         npoints=npoints, split='val', nmask=10,
        #         shape_types=["03001627"],
        #         real_test=False, part_net_seg=True, partnet_split=False, args=self.opt)
        # self.dataset_test = torch.utils.data.DataLoader(dataset_test,
        #                                                 batch_size=self.opt.batch_size,
        #                                                 shuffle=False,
        #                                                 num_workers=self.opt.num_thread)

    ''' Setup model '''
    def _setup_model(self):
        if self.opt.mode == 'train': #
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

        if self.stage == 1:
            glb_stage_module = import_module('SPConvNets.models')
            # ori_kpconv_kanchor =
            # self.opt.equi_settings.kpconv_kanchor = 1
            self.glb_stage_model = getattr(glb_stage_module, self.opt.model.model).build_model_from(self.opt, param_outfile)
            self.glb_stage_model.stage = 0 # set stage to 0
            self.opt.equi_settings.kpconv_kanchor = self.model.kpconv_kanchor

        # if self.dataset_type == DATASET_MOTION and self.shape_type != DATASET_DRAWER:
        #     self.ref_shape_slot = self.dataset.dataset.get_shape_by_idx(0)
        #     print(f"Got reference shape: {self.ref_shape_slot.size()}")
        #     self.model.ref_shape_slot = self.ref_shape_slot.cuda()
        #
        #     self.model.ref_whole_shape = self.dataset.dataset.get_whole_shape_by_idx(0).cuda()
        # self.

    def train_iter(self):
        for i in range(self.opt.num_iterations):
            self.timer.set_point('train_iter')
            # self.lr_schedule.step()
            self.step()
            # print({'Time': self.timer.reset_point('train_iter')})
            self.summary.update({'Time': self.timer.reset_point('train_iter')})

            if i % self.opt.log_freq == 0:
                if hasattr(self, 'epoch_counter'):
                    step = f'Epoch {self.epoch_counter}, Iter {i}'
                else:
                    step = f'Iter {i}'
                self._print_running_stats(step)

            if i > 0 and i < 10000 and i % self.opt.save_freq == 0:
                self._save_network(f'Iter{i}')
                self.test()

    def safe_load_ckpt(self, model, state_dicts):
        ori_dict = state_dicts
        part_dict = dict()
        model_dict = model.state_dict()
        tot_params_n = 0
        for k in ori_dict:
            if k in model_dict and 'glb' in k: #### have the value
                v = ori_dict[k]
                part_dict[k] = v
                tot_params_n += 1
            # if k in model_dict:
            #     v = ori_dict[k]
            #     part_dict[k] = v
            #     tot_params_n += 1
        model_dict.update(part_dict)
        model.load_state_dict(model_dict)
        self.logger.log('Setup', f"Resume glb-backbone finished!! Total number of parameters: {tot_params_n}.")
        #

    def safe_load_ckpt_common(self, model, state_dicts):
        ori_dict = state_dicts
        part_dict = dict()
        model_dict = model.state_dict()
        tot_params_n = 0
        for k in ori_dict: #
            if k in model_dict: #### have the value
                v = ori_dict[k]
                part_dict[k] = v
                tot_params_n += 1
        model_dict.update(part_dict)
        model.load_state_dict(model_dict)
        self.logger.log('Setup', f"Resume glb-backbone finished!! Total number of parameters: {tot_params_n}.")
        #

    ''' Load checkpoint '''
    def _resume_from_ckpt(self, resume_path):
        if resume_path is None:
            self.logger.log('Setup', f'Seems like we train from scratch!')
            return
        self.logger.log('Setup', f'Resume from checkpoint: {resume_path}')

        state_dicts = torch.load(resume_path, map_location="cpu")
        if self.stage == 1  and self.run_mode == 'train': #
            # self.glb_stage_model.load_state_dict(state_dicts)
            self.safe_load_ckpt(self.glb_stage_model, state_dicts)
        else:
            # must still be at stage = 1
            # self.model.load_state_dict(state_dicts)
            # load parameter weights for the model
            try:
                self.safe_load_ckpt_common(self.model, state_dicts) # st
            except:
                pass
            if self.glb_resume_path is not None:
                # load parameter weigths for the glb model
                glb_state_dicts = torch.load(self.glb_resume_path, map_location="cpu")
                self.safe_load_ckpt(self.glb_stage_model, glb_state_dicts)
            # glb_resume_path = s

        # # if self.stage == 1
        # if self.stage == 1 and self.run_mode == 'train':
        #     # self.model.glb_backbone.load_state_dict(state_dicts['glb_backbone'])
        #
        #     ori_dict = state_dicts
        #     part_dict = dict()
        #     model_dict = self.model.state_dict()
        #     tot_params_n = 0
        #     for k in ori_dict:
        #         if "glb_backbone" in k:
        #             v = ori_dict[k]
        #             part_dict[k] = v
        #             tot_params_n += 1
        #     model_dict.update(part_dict)
        #     self.model.load_state_dict(model_dict)
        #     self.logger.log('Setup', f"Resume glb-backbone finished!! Total number of parameters: {tot_params_n}.")
        #
        # else:
        #     self.model.load_state_dict(state_dicts)

        self.logger.log('Setup', f'Resume finished! Great!')

    ''' Setup model with multi-gpu '''
    def _setup_model_multi_gpu(self):
        # if torch.cuda.device_count() > 1:
        if self._use_multi_gpu:
            self.logger.log('Setup', 'Using Multi-gpu and DataParallel!')
            self._use_multi_gpu = True
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = self.model.cuda(self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   find_unused_parameters=False if self.use_equi in [23] else True)

            if self.stage == 1:
                self.glb_stage_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.glb_stage_model)
                self.glb_stage_model = self.glb_stage_model.cuda(self.local_rank)
                self.glb_stage_model = torch.nn.parallel.DistributedDataParallel(self.glb_stage_model, device_ids=[self.local_rank],
                                                                       find_unused_parameters=False if self.use_equi in [
                                                                           23] else True)
        else:
            self.logger.log('Setup', 'Using Single-gpu!')
            self._use_multi_gpu = False
            self.model = self.model.cuda()
            self.glb_stage_model = self.glb_stage_model.cuda()

    ''' Setup metric '''
    def _setup_metric(self):
        # attention model?
        # using attention mechanism
        if self.attention_model:
            self.metric = vgtk.AttentionCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin)
            # self.r_metric = AnchorMatchingLoss()
        else:
            self.metric = vgtk.CrossEntropyLossPerP()

    def _save_network(self, step, label=None, path=None):
        ##### Save network for further use #####
        label = self.opt.experiment_id if label is None else label
        if path is None:
            save_filename = '%s_net_%s.pth' % (label, step)
            save_path = os.path.join(self.root_dir, 'ckpt', save_filename)
        else:
            save_path = f'{path}.pth'

        if self._use_multi_gpu:
            # params = self.model.module.cpu().state_dict()
            params = self.model.module.state_dict()
        else:
            # params = self.model.cpu().state_dict()
            params = self.model.state_dict()
        torch.save(params, save_path)

        self.logger.log('Training', f'Checkpoint saved to: {save_path}!')

    # def save_predicted_by_step(self, out_feats):
    #     save_root_path = os.path.join(self.root_dir, "out_step")
    #     if not os.path.exists(save_root_path):
    #         os.mkdir(save_root_path)
    #     save_path = os.path.join(save_root_path, f"out_feats_Iter_{self.n_step}.npy")
    #     np.save(save_path, out_feats)
    #     print(f"Out features in step {self.n_step} saved to path {save_path}!")

    # For epoch-based training
    def epoch_step(self):
        for it, data in tqdm(enumerate(self.dataset)):
            self._optimize(data)

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
            if data['label'].shape[0] < self.opt.batch_size:
                raise StopIteration
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            # reset dataset iterator
            self.dataset_iter = iter(self.dataset)
            # get data for the next iteration
            data = next(self.dataset_iter)

        if self.stage == 0:
            self._optimize_stage_zero(data)
        else:
            self._optimize(data)
        self.iter_counter += 1

    def save_predicted_infos(self, idxes_list, out_feats, test=False):
        idxes_list_str = [str(ii) for ii in idxes_list]
        idxes_save_str = ",".join(idxes_list_str)
        # if test == True:
        #     save_pth = os.path.join(self.predicted_info_save_folder, idxes_save_str + ".npy")
        # np.save(os.path.join(self.predicted_info_save_folder, idxes_save_str + ".npy"), out_feats)

    def save_predicted_infos_all_iters(self, idxes_list, out_feats):
        idxes_list_str = [str(ii) for ii in idxes_list]
        idxes_save_str = ",".join(idxes_list_str)
        # np.save(os.path.join(self.predicted_info_save_folder, idxes_save_str + "_all_iters" + ".npy"), out_feats)

    ''' Train '''
    def _optimize(self, data):
        # set to train mode
        # optimize model
        if self._use_multi_gpu:
            self.model.module.train()
            # self.model.module.glb_backbone.eval()
            # self.glb_stage_model.module.train()
            self.glb_stage_model.module.eval()
        else:
            self.model.train()
            # self.model.glb_backbone.eval()
            # self.glb_stage_model.train()
            self.glb_stage_model.eval()
        self.metric.train()

        if self.n_step < 1000:
            self.model.module.annealing_k = 12
        elif self.n_step < 2000:
            self.model.module.annealing_k = 6
        else:
            self.model.module.annealing_k = 1

        # if self.n_step > 0:
        #     if os.path.exists("axis_prior_0.npy"):
        #         axis_prior_slot_pairs = torch.from_numpy(np.load("axis_prior_0.npy", allow_pickle=True)).cuda()
        #         self.model.module.axis_prior_slot_pairs.data = axis_prior_slot_pairs
        #     if os.path.exists("slot_pair_mult_R_queue_0.npy"):
        #         slot_pair_mult_R_queue = torch.from_numpy(np.load("slot_pair_mult_R_queue_0.npy", allow_pickle=True)).cuda()
        #         self.model.module.slot_pair_mult_R_queue.data = slot_pair_mult_R_queue

        # weighted_step = math.exp(-1. * (self.n_step) / self.n_dec_steps)
        # weighted_step = math.exp(-1. * (self.n_step) / 200)
        # # weighted_step = math.exp(-1. * (self.n_step) / 400) # if using larger n_dec_step
        # # weighted_step = math.exp(-1. * (self.n_step) / 800) # if using larger n_dec_step
        # # weighted_step = math.exp(-1. * (self.n_step) / 100) # if using a smaller n_dec_step
        # # weighted_step = math.exp(-1. * (self.n_step) / 50) # if using a smaller n_dec_step
        # # weighted_step = math.exp((self.n_step) / self.n_dec_steps)
        #
        # weighted_step = 1. - weighted_step
        # # weighted_step = 1.
        # # weighted_step = self.n_step / 800
        # # weighted_step = weighted_step - 1.
        # weighted_step = max(0.0, min(1.0, weighted_step))
        #
        # weighted_step = 1.0 if (self.use_equi == 23 and self.gt_oracle_seg == 1) else weighted_step
        weighted_step = 1.0 # if (self.use_equi == 23 and self.gt_oracle_seg == 1) else weighted_step

        # if local rank is zero, print weighted_step #### if local zeros
        if self.local_rank == 0:
            print(f"weighted_step current epoch: {weighted_step}")
        self.model.module.slot_recon_factor = self.slot_recon_factor * weighted_step

        if self.lr_adjust == 1:
            self.adjust_lr_by_loss()
        elif self.lr_adjust == 2:
            self.adjust_lr_by_step()
        # input tensors
        in_tensors = data['pc'].cuda(non_blocking=True)
        data_idxes = data['idx'].detach().cpu().numpy().tolist()
        data_idxes = [ str(ii) for ii in data_idxes]
        # in_tensors = torch.

        bdim = in_tensors.shape[0]
        in_label = data['label'].cuda(non_blocking=True) # .reshape(-1)
        in_pose = data['pose'].cuda(non_blocking=True) #  if self.opt.debug_mode == 'knownatt' else None
        in_pose_segs = data['pose_segs'].cuda(non_blocking=True)
        ori_pc = data['ori_pc'].cuda(non_blocking=True)
        canon_pc = data['canon_pc'].cuda(non_blocking=True)
        oorr_pc = data['oorr_pc'] # .cuda(non_blocking=True)
        oorr_label = data['oorr_label'] # .cuda(non_blocking=True)
        part_axis = data['part_axis'] # ground-trut axis...
        part_axis = part_axis.cuda(non_blocking=True)

        if self.shape_type != 'drawer':
            part_pv_offset = data['part_pv_offset'].cuda(non_blocking=True)
            # part_pv_point = data['cur_part_pv_point']
        else:
            part_pv_offset = torch.zeros((part_axis.size(0), part_axis.size(1)), dtype=torch.float32).cuda(non_blocking=True)
        if 'part_pv_point' in data:
            part_pv_point = data['part_pv_point'].cuda(non_blocking=True)
            part_angles = data['part_angles'].cuda()
        else:
            part_pv_point = torch.zeros((part_axis.size(0), in_pose_segs.size(1), 3), dtype=torch.float32).cuda(
                non_blocking=True)
            part_angles = torch.zeros((in_pose_segs.size(1)), dtype=torch.float32).cuda(
                non_blocking=True)
        if self.est_normals == 1:
            cur_normals = data['cur_normals'].cuda(non_blocking=True)
            cur_canon_normals = data['cur_canon_normals'].cuda(non_blocking=True)
        else:
            cur_normals = None
            cur_canon_normals = None
        # be af dists
        be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
        minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)

        # import ipdb; ipdb.set_trace()
        # print("input shapes = ", in_tensors.size(), in_label.size(), in_pose.size())

        # bz, N = in_tensors.size(0), in_tensors.size(2)

        ###################### ----------- debug only ---------------------
        # in_tensorsR = data['pcR'].to(self.opt.device)
        # import ipdb; ipdb.set_trace()
        ##################### --------------------------------------------

        # feed into the model: in_tensors, in_pos, and no rotation value
        # pred, feat =
        # loss, pred = self.model(in_tensors, in_pose, None)

        sv_dict = {'part_pv_points': part_pv_point.detach().cpu().numpy(),
                   'part_axis': part_axis.detach().cpu().numpy(),
                   'part_angles': part_angles.detach().cpu().numpy(), # part angles...,
                   'in_tensors': in_tensors.detach().cpu().numpy()
                   }

        #### Ground-truth labels ####
        label = torch.eye(self.model.module.num_slots)[in_label].cuda()

        if self.global_rot == 1 and self.resume_path is not None: # resume path is not None...
            with torch.no_grad():
                glb_recon_loss = self.glb_stage_model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc, normals=cur_normals, canon_normals=cur_canon_normals)
            in_tensors = self.glb_stage_model.module.inv_trans_ori_pts if self._use_multi_gpu else self.glb_stage_model.inv_trans_ori_pts
            glb_R = self.glb_stage_model.module.glb_R if self._use_multi_gpu else self.glb_stage_model.glb_R
            glb_T = self.glb_stage_model.module.glb_T if self._use_multi_gpu else self.glb_stage_model.glb_T

            sv_dict['glb_R'] = glb_R.detach().cpu().numpy()
            sv_dict['glb_T'] = glb_T.detach().cpu().numpy()

            sv_dict['in_tensors_af_glb'] = in_tensors.detach().cpu().numpy()

            # glb_R: bz x 3 x 3
            # glb_T: bz x 3
            # in_tensors = torch.matmul(safe_transpose(glb_R, -1, -2), in_tensors - glb_T.unsqueeze(-1))
            # in_pose: bz x N x 4 x 4
            # in_tensors: bz x 3 x N
            ###### Centralize in_tensors ######


            in_tensors = in_tensors - torch.mean(in_tensors, dim=-1, keepdim=True)



            # part_axis: bz x n_part_mov x 3
            # glb_R: bz x 3 x 3
            # part_axis: bz x 1 x 3 x 3 xxxx bz x n_part_mov x 3 x 1 --> bz x n_part_mov x 3 x 1 --> bz x n_part_mov x 3
            # inv transform part axises...
            part_axis = torch.matmul(safe_transpose(glb_R, -1, -2).unsqueeze(1), part_axis.unsqueeze(-1)).squeeze(-1)

            # # pose: bz x N x 4 x 4; glb_T: bz x 3
            # in_pose[:, :, :3, 3] = in_pose[:, :, :3, 3] - glb_T.unsqueeze(1)
            # in_pose[:, :, :3] = torch.matmul(safe_transpose(glb_R, -1, -2), in_pose[:, :, :3])

        # oorr_pc = torch.matmul(safe_transpose(glb_R, -1, -2), oorr_pc - glb_T.unsqueeze(-1))

        loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc,
                          normals=cur_normals, canon_normals=cur_canon_normals)

        # Need not to further transpose predictions
        # # transform `pred` to prediction probability
        # pred = torch.clamp(pred, min=-20, max=20)
        # pred = torch.softmax(pred, dim=-1)
        # bz x npred-class x N

        # pred = pred.contiguous().transpose(1, 2).contiguous()
        #
        # if pred.size(1) < 200:
        #     pred = torch.cat(
        #         [pred, torch.zeros((bz, 200 - pred.size(1), N), dtype=torch.float32).cuda(non_blocking=True)], dim=1
        #     )

        label = torch.eye(self.model.module.num_slots)[in_label].cuda(non_blocking=True)
        label, gt_conf = self.get_gt_conf(label)

        # get labels for all points
        oorr_label = torch.eye(self.model.module.num_slots)[oorr_label] # .cuda(non_blocking=True)
        # transform labels to labels and gt-part confidence values
        oorr_label, oorr_gt_conf = self.get_gt_conf(oorr_label)

        # pred_part_axis: bz x 3; pred_axis: bz x n_mov_parts x 3
        # pred_part_axis = self.model.module.defined_axises if self._use_multi_gpu else self.model.defined_axises
        pred_part_axis = self.model.module.real_defined_axises if self._use_multi_gpu else self.model.real_defined_axises
        part_axis = part_axis / torch.norm(part_axis, dim=-1, keepdim=True, p=2)
        if self.shape_type != 'drawer':
            pred_part_pv_point_offset = self.model.module.offset_pivot_points if self._use_multi_gpu else self.model.offset_pivot_points
            dist_pred_gt_offset = torch.abs(pred_part_pv_point_offset - part_pv_offset).mean().item()
        else:
            dist_pred_gt_offset = 0.0

        # dot_prod: bz x n_mov_parts
        # print(f"pred_part_axis: {pred_part_axis.size()}, part_axis: {part_axis.size()}")
        dot_prod = torch.abs(torch.sum(pred_part_axis.unsqueeze(1) * part_axis, dim=-1))
        mean_dot_prod_val = dot_prod.mean().item()
        # if self.local_rank == 0:
        #     print(pred_part_axis.size(), part_axis.size(), mean_dot_prod_val, pred_part_axis, part_axis, "offset_dist", dist_pred_gt_offset)

        accs = []
        accs_2 = []

        if 'pred_R_slots' in self.model.module.out_feats:
            for i_iter in range(self.n_iters):
                if i_iter == 0:
                    # bz x nmasks x N
                    cur_pred = self.model.module.attn_iter_0

                    iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)

                    # all_pred: bz x NN x nmasks
                    all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2), indices=minn_idx, dim=1)
                    iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2), gt_x=oorr_label, gt_conf=oorr_gt_conf)

                    accs.append(iou_value.mean())
                    accs_2.append(iou_value_2.mean())
                elif i_iter == 1:
                    cur_pred = self.model.module.attn_iter_1
                    iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)
                    accs.append(iou_value.mean())

                    # all_pred: bz x NN x nmasks
                    all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2), indices=minn_idx, dim=1)
                    iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2),
                                                                              gt_x=oorr_label, gt_conf=oorr_gt_conf)

                    # accs.append(iou_value.mean())
                    accs_2.append(iou_value_2.mean())
                elif i_iter == 2:
                    cur_pred = self.model.module.attn_iter_2
                    iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)
                    accs.append(iou_value.mean())

                    # all_pred: bz x NN x nmasks
                    all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2), indices=minn_idx, dim=1)
                    iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2),
                                                                              gt_x=oorr_label, gt_conf=oorr_gt_conf)

                    # accs.append(iou_value.mean())
                    accs_2.append(iou_value_2.mean())

            ''' Previous prediction and iou calculation '''
            # pred = torch.zeros((bz, 200, N), dtype=torch.float32).cuda(non_blocking=True)
            # iou_value, _, _ = iou(pred, gt_x=label, gt_conf=gt_conf)
            ''' Previous prediction and iou calculation '''

            curr_attn = self.model.module.attn_saved # cur_attn: bz x n_s x N ---> attention weights from points to slots...

            sv_dict['curr_attn'] = curr_attn.detach().cpu().numpy()

            # if self.shape_type == 'drawer':
            #     hard_labels = torch.argmax(curr_attn, dim=1) # bz x N
            #     pred_label_to_real_label = {1: 0, 0: 1, 2: 3, 3: 2, 4: 2, 5: 3}
            #     for i_bz in range(hard_labels.size(0)):
            #         for i_pts in range(hard_labels.size(1)):
            #             cur_bz_cur_pts_label = int(hard_labels[i_bz, i_pts].item())
            #             reindexed_pts_label = pred_label_to_real_label[cur_bz_cur_pts_label]
            #             hard_labels[i_bz, i_pts] = reindexed_pts_label
            #     curr_attn = torch.eye(self.model.module.num_slots)[hard_labels.long()].cuda(non_blocking=True)
            #     curr_attn = safe_transpose(curr_attn, 1, 2)

            iou_value, matching_idx_gt, matching_idx_pred = iou(curr_attn, gt_x=label, gt_conf=gt_conf)

            # iou_loss = -iou_value.mean() * 100
            iou_loss = -iou_value.mean() # * 100 # attn_saved... attn:

            if self.n_iters == 2:
                # current attention
                curr_attn_1 = self.model.module.attn_saved_1
                iou_value_1, matching_idx_gt, matching_idx_pred = iou(curr_attn_1, gt_x=label, gt_conf=gt_conf)

                iou_loss_1 = -iou_value_1.mean() * 100
                iou_loss = (iou_loss + iou_loss_1) / 2.
        else:
            accs = [torch.tensor([0.0], dtype=torch.float32).cuda() for _ in range(self.n_iters)]
            accs_2 = [torch.tensor([0.0], dtype=torch.float32).cuda() for _ in range(self.n_iters)]
            iou_loss = None
            iou_loss = torch.tensor([0.0], dtype=torch.float32).cuda()

        # print(iou_loss)

        sv_dict['out_feats'] = self.model.module.out_feats

        ##############################################
        # predR, featR = self.model(in_tensorsR, in_Rlabel)
        # print(torch.sort(featR[0,0])[0])
        # print(torch.sort(feat[0,0])[0])
        # import ipdb; ipdb.set_trace()
        ##############################################

        # self.optimizer.zero_grad()

        # if self.attention_model:
        #     in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
        #     self.loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000)
        # else:
        # cls_loss, acc = self.metric(pred, in_label)
        # self.loss = cls_loss
        # self.loss = loss + iou_loss
        # print(f"iou_loss: {iou_loss}")
        if iou_loss is not None:
            # self.loss = loss  # + iou_loss
            # print(f"iou_loss: {iou_loss.item()}")
            if self.shape_type == 'drawer':
                # self.loss = loss + iou_loss
                self.loss = iou_loss + loss * 0.00000001
                self.loss = iou_loss + loss * 0.1
                self.loss = iou_loss + loss
                self.loss = loss
            else:
                self.loss = loss # + iou_loss
                # self.loss = loss + 0.0001 *  iou_loss
                # self.loss = loss + 0.2 *  iou_loss
                # self.loss = loss # + 0.0002 *  iou_loss
            # acc = iou_value.mean()
        else:
            self.loss = loss
            # self.loss = loss + iou_loss

        # out_pred_R_np = self.model.module.out_feats['pred_R_slots']
        # # out_pred_R: bz x n_s x 3 x 3
        # out_pred_R = torch.from_numpy(out_pred_R_np).float().cuda()
        # gt_pose = data['pose_segs']
        # # gt_R: bz x n_s x 3 x 3
        # gt_R = gt_pose[:, :, :3, :3].cuda()
        #
        # out_pred_R = batched_index_select(values=out_pred_R, indices=matching_idx_pred.long(), dim=1)
        # gt_R = batched_index_select(values=gt_R, indices=matching_idx_gt.long(), dim=1)
        #
        # avg_R_dist = calculate_res_relative_Rs(out_pred_R, gt_R)

        # pred part axis?

        ''' Sync values from all gpus '''
        torch.distributed.barrier()

        # pred_part_axies =

        self.loss = self.reduce_mean(self.loss, self.nprocs)
        loss = self.reduce_mean(loss, self.nprocs)

        iou_loss = self.reduce_mean(iou_loss, self.nprocs)

        if self.local_rank == 0:
            with open(self.loss_log_saved_path, "a") as wf:
                wf.write(f"Loss: {round(float(loss.item()), 4)}, Iou: {round(float(-1. * iou_loss.item()), 4)}\n")
                wf.close()

        new_accs = []
        for acc in accs:
            new_acc = self.reduce_mean(acc, self.nprocs)
            new_accs.append(new_acc)
        accs = new_accs
        # acc = self.reduce_mean(acc, self.nprocs)

        # avg_R_dist = self.reduce_mean(avg_R_dist, self.nprocs)

        # acc = torch.zeros((1,))
        if accs[-1].item() > self.best_acc_ever_reached:
            self.best_acc_ever_reached = accs[-1].item()

        ''' Optimize loss '''
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Log training stats
        # if self.attention_model:
        #     log_info = {
        #         'Loss': cls_loss.item(),
        #         'Acc': 100 * acc.item(),
        #         'R_Loss': r_loss.item(),
        #         'R_Acc': 100 * r_acc.item(),
        #     }
        # else:
        log_info = {
            'Loss': loss.item(),
            'dot_axis_pred': mean_dot_prod_val
            # 'Acc': 100 * acc.item(),
        }
        for i_iter in range(self.n_iters):
            log_info[f'Acc_{i_iter}'] = accs[i_iter].item() * 100
            log_info[f'Acc_2_{i_iter}'] = accs_2[i_iter].item() * 100
        # log_info['Avg_R_dist'] = avg_R_dist.item()

        # out_feats = self.model.module.out_feats
        # out_feats_all_iters = self.model.module.out_feats_all_iters

        # self.save_predicted_infos(data_idxes, out_feats)
        # self.save_predicted_infos_all_iters(data_idxes, out_feats_all_iters)

        # self.logger.log("Training", "Accuracy: %.1f, Loss: %.2f!" % (100 * acc.item(), cls_loss.item()))

        self.summary.update(log_info)
        if self.local_rank == 0:

            stats = self.summary.get()
            self.logger.log('Training', f'{stats}')
            if not os.path.exists(self.model.module.log_fn):
                os.mkdir(self.model.module.log_fn)
            with open(os.path.join(self.model.module.log_fn, "logs.txt"), "a") as wf:
                # wf.write(f"Loss: {loss.item()}, Acc: {acc.item()}\n")
                wf.write(f"Loss: {loss.item()}\n")
                for i_iter in range(self.n_iters):
                    wf.write(f', Acc_{i_iter}: {accs[i_iter].item()}')
                # wf.write(f", avg_R_dist: {avg_R_dist.item()}")
                wf.write('\n')
                wf.close()

        # out_feats = self.model.module.out_feats
        ### save the sv_dict ###
        # out_feats = self.model.module.sv_dict
        # out_feats = sv_dict
        # idxes_str = ",".join(data_idxes)
        # feat_save_fn = os.path.join(self.model.module.log_fn, f"out_feats_{idxes_str}_rnk_{self.local_rank}.npy")
        # np.save(feat_save_fn, out_feats)

        # feat_all_iters_save_fn = os.path.join(self.model.module.log_fn, f"out_feats_{idxes_str}_all_iters_rnk_{self.local_rank}.npy")
        # np.save(feat_all_iters_save_fn, out_feats_all_iters)

        # print(stats) # print(stats) --- get statistics and dump statistics
        self.last_loss = float(self.summary.get_item('Loss'))
        # print(f"Best current: {self.best_acc_ever_reached * 100}")
        # with open(os.path.join(self.model.log_fn, ))

        # if self.local_rank == 0 and self.n_step % 100 == 0:
        #     self.save_predicted_by_step(self.model.module.out_feats)

    def _optimize_stage_zero(self, data):
        # set to train mode
        # optimize model
        if self._use_multi_gpu:
            self.model.module.train()
        else:
            self.model.train()
        self.metric.train()

        weighted_step = 1.0 # if (self.use_equi == 23 and self.gt_oracle_seg == 1) else weighted_step

        # if local rank is zero, print weighted_step
        if self.local_rank == 0:
            print(f"weighted_step current epoch: {weighted_step}")
        self.model.module.slot_recon_factor = self.slot_recon_factor * weighted_step

        if self.lr_adjust == 1:
            self.adjust_lr_by_loss()
        elif self.lr_adjust == 2:
            self.adjust_lr_by_step()
        # input tensors
        in_tensors = data['pc'].cuda(non_blocking=True)
        data_idxes = data['idx'].detach().cpu().numpy().tolist()
        data_idxes = [ str(ii) for ii in data_idxes]
        # in_tensors = torch.

        bdim = in_tensors.shape[0]
        in_label = data['label'].cuda(non_blocking=True) # .reshape(-1)
        in_pose = data['pose'].cuda(non_blocking=True) #  if self.opt.debug_mode == 'knownatt' else None
        in_pose_segs = data['pose_segs'].cuda(non_blocking=True)
        ori_pc = data['ori_pc'].cuda(non_blocking=True)
        canon_pc = data['canon_pc'].cuda(non_blocking=True)
        oorr_pc = data['oorr_pc']
        oorr_label = data['oorr_label']
        if self.est_normals == 1:
            cur_normals = data['cur_normals'].cuda(non_blocking=True)
            cur_canon_normals = data['cur_canon_normals'].cuda(non_blocking=True)
        else:
            cur_normals = None
            cur_canon_normals = None
        # be af dists
        be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
        minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)

        # import ipdb; ipdb.set_trace()
        # print("input shapes = ", in_tensors.size(), in_label.size(), in_pose.size())

        bz, N = in_tensors.size(0), in_tensors.size(2)

        #### Ground-truth labels ####
        label = torch.eye(self.model.module.num_slots)[in_label].cuda()

        loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc, normals=cur_normals, canon_normals=cur_canon_normals)

        self.loss = loss

        # glb_ori_recon_dist = self.model.module.glb_recon_ori_dist
        # glb_ori_recon_dist = self.reduce_mean(glb_ori_recon_dist, self.nprocs)

        ''' Sync values from all gpus '''
        torch.distributed.barrier()

        self.loss = self.reduce_mean(self.loss, self.nprocs)
        loss = self.reduce_mean(loss, self.nprocs)

        ''' Optimize loss '''
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        log_info = {
            'Loss': loss.item()
        }

        #### Update logging information ####
        self.summary.update(log_info)
        if self.local_rank == 0:
            stats = self.summary.get()
            self.logger.log('Training', f'{stats}')
            if not os.path.exists(self.model.module.log_fn):
                os.mkdir(self.model.module.log_fn)

        out_feats = self.model.module.out_feats
        idxes_str = ",".join(data_idxes)
        feat_save_fn = os.path.join(self.model.module.log_fn, f"out_feats_{idxes_str}_rnk_{self.local_rank}.npy")
        np.save(feat_save_fn, out_feats)

    def _stage_one_eval(self, cur_dataset): # should set the model's stage to 1;
        # save both for train- and test- dataset?
        aligned_data_save_pth = "/share/xueyi/proj_data/aligned_motion_data"
        aligned_data_save_pth = os.path.join(aligned_data_save_pth, self.shape_type)
        if not os.path.exists(aligned_data_save_pth):
            os.mkdir(aligned_data_save_pth)

        # set to train mode
        # optimize model
        if self._use_multi_gpu:
            # self.model.module.train()
            self.model.module.eval()
        else:
            # self.model.train()
            self.model.eval()
        #### loss weight is not that important now ####
        weighted_step = 1.0 # if (self.use_equi == 23 and self.gt_oracle_seg == 1) else weighted_step

        # if local rank is zero, print weighted_step
        if self.local_rank == 0:
            print(f"weighted_step current epoch: {weighted_step}")
        self.model.module.slot_recon_factor = self.slot_recon_factor * weighted_step

        losses = []
        losses_nn = []

        with torch.no_grad():
            for it, data in enumerate(cur_dataset):
                # input tensors
                in_tensors = data['pc'].cuda(non_blocking=True)
                data_idxes = data['idx'].detach().cpu().numpy().tolist()
                data_idxes = [str(ii) for ii in data_idxes]

                shp_idx = data['shp_idx'].cuda(non_blocking=True)
                sample_idx = data['sample_idx'].cuda(non_blocking=True)

                bdim = in_tensors.shape[0]
                in_label = data['label'].cuda(non_blocking=True) # .reshape(-1)
                in_pose = data['pose'].cuda(non_blocking=True) #  if self.opt.debug_mode == 'knownatt' else None
                in_pose_segs = data['pose_segs'].cuda(non_blocking=True)
                ori_pc = data['ori_pc'].cuda(non_blocking=True)
                canon_pc = data['canon_pc'].cuda(non_blocking=True)
                oorr_pc = data['oorr_pc']
                oorr_label = data['oorr_label']
                if self.est_normals == 1:
                    cur_normals = data['cur_normals'].cuda(non_blocking=True)
                    cur_canon_normals = data['cur_canon_normals'].cuda(non_blocking=True)
                else:
                    cur_normals = None
                    cur_canon_normals = None
                # be af dists
                be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
                # minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)

                bz, N = in_tensors.size(0), in_tensors.size(2)

                ###################### ----------- debug only ---------------------
                # in_tensorsR = data['pcR'].to(self.opt.device)
                # import ipdb; ipdb.set_trace()
                ##################### --------------------------------------------

                # feed into the model: in_tensors, in_pos, and no rotation value
                # pred, feat =
                # loss, pred = self.model(in_tensors, in_pose, None)

                #### Ground-truth labels ####
                label = torch.eye(self.model.module.num_slots)[in_label].cuda()
                label, gt_conf = self.get_gt_conf(label)

                # get labels for all points; oorr_labels for input points
                oorr_label = torch.eye(self.model.module.num_slots)[oorr_label]  # .cuda(non_blocking=True)
                # transform labels to labels and gt-part confidence values
                oorr_label, oorr_gt_conf = self.get_gt_conf(oorr_label)

                loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc, normals=cur_normals, canon_normals=cur_canon_normals)

                loss = loss.mean()
                self.loss = loss
                torch.distributed.barrier()

                self.loss = self.reduce_mean(self.loss, self.nprocs)
                loss = self.reduce_mean(loss, self.nprocs)

                losses.append(float(loss.item()) * bz)
                losses_nn.append(bz)
                cur_avg_loss = sum(losses) / sum(losses_nn)
                print(f"Iteration: {it}, avg_loss: {cur_avg_loss}")

                #### of no meaning... ####
                # if self._use_multi_gpu:
                #     curr_attn = self.model.module.attn_saved
                # else:
                #     curr_attn = self.model.attn_saved
                # # iou_value, matching_idx_gt, matching_idx_pred = iou(curr_attn, gt_x=label, gt_conf=gt_conf)
                #### of no meaning... ####

                #### Get predicted globa pose for further data transformation ####
                if self._use_multi_gpu:
                    # bz x 3 x 4?
                    pred_glb_pose = self.model.module.pred_glb_pose
                else:
                    pred_glb_pose = self.model.pred_glb_pose
                # pred_glb_R: bz x 3 x 3; pred_glb_T: bz x 3
                pred_glb_R, pred_glb_T = pred_glb_pose[:, :3, :3], pred_glb_pose[:, :3, 3]
                # in_pose: bz x N x 3(4) x 4
                in_pose[:, :, :3, 3] = in_pose[:, :, :3, 3] - pred_glb_T.unsqueeze(1)
                in_pose[:, :, :3, :] = torch.matmul(safe_transpose(pred_glb_R, -1, -2).unsqueeze(1), in_pose[:, :, :3, :])
                in_pose_segs[:, :, :3, 3] = in_pose_segs[:, :, :3, 3] - pred_glb_T.unsqueeze(1)
                in_pose_segs[:, :, :3, :] = torch.matmul(safe_transpose(pred_glb_R, -1, -2).unsqueeze(1),
                                                    in_pose_segs[:, :, :3, :])
                # in_tensors: bz x 3 x N
                in_tensors = torch.matmul(safe_transpose(pred_glb_R, -1, -2), in_tensors - pred_glb_T.unsqueeze(-1))
                oorr_pc = torch.matmul(safe_transpose(pred_glb_R, -1, -2), oorr_pc - pred_glb_T.unsqueeze(-1))
                if self.est_normals == 1:
                    # cur_normals: bz x 3 x N
                    cur_normals = torch.matmul(safe_transpose(pred_glb_R, -1, -2), cur_normals)
                    cur_canon_normals = torch.matmul(safe_transpose(pred_glb_R, -1, -2), cur_canon_normals)

                for i_bz in range(bz):
                    cur_pc = in_tensors[i_bz].detach().cpu().numpy()
                    cur_data_idx = np.array([int(data_idxes[i_bz])], dtype=np.long)
                    cur_label = in_label[i_bz].detach().cpu().numpy()
                    cur_pose = in_pose[i_bz].detach().cpu().numpy()
                    cur_pose_segs = in_pose_segs[i_bz].detach().cpu().numpy()
                    cur_ori_pc = ori_pc[i_bz].detach().cpu().numpy()
                    cur_canon_pc = canon_pc[i_bz].detach().cpu().numpy()
                    cur_oorr_pc = oorr_pc[i_bz].detach().cpu().numpy()
                    cur_oorr_label = oorr_label[i_bz].detach().cpu().numpy()

                    cur_shp_idx = shp_idx[i_bz].detach().cpu().numpy().item()
                    cur_sample_idx = sample_idx[i_bz].detach().cpu().numpy().item()

                    cur_aligned_data = {
                        'pc': cur_pc,
                        'idx': cur_data_idx,
                        'label': cur_label,
                        'pose': cur_pose,
                        'pose_segs': cur_pose_segs,
                        'ori_pc': cur_ori_pc,
                        'canon_pc': cur_canon_pc,
                        'oorr_pc': cur_oorr_pc,
                        'oorr_label': cur_oorr_label,
                        'shp_idx': np.array([cur_shp_idx], dtype=np.long),
                        'sample_idx': np.array([cur_sample_idx], dtype=np.long)
                    }
                    if self.est_normals == 1:
                        cur_bz_normals = cur_normals[i_bz].detach().cpu().numpy()
                        cur_bz_canon_normals = cur_canon_normals[i_bz].detach().cpu().numpy()
                        cur_aligned_data['cur_normals'] = cur_bz_normals
                        cur_aligned_data['cur_canon_normals'] = cur_bz_canon_normals
                    shp_folder = "%.4d" % cur_shp_idx
                    shp_folder = os.path.join(aligned_data_save_pth, shp_folder)
                    if not os.path.exists(shp_folder):
                        os.mkdir(shp_folder)
                    sample_fn = "%.4d" % cur_sample_idx
                    sample_fn = os.path.join(shp_folder, sample_fn + ".npy")
                    #### Save aligned data ####
                    np.save(sample_fn, cur_aligned_data)

    def adjust_lr_by_loss(self):
        # if self.local_rank == 0:
        # print(f"self.local_rank: {self.local_rank} self.last_loss: {self.last_loss}, self.best_loss: {self.best_loss}")
        if self.last_loss < self.best_loss:
            # print()
            self.best_loss = self.last_loss
            self.not_increased_epoch = 0
        else:
            self.not_increased_epoch += 1
            if self.not_increased_epoch >= 30:
                print(f"Adjusting learning rate by {self.lr_decay_factor}!")
                self.adjust_learning_rate_by_factor(self.lr_decay_factor)
                self.not_increased_epoch = 0

    def adjust_lr_by_step(self):
        if self.n_step > 0 and self.n_step % self.n_dec_steps == 0:
            self.adjust_learning_rate_by_factor(self.lr_decay_factor)
            print(f"Adjusting learning rate by {self.lr_decay_factor}!")
        self.n_step += 1

    def adjust_learning_rate_by_factor(self, scale_factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * scale_factor, 1e-7)

    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self):
        return None
        self.eval()
        return None

    def get_gt_conf(self, label):
        # bz x nmask x N?
        #
        if label.size(1) > label.size(2):
            label = label.transpose(1, 2)
        gt_conf = torch.sum(label, 2)
        # gt_conf = torch.where(gt_conf > 0, 1, 0).float()
        gt_conf = torch.where(gt_conf > 0, torch.ones_like(gt_conf), torch.zeros_like(gt_conf)).float()
        return label, gt_conf

    def eval_bak(self):
        ''' Need j '''
        # Use one gpu for eval
        # evaluate test dataset
        self.logger.log('Testing','Evaluating test set!')
        # self.model.module.eval()
        # self.metric.eval()

        ''' Set queeue lenght to full length... '''
        self.model.module.queue_len = self.model.module.queue_tot_len


        if self.pre_compute_delta == 1:
            # self.model.module.gt_oracle_seg = 1
            # better evaluate it in a single card?
            #
            set_dr = []
            set_dt = []
            shape_type_to_n_parts = {"eyeglasses": 3, "oven": 2, "washing_machine": 2, "laptop": 2, "drawer": 4, "safe": 2}
            n_parts = shape_type_to_n_parts[self.shape_type]
            set_dr = {}; set_dt = {}
            delta_rs, delta_ts = {}, {}
            delta_r_scores, delta_t_scores = {}, {}
            # remember rotation and translation
            # delta_r
            for i_p in range(n_parts + 2):
                set_dr[i_p] = []
                set_dt[i_p] = []
            with torch.no_grad():
                for it, data in enumerate(self.dataset):
                    #
                    in_tensors = data['pc'].cuda(non_blocking=True)
                    bdim = in_tensors.shape[0]
                    in_label = data['label'].cuda(non_blocking=True)  # .reshape(-1)
                    in_pose = data['pose'].cuda(non_blocking=True)  # if self.opt.debug_mode == 'knownatt' else None
                    in_pose_segs = data['pose_segs'].cuda(non_blocking=True)
                    ori_pc = data['ori_pc'].cuda(non_blocking=True)
                    canon_pc = data['canon_pc'].cuda(non_blocking=True)

                    oorr_pc = data['oorr_pc']
                    oorr_label = data['oorr_label']
                    be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
                    minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)

                    # print("intensor_size:", in_tensors.size())

                    data_idxes = data['idx'].detach().cpu().numpy().tolist()
                    # a complete or a
                    data_idxes = [str(ii) for ii in data_idxes]

                    bz = in_tensors.size(0)
                    N = in_tensors.size(2)

                    # get one-hot label
                    label = torch.eye(self.opt.nmasks)[in_label].cuda()
                    # get loss by forwarding data through the model
                    loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                                      canon_pc=canon_pc)

                    print(f"loss: {loss}")

                    ''' NOTE: We use GT Seg!!! '''
                    # pred_R: bz x n_s x 3 x 3
                    # selected predicted rotations for each slot
                    pred_R = self.model.module.pred_R
                    # pred_T: bz x n_s x 3
                    # pred_T = self.model.module.ori_pred_T
                    # so what does ori_pred_T represent
                    # selected predicted translations for each slot
                    pred_T = self.model.module.pred_T
                    # pred_T = self.model.module.pred_T
                    n_s = pred_T.size(1)

                    # Get predicted attention
                    # bz x n_s x N
                    pred_attn_ori = self.model.module.attn_saved if self.n_iters == 1 else self.model.module.attn_saved_1
                    # pred_labels: bz x N
                    pred_labels = torch.argmax(pred_attn_ori, dim=1).long()
                    # pred_hard_one_hot_labels: bz x N x n_s --> bz x n_s x N
                    pred_hard_one_hot_labels = torch.eye(pred_attn_ori.size(1), dtype=torch.float32).cuda()[pred_labels]
                    pred_hard_one_hot_labels = torch.transpose(pred_hard_one_hot_labels.contiguous(), -1, -2)

                    # boundary_pts = [np.min(sampled_pcts, axis=0), np.max(sampled_pcts, axis=0)]
                    # center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
                    # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
                    # for
                    # in_tensors: bz x 3 x N --> bz x N x 3 --> bz x n_s x N x 3
                    expand_xyz = safe_transpose(in_tensors, -1, -2).unsqueeze(1).contiguous().repeat(1, self.num_slots, 1, 1)
                    #



                    # pred_R; pred_T
                    for i_p in range(n_s): # get rotation and translation for each slot
                        cur_part_R = pred_R[:, i_p, ...]
                        cur_part_T = pred_T[:, i_p, ...]
                        # remember predicted translation and rotation
                        set_dr[i_p].append(cur_part_R)
                        set_dt[i_p].append(cur_part_T)

                    # out_feats
                    out_feats = self.model.module.out_feats
                    idxes_str = ",".join(data_idxes)
                    feat_save_fn = os.path.join(self.model.module.log_fn, f"eval_tr_out_feats_{idxes_str}_rnk_{self.local_rank}.npy")
                    np.save(feat_save_fn, out_feats)

                # for i_p in range(n_parts):
                for i_p in range(n_s):
                    # after cat: tot_bz x 3 x 3
                    # set_dr
                    set_dr[i_p] = torch.cat(set_dr[i_p], dim=0)
                    # after cat: tot_bz x 3
                    set_dt[i_p] = torch.cat(set_dt[i_p], dim=0)

                    # todo: add flip axis and chosen axis for parts in each category
                    # rotations for this slot
                    print(set_dr[i_p].size())
                    # ransac fit rotation from canonical posed ...
                    delta_r, r_score = ransac_fit_r(set_dr[i_p], chosen_axis=None, flip_axis=None)
                    # todo: delta_r is not used here?
                    delta_t, t_score = ransac_fit_t(set_dt[i_p], set_dr[i_p], delta_r.squeeze())
                    delta_rs[i_p] = delta_r
                    #
                    print(f"current delta r, shape: {delta_r.size()}, values: {delta_r}")
                    delta_ts[i_p] = delta_t
                    #
                    delta_r_scores[i_p] = r_score
                    delta_t_scores[i_p] = t_score
                    # print(f"category {self.shape_type}, part {i_p}, delta_r {delta_r}, delta_t {delta_t}, r_score {r_score}, t_score {t_score}")
                    print(f"category {self.shape_type}, part {i_p}, r_score {r_score}, t_score {t_score}")
        else:
            delta_rs = {}
            delta_ts = {}
            for i_s in range(self.num_slots):
                delta_rs[i_s] = torch.eye(3).cuda()
                delta_ts[i_s] = torch.zeros((3,)).cuda()

            # self.model.module.gt_oracle_seg = 0

        ################## DEBUG ###############################
        # for module in self.model.modules():
        #     if isinstance(module, torch.nn.modules.BatchNorm1d):
        #         module.train()
        #     if isinstance(module, torch.nn.modules.BatchNorm2d):
        #         module.train()
        #     if isinstance(module, torch.nn.modules.BatchNorm3d):
        #         module.train()
            # if isinstance(module, torch.nn.Dropout):
            #     module.train()
        #####################################################

        with torch.no_grad():
            accs = [[] for i_iter in range(self.n_iters)]
            accs_2 = [[] for i_iter in range(self.n_iters)]
            # lmc = np.zeros([40,60], dtype=np.int32)

            # idx to evaluation infos
            eval_infos = {}

            all_labels = []
            all_feats = []
            avg_R_dists = []

            part_idx_to_rot_diff = {}
            part_idx_to_canon_rot_diff = {}
            slot_idx_to_rot_diff = {}
            slot_idx_to_canon_rot_diff = {}
            part_idx_to_canon_rot_diff_zz = {}
            part_idx_to_rot_diff_zz = {}
            part_idx_to_trans_diff = {}
            part_idx_to_trans_diff_zz = {}
            part_idx_to_trans_diff_2 = {}
            part_idx_to_trans_diff_2_zz = {}
            part_rel_rot_diff = []
            part_pair_to_part_rel_rot_diff = {}
            part_pair_to_part_rel_rot_delta_diff = {}

            part_idx_to_pred_posed_canon_diff = {}
            part_idx_to_pred_posed_posed_diff = {}

            for it, data in enumerate(self.dataset_test):
                in_tensors = data['pc'].cuda(non_blocking=True)
                bdim = in_tensors.shape[0]
                in_label = data['label'].cuda(non_blocking=True) # .reshape(-1)
                # per-point pose
                in_pose = data['pose'].cuda(non_blocking=True)  # if self.opt.debug_mode == 'knownatt' else None
                # per-part pose
                in_pose_segs = data['pose_segs'].cuda(non_blocking=True)
                ori_pc = data['ori_pc'].cuda(non_blocking=True)
                canon_pc = data['canon_pc'].cuda(non_blocking=True)
                part_state_rots = data['part_state_rots'].cuda(non_blocking=True)
                part_ref_rots = data['part_ref_rots'].cuda(non_blocking=True) # [0]
                if 'part_ref_trans' not in data:
                    part_ref_trans = torch.zeros((in_tensors.size(0), in_pose_segs.size(1), 3), dtype=torch.float).cuda()
                else:
                    part_ref_trans = data['part_ref_trans'].cuda(non_blocking=True) # [0]

                oorr_pc = data['oorr_pc']
                oorr_label = data['oorr_label']
                be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
                minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)

                data_idxes = data['idx'].detach().cpu().numpy().tolist()
                data_idxes = [str(ii) for ii in data_idxes]

                bz = in_tensors.size(0)
                N = in_tensors.size(2)

                label = torch.eye(self.opt.nmasks)[in_label].cuda(non_blocking=True)
                label, gt_conf = self.get_gt_conf(label)

                loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc)

                out_feats = self.model.module.out_feats
                out_feats_all_iters = self.model.module.out_feats_all_iters

                out_feats["part_state_rots"] = part_state_rots.detach().cpu().numpy()
                out_feats["part_ref_rots"] = part_ref_rots.detach().cpu().numpy()
                out_feats["part_ref_trans"] = part_ref_trans.detach().cpu().numpy()

                # Need not to further transpose predictions
                # # transform `pred` to prediction probability
                # pred = torch.clamp(pred, min=-20, max=20)
                # pred = torch.softmax(pred, dim=-1)
                # bz x npred-class x N

                # pred = torch.zeros((bz, 200, N), dtype=torch.float32).cuda(non_blocking=True)

                # loss, pred = self.model(in_tensors, in_pose)
                #
                # # # transform `pred` to prediction probability
                # # pred = torch.clamp(pred, min=-20, max=20)
                # # pred = torch.softmax(pred, dim=-1)
                # pred = pred.contiguous().transpose(1, 2).contiguous()
                #
                # if pred.size(1) < 200:
                #     pred = torch.cat(
                #         [pred, torch.zeros((bz, 200 - pred.size(1), N), dtype=torch.float32).cuda()], dim=1
                #     )

                # from in_label to label

                # label = torch.eye(self.model.module.num_slots)[in_label].cuda(non_blocking=True)
                # label, gt_conf = self.get_gt_conf(label)

                oorr_label = torch.eye(self.model.module.num_slots)[oorr_label] # .cuda(non_blocking=True)
                oorr_label, oorr_gt_conf = self.get_gt_conf(oorr_label)

                cur_accs = []
                cur_accs_2 = []

                for i_iter in range(self.n_iters):
                    if i_iter == 0:
                        cur_pred = self.model.module.attn_iter_0
                        iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)

                        cur_accs.append(iou_value.mean())
                        # all_pred: bz x NN x nmasks
                        all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2),
                                                        indices=minn_idx, dim=1)
                        iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2),
                                                                                  gt_x=oorr_label, gt_conf=oorr_gt_conf)

                        cur_accs_2.append(iou_value_2.mean())
                    elif i_iter == 1:
                        cur_pred = self.model.module.attn_iter_1
                        iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)
                        cur_accs.append(iou_value.mean())
                        # all_pred: bz x NN x nmasks
                        all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2),
                                                        indices=minn_idx, dim=1)
                        iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2),
                                                                                  gt_x=oorr_label, gt_conf=oorr_gt_conf)
                        cur_accs_2.append(iou_value_2.mean())

                    elif i_iter == 2:
                        cur_pred = self.model.module.attn_iter_2
                        iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)
                        cur_accs.append(iou_value.mean())
                        # all_pred: bz x NN x nmasks
                        all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2),
                                                        indices=minn_idx, dim=1)
                        iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2),
                                                                                  gt_x=oorr_label, gt_conf=oorr_gt_conf)
                        cur_accs_2.append(iou_value_2.mean())

                # iou_value, _, _ = iou(pred, gt_x=label, gt_conf=gt_conf)

                # loss = -iou_value.mean()

                # if self.attention_model:
                #     in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
                #     loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000)
                #     attention = F.softmax(feat,1)
                #
                #     if self.opt.train_loss.attention_loss_type == 'no_cls':
                #         acc = r_acc
                #         loss = r_loss
                #
                #     # max_id = attention.max(-1)[1].detach().cpu().numpy()
                #     # labels = data['label'].cpu().numpy().reshape(-1)
                #     # for i in range(max_id.shape[0]):
                #     #     lmc[labels[i], max_id[i]] += 1
                # else:
                # cls_loss, acc = self.metric(pred, in_label)
                # loss = cls_loss
                #
                out_pred_R_np = self.model.module.out_feats['pred_R_slots']
                out_pred_R = torch.from_numpy(out_pred_R_np).float().cuda()
                out_pred_T = self.model.module.pred_T
                # out_pred_T = self.model.module.ori_pred_T
                gt_pose = data['pose_segs']
                gt_R = gt_pose[:, :, :3, :3].cuda()

                # matching_idx_gt: bz x n_slots
                if self.shape_type == 'drawer':
                    matching_idx_gt = torch.tensor([0, 1, 2, 3], dtype=torch.long).cuda().unsqueeze(0).repeat(bz, 1)
                    matching_idx_pred = torch.tensor([2, 3, 0, 1], dtype=torch.long).cuda().unsqueeze(0).repeat(bz, 1)

                out_pred_R = batched_index_select(values=out_pred_R, indices=matching_idx_pred.long(), dim=1)
                out_pred_T = batched_index_select(values=out_pred_T, indices=matching_idx_pred.long(), dim=1)
                gt_R = batched_index_select(values=gt_R, indices=matching_idx_gt.long(), dim=1)

                avg_R_dist = calculate_res_relative_Rs(out_pred_R, gt_R)

                out_feats["matching_idx_pred"] = matching_idx_pred.detach().cpu().numpy()
                out_feats["matching_idx_gt"] = matching_idx_gt.detach().cpu().numpy()

                torch.distributed.barrier()

                cur_new_accs = []
                cur_new_accs_2 = []
                for acc in cur_accs:
                    new_acc = self.reduce_mean(acc, self.nprocs)
                    cur_new_accs.append(new_acc)
                for acc in cur_accs_2:
                    # new_acc = self.reduce_mean(acc, self.nprocs)
                    cur_new_accs_2.append(acc)
                cur_accs = cur_new_accs
                cur_accs_2 = cur_new_accs_2

                # acc = iou_value.mean()
                all_labels.append(in_label.cpu().numpy())

                loss = self.reduce_mean(loss, self.nprocs)
                avg_R_dist = self.reduce_mean(avg_R_dist, self.nprocs)

                avg_R_dists.append(avg_R_dist.item())
                # acc = self.reduce_mean(acc, self.nprocs)
                # all_feats.append(feat.cpu().numpy())

                loss_ = self.model(safe_transpose(canon_pc, 1, 2), in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc)
                # out-pred-R
                canon_out_pred_R_np = self.model.module.out_feats['pred_R_slots']
                canon_out_pred_R = torch.from_numpy(canon_out_pred_R_np).float().cuda()
                canon_out_pred_T = self.model.module.pred_T
                # canon_out_pred_T = self.model.module.ori_pred_T

                canon_out_pred_R = batched_index_select(values=canon_out_pred_R, indices=matching_idx_pred.long(), dim=1)
                canon_out_pred_T = batched_index_select(values=canon_out_pred_T, indices=matching_idx_pred.long(), dim=1)

                # canonical out prediction T
                real_pred_T = out_pred_T - torch.matmul(torch.matmul(out_pred_R, safe_transpose(canon_out_pred_R, -1, -2)), canon_out_pred_T.unsqueeze(-1)).squeeze(-1)
                # real_pred_T

                # if use pre compute delta
                if self.pre_compute_delta >= 1:
                    # pred_R: bz x n_s x 3
                    pred_R = self.model.module.pred_R
                    # pred_T: bz x n_s x 3
                    # pred_T = self.model.module.pred_T
                    pred_T = self.model.module.ori_pred_T

                    pred_R_parts, pred_T_parts = [], []

                    cur_iter_rot_diff = []
                    cur_iter_rot_diff_canon = []

                    # matching_idx_gt
                    for i_bz in range(bz):
                        cur_bz_pred_Rs = []
                        cur_bz_delta_Rs = []
                        cur_bz_ref_Rs = []
                        cur_iter_cur_bz_rot_diff = {}
                        cur_iter_cur_bz_rot_diff_canon = {}
                        for it_p in range(matching_idx_gt.size(1)):
                            cur_gt_part_idx = int(matching_idx_gt[i_bz, it_p].item())
                            cur_pred_part_idx = int(matching_idx_pred[i_bz, it_p].item())
                            cur_bz_cur_match_pred_R = out_pred_R[i_bz, it_p, :, :]
                            cur_bz_cur_match_canon_pred_R = canon_out_pred_R[i_bz, it_p, :, :]
                            # cur_bz_cur_match_pred_R = cur_bz_cur_match_pred_R.contiguous().view(3, 3).contiguous()
                            cur_bz_cur_match_pred_T = out_pred_T[i_bz, it_p, :]
                            cur_bz_cur_match_real_pred_T = real_pred_T[i_bz, it_p, :]
                            # cur_bz_cur_match_delta_R = delta_rs[cur_gt_part_idx].contiguous().view(3, 3).contiguous()
                            # Get the predicted delta_R for the current part
                            cur_bz_cur_match_delta_R = delta_rs[cur_pred_part_idx].contiguous().view(3, 3).contiguous()
                            # cur_bz_cur_match_delta_T = delta_ts[cur_gt_part_idx]
                            cur_bz_cur_match_delta_T = delta_ts[cur_pred_part_idx]
                            # delta
                            cur_bz_cur_match_pred_rot = torch.matmul(cur_bz_cur_match_pred_R, cur_bz_cur_match_delta_R.contiguous().transpose(-1, -2).contiguous())
                            # cur_bz_cur_match_pred_canon_rot = torch.matmul(cur_bz_cur_match_pred_R, cur_bz_cur_match_delta_R.contiguous().transpose(0, 1).contiguous())
                            cur_bz_cur_match_pred_canon_rot = torch.matmul(cur_bz_cur_match_pred_R, cur_bz_cur_match_canon_pred_R.contiguous().transpose(-1, -2).contiguous())
                            cur_bz_cur_match_pred_pred_rot = torch.matmul(cur_bz_cur_match_pred_R, cur_bz_cur_match_pred_R.contiguous().transpose(-1, -2).contiguous())

                            # cur_bz_cur_match_gt_rot = in_pose_segs[i_bz, cur_gt_part_idx, :3, :3]
                            cur_bz_cur_match_gt_trans = in_pose_segs[i_bz, cur_gt_part_idx, :3, 3]
                            #
                            cur_bz_cur_match_gt_rot = torch.matmul(part_state_rots[i_bz, cur_gt_part_idx], safe_transpose(part_ref_rots[i_bz, cur_gt_part_idx], -1, -2))

                            cur_bz_cur_match_gt_canon_trans = part_ref_trans[i_bz, cur_gt_part_idx]
                            cur_bz_cur_match_gt_state_trans = in_pose_segs[i_bz, cur_gt_part_idx, :3, 3]
                            real_gt_T = cur_bz_cur_match_gt_state_trans - torch.matmul(
                                torch.matmul(part_state_rots[i_bz, cur_gt_part_idx], safe_transpose(part_ref_rots[i_bz, cur_gt_part_idx], -1, -2)), cur_bz_cur_match_gt_canon_trans.unsqueeze(-1)).squeeze(-1)

                            if cur_gt_part_idx not in part_idx_to_pred_posed_canon_diff:
                                part_idx_to_pred_posed_canon_diff[cur_gt_part_idx] = [cur_bz_cur_match_pred_canon_rot.unsqueeze(0)]
                            else:
                                part_idx_to_pred_posed_canon_diff[cur_gt_part_idx].append(cur_bz_cur_match_pred_canon_rot.unsqueeze(0))

                            if cur_gt_part_idx == 0:
                                cur_bz_cur_match_diff = rot_diff_degree(cur_bz_cur_match_pred_rot.unsqueeze(0),
                                                                        cur_bz_cur_match_gt_rot.unsqueeze(0)).item()

                                # cur_bz_cur_match_canon_diff = rot_diff_degree(cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                #                                         cur_bz_cur_match_canon_pred_R.unsqueeze(0))

                                cur_bz_cur_match_canon_diff = rot_diff_degree(
                                    cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                    cur_bz_cur_match_gt_rot.unsqueeze(0)).item()
                                cur_bz_cur_match_pred_pred_diff = rot_diff_degree(
                                    cur_bz_cur_match_pred_pred_rot.unsqueeze(0), cur_bz_cur_match_pred_pred_rot.unsqueeze(0)
                                ).item()
                            else:
                                # cur_bz_cur_match_diff = rot_diff_degree(cur_bz_cur_match_pred_rot.unsqueeze(0), cur_bz_cur_match_gt_rot.unsqueeze(0)) - 90
                                # cur_bz_cur_match_canon_diff = rot_diff_degree(cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                #                                         cur_bz_cur_match_canon_pred_R.unsqueeze(0)) - 90
                                cur_bz_cur_match_diff = rot_diff_degree(cur_bz_cur_match_pred_rot.unsqueeze(0),
                                                                        cur_bz_cur_match_gt_rot.unsqueeze(0)).item()

                                # cur_bz_cur_match_canon_diff = rot_diff_degree(cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                    # cur_bz_cur_match_canon_pred_R.unsqueeze(0))

                                cur_bz_cur_match_canon_diff = rot_diff_degree(
                                    cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                    cur_bz_cur_match_gt_rot.unsqueeze(0)).item()

                                cur_bz_cur_match_pred_pred_diff = rot_diff_degree(
                                    cur_bz_cur_match_pred_pred_rot.unsqueeze(0), cur_bz_cur_match_pred_pred_rot.unsqueeze(0)
                                ).item()

                            cur_iter_cur_bz_rot_diff[cur_gt_part_idx] = cur_bz_cur_match_diff
                            cur_iter_cur_bz_rot_diff_canon[cur_gt_part_idx] = cur_bz_cur_match_canon_diff

                            # get rotation difference
                            # cur_bz_cur_match_diff = abs(cur_bz_cur_match_diff)
                            # cur_bz_cur_match_diff = cur_bz_cur_match_diff
                            # part_idx_to_canon_rot_diff_zz

                            cur_bz_cur_match_diff_t = torch.norm(
                                cur_bz_cur_match_pred_T - cur_bz_cur_match_delta_T - real_gt_T, dim=-1).mean().item()
                            # cur_bz_cur_match_diff_t_2 = torch.norm(cur_bz_cur_match_real_pred_T - cur_bz_cur_match_gt_trans, dim=-1).mean().item()
                            cur_bz_cur_match_diff_t_2 = torch.norm(cur_bz_cur_match_real_pred_T - real_gt_T,
                                                                   dim=-1).mean().item()

                            print(cur_bz_cur_match_diff, cur_bz_cur_match_canon_diff, cur_bz_cur_match_pred_pred_diff)
                            if not cur_gt_part_idx in part_idx_to_canon_rot_diff_zz:
                                part_idx_to_canon_rot_diff_zz[cur_gt_part_idx] = [cur_bz_cur_match_canon_diff]
                                part_idx_to_rot_diff_zz[cur_gt_part_idx] = [cur_bz_cur_match_diff]
                                part_idx_to_trans_diff_zz[cur_gt_part_idx] = [cur_bz_cur_match_diff_t]
                                part_idx_to_trans_diff_2_zz[cur_gt_part_idx] = [cur_bz_cur_match_diff_t_2]
                            else:
                                part_idx_to_canon_rot_diff_zz[cur_gt_part_idx].append(cur_bz_cur_match_canon_diff)
                                part_idx_to_rot_diff_zz[cur_gt_part_idx].append(cur_bz_cur_match_diff)
                                part_idx_to_trans_diff_zz[cur_gt_part_idx].append(cur_bz_cur_match_diff_t)
                                part_idx_to_trans_diff_2_zz[cur_gt_part_idx].append(cur_bz_cur_match_diff_t_2)
                            cur_bz_cur_match_diff = min(180. - cur_bz_cur_match_diff, cur_bz_cur_match_diff)
                            # cur_bz_cur_match_canon_diff = abs(cur_bz_cur_match_canon_diff)
                            # cur_bz_cur_match_canon_diff = cur_bz_cur_match_canon_diff
                            cur_bz_cur_match_canon_diff = min(180. - cur_bz_cur_match_canon_diff, cur_bz_cur_match_canon_diff)
                            # get translation difference
                            # try:
                            # cur_bz_cur_match_diff_t = torch.norm(cur_bz_cur_match_pred_T - cur_bz_cur_match_delta_T - cur_bz_cur_match_gt_trans, dim=-1).mean().item()

                            # except:

                            print(cur_bz_cur_match_diff_t_2)

                            if not cur_gt_part_idx in part_idx_to_rot_diff:
                                part_idx_to_rot_diff[cur_gt_part_idx] = [cur_bz_cur_match_diff]
                                part_idx_to_canon_rot_diff[cur_gt_part_idx] = [cur_bz_cur_match_canon_diff]
                                part_idx_to_trans_diff[cur_gt_part_idx]  = [cur_bz_cur_match_diff_t]
                                part_idx_to_trans_diff_2[cur_gt_part_idx] = [cur_bz_cur_match_diff_t_2]
                                part_idx_to_pred_posed_posed_diff[cur_gt_part_idx] = [cur_bz_cur_match_pred_pred_diff]
                            else:
                                part_idx_to_rot_diff[cur_gt_part_idx].append(cur_bz_cur_match_diff)
                                part_idx_to_canon_rot_diff[cur_gt_part_idx].append(cur_bz_cur_match_canon_diff)
                                part_idx_to_trans_diff[cur_gt_part_idx].append(cur_bz_cur_match_diff_t)
                                part_idx_to_trans_diff_2[cur_gt_part_idx].append(cur_bz_cur_match_diff_t_2)
                                part_idx_to_pred_posed_posed_diff[cur_gt_part_idx].append(cur_bz_cur_match_pred_pred_diff)

                            if cur_pred_part_idx not in slot_idx_to_rot_diff:
                                slot_idx_to_rot_diff[cur_pred_part_idx] = [cur_bz_cur_match_diff]
                                slot_idx_to_canon_rot_diff[cur_pred_part_idx] = [cur_bz_cur_match_canon_diff]
                            else:
                                slot_idx_to_rot_diff[cur_pred_part_idx].append(cur_bz_cur_match_diff)
                                slot_idx_to_canon_rot_diff[cur_pred_part_idx].append(cur_bz_cur_match_canon_diff)

                            cur_bz_pred_Rs.append(cur_bz_cur_match_pred_R)
                            cur_bz_delta_Rs.append(cur_bz_cur_match_delta_R)
                            cur_bz_ref_Rs.append(cur_bz_cur_match_pred_rot)

                        cur_iter_rot_diff.append(cur_iter_cur_bz_rot_diff)
                        cur_iter_rot_diff_canon.append(cur_iter_cur_bz_rot_diff_canon)

                        part_rel_R = torch.matmul(cur_bz_ref_Rs[0], safe_transpose(cur_bz_ref_Rs[1], 0, 1))
                        gt_part_rel_R = torch.matmul(part_ref_rots[i_bz, 0], safe_transpose(part_ref_rots[i_bz, 1], 0, 1))
                        cur_bz_part_rel_R_rot_diff = rot_diff_degree(part_rel_R.unsqueeze(0),
                                                                gt_part_rel_R.unsqueeze(0)).item()

                        part_rel_rot_diff.append(cur_bz_part_rel_R_rot_diff)

                        # for it_p in range(matching_idx_gt.size(1)):
                        #     cur_gt_part_idx = int(matching_idx_gt[i_bz, it_p].item())

                        for ip_a in range(matching_idx_gt.size(1) - 1):
                            gt_part_idx_a = int(matching_idx_gt[i_bz, ip_a].item())
                            pred_part_idx_a = int(matching_idx_pred[i_bz, ip_a].item())
                            for ip_b in range(ip_a + 1, matching_idx_gt.size(1)):
                                gt_part_idx_b = int(matching_idx_gt[i_bz, ip_b].item())
                                pred_part_idx_b = int(matching_idx_pred[i_bz, ip_b].item())

                                pred_R_a = out_pred_R[i_bz, ip_a, :, :]
                                canon_pred_R_a = canon_out_pred_R[i_bz, ip_a, :, :]
                                pred_R_b = out_pred_R[i_bz, ip_b, :, :]
                                canon_pred_R_b = canon_out_pred_R[i_bz, ip_b, :, :]

                                delta_R_a = delta_rs[pred_part_idx_a].contiguous().view(3, 3).contiguous()
                                delta_R_b = delta_rs[pred_part_idx_b].contiguous().view(3, 3).contiguous()

                                pred_R_a = torch.matmul(pred_R_a, safe_transpose(canon_pred_R_a, -1, -2))
                                pred_R_b = torch.matmul(pred_R_b, safe_transpose(canon_pred_R_b, -1, -2))

                                pred_R_a_delta = torch.matmul(pred_R_a, safe_transpose(delta_R_a, -1, -2))
                                pred_R_b_delta = torch.matmul(pred_R_b, safe_transpose(delta_R_b, -1, -2))

                                rel_rot_R = torch.matmul(pred_R_a, safe_transpose(pred_R_b, -1, -2))
                                rel_rot_R_delta = torch.matmul(pred_R_a_delta, safe_transpose(pred_R_b_delta, -1, -2))

                                gt_R_a = part_state_rots[i_bz, gt_part_idx_a]
                                gt_R_b = part_state_rots[i_bz, gt_part_idx_b]

                                gt_canon_R_a = part_ref_rots[i_bz, gt_part_idx_a]
                                gt_canon_R_b = part_ref_rots[i_bz, gt_part_idx_b]

                                gt_R_a = torch.matmul(gt_R_a, safe_transpose(gt_canon_R_a, -1, -2))
                                gt_R_b = torch.matmul(gt_R_b, safe_transpose(gt_canon_R_b, -1, -2))

                                # relative rotation between part a and part b
                                gt_rel_rot_R = torch.matmul(gt_R_a, safe_transpose(gt_R_b, 0, 1))

                                part_rel_gt_rot_diff = rot_diff_degree(
                                    rel_rot_R.unsqueeze(0),
                                    gt_rel_rot_R.unsqueeze(0))

                                part_rel_delta_gt_rot_diff = rot_diff_degree(
                                    rel_rot_R_delta.unsqueeze(0),
                                    gt_rel_rot_R.unsqueeze(0)
                                )

                                part_rel_gt_rot_diff = min(part_rel_gt_rot_diff, 180. - part_rel_gt_rot_diff)
                                part_rel_delta_gt_rot_diff = min(part_rel_delta_gt_rot_diff, 180. - part_rel_delta_gt_rot_diff)

                                if gt_part_idx_a < gt_part_idx_b:
                                    cur_part_pari = (gt_part_idx_a, gt_part_idx_b)
                                else:
                                    cur_part_pari = (gt_part_idx_b, gt_part_idx_a)
                                # cur_part_pari = (gt_part_idx_a, gt_part_idx_b)
                                # cur_part_pair_inv = (gt_part_idx_b, gt_part_idx_a)
                                if cur_part_pari not in part_pair_to_part_rel_rot_diff:
                                    part_pair_to_part_rel_rot_diff[cur_part_pari] = [part_rel_gt_rot_diff]
                                    part_pair_to_part_rel_rot_delta_diff[cur_part_pari] = [part_rel_delta_gt_rot_diff]
                                else:
                                    part_pair_to_part_rel_rot_diff[cur_part_pari].append(part_rel_gt_rot_diff)
                                    part_pair_to_part_rel_rot_delta_diff[cur_part_pari].append(part_rel_delta_gt_rot_diff)

                log_str = "Loss: %.2f" % loss.item()
                for i_iter in range(self.n_iters):
                    log_str += f" Acc_{i_iter}: %.2f" % (100 * cur_accs[i_iter].item())
                    log_str += f" Acc_2_{i_iter}: %.2f" % (100 * cur_accs_2[i_iter].item())
                log_str += f" avg_R_dist: %.4f"%(avg_R_dist.item())
                for i_iter in range(self.n_iters):
                    cur_acc_item = float(cur_accs[i_iter].detach().item())
                    cur_acc_2_item = float(cur_accs_2[i_iter].detach().item())
                    accs[i_iter].append(cur_acc_item)
                    accs_2[i_iter].append(cur_acc_2_item)

                canon_out_feats = self.model.module.out_feats
                canon_out_feats_all_iters = self.model.module.out_feats_all_iters

                # self.save_predicted_infos(data_idxes, out_feats)
                # self.save_predicted_infos_all_iters(data_idxes, out_feats_all_iters)

                out_feats['rot_diff'] = cur_iter_rot_diff
                out_feats['rot_diff_canon'] = cur_iter_rot_diff_canon

                idxes_str = ",".join(data_idxes)
                feat_save_fn = os.path.join(self.model.module.log_fn,
                                            f"test_out_feats_{idxes_str}_rnk_{self.local_rank}.npy")
                np.save(feat_save_fn, out_feats)

                feat_all_iters_save_fn = os.path.join(self.model.module.log_fn,
                                                      f"test_out_feats_{idxes_str}_all_iters_rnk_{self.local_rank}.npy")
                np.save(feat_all_iters_save_fn, out_feats_all_iters)

                # idxes_str = ",".join(data_idxes)
                feat_save_fn = os.path.join(self.model.module.log_fn,
                                            f"test_canon_out_feats_{idxes_str}_rnk_{self.local_rank}.npy")
                np.save(feat_save_fn, canon_out_feats)

                feat_all_iters_save_fn = os.path.join(self.model.module.log_fn,
                                                      f"test_canon_out_feats_{idxes_str}_all_iters_rnk_{self.local_rank}.npy")
                np.save(feat_all_iters_save_fn, canon_out_feats_all_iters)

                # accs.append(cur_accs[-1].detach().cpu().numpy())

                # self.logger.log("Testing", "Accuracy: %.1f, Loss: %.2f!"%(100*acc.item(), loss.item()))
                # if self.attention_model:
                #     self.logger.log("Testing", "Rot Acc: %.1f, Rot Loss: %.2f!"%(100*r_acc.item(), r_loss.item()))

            np.save(f"part_idx_to_canon_rot_diff_zz_{self.local_rank}.npy", part_idx_to_canon_rot_diff_zz)
            np.save(f"part_idx_to_rot_diff_zz_{self.local_rank}.npy", part_idx_to_rot_diff_zz)
            np.save(f"part_idx_to_trans_diff_zz_{self.local_rank}.npy", part_idx_to_trans_diff_zz)
            np.save(f"part_idx_to_trans_diff_2_zz_{self.local_rank}.npy", part_idx_to_trans_diff_2_zz)

            # accs = np.array(accs, dtype=np.float32)
            avg_accs = []
            avg_accs_2 = []
            for i_iter in range(self.n_iters):
                avg_accs.append(sum(accs[i_iter]) / len(accs[i_iter]))
                avg_accs_2.append(sum(accs_2[i_iter]) / len(accs_2[i_iter]))
            avg_R_dist = sum(avg_R_dists) / float(len(avg_R_dists))
            if self.local_rank == 0:
                log_str = ""
                for i_iter in range(self.n_iters):
                    log_str += f" Avg_Acc_{i_iter}: %.2f" % (100 * avg_accs[i_iter])
                    log_str += f" Avg_Acc_2_{i_iter}: %.2f" % (100 * avg_accs_2[i_iter])
                # log_str += " avg_R_dist: %.4f" % float(avg_R_dist.item())
                log_str += " avg_R_dist: %.4f" % float(avg_R_dist)
                # self.logger.log('Testing', 'Average accuracy is %.2f!!!!'%(100*accs.mean()))
                self.logger.log('Testing', log_str)
                # self.test_accs.append(100*accs.mean())
                self.test_accs.append(100*avg_accs[i_iter]) # record average acc
                best_acc = np.array(self.test_accs).max() # get best test acc so far
                self.logger.log('Testing', 'Best accuracy so far is %.2f!!!!' % (best_acc)) # log best acc so far

                if self.pre_compute_delta >= 1:
                    log_str = ""
                    for i_p in part_idx_to_rot_diff:
                        cur_part_rot_diff = sum(part_idx_to_rot_diff[i_p]) / len(part_idx_to_rot_diff[i_p])
                        sorted_cur_part_rot_diff = sorted(part_idx_to_rot_diff[i_p])
                        medium_cur_part_rot_diff = sorted_cur_part_rot_diff[len(sorted_cur_part_rot_diff) // 2]
                        cur_part_canon_rot_diff = sum(part_idx_to_canon_rot_diff[i_p]) / len(part_idx_to_canon_rot_diff[i_p])
                        sorted_cur_part_canon_rot_diff = sorted(part_idx_to_canon_rot_diff[i_p])
                        medium_cur_part_canon_rot_diff = sorted_cur_part_canon_rot_diff[len(sorted_cur_part_canon_rot_diff) // 2]
                        # cur_slot_rot_diff = sum(slot_idx_to_rot_diff[i_p]) / len(slot_idx_to_rot_diff[i_p])
                        # cur_slot_canon_rot_diff = sum(slot_idx_to_canon_rot_diff[i_p]) / len(
                        #     slot_idx_to_canon_rot_diff[i_p])
                        cur_part_trans_diff = sum(part_idx_to_trans_diff[i_p]) / len(part_idx_to_trans_diff[i_p])
                        sorted_cur_part_trans_diff = sorted(part_idx_to_trans_diff[i_p])
                        medium_cur_part_trans_diff = sorted_cur_part_trans_diff[len(sorted_cur_part_trans_diff) // 2]
                        cur_part_trans_diff_2 = sum(part_idx_to_trans_diff_2[i_p]) / len(part_idx_to_trans_diff_2[i_p])
                        sorted_cur_part_trans_diff_2 = sorted(part_idx_to_trans_diff_2[i_p])
                        medium_cur_part_trans_diff_2 = sorted_cur_part_trans_diff_2[len(sorted_cur_part_trans_diff_2) // 2]
                        cur_part_pred_pred_diff = sum(part_idx_to_pred_posed_posed_diff[i_p]) / len(part_idx_to_pred_posed_posed_diff[i_p])
                        log_str += f"part idx: {i_p}, rot_diff: {cur_part_rot_diff}/{medium_cur_part_rot_diff}, canon_rot_diff: {cur_part_canon_rot_diff}/{medium_cur_part_canon_rot_diff}, posed_posed_diff: {cur_part_pred_pred_diff}, trans_diff: {cur_part_trans_diff}/{medium_cur_part_trans_diff}, trans_diff_2: {cur_part_trans_diff_2}/{medium_cur_part_trans_diff_2}\n"
                    avg_part_rel_rot_diff = sum(part_rel_rot_diff) / len(part_rel_rot_diff)
                    # log_str
                    log_str += f"part_rel_rot_diff: {avg_part_rel_rot_diff}\n"
                    for cur_part_pair in part_pair_to_part_rel_rot_diff:
                        curr_rot_diffs = part_pair_to_part_rel_rot_diff[cur_part_pair]
                        curr_rot_diffs_delta = part_pair_to_part_rel_rot_delta_diff[cur_part_pair]
                        avg_curr_rot_diff = sum(curr_rot_diffs) / len(curr_rot_diffs)
                        avg_curr_rot_diff_delta = sum(curr_rot_diffs_delta) / len(curr_rot_diffs_delta)
                        log_str += f"part pair: {cur_part_pair}, rot diff: {avg_curr_rot_diff}, rot_diff_delta: {avg_curr_rot_diff_delta}\n"
                    # trans diff is not accurate...
                    self.logger.log('Testing', log_str)

                    log_str = ""
                    for i_s in slot_idx_to_rot_diff:
                        cur_slot_rot_diff = sum(slot_idx_to_rot_diff[i_s]) / len(slot_idx_to_rot_diff[i_s])
                        sorted_cur_slot_rot_diff = sorted(slot_idx_to_rot_diff[i_s])
                        medium_cur_slot_rot_diff = sorted_cur_slot_rot_diff[len(sorted_cur_slot_rot_diff) // 2]
                        cur_slot_canon_rot_diff = sum(slot_idx_to_canon_rot_diff[i_s]) / len(
                            slot_idx_to_canon_rot_diff[i_s])
                        sorted_cur_slot_canon_rot_diff = sorted(slot_idx_to_canon_rot_diff[i_s])
                        medium_cur_slot_canon_rot_diff = sorted_cur_slot_canon_rot_diff[len(sorted_cur_slot_canon_rot_diff) // 2]
                        log_str += f"slot idx: {i_s}, slot_rot_diff: {cur_slot_rot_diff}, canon_slot_rot_diff: {cur_slot_canon_rot_diff}/{medium_cur_slot_canon_rot_diff}\n"
                    self.logger.log('Testing', log_str)

                    for gt_part_idx in part_idx_to_pred_posed_canon_diff:
                        # part_idx_to_pred_posed_canon_diff[gt_part_idx] = np.concatenate(part_idx_to_pred_posed_canon_diff[gt_part_idx], axis=0)
                        part_idx_to_pred_posed_canon_diff[gt_part_idx] = torch.cat(part_idx_to_pred_posed_canon_diff[gt_part_idx], dim=0).detach().cpu().numpy()
                    np.save(f"{self.shape_type}_part_idx_to_pred_posed_canon_diff.npy", part_idx_to_pred_posed_canon_diff)

            # self.logger.log("Testing", 'Here to peek at the lmc') # we should infer pose information?
            # self.logger.log("Testing", str(lmc))
            # import ipdb; ipdb.set_trace()
            # n = 1
            # mAP = modelnet_retrieval_mAP(all_feats,all_labels,n)
            # self.logger.log('Testing', 'Mean average precision at %d is %f!!!!'%(n, mAP))

        # self.model.module.train()
        # self.metric.train()

    def eval(self):
        ''' Need '''
        # Use one gpu for eval
        # evaluate test dataset
        self.logger.log('Testing','Evaluating test set!')
        self.model.module.eval()
        self.glb_stage_model.module.eval()
        # self.metric.eval()

        ''' Set queeue lenght to full length... '''
        # self.model.module.queue_len = self.model.module.queue_tot_len

        if self.pre_compute_delta == 1:
            # self.model.module.gt_oracle_seg = 1
            # better evaluate it in a single card?
            #
            # set_dr = []
            # set_dt = []
            shape_type_to_n_parts = {"eyeglasses": 3, "oven": 2, "washing_machine": 2, "laptop": 2, "drawer": 4, "safe": 2}
            n_parts = shape_type_to_n_parts[self.shape_type]
            set_dr = {}; set_dt = {}
            delta_rs, delta_ts = {}, {}
            delta_r_scores, delta_t_scores = {}, {}
            # remember rotation and translation
            # delta_r
            # part idx to others
            for i_p in range(n_parts + 2):
                set_dr[i_p] = []
                set_dt[i_p] = []
            with torch.no_grad():
                for it, data in enumerate(self.dataset):
                    #
                    # in_tensors = data['pc'].cuda(non_blocking=True)
                    # bdim = in_tensors.shape[0]
                    # in_label = data['label'].cuda(non_blocking=True)  # .reshape(-1)
                    # in_pose = data['pose'].cuda(non_blocking=True)  # if self.opt.debug_mode == 'knownatt' else None
                    # in_pose_segs = data['pose_segs'].cuda(non_blocking=True)
                    # ori_pc = data['ori_pc'].cuda(non_blocking=True)
                    # canon_pc = data['canon_pc'].cuda(non_blocking=True)
                    #
                    # if self.est_normals == 1:
                    #     cur_normals = data['cur_normals'].cuda(non_blocking=True)
                    #     cur_canon_normals = data['cur_canon_normals'].cuda(non_blocking=True)
                    # else:
                    #     cur_normals = None
                    #     cur_canon_normals = None
                    #
                    # # original point cloud
                    # oorr_pc = data['oorr_pc']
                    # # point labels
                    # oorr_label = data['oorr_label']
                    # be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
                    # minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)
                    #
                    # # print("intensor_size:", in_tensors.size())
                    #
                    # data_idxes = data['idx'].detach().cpu().numpy().tolist()
                    # # a complete or a
                    # data_idxes = [str(ii) for ii in data_idxes]
                    #
                    # bz = in_tensors.size(0)
                    # N = in_tensors.size(2)
                    #
                    # assert bz == 1
                    #
                    # # get one-hot label
                    # label = torch.eye(self.opt.nmasks)[in_label].cuda()
                    #
                    # with torch.no_grad():
                    #     glb_recon_loss = self.glb_stage_model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label,
                    #                                           pose_segs=in_pose_segs, canon_pc=canon_pc,
                    #                                           normals=cur_normals, canon_normals=cur_canon_normals)
                    # in_tensors = self.glb_stage_model.module.inv_trans_ori_pts if self._use_multi_gpu else self.glb_stage_model.inv_trans_ori_pts
                    # glb_R = self.glb_stage_model.module.glb_R if self._use_multi_gpu else self.glb_stage_model.glb_R
                    # glb_T = self.glb_stage_model.module.glb_T if self._use_multi_gpu else self.glb_stage_model.glb_T
                    #
                    # # oorr_pc = torch.matmul(safe_transpose(glb_R, -1, -2), oorr_pc - glb_T.unsqueeze(-1))
                    #
                    # loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                    #                   canon_pc=canon_pc,
                    #                   normals=cur_normals, canon_normals=cur_canon_normals)
                    #
                    # # # get loss by forwarding data through the model
                    # # loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                    # #                   canon_pc=canon_pc)
                    #
                    # print(f"loss: {loss}")
                    #
                    # ''' NOTE: We use GT Seg!!! '''
                    # # pred_R: bz x n_s x 3 x 3
                    # # selected predicted rotations for each slot
                    # pred_R = self.model.module.pred_R
                    # # pred_T: bz x n_s x 3
                    # # pred_T = self.model.module.ori_pred_T
                    # # so what does ori_pred_T represent
                    # # selected predicted translations for each slot
                    # pred_T = self.model.module.pred_T
                    # # pred_T = self.model.module.pred_T
                    # n_s = pred_T.size(1)
                    #
                    # # Get predicted attention
                    # # bz x n_s x N
                    # pred_attn_ori = self.model.module.attn_saved if self.n_iters == 1 else self.model.module.attn_saved_1
                    # # pred_labels: bz x N
                    # pred_labels = torch.argmax(pred_attn_ori, dim=1).long()
                    # # pred_hard_one_hot_labels: bz x N x n_s --> bz x n_s x N
                    # # pred_hard_one_hot_labels = torch.eye(pred_attn_ori.size(1), dtype=torch.float32).cuda()[pred_labels]
                    # # pred_hard_one_hot_labels = torch.transpose(pred_hard_one_hot_labels.contiguous(), -1, -2)
                    #
                    # # boundary_pts = [np.min(sampled_pcts, axis=0), np.max(sampled_pcts, axis=0)]
                    # # center_pt = (boundary_pts[0] + boundary_pts[1]) / 2
                    # # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
                    # # for
                    # # in_tensors: bz x 3 x N --> bz x N x 3 --> bz x n_s x N x 3
                    # expand_xyz = safe_transpose(in_tensors, -1, -2).unsqueeze(1).contiguous().repeat(1, self.num_slots, 1, 1)
                    #
                    # pred_seg_to_pts_idxes = {} # predicted seg label to point indexes
                    # for i_pts in range(pred_labels.size(1)):
                    #     cur_pts_pred_label = int(pred_labels[0, i_pts].item())
                    #     if cur_pts_pred_label not in pred_seg_to_pts_idxes:
                    #         pred_seg_to_pts_idxes[cur_pts_pred_label] = [i_pts]
                    #     else:
                    #         pred_seg_to_pts_idxes[cur_pts_pred_label].append(i_pts)

                    in_tensors = data['pc'].cuda(non_blocking=True) # input tensor
                    bdim = in_tensors.shape[0]
                    in_label = data['label'].cuda(non_blocking=True)  # .reshape(-1)
                    # per-point pose
                    in_pose = data['pose'].cuda(non_blocking=True)  # if self.opt.debug_mode == 'knownatt' else None
                    # per-part pose
                    in_pose_segs = data['pose_segs'].cuda(non_blocking=True)
                    ori_pc = data['ori_pc'].cuda(non_blocking=True)
                    canon_pc = data['canon_pc'].cuda(non_blocking=True)
                    part_axis = data['part_axis'].cuda(non_blocking=True)
                    if self.est_normals == 1:
                        cur_normals = data['cur_normals'].cuda(non_blocking=True)
                        cur_canon_normals = data['cur_canon_normals'].cuda(non_blocking=True)
                    else:
                        cur_normals = None
                        cur_canon_normals = None
                    # part_state_rots = data['part_state_rots'].cuda(non_blocking=True)
                    # part_ref_rots = data['part_ref_rots'].cuda(non_blocking=True)  # [0]

                    data_idxes = data['idx'].detach().cpu().numpy().tolist()
                    data_idxes = [str(ii) for ii in data_idxes]

                    bz = in_tensors.size(0)
                    N = in_tensors.size(2)

                    label = torch.eye(self.opt.nmasks)[in_label].cuda(non_blocking=True)
                    # label, gt_conf = self.get_gt_conf(label)


                    # oorr_label = torch.eye(self.model.module.num_slots)[oorr_label]  # .cuda(non_blocking=True)
                    # oorr_label, oorr_gt_conf = self.get_gt_conf(oorr_label)

                    cur_accs = []
                    cur_accs_2 = []

                    if self.global_rot == 1 and self.glb_resume_path is not None:
                        # then we can use global alignment module
                        with torch.no_grad():
                            glb_recon_loss = self.glb_stage_model(safe_transpose(canon_pc, 1, 2), in_pose,
                                                                  ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                                                                  canon_pc=canon_pc, normals=cur_normals,
                                                                  canon_normals=cur_canon_normals)
                        in_tensors_canon = self.glb_stage_model.module.inv_trans_ori_pts if self._use_multi_gpu else self.glb_stage_model.inv_trans_ori_pts
                        glb_R_canon = self.glb_stage_model.module.glb_R if self._use_multi_gpu else self.glb_stage_model.glb_R
                        glb_T_canon = self.glb_stage_model.module.glb_T if self._use_multi_gpu else self.glb_stage_model.glb_T
                        # part_axis = torch.matmul(safe_transpose(glb_R, -1, -2).unsqueeze(1),
                        #                          part_axis.unsqueeze(-1)).squeeze(-1)

                    else:
                        glb_R_canon = torch.eye(3, dtype=torch.float32).cuda().unsqueeze(0).contiguous().repeat(bz, 1,
                                                                                                                1).contiguous()
                        glb_T_canon = torch.zeros((bz, 3), dtype=torch.float32).cuda()
                        in_tensors_canon = safe_transpose(canon_pc, 1, 2)

                    loss = self.model(in_tensors_canon, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                                      canon_pc=canon_pc,
                                      normals=cur_normals, canon_normals=cur_canon_normals)

                    label = torch.eye(self.opt.nmasks)[in_label].cuda(non_blocking=True)
                    label, gt_conf = self.get_gt_conf(label) # get labels

                    canon_out_pred_R_np = self.model.module.out_feats['pred_R_slots']
                    canon_out_pred_R = torch.from_numpy(canon_out_pred_R_np).float().cuda()
                    canon_out_pred_T = self.model.module.pred_T
                    # canon_out_pred_T = self.model.module.ori_pred_T

                    canon_out_pred_R = torch.matmul(glb_R_canon.unsqueeze(1), canon_out_pred_R)
                    ''' if no glb '''
                    canon_out_pred_T = torch.matmul(glb_R_canon.unsqueeze(1), canon_out_pred_T.unsqueeze(-1)).squeeze(
                        -1) + glb_T_canon.unsqueeze(1)

                    ''' If we use global rotation and translation directly... '''
                    # canon_out_pred_R = glb_R_canon.unsqueeze(1).contiguous().repeat(1, self.num_slots, 1, 1)
                    # canon_out_pred_T = glb_T_canon.unsqueeze(1).contiguous().repeat(1, self.num_slots, 1)
                    ''' If we use global rotation and translation directly... '''

                    cur_pred = self.model.module.attn_iter_1 if self.n_iters == 2 else self.model.module.attn_iter_0
                    # todo: add canon pts's labels
                    iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)
                    cur_accs.append(iou_value.mean())

                    # canonical out pred R --> slots's rotations
                    # canon_out_pred_R = batched_index_select(values=canon_out_pred_R, indices=matching_idx_pred.long(),
                    #                                         dim=1)
                    # canon_out_pred_T = batched_index_select(values=canon_out_pred_T, indices=matching_idx_pred.long(),
                    #                                         dim=1)


                    ''' Get predicted attention weights for each point '''
                    canon_pred_attn_ori = self.model.module.attn_saved if self.n_iters == 1 else self.model.module.attn_saved_1
                    # pred_labels: bz x N
                    # canon labels
                    canon_pred_labels = torch.argmax(canon_pred_attn_ori, dim=1).long()

                    canon_seg_label_to_pts_idxes = {}
                    for i_pts in range(canon_pred_labels.size(1)):
                        cur_pts_label = int(canon_pred_labels[0, i_pts].item())
                        if cur_pts_label not in canon_seg_label_to_pts_idxes:
                            canon_seg_label_to_pts_idxes[cur_pts_label] = [i_pts]
                        else:
                            canon_seg_label_to_pts_idxes[cur_pts_label].append(i_pts)

                    for pred_label in canon_seg_label_to_pts_idxes:
                        print(f"current pred_label: {pred_label}")
                        cur_pred_pts_idxes = canon_seg_label_to_pts_idxes[pred_label]
                        cur_pred_pts_idxes = torch.tensor(cur_pred_pts_idxes, dtype=torch.long).cuda()
                        # Get predicted part_R and part_T: cur_part_R: 3 x 3; cur_part_T: 3
                        cur_part_R = canon_out_pred_R[0, pred_label, ...]
                        cur_part_T = canon_out_pred_T[0, pred_label, ...]
                        # set_dr[i_p].append(cur_part_R)
                        # set_dt[i_p].append(cur_part_T)
                        # 3 x n_parts --> current part predicted xyz
                        ''' Use original points for bounding box prediction '''
                        # cur_part_xyz = in_tensors[0, :, cur_pred_pts_idxes]
                        ''' Use original points for bounding box prediction '''
                        ''' Use canonical points for bounding box prediction '''
                        cur_part_xyz = safe_transpose(canon_pc, 1, 2)[0, :, cur_pred_pts_idxes]
                        ''' Use canonical points for bounding box prediction '''
                        # center_pt: 3 x 1
                        boundary_pts_minn, _ = torch.min(cur_part_xyz, dim=-1, keepdim=True)
                        boundary_pts_maxx, _ = torch.max(cur_part_xyz, dim=-1, keepdim=True)
                        # center_pt: 3 x 1
                        center_pt = (boundary_pts_minn + boundary_pts_maxx) / 2.
                        center_pt = center_pt.squeeze(-1)
                        # put points to the center of the bounding box of predicted part for category-level translation estimation
                        cur_part_T = cur_part_T - center_pt
                        set_dr[pred_label].append(cur_part_R.unsqueeze(0)) # rotation and predicted label...
                        set_dt[pred_label].append(cur_part_T.unsqueeze(0))

                    # out_feats
                    out_feats = self.model.module.out_feats
                    idxes_str = ",".join(data_idxes)
                    feat_save_fn = os.path.join(self.model.module.log_fn, f"eval_tr_out_feats_{idxes_str}_rnk_{self.local_rank}.npy")
                    np.save(feat_save_fn, out_feats)

                # for i_p in range(n_parts):
                for i_p in range(self.num_slots):
                    if len(set_dr[i_p]) == 0:
                        delta_r = torch.eye(3, dtype=torch.float32).cuda()
                        delta_t = torch.zeros((3,), dtype=torch.float32).cuda()
                        r_score, t_score = 0., 0.
                        delta_rs[i_p] = delta_r
                        delta_ts[i_p] = delta_t
                        delta_r_scores[i_p] = r_score
                        delta_t_scores[i_p] = t_score
                        continue
                    # after cat: tot_bz x 3 x 3
                    # set_dr
                    set_dr[i_p] = torch.cat(set_dr[i_p], dim=0)
                    # after cat: tot_bz x 3
                    set_dt[i_p] = torch.cat(set_dt[i_p], dim=0)
                    # todo: add flip axis and chosen axis for parts in each category
                    # rotations for this slot
                    print(set_dr[i_p].size())
                    # ransac fit rotation from canonical posed ...
                    delta_r, r_score = ransac_fit_r(set_dr[i_p], chosen_axis=None, flip_axis=None)
                    # todo: delta_r is not used here?
                    delta_t, t_score = ransac_fit_t(set_dt[i_p], set_dr[i_p], delta_r.squeeze())
                    delta_rs[i_p] = delta_r
                    print(f"current delta r, shape: {delta_r.size()}, values: {delta_r}")
                    delta_ts[i_p] = delta_t
                    delta_r_scores[i_p] = r_score
                    delta_t_scores[i_p] = t_score
                    # print(f"category {self.shape_type}, part {i_p}, delta_r {delta_r}, delta_t {delta_t}, r_score {r_score}, t_score {t_score}")
                    print(f"category {self.shape_type}, part {i_p}, r_score {r_score}, t_score {t_score}")
        else:
            delta_rs = {}
            delta_ts = {}
            for i_s in range(self.num_slots):
                delta_rs[i_s] = torch.eye(3).cuda()
                delta_ts[i_s] = torch.zeros((3,)).cuda()

        ################## DEBUG ###############################
        # for module in self.model.modules():
        #     if isinstance(module, torch.nn.modules.BatchNorm1d):
        #         module.train()
        #     if isinstance(module, torch.nn.modules.BatchNorm2d):
        #         module.train()
        #     if isinstance(module, torch.nn.modules.BatchNorm3d):
        #         module.train()
            # if isinstance(module, torch.nn.Dropout):
            #     module.train()
        #####################################################

        with torch.no_grad():
            accs = [[] for i_iter in range(self.n_iters)]
            accs_2 = [[] for i_iter in range(self.n_iters)]
            # lmc = np.zeros([40,60], dtype=np.int32)
            eval_infos = {}

            all_labels = []
            all_feats = []
            avg_R_dists = []

            axis_angle_val = []
            avg_dist_pred_gt_offset = []

            avg_glb_ori_recon_dists = []

            part_idx_to_rot_diff = {}
            part_idx_to_canon_rot_diff = {}
            slot_idx_to_rot_diff = {}
            slot_idx_to_canon_rot_diff = {}
            part_idx_to_canon_rot_diff_zz = {}
            part_idx_to_rot_diff_zz = {}
            part_idx_to_trans_diff = {}
            part_idx_to_trans_diff_zz = {}
            part_idx_to_trans_diff_2 = {}
            part_idx_to_trans_diff_2_zz = {}
            part_rel_rot_diff = []
            part_pair_to_part_rel_rot_diff = {}
            part_pair_to_part_rel_rot_delta_diff = {}

            part_idx_to_pred_posed_canon_diff = {}
            part_idx_to_pred_posed_posed_diff = {}

            glb_recon_chamfer_l1 = []
            slot_recon_chamfer_l1 = []

            for it, data in enumerate(self.dataset_test):
                in_tensors = data['pc'].cuda(non_blocking=True)
                bdim = in_tensors.shape[0]
                in_label = data['label'].cuda(non_blocking=True) # .reshape(-1)
                # per-point pose
                in_pose = data['pose'].cuda(non_blocking=True)  # if self.opt.debug_mode == 'knownatt' else None
                # per-part pose
                in_pose_segs = data['pose_segs'].cuda(non_blocking=True)
                ori_pc = data['ori_pc'].cuda(non_blocking=True)
                canon_pc = data['canon_pc'].cuda(non_blocking=True) #
                part_axis = data['part_axis'].cuda(non_blocking=True)
                if self.shape_type != 'drawer':
                    part_pv_offset = data['part_pv_offset'].cuda(non_blocking=True)
                else:
                    part_pv_offset = torch.zeros((part_axis.size(0), part_axis.size(1)), dtype=torch.float32).cuda(
                        non_blocking=True)
                if self.est_normals == 1:
                    cur_normals = data['cur_normals'].cuda(non_blocking=True)
                    cur_canon_normals = data['cur_canon_normals'].cuda(non_blocking=True)
                else:
                    cur_normals = None
                    cur_canon_normals = None
                part_state_rots = data['part_state_rots'].cuda(non_blocking=True)
                part_ref_rots = data['part_ref_rots'].cuda(non_blocking=True) # [0]
                # part reference translation
                if 'part_ref_trans' not in data:
                    part_ref_trans = torch.zeros((in_tensors.size(0), in_pose_segs.size(1), 3), dtype=torch.float).cuda()
                else:
                    part_ref_trans = data['part_ref_trans'].cuda(non_blocking=True) # [0]

                # reference translation considering bounding box
                part_ref_trans_bbox = data['part_ref_trans_bbox'].cuda(non_blocking=True)
                part_state_trans_bbox = data['part_state_trans_bbox'].cuda(non_blocking=True)

                oorr_pc = data['oorr_pc']
                oorr_label = data['oorr_label']
                be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
                minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)

                #
                data_idxes = data['idx'].detach().cpu().numpy().tolist()
                data_idxes = [str(ii) for ii in data_idxes]

                bz = in_tensors.size(0)
                N = in_tensors.size(2)

                label = torch.eye(self.opt.nmasks)[in_label].cuda(non_blocking=True)
                # label, gt_conf = self.get_gt_conf(label)

                in_tensors_ori = in_tensors.clone()
                
                sv_dict = {}
                data_np = {
                    k: data[k].detach().cpu().numpy() for k in data
                }
                sv_dict['data_np'] = data_np

                if self.global_rot == 1 and self.glb_stage_model is not None:
                    with torch.no_grad():
                        glb_recon_loss = self.glb_stage_model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label,
                                                              pose_segs=in_pose_segs, canon_pc=canon_pc,
                                                              normals=cur_normals, canon_normals=cur_canon_normals)
                    in_tensors = self.glb_stage_model.module.inv_trans_ori_pts if self._use_multi_gpu else self.glb_stage_model.inv_trans_ori_pts
                    glb_R = self.glb_stage_model.module.glb_R if self._use_multi_gpu else self.glb_stage_model.glb_R
                    glb_T = self.glb_stage_model.module.glb_T if self._use_multi_gpu else self.glb_stage_model.glb_T
                    part_axis = torch.matmul(safe_transpose(glb_R, -1, -2).unsqueeze(1),
                                             part_axis.unsqueeze(-1)).squeeze(-1)

                    if 'partial' in self.dataset_type:
                        curr_glb_l1_recon = float(self.glb_stage_model.module.glb_ori_to_recon_dist.item())
                    else:
                        curr_glb_l1_recon = float(self.glb_stage_model.module.glb_recon_ori_dist.item())
                    glb_recon_chamfer_l1.append(curr_glb_l1_recon)
                else:
                    glb_R = torch.eye(3, dtype=torch.float32).cuda().unsqueeze(0).contiguous().repeat(bz, 1, 1).contiguous()
                    glb_T = torch.zeros((bz, 3), dtype=torch.float32).cuda()

                ### === add glb R and glb T === ###
                sv_dict['glb_R'] = glb_R.detach().cpu().numpy()
                sv_dict['glb_T'] = glb_T.detach().cpu().numpy()
                ### === add glb R and glb T === ###

                # oorr_pc = torch.matmul(safe_transpose(glb_R, -1, -2), oorr_pc - glb_T.unsqueeze(-1))

                loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                                  canon_pc=canon_pc,
                                  normals=cur_normals, canon_normals=cur_canon_normals)

                label = torch.eye(self.opt.nmasks)[in_label].cuda(non_blocking=True)
                label, gt_conf = self.get_gt_conf(label)

                curr_slot_l1_recon = float(self.model.module.ori_to_recon.item()) if 'partial' in self.dataset_type else float(self.model.module.glb_recon_ori_dist.item())
                slot_recon_chamfer_l1.append(curr_slot_l1_recon)

                # loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc)

                out_feats = self.model.module.out_feats
                # out_feats_all_iters = self.model.module.out_feats_all_iters

                # Get model's output features and register some other features in the feature dictionary
                out_feats["part_state_rots"] = part_state_rots.detach().cpu().numpy()
                out_feats["part_ref_rots"] = part_ref_rots.detach().cpu().numpy()
                out_feats["part_ref_trans"] = part_ref_trans.detach().cpu().numpy()

                # out_feats_all_iters = self.model.module.out_feats_all_iters

                # Need not to further transpose predictions
                # # transform `pred` to prediction probability
                # pred = torch.clamp(pred, min=-20, max=20)
                # pred = torch.softmax(pred, dim=-1)
                # bz x npred-class x N

                # pred = torch.zeros((bz, 200, N), dtype=torch.float32).cuda(non_blocking=True)

                # loss, pred = self.model(in_tensors, in_pose)
                #
                # # # transform `pred` to prediction probability
                # # pred = torch.clamp(pred, min=-20, max=20)
                # # pred = torch.softmax(pred, dim=-1)
                # pred = pred.contiguous().transpose(1, 2).contiguous()
                #
                # if pred.size(1) < 200:
                #     pred = torch.cat(
                #         [pred, torch.zeros((bz, 200 - pred.size(1), N), dtype=torch.float32).cuda()], dim=1
                #     )

                # from in_label to label

                # label = torch.eye(self.model.module.num_slots)[in_label].cuda(non_blocking=True)
                # label, gt_conf = self.get_gt_conf(label)

                oorr_label = torch.eye(self.model.module.num_slots)[oorr_label] # .cuda(non_blocking=True)
                oorr_label, oorr_gt_conf = self.get_gt_conf(oorr_label)

                # pred_part_axis: bz x 3; pred_axis: bz x n_mov_parts x 3
                # pred_part_axis = self.model.module.defined_axises if self._use_multi_gpu else self.model.defined_axises
                # pred_part_axis = self.model.module.real_defined_axises if self._use_multi_gpu else self.model.real_defined_axises # real defined axis...
                # part_axis = part_axis / torch.norm(part_axis, dim=-1, keepdim=True, p=2)

                # pred_part_axis: bz x 3; pred_axis: bz x n_mov_parts x 3
                # pred_part_axis = self.model.module.defined_axises if self._use_multi_gpu else self.model.defined_axises
                pred_part_axis = self.model.module.real_defined_axises if self._use_multi_gpu else self.model.real_defined_axises
                part_axis = part_axis / torch.norm(part_axis, dim=-1, keepdim=True, p=2)
                if self.shape_type != 'drawer':
                    pred_part_pv_point_offset = self.model.module.offset_pivot_points if self._use_multi_gpu else self.model.offset_pivot_points
                    dist_pred_gt_offset = torch.abs(pred_part_pv_point_offset - part_pv_offset).mean().item()
                else:
                    dist_pred_gt_offset = 0.0
                # dist_pred_gt_offset = torch.abs(pred_part_pv_point_offset - part_pv_offset).mean().item()
                avg_dist_pred_gt_offset.append(dist_pred_gt_offset)

                # dot_prod: bz x n_mov_parts
                dot_prod = torch.abs(torch.sum(pred_part_axis.unsqueeze(1) * part_axis, dim=-1))
                mean_dot_prod_val = float(dot_prod.mean().item()) # mean dot prod val...
                print(f"dot_prod_axis_pred: {mean_dot_prod_val}, {math.acos(min(mean_dot_prod_val, 1.0))}, {math.acos(min(mean_dot_prod_val, 1.0)) / np.pi}")
                mean_angle = math.acos(min(mean_dot_prod_val, 1.0)) / np.pi * 180.0
                axis_angle_val.append(mean_angle)
                
                ### ==== add pred_axis ==== ###
                sv_dict['pred_part_axis'] = pred_part_axis.detach().cpu().numpy()
                ### ==== add pred_axis ==== ###


                cur_accs = []
                cur_accs_2 = []

                for i_iter in range(self.n_iters):
                    if i_iter == 0:
                        cur_pred = self.model.module.attn_iter_0

                        if self.shape_type == 'drawer':
                            hard_labels = torch.argmax(cur_pred, dim=1)  # bz x N
                            pred_label_to_real_label = {0: 0, 1: 2, 2: 1, 3: 3, 4: 3, 5: 3}
                            for i_bz in range(hard_labels.size(0)):
                                for i_pts in range(hard_labels.size(1)):
                                    cur_bz_cur_pts_label = int(hard_labels[i_bz, i_pts].item())
                                    reindexed_pts_label = pred_label_to_real_label[cur_bz_cur_pts_label]
                                    hard_labels[i_bz, i_pts] = reindexed_pts_label
                            cur_pred = torch.eye(self.model.module.num_slots)[hard_labels.long()].cuda(
                                non_blocking=True)
                            cur_pred = safe_transpose(cur_pred, 1, 2)



                        # iou_value, matching_idx_gt, matching_idx_pred = iou(curr_attn, gt_x=label, gt_conf=gt_conf)

                        # iou_loss = -iou_value.mean() * 100
                        # iou_loss = -iou_value.mean()  # * 100 # attn_saved... attn:



                        iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)

                        cur_accs.append(iou_value.mean())
                        # all_pred: bz x NN x nmasks
                        all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2),
                                                        indices=minn_idx, dim=1)
                        iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2),
                                                                                  gt_x=oorr_label, gt_conf=oorr_gt_conf)

                        cur_accs_2.append(iou_value_2.mean())
                    elif i_iter == 1:
                        cur_pred = self.model.module.attn_iter_1

                        if self.shape_type == 'drawer':
                            hard_labels = torch.argmax(cur_pred, dim=1)  # bz x N
                            pred_label_to_real_label = {0: 0, 1: 2, 2: 1, 3: 3, 4: 3, 5: 3}
                            for i_bz in range(hard_labels.size(0)):
                                for i_pts in range(hard_labels.size(1)):
                                    cur_bz_cur_pts_label = int(hard_labels[i_bz, i_pts].item())
                                    reindexed_pts_label = pred_label_to_real_label[cur_bz_cur_pts_label]
                                    hard_labels[i_bz, i_pts] = reindexed_pts_label
                            cur_pred = torch.eye(self.model.module.num_slots)[hard_labels.long()].cuda(
                                non_blocking=True)
                            cur_pred = safe_transpose(cur_pred, 1, 2)




                        # iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)
                        #
                        # # iou_loss = -iou_value.mean() * 100
                        # iou_loss = -iou_value.mean()  # * 100 # attn_saved... attn:




                        iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)
                        cur_accs.append(iou_value.mean())
                        # all_pred: bz x NN x nmasks
                        all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2),
                                                        indices=minn_idx, dim=1)
                        iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2),
                                                                                  gt_x=oorr_label, gt_conf=oorr_gt_conf)
                        cur_accs_2.append(iou_value_2.mean())

                    elif i_iter == 2:
                        cur_pred = self.model.module.attn_iter_2
                        iou_value, matching_idx_gt, matching_idx_pred = iou(cur_pred, gt_x=label, gt_conf=gt_conf)
                        cur_accs.append(iou_value.mean())
                        # all_pred: bz x NN x nmasks
                        all_pred = batched_index_select(values=safe_transpose(cur_pred.detach().cpu(), 1, 2),
                                                        indices=minn_idx, dim=1)
                        iou_value_2, matching_idx_gt_2, matching_idx_pred_2 = iou(safe_transpose(all_pred, 1, 2),
                                                                                  gt_x=oorr_label, gt_conf=oorr_gt_conf)
                        cur_accs_2.append(iou_value_2.mean())

                # iou_value, _, _ = iou(pred, gt_x=label, gt_conf=gt_conf)

                # loss = -iou_value.mean()

                # if self.attention_model:
                #     in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
                #     loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000)
                #     attention = F.softmax(feat,1)
                #
                #     if self.opt.train_loss.attention_loss_type == 'no_cls':
                #         acc = r_acc
                #         loss = r_loss
                #
                #     # max_id = attention.max(-1)[1].detach().cpu().numpy()
                #     # labels = data['label'].cpu().numpy().reshape(-1)
                #     # for i in range(max_id.shape[0]):
                #     #     lmc[labels[i], max_id[i]] += 1
                # else:
                # cls_loss, acc = self.metric(pred, in_label)
                # loss = cls_loss
                #

                # Get predicted attention
                # bz x n_s x N
                pred_attn_ori = self.model.module.attn_saved if self.n_iters == 1 else self.model.module.attn_saved_1
                # pred_labels: bz x N; From predicted attention to predicted per-point label
                pred_labels = torch.argmax(pred_attn_ori, dim=1).long()

                seg_label_to_pts_idxes = {}
                for i_pts in range(pred_labels.size(1)):
                    cur_pts_label = int(pred_labels[0, i_pts].item())
                    if cur_pts_label not in seg_label_to_pts_idxes:
                        seg_label_to_pts_idxes[cur_pts_label] = [i_pts]
                    else:
                        seg_label_to_pts_idxes[cur_pts_label].append(i_pts)


                out_pred_R_np = self.model.module.out_feats['pred_R_slots'] #
                out_pred_R = torch.from_numpy(out_pred_R_np).float().cuda()
                out_pred_T = self.model.module.pred_T
                
                ### ==== add pred_axis ==== ###
                sv_dict['out_pred_R'] = out_pred_R.detach().cpu().numpy()
                sv_dict['out_pred_T'] = out_pred_T.detach().cpu().numpy()
                sv_dict['pred_labels'] = pred_labels.detach().cpu().numpy()
                ### ==== add pred_axis ==== ###

                # glb_R: bz x 3 x 3
                out_pred_R = torch.matmul(glb_R.unsqueeze(1), out_pred_R)
                ''' If no glb '''
                out_pred_T = torch.matmul(glb_R.unsqueeze(1), out_pred_T.unsqueeze(-1)).squeeze(-1) + glb_T.unsqueeze(1)
                
                

                ''' If we use global rotation and translation directly... '''
                # out_pred_R = glb_R.unsqueeze(1).contiguous().repeat(1, self.num_slots, 1, 1)
                # out_pred_T = glb_T.unsqueeze(1).contiguous().repeat(1, self.num_slots, 1)
                ''' If we use global rotation and translation directly... '''

                # out_pred_T = self.model.module.ori_pred_T
                # gt pose
                gt_pose = data['pose_segs']
                gt_R = gt_pose[:, :, :3, :3].cuda()

                ''' Set matching idx manually... '''
                # if self.shape_type == 'drawer':
                #     matching_idx_gt = torch.tensor([0, 1, 2, 3], dtype=torch.long).cuda().unsqueeze(0).repeat(bz, 1)
                #     # matching_idx_pred = torch.tensor([2, 3, 0, 1], dtype=torch.long).cuda().unsqueeze(0).repeat(bz, 1)
                #     matching_idx_pred = torch.tensor([0, 2, 3, 1], dtype=torch.long).cuda().unsqueeze(0).repeat(bz, 1)
                #     # matching_idx_pred = torch.tensor([2, 3, 0, 1], dtype=torch.long).cuda().unsqueeze(0).repeat(bz, 1)
                ''' Set matching idx manually... '''

                # matching_idx_gt = torch.tensor([0, 1], dtype=torch.long).cuda().unsqueeze(0).repeat(bz, 1)
                # matching_idx_pred = torch.tensor([2, 1], dtype=torch.long).cuda().unsqueeze(0).repeat(bz, 1)

                # to matched slots..?
                out_pred_R = batched_index_select(values=out_pred_R, indices=matching_idx_pred.long(), dim=1)
                out_pred_T = batched_index_select(values=out_pred_T, indices=matching_idx_pred.long(), dim=1)
                # change to the ground-truth matched order; gt_R ---
                gt_R = batched_index_select(values=gt_R, indices=matching_idx_gt.long(), dim=1)

                avg_R_dist = calculate_res_relative_Rs(out_pred_R, gt_R)

                out_feats["matching_idx_pred"] = matching_idx_pred.detach().cpu().numpy()
                out_feats["matching_idx_gt"] = matching_idx_gt.detach().cpu().numpy()

                torch.distributed.barrier()

                cur_new_accs = []
                cur_new_accs_2 = []
                for acc in cur_accs:
                    new_acc = self.reduce_mean(acc, self.nprocs)
                    cur_new_accs.append(new_acc)
                for acc in cur_accs_2:
                    # new_acc = self.reduce_mean(acc, self.nprocs)
                    cur_new_accs_2.append(acc)
                cur_accs = cur_new_accs
                cur_accs_2 = cur_new_accs_2

                print(f"accs: {cur_accs}")

                # acc = iou_value.mean()
                all_labels.append(in_label.cpu().numpy())

                glb_ori_recon_dist = self.model.module.glb_recon_ori_dist
                glb_ori_recon_dist = self.reduce_mean(glb_ori_recon_dist, self.nprocs)

                avg_glb_ori_recon_dists.append(glb_ori_recon_dist.item())

                loss = self.reduce_mean(loss, self.nprocs)
                avg_R_dist = self.reduce_mean(avg_R_dist, self.nprocs)

                avg_R_dists.append(avg_R_dist.item())

                # acc = self.reduce_mean(acc, self.nprocs)
                # all_feats.append(feat.cpu().numpy())

                label = torch.eye(self.opt.nmasks)[in_label].cuda(non_blocking=True)
                # label, gt_conf = self.get_gt_conf(label)

                # Get global rotation and translation
                if self.global_rot == 1 and self.glb_resume_path is not None:
                    with torch.no_grad():
                        glb_recon_loss = self.glb_stage_model(safe_transpose(canon_pc, 1, 2), in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc, normals=cur_normals, canon_normals=cur_canon_normals)
                    in_tensors_canon = self.glb_stage_model.module.inv_trans_ori_pts if self._use_multi_gpu else self.glb_stage_model.inv_trans_ori_pts
                    glb_R_canon = self.glb_stage_model.module.glb_R if self._use_multi_gpu else self.glb_stage_model.glb_R
                    glb_T_canon = self.glb_stage_model.module.glb_T if self._use_multi_gpu else self.glb_stage_model.glb_T
                else:
                    glb_R_canon = torch.eye(3, dtype=torch.float32).cuda().unsqueeze(0).contiguous().repeat(bz, 1,
                                                                                                      1).contiguous()
                    glb_T_canon = torch.zeros((bz, 3), dtype=torch.float32).cuda()
                    in_tensors_canon = safe_transpose(canon_pc, 1, 2)

                # oorr_pc = torch.matmul(safe_transpose(glb_R, -1, -2), oorr_pc - glb_T.unsqueeze(-1))

                loss = self.model(in_tensors_canon, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                                  canon_pc=canon_pc,
                                  normals=cur_normals, canon_normals=cur_canon_normals)

                # loss = self.model(in_tensors, in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                #                   canon_pc=canon_pc,
                #                   normals=cur_normals, canon_normals=cur_canon_normals)

                label = torch.eye(self.opt.nmasks)[in_label].cuda(non_blocking=True)
                label, gt_conf = self.get_gt_conf(label)

                # loss_ = self.model(safe_transpose(canon_pc, 1, 2), in_pose, ori_pc=ori_pc, rlabel=label, pose_segs=in_pose_segs,
                #                    canon_pc=canon_pc)
                # out-pred-R

                canon_out_pred_R_np = self.model.module.out_feats['pred_R_slots']
                canon_out_pred_R = torch.from_numpy(canon_out_pred_R_np).float().cuda()
                canon_out_pred_T = self.model.module.pred_T
                # canon_out_pred_T = self.model.module.ori_pred_T

                canon_out_pred_R = torch.matmul(glb_R_canon.unsqueeze(1), canon_out_pred_R)
                ''' if no glb '''
                canon_out_pred_T = torch.matmul(glb_R_canon.unsqueeze(1), canon_out_pred_T.unsqueeze(-1)).squeeze(-1) + glb_T_canon.unsqueeze(1)

                ''' If we use global rotation and translation directly... '''
                # canon_out_pred_R = glb_R_canon.unsqueeze(1).contiguous().repeat(1, self.num_slots, 1, 1)
                # canon_out_pred_T = glb_T_canon.unsqueeze(1).contiguous().repeat(1, self.num_slots, 1)
                ''' If we use global rotation and translation directly... '''

                # canonical out pred R --> slots's rotations
                canon_out_pred_R = batched_index_select(values=canon_out_pred_R, indices=matching_idx_pred.long(), dim=1)
                canon_out_pred_T = batched_index_select(values=canon_out_pred_T, indices=matching_idx_pred.long(), dim=1)

                # canonical out prediction T
                # real_T = out_T - (out_R x canon_out_R^T x canon_out_T)
                # real pred T
                real_pred_T = out_pred_T - torch.matmul(torch.matmul(out_pred_R, safe_transpose(canon_out_pred_R, -1, -2)), canon_out_pred_T.unsqueeze(-1)).squeeze(-1)
                # real_pred_T

                ''' Get predicted attention weights for each point '''
                canon_pred_attn_ori = self.model.module.attn_saved if self.n_iters == 1 else self.model.module.attn_saved_1
                # pred_labels: bz x N
                # canon labels
                canon_pred_labels = torch.argmax(canon_pred_attn_ori, dim=1).long()

                canon_seg_label_to_pts_idxes = {}
                for i_pts in range(canon_pred_labels.size(1)):
                    cur_pts_label = int(canon_pred_labels[0, i_pts].item())
                    if cur_pts_label not in canon_seg_label_to_pts_idxes:
                        canon_seg_label_to_pts_idxes[cur_pts_label] = [i_pts]
                    else:
                        canon_seg_label_to_pts_idxes[cur_pts_label].append(i_pts)

                # if use pre compute delta
                if self.pre_compute_delta >= 1:
                    # pred_R: bz x n_s x 3
                    # pred_R = self.model.module.pred_R
                    # # pred_T: bz x n_s x 3
                    # # pred_T = self.model.module.pred_T
                    # pred_T = self.model.module.ori_pred_T
                    # pred_T = self.model.module.ori_pred_T

                    pred_R_parts, pred_T_parts = [], []

                    cur_iter_rot_diff = []
                    cur_iter_rot_diff_canon = []

                    # matching_idx_gt
                    for i_bz in range(bz):
                        try:
                            cur_bz_pred_Rs = []
                            cur_bz_delta_Rs = []
                            cur_bz_ref_Rs = []
                            cur_iter_cur_bz_rot_diff = {}
                            cur_iter_cur_bz_rot_diff_canon = {}
                            for it_p in range(matching_idx_gt.size(1)):
                                cur_gt_part_idx = int(matching_idx_gt[i_bz, it_p].item())
                                cur_pred_part_idx = int(matching_idx_pred[i_bz, it_p].item())
                                cur_bz_cur_match_pred_R = out_pred_R[i_bz, it_p, :, :]
                                cur_bz_cur_match_canon_pred_R = canon_out_pred_R[i_bz, it_p, :, :]
                                # cur_bz_cur_match_pred_R = cur_bz_cur_match_pred_R.contiguous().view(3, 3).contiguous()
                                cur_bz_cur_match_pred_T = out_pred_T[i_bz, it_p, :]
                                cur_bz_cur_match_canon_pred_T = canon_out_pred_T[i_bz, it_p, :]

                                cur_bz_cur_match_pts_idxes = torch.tensor(seg_label_to_pts_idxes[cur_pred_part_idx], dtype=torch.long).cuda()
                                cur_bz_cur_match_canon_pts_idxes = torch.tensor(canon_seg_label_to_pts_idxes[cur_pred_part_idx], dtype=torch.long).cuda()
                                # cur_bz_cur_match_pts_xyz = in_tensors[0, :, cur_bz_cur_match_pts_idxes]
                                cur_bz_cur_match_pts_xyz = in_tensors_ori[0, :, cur_bz_cur_match_pts_idxes]
                                cur_bz_cur_match_canon_pts_xyz = safe_transpose(canon_pc, -1, -2)[0, :, cur_bz_cur_match_canon_pts_idxes]

                                ''' Get boudnary coordinates for points in current predicted match '''
                                #
                                cur_bz_cur_match_pts_xyz_minn, _ = torch.min(cur_bz_cur_match_pts_xyz, dim=-1)
                                cur_bz_cur_match_pts_xyz_maxx, _ = torch.max(cur_bz_cur_match_pts_xyz, dim=-1)
                                center_pt = (cur_bz_cur_match_pts_xyz_minn + cur_bz_cur_match_pts_xyz_maxx) / 2.
                                ori_cur_bz_cur_match_pred_T_norm = torch.norm(cur_bz_cur_match_pred_T, dim=-1).mean().item()
                                cur_bz_cur_match_pred_T = cur_bz_cur_match_pred_T - center_pt

                                ''' Get boudnary coordinates for canonical points in current  predicted match '''
                                cur_bz_cur_match_canon_pts_xyz_minn, _ = torch.min(cur_bz_cur_match_canon_pts_xyz, dim=-1)
                                cur_bz_cur_match_canon_pts_xyz_maxx, _ = torch.max(cur_bz_cur_match_canon_pts_xyz, dim=-1)
                                center_pt = (cur_bz_cur_match_canon_pts_xyz_minn + cur_bz_cur_match_canon_pts_xyz_maxx) / 2.
                                cur_bz_cur_match_canon_pred_T = cur_bz_cur_match_canon_pred_T - center_pt

                                # pc1 = R1(pc) + T1; pc2 = R2(pc) + T2;
                                # pc = R1^{-1}(pc1 - T1)
                                # pc2 = R2(R1^{-1}pc1 - R1^{-1}T1) + T2
                                # pc2 = R2R1^{-1}pc1 - R2R1^{-1}T1 + T2
                                ''' Then, calculate real predicted translation vector ''' # pred T
                                #

                                cur_bz_cur_match_pred_R_for_trans = torch.matmul(safe_transpose(glb_R[i_bz], -1, -2), cur_bz_cur_match_pred_R)
                                cur_bz_cur_match_canon_pred_R_for_trans = torch.matmul(safe_transpose(glb_R_canon[i_bz], -1, -2), cur_bz_cur_match_canon_pred_R)

                                ''' Current match real predicted translation '''
                                cur_bz_cur_match_real_pred_T = cur_bz_cur_match_pred_T - torch.matmul(
                                    torch.matmul(cur_bz_cur_match_pred_R, safe_transpose(cur_bz_cur_match_canon_pred_R, -1, -2)), cur_bz_cur_match_canon_pred_T.unsqueeze(-1)).squeeze(-1)

                                ''' Current match real predicted for trans '''
                                # cur_bz_cur_match_real_pred_T = cur_bz_cur_match_pred_T - torch.matmul(
                                #     torch.matmul(cur_bz_cur_match_pred_R_for_trans,
                                #                  safe_transpose(cur_bz_cur_match_canon_pred_R_for_trans, -1, -2)),
                                #     cur_bz_cur_match_canon_pred_T.unsqueeze(-1)).squeeze(-1)

                                ####
                                # cur_bz_cur_match_real_pred_T = real_pred_T[i_bz, it_p, :]

                                # cur_bz_cur_match_delta_R = delta_rs[cur_gt_part_idx].contiguous().view(3, 3).contiguous()
                                # Get the predicted delta_R for the current part
                                cur_bz_cur_match_delta_R = delta_rs[cur_pred_part_idx].contiguous().view(3, 3).contiguous()
                                # cur_bz_cur_match_delta_T = delta_ts[cur_gt_part_idx]
                                cur_bz_cur_match_delta_T = delta_ts[cur_pred_part_idx]
                                # delta
                                cur_bz_cur_match_pred_rot = torch.matmul(cur_bz_cur_match_pred_R, cur_bz_cur_match_delta_R.contiguous().transpose(-1, -2).contiguous())
                                # cur_bz_cur_match_pred_canon_rot = torch.matmul(cur_bz_cur_match_pred_R, cur_bz_cur_match_delta_R.contiguous().transpose(0, 1).contiguous())
                                cur_bz_cur_match_pred_canon_rot = torch.matmul(cur_bz_cur_match_pred_R, cur_bz_cur_match_canon_pred_R.contiguous().transpose(-1, -2).contiguous())
                                cur_bz_cur_match_pred_pred_rot = torch.matmul(cur_bz_cur_match_pred_R, cur_bz_cur_match_pred_R.contiguous().transpose(-1, -2).contiguous())

                                cur_bz_cur_match_real_pred_T_delta = cur_bz_cur_match_pred_T - torch.matmul(
                                    torch.matmul(cur_bz_cur_match_pred_R,
                                                 safe_transpose(cur_bz_cur_match_delta_R, -1, -2)),
                                    cur_bz_cur_match_delta_T.unsqueeze(-1)).squeeze(-1)

                                # cur_bz_cur_match_gt_rot = in_pose_segs[i_bz, cur_gt_part_idx, :3, :3]
                                cur_bz_cur_match_gt_trans = in_pose_segs[i_bz, cur_gt_part_idx, :3, 3]
                                #
                                cur_bz_cur_match_gt_rot = torch.matmul(part_state_rots[i_bz, cur_gt_part_idx], safe_transpose(part_ref_rots[i_bz, cur_gt_part_idx], -1, -2))

                                # cur_bz_cur_match_gt_canon_trans = part_ref_trans[i_bz, cur_gt_part_idx]
                                # cur_bz_cur_match_gt_state_trans = in_pose_segs[i_bz, cur_gt_part_idx, :3, 3]

                                ''' Get state trans and canonical trans with bounding boxes centralized '''
                                cur_bz_cur_match_gt_canon_trans = part_ref_trans_bbox[i_bz, cur_gt_part_idx]
                                cur_bz_cur_match_gt_state_trans = part_state_trans_bbox[i_bz, cur_gt_part_idx]
                                ''' GT state trans and GT ref trans should alsot be modified '''

                                real_gt_T = cur_bz_cur_match_gt_state_trans - torch.matmul(
                                    torch.matmul(part_state_rots[i_bz, cur_gt_part_idx], safe_transpose(part_ref_rots[i_bz, cur_gt_part_idx], -1, -2)), cur_bz_cur_match_gt_canon_trans.unsqueeze(-1)).squeeze(-1)

                                if cur_gt_part_idx not in part_idx_to_pred_posed_canon_diff:
                                    part_idx_to_pred_posed_canon_diff[cur_gt_part_idx] = [cur_bz_cur_match_pred_canon_rot.unsqueeze(0)]
                                else:
                                    part_idx_to_pred_posed_canon_diff[cur_gt_part_idx].append(cur_bz_cur_match_pred_canon_rot.unsqueeze(0))

                                if cur_gt_part_idx == 0:
                                    cur_bz_cur_match_diff = rot_diff_degree(cur_bz_cur_match_pred_rot.unsqueeze(0),
                                                                            cur_bz_cur_match_gt_rot.unsqueeze(0)).item()

                                    # cur_bz_cur_match_canon_diff = rot_diff_degree(cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                    #                                         cur_bz_cur_match_canon_pred_R.unsqueeze(0))

                                    cur_bz_cur_match_canon_diff = rot_diff_degree(
                                        cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                        cur_bz_cur_match_gt_rot.unsqueeze(0)).item()
                                    cur_bz_cur_match_pred_pred_diff = rot_diff_degree(
                                        cur_bz_cur_match_pred_pred_rot.unsqueeze(0), cur_bz_cur_match_pred_pred_rot.unsqueeze(0)
                                    ).item()
                                else:
                                    # cur_bz_cur_match_diff = rot_diff_degree(cur_bz_cur_match_pred_rot.unsqueeze(0), cur_bz_cur_match_gt_rot.unsqueeze(0)) - 90
                                    # cur_bz_cur_match_canon_diff = rot_diff_degree(cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                    #                                         cur_bz_cur_match_canon_pred_R.unsqueeze(0)) - 90
                                    cur_bz_cur_match_diff = rot_diff_degree(cur_bz_cur_match_pred_rot.unsqueeze(0),
                                                                            cur_bz_cur_match_gt_rot.unsqueeze(0)).item()

                                    # cur_bz_cur_match_canon_diff = rot_diff_degree(cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                        # cur_bz_cur_match_canon_pred_R.unsqueeze(0))

                                    cur_bz_cur_match_canon_diff = rot_diff_degree(
                                        cur_bz_cur_match_pred_canon_rot.unsqueeze(0),
                                        cur_bz_cur_match_gt_rot.unsqueeze(0)).item()

                                    cur_bz_cur_match_pred_pred_diff = rot_diff_degree(
                                        cur_bz_cur_match_pred_pred_rot.unsqueeze(0), cur_bz_cur_match_pred_pred_rot.unsqueeze(0)
                                    ).item()

                                cur_iter_cur_bz_rot_diff[cur_gt_part_idx] = cur_bz_cur_match_diff
                                cur_iter_cur_bz_rot_diff_canon[cur_gt_part_idx] = cur_bz_cur_match_canon_diff

                                # get rotation difference
                                # cur_bz_cur_match_diff = abs(cur_bz_cur_match_diff)
                                # cur_bz_cur_match_diff = cur_bz_cur_match_diff
                                # part_idx_to_canon_rot_diff_zz

                                # cur_bz_cur_match_diff_t = torch.norm(
                                #     cur_bz_cur_match_pred_T - cur_bz_cur_match_delta_T - real_gt_T, dim=-1).mean().item()
                                cur_bz_cur_match_diff_t = torch.norm(cur_bz_cur_match_real_pred_T_delta - real_gt_T,
                                    dim=-1).mean().item()
                                # cur_bz_cur_match_diff_t_2 = torch.norm(cur_bz_cur_match_real_pred_T - cur_bz_cur_match_gt_trans, dim=-1).mean().item()
                                cur_bz_cur_match_diff_t_2 = torch.norm(cur_bz_cur_match_real_pred_T - real_gt_T,
                                                                       dim=-1).mean().item()
                                cur_bz_cur_match_real_pred_T_norm = torch.norm(cur_bz_cur_match_real_pred_T, dim=-1).mean().item()
                                real_gt_T_norm = torch.norm(real_gt_T, dim=-1).mean().item()
                                cur_bz_cur_match_pred_T_norm = torch.norm(cur_bz_cur_match_pred_T, dim=-1).mean().item()
                                cur_bz_cur_match_canon_pred_T_norm = torch.norm(cur_bz_cur_match_canon_pred_T, dim=-1).mean().item()

                                print(cur_bz_cur_match_diff, cur_bz_cur_match_canon_diff, cur_bz_cur_match_pred_pred_diff)
                                if not cur_gt_part_idx in part_idx_to_canon_rot_diff_zz:
                                    part_idx_to_canon_rot_diff_zz[cur_gt_part_idx] = [cur_bz_cur_match_canon_diff]
                                    part_idx_to_rot_diff_zz[cur_gt_part_idx] = [cur_bz_cur_match_diff]
                                    part_idx_to_trans_diff_zz[cur_gt_part_idx] = [cur_bz_cur_match_diff_t]
                                    part_idx_to_trans_diff_2_zz[cur_gt_part_idx] = [cur_bz_cur_match_diff_t_2]
                                else:
                                    part_idx_to_canon_rot_diff_zz[cur_gt_part_idx].append(cur_bz_cur_match_canon_diff)
                                    part_idx_to_rot_diff_zz[cur_gt_part_idx].append(cur_bz_cur_match_diff)
                                    part_idx_to_trans_diff_zz[cur_gt_part_idx].append(cur_bz_cur_match_diff_t)
                                    part_idx_to_trans_diff_2_zz[cur_gt_part_idx].append(cur_bz_cur_match_diff_t_2)
                                cur_bz_cur_match_diff = min(180. - cur_bz_cur_match_diff, cur_bz_cur_match_diff)
                                # cur_bz_cur_match_canon_diff = abs(cur_bz_cur_match_canon_diff)
                                # cur_bz_cur_match_canon_diff = cur_bz_cur_match_canon_diff

                                cur_bz_cur_match_canon_diff = min(180. - cur_bz_cur_match_canon_diff, cur_bz_cur_match_canon_diff)
                                # get translation difference
                                # try:
                                # cur_bz_cur_match_diff_t = torch.norm(cur_bz_cur_match_pred_T - cur_bz_cur_match_delta_T - cur_bz_cur_match_gt_trans, dim=-1).mean().item()

                                # except:

                                # print(f"real_pred_T_norm: {cur_bz_cur_match_real_pred_T_norm}, real_gt_T_norm: {real_gt_T_norm}, diff_norm: {cur_bz_cur_match_diff_t_2}, cur_bz_cur_match_pred_T_norm: {cur_bz_cur_match_pred_T_norm}, cur_bz_cur_match_canon_pred_T_norm: {cur_bz_cur_match_canon_pred_T_norm}, ori_cur_bz_cur_match_pred_T_norm: {ori_cur_bz_cur_match_pred_T_norm}")
                                print(f"delta_diff_norm: {cur_bz_cur_match_diff_t}, diff_norm: {cur_bz_cur_match_diff_t_2}")

                                if not cur_gt_part_idx in part_idx_to_rot_diff:
                                    part_idx_to_rot_diff[cur_gt_part_idx] = [cur_bz_cur_match_diff]
                                    part_idx_to_canon_rot_diff[cur_gt_part_idx] = [cur_bz_cur_match_canon_diff]
                                    part_idx_to_trans_diff[cur_gt_part_idx]  = [cur_bz_cur_match_diff_t]
                                    part_idx_to_trans_diff_2[cur_gt_part_idx] = [cur_bz_cur_match_diff_t_2]
                                    part_idx_to_pred_posed_posed_diff[cur_gt_part_idx] = [cur_bz_cur_match_pred_pred_diff]
                                else:
                                    part_idx_to_rot_diff[cur_gt_part_idx].append(cur_bz_cur_match_diff)
                                    part_idx_to_canon_rot_diff[cur_gt_part_idx].append(cur_bz_cur_match_canon_diff)
                                    part_idx_to_trans_diff[cur_gt_part_idx].append(cur_bz_cur_match_diff_t)
                                    part_idx_to_trans_diff_2[cur_gt_part_idx].append(cur_bz_cur_match_diff_t_2)
                                    part_idx_to_pred_posed_posed_diff[cur_gt_part_idx].append(cur_bz_cur_match_pred_pred_diff)

                                if cur_pred_part_idx not in slot_idx_to_rot_diff:
                                    slot_idx_to_rot_diff[cur_pred_part_idx] = [cur_bz_cur_match_diff]
                                    slot_idx_to_canon_rot_diff[cur_pred_part_idx] = [cur_bz_cur_match_canon_diff]
                                else:
                                    slot_idx_to_rot_diff[cur_pred_part_idx].append(cur_bz_cur_match_diff)
                                    slot_idx_to_canon_rot_diff[cur_pred_part_idx].append(cur_bz_cur_match_canon_diff)

                                cur_bz_pred_Rs.append(cur_bz_cur_match_pred_R)
                                cur_bz_delta_Rs.append(cur_bz_cur_match_delta_R)
                                cur_bz_ref_Rs.append(cur_bz_cur_match_pred_rot)

                            cur_iter_rot_diff.append(cur_iter_cur_bz_rot_diff)
                            cur_iter_rot_diff_canon.append(cur_iter_cur_bz_rot_diff_canon)

                            part_rel_R = torch.matmul(cur_bz_ref_Rs[0], safe_transpose(cur_bz_ref_Rs[1], 0, 1))
                            gt_part_rel_R = torch.matmul(part_ref_rots[i_bz, 0], safe_transpose(part_ref_rots[i_bz, 1], 0, 1))
                            cur_bz_part_rel_R_rot_diff = rot_diff_degree(part_rel_R.unsqueeze(0),
                                                                    gt_part_rel_R.unsqueeze(0)).item()

                            part_rel_rot_diff.append(cur_bz_part_rel_R_rot_diff)

                            # for it_p in range(matching_idx_gt.size(1)):
                            #     cur_gt_part_idx = int(matching_idx_gt[i_bz, it_p].item())

                            for ip_a in range(matching_idx_gt.size(1) - 1):
                                gt_part_idx_a = int(matching_idx_gt[i_bz, ip_a].item())
                                pred_part_idx_a = int(matching_idx_pred[i_bz, ip_a].item())
                                for ip_b in range(ip_a + 1, matching_idx_gt.size(1)):
                                    gt_part_idx_b = int(matching_idx_gt[i_bz, ip_b].item())
                                    pred_part_idx_b = int(matching_idx_pred[i_bz, ip_b].item())

                                    pred_R_a = out_pred_R[i_bz, ip_a, :, :]
                                    canon_pred_R_a = canon_out_pred_R[i_bz, ip_a, :, :]
                                    pred_R_b = out_pred_R[i_bz, ip_b, :, :]
                                    canon_pred_R_b = canon_out_pred_R[i_bz, ip_b, :, :]

                                    delta_R_a = delta_rs[pred_part_idx_a].contiguous().view(3, 3).contiguous()
                                    delta_R_b = delta_rs[pred_part_idx_b].contiguous().view(3, 3).contiguous()

                                    pred_R_a = torch.matmul(pred_R_a, safe_transpose(canon_pred_R_a, -1, -2))
                                    pred_R_b = torch.matmul(pred_R_b, safe_transpose(canon_pred_R_b, -1, -2))

                                    pred_R_a_delta = torch.matmul(pred_R_a, safe_transpose(delta_R_a, -1, -2))
                                    pred_R_b_delta = torch.matmul(pred_R_b, safe_transpose(delta_R_b, -1, -2))

                                    rel_rot_R = torch.matmul(pred_R_a, safe_transpose(pred_R_b, -1, -2))
                                    rel_rot_R_delta = torch.matmul(pred_R_a_delta, safe_transpose(pred_R_b_delta, -1, -2))

                                    gt_R_a = part_state_rots[i_bz, gt_part_idx_a]
                                    gt_R_b = part_state_rots[i_bz, gt_part_idx_b]

                                    gt_canon_R_a = part_ref_rots[i_bz, gt_part_idx_a]
                                    gt_canon_R_b = part_ref_rots[i_bz, gt_part_idx_b]

                                    gt_R_a = torch.matmul(gt_R_a, safe_transpose(gt_canon_R_a, -1, -2))
                                    gt_R_b = torch.matmul(gt_R_b, safe_transpose(gt_canon_R_b, -1, -2))

                                    # relative rotation between part a and part b
                                    gt_rel_rot_R = torch.matmul(gt_R_a, safe_transpose(gt_R_b, 0, 1))

                                    part_rel_gt_rot_diff = rot_diff_degree(
                                        rel_rot_R.unsqueeze(0),
                                        gt_rel_rot_R.unsqueeze(0))

                                    part_rel_delta_gt_rot_diff = rot_diff_degree(
                                        rel_rot_R_delta.unsqueeze(0),
                                        gt_rel_rot_R.unsqueeze(0)
                                    )

                                    part_rel_gt_rot_diff = min(part_rel_gt_rot_diff, 180. - part_rel_gt_rot_diff)
                                    part_rel_delta_gt_rot_diff = min(part_rel_delta_gt_rot_diff, 180. - part_rel_delta_gt_rot_diff)

                                    if gt_part_idx_a < gt_part_idx_b:
                                        cur_part_pari = (gt_part_idx_a, gt_part_idx_b)
                                    else:
                                        cur_part_pari = (gt_part_idx_b, gt_part_idx_a)
                                    # cur_part_pari = (gt_part_idx_a, gt_part_idx_b)
                                    # cur_part_pair_inv = (gt_part_idx_b, gt_part_idx_a)
                                    if cur_part_pari not in part_pair_to_part_rel_rot_diff:
                                        part_pair_to_part_rel_rot_diff[cur_part_pari] = [part_rel_gt_rot_diff]
                                        part_pair_to_part_rel_rot_delta_diff[cur_part_pari] = [part_rel_delta_gt_rot_diff]
                                    else:
                                        part_pair_to_part_rel_rot_diff[cur_part_pari].append(part_rel_gt_rot_diff)
                                        part_pair_to_part_rel_rot_delta_diff[cur_part_pari].append(part_rel_delta_gt_rot_diff)
                        except:
                            continue

                log_str = "Loss: %.2f" % loss.item()
                for i_iter in range(self.n_iters):
                    log_str += f" Acc_{i_iter}: %.2f" % (100 * cur_accs[i_iter].item())
                    log_str += f" Acc_2_{i_iter}: %.2f" % (100 * cur_accs_2[i_iter].item())
                log_str += f" avg_R_dist: %.4f"%(avg_R_dist.item())
                for i_iter in range(self.n_iters):
                    cur_acc_item = float(cur_accs[i_iter].detach().item())
                    cur_acc_2_item = float(cur_accs_2[i_iter].detach().item())
                    accs[i_iter].append(cur_acc_item)
                    accs_2[i_iter].append(cur_acc_2_item)

                canon_out_feats = self.model.module.out_feats
                # canon_out_feats_all_iters = self.model.module.out_feats_all_iters

                # self.save_predicted_infos(data_idxes, out_feats)
                # self.save_predicted_infos_all_iters(data_idxes, out_feats_all_iters)

                out_feats['rot_diff'] = cur_iter_rot_diff
                out_feats['rot_diff_canon'] = cur_iter_rot_diff_canon

                idxes_str = ",".join(data_idxes)
                feat_save_fn = os.path.join(self.model.module.log_fn,
                                            f"test_out_feats_{idxes_str}_rnk_{self.local_rank}.npy")
                np.save(feat_save_fn, out_feats)

                # out_feats_all_iters

                # idxes_str = ",".join(data_idxes)
                # all_iters_feat_save_fn = os.path.join(self.model.module.log_fn,
                #                             f"test_out_feats_{idxes_str}_rnk_{self.local_rank}_all_iters.npy")
                # np.save(all_iters_feat_save_fn, out_feats_all_iters)

                # feat_all_iters_save_fn = os.path.join(self.model.module.log_fn,
                #                                       f"test_out_feats_{idxes_str}_all_iters_rnk_{self.local_rank}.npy")
                # np.save(feat_all_iters_save_fn, out_feats_all_iters)

                # idxes_str = ",".join(data_idxes)
                feat_save_fn = os.path.join(self.model.module.log_fn,
                                            f"test_canon_out_feats_{idxes_str}_rnk_{self.local_rank}.npy")
                np.save(feat_save_fn, canon_out_feats)
                
                
                ##### ===== save evaluated data ===== #####
                os.makedirs(self.opt.equi_settings.eval_data_sv_dict_fn, exist_ok=True)
                cur_test_data_sv_dict_fn = os.path.join(self.opt.equi_settings.eval_data_sv_dict_fn, f"iter_{it}.npy")
                np.save(cur_test_data_sv_dict_fn, sv_dict)
                print(f"Evalation data of it {it} saved to {cur_test_data_sv_dict_fn}")
                ##### ===== save evaluated data ===== #####

                # feat_all_iters_save_fn = os.path.join(self.model.module.log_fn,
                #                                       f"test_canon_out_feats_{idxes_str}_all_iters_rnk_{self.local_rank}.npy")
                # np.save(feat_all_iters_save_fn, canon_out_feats_all_iters)

                # accs.append(cur_accs[-1].detach().cpu().numpy())

                # self.logger.log("Testing", "Accuracy: %.1f, Loss: %.2f!"%(100*acc.item(), loss.item()))
                # if self.attention_model:
                #     self.logger.log("Testing", "Rot Acc: %.1f, Rot Loss: %.2f!"%(100*r_acc.item(), r_loss.item()))

            np.save(f"part_idx_to_canon_rot_diff_zz_{self.local_rank}.npy", part_idx_to_canon_rot_diff_zz)
            np.save(f"part_idx_to_rot_diff_zz_{self.local_rank}.npy", part_idx_to_rot_diff_zz)
            np.save(f"part_idx_to_trans_diff_zz_{self.local_rank}.npy", part_idx_to_trans_diff_zz)
            np.save(f"part_idx_to_trans_diff_2_zz_{self.local_rank}.npy", part_idx_to_trans_diff_2_zz)

            # accs = np.array(accs, dtype=np.float32)
            avg_accs = []
            avg_accs_2 = []
            for i_iter in range(self.n_iters):
                avg_accs.append(sum(accs[i_iter]) / len(accs[i_iter]))
                avg_accs_2.append(sum(accs_2[i_iter]) / len(accs_2[i_iter]))
            avg_R_dist = sum(avg_R_dists) / float(len(avg_R_dists))
            avg_glb_ori_recon_dist = sum(avg_glb_ori_recon_dists) / float(len(avg_glb_ori_recon_dists))
            avg_axis_angle_value = sum(axis_angle_val) / float(len(axis_angle_val))
            avg_dist_pred_gt_offset = sum(avg_dist_pred_gt_offset) / float(len(avg_dist_pred_gt_offset))
            # glb_recon_chamfer_l1 = []
            # slot_recon_chamfer_l1 = []
            avg_glb_recon_chamfer_l1 = sum(glb_recon_chamfer_l1) / float(len(glb_recon_chamfer_l1))
            avg_slot_recon_chamfer_l1 = sum(slot_recon_chamfer_l1) / float(len(slot_recon_chamfer_l1))
            if self.local_rank == 0:
                log_str = ""
                for i_iter in range(self.n_iters):
                    log_str += f" Avg_Acc_{i_iter}: %.2f" % (100 * avg_accs[i_iter])
                    log_str += f" Avg_Acc_2_{i_iter}: %.2f" % (100 * avg_accs_2[i_iter])
                # log_str += " avg_R_dist: %.4f" % float(avg_R_dist.item())
                log_str += " avg_R_dist: %.4f" % float(avg_R_dist)
                log_str += " avg_ori_recon_dist: %.4f" % float(avg_glb_ori_recon_dist)
                log_str += " axis_angle_dist: %.4f" % float(avg_axis_angle_value)
                log_str += " avg_dist_pred_gt_offset: %.4f" % float(avg_dist_pred_gt_offset)
                log_str += " avg_glb_recon_chamfer_l1: %.4f" % float(avg_glb_recon_chamfer_l1)
                log_str += " avg_slot_recon_chamfer_l1: %.4f" % float(avg_slot_recon_chamfer_l1)
                # self.logger.log('Testing', 'Average accuracy is %.2f!!!!'%(100*accs.mean()))
                self.logger.log('Testing', log_str)
                # self.test_accs.append(100*accs.mean())
                self.test_accs.append(100*avg_accs[i_iter]) # record average acc
                best_acc = np.array(self.test_accs).max() # get best test acc so far
                self.logger.log('Testing', 'Best accuracy so far is %.2f!!!!' % (best_acc)) # log best acc so far

                if self.pre_compute_delta >= 1:
                    log_str = ""
                    for i_p in part_idx_to_rot_diff:
                        cur_part_rot_diff = sum(part_idx_to_rot_diff[i_p]) / len(part_idx_to_rot_diff[i_p])
                        sorted_cur_part_rot_diff = sorted(part_idx_to_rot_diff[i_p])
                        medium_cur_part_rot_diff = sorted_cur_part_rot_diff[len(sorted_cur_part_rot_diff) // 2]
                        cur_part_canon_rot_diff = sum(part_idx_to_canon_rot_diff[i_p]) / len(part_idx_to_canon_rot_diff[i_p])
                        sorted_cur_part_canon_rot_diff = sorted(part_idx_to_canon_rot_diff[i_p])
                        medium_cur_part_canon_rot_diff = sorted_cur_part_canon_rot_diff[len(sorted_cur_part_canon_rot_diff) // 2]
                        # cur_slot_rot_diff = sum(slot_idx_to_rot_diff[i_p]) / len(slot_idx_to_rot_diff[i_p])
                        # cur_slot_canon_rot_diff = sum(slot_idx_to_canon_rot_diff[i_p]) / len(
                        #     slot_idx_to_canon_rot_diff[i_p])
                        cur_part_trans_diff = sum(part_idx_to_trans_diff[i_p]) / len(part_idx_to_trans_diff[i_p])
                        sorted_cur_part_trans_diff = sorted(part_idx_to_trans_diff[i_p])
                        medium_cur_part_trans_diff = sorted_cur_part_trans_diff[len(sorted_cur_part_trans_diff) // 2]
                        cur_part_trans_diff_2 = sum(part_idx_to_trans_diff_2[i_p]) / len(part_idx_to_trans_diff_2[i_p])
                        sorted_cur_part_trans_diff_2 = sorted(part_idx_to_trans_diff_2[i_p])
                        medium_cur_part_trans_diff_2 = sorted_cur_part_trans_diff_2[len(sorted_cur_part_trans_diff_2) // 2]
                        cur_part_pred_pred_diff = sum(part_idx_to_pred_posed_posed_diff[i_p]) / len(part_idx_to_pred_posed_posed_diff[i_p])
                        log_str += f"part idx: {i_p}, rot_diff: {cur_part_rot_diff}/{medium_cur_part_rot_diff}, canon_rot_diff: {cur_part_canon_rot_diff}/{medium_cur_part_canon_rot_diff}, posed_posed_diff: {cur_part_pred_pred_diff}, trans_diff: {cur_part_trans_diff}/{medium_cur_part_trans_diff}, trans_diff_2: {cur_part_trans_diff_2}/{medium_cur_part_trans_diff_2}\n"
                    avg_part_rel_rot_diff = sum(part_rel_rot_diff) / len(part_rel_rot_diff)
                    # log_str
                    log_str += f"part_rel_rot_diff: {avg_part_rel_rot_diff}\n"
                    for cur_part_pair in part_pair_to_part_rel_rot_diff:
                        curr_rot_diffs = part_pair_to_part_rel_rot_diff[cur_part_pair]
                        curr_rot_diffs_delta = part_pair_to_part_rel_rot_delta_diff[cur_part_pair]
                        avg_curr_rot_diff = sum(curr_rot_diffs) / len(curr_rot_diffs)
                        avg_curr_rot_diff_delta = sum(curr_rot_diffs_delta) / len(curr_rot_diffs_delta)
                        log_str += f"part pair: {cur_part_pair}, rot diff: {avg_curr_rot_diff}, rot_diff_delta: {avg_curr_rot_diff_delta}\n"
                    # trans diff is not accurate...
                    self.logger.log('Testing', log_str)

                    log_str = ""
                    for i_s in slot_idx_to_rot_diff:
                        cur_slot_rot_diff = sum(slot_idx_to_rot_diff[i_s]) / len(slot_idx_to_rot_diff[i_s])
                        sorted_cur_slot_rot_diff = sorted(slot_idx_to_rot_diff[i_s])
                        medium_cur_slot_rot_diff = sorted_cur_slot_rot_diff[len(sorted_cur_slot_rot_diff) // 2]
                        cur_slot_canon_rot_diff = sum(slot_idx_to_canon_rot_diff[i_s]) / len(
                            slot_idx_to_canon_rot_diff[i_s])
                        sorted_cur_slot_canon_rot_diff = sorted(slot_idx_to_canon_rot_diff[i_s])
                        medium_cur_slot_canon_rot_diff = sorted_cur_slot_canon_rot_diff[len(sorted_cur_slot_canon_rot_diff) // 2]
                        log_str += f"slot idx: {i_s}, slot_rot_diff: {cur_slot_rot_diff}, canon_slot_rot_diff: {cur_slot_canon_rot_diff}/{medium_cur_slot_canon_rot_diff}\n"
                    self.logger.log('Testing', log_str)

                    for gt_part_idx in part_idx_to_pred_posed_canon_diff:
                        # part_idx_to_pred_posed_canon_diff[gt_part_idx] = np.concatenate(part_idx_to_pred_posed_canon_diff[gt_part_idx], axis=0)
                        part_idx_to_pred_posed_canon_diff[gt_part_idx] = torch.cat(part_idx_to_pred_posed_canon_diff[gt_part_idx], dim=0).detach().cpu().numpy()
                    np.save(f"{self.shape_type}_part_idx_to_pred_posed_canon_diff.npy", part_idx_to_pred_posed_canon_diff)

            # self.logger.log("Testing", 'Here to peek at the lmc') # we should infer pose information?
            # self.logger.log("Testing", str(lmc))
            # import ipdb; ipdb.set_trace()
            # n = 1
            # mAP = modelnet_retrieval_mAP(all_feats,all_labels,n)
            # self.logger.log('Testing', 'Mean average precision at %d is %f!!!!'%(n, mAP))

        # self.model.module.train()
        # self.metric.train()
