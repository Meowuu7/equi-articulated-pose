from importlib import import_module
# from SPConvNets import Dataloader_ModelNet40
# from SPConvNets.datasets.ToySegDataset import Dataloader_ToySeg
from SPConvNets.datasets.MotionSegDataset import PartSegmentationMetaInfoDataset
# from SPConvNets.datasets.shapenetpart_dataset import ShapeNetPartDataset
# from SPConvNets.datasets.MotionDataset import MotionDataset
from SPConvNets.datasets.MotionDataset2 import MotionDataset as MotionDataset2
from SPConvNets.datasets.MotionSAPIENDataset import MotionDataset as MotionSAPIENDataset
from SPConvNets.datasets.MotionSAPIENDatasetNPCS import MotionDataset as MotionSAPIENDatasetNPCS
from SPConvNets.datasets.MotionDatasetNPCS import MotionDataset as MotionDatasetNPCS
# from SPConvNets.datasets.MotionHOIDataset import MotionDataset as MotionHOIDataset
from SPConvNets.datasets.MotionHOIDatasetNPCS import MotionDataset as MotionHOIDatasetNPCS
from tqdm import tqdm

try:
    from SPConvNets.datasets.MotionDatasetNPCSPartial import  MotionDataset as MotionDatasetNPCSPartial
except:
    pass


try:
    from SPConvNets.datasets.MotionHOIDatasetNPCSPartial import  MotionDataset as MotionHOIDatasetNPCSPartial
except:
    pass

try:
    from SPConvNets.datasets.MotionSAPIENDatasetNPCSPartial import  MotionDataset as MotionSAPIENDatasetNPCSPartial
except:
    pass

import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from .utils.loss_util import iou, calculate_res_relative_Rs, batched_index_select
import torch.nn as nn
# from datas
import math

from SPConvNets.eval_utils import *

import torch.backends.cudnn as cudnn
import torch.distributed as dist

from SPConvNets.pose_utils import *

from SPConvNets.models.common_utils import *
from SPConvNets.ransac import *


class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        ''' dummy '''
        ''' Set device '''
        if torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend='nccl')

            tmpp_local_rnk = int(os.environ['LOCAL_RANK'])
            print("os_environed:", tmpp_local_rnk)
            self.local_rank = tmpp_local_rnk
            torch.cuda.set_device(self.local_rank)
            print(f"local_rank: {self.local_rank}")

            self._use_multi_gpu = True

            cudnn.benchmark = True
        else:
            self._use_multi_gpu = False
            self.local_rank = 0

        opt.device = 0

        self.n_step = 0
        self.n_dec_steps = opt.equi_settings.n_dec_steps
        self.lr_adjust = opt.equi_settings.lr_adjust
        self.lr_decay_factor = opt.equi_settings.lr_decay_factor
        self.pre_compute_delta = opt.equi_settings.pre_compute_delta
        self.num_slots = opt.nmasks
        # torch.cuda.set_device(opt.parallel.local_rank)
        # self.local_rank = opt.parallel.local_rank

        self.slot_recon_factor = opt.equi_settings.slot_recon_factor


        ''' Set shape type '''
        self.opt_shape_type = opt.equi_settings.shape_type

        ''' Set number of procs '''
        opt.nprocs = torch.cuda.device_count()
        print("device count:", opt.nprocs)
        nprocs = opt.nprocs
        self.nprocs = nprocs

        self.dataset_type = opt.equi_settings.dataset_type

        self.attention_model = opt.model.flag.startswith('attention') and opt.debug_mode != 'knownatt'
        super(Trainer, self).__init__(opt)

        # if self.attention_model:
        #     self.summary.register(['Loss', 'Acc', 'R_Loss', 'R_Acc'])
        # else:
        self.n_iters = opt.equi_settings.num_iters
        reg_strs = []
        reg_strs.append('Loss')
        reg_strs.append('NPCS Loss')
        reg_strs.append('Acc')
        reg_strs.append('Axis')
        reg_strs.append('Offset')
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



        ''' Setup predicted features save folder '''
        predicted_info_folder = "predicted_info"
        if not os.path.exists(predicted_info_folder):
            os.mkdir(predicted_info_folder)
        cur_shape_info_folder = os.path.join(predicted_info_folder, self.shape_type)
        if not os.path.exists(cur_shape_info_folder):
            os.mkdir(cur_shape_info_folder)
        cur_setting_info_folder =  self.model.module.log_fn if self._use_multi_gpu else self.model.log_fn
        cur_setting_info_folder = os.path.join(cur_shape_info_folder, cur_setting_info_folder)
        if not os.path.exists(cur_setting_info_folder):
            os.mkdir(cur_setting_info_folder)
        self.predicted_info_save_folder = cur_setting_info_folder
        # print(f"predicted_info_save_folder: {self.predicted_info_save_folder}")

        if self.local_rank == 0:
            if not os.path.exists(cur_setting_info_folder):
                os.mkdir(cur_setting_info_folder)



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

            if self.shape_type == DATASET_MOTION:
                cur_dataset = MotionDatasetNPCS if self.shape_type != DATASET_DRAWER else MotionSAPIENDatasetNPCS
            else:
                cur_dataset = MotionDatasetNPCSPartial if self.shape_type != DATASET_DRAWER else MotionSAPIENDatasetNPCSPartial

            # cur_dataset = MotionDatasetNPCS if self.shape_type != DATASET_DRAWER else MotionSAPIENDatasetNPCS
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
                                                                shuffle=True,
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
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, shuffle=True)
                self.dataset = torch.utils.data.DataLoader(dataset,
                                                           batch_size=self.opt.batch_size,
                                                           sampler=train_sampler,
                                                           num_workers=self.opt.num_thread)
                self.dataset_iter = iter(self.dataset)

            dataset_test = MotionDataset2(
                root="./data/MDV02",
                npoints=npoints, split='test', nmask=10,
                shape_type=shape_type, args=self.opt, global_rot=global_rot)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)
            self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                            batch_size=self.opt.batch_size,
                                                            sampler=test_sampler,
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

            cur_dataset = MotionHOIDatasetNPCS if self.dataset_type == DATASET_HOI4D else MotionHOIDatasetNPCSPartial


            val_str = "val" if self.shape_type != DATASET_DRAWER else "test"

            ''' Shapes from motion segmentation dataset '''
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
                                                                shuffle=True,
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

            if i > 0 and i < 5000 and i % self.opt.save_freq == 0:
                self._save_network(f'Iter{i}')
                self.test()

    ''' Load checkpoint '''
    def _resume_from_ckpt(self, resume_path):
        if resume_path is None:
            self.logger.log('Setup', f'Seems like we train from scratch!')
            return
        self.logger.log('Setup', f'Resume from checkpoint: {resume_path}')

        state_dicts = torch.load(resume_path, map_location="cpu")

        # self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(state_dicts)
        # self.model = self.model.module
        # self.optimizer.load_state_dict(state_dicts['optimizer'])
        # self.start_epoch = state_dicts['epoch']
        # self.start_iter = state_dicts['iter']
        self.logger.log('Setup', f'Resume finished! Great!')

    ''' Setup model with multi-gpu '''
    def _setup_model_multi_gpu(self):
        if torch.cuda.device_count() > 1:
            self.logger.log('Setup', 'Using Multi-gpu and DataParallel!')
            self._use_multi_gpu = True
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = self.model.cuda(self.local_rank)
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model,  device_ids=[self.local_rank], find_unused_parameters=True)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,  device_ids=[self.local_rank], find_unused_parameters=False)
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model,  device_ids=[self.local_rank],)
            # self.model = nn.DataParallel(self.model)
        else:
            self.logger.log('Setup', 'Using Single-gpu!')
            self._use_multi_gpu = False
            self.model = self.model.cuda()

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

        # if torch.cuda.is_available():
        #     # torch.cuda.device(gpu_id)
        #     # self.model.to(self.opt.device)
        #     self.model.to(self.local_rank)
        self.logger.log('Training', f'Checkpoint saved to: {save_path}!')

    def save_predicted_by_step(self, out_feats):
        save_root_path = os.path.join(self.root_dir, "out_step")
        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)
        save_path = os.path.join(save_root_path, f"out_feats_Iter_{self.n_step}.npy")
        np.save(save_path, out_feats)
        print(f"Out features in step {self.n_step} saved to path {save_path}!")

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

        self._optimize(data)
        self.iter_counter += 1

    def save_predicted_infos(self, idxes_list, out_feats, test=False):
        idxes_list_str = [str(ii) for ii in idxes_list]
        idxes_save_str = ",".join(idxes_list_str)
        if test == True:
            save_pth = os.path.join(self.predicted_info_save_folder, idxes_save_str + ".npy")
        np.save(os.path.join(self.predicted_info_save_folder, idxes_save_str + ".npy"), out_feats)

    def save_predicted_infos_all_iters(self, idxes_list, out_feats):
        idxes_list_str = [str(ii) for ii in idxes_list]
        idxes_save_str = ",".join(idxes_list_str)
        np.save(os.path.join(self.predicted_info_save_folder, idxes_save_str + "_all_iters" + ".npy"), out_feats)


    ''' Train '''
    def _optimize(self, data):
        # set to train mode
        # optimize model
        if self._use_multi_gpu:
            self.model.module.train()
        else:
            self.model.train()
        self.metric.train()

        weighted_step = math.exp(-1. * (self.n_step) / self.n_dec_steps)
        weighted_step = math.exp(-1. * (self.n_step) / 200)

        weighted_step = 1. - weighted_step
        # weighted_step = 1.
        # weighted_step = self.n_step / 800
        # weighted_step = weighted_step - 1.
        weighted_step = max(0.0, min(1.0, weighted_step))


        # if local rank is zero, print weighted_step
        # if self.local_rank == 0:
        #     print(f"weighted_step current epoch: {weighted_step}")

        if self._use_multi_gpu:
            self.model.module.slot_recon_factor = self.slot_recon_factor * weighted_step
        else:
            self.model.slot_recon_factor = self.slot_recon_factor * weighted_step



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
        # ori_pc = data['ori_pc'].cuda(non_blocking=True)
        canon_pc = data['canon_pc'].cuda(non_blocking=True)
        oorr_pc = data['oorr_pc']
        oorr_label = data['oorr_label']
        part_axis = data['part_axis']  # ground-trut axis...
        part_axis = part_axis.cuda(non_blocking=True)
        be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
        minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)

        if self.shape_type != 'drawer':
            part_pv_offset = data['part_pv_offset'].cuda(non_blocking=True)
            # part_pv_point = data['cur_part_pv_point']
        else:
            part_pv_offset = torch.zeros((part_axis.size(0), part_axis.size(1)), dtype=torch.float32).cuda(non_blocking=True)

        bz, N = in_tensors.size(0), in_tensors.size(2)

        label = torch.eye(self.opt.nmasks)[in_label].cuda()

        loss = self.model(in_tensors, in_pose, ori_pc=oorr_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc)

        if self._use_multi_gpu:
            label = torch.eye(self.model.module.num_slots)[in_label].cuda(non_blocking=True)
            oorr_label = torch.eye(self.model.module.num_slots)[oorr_label]  # .cuda(non_blocking=True)
        else:
            label = torch.eye(self.model.num_slots)[in_label].cuda(non_blocking=True)
            oorr_label = torch.eye(self.model.num_slots)[oorr_label]  # .cuda(non_blocking=True)

        if self._use_multi_gpu:
            pred_axis = self.model.module.pred_axis
            pred_pv = self.model.module.pred_pv
        else:
            pred_axis = self.model.pred_axis
            pred_pv = self.model.pred_pv

        dot_prod = torch.abs(torch.sum(pred_axis.unsqueeze(1) * part_axis, dim=-1))
        mean_dot_prod_val = dot_prod.mean().item()

        axis_loss = -1.0 * dot_prod.mean()

        pred_part_pv_point_offset = pred_pv[...,0]
        dist_pred_gt_offset = torch.abs(pred_part_pv_point_offset - part_pv_offset).mean().item()
        dist_pred_gt_offset_loss = ((pred_part_pv_point_offset - part_pv_offset) ** 2).mean()

        label, gt_conf = self.get_gt_conf(label)


        oorr_label, oorr_gt_conf = self.get_gt_conf(oorr_label)

        ''' Previous prediction and iou calculation '''
        # pred = torch.zeros((bz, 200, N), dtype=torch.float32).cuda(non_blocking=True)
        # iou_value, _, _ = iou(pred, gt_x=label, gt_conf=gt_conf)
        ''' Previous prediction and iou calculation '''

        if self._use_multi_gpu:
            curr_attn = self.model.module.attn
        else:
            curr_attn = self.model.attn
        iou_value, matching_idx_gt, matching_idx_pred = iou(curr_attn, gt_x=label, gt_conf=gt_conf)
        iou_value = iou_value.mean()
        iou_loss = -iou_value.mean()

        self.loss = loss + iou_loss + dist_pred_gt_offset_loss + axis_loss
        # acc = iou_value.mean()

        if self._use_multi_gpu:
            torch.distributed.barrier()

            self.loss = self.reduce_mean(self.loss, self.nprocs)
            iou_value = self.reduce_mean(iou_value, self.nprocs)
            dist_pred_gt_offset_loss = self.reduce_mean(dist_pred_gt_offset_loss, self.nprocs)
            axis_loss = self.reduce_mean(axis_loss, self.nprocs)
        # loss = self.reduce_mean(loss, self.nprocs)

        ''' Optimize loss '''
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        log_info = {
            'Loss': self.loss.item(),
            'NPCS Loss': loss.item(),
            'Acc': iou_value.mean().item(),
            'Axis': mean_dot_prod_val,
            'Offset': dist_pred_gt_offset
        }

        self.summary.update(log_info)
        if self.local_rank == 0:

            stats = self.summary.get()
            self.logger.log('Training', f'{stats}')
            cur_fn = self.model.module.log_fn if self._use_multi_gpu else self.model.log_fn
            if not os.path.exists(cur_fn):
                os.mkdir(cur_fn)
            with open(os.path.join(cur_fn, "logs.txt"), "a") as wf:
                # wf.write(f"Loss: {loss.item()}, Acc: {acc.item()}\n")
                wf.write(f"Loss: {self.loss.item()}; NPCS Loss: {loss.item()}; Acc: {iou_value.item()}\n")

                wf.close()


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

    def eval(self):

        # evaluate test dataset
        self.logger.log('Testing','Evaluating test set!')
        # self.model.module.eval()
        self.model.eval()
        self.metric.eval()

        with torch.no_grad():

            part_idx_to_rot_diff = {}
            part_idx_to_trans_diff = {}
            part_idx_to_scale_diff = {}
            iou_all = []
            avg_angle_diff = []
            part_idx_to_offset_diff = {}

            for it, data in enumerate(self.dataset_test):
                print(it)
                in_tensors = data['pc'].cuda(non_blocking=True)
                bdim = in_tensors.shape[0]
                #
                in_label = data['label'].cuda(non_blocking=True) # .reshape(-1)
                # per-point pose
                in_pose = data['pose'].cuda(non_blocking=True) # if self.opt.debug_mode == 'knownatt' else None
                # per-part pose
                in_pose_segs = data['pose_segs'].cuda(non_blocking=True).float()
                # ori_pc = data['ori_pc'].cuda(non_blocking=True)
                canon_pc = data['canon_pc'].cuda(non_blocking=True)
                if 'npcs_trans' not in data:
                    npcs_trans = torch.zeros((in_tensors.size(0), in_pose_segs.size(1), 3), dtype=torch.float32).cuda()
                else:
                    npcs_trans = data['npcs_trans'].cuda(non_blocking=True).float()
                # part_state_rots = data['part_state_rots'].cuda(non_blocking=True)
                # part_ref_rots = data['part_ref_rots'].cuda(non_blocking=True) # [0]

                axis_angles = []
                axis_offsets = []

                oorr_pc = data['oorr_pc']
                oorr_label = data['oorr_label']
                part_axis = data['part_axis']  # ground-trut axis...
                part_axis = part_axis.cuda(non_blocking=True)
                be_af_dists = torch.sum((oorr_pc.unsqueeze(-1) - in_tensors.detach().cpu().unsqueeze(-2)) ** 2, dim=1)
                minn_dist, minn_idx = torch.min(be_af_dists, dim=-1)

                data_idxes = data['idx'].detach().cpu().numpy().tolist()
                data_idxes = [str(ii) for ii in data_idxes]
                data_idxes_to_name = ",".join(data_idxes)

                bz = in_tensors.size(0)
                N = in_tensors.size(2)

                label = torch.eye(self.opt.nmasks)[in_label].cuda(non_blocking=True)
                label, gt_conf = self.get_gt_conf(label)

                # Get iou_loss and npcs_loss
                loss = self.model(in_tensors, in_pose, ori_pc=canon_pc, rlabel=label, pose_segs=in_pose_segs, canon_pc=canon_pc)

                if self._use_multi_gpu:
                    oorr_label = torch.eye(self.model.module.num_slots)[oorr_label] # .cuda(non_blocking=True)
                    curr_attn = self.model.module.attn
                else:
                    oorr_label = torch.eye(self.model.num_slots)[oorr_label]  # .cuda(non_blocking=True)
                    curr_attn = self.model.attn

                if self._use_multi_gpu:
                    pred_axis = self.model.module.pred_axis
                    pred_pv = self.model.module.pred_pv
                else:
                    pred_axis = self.model.pred_axis
                    pred_pv = self.model.pred_pv

                if self.shape_type != 'drawer':
                    part_pv_offset = data['part_pv_offset'].cuda(non_blocking=True)
                    # part_pv_point = data['cur_part_pv_point']
                else:
                    part_pv_offset = torch.zeros((part_axis.size(0), part_axis.size(1)), dtype=torch.float32).cuda(
                        non_blocking=True)

                dot_prod = torch.abs(torch.sum(pred_axis.unsqueeze(1) * part_axis, dim=-1))
                mean_dot_prod_val = dot_prod.mean().item()

                # axis_loss = -1.0 * dot_prod.mean()

                pred_part_pv_point_offset = pred_pv[..., 0]
                dist_pred_gt_offset = torch.abs(pred_part_pv_point_offset - part_pv_offset).mean(0)
                # dist_pred_gt_offset_loss = ((pred_part_pv_point_offset - part_pv_offset) ** 2).mean()

                mean_angle = math.acos(min(mean_dot_prod_val, 1.0)) / np.pi * 180.0

                avg_angle_diff.append(mean_angle)

                for ii_p in range(dist_pred_gt_offset.size(0)):
                    cur_iip_offset = float(dist_pred_gt_offset[ii_p].item())
                    if ii_p not in part_idx_to_offset_diff:
                        part_idx_to_offset_diff[ii_p] = [cur_iip_offset]
                    else:
                        part_idx_to_offset_diff[ii_p].append(cur_iip_offset)

                oorr_label, oorr_gt_conf = self.get_gt_conf(oorr_label)

                iou_value, matching_idx_gt, matching_idx_pred = iou(curr_attn, gt_x=label, gt_conf=gt_conf)
                cur_pred_label = torch.argmax(curr_attn, dim=1)

                iou_value = iou_value.mean()
                iou_loss = -iou_value.mean()

                iou_all.append(float(iou_value.mean().item()))

                print(f"iou: {iou_value.mean().item()}")

                self.loss = iou_loss + loss

                # Get rotation, translation and scale factors
                # pred_npcs: bz x 3 x N
                if self._use_multi_gpu:
                    pred_npcs = self.model.module.pred_npcs # just used
                else:
                    pred_npcs = self.model.pred_npcs  # just used

                print(in_tensors.size(), pred_npcs.size(), matching_idx_gt.size(), matching_idx_pred.size())

                save_dict = {
                    'cur_pred_label': cur_pred_label.detach().cpu().numpy(),
                    'ori_pc': in_tensors.detach().cpu().numpy(),
                    'canon_pc': canon_pc.detach().cpu().numpy()
                }

                for i_bz in range(in_pose_segs.size(0)): # number of gt parts
                    # Get gt_seg_idx to pred_seg_idx
                    cur_bz_matching_idx_gt = matching_idx_gt[i_bz] #
                    cur_bz_matching_idx_pred = matching_idx_pred[i_bz]
                    gt_seg_to_pred_seg = {}
                    for i_ss in range(cur_bz_matching_idx_gt.size(0)):
                    # for i_ss in range(in_pose_segs.size(1)):
                        cur_gt_seg_idx = int(cur_bz_matching_idx_gt[i_ss].item())
                        cur_pred_seg_idx = int(cur_bz_matching_idx_pred[i_ss].item())
                        print(i_ss, f"gt idx: {cur_gt_seg_idx}, pred idx: {cur_pred_seg_idx}")
                        if cur_gt_seg_idx not in gt_seg_to_pred_seg:
                            gt_seg_to_pred_seg[cur_gt_seg_idx] = cur_pred_seg_idx

                    # gt_seg_to_pred_seg = {0: 3, 1: 1}
                    # pred label
                    print(f"gt_seg_to_pred_seg: {gt_seg_to_pred_seg}")
                    print(f"cur_pred_label: {cur_pred_label.size()}")
                    seg_idx_to_pts_idx = {}
                    for i_pts in range(cur_pred_label.size(1)):
                        cur_pts_pred_label = int(cur_pred_label[i_bz, i_pts].item())
                        if cur_pts_pred_label not in seg_idx_to_pts_idx:
                            seg_idx_to_pts_idx[cur_pts_pred_label] = [i_pts]
                        else:
                            seg_idx_to_pts_idx[cur_pts_pred_label].append(i_pts)
                    for i_seg in seg_idx_to_pts_idx:
                        seg_idx_to_pts_idx[i_seg] = torch.tensor(seg_idx_to_pts_idx[i_seg], dtype=torch.long).long().cuda()
                    print(f"seg_idx_to_pts_idx.keys(): {seg_idx_to_pts_idx.keys()}")

                    ''' Save releated infos for debuging '''
                    # save related infos
                    if not os.path.exists(self.model.log_fn):
                        os.mkdir(self.model.log_fn)
                    np.save(os.path.join(self.model.log_fn, f"save_dict_{data_idxes_to_name}.npy"), save_dict)

                    dataset_sources = []
                    dataset_targets = []

                    gt_part_idx_to_est_rot = {}
                    # gt_part_idx_to_e

                    for i_p in range(in_pose_segs.size(1)):
                        cur_bz_cur_seg_pose = in_pose_segs[i_bz, i_p]
                        cur_bz_cur_seg_npcs_trans = npcs_trans[i_bz, i_p].detach().cpu()
                        # cur_rot: 3 x 3; cur_trans: 3 x 3
                        cur_rot, cur_trans = cur_bz_cur_seg_pose[:3, :3], cur_bz_cur_seg_pose[:3, 3]
                        cur_rot = cur_rot.detach().cpu() # .numpy()
                        cur_trans = cur_trans.detach().cpu().numpy()
                        # from npcs to current state
                        npcs_rot_trans = torch.matmul(cur_rot, cur_bz_cur_seg_npcs_trans.unsqueeze(-1)).squeeze(-1)
                        cur_trans = cur_trans - npcs_rot_trans.numpy() #

                        if i_p in gt_seg_to_pred_seg:
                            cur_pred_seg_idx = gt_seg_to_pred_seg[i_p]

                            if cur_pred_seg_idx not in seg_idx_to_pts_idx:
                                cur_pred_seg_idx = 0 if cur_pred_seg_idx == 2 else 2

                            # cur_pred_seg_transformed_pts: n_pts_part x 3
                            # cur_pred_seg_canon_pts: n_pts_part x 3
                            # from predicted canonical point clouds to original transformed point clouds
                            cur_pred_seg_transformed_pts = safe_transpose(in_tensors, 1, 2)[i_bz, seg_idx_to_pts_idx[cur_pred_seg_idx]]
                            cur_pred_seg_canon_pts = safe_transpose(pred_npcs, 1, 2)[i_bz, seg_idx_to_pts_idx[cur_pred_seg_idx]]
                            # cur_pred_seg_canon_pts = canon_pc[i_bz, seg_idx_to_pts_idx[cur_pred_seg_idx]]

                            niter = 1000
                            inlier_th = 0.05

                            dataset = dict()
                            dataset['source'] = cur_pred_seg_canon_pts.detach().cpu().numpy()
                            dataset['target'] = cur_pred_seg_transformed_pts.detach().cpu().numpy()
                            dataset['nsource'] = dataset['source'].shape[0]

                            best_model, best_inliers = ransac(dataset, single_transformation_estimator,
                                                              single_transformation_verifier, inlier_th, niter)
                            # rdiff = 180.0
                            # for ipp in range(in_pose_segs.size(1)):
                                # cur_rot, cur_trans = cur_bz_cur_seg_pose[:3, :3], cur_bz_cur_seg_pose[:3, 3]
                            rdiff = rot_diff_degree(torch.from_numpy(best_model['rotation']), cur_rot)
                            rdiff = min(rdiff, 180. - rdiff)
                                # rdiff = min(rdiff, cur_rdiff)
                            tdiff = float(np.linalg.norm(best_model['translation'] - cur_trans).item())
                            sdiff = float(np.linalg.norm(best_model['scale'] - 1.0).item())

                            if i_p not in part_idx_to_rot_diff:
                                part_idx_to_rot_diff[i_p] = [rdiff]
                                part_idx_to_trans_diff[i_p] = [tdiff]
                                part_idx_to_scale_diff[i_p] = [sdiff]
                            else:
                                part_idx_to_rot_diff[i_p].append(rdiff)
                                part_idx_to_trans_diff[i_p].append(tdiff)
                                part_idx_to_scale_diff[i_p].append(sdiff)

                            print(f"curr rot diff: {rdiff}, trans diff: {tdiff}, sacle diff: {sdiff}")

                if self.local_rank == 0:
                    for i_p in part_idx_to_scale_diff:
                        avg_rot_diff = sum(part_idx_to_rot_diff[i_p]) / len(part_idx_to_rot_diff[i_p])
                        sorted_part_diffs = sorted(part_idx_to_rot_diff[i_p])
                        med_rot_diff = sorted_part_diffs[len(sorted_part_diffs) // 2]
                        avg_trans_diff = sum(part_idx_to_trans_diff[i_p]) / len(part_idx_to_trans_diff[i_p])
                        sorted_trans_diffs = sorted(part_idx_to_trans_diff[i_p])
                        med_trans_diff = sorted_trans_diffs[len(sorted_trans_diffs) // 2]
                        print(
                            f"part_idx: {i_p}, rot_diff_mean: {avg_rot_diff}/{med_rot_diff}, trans_diff_mean: {avg_trans_diff}/{med_trans_diff}, scale_diff_mean: {sum(part_idx_to_scale_diff[i_p]) / len(part_idx_to_scale_diff[i_p])}, iou_mean: {sum(iou_all) / len(iou_all)}")
                    for ii_p in part_idx_to_offset_diff:
                        cur_part_avg_offset = sum(part_idx_to_offset_diff[ii_p]) / float(len(part_idx_to_offset_diff[ii_p]))
                        print(f"avg_offset: {cur_part_avg_offset}")
                    avg_angle = sum(avg_angle_diff) / float(len(avg_angle_diff))
                    print(f"avg_angle: {avg_angle}")

            if self.local_rank == 0:
                avg_rot_diff = sum(part_idx_to_rot_diff[i_p]) / len(part_idx_to_rot_diff[i_p])
                sorted_part_diffs = sorted(part_idx_to_rot_diff[i_p])
                med_rot_diff = sorted_part_diffs[len(sorted_part_diffs) // 2]
                avg_trans_diff = sum(part_idx_to_trans_diff[i_p]) / len(part_idx_to_trans_diff[i_p])
                sorted_trans_diffs = sorted(part_idx_to_trans_diff[i_p])
                med_trans_diff = sorted_trans_diffs[len(sorted_trans_diffs) // 2]
                print(
                    f"part_idx:  {i_p}, rot_diff_mean: {avg_rot_diff}/{med_rot_diff}, trans_diff_mean: {avg_trans_diff}/{med_trans_diff}, scale_diff_mean: {sum(part_idx_to_scale_diff[i_p]) / len(part_idx_to_scale_diff[i_p])}, iou_mean: {sum(iou_all) / len(iou_all)}")
                for ii_p in part_idx_to_offset_diff:
                    cur_part_avg_offset = sum(part_idx_to_offset_diff[ii_p]) / float(len(part_idx_to_offset_diff[ii_p]))
                    print(f"avg_offset: {cur_part_avg_offset}")
                avg_angle = sum(avg_angle_diff) / float(len(avg_angle_diff))
                print(f"avg_angle: {avg_angle}")

