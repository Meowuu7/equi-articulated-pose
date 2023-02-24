from importlib import import_module
from SPConvNets import Dataloader_ModelNet40
from SPConvNets.datasets.ToySegDataset import Dataloader_ToySeg
from SPConvNets.datasets.MotionSegDataset import PartSegmentationMetaInfoDataset
from SPConvNets.datasets.shapenetpart_dataset import ShapeNetPartDataset
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from .utils.loss_util import iou
import torch.nn as nn
# from datas


class Trainer(vgtk.Trainer):
    def __init__(self, opt):
        ''' dummy '''
        ''' Set device '''
        # torch.cuda.set_device(opt.parallel.local_rank)
        # torch.distributed.init_process_group(backend='nccl')


        self.attention_model = opt.model.flag.startswith('attention') and opt.debug_mode != 'knownatt'
        super(Trainer, self).__init__(opt)

        # if self.attention_model:
        #     self.summary.register(['Loss', 'Acc', 'R_Loss', 'R_Acc'])
        # else:
        self.summary.register(['Loss', 'Acc'])
        self.epoch_counter = 0 # epoch counter
        self.iter_counter = 0 # inter counter
        self.test_accs = [] # test metrics
        # self.
        self.best_acc_ever_reached = 0.0
        self.best_loss = 9999.0
        self.not_increased_epoch = 0
        self.last_loss = 9999.0


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

        ''' Shapes from motion segmentation dataset '''
        if self.opt.mode == 'train':

            dataset = PartSegmentationMetaInfoDataset(
                root="/home/xueyi/inst-segmentation/data/part-segmentation/data/motion_part_split_meta_info",
                npoints=npoints, split='train', nmask=10,
                # shape_types=["Sitting Furniture"],
                shape_types=["Laptop"],
                real_test=False, part_net_seg=True, partnet_split=False, args=self.opt)
            self.dataset = torch.utils.data.DataLoader(dataset,
                                                        batch_size=self.opt.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)

        dataset_test = PartSegmentationMetaInfoDataset(
                root="/home/xueyi/inst-segmentation/data/part-segmentation/data/motion_part_split_meta_info",
                npoints=npoints, split='val', nmask=10,
                # shape_types=["Sitting Furniture"],
                shape_types=["Laptop"],
                real_test=False, part_net_seg=True, partnet_split=False, args=self.opt)
        self.dataset_test = torch.utils.data.DataLoader(dataset_test,
                                                        batch_size=self.opt.batch_size,
                                                        shuffle=False,
                                                        num_workers=self.opt.num_thread)

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


    def _setup_model(self):
        if self.opt.mode == 'train': #
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

    # def _setup_model_multi_gpu(self):
    #     if torch.cuda.device_count() > 1:
    #         self.logger.log('Setup', 'Using Multi-gpu and DataParallel!')
    #         self._use_multi_gpu = True
    #         self.model = self.model.cuda()
    #         self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
    #         # self.model = nn.DataParallel(self.model)
    #     else:
    #         self.logger.log('Setup', 'Using Single-gpu!')
    #         self._use_multi_gpu = False
    #         self.model = self.model.cuda()

    def _setup_metric(self):
        # attention model?
        # using attention mechanism
        if self.attention_model:
            self.metric = vgtk.AttentionCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin)
            # self.r_metric = AnchorMatchingLoss()
        else:
            self.metric = vgtk.CrossEntropyLossPerP()

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

    ''' Train '''
    def _optimize(self, data):
        # set to train mode

        self.model.train()
        self.metric.train()
        self.adjust_lr_by_loss()
        # input tensors
        in_tensors = data['pc'].to(self.opt.device)
        # in_tensors = torch.

        bdim = in_tensors.shape[0]
        in_label = data['label'].to(self.opt.device) # .reshape(-1)
        in_pose = data['pose'].to(self.opt.device) #  if self.opt.debug_mode == 'knownatt' else None
        # import ipdb; ipdb.set_trace()
        # print("input shapes = ", in_tensors.size(), in_label.size(), in_pose.size())

        bz, N = in_tensors.size(0), in_tensors.size(2)


        ###################### ----------- debug only ---------------------
        # in_tensorsR = data['pcR'].to(self.opt.device)
        # import ipdb; ipdb.set_trace()
        ##################### --------------------------------------------

        # feed into the model: in_tensors, in_pos, and no rotation value
        # pred, feat =
        loss, pred = self.model(in_tensors, in_pose, None)


        # Need not to further transpose predictions
        # # transform `pred` to prediction probability
        # pred = torch.clamp(pred, min=-20, max=20)
        # pred = torch.softmax(pred, dim=-1)
        # bz x npred-class x N
        pred = pred.contiguous().transpose(1, 2).contiguous()

        if pred.size(1) < 200:
            pred = torch.cat(
                [pred, torch.zeros((bz, 200 - pred.size(1), N), dtype=torch.float32).cuda()], dim=1
            )

        label = torch.eye(200)[in_label].cuda()
        label, gt_conf = self.get_gt_conf(label)

        iou_value, _, _ = iou(pred, gt_x=label, gt_conf=gt_conf)
        # loss = -iou_value.mean()

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
        self.loss = loss
        acc = iou_value.mean()
        # acc = torch.zeros((1,))
        if acc.item() > self.best_acc_ever_reached:
            self.best_acc_ever_reached = acc.item()

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
            'Acc': 100 * acc.item(),
        }

        # self.logger.log("Training", "Accuracy: %.1f, Loss: %.2f!" % (100 * acc.item(), cls_loss.item()))

        self.summary.update(log_info)

        stats = self.summary.get()
        self.logger.log('Training', f'{stats}')
        # print(stats)
        self.last_loss = float(self.summary.get_item('Loss'))
        # print(f"Best current: {self.best_acc_ever_reached * 100}")

    def adjust_lr_by_loss(self):
        if self.last_loss < self.best_loss:
            # print()
            self.best_loss = self.last_loss
            self.not_increased_epoch = 0
        else:
            self.not_increased_epoch += 1
            if self.not_increased_epoch >= 30:
                print("Adjusting learning rate by 0.7!")
                self.adjust_learning_rate_by_factor(0.7)
                self.not_increased_epoch = 0

    def adjust_learning_rate_by_factor(self, scale_factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * scale_factor, 1e-7)

    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self):
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
        self.model.eval()
        self.metric.eval()

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
            accs = []
            # lmc = np.zeros([40,60], dtype=np.int32)

            all_labels = []
            all_feats = []

            for it, data in enumerate(self.dataset_test):
                in_tensors = data['pc'].to(self.opt.device)
                bdim = in_tensors.shape[0]
                in_label = data['label'].to(self.opt.device) # .reshape(-1)
                in_pose = data['pose'].to(self.opt.device)  # if self.opt.debug_mode == 'knownatt' else None

                bz = in_tensors.size(0)
                N = in_tensors.size(2)

                loss, pred = self.model(in_tensors, in_pose)

                # # transform `pred` to prediction probability
                # pred = torch.clamp(pred, min=-20, max=20)
                # pred = torch.softmax(pred, dim=-1)
                pred = pred.contiguous().transpose(1, 2).contiguous()

                if pred.size(1) < 200:
                    pred = torch.cat(
                        [pred, torch.zeros((bz, 200 - pred.size(1), N), dtype=torch.float32).cuda()], dim=1
                    )

                # from in_label to label
                label = torch.eye(200)[in_label].cuda()
                label, gt_conf = self.get_gt_conf(label)

                iou_value, _, _ = iou(pred, gt_x=label, gt_conf=gt_conf)
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

                acc = iou_value.mean()
                all_labels.append(in_label.cpu().numpy())
                # all_feats.append(feat.cpu().numpy())

                accs.append(acc.detach().cpu().numpy())
                self.logger.log("Testing", "Accuracy: %.1f, Loss: %.2f!"%(100*acc.item(), loss.item()))
                # if self.attention_model:
                #     self.logger.log("Testing", "Rot Acc: %.1f, Rot Loss: %.2f!"%(100*r_acc.item(), r_loss.item()))

            accs = np.array(accs, dtype=np.float32)

            self.logger.log('Testing', 'Average accuracy is %.2f!!!!'%(100*accs.mean()))
            self.test_accs.append(100*accs.mean())
            best_acc = np.array(self.test_accs).max()
            self.logger.log('Testing', 'Best accuracy so far is %.2f!!!!'%(best_acc))

            # self.logger.log("Testing", 'Here to peek at the lmc') # we should infer pose information?
            # self.logger.log("Testing", str(lmc))
            # import ipdb; ipdb.set_trace()
            # n = 1
            # mAP = modelnet_retrieval_mAP(all_feats,all_labels,n)
            # self.logger.log('Testing', 'Mean average precision at %d is %f!!!!'%(n, mAP))

        self.model.train()
        self.metric.train()

