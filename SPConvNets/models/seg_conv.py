import torch
import torch.nn as nn
from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet
from scipy.optimize import linear_sum_assignment
# from .point_convolution_universal import LocalConvNet, EdgeConv
from .DGCNN import PrimitiveNet

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PrimitiveFittingNet(nn.Module):
    def __init__(self, args=None):
        super(PrimitiveFittingNet, self).__init__()

        ''' SET arguments '''

        args.input_normal = False
        args.backbone = 'DGCNN'
        args.dgcnn_layers = 3
        args.pred_nmasks = 3

        self.args = args
        self.args.radius = args.radius = "0.2,0.2,0.4" # effect of this parameter?


        self.xyz_dim = 3
        self.mask_dim = 3 # prediction mask dim is set to 3
        # representation dimension

        self.use_normal_loss = False
        self.cls_backbone = True
        self.with_normal = False
        radius = self.args.radius.split(",")
        self.radius = [float(rr) for rr in radius]
        ''' SET arguments '''

        ''' 6-layer feature extraction backbone '''
        args.dgcnn_out_dim = 128
        map_feat_dim = 128
        args.dgcnn_in_feat_dim = 3
        self.in_feat_dim  =  args.in_feat_dim = 3
        print("Using DGCNN")
        self.conv_uniunet = PrimitiveNet(args)
        ''' 6-layer feature extraction backbone '''

        nmasks = self.args.pred_nmasks
        self.cls_layers = construct_conv1d_modules(
            [map_feat_dim, map_feat_dim, nmasks], n_in=map_feat_dim, last_act=False, bn=False
        )

    def forward(
            self, pos: torch.FloatTensor,
            feats: {},
        ):
        pos = pos.float()
        # bz, N = pos.size(0), pos.size(1)
        # nsmp = 256
        # print(pos.size())

        pos = pos.contiguous().transpose(1, 2).contiguous()

        fps_idx = None

        pos = pos.contiguous()

        statistics = {}

        ''' GET features '''
        # feat = []
        # for k in feats:
        #     feat.append(feats[k])
        # feat = torch.cat(feat, dim=-1) if len(feat) > 0 else None
        ''' GET features '''

        ''' GET feature embeddings '''
        # infeat = torch.cat([pos, feats["normals"]], dim=-1)
        infeat = pos

        x, type_per_point, normal_per_point = self.conv_uniunet(xyz=infeat, normal=infeat, inds=None, postprocess=False)
        # statistics['type_per_point'] = type_per_point
        # statistics['normal_per_point'] = normal_per_point
        ''' GET feature embeddings '''

        # losses = []
        # sims = []
        # gt_l = torch.zeros((1, ), dtype=torch.float32, device=pos.device)
        #
        # confpred = None

        # statistics['x'] = x
        ''' top-down segmentation '''
        segpred = CorrFlowPredNet.apply_module_with_conv1d_bn(
            x, self.cls_layers
        )
        # segpred = torch.clamp(segpred, min=-20, max=20)
        # segpred = torch.softmax(segpred, dim=-1)
        #
        # segpred = segpred.transpose(1, 2).contiguous()
        ''' top-down segmentation '''

        # masks_dim = masks.size(-1)
        # todo: test the effectiveness of aligning with masks
        # if masks_dim > segpred.size(1):
        #     segpred = torch.cat(
        #         [segpred, torch.zeros((bz, masks_dim - segpred.size(1), N), dtype=torch.float32, device=pos.device)],
        #         dim=1
        #     )
        #     if self.args.with_conf_loss:
        #         confpred = torch.cat(
        #             [confpred, torch.zeros((bz, masks_dim - confpred.size(1)), dtype=torch.float32, device=pos.device)],
        #             dim=-1
        #         )

        return segpred, None


def build_model(opt):
    cur_model = PrimitiveFittingNet(opt).cuda()
    return cur_model

def build_model_from(opt, outfile_path=None):
    return build_model(opt)