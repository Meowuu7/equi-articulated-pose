import torch
try:
    from torch_cluster import fps
except:
    pass

import torch.nn as nn
import numpy as np

try:
    import open3d as o3d
except:
    pass

import os

from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
from SPConvNets.models.common_utils import *

def check_and_make_dir(dir_fn):
    if not os.path.exists(dir_fn):
        os.mkdir(dir_fn)

def set_bn_not_training(module):
    if isinstance(module, nn.ModuleList):
        for block in module:
            set_bn_not_training(block)
    elif isinstance(module, nn.Sequential):
        for block in module:
            if isinstance(block, nn.BatchNorm1d) or isinstance(block, nn.BatchNorm2d):
                block.is_training = False
    else:
        raise ValueError("Not recognized module to set not training!")

def set_grad_to_none(module):
    if isinstance(module, nn.ModuleList):
        for block in module:
            set_grad_to_none(block)
    elif isinstance(module, nn.Sequential):
        for block in module:
            for param in block.parameters():
                param.grad = None
    else:
        raise ValueError("Not recognized module to set not training!")


def init_weight(blocks):
    for module in blocks:
        if isinstance(module, nn.Sequential):
            for subm in module:
                if isinstance(subm, nn.Linear):
                    nn.init.xavier_uniform_(subm.weight)
                    nn.init.zeros_(subm.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


def construct_conv1d_modules(mlp_dims, n_in, last_act=True, bn=True, others_bn=True):
    rt_module_list = nn.ModuleList()
    for i, dim in enumerate(mlp_dims):
        inc, ouc = n_in if i == 0 else mlp_dims[i-1], dim
        if i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act):
            # if others_bn and ouc % 4 == 0:
            if others_bn: # and ouc % 4 == 0:
                blk = nn.Sequential(
                        nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                        nn.BatchNorm1d(num_features=ouc, eps=1e-5, momentum=0.1),
                    # nn.GroupNorm(num_groups=4, num_channels=ouc),
                        nn.ReLU()
                    )
            else:
                blk = nn.Sequential(
                    nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                    nn.ReLU()
                )
        # elif bn  and ouc % 4 == 0:
        elif bn: #  and ouc % 4 == 0:
            blk = nn.Sequential(
                nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                nn.BatchNorm1d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
            )
        else:
            blk = nn.Sequential(
                nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
            )
        rt_module_list.append(blk)
    init_weight(rt_module_list)
    return rt_module_list


def construct_conv_modules(mlp_dims, n_in, last_act=True, bn=True):
    rt_module_list = nn.ModuleList()
    for i, dim in enumerate(mlp_dims):
        inc, ouc = n_in if i == 0 else mlp_dims[i-1], dim
        # if (i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act))  and ouc % 4 == 0:
        if (i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act)): #  and ouc % 4 == 0:
            blk = nn.Sequential(
                    nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
                    nn.BatchNorm2d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
                    nn.ReLU()
                )
        # elif bn  and ouc % 4 == 0:
        elif bn: #  and ouc % 4 == 0:
            blk = nn.Sequential(
                nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
                nn.BatchNorm2d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
            )
        else:
            blk = nn.Sequential(
                nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
            )
        rt_module_list.append(blk)
    init_weight(rt_module_list)
    return rt_module_list


class CorrFlowPredNet(nn.Module):
    def __init__(self, corr_feat_dim: int=32):
        super(CorrFlowPredNet, self).__init__()

    @staticmethod
    def apply_module_with_conv2d_bn(x, module):
        x = x.transpose(2, 3).contiguous().transpose(1, 2).contiguous()
        # print(x.size())
        for layer in module:
            for sublayer in layer:
                x = sublayer(x.contiguous())
            x = x.float()
        x = torch.transpose(x, 1, 2).transpose(2, 3)
        return x

    @staticmethod
    def apply_module_with_conv1d_bn(x, module):
        x = x.transpose(1, 2).contiguous()
        # print(x.size())
        for layer in module:
            for sublayer in layer:
                x = sublayer(x.contiguous())
            x = x.float()
        x = torch.transpose(x, 1, 2)
        return x

    # @staticmethod
def apply_module_with_conv2d_bn(x, module):
    x = x.transpose(2, 3).contiguous().transpose(1, 2).contiguous()
    # print(x.size())
    for layer in module:
        for sublayer in layer:
            x = sublayer(x.contiguous())
        x = x.float()
    x = torch.transpose(x, 1, 2).transpose(2, 3)
    return x

# @staticmethod
def apply_module_with_conv1d_bn(x, module):
    x = x.transpose(1, 2).contiguous()
    # print(x.size())
    for layer in module:
        for sublayer in layer:
            x = sublayer(x.contiguous())
        x = x.float()
    x = torch.transpose(x, 1, 2)
    return x

def estimate_normals(pos):
    # pos.size = bz x N x 3
    normals = []
    for i in range(pos.size(0)):
        pts = pos[i].detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        nms = np.array(pcd.normals)
        normals.append(torch.from_numpy(nms).to(pos.device).float().unsqueeze(0))
    normals = torch.cat(normals, dim=0)
    return normals


def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    feat_dim = pos.size(-1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()

    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.contiguous().view(-1, feat_dim).contiguous() # (bz x N, 3)
    # sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=False)
    # shape of sampled_idx?
    return sampled_idx

def initialize_model_modules(modules):
    for zz in modules:
        if isinstance(zz, nn.Sequential):
            for zzz in zz:
                if isinstance(zzz, nn.Conv2d):
                    torch.nn.init.xavier_uniform_(zzz.weight)
                    if zzz.bias is not None:
                        torch.nn.init.zeros_(zzz.bias)
        elif isinstance(zz, nn.Conv2d):
            torch.nn.init.xavier_uniform_(zz.weight)
            if zz.bias is not None:
                torch.nn.init.zeros_(zz.bias)

class PartDecoder(nn.Module):

    def __init__(self, feat_len, recon_M=128):
        super(PartDecoder, self).__init__()

        self.mlp1 = nn.Linear(feat_len + 3, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, 3)

        self.recon_M = recon_M

    def forward(self, feat, pc):
        num_point = pc.shape[0]
        batch_size = feat.shape[0]

        net = torch.cat([feat.unsqueeze(dim=1).repeat(1, num_point, 1), \
                         pc.unsqueeze(dim=0).repeat(batch_size, 1, 1)], dim=-1)

        net = F.leaky_relu(self.mlp1(net))
        net = F.leaky_relu(self.mlp2(net))
        net = self.mlp3(net)

        if self.recon_M < num_point:
            fps_idx = farthest_point_sampling(net, self.recon_M)
            net = net.contiguous().view(
                batch_size * num_point, -1)[fps_idx, :].contiguous().view(batch_size, self.recon_M, -1)

        return net

class DecoderFCAxis(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, bn=False):
        super(DecoderFCAxis, self).__init__()
        self.n_features = list(n_features)
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf) # Linear layer
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 3)
        model.append(fc_layer)
        # add by XL
        # acti_layer = nn.Sigmoid()
        # model.append(acti_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        for module in self.model:
            if isinstance(module, nn.BatchNorm1d):
                x = module(x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                x = module(x)

        # x: bz x 3 -> bz x 3 (normalized axis)
        x = x / torch.clamp(torch.norm(x, dim=-1, keepdim=True, p=2), min=1e-8)
        x = x.sum(0)
        x = x / torch.clamp(torch.norm(x, dim=-1, keepdim=True, p=2), min=1e-8)

        # x = self.model(x)
        # x = x.view((-1, 3, self.output_pts))
        return x


class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, use_sigmoid=True):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.use_sigmoid = use_sigmoid

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf) # Linear layer
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)
        # add by XL
        if self.use_sigmoid:
            acti_layer = nn.Sigmoid()
            model.append(acti_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x): # bz x dim --> what does x represent?
        for module in self.model:
            if isinstance(module, nn.BatchNorm1d):
                x = module(x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                x = module(x)

        # x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        return x


class DecoderFCWithPVP(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, with_conf=False):
        super(DecoderFCWithPVP, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.with_conf = with_conf

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf) # Linear layer
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model.append(acti_layer)
        self.model = nn.Sequential(*model)

        model_ppv = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            model_ppv.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                model_ppv.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model_ppv.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 6)
        model_ppv.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model_ppv.append(acti_layer)
        self.model_ppv = nn.Sequential(*model_ppv)

        ''' Construt conf prediction block '''
        model_conf = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            model_conf.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                model_conf.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model_conf.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 6)
        model_conf.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model_conf.append(acti_layer)
        self.model_conf = nn.Sequential(*model_conf)

    def forward(self, x, pv_point_inv_feat=None, central_point_inv_feat=None):
        ppv = x.clone() if pv_point_inv_feat is None else pv_point_inv_feat
        conf = x.clone() if central_point_inv_feat is None else central_point_inv_feat
        for module in self.model:
            if isinstance(module, nn.BatchNorm1d):
                x = module(x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                x = module(x)
        # x = self.model(x) #
        x = x.view((-1, 3, self.output_pts))

        ''' Predict pivot point '''
        for module in self.model_ppv:
            if isinstance(module, nn.BatchNorm1d):
                ppv = module(ppv.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                ppv = module(ppv)
        ppv = ppv.contiguous().view((-1, 6)).contiguous()
        pivot_point, central_point = ppv[:, :3], ppv[:, 3:]

        if self.with_conf:
            ''' Predict pivot point '''
            for module in self.model_conf:
                if isinstance(module, nn.BatchNorm1d):
                    conf = module(conf.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
                else:
                    conf = module(conf)
            conf = conf.contiguous().view((-1, 1)).contiguous()
        if self.with_conf:
            return x, pivot_point, central_point, conf
        else:
            return x, pivot_point, central_point


class DecoderFCWithPVPAtlas(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, with_conf=False, prior_dim=3):
        super(DecoderFCWithPVPAtlas, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.with_conf = with_conf

        self.atlas_prior_dim = prior_dim  # latent-dim + prior-dim = decoder input dim

        self.path = torch.nn.Parameter(torch.FloatTensor(self.atlas_prior_dim, self.output_pts, ),
                                       requires_grad=True)
        # self.path.data.uniform_(0, 1)
        self.path.data.uniform_(-0.5, 0.5)

        model = []
        prev_nf = self.latent_dim + self.atlas_prior_dim

        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers; n_features
            fc_layer = nn.Conv2d(in_channels=prev_nf, out_channels=nf, kernel_size=(1, 1),
                              stride=(1, 1), bias=True)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm2d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Conv2d(in_channels=n_features[-1], out_channels=3, kernel_size=(1, 1),
                              stride=(1, 1), bias=True)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid() # activate layer
        model.append(acti_layer)
        self.model = nn.Sequential(*model)

        # # model = []
        # # prev_nf = self.latent_dim
        # #### Construct points prediction model ####
        # for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
        #     fc_layer = nn.Linear(prev_nf, nf) # Linear layer
        #     model.append(fc_layer)
        #
        #     if bn:
        #         bn_layer = nn.BatchNorm1d(nf) # batch norm
        #         model.append(bn_layer)
        #
        #     act_layer = nn.LeakyReLU(inplace=True)
        #     model.append(act_layer)
        #     prev_nf = nf
        #
        # fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        # model.append(fc_layer)
        # # add by XL
        # acti_layer = nn.Sigmoid()
        # model.append(acti_layer)
        # self.model = nn.Sequential(*model)

        model_ppv = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            model_ppv.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                model_ppv.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model_ppv.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 6)
        model_ppv.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model_ppv.append(acti_layer)
        self.model_ppv = nn.Sequential(*model_ppv)

        ''' Construt conf prediction block '''
        model_conf = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            model_conf.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                model_conf.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model_conf.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 6)
        model_conf.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model_conf.append(acti_layer)
        self.model_conf = nn.Sequential(*model_conf)

    def forward(self, x, pv_point_inv_feat=None, central_point_inv_feat=None):
        # ppv = x.clone()
        # conf = x.clone()

        ppv = x.clone() if pv_point_inv_feat is None else pv_point_inv_feat
        conf = x.clone() if central_point_inv_feat is None else central_point_inv_feat

        # for module in self.model:
        #     if isinstance(module, nn.BatchNorm1d):
        #         x = module(x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
        #     else:
        #         x = module(x)
        # # x = self.model(x) #
        # x = x.view((-1, 3, self.output_pts))

        bz = x.size(0) # batch size
        # decoder_in_feats: bz x (prior_dim + 3) x npts
        decoder_in_feats = torch.cat( # input features for the decorder
            [x.unsqueeze(-1).repeat(1, 1, self.output_pts), self.path.unsqueeze(0).repeat(bz, 1, 1)], dim=1)
        # x: bz x 3 x npts x 1; x = self.model...
        x = self.model(decoder_in_feats.unsqueeze(-1))
        x = x.squeeze(-1)

        ''' Predict pivot point '''
        for module in self.model_ppv:
            if isinstance(module, nn.BatchNorm1d):
                ppv = module(ppv.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                ppv = module(ppv)
        ppv = ppv.contiguous().view((-1, 6)).contiguous()
        pivot_point, central_point = ppv[:, :3], ppv[:, 3:]

        if self.with_conf:
            ''' Predict pivot point '''
            for module in self.model_conf:
                if isinstance(module, nn.BatchNorm1d):
                    conf = module(conf.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
                else:
                    conf = module(conf)
            conf = conf.contiguous().view((-1, 1)).contiguous()
        if self.with_conf:
            return x, pivot_point, central_point, conf
        else:
            return x, pivot_point, central_point


class DecoderFCWithPVPAxis(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False):
        super(DecoderFCWithPVPAxis, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf) # Linear layer
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model.append(acti_layer)
        self.model = nn.Sequential(*model)

        # pivot-point: 3-dim
        model_ppv = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            model_ppv.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                model_ppv.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model_ppv.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 6)
        model_ppv.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model_ppv.append(acti_layer)
        # Set the model for pivot point prediction
        self.model_ppv = nn.Sequential(*model_ppv)

    def forward(self, x):
        ppv = x.clone()
        for module in self.model:
            if isinstance(module, nn.BatchNorm1d):
                x = module(x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                x = module(x)

        # x = self.model(x) #
        x = x.view((-1, 3, self.output_pts))

        for module in self.model_ppv:
            if isinstance(module, nn.BatchNorm1d):
                ppv = module(ppv.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                ppv = module(ppv)
        ppv = ppv.contiguous().view((-1, 6)).contiguous()
        pivot_point, central_point = ppv[:, :3], ppv[:, 3:]

        return x, pivot_point, central_point


class DecoderFCAtlas(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, prior_dim=3):
        super(DecoderFCAtlas, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts # number of output points
        self.latent_dim = latent_dim # latent vector dimension...
        self.atlas_prior_dim = prior_dim # latent-dim + prior-dim = decoder input dim

        self.path = torch.nn.Parameter(torch.FloatTensor(self.atlas_prior_dim, self.output_pts, ),
                                       requires_grad=True)
        self.path.data.uniform_(0, 1)
        # self.path.data.uniform_(-0.5, 0.5)

        model = []
        prev_nf = self.latent_dim + self.atlas_prior_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers; n_features
            fc_layer = nn.Conv2d(in_channels=prev_nf, out_channels=nf, kernel_size=(1, 1),
                              stride=(1, 1), bias=True)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm2d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Conv2d(in_channels=n_features[-1], out_channels=3, kernel_size=(1, 1),
                              stride=(1, 1), bias=True)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid() # activate layer
        model.append(acti_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        # x: bz x dim --> inv feature for the slot
        # patch: prior_dim x npts
        bz = x.size(0)
        # decoder_in_feats: bz x (prior_dim + 3) x npts
        decoder_in_feats = torch.cat([x.unsqueeze(-1).repeat(1, 1, self.output_pts), self.path.unsqueeze(0).repeat(bz, 1, 1)], dim=1)
        # x: bz x 3 x npts x 1
        x = self.model(decoder_in_feats.unsqueeze(-1))
        x = x.squeeze(-1)

        return x

class DecoderFCWithPVPConstantCommon(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, with_conf=False, prior_dim=3):
        super(DecoderFCWithPVPConstantCommon, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.with_conf = with_conf

        model = []
        prev_nf = self.latent_dim
        common_layers = [1024, 1024]
        for idx, nf in enumerate(common_layers):  # n_features is used for constructing layers
            fc_layer = nn.Conv2d(in_channels=prev_nf, out_channels=nf, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm2d(nf)  # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Conv2d(in_channels=common_layers[-1], out_channels=3 * output_pts, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model.append(acti_layer)

        self.common_model = nn.Sequential(*model)

        model = []
        prev_nf = self.latent_dim + 3
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Conv2d(in_channels=prev_nf, out_channels=nf, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm2d(nf)  # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Conv2d(in_channels=n_features[-1], out_channels=3, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model.append(acti_layer)
        self.flow_model = nn.Sequential(*model)


        # # model = []
        # # prev_nf = self.latent_dim
        # #### Construct points prediction model ####
        # for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
        #     fc_layer = nn.Linear(prev_nf, nf) # Linear layer
        #     model.append(fc_layer)
        #
        #     if bn:
        #         bn_layer = nn.BatchNorm1d(nf) # batch norm
        #         model.append(bn_layer)
        #
        #     act_layer = nn.LeakyReLU(inplace=True)
        #     model.append(act_layer)
        #     prev_nf = nf
        #
        # fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        # model.append(fc_layer)
        # # add by XL
        # acti_layer = nn.Sigmoid()
        # model.append(acti_layer)
        # self.model = nn.Sequential(*model)

        model_ppv = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            model_ppv.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                model_ppv.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model_ppv.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 6)
        model_ppv.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model_ppv.append(acti_layer)
        self.model_ppv = nn.Sequential(*model_ppv)

        ''' Construt conf prediction block '''
        model_conf = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            model_conf.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                model_conf.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model_conf.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 6)
        model_conf.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model_conf.append(acti_layer)
        self.model_conf = nn.Sequential(*model_conf)

    def forward(self, x, ):
        ppv = x.clone()
        conf = x.clone()

        # for module in self.model:
        #     if isinstance(module, nn.BatchNorm1d):
        #         x = module(x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
        #     else:
        #         x = module(x)
        # # x = self.model(x) #
        # x = x.view((-1, 3, self.output_pts))

        bz = x.size(0)
        # cur_slot_prior: dim --> bz x dim x 1 x 1
        cur_slot_prior = torch.ones((self.latent_dim), dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1).repeat(bz, 1, 1, 1)
        # category_pts: bz x (3 * n_pts) x 1 x 1
        category_pts = self.common_model(cur_slot_prior).squeeze(-1).squeeze(-1)
        # category_pts: bz x 3 x n_pts
        category_pts = category_pts.contiguous().view(bz, 3, self.output_pts).contiguous() - 0.5
        expanded_x_feats = x.unsqueeze(-1).unsqueeze(-1).contiguous().repeat(1, 1, self.output_pts, 1).contiguous()
        cat_feats = torch.cat([expanded_x_feats, category_pts.unsqueeze(-1)], dim=1).contiguous()
        # pts_flow: bz x 3 x n_pts x 1
        pts_flow = self.flow_model(cat_feats).squeeze(-1)
        # pts_flow = 0.2 * (pts_flow - 0.5)
        pts_flow = 0.1 * (pts_flow - 0.5)
        # pts_flow = 0.05 * (pts_flow - 0.5)
        # pts_flow = 0.0 * (pts_flow - 0.5)
        res_pts = category_pts + pts_flow
        res_pts = res_pts + 0.5
        x = res_pts.squeeze(-1)

        ''' Predict pivot point '''
        for module in self.model_ppv:
            if isinstance(module, nn.BatchNorm1d):
                ppv = module(ppv.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                ppv = module(ppv)
        ppv = ppv.contiguous().view((-1, 6)).contiguous()
        pivot_point, central_point = ppv[:, :3], ppv[:, 3:]

        if self.with_conf:
            ''' Predict pivot point '''
            for module in self.model_conf:
                if isinstance(module, nn.BatchNorm1d):
                    conf = module(conf.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
                else:
                    conf = module(conf)
            conf = conf.contiguous().view((-1, 1)).contiguous()
        if self.with_conf:
            return x, pivot_point, central_point, conf
        else:
            return x, pivot_point, central_point


class DecoderConstantCommon(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, use_sigmoid=True):
        super(DecoderConstantCommon, self).__init__()

        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.use_sigmoid = use_sigmoid

        model = []
        prev_nf = self.latent_dim
        common_layers = [1024, 1024]
        for idx, nf in enumerate(common_layers):  # n_features is used for constructing layers
            fc_layer = nn.Conv2d(in_channels=prev_nf, out_channels=nf, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm2d(nf)  # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Conv2d(in_channels=common_layers[-1], out_channels=3 * output_pts, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)
        model.append(fc_layer)
        # add by XL
        if self.use_sigmoid:
            acti_layer = nn.Sigmoid()
            model.append(acti_layer)

        self.common_model = nn.Sequential(*model)

        model = []
        prev_nf = self.latent_dim + 3
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Conv2d(in_channels=prev_nf, out_channels=nf, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm2d(nf)  # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Conv2d(in_channels=n_features[-1], out_channels=3, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model.append(acti_layer)
        self.flow_model = nn.Sequential(*model)

    def forward(self, x):
        # print("here1")
        # x: bz x dim --> inv feature for the slot
        bz = x.size(0)
        # cur_slot_prior: dim --> bz x dim x 1 x 1
        cur_slot_prior = torch.ones((self.latent_dim), dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(bz, 1, 1, 1)
        # category_pts: bz x (3 * n_pts) x 1 x 1
        category_pts = self.common_model(cur_slot_prior).squeeze(-1).squeeze(-1)
        # category_pts: bz x 3 x n_pts
        if self.use_sigmoid:
            category_pts = category_pts.contiguous().view(bz, 3, self.output_pts).contiguous() - 0.5

        expanded_x_feats = x.unsqueeze(-1).unsqueeze(-1).contiguous().repeat(1, 1, self.output_pts, 1).contiguous()
        cat_feats = torch.cat([expanded_x_feats, category_pts.unsqueeze(-1)], dim=1).contiguous()
        # pts_flow: bz x 3 x n_pts x 1
        pts_flow = self.flow_model(cat_feats).squeeze(-1)
        # pts_flow = 0.2 * (pts_flow - 0.5)
        # pts_flow = 0.1 * (pts_flow - 0.5)

        pts_flow = 0.0 * (pts_flow - 0.5)
        res_pts = category_pts + pts_flow
        return res_pts


class DecoderFCWithCuboic(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, pred_rot=False):
        super(DecoderFCWithCuboic, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.pred_rot = pred_rot

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf) # Linear layer
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model.append(acti_layer)
        self.model = nn.Sequential(*model)
        ''' End of the construction for MLP points deocder '''

        # from x to other scaling and rotation related features

        ''' Construct cuboic related model '''
        # scaling factor? ---> how to predict a scaling factor? [-0.5, 0.5], then the factor lies in...
        cuboic_model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf) # Linear layer
            cuboic_model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf) # batch norm
                cuboic_model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            cuboic_model.append(act_layer)
            prev_nf = nf

        if self.pred_rot:
            fc_layer = nn.Linear(self.n_features[-1], 7)
        else:
            fc_layer = nn.Linear(self.n_features[-1], 3)
        # fc_layer = nn.Linear(self.n_features[-1], 7)
        # fc_layer = nn.Linear(self.n_features[-1], 3)

        cuboic_model.append(fc_layer)

        if not self.pred_rot:
            acti_layer = nn.Sigmoid()
            cuboic_model.append(acti_layer)
        self.cuboic_model = nn.Sequential(*cuboic_model)
        ''' End of the construction of cuboic related scaling and rotation model '''


    def forward(self, x): # we should panelize maximum distance, outlier most...
        '''
            x: bz x dim --> feature for points decoding
        '''

        bz = x.size(0)

        # Save the cuboid x for future use
        cuboic_x = x.clone()

        ''' Decode points '''
        for module in self.model:
            if isinstance(module, nn.BatchNorm1d) and len(x.size()) > 2:
                x = module(x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                x = module(x)

        x = x.view((-1, 3, self.output_pts))

        for module in self.cuboic_model:
            if isinstance(module, nn.BatchNorm1d) and len(cuboic_x.size()) > 2:
                cuboic_x = module(cuboic_x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                cuboic_x = module(cuboic_x)
        # # cuboic_x: bz x 7
        # cuboic_scaling, cuboic_quat = cuboic_x[..., :3], cuboic_x[..., 3:]
        # cuboic_scaling = torch.sigmoid(cuboic_scaling)
        # # rot points?
        # # eight points?
        # dist2: bz x 3 x N
        # print(cuboic_x.size(), x.size())
        if self.pred_rot:
            cuboic_x, cuboic_quat = cuboic_x[:, :3], cuboic_x[:, 3:]
            # Get predicted cuboic_x's range
            cuboic_x = torch.sigmoid(cuboic_x)
            # cuboic_R: bz x 3 x 3
            cuboic_R = compute_rotation_matrix_from_quaternion(cuboic_quat)
        else:
            cuboic_R = torch.eye(3, dtype=torch.float32).cuda().unsqueeze(0).contiguous().repeat(bz, 1, 1)
        dist2 = torch.abs(cuboic_x.unsqueeze(-1) - x)
        # tot_dist: bz x 3 x N x 2
        tot_dist = torch.cat([x.unsqueeze(-1), dist2.unsqueeze(-1)], dim=-1)
        # should mask out inlier xyzs?
        # mask out inlier xyzs?
        # inlier_xyzs = (0. <= x <= cuboic_x.unsqueeze(-1)).float()
        inlier_xyzs = (x <= cuboic_x.unsqueeze(-1)).float()
        inlier_pts_indicators = (inlier_xyzs.sum(1) > 2.5).float()
        tot_dist_outliers = tot_dist.clone()
        tot_dist_outliers[inlier_xyzs.unsqueeze(-1).repeat(1, 1, 1, 2) > 0.5] = 0.0 # we should mask out inlier xyzs for outlier distance computing
        # distance from points to cuboics for outlier points
        outlier_dist, _ = torch.max(tot_dist_outliers, dim=-1)
        outlier_dist, _ = torch.max(outlier_dist, dim=1)

        inlier_dist, _ = torch.min(tot_dist, dim=-1)
        inlier_dist, _ = torch.min(inlier_dist, dim=1)

        minn_tot_dist = inlier_dist * inlier_pts_indicators + (1. - inlier_pts_indicators) * outlier_dist
        minn_tot_dist = torch.mean(minn_tot_dist, dim=-1)

        # # minn_tot_dist: bz x 3 x N
        # minn_tot_dist, _ = torch.min(tot_dist, dim=-1)
        # # minn_tot_dist: bz x N
        # minn_tot_dist, _ = torch.min(minn_tot_dist, dim=1)
        # ''' Reconstruction loss for cuboic fitting '''
        # # minn_tot_dist: bz
        # minn_tot_dist = torch.mean(minn_tot_dist, dim=-1)
        #
        # x = x.view((-1, 3, self.output_pts))
        # for each slot, we get its

        return x, minn_tot_dist, cuboic_x, cuboic_R


class DecoderFCWithPVPCuboic(nn.Module): # pivot
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False, pred_rot=False):
        super(DecoderFCWithPVPCuboic, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.pred_rot = pred_rot

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features): # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf) # Linear layer
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf) # batch norm
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model.append(acti_layer)
        self.model = nn.Sequential(*model)

        ''' Set pivot point prediction module '''
        model_ppv = [] # model for pivot point prediction...
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            model_ppv.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                model_ppv.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model_ppv.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], 6)
        model_ppv.append(fc_layer)
        # add by XL
        acti_layer = nn.Sigmoid()
        model_ppv.append(acti_layer)
        self.model_ppv = nn.Sequential(*model_ppv)
        ''' Set pivot point prediction module '''

        ''' Construct cuboic related model '''
        # scaling factor? ---> how to predict a scaling factor? [-0.5, 0.5], then the factor lies in...
        cuboic_model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):  # n_features is used for constructing layers
            fc_layer = nn.Linear(prev_nf, nf)  # Linear layer
            cuboic_model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)  # batch norm
                cuboic_model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            cuboic_model.append(act_layer)
            prev_nf = nf

        if self.pred_rot:
            fc_layer = nn.Linear(self.n_features[-1], 7)
        else:
            fc_layer = nn.Linear(self.n_features[-1], 3)

        cuboic_model.append(fc_layer)

        if not self.pred_rot:
            acti_layer = nn.Sigmoid()
            cuboic_model.append(acti_layer)
        self.cuboic_model = nn.Sequential(*cuboic_model)
        ''' End of the construction of cuboic related scaling and rotation model '''



    def forward(self, x):

        bz = x.size(0)
        cuboic_x = x.clone()
        ppv = x.clone()
        for module in self.model:
            if isinstance(module, nn.BatchNorm1d):
                x = module(x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                x = module(x)

        # x = self.model(x) #
        x = x.view((-1, 3, self.output_pts))

        ''' Predict pivot point '''
        for module in self.model_ppv:
            if isinstance(module, nn.BatchNorm1d):
                ppv = module(ppv.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                ppv = module(ppv)
        ppv = ppv.contiguous().view((-1, 6)).contiguous()
        pivot_point, central_point = ppv[:, :3], ppv[:, 3:]
        ''' Predict pivot point '''

        ''' Predict cuboid related parameters '''
        for module in self.cuboic_model:
            if isinstance(module, nn.BatchNorm1d) and len(cuboic_x.size()) > 2:
                cuboic_x = module(cuboic_x.contiguous().transpose(1, 2).contiguous()).contiguous().transpose(1, 2).contiguous()
            else:
                cuboic_x = module(cuboic_x)
        if self.pred_rot:
            cuboic_x, cuboic_quat = cuboic_x[:, :3], cuboic_x[:, 3:]
            # Get predicted cuboic_x's range
            cuboic_x = torch.sigmoid(cuboic_x)
            # cuboic_R: bz x 3 x 3
            cuboic_R = compute_rotation_matrix_from_quaternion(cuboic_quat)
        else:
            cuboic_R = torch.eye(3, dtype=torch.float32).cuda().unsqueeze(0).contiguous().repeat(bz, 1, 1)
        ''' Predict cuboid related parameters '''

        # Get pivot point, central point, cuboid coordinates and rotation matrix...
        return x, pivot_point, central_point, cuboic_x, cuboic_R


def get_surface_points(k=6):
    '''
        Get surface points from [0, 1] x-y, y-z, x-z planes
    '''
    rng = 1. / k
    pts_x = [_ * rng for _ in range(k)]
    pts_x = torch.tensor(pts_x, dtype=torch.float32).cuda()
    pts_y = [_ * rng for _ in range(k)]
    pts_xy = []
    for i_x, cur_x in enumerate(pts_x):
        for i_y, cur_y in enumerate(pts_y):
            pts_xy.append([cur_x - 0.5, cur_y - 0.5])
    pts_xy = torch.tensor(pts_xy, dtype=torch.float32).cuda()
    tot_pts_n = pts_xy.size(0)
    # pts_xy_xy: tot_pts_n x 3
    zero_paddings = torch.zeros((tot_pts_n, 1), dtype=torch.float32).cuda()
    pts_xy_xy = torch.cat([pts_xy, zero_paddings], dim=-1)
    pts_xy_xz = torch.cat([pts_xy[:, 0].unsqueeze(-1), zero_paddings, pts_xy[:, 1].unsqueeze(-1)], dim=-1)
    pts_xy_yz = torch.cat([zero_paddings, pts_xy], dim=-1)
    return pts_xy_xy, pts_xy_xz, pts_xy_yz


def get_cuboic_constraint_loss(pred_R, pred_T, ori_pts, slot_recon_cuboic, slot_cuboic_R, hard_one_hot_labels, attn_ori, forb_slot_idx=None):
    '''
        pred_R: bz x n_s x 3 x 3
        pred_T: bz x n_s x 3 # p
        slot_cuboic_R: bz x n_s x 3 x 3
        ori_pts: bz x 3 x N
        slot_recon_cuboic: bz x n_s x 3 --> we assume that the reconstruction should not be rotated now...
        hard_one_hot_labels: bz x n_s x N
    '''
    # inv_trans_pts: bz x n_s x 3 x N
    # inv_trans_pts should be constrainted into the box with min_x, min_y, min_z = -s_x / 2., -s_y / 2., -s_z / 2.; and max_x, max_y, max_z = s_x / 2., s_y / 2., s_z / 2.
    # Inversed transformed points
    inv_trans_pts = torch.matmul(safe_transpose(pred_R, -1, -2), ori_pts.unsqueeze(1) - pred_T.unsqueeze(-1))
    # transform points
    # inv_trans_pts = inv_trans_pts + 0.5
    # Get inv_trans_pts: inv_cuboic_R^T inv_trans_pts
    # inv_trans_pts: bz x n_s x 3 x N
    inv_trans_pts = torch.matmul(safe_transpose(slot_cuboic_R, -1, -2), inv_trans_pts)

    # slot_recon_cuboic: bz x n_s x 3 # xyz
    minn_recon_cuboic = -1. * slot_recon_cuboic / 2.
    maxx_recon_cuboic = 1. * slot_recon_cuboic / 2.
    dist_1 = torch.abs((minn_recon_cuboic.unsqueeze(-1) - inv_trans_pts) ** 2)
    dist_2 = torch.abs((maxx_recon_cuboic.unsqueeze(-1) - inv_trans_pts) ** 2)

    # dist_1: bz x n_s x 3 x N; dist_2: bz x n_s x 3 x N

    # dist_1 = torch.abs(inv_trans_pts)
    # dist_2 = torch.abs(slot_recon_cuboic.unsqueeze(-1) - inv_trans_pts)

    # dists = torch.cat([dist_1.unsqueeze(2), dist_2.unsqueeze(2)], dim=2)
    dists = torch.cat([dist_1.unsqueeze(-1), dist_2.unsqueeze(-1)], dim=-1)

    ''' Inlier xyz '''
    # should mask out inlier xyzs?
    # mask out inlier xyzs?
    # inlier_xyzs = (0. <= x <= cuboic_x.unsqueeze(-1)).float()

    ###### Get inlier xyzs ######
    # inlier_xyzs = (inv_trans_pts <= slot_recon_cuboic.unsqueeze(-1)).float()
    # inv_trans_pts: bz x n_s x 3 x N
    inlier_xyzs = (inv_trans_pts <= (slot_recon_cuboic / 2.).unsqueeze(-1)).float() * (inv_trans_pts >= (-1.0 * slot_recon_cuboic / 2.).unsqueeze(-1)).float()
    ###### Get inlier xyzs ######
    # Get inlier points indicators: the indicator for whether the point is in the cuboid's range
    # inlier_pts_indicators: bz x n_s x N; inlier_xyzs: bz x n_s x 3 x N
    inlier_pts_indicators = (inlier_xyzs.sum(2) > 2.5).float()
    tot_dist_outliers = dists.clone() # inlie
    tot_dist_outliers[inlier_xyzs.unsqueeze(-1).repeat(1, 1, 1, 1, 2) > 0.5] = 0.0  # we should mask out inlier xyzs for outlier distance computing
    # distance from points to cuboics for outlier points
    outlier_dist, _ = torch.min(tot_dist_outliers, dim=-1) # min for the
    outlier_dist, _ = torch.max(outlier_dist, dim=2)

    inlier_dist, _ = torch.min(dists, dim=-1)

    # topk_inlier_dists: bz x n_s x 2 x N
    topk_inlier_dists, _ = torch.topk(inlier_dist, k=2, largest=False, dim=2)
    # topk_inlier_dists: bz x n_s x N
    inlier_dist = torch.mean(topk_inlier_dists, dim=2)

    # inlier_dist, _ = torch.min(inlier_dist, dim=2)

    minn_tot_dist = inlier_dist * inlier_pts_indicators + (1. - inlier_pts_indicators) * outlier_dist
    # minn_tot_dist = torch.mean(minn_tot_dist, dim=-1)
    minn_dists = minn_tot_dist

    ''' Previous: outliers not considered '''
    # # minn_dists: bz x n_s x 3 x N
    # minn_dists, _ = torch.min(dists, dim=2)
    # minn_dists, _ = torch.min(minn_dists, dim=2)
    ''' Previous: outliers not considered '''

    soft_weights = safe_transpose(hard_one_hot_labels, -1, -2) * attn_ori
    # cuboic_constraint_loss: bz x n_s
    # cuboic
    cuboic_constraint_loss = torch.sum(minn_dists * soft_weights, dim=-1) / torch.clamp(torch.sum(soft_weights, dim=-1), min=1e-8)
    hard_slot_indicators = (torch.sum(safe_transpose(hard_one_hot_labels, -1, -2), dim=-1) > 0.5).float()
    if forb_slot_idx is not None:
        hard_slot_indicators[:, forb_slot_idx] = 0.0
    cuboic_constraint_loss = torch.sum(cuboic_constraint_loss * hard_slot_indicators, dim=-1) / torch.sum(hard_slot_indicators, dim=-1)
    # print(f"cuboic_constraint_loss: {cuboic_constraint_loss.item()}, {inlier_pts_indicators.mean().item()}")
    return cuboic_constraint_loss


def get_cuboic_constraint_loss_cd_based(pred_R, pred_T, ori_pts, slot_recon_cuboic, slot_cuboic_R, hard_one_hot_labels, attn_ori):
    '''
        pred_R: bz x n_s x 3 x 3
        pred_T: bz x n_s x 3 # p
        slot_cuboic_R: bz x n_s x 3 x 3
        ori_pts: bz x 3 x N
        slot_recon_cuboic: bz x n_s x 3 --> we assume that the reconstruction should not be rotated now...
        hard_one_hot_labels: bz x n_s x N
    '''
    # inv_trans_pts: bz x n_s x 3 x N
    # inv_trans_pts should be constrainted into the box with min_x, min_y, min_z = -s_x / 2., -s_y / 2., -s_z / 2.; and max_x, max_y, max_z = s_x / 2., s_y / 2., s_z / 2.
    # Inversed transformed points
    inv_trans_pts = torch.matmul(safe_transpose(pred_R, -1, -2), ori_pts.unsqueeze(1) - pred_T.unsqueeze(-1))
    # transform points
    # inv_trans_pts = inv_trans_pts + 0.5
    # Get inv_trans_pts: inv_cuboic_R^T inv_trans_pts
    # inv_trans_pts: bz x n_s x 3 x N
    inv_trans_pts = torch.matmul(safe_transpose(slot_cuboic_R, -1, -2), inv_trans_pts)

    # slot_recon_cuboic: bz x n_s x 3
    # minn_recon_cuboic = -1. * slot_recon_cuboic / 2.
    # maxx_recon_cuboic = 1. * slot_recon_cuboic / 2.
    # dist_1 = torch.abs((minn_recon_cuboic.unsqueeze(-1) - inv_trans_pts) ** 2)
    # dist_2 = torch.abs((maxx_recon_cuboic.unsqueeze(-1) - inv_trans_pts) ** 2)

    slot_recon_cuboic_scc = torch.zeros((slot_recon_cuboic.size(0), slot_recon_cuboic.size(1), 3, 3), dtype=torch.float32).cuda()
    slot_recon_cuboic_scc[:, :, 0, 0] = slot_recon_cuboic[:, :, 0]
    slot_recon_cuboic_scc[:, :, 1, 1] = slot_recon_cuboic[:, :, 1]
    slot_recon_cuboic_scc[:, :, 2, 2] = slot_recon_cuboic[:, :, 2]

    # mult_cuboic_factor: xxxx x 3 x 3
    # mult_cuboic_factor = torch.ones((minn_recon_cuboic.size(0), minn_recon_cuboic.size(1), 3, 3),
    #                                     dtype=torch.float32).cuda()
    ones_padding = torch.ones((slot_recon_cuboic.size(0), slot_recon_cuboic.size(1), 1), dtype=torch.float32).cuda()
    mult_cuboic_xy = torch.cat([slot_recon_cuboic[:, :, :2], ones_padding], dim=-1)
    mult_cuboic_xz = torch.cat([slot_recon_cuboic[:, :, 0].unsqueeze(-1), ones_padding, slot_recon_cuboic[:, :, 2].unsqueeze(-1)], dim=-1)
    mult_cuboic_yz = torch.cat([ones_padding, slot_recon_cuboic[:, :, 1:]], dim=-1)

    # get a square containing k x k points
    # (k x k) x 3 -->
    pts_xy_xy, pts_xy_xz, pts_xy_yz = get_surface_points(k=6)
    # pts_xy_xy: bz x n_s x kk x 3
    pts_xy_xy = (pts_xy_xy).unsqueeze(0).unsqueeze(0) * mult_cuboic_xy.unsqueeze(-2)
    pts_xy_yz = (pts_xy_yz).unsqueeze(0).unsqueeze(0) * mult_cuboic_yz.unsqueeze(-2)
    pts_xy_xz = (pts_xy_xz).unsqueeze(0).unsqueeze(0) * mult_cuboic_xz.unsqueeze(-2)
    # pts_xy_up: bz x n_s x kk x 3
    pts_xy_up, pts_xy_dn = pts_xy_xy + slot_recon_cuboic_scc[:, :, 2, :].unsqueeze(-2) / 2., pts_xy_xy - slot_recon_cuboic_scc[:, :, 2, :].unsqueeze(-2) / 2.
    pts_xz_up, pts_xz_dn = pts_xy_xz + slot_recon_cuboic_scc[:, :, 1, :].unsqueeze(-2) / 2., pts_xy_xz - slot_recon_cuboic_scc[:, :, 1, :].unsqueeze(-2) / 2.
    pts_yz_up, pts_yz_dn = pts_xy_yz + slot_recon_cuboic_scc[:, :, 0, :].unsqueeze(-2) / 2., pts_xy_yz - slot_recon_cuboic_scc[:, :, 0, :].unsqueeze(-2) / 2.
    # cuboid_sampled_pts: bz x n_s x (kk x 6) x 3
    cuboid_sampled_pts = torch.cat([
        pts_xy_up, pts_xy_dn, pts_xz_up, pts_xz_dn, pts_yz_up, pts_yz_dn
    ], dim=-2)
    # inv_trans_pts: bz x n_s x 3 x N
    # dist_cuboid_sampled_to_inv_trans: bz x n_s x (kk x 6) x N
    dist_cuboid_sampled_to_inv_trans = torch.sum(
        (cuboid_sampled_pts.unsqueeze(-2) - safe_transpose(inv_trans_pts, -1, -2).unsqueeze(-3)) ** 2, dim=-1
    )
    # hard_one_hot_labels: bz x n_s x N --> expanded_hard_one_hot_labels: bz x n_s x (kk x 6) x N
    expanded_hard_one_hot_labels = safe_transpose(hard_one_hot_labels, -1, -2).unsqueeze(-2).contiguous().repeat(1, 1, dist_cuboid_sampled_to_inv_trans.size(-2), 1).contiguous()
    dist_cuboid_sampled_to_inv_trans[expanded_hard_one_hot_labels < 0.5] = 9999.0
    # cd_sampled_to_trans: bz x n_s x (kk x 6)
    cd_sampled_to_trans, _ = torch.min(dist_cuboid_sampled_to_inv_trans, dim=-1)
    # cd_trans_to_sampled: bz x n_s x N
    cd_trans_to_sampled, _ = torch.min(dist_cuboid_sampled_to_inv_trans, dim=-2)

    cd_trans_to_sampled = torch.sum(cd_trans_to_sampled * safe_transpose(hard_one_hot_labels, -1, -2), dim=-1) / torch.clamp(torch.sum(safe_transpose(hard_one_hot_labels, -1, -2), dim=-1), min=1e-8)

    # hard_slot_indicators: bz x n_s
    hard_slot_indicators = (torch.sum(safe_transpose(hard_one_hot_labels, -1, -2), dim=-1) > 0.9).float()
    # cd_sampled_trans = cd_trans_to_sampled.mean(dim=-1) + cd_sampled_to_trans.mean(dim=-1)
    cd_sampled_trans = cd_trans_to_sampled + cd_sampled_to_trans.mean(dim=-1)
    # print("cd_sampled_trans,", cd_sampled_trans, "hard_slot_indicators,", hard_slot_indicators, "cd_trans_to_sampled, ", cd_trans_to_sampled.mean(dim=-1), "cd_sampled_to_trans, ", cd_sampled_to_trans.mean(dim=-1))
    # cd_sampled_trans[cd_sampled_trans > 100.] = 0.


    cuboic_constraint_loss = torch.sum(cd_sampled_trans * hard_slot_indicators, dim=-1) / torch.sum(
        hard_slot_indicators, dim=-1)

    return cuboic_constraint_loss, cuboid_sampled_pts


from SPConvNets.utils.loss_util import batched_index_select
##### Get constraint loss with normals #####
def get_cuboic_constraint_loss_with_normals(pred_R, pred_T, ori_pts, ori_pts_normals, slot_recon_cuboic, slot_cuboic_R, hard_one_hot_labels, attn_ori):
    '''
        pred_R: bz x n_s x 3 x 3
        pred_T: bz x n_s x 3 # p
        slot_cuboic_R: bz x n_s x 3 x 3
        ori_pts: bz x 3 x N
        ori_pts_normals: bz x 3 x N
        slot_recon_cuboic: bz x n_s x 3 --> we assume that the reconstruction should not be rotated now...
        hard_one_hot_labels: bz x n_s x N
    '''
    bz, n_s = pred_R.size(0), pred_R.size(1)
    N = ori_pts.size(-1)
    # inv_trans_pts: bz x n_s x 3 x N
    # inv_trans_pts should be constrainted into the box with min_x, min_y, min_z = -s_x / 2., -s_y / 2., -s_z / 2.; and max_x, max_y, max_z = s_x / 2., s_y / 2., s_z / 2.
    # Inversed transformed points
    inv_trans_pts = torch.matmul(safe_transpose(pred_R, -1, -2), ori_pts.unsqueeze(1) - pred_T.unsqueeze(-1))
    inv_trans_normals = torch.matmul(safe_transpose(pred_R, -1, -2), ori_pts_normals.unsqueeze(1))

    # transform points
    # inv_trans_pts = inv_trans_pts + 0.5
    # Get inv_trans_pts: inv_cuboic_R^T inv_trans_pts
    # inv_trans_pts: bz x n_s x 3 x N
    inv_trans_pts = torch.matmul(safe_transpose(slot_cuboic_R, -1, -2), inv_trans_pts)
    inv_trans_normals = torch.matmul(safe_transpose(slot_cuboic_R, -1, -2), inv_trans_normals)

    # slot_recon_cuboic: bz x n_s x 3
    minn_recon_cuboic = -1. * slot_recon_cuboic / 2.
    maxx_recon_cuboic = 1. * slot_recon_cuboic / 2.
    dist_1 = torch.abs((minn_recon_cuboic.unsqueeze(-1) - inv_trans_pts) ** 2)
    dist_2 = torch.abs((maxx_recon_cuboic.unsqueeze(-1) - inv_trans_pts) ** 2)
    # x: y-z plane, y: x-z plane, z: x-y plane;
    # plane_axises: 3 x 3
    plane_axises = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32
    ).cuda()
    # plane_axises: 3 x 2 x 3
    plane_axises = torch.cat([-plane_axises.unsqueeze(1), plane_axises.unsqueeze(1)], dim=1)
    # inv_trans_normals: bz x n_s x 3 x N
    # matching_scores: bz x n_s x 3 x 2 x N --> points in every batch's matching scores with six plane normals
    # print(f"inv_trans_normals: {inv_trans_normals.size()}, plane_axises: {plane_axises.size()}")
    matching_scores = torch.sum(inv_trans_normals.unsqueeze(2).unsqueeze(2) * plane_axises.unsqueeze(0).unsqueeze(1).unsqueeze(-1), dim=-2)

    # dist_1: bz x n_s x 3 x N; dist_2: bz x n_s x 3 x N

    # dist_1 = torch.abs(inv_trans_pts)
    # dist_2 = torch.abs(slot_recon_cuboic.unsqueeze(-1) - inv_trans_pts)

    # dists = torch.cat([dist_1.unsqueeze(2), dist_2.unsqueeze(2)], dim=2)
    dists = torch.cat([dist_1.unsqueeze(-1), dist_2.unsqueeze(-1)], dim=-1)

    ''' Inlier xyz '''
    # should mask out inlier xyzs?
    # mask out inlier xyzs?
    # inlier_xyzs = (0. <= x <= cuboic_x.unsqueeze(-1)).float()

    ###### Get inlier xyzs ######
    # inlier_xyzs = (inv_trans_pts <= slot_recon_cuboic.unsqueeze(-1)).float()
    # inv_trans_pts: bz x n_s x 3 x N
    inlier_xyzs = (inv_trans_pts <= (slot_recon_cuboic / 2.).unsqueeze(-1)).float() * (inv_trans_pts >= (-1.0 * slot_recon_cuboic / 2.).unsqueeze(-1)).float()
    ###### Get inlier xyzs ######
    # Get inlier points indicators: the indicator for whether the point is in the cuboid's range
    # inlier_pts_indicators: bz x n_s x N; inlier_xyzs: bz x n_s x 3 x N
    inlier_pts_indicators = (inlier_xyzs.sum(2) > 2.5).float()

    # # matching_scores: bz x n_s x 3 x 2 x N
    # matching_scores = -matching_scores * inlier_pts_indicators.unsqueeze(2).unsqueeze(2) + matching_scores * (1. - inlier_pts_indicators.unsqueeze(2).unsqueeze(2))
    matching_scores = matching_scores.contiguous().view(bz, n_s, 6, N).contiguous()
    # maxx_matching_idx: bz x n_s x N
    maxx_matching_scores, maxx_matching_idx = torch.max(matching_scores, dim=-2)

    # dists: bz x n_s x 3 x N x 2
    tot_dist_outliers = dists.clone() # inlie

    # distance from points to cuboics for outlier points
    # tot_dist_outliers: bz x n_s x N

    ''' V2 -- outlier distance calculation '''
    # tot_dist_outliers = batched_index_select(
    #     values=safe_transpose(safe_transpose(tot_dist_outliers, -1, -2).contiguous().view(bz, n_s, 6, N).contiguous(), -1, -2),
    #     indices=safe_transpose(maxx_matching_idx.contiguous().unsqueeze(2).long().contiguous(), -1, -2), dim=3).squeeze(
    #     3)
    #
    # outlier_dist = tot_dist_outliers
    ''' V2 -- outlier distance calculation '''

    ''' V1 -- outlier distance calculation '''
    tot_dist_outliers[inlier_xyzs.unsqueeze(-1).repeat(1, 1, 1, 1, 2) > 0.5] = 0.0  # we should mask out inlier xyzs for outlier distance computing
    outlier_dist, _ = torch.min(tot_dist_outliers, dim=-1) # min for the
    outlier_dist, _ = torch.max(outlier_dist, dim=2)
    ''' V1 -- outlier distance calculation '''

    ''' V1 -- inlier distance calculation '''
    # inlier_dist, _ = torch.min(dists, dim=-1)
    #
    # # topk_inlier_dists: bz x n_s x 2 x N
    # topk_inlier_dists, _ = torch.topk(inlier_dist, k=2, largest=False, dim=2)
    # # topk_inlier_dists: bz x n_s x N
    # inlier_dist = torch.mean(topk_inlier_dists, dim=2)
    ''' V1 -- inlier distance calculation '''

    tot_dist_inliers = batched_index_select(
        values=safe_transpose(safe_transpose(dists, -1, -2).contiguous().view(bz, n_s, 6, N).contiguous(), -1, -2),
        indices=safe_transpose(maxx_matching_idx.contiguous().unsqueeze(2).long().contiguous(), -1, -2), dim=3).squeeze(3)

    inlier_dist = tot_dist_inliers

    # inlier_dist, _ = torch.min(inlier_dist, dim=2)

    # print(f"inlier_dist: {inlier_dist.size()}, outlier_dist: {outlier_dist.size()}, inlier_pts_indicators: {inlier_pts_indicators.size()}, maxx_matching_idx: {maxx_matching_idx.size()}, dists: {dists.size()}")

    minn_tot_dist = inlier_dist * inlier_pts_indicators + (1. - inlier_pts_indicators) * outlier_dist
    # minn_tot_dist = torch.mean(minn_tot_dist, dim=-1)
    minn_dists = minn_tot_dist

    ''' Previous: outliers not considered '''
    # # minn_dists: bz x n_s x 3 x N
    # minn_dists, _ = torch.min(dists, dim=2)
    # minn_dists, _ = torch.min(minn_dists, dim=2)
    ''' Previous: outliers not considered '''

    soft_weights = safe_transpose(hard_one_hot_labels, -1, -2) * attn_ori
    # cuboic_constraint_loss: bz x n_s
    # cuboic
    cuboic_constraint_loss = torch.sum(minn_dists * soft_weights, dim=-1) / torch.clamp(torch.sum(soft_weights, dim=-1), min=1e-8)
    hard_slot_indicators = (torch.sum(safe_transpose(hard_one_hot_labels, -1, -2), dim=-1) > 0.5).float()
    cuboic_constraint_loss = torch.sum(cuboic_constraint_loss * hard_slot_indicators, dim=-1) / torch.sum(hard_slot_indicators, dim=-1)
    # print(f"cuboic_constraint_loss: {cuboic_constraint_loss.item()}, {inlier_pts_indicators.mean().item()}")
    return cuboic_constraint_loss

# slot_pivot_points (here the first predicted pv point); central point; cuboid x; cuboid r; avg slot axis (direction of the predicted axis); normals... we can define them...
def get_cuboic_constraint_loss_with_axis_cuboid(slot_pivot_points, slot_central_points, slot_cuboid_x, slot_cuboid_R, avg_slot_axis):
    '''
        slot_pivot_poitns: bz x n_s x 3
        slot_central_points: bz x n_s x 3
        slot_cuboid_x: bz x n_s x 3
        slot_cuboid_R: bz x n_s x 3 x 3
        avg_slot_axis: bz x 3
    '''
    cuboid_normal_vectors = torch.tensor(
        [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
    ).cuda()
    # print(f"shape of cuboid_normal_vectors: {cuboid_normal_vectors.size()}") # expected: 3 x 2 x 3

    n_s = slot_central_points.size(1)
    slot_pivot_points = slot_pivot_points[:, 0, :].contiguous().unsqueeze(1).repeat(1, n_s, 1).contiguous()
    inv_central_transformed_pivot_points = slot_pivot_points - slot_central_points
    inv_trans_pivot_points = torch.matmul(safe_transpose(slot_cuboid_R, -1, -2), inv_central_transformed_pivot_points.unsqueeze(-1)).squeeze(-1)
    minn_slot_cuboid_xyz = -0.5 * slot_cuboid_x
    maxx_slot_cuboid_xyz = 0.5 * slot_cuboid_x
    dist_pv_minn_slot_cuboid_xyz = torch.abs(inv_trans_pivot_points - minn_slot_cuboid_xyz)
    dist_pv_maxx_slot_cuboid_xyz = torch.abs(inv_trans_pivot_points - maxx_slot_cuboid_xyz)
    cat_dist_pv_slot_cuboid_xyz = torch.cat([dist_pv_minn_slot_cuboid_xyz.unsqueeze(-1), dist_pv_maxx_slot_cuboid_xyz.unsqueeze(-1)], dim=-1)
    # cat_dist_pv_slot_cuboid_xyz: bz x n_s x 3 x 2 --> bz x n_s x 3; minn_dist_same_cat_idxes: bz x n_s x 3
    cat_dist_pv_slot_cuboid_xyz, minn_dist_same_cat_idxes = torch.min(cat_dist_pv_slot_cuboid_xyz, dim=-1)
    # cat_dist_pv_slot_cuboid_xyz: bz x n_s x 3 --> bz x n_s; minn_dist_xyz_idxes: bz x n_s
    cat_dist_pv_slot_cuboid_xyz, minn_dist_xyz_idxes = torch.min(cat_dist_pv_slot_cuboid_xyz, dim=-1)
    # selected_normals: bz x n_s x 2 x 3
    selected_normals = batched_index_select(values=cuboid_normal_vectors, indices=minn_dist_xyz_idxes.unsqueeze(-1).long(), dim=0).squeeze(2)
    # selected_cat_idxes: bz x n_s
    selected_cat_idxes = batched_index_select(values=minn_dist_same_cat_idxes, indices=minn_dist_xyz_idxes.unsqueeze(-1).long(), dim=2).squeeze(2)
    # selected_normals: bz x n_s x 3
    selected_normals = batched_index_select(values=selected_normals, indices=selected_cat_idxes.unsqueeze(-1).long(), dim=2).squeeze(2)
    trans_normals = torch.matmul(slot_cuboid_R, selected_normals.unsqueeze(-1)).squeeze(-1)
    # the predicted cuboid should not be affected...? 
    dot_axis_normals = torch.sum(avg_slot_axis.unsqueeze(1) * trans_normals.detach(), dim=-1).mean(dim=-1) # they should be orthogonal
    dot_axis_normals = torch.abs(dot_axis_normals)
    # dot_axis_normals: bz
    return dot_axis_normals



def get_weights_reguralization_loss(attn_ori):
    # attn_ori: bz x n_s x N
    # slot_weights: bz x n_s
    slot_weights = torch.mean(attn_ori, dim=-1)
    eps = 0.01
    loss_weights_reg = torch.sum(torch.sqrt(slot_weights + eps), dim=-1) ** 2
    # print("loss_weights_reg: ")
    # loss_weights_reg = torch.mean(loss_weights_reg)
    return loss_weights_reg


# [cos_, 0, sin_
#  0, 1., 0.
#  -sin_, 0., cos_]
# rot_matrices: 4 x 3 x 3
def get_2D_res_anchors():
    # angles = [_ * (np.pi / 4.) for _ in range(4)]
    angles = [_ * (np.pi / 2.) for _ in range(4)]
    rot_matrices = []
    for i, theta in enumerate(angles):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        n_x, n_y, n_z = 0.0, 1.0, 0.0
        cur_matrix = np.array(
            [[cos_theta, 0., sin_theta],
             [0., 1., 0.],
             [-sin_theta, 0., cos_theta]], dtype=np.float
        )
        rot_matrices.append(torch.from_numpy(cur_matrix).float().unsqueeze(0))
    rot_matrices = torch.cat(rot_matrices, dim=0).cuda()
    return rot_matrices