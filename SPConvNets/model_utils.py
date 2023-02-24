
import math
import os
import numpy as np
import time
from collections import namedtuple
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import vgtk.so3conv as sptk


# outblock for rotation regression model
class SO3OutBlockRT(nn.Module):
    def __init__(self, params, norm=None, pooling_method='mean', global_scalar=False, use_anchors=False,
                 feat_mode_num=60, num_heads=1):
        super(SO3OutBlockRT, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        na = params['kanchor']

        self.linear = nn.ModuleList()
        self.temperature = params['temperature']
        # self.representation = params['representation']
        self.global_scalar = global_scalar
        self.use_anchors   = use_anchors
        self.feat_mode_num=feat_mode_num
        self.num_heads = num_heads
        # self.attention_layer = nn.Conv2d(mlp[-1], 1, (1,1))
        if norm is not None:
            self.norm = nn.ModuleList()
        else:
            self.norm = None
        self.pooling_method = pooling_method
        if self.pooling_method == 'pointnet':
            self.pointnet = sptk.PointnetSO3Conv(mlp[-1], mlp[-1], na)

        self.attention_layer = nn.Conv1d(mlp[-1], 1 * num_heads, (1))
        if self.feat_mode_num < 2:
            self.regressor_layer = nn.Conv1d(mlp[-1],4,(1))
        else:
            self.regressor_layer = nn.Conv1d(mlp[-1], 4 * num_heads, (1))
        self.regressor_scalar_layer = nn.Conv1d(mlp[-1], 1 * num_heads, (1)) # [B, C, A] --> [B, 1, A] scalar, local

        # ------------------ uniary conv ----------------
        for c in mlp: #
            self.linear.append(nn.Conv2d(c_in, c, 1))
            if norm is not None:
                self.norm.append(nn.BatchNorm2d(c))
            c_in = c

        # ----------------- dense regression per mode per point ------------------
        self.regressor_dense_layer = nn.Sequential(nn.Conv2d(2 * mlp[-1], mlp[-1], 1),
                                                   nn.BatchNorm2d(mlp[-1]),
                                                   nn.LeakyReLU(inplace=True),
                                                   nn.Conv2d(c, 3 * num_heads, 1)) # randomly

    def forward(self, x, anchors=None):
        x_out = x.feats         # nb, nc, np, na -> nb, nc, na;
        nb, nc, np, na = x_out.shape
        # if x_out.shape[-1] == 1:
        #     x_out = x_out.repeat(1, 1, 1, 60).contiguous()
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if self.norm is not None: # apply normalization on obtained feature
                x_out = self.norm[lid](x_out)
            x_out = F.relu(x_out)

        # x_out: bz x dim x na
        shared_feat = x_out
        # mean pool at xyz ->  BxCxA
        if self.pooling_method == 'mean':
            x_out = x_out.mean(2) # max perform better? or point-based xyz conv
        elif self.pooling_method == 'max':
            x_out = x_out.max(2)[0]
        elif self.pooling_method == 'pointnet':
            x_in = sptk.SphericalPointCloud(x.xyz, x_out, None)
            x_out = self.pointnet(x_in)
        # regressor_dense_layer
        t_out = self.regressor_dense_layer(torch.cat([x_out.unsqueeze(2).repeat(1, 1, shared_feat.shape[2], 1).contiguous(),
                                                      shared_feat], dim=1))  # dense branch, [B, 3 * num_heads, P, A]
        t_out = t_out.reshape((nb, self.num_heads, 3) + t_out.shape[-2:])  # [B, num_heads, 3, P, A]
        # anchors = torch.from_numpy(L.get_anchors(self.config.model.kanchor)).to(self.output_pts)
        if self.global_scalar: # regressor
            y_t = self.regressor_scalar_layer(shared_feat.max(dim=-1)[0]).reshape(nb, self.num_heads, -1)  # [B, num_heads, P] --> [B, num_heads, P, A]
            # y_t = F.normalize(t_out, p=2, dim=2) * y_t.unsqueeze(-1)   #  [B, num_heads, 3, P, A]
            y_t = F.normalize(t_out, p=2, dim=2) * y_t.unsqueeze(2).unsqueeze(-1)
            if self.use_anchors:
                y_t = (torch.matmul(anchors.unsqueeze(1),
                                   y_t.permute(0, 1, 4, 2, 3).contiguous())
                       + x.xyz.unsqueeze(1).unsqueeze(1))  # [nb, num_heads, A, 3, P]
            else:
                y_t = y_t.permute(0, 1, 4, 2, 3).contiguous() + x.xyz.unsqueeze(1).unsqueeze(1)  # nb, num_heads, 60, 3, 64
        else:
            y_t = torch.matmul(anchors.unsqueeze(1),
                               t_out.permute(0, 1, 4, 2, 3).contiguous()) \
                  + x.xyz.unsqueeze(1).unsqueeze(1)  # nb, num_heads, 60, 3, 64
        # print(y_t.size())
        y_t = y_t.mean(dim=-1).permute(0, 1, 3, 2).contiguous()  # [B, num_heads, 3, A]

        attention_wts = self.attention_layer(x_out)  # [B, num_heads, A]
        confidence = F.softmax(attention_wts * self.temperature, dim=2)
        # regressor
        output = {}
        y = self.regressor_layer(x_out)  # [B, num_heads, 4, A]
        output['1'] = confidence  #
        output['R'] = y
        output['T'] = y_t

        if self.num_heads == 1:
            for key, value in output.items():
                output[key] = value.squeeze(1)

        return output

class SO3OutBlockR(nn.Module):
    def __init__(self, params, norm=None, pooling_method='mean', pred_t=False, feat_mode_num=60, num_heads=1):
        super(SO3OutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        na = params['kanchor']
        # rp = params['representation']
        rp = 'quat'

        self.linear = nn.ModuleList()
        self.temperature = params['temperature']
        # self.representation = params['representation']
        self.feat_mode_num = feat_mode_num

        if rp == 'up_axis':
            self.out_channel = 3
            print('---SO3OutBlockR output up axis')
        elif rp == 'quat':
            self.out_channel = 4
        elif rp == 'ortho6d':
            self.out_channel = 6
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%rp)

        if norm is not None:
            self.norm = nn.ModuleList()
        else:
            self.norm = None
        self.pooling_method = pooling_method
        if self.pooling_method == 'pointnet':
            self.pointnet = sptk.PointnetSO3Conv(mlp[-1], mlp[-1], na)

        self.attention_layer = nn.Conv1d(mlp[-1], 1, (1))
        self.regressor_layer = nn.Conv1d(mlp[-1],self.out_channel,(1))

        self.pred_t = pred_t
        if pred_t:
            self.regressor_t_layer = nn.Conv1d(mlp[-1], 3 * num_heads, (1, ))

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            if norm is not None:
                self.norm.append(nn.BatchNorm2d(c))
            c_in = c

    def forward(self, x, anchors=None):
        nb = len(x.feats)
        x_out = x.feats
        # if x_out.shape[-1] == 1:
        #     x_out = x_out.repeat(1, 1, 1, 60).contiguous()
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if self.norm is not None:
                x_out = self.norm[lid](x_out)
            x_out = F.relu(x_out)  # [B, C, N, A]

        # mean pool at xyz ->  BxCxA
        if self.pooling_method == 'mean':
            x_out = x_out.mean(2) # max perform better? or point-based xyz conv
        elif self.pooling_method == 'max':
            x_out = x_out.max(2)[0]
        elif self.pooling_method == 'pointnet':
            x_in = sptk.SphericalPointCloud(x.xyz, x_out, None)
            x_out = self.pointnet(x_in)
        attention_wts = self.attention_layer(x_out)  # [B, 1, A]
        confidence = F.softmax(attention_wts * self.temperature, dim=2).squeeze(1)
        # regressor
        output = {}
        # if self.feat_mode_num < 2:
        #     y = self.regressor_layer(x_out[:, :, 0:1]).squeeze(-1).view(x.xyz.shape[0], 4, -1).contiguous()
        # else:
        y = self.regressor_layer(x_out) # [B, nr, A] # features from --- we must perform a clustering process
        output['1'] = confidence #
        output['R'] = y
        if self.pred_t:
            y_t = self.regressor_t_layer(x_out) # [B, 3, A]
            output['T'] = y_t
        else:
            output['T'] = None

        return output


class SO3OutBlockRTWithMask(nn.Module):
    def __init__(self, params, norm=None, pooling_method='mean', global_scalar=False, use_anchors=False,
                 feat_mode_num=60, num_heads=1, pred_R=True, representation='quat'):
        super(SO3OutBlockRTWithMask, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        na = params['kanchor']

        self.linear = nn.ModuleList()
        self.temperature = params['temperature']
        # self.representation = params['representation']
        self.global_scalar = global_scalar
        self.use_anchors   = use_anchors
        self.feat_mode_num=feat_mode_num
        self.num_heads = num_heads
        self.representation = representation # representation in ['quat', 'angle']

        self.pred_R = pred_R
        # self.attention_layer = nn.Conv2d(mlp[-1], 1, (1,1))
        if norm is not None:
            self.norm = nn.ModuleList()
        else:
            self.norm = None
        self.pooling_method = pooling_method
        if self.pooling_method == 'pointnet':
            self.pointnet = sptk.PointnetSO3Conv(mlp[-1], mlp[-1], na)

        if self.pred_R:
            # self.attention_layer = nn.Conv1d(mlp[-1], 1 * num_heads, (1))
            if self.representation == 'quat':
                if self.feat_mode_num < 2:
                    self.regressor_layer = nn.Conv1d(mlp[-1],4,(1))
                else:
                    self.regressor_layer = nn.Conv1d(mlp[-1], 4 * num_heads, (1))
            else:
                if self.feat_mode_num < 2:
                    self.regressor_layer = nn.Conv1d(mlp[-1],1,(1))
                else:
                    self.regressor_layer = nn.Conv1d(mlp[-1], 1 * num_heads, (1))

        if self.global_scalar:
            self.regressor_scalar_layer = nn.Conv1d(mlp[-1], 1 * num_heads, (1)) # [B, C, A] --> [B, 1, A] scalar, local

        # ------------------ uniary conv ----------------
        for c in mlp: #
            self.linear.append(nn.Conv2d(c_in, c, 1))
            if norm is not None:
                self.norm.append(nn.BatchNorm2d(c))
            c_in = c

        # ----------------- dense regression per mode per point ------------------
        self.regressor_dense_layer = nn.Sequential(nn.Conv2d(2 * mlp[-1], mlp[-1], 1),
                                                   nn.BatchNorm2d(mlp[-1]),
                                                   nn.LeakyReLU(inplace=True),
                                                   nn.Conv2d(c, 3 * num_heads, 1)) # randomly

    def forward(self, x, mask, anchors=None, soft_mask=None):
        x_out = x.feats         # nb, nc, np, na -> nb, nc, na;
        # features
        # x_out: bz x dim x N x na
        # todo: is just masking out input features a enough strategy?
        if mask is not None:
            x_out = x_out * mask.unsqueeze(1).unsqueeze(-1)
        # if soft_mask is not None:
        #     x_out = x_out * soft_mask.unsqueeze(1).unsqueeze(-1)
        nb, nc, np, na = x_out.shape
        # if x_out.shape[-1] == 1:
        #     x_out = x_out.repeat(1, 1, 1, 60).contiguous()
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if self.norm is not None: # apply normalization on obtained feature
                x_out = self.norm[lid](x_out)
            x_out = F.relu(x_out)

        # x_out: bz x dim x na
        shared_feat = x_out
        # mean pool at xyz ->  BxCxA
        if self.pooling_method == 'mean':
            x_out = x_out.mean(2) # max perform better? or point-based xyz conv
        elif self.pooling_method == 'max':
            # x_out: bz x dim x N x na
            if mask is not None:
                expanded_mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1, x_out.size(1), 1, x_out.size(-1))
                # it is reasonable to set masked values to zero due to the relu operation applied on x_out before
                x_out[expanded_mask < 0.5] = 0.
            # if soft_mask is not None:
            #     x_out = x_out * soft_mask.unsqueeze(1).unsqueeze(-1)
            x_out = x_out.max(2)[0]
        elif self.pooling_method == 'pointnet':
            x_in = sptk.SphericalPointCloud(x.xyz, x_out, None)
            x_out = self.pointnet(x_in)
        # regressor_dense_layer
        t_out = self.regressor_dense_layer(torch.cat([x_out.unsqueeze(2).repeat(1, 1, shared_feat.shape[2], 1).contiguous(),
                                                      shared_feat], dim=1))  # dense branch, [B, 3 * num_heads, P, A]
        t_out = t_out.reshape((nb, self.num_heads, 3) + t_out.shape[-2:])  # [B, num_heads, 3, P, A]
        # anchors = torch.from_numpy(L.get_anchors(self.config.model.kanchor)).to(self.output_pts)
        if self.global_scalar: # regressor
            y_t = self.regressor_scalar_layer(shared_feat.max(dim=-1)[0]).reshape(nb, self.num_heads, -1)  # [B, num_heads, P] --> [B, num_heads, P, A]
            # y_t = F.normalize(t_out, p=2, dim=2) * y_t.unsqueeze(-1)   #  [B, num_heads, 3, P, A]
            y_t = F.normalize(t_out, p=2, dim=2) * y_t.unsqueeze(2).unsqueeze(-1)
            if self.use_anchors:
                y_t = (torch.matmul(anchors.unsqueeze(1),
                                   y_t.permute(0, 1, 4, 2, 3).contiguous())
                       + x.xyz.unsqueeze(1).unsqueeze(1))  # [nb, num_heads, A, 3, P]
            else:
                y_t = y_t.permute(0, 1, 4, 2, 3).contiguous() + x.xyz.unsqueeze(1).unsqueeze(1)  # nb, num_heads, 60, 3, 64
        else:
            # na x 1 x 3 x 3
            # bz x na x 3 x 3 --> bz x 1 x na x 3 x 3 @ bz x 1 x na x 3 x np --> bz x 1 x na x 3 x np
            # x.xyz: bz x 3 x np --> bz x 1 x 1 x 3 x np
            y_t = torch.matmul(anchors.unsqueeze(1),
                               t_out.permute(0, 1, 4, 2, 3).contiguous()) \
                  + x.xyz.unsqueeze(1).unsqueeze(1)  # nb, num_heads, 60, 3, 64
            # j
        # print(y_t.size())
        #

        if mask is not None:
            y_t = torch.sum(y_t * mask.unsqueeze(1).unsqueeze(1).unsqueeze(1), dim=-1) / torch.clamp(torch.sum(mask.unsqueeze(1).unsqueeze(1).unsqueeze(1), dim=-1), min=1e-8)
            y_t = y_t.contiguous().permute(0, 1, 3, 2).contiguous()
        elif soft_mask is not None:
            y_t = torch.sum(y_t * soft_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1), dim=-1) / torch.clamp(
                torch.sum(soft_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1), dim=-1), min=1e-8)
            y_t = y_t.contiguous().permute(0, 1, 3, 2).contiguous()
        else:
            y_t = y_t.mean(dim=-1).permute(0, 1, 3, 2).contiguous()  # [B, num_heads, 3, A]

        # attention_wts = self.attention_layer(x_out)  # [B, num_heads, A]
        # confidence = F.softmax(attention_wts * self.temperature, dim=2)
        # regressor
        output = {}
        if self.pred_R:
            y = self.regressor_layer(x_out)  # [B, num_heads, 4, A]
        else:
            y = None
        # output['1'] = confidence  #
        output['R'] = y
        output['T'] = y_t

        if self.num_heads == 1:
            for key, value in output.items():
                if value is not None:
                    output[key] = value.squeeze(1)

        return output


class SO3OutBlockRWithMask(nn.Module):
    def __init__(self, params, norm=None, pooling_method='mean', pred_t=False, feat_mode_num=60, num_heads=1, representation='quat'):
        super(SO3OutBlockRWithMask, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        na = params['kanchor']
        # rp = params['representation']
        rp = representation
        # if representation == 'angle', then we just predict an angle along the axis

        self.linear = nn.ModuleList()
        self.temperature = params['temperature']
        # self.representation = params['representation']
        self.feat_mode_num = feat_mode_num

        if rp == 'up_axis':
            self.out_channel = 3 # out channel; 36 / 180 angle difference?
            print('---SO3OutBlockR output up axis')
        elif rp == 'quat':
            self.out_channel = 4
        elif rp == 'ortho6d':
            self.out_channel = 6
        elif rp == 'angle':
            self.out_channel = 1
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%rp)

        if norm is not None:
            self.norm = nn.ModuleList()
        else:
            self.norm = None
        self.pooling_method = pooling_method
        if self.pooling_method == 'pointnet':
            self.pointnet = sptk.PointnetSO3Conv(mlp[-1], mlp[-1], na)

        # self.attention_layer = nn.Conv1d(mlp[-1], 1, (1))
        self.regressor_layer = nn.Conv1d(mlp[-1],self.out_channel,(1))

        self.pred_t = pred_t
        if pred_t:
            self.regressor_t_layer = nn.Conv1d(mlp[-1], 3 * num_heads, (1, ))

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            if norm is not None:
                self.norm.append(nn.BatchNorm2d(c))
            c_in = c

    def forward(self, x, mask=None, soft_mask=None, anchors=None):
        nb = len(x.feats)
        x_out = x.feats
        if mask is not None:
            x_out = x_out * mask.unsqueeze(1).unsqueeze(-1)
        if soft_mask is not None:
            x_out = x_out * soft_mask.unsqueeze(1).unsqueeze(-1)
        # if x_out.shape[-1] == 1:
        #     x_out = x_out.repeat(1, 1, 1, 60).contiguous()
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if self.norm is not None:
                x_out = self.norm[lid](x_out)
            x_out = F.relu(x_out)  # [B, C, N, A]

        # mean pool at xyz ->  BxCxA
        if self.pooling_method == 'mean':
            x_out = x_out.mean(2) # max perform better? or point-based xyz conv
        elif self.pooling_method == 'max':
            if mask is not None:
                x_out = x_out * mask.unsqueeze(1).unsqueeze(-1)
            if soft_mask is not None:
                x_out = x_out * soft_mask.unsqueeze(1).unsqueeze(-1)
            x_out = x_out.max(2)[0]
        elif self.pooling_method == 'pointnet':
            x_in = sptk.SphericalPointCloud(x.xyz, x_out, None)
            x_out = self.pointnet(x_in)
        # attention_wts = self.attention_layer(x_out)  # [B, 1, A]
        # confidence = F.softmax(attention_wts * self.temperature, dim=2).squeeze(1)
        # regressor
        output = {}
        # if self.feat_mode_num < 2:
        #     y = self.regressor_layer(x_out[:, :, 0:1]).squeeze(-1).view(x.xyz.shape[0], 4, -1).contiguous()
        # else:
        y = self.regressor_layer(x_out) # [B, nr, A] # features from --- we must perform a clustering process

        # output['1'] = confidence #
        output['R'] = y
        if self.pred_t:
            y_t = self.regressor_t_layer(x_out) # [B, 3, A]
            output['T'] = y_t
        else:
            output['T'] = None

        return output


def from_rotation_mtx_to_axis(rots):
    ori_rots_size = rots.size()[:-1] # number of rotation matrices
    exp_rots = rots.contiguous().view(-1, 3, 3).contiguous()
    exp_axises = []

    for i in range(exp_rots.size(0)):
        cur_rot = exp_rots[i]
        trace = cur_rot[0, 0] + cur_rot[1, 1] + cur_rot[2, 2] # current rotation
        trace = (trace - 1.) / 2.
        trace = float(trace.item())
        trace = min(1.0, max(trace, -1.0))
        if abs(trace - 1.0) < 1e-8:
            axes = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float).cuda()
        elif abs(trace + 1.0) < 1e-8:
            ax_x = torch.sqrt((cur_rot[0, 0] + 1.) / 2.)
            if float(ax_x.item()) > 1e-8:
                ax_y = cur_rot[0, 1] + cur_rot[1, 0]
                ax_y = ax_y / (4. * ax_x)
                ax_z = cur_rot[0, 2] + cur_rot[2, 0]
                ax_z = ax_z / (4. * ax_x)
                axes = torch.tensor([ax_x, ax_y, ax_z], dtype=torch.float).cuda()
            else:
                ax_y = torch.sqrt((cur_rot[1, 1] + 1) / 2.0)
                if float(ax_y.item()) > 1e-8:
                    ax_z = cur_rot[1, 2] + cur_rot[2, 1]
                    ax_z = ax_z / (4. * ax_y)
                    axes = torch.tensor([0, ax_y, ax_z], dtype=torch.float).cuda()
                else:
                    ax_z = torch.sqrt((cur_rot[2, 2] + 1) / 2.0)
                    axes = torch.tensor([0, 0, ax_z], dtype=torch.float).cuda()
        else:
            angle = math.acos(trace)  # acos
            # print(angle / np.pi * 180.)
            sine = math.sin(angle)  # angle
            ax_x, ax_y, ax_z = cur_rot[2, 1] - cur_rot[1, 2], cur_rot[0, 2] - cur_rot[2, 0], cur_rot[1, 0] - \
                               cur_rot[0, 1]
            axes = torch.tensor([ax_x, ax_y, ax_z], dtype=torch.float).cuda()
            axes = axes / (2.0 * sine)
        exp_axises.append(axes.unsqueeze(0))
    # exp_dim x 3
    exp_axises = torch.cat(exp_axises, dim=0)
    # exp_dim x 3; axises
    exp_axises = exp_axises.contiguous().view(*ori_rots_size).contiguous()
    return exp_axises


def compute_rotation_matrix_from_angle(anchors, angles, defined_axis=None):
    '''
        anchors: na x 3 x 3
        angles: -1 x na x 1
    '''
    angles = angles.squeeze(-1)
    n_angles = angles.size(0)
    na = anchors.size(0)
    # anchor_axises: na x 3
    if defined_axis is None:
        anchor_axises = from_rotation_mtx_to_axis(anchors)
    else:
        anchor_axises = defined_axis
    u, v, w = anchor_axises[:, 0].unsqueeze(0), anchor_axises[:, 1].unsqueeze(0), anchor_axises[:, 2].unsqueeze(0)

    costheta = torch.cos(angles)
    sintheta = torch.sin(angles)

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    m = torch.zeros((n_angles, na, 3, 3), dtype=torch.float32).cuda()
    # print(uu.size(), costheta.size())
    m[:, :, 0, 0] = uu + (vv + ww) * costheta
    m[:, :, 1, 0] = uv * (1 - costheta) + w * sintheta
    m[:, :, 2, 0] = uw * (1 - costheta) - v * sintheta

    m[:, :, 0, 1] = uv * (1 - costheta) - w * sintheta
    m[:, :, 1, 1] = vv + (uu + ww) * costheta
    m[:, :, 2, 1] = vw * (1 - costheta) + u * sintheta

    m[:, :, 0, 2] = uw * (1 - costheta) + v * sintheta
    m[:, :, 1, 2] = vw * (1 - costheta) - u * sintheta
    m[:, :, 2, 2] = ww + (uu + vv) * costheta

    return m
