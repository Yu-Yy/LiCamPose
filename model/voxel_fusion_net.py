# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.v2vnet import V2VNet
from model.project_layer import ProjectLayer

# it needs to re-calculate the standard for the model's confidence
 

class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.TRAIN.BETA

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)  # turn to the probability
        heatmap = x.squeeze(-1).clone()
        # calculate the entropy
        entropy = -torch.sum(x * torch.log(x + 1e-6), dim=2) # calculate the entropy ！！！# B x J x 
        # test the value
        # mean_value = torch.mean(x, dim=2, keepdim=True)
        # mask = x > mean_value
        # # calculate the mask's regions's softmax value via each channel
        # # masked softmax
        # exp_x = torch.exp(x)
        # exp_x = exp_x * mask
        # exp_x = exp_x / torch.sum(exp_x, dim=2, keepdim=True)
        # new_entropy = -torch.sum(exp_x * torch.log(exp_x + 1e-6), dim=2)
        # tst, tst2 = x[0,0], x[0,2]
        # print(tst.max(), tst2.max())
        # print(torch.sum(tst > tst.mean()), torch.sum(tst2 > tst2.mean()))
        # print(entropy[0,0], entropy[0,2])
        # print(new_entropy[0,0], new_entropy[0,2])
        # import pdb;pdb.set_trace()
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x, entropy, heatmap

class PerJointL1Loss(nn.Module):
    def __init__(self):
        super(PerJointL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, output, target, use_target_weight=False, target_weight=None):
        if use_target_weight:
            batch_size = output.size(0)
            num_joints = output.size(1)

            pred = output.reshape((batch_size, num_joints, -1))
            gt = target.reshape((batch_size, num_joints, -1))
            loss = self.criterion(pred.mul(target_weight), gt.mul(target_weight))
        else:
            loss = self.criterion(output, target)
        return loss

class VoxelFusionNet(nn.Module):
    def __init__(self, cfg):
        super(VoxelFusionNet, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE

        self.project_layer = ProjectLayer(cfg)
        self.v2v_2d_net = V2VNet(cfg.NUM_JOINTS, cfg.NETWORK.MID_FDIM)
        self.v2v_3d_net = V2VNet(1, cfg.NETWORK.MID_FDIM)
        self.v2v_voxelfusion = V2VNet(cfg.NETWORK.MID_FDIM * 2, cfg.NUM_JOINTS)
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def forward(self, input3d, all_heatmaps, projectionM, grid_centers):
        batch_size = all_heatmaps[0].shape[0]
        num_joints = all_heatmaps[0].shape[1]
        device = all_heatmaps[0].device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)
        cubes, grids = self.project_layer(all_heatmaps, projectionM,
                                          self.grid_size, grid_centers, self.cube_size)
        # 2d part
        cubes_2d = self.v2v_2d_net(cubes)
        # 3d point part
        cubes_3d = self.v2v_3d_net(input3d)
        # VoxelFusion
        cubes_fusion = torch.cat([cubes_2d, cubes_3d], dim=1)
        valid_cubes = self.v2v_voxelfusion(cubes_fusion)
        pred, entropy, heatmap = self.soft_argmax_layer(valid_cubes, grids)
        pred = torch.cat([pred, entropy], dim=2)
        return pred, heatmap
