# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.v2vnet import V2VNet
from model.project_layer import ProjectLayer


class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.TRAIN.BETA

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)
        # calculate the entropy
        entropy = -torch.sum(x * torch.log(x + 1e-6), dim=2) # B x J x 
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x, entropy


class PoseRegressionNet_2D(nn.Module):
    def __init__(self, cfg):
        super(PoseRegressionNet_2D, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE

        self.project_layer = ProjectLayer(cfg)
        self.v2v_net = V2VNet(cfg.NUM_JOINTS, cfg.NUM_JOINTS)
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def forward(self, all_heatmaps, projectionM, grid_centers):
        batch_size = all_heatmaps[0].shape[0]
        num_joints = all_heatmaps[0].shape[1]
        device = all_heatmaps[0].device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)
        cubes, grids = self.project_layer(all_heatmaps, projectionM,
                                          self.grid_size, grid_centers, self.cube_size)
        # index = grid_centers[:, 3] >= 0
        valid_cubes = self.v2v_net(cubes)
        pred, entropy = self.soft_argmax_layer(valid_cubes, grids)
        pred = torch.cat([pred, entropy], dim=2)

        return pred
 