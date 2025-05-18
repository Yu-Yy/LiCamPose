# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.v2vnet import V2VNet


class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.TRAIN.BETA

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2) # the function of beta?
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x


class PoseRegressionNet(nn.Module):
    def __init__(self, cfg):
        super(PoseRegressionNet, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE
        self.num_joints = cfg.NUM_JOINTS
        # self.project_layer = ProjectLayer(cfg)
        self.v2v_net = V2VNet(1, cfg.NUM_JOINTS) # input the one channel data 
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):  # nbins is not accurate
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def forward(self, input3d, grid_centers):
        # input3d 
        # input3d size is B X C X D X H X W, grid centers shape is B X 3
        batch_size = input3d.shape[0]
        device = input3d.device
        nbins = self.cube_size[0] * self.cube_size[1] * self.cube_size[2]
        pred = torch.zeros(batch_size, self.num_joints, 3, device=device)
        grids = torch.zeros(batch_size, nbins, 3, device = device)
        for i in range(batch_size):
            grid = self.compute_grid(self.grid_size, grid_centers[i], self.cube_size, device)
            grids[i:i+1] = grid

        valid_cubes = self.v2v_net(input3d)
        pred = self.soft_argmax_layer(valid_cubes, grids)  # return the voxel's coordinate

        return pred
