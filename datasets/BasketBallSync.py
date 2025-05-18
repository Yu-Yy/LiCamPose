"""
This script output the pointcloud generate 3d voxel and the syncthetic 2d heatmap
1. Using the camera parameter of the basketbal dataset to generate the heatmap
2. Using the unity parameter to generate the heatmap 
"""
import torch
import numpy as np
import torch.utils.data as data
from glob import glob
import os
import open3d as o3d
import cv2
import copy
import random
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import rotate_points, get_scale
from utils.cameras_cpu import project_pose, project_pose_sync
import json

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]

class SyntheticDataM(data.Dataset):
    def __init__(self, cfg, datadir, is_train=True, is_augment=True):
        super().__init__()
        self.train_peds = range(8)
        self.test_peds = range(8,10)
        self.is_train = is_train
        self.augment = is_augment
        if is_train:
            self.peds = self.train_peds
        else:
            self.peds = self.test_peds
        self.data_dir = datadir
        self.points_dir = os.path.join(datadir, 'points', '*.txt')
        self.files = glob(self.points_dir)
        self.timestamp = len(self.files)
        self.space_size = np.array(cfg.PICT_STRUCT.GRID_SIZE)
        self.initial_cube_size = np.array(cfg.PICT_STRUCT.CUBE_SIZE)
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
        self.kp_num = cfg.NUM_JOINTS
        self.cam_file = os.path.join(datadir, 'images')
        self.views_num = len(glob(os.path.join(self.cam_file, 'camera_*.txt')))
        self.cam = []
        for i in range(self.views_num):
            camM = np.loadtxt(os.path.join(self.cam_file, f'camera_{i+1}.txt'), dtype=np.float32)
            self.cam.append(camM)
        # load the pred_2d result
        self.use_pred2d = cfg.USE_PRED2D
        if cfg.USE_PRED2D:
            self.pred_2d_folder = os.path.join(self.data_dir, 'pred_2d_folder')
        

    def __len__(self):
        return len(self.peds) * self.timestamp # TODO: fixed the timestamp length 3517 

    def __augment__(self, lidar, joints):
        # lidar is N x 3
        # joints is J x 3
        # ang is applied on the xy plane
        ang = np.random.uniform(-np.pi, np.pi)
        rotation_matrix = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])

        lidar[:,:2] = lidar[:,:2] @ rotation_matrix.T
        joints[:,:2] = joints[:,:2] @ rotation_matrix.T
        return lidar, joints

    def generate_3d_input(self, lidar, kp, space_center): # different person has different space center
        space_size = self.space_size
        cube_size = self.initial_cube_size
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        i_x = np.searchsorted(grid1Dx, lidar[:,0])
        i_y = np.searchsorted(grid1Dy, lidar[:,1])
        i_z = np.searchsorted(grid1Dz, lidar[:,2])

        i_x[i_x == cube_size[0]] = cube_size[0] - 1
        i_y[i_y == cube_size[1]] = cube_size[1] - 1
        i_z[i_z == cube_size[2]] = cube_size[2] - 1

        i = np.concatenate([i_x[:,np.newaxis], i_y[:,np.newaxis], i_z[:,np.newaxis]], axis=1)
        i_uniq, inv_ind, voxel_counts = np.unique(i, axis=0, return_inverse=True, return_counts=True)
        input_flatten = np.zeros(cube_size[0] * cube_size[1] * cube_size[2], dtype=np.float32)
        i_flatten = i_uniq[:,0] * cube_size[1] * cube_size[2] + i_uniq[:,1] * cube_size[2] + i_uniq[:,2]
        input_flatten[i_flatten] = voxel_counts
        input_3d_count = input_flatten.reshape(cube_size[0], cube_size[1], cube_size[2])      
        input_3d = (input_3d_count > 0).astype(np.float32)

        kp_x = np.searchsorted(grid1Dx, kp[:,0])
        kp_y = np.searchsorted(grid1Dy, kp[:,1])
        kp_z = np.searchsorted(grid1Dz, kp[:,2])

        kp_x[kp_x == cube_size[0]] = cube_size[0] - 1
        kp_y[kp_y == cube_size[1]] = cube_size[1] - 1
        kp_z[kp_z == cube_size[2]] = cube_size[2] - 1
        kp_idx = np.concatenate([kp_x[:,np.newaxis], kp_y[:,np.newaxis], kp_z[:,np.newaxis]], axis=1)
        kp_idx_flatten = kp_idx[:,0] * cube_size[1] * cube_size[2] + kp_idx[:,1] * cube_size[2] + kp_idx[:,2]

        return input_3d, kp_idx_flatten

    def generate_heatmap(self, kp_3d):
        joints_3d = copy.deepcopy(kp_3d) # J x 3
        joints_3d_vis = np.ones_like(joints_3d)
        width = self.image_size[0]
        height = self.image_size[1]
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = get_scale((width, height), self.image_size)
        r = 0
        joints = []
        joints_vis = [] # collect different views 
        pose2d_gt = []
        for cam in self.cam:
            # trans the order for joints_3d
            pose2d = project_pose_sync(joints_3d[:,[0,2,1]], cam, self.image_size[0], self.image_size[1]) # transfer the y and z axis for normal projection

            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                     pose2d[:, 0] <= width - 1)
            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                     pose2d[:, 1] <= height - 1)
            check = np.bitwise_and(x_check, y_check)
            vis = joints_3d_vis[:, 0] > 0
            vis[np.logical_not(check)] = 0

            joints.append(pose2d)
            joints_vis.append(np.repeat(np.reshape(vis, (-1, 1)), 2, axis=1))
            pose2d_gt.append(np.concatenate([pose2d, np.reshape(vis, (-1, 1))], axis=1))


        return self._generate_input_heatmap_(joints, joints_vis), pose2d_gt
        
    def _generate_input_heatmap_(self, joints, joints_vis):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: input_heatmap
        '''
        vposes = len(joints)
        num_joints = joints[0].shape[0]
        input_heatmap = []
        feat_stride = self.image_size / self.heatmap_size

        for n in range(vposes):
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            obscured = random.random() < 0.05
            if obscured:
                target = torch.from_numpy(target)
                input_heatmap.append(target)
                continue
            human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
            if human_scale == 0:
                target = torch.from_numpy(target)
                input_heatmap.append(target)
                continue

            cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
            tmp_size = cur_sigma * 3
            for joint_id in range(num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if joints_vis[n][joint_id, 0] == 0 or \
                        ul[0] >= self.heatmap_size[0] or \
                        ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # scale = 1 - np.abs(np.random.randn(1) * 0.25)
                scale = 0.9 + np.random.randn(1) * 0.03 if random.random() < 0.6 else 1.0
                if joint_id in [4, 10, 7, 13]:
                    scale = scale * 0.5 if random.random() < 0.1 else scale
                elif joint_id in [5, 11, 8, 14]:
                    scale = scale * 0.2 if random.random() < 0.1 else scale
                else:
                    scale = scale * 0.5 if random.random() < 0.05 else scale
                g = np.exp(
                    -((x - x0) ** 2 + (y - y0) ** 2) / (2 * cur_sigma ** 2)) * scale

                # Usable gaussian range
                g_x = max(0,
                            -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0,
                            -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
            target = np.clip(target, 0, 1)

            # changed to the torch
            target = torch.from_numpy(target)
            input_heatmap.append(target)

        return input_heatmap

    def __getitem__(self, index):
        time_idx = index // len(self.peds)
        ped_idx = index % len(self.peds)
        ped = self.peds[ped_idx] # select the corresponding ped
        points_file = os.path.join(self.data_dir, 'points_ped',f'{ped}', f'{time_idx:>05d}.ply')
        joints_file = os.path.join(self.data_dir, 'joints',f'{ped}',f'joints_{time_idx}.txt')
        # test 
        if not os.path.exists(points_file):
            return None, None, None, None, None 
        pcd = o3d.io.read_point_cloud(points_file)
        pointcloud = np.array(pcd.points)

        kp = np.loadtxt(joints_file)
        kp = kp[:,[0,2,1]]
        # augment
        if self.is_train and self.augment and not self.use_pred2d:
            center = 0.5 * (np.max(pointcloud[:,:2], axis=0, keepdims=True) + np.min(pointcloud[:,:2], axis=0, keepdims=True)) # center is not the mean
            height = np.min(pointcloud[:,2:3], axis=0, keepdims=True)
            uni = np.concatenate([center, height], axis=1)    
            lidar_load = pointcloud - uni
            kp_load = kp - uni
            lidar_load, kp_load = self.__augment__(lidar_load, kp_load)
            lidar_load = lidar_load + uni
            kp_load = kp_load + uni
        # return to the original position
        else:
            lidar_load = pointcloud
            kp_load = kp
        lidar_center = 0.5 * (np.max(lidar_load, axis=0) + np.min(lidar_load, axis=0))
        input_3d, kp_idx = self.generate_3d_input(lidar_load, kp_load,lidar_center)
        input_3d = torch.from_numpy(input_3d)
        kp_idx = torch.from_numpy(kp_idx)
        input_3d = input_3d.unsqueeze(0) # 1 x d x w x h
        lidar_center = torch.from_numpy(lidar_center) # 3
        # generate the heatmap and output the camera parameter
        if self.use_pred2d:
            pred_2d_info = []
            pred_valid_label = 0
            for v in range(self.views_num):
                kp_file = os.path.join(self.pred_2d_folder, f'{ped}', f'{v+1}', f'{time_idx:0>5d}.json')
                if os.path.exists(kp_file):
                    pred_valid_label += 1
                    with open(kp_file, 'r') as f:
                        kp_2d = json.load(f)
                    pred_2d_info.append(np.array(kp_2d))
                else:
                    pred_2d_info.append(np.zeros((self.kp_num, 3)))
            if pred_valid_label == 0:
                return None, None, None, None, None, None, None, None
            input_heatmap = self.process_2d_infomation(pred_2d_info)
        else:
            # using the random simulation views to generate the heatmap
            input_heatmap, pose_2dgt = self.generate_heatmap(kp_load) # It is a list about the views

        kp_load = torch.from_numpy(kp_load)
        output_cam = []
        for cam in self.cam:
            output_cam.append(torch.from_numpy(cam))
            
        return input_3d, kp_load, lidar_center, input_heatmap, output_cam #kp_idx #, ped, time_idx

    def process_2d_infomation(self, pred_2d):
        # pred_2d is [[J x 3]]
        joints_vis = [x[:,2:3] for x in pred_2d]

        return self.generate_input_heatmap(pred_2d, joints_vis)

    def generate_input_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints]
        :return: input_heatmap
        '''
        vposes = len(joints)
        num_joints = joints[0].shape[0]
        input_heatmap = []
        feat_stride = self.image_size / self.heatmap_size
        for n in range(vposes):
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            human_scale = 2 * self.compute_human_scale(joints[n][:, 0:2] / feat_stride, joints_vis[n])
            if human_scale == 0:
                target = torch.from_numpy(target)
                input_heatmap.append(target)
                continue

            cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
            tmp_size = cur_sigma * 3
            for joint_id in range(num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if joints_vis[n][joint_id, 0] <=0.3 or \
                        ul[0] >= self.heatmap_size[0] or \
                        ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    continue  # setting the invisible joints (< 0.3) as 0
                
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2

                max_value = joints[n][joint_id][2] if len(joints[n][joint_id]) == 3 else 1.0
                g = np.exp(
                    -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2)) * max_value

                # Usable gaussian range
                g_x = max(0,
                            -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0,
                            -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
            target = np.clip(target, 0, 1)
            target = torch.from_numpy(target)
            input_heatmap.append(target)

        return input_heatmap
    @staticmethod
    def compute_human_scale(pose, joints_vis):
        idx = joints_vis[:, 0] >= 0.3 # using the general one
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        return np.clip(np.maximum(maxy - miny, maxx - minx) ** 2, 1.0 / 4 * 96 ** 2, 4 * 96 ** 2)
