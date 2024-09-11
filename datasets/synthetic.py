import numpy as np
import torch.utils.data as data
import torch
from glob import glob
import os
import open3d as o3d

class SyntheticData(data.Dataset):
    def __init__(self, cfg, datadir, is_train = True):
        super().__init__()
        self.train_peds = range(8)
        self.test_peds = range(8,10)
        self.is_train = is_train
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
        
    def __len__(self):
        return len(self.peds) * self.timestamp # TODO: fixed the length for the equal test 3517

    # def __getpc__(self, pc, kp):
    #     # confirm the bbox range
    #     r_min = np.min(kp, axis=0) - 0.1
    #     r_max = np.max(kp, axis=0) + 0.1
    #     select = pc[:,0] > r_min[0] and pc[:,0] < r_max[0] and pc[:,1] > r_min[1] and pc[:,1] < r_max[1]\
    #         and pc[:,2] > r_min[2] and pc[:,2] < r_max[2] and pc[:,2] > 0.1
    #     points_peds = pc[select,:]

    #     center = 0.5 * (np.max(points_peds[:,:2], axis=0, keepdims=True) + np.min(points_peds[:,:2], axis=0, keepdims=True)) # center is not the mean

    #     height = np.min(points_peds[:,2:3], axis=0, keepdims=True)
    #     uni = np.concatenate([center, height], axis=1)   # tranfer to relative space's points and joints
    #     lidar_load = points_peds - uni
    #     kp_load = kp - uni

    #     return lidar_load, kp_load

    def __augment__(self, lidar, joints):
        # lidar is N x 3
        # joints is J x 3
        # ang is applied on the xy plane
        ang = np.random.uniform(-np.pi, np.pi)
        rotation_matrix = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])

        lidar[:,:2] = lidar[:,:2] @ rotation_matrix.T
        joints[:,:2] = joints[:,:2] @ rotation_matrix.T
        return lidar, joints

    def generate_3d_input(self, lidar, space_center): # different person has different space center
        space_size = self.space_size
        cube_size = self.initial_cube_size
        # input_3d = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        i_x = np.searchsorted(grid1Dx, lidar[:,0])
        i_y = np.searchsorted(grid1Dy, lidar[:,1])
        i_z = np.searchsorted(grid1Dz, lidar[:,2])

        i_x[i_x == cube_size[0]] = cube_size[0] - 1
        i_y[i_y == cube_size[1]] = cube_size[1] - 1
        i_z[i_z == cube_size[2]] = cube_size[2] - 1

        # # || TODO: VOXEL COUNTING
        i = np.concatenate([i_x[:,np.newaxis], i_y[:,np.newaxis], i_z[:,np.newaxis]], axis=1)
        i_uniq, inv_ind, voxel_counts = np.unique(i, axis=0, return_inverse=True, return_counts=True)
        input_flatten = np.zeros(cube_size[0] * cube_size[1] * cube_size[2], dtype=np.float32)
        i_flatten = i_uniq[:,0] * cube_size[1] * cube_size[2] + i_uniq[:,1] * cube_size[2] + i_uniq[:,2]
        input_flatten[i_flatten] = voxel_counts
        input_3d_count = input_flatten.reshape(cube_size[0], cube_size[1], cube_size[2])      
        input_3d = (input_3d_count > 0).astype(np.float32)

        return input_3d

    def __getitem__(self, index):
        time_idx = index // len(self.peds)
        ped_idx = index % len(self.peds)
        ped = self.peds[ped_idx] # select the corresponding ped
        points_file = os.path.join(self.data_dir, 'points_ped',f'{ped}', f'{time_idx:>05d}.ply')
        joints_file = os.path.join(self.data_dir, 'joints',f'{ped}',f'joints_{time_idx}.txt')
        # test 
        if not os.path.exists(points_file):
            return None, None, None
        pcd = o3d.io.read_point_cloud(points_file)
        pointcloud = np.array(pcd.points)
        # if pointcloud.shape[0] < 50:
        #     return None, None, None
        kp = np.loadtxt(joints_file)
        kp = kp[:,[0,2,1]]
        center = 0.5 * (np.max(pointcloud[:,:2], axis=0, keepdims=True) + np.min(pointcloud[:,:2], axis=0, keepdims=True)) # center is not the mean
        height = np.min(pointcloud[:,2:3], axis=0, keepdims=True)
        uni = np.concatenate([center, height], axis=1)    
        lidar_load = pointcloud - uni
        kp_load = kp - uni
        # augment
        if self.is_train:
            lidar_load, kp_load = self.__augment__(lidar_load, kp_load)
        lidar_center = 0.5 * (np.max(lidar_load, axis=0) + np.min(lidar_load, axis=0))
        input_3d = self.generate_3d_input(lidar_load, lidar_center)
        input_3d = torch.from_numpy(input_3d)
        input_3d = input_3d.unsqueeze(0) # 1 x d x w x h
        kp_load = torch.from_numpy(kp_load)
        lidar_center = torch.from_numpy(lidar_center) # 3

        return input_3d, kp_load, lidar_center
        
