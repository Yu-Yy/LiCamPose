import numpy as np
import torch.utils.data as data
import torch
from glob import glob
import os
import open3d as o3d
import os.path as osp
import json

# This is the only-vadiation dataset
class BasketBall(data.Dataset):
    def __init__(self, cfg, datadir, is_transfer = True): #
        super().__init__()
        self.space_size = np.array(cfg.PICT_STRUCT.GRID_SIZE)
        self.num_joints = cfg.NUM_JOINTS
        self.initial_cube_size = np.array(cfg.PICT_STRUCT.CUBE_SIZE)
        self.datadir = datadir
        self.points_ped_folder = os.path.join(datadir, 'points_ped')
        self.points_ped_folder_files = glob(os.path.join(self.points_ped_folder,'*.ply'))
        self.kp_2d_folder = osp.join(datadir, 'pred2d_pose')
        self.views_num = 4 #
        self.kp_num = cfg.NUM_JOINTS
        self.image_size = np.array([2048, 1536]) # fixed the array size 
        self.heatmap_size = np.array([512, 384]) # fixed the heatmap size cfg.NETWORK.HEATMAP_SIZE
        self.sigma = cfg.NETWORK.SIGMA
        # another camrea_view mod, the orignal meta
        self.cameras = self._get_cam()
        # get the valid index
        self.is_transfer = is_transfer # represent the transfer learning
        self.camera_names = ['camera_0', 'camera_1', 'camera_2', 'camera_3'] 

    def __len__(self):
        # it should not fix
        return len(self.points_ped_folder_files) #1000 * 10 # TODO: fix length

    def _get_cam(self):
        cam_file = osp.join('datasets', 'calibrations','BasketBall', "calibration.json")
        with open(cam_file) as cfile:
            cameras = json.load(cfile)
        new_cam = {}
        for v in range(self.views_num):
            new_cam[v] = {}
        for idx, (id, cam) in enumerate(cameras.items()): # id is 0 1 2 3
            for k, v in cam.items():
                new_cam[idx][k] = np.array(v)
            
            new_cam[idx]['k'] = np.zeros_like(new_cam[idx]['k'])
            new_cam[idx]['p'] = np.zeros_like(new_cam[idx]['p']) # zerolized the distortion part
        return new_cam
    
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

        # # 
        i = np.concatenate([i_x[:,np.newaxis], i_y[:,np.newaxis], i_z[:,np.newaxis]], axis=1)
        i_uniq, inv_ind, voxel_counts = np.unique(i, axis=0, return_inverse=True, return_counts=True)
        input_flatten = np.zeros(cube_size[0] * cube_size[1] * cube_size[2], dtype=np.float32)
        i_flatten = i_uniq[:,0] * cube_size[1] * cube_size[2] + i_uniq[:,1] * cube_size[2] + i_uniq[:,2]
        input_flatten[i_flatten] = voxel_counts
        input_3d_count = input_flatten.reshape(cube_size[0], cube_size[1], cube_size[2])      
        input_3d = (input_3d_count > 0).astype(np.float32)

        return input_3d

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
                if joints_vis[n][joint_id, 0] == 0 or \
                        ul[0] >= self.heatmap_size[0] or \
                        ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    continue
                
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

    def __getitem__(self, index):
        pcd_file = self.points_ped_folder_files[index]
        basename = os.path.basename(pcd_file)
        basename = basename.split('.')[0]
        time_idx, ped = basename.split('_')
        time_idx = int(time_idx)
        ped = int(ped)

        pcd = o3d.io.read_point_cloud(pcd_file)
        lidar_load = np.array(pcd.points)
        lidar_center = 0.5 * (np.max(lidar_load, axis=0) + np.min(lidar_load, axis=0))
        input_3d = self.generate_3d_input(lidar_load, lidar_center)
        input_3d = torch.from_numpy(input_3d)
        input_3d = input_3d.unsqueeze(0) # 1 x d x w x h
        lidar_center = torch.from_numpy(lidar_center)
        # find the kp info
        pred_2d_info = []
        for v in range(self.views_num):
            kp_file = osp.join(self.kp_2d_folder, self.camera_names[v], f'{time_idx:0>5d}_{ped:0>4d}.json')
            if osp.exists(kp_file):
                with open(kp_file, 'r') as f:
                    kp_2d = json.load(f)
                pred_2d_info.append(np.array(kp_2d))
            else:
                pred_2d_info.append(np.zeros((self.kp_num, 3)))

        input_heatmap = self.process_2d_infomation(pred_2d_info)
        # return the test information
        ped = torch.tensor(ped)
        time_idx = torch.tensor(time_idx)
        for v in range(self.views_num):
            pred_2d_info[v] = torch.tensor(pred_2d_info[v])
        # ref_pose = torch.tensor(ref_pose)
        return input_3d, input_heatmap, pred_2d_info, self.cameras, lidar_center,  ped, time_idx

    @staticmethod
    def compute_human_scale(pose, joints_vis):
        idx = joints_vis[:, 0] >= 0.2
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        return np.clip(np.maximum(maxy - miny, maxx - minx) ** 2, 1.0 / 4 * 96 ** 2, 4 * 96 ** 2)
