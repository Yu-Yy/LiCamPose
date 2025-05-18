import numpy as np
import torch.utils.data as data
import torch
from glob import glob
import os
import open3d as o3d
import os.path as osp
import json
import pickle
from utils.cameras_cpu import project_pose, undistort_pose2d

validate_scenes = ["160422_ultimatum1","160224_haggling1" ,"160226_haggling1","161202_haggling1",\
                 "160906_ian1", "160906_ian2", "160906_ian3","160906_band1","160906_band2","160906_band3"]


M = np.array([[1,0,0],[0,0,-1],[0,1,0]])

# This is the only-vadiation dataset
class Panoptic(data.Dataset):
    def __init__(self, cfg, datadir, is_transfer = True): #
        super().__init__()
        self.space_size = np.array(cfg.PICT_STRUCT.GRID_SIZE)
        self.initial_cube_size = np.array(cfg.PICT_STRUCT.CUBE_SIZE)
        self.datadir = datadir
        self.scene = datadir.split('/')[-1]
        self.points_ped_folder = os.path.join(datadir, 'sorted_data','points_ped')
        self.points_ped_folder_files = glob(os.path.join(self.points_ped_folder,'*.ply'))
        self.gt3d_folder = osp.join(datadir, 'hdPose3d_stage1_coco19')
        self.kp_2d_folder = osp.join(datadir, 'sorted_data', 'pred_2d')
        self.views_num = 5 # TODO: female is 3
        self.kp_num = cfg.NUM_JOINTS
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE) # fixed the array size #np.array(cfg.NETWORK.IMAGE_SIZE) # changed the image size
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE) # fixed the heatmap size cfg.NETWORK.HEATMAP_SIZE
        self.sigma = cfg.NETWORK.SIGMA
        # find each scene's and each view's camera parameter
        self.cameras = self._get_cam()
        # get the valid index
        self.is_transfer = is_transfer # represent the transfer learning
        self.camera_names = ['00_03', '00_06', '00_12', '00_13', '00_23']


    def __len__(self):
        # it should not fix
        return len(self.points_ped_folder_files) #1000 * 10 # TODO: fix length

    def _get_cam(self):
        cam_path = glob(osp.join(self.datadir,'calibration_*.json'))[0]
        with open(cam_path, 'r') as f:
            calib_data = json.load(f)
        cameras = {}
        cam_list = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)] # fixed camera
        idx = 0
        for cam in calib_data["cameras"]:
            if (cam['panel'], cam['node']) in cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['T'] = np.array(cam['t']).reshape((3, 1)) / 100
                sel_cam['fx'] = np.array(sel_cam['K'][0, 0])
                sel_cam['fy'] = np.array(sel_cam['K'][1, 1])
                sel_cam['cx'] = np.array(sel_cam['K'][0, 2])
                sel_cam['cy'] = np.array(sel_cam['K'][1, 2])
                sel_cam['k'] = sel_cam['distCoef'][[0, 1, 4]].reshape(3, 1)
                sel_cam['p'] = sel_cam['distCoef'][[2, 3]].reshape(2, 1)
                cameras[idx] = sel_cam 
                idx += 1
            
        return cameras
    
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

        kpgt_file = os.path.join(self.gt3d_folder, f'body3DScene_{time_idx:08d}.json') 
        with open(kpgt_file , 'r') as f:
            kp_row_data = json.load(f)
        kp3d_body_datas = kp_row_data['bodies'] 
        for kp3d_body_data in kp3d_body_datas:
            if kp3d_body_data['id'] == ped:
                kp_gt3d = np.array(kp3d_body_data['joints19'])
                kp_gt3d = kp_gt3d.reshape(-1, 4)
                kp_gt3d[:,:3] = kp_gt3d[:,:3] / 100 # cm to m
                kp_gt3d = kp_gt3d[:15,:]
                kp_gt3d[:,:3] = kp_gt3d[:,:3].dot(M)
                break
        kp_gt3d = torch.from_numpy(kp_gt3d).float()
        pcd = o3d.io.read_point_cloud(pcd_file)
        lidar_load = np.array(pcd.points)
        lidar_load = lidar_load.dot(M)

        lidar_center = 0.5 * (np.max(lidar_load, axis=0) + np.min(lidar_load, axis=0))
        input_3d = self.generate_3d_input(lidar_load, lidar_center)
        input_3d = torch.from_numpy(input_3d)
        input_3d = input_3d.unsqueeze(0) # 1 x d x w x h
        lidar_center = torch.from_numpy(lidar_center)
        # find the kp info
        pred_2d_info = []
        for v in range(self.views_num):
            kp_file = osp.join(self.kp_2d_folder, self.camera_names[v], f'{time_idx:0>8d}_{ped:0>3d}.json')
            if osp.exists(kp_file):
                with open(kp_file, 'r') as f:
                    kp_2d = json.load(f)
                kp_2d = np.array(kp_2d)
                pred_2d_info.append(kp_2d)
            else:
                pred_2d_info.append(np.zeros((self.kp_num, 3)))
        input_heatmap = self.process_2d_infomation(pred_2d_info)  # TODO: the distort is only for the dlt testing

        kp_2d_gt = []
        width = 1920
        height = 1080
        for v in range(self.views_num):
            cam = self.cameras[v]
            valid_3d = (kp_gt3d[..., 3].numpy() > 0.1)
            project_2d = project_pose(kp_gt3d[...,:3].numpy(), cam)
            project_2d = undistort_pose2d(project_2d, cam)
            x_check = np.bitwise_and(project_2d[:, 0] >= 0,
                                     project_2d[:, 0] <= width - 1)
            y_check = np.bitwise_and(project_2d[:, 1] >= 0,
                                     project_2d[:, 1] <= height - 1)
            check = np.bitwise_and(x_check, y_check)
            check = np.bitwise_and(check, valid_3d)
            project_2d = np.concatenate([project_2d, check[:, None]], axis=1)
            kp_2d_gt.append(project_2d)

        # all is zeros
        valid = 0
        for v in range(self.views_num):
            valid += torch.sum(input_heatmap[v])
        if valid == 0:
            return None, None, None, None, None, None, None, None, None, None
        # return the test information
        ped = torch.tensor(ped)
        time_idx = torch.tensor(time_idx)
        for v in range(self.views_num):
            pred_2d_info[v] = torch.tensor(pred_2d_info[v])
        return input_3d, input_heatmap, pred_2d_info, kp_2d_gt, self.cameras, lidar_center,  ped, time_idx, self.scene, kp_gt3d

    @staticmethod
    def compute_human_scale(pose, joints_vis):
        idx = joints_vis[:, 0] >= 0.2
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        return np.clip(np.maximum(maxy - miny, maxx - minx) ** 2, 1.0 / 4 * 96 ** 2, 4 * 96 ** 2)
