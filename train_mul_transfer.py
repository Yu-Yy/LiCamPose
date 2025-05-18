from utils import utils
import torch
import numpy as np
import os
import argparse # prepare for the arg config
from config import config
from config import update_config
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import shutil
# model and dataset part
from datasets.BasketBall import BasketBall
from model.voxel_fusion_net import VoxelFusionNet
from utils import cameras
import pickle
# using psedo label as the supervised, alongwith the bonelength

from torch.utils.data.dataloader import default_collate
JOINTS_DEF = {
    0: 'neck',
    1: 'nose',
    2: 'mid-hip',
    3: 'l-shoulder',
    4: 'l-elbow',
    5: 'l-wrist',
    6: 'l-hip',
    7: 'l-knee',
    8: 'l-ankle',
    9: 'r-shoulder',
    10: 'r-elbow',
    11: 'r-wrist',
    12: 'r-hip',
    13: 'r-knee',
    14: 'r-ankle',
}

bone_link = np.array([[0, 1], # 0 face
         [0, 2], # 1 body
         [0, 3], # 2 left arm
         [3, 4], # 3    left arm
         [4, 5], # 4   left arm
         [0, 9], # 5 right arm
         [9, 10], # 6  right arm
         [10, 11], # 7 right arm
         [2, 6], # 8 left leg
         [2, 12], # 9 right leg
         [6, 7],   # 10 left leg
         [7, 8],   # 11 left leg
         [12, 13], # 12 right leg
         [13, 14]]) # 13 right leg

symmetric_idx = np.array([[2,5],[3,6],[4,7],[8,9],[10,12],[11,13]])

# bone length for 1 meter, and the bone length symmetric
def get_bone_length(kp):
    '''
    get all the bone length
    '''
    total_length = torch.norm(kp[:,bone_link[:,0],:] - kp[:,bone_link[:,1],:], dim = -1) # [B, 14]
    return total_length

def cal_leg_angle_loss(kp):
    '''
    kp : B, K, 3
    It needs to calculate the forward direction and the leg direction
    '''
    neck_hip = kp[:,2,:] - kp[:,0,:] # B, 3
    neck_ls = kp[:,3,:] - kp[:,0,:] # B, 3
    forward_direction = torch.cross(neck_hip, neck_ls, dim = -1) # B, 3
    forward_direction = forward_direction / torch.norm(forward_direction, dim = -1, keepdim = True)
    # get the leg direction
    # left leg
    left_leg_m = (kp[:,6,:] + kp[:,8,:]) / 2
    left_leg_direction = left_leg_m - kp[:,7,:]
    left_leg_direction = left_leg_direction / torch.norm(left_leg_direction, dim = -1, keepdim = True)
    # right leg
    right_leg_m = (kp[:,12,:] + kp[:,14,:]) / 2
    right_leg_direction = right_leg_m - kp[:,13,:]
    right_leg_direction = right_leg_direction / torch.norm(right_leg_direction, dim = -1, keepdim = True)
    # get the angle
    left_angle_loss = torch.mean(torch.clamp(torch.sum(left_leg_direction * forward_direction, dim=-1), 0))
    right_angle_loss = torch.mean(torch.clamp(torch.sum(right_leg_direction * forward_direction, dim=-1), 0))

    # add the nose/head loss to supervise the head direction
    head_direction = kp[:,1,:] - kp[:,0,:]
    head_direction = head_direction / torch.norm(head_direction, dim = -1, keepdim = True)
    head_angle_loss = torch.mean(torch.clamp(torch.sum( -head_direction * forward_direction, dim=-1), 0))
    return left_angle_loss + right_angle_loss + head_angle_loss

def my_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = config.DDP.MASTER_ADDRESS
    os.environ['MASTER_PORT'] = config.DDP.PORT #'6020' # set the port 
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main(args):
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    logging = utils.Logger(config.DDP.GLOBAL_RANK, config.OUTPUT_DIR)
    writer = utils.Writer(config.DDP.GLOBAL_RANK, os.path.join(config.OUTPUT_DIR, config.LOG_DIR))

    if not os.path.exists(os.path.join(config.OUTPUT_DIR, 'joints_fig')):
        os.makedirs(os.path.join(config.OUTPUT_DIR, 'joints_fig'))


    gpus = [int(i) for i in config.GPUS.split(',')]
    # DATASET SETTING
    train_dataset = BasketBall(config, config.DATA_DIR, is_transfer = True) # it do not have the valid part, it just a finetune
    # train_sampler, valid_sampler = None, None
    
    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size = config.TRAIN.BATCH_SIZE* len(gpus), collate_fn=my_collate_fn,
        shuffle=config.TRAIN.SHUFFLE,pin_memory=True, num_workers=config.WORKERS, drop_last=True)
    
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # warmup is option
    print('=> Constructing models ..')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = VoxelFusionNet(config)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    # set up optimizer
    cnn_optimizer = torch.optim.Adam(model.module.parameters(), lr=config.TRAIN.LR)
    # load the supervised sync model
    sync_model_file = os.path.join('output_BasketBallSync', 'best_model.pt')
    logging.info(f'load the model from sync model {sync_model_file}')
    checkpoint = torch.load(sync_model_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    checkpoint_file = os.path.join(config.OUTPUT_DIR, 'checkpoint.pt')
    best_file = os.path.join(config.OUTPUT_DIR, 'best_model.pt')
    min_loss = torch.tensor(np.inf).to(device)
    if config.TRAIN.RESUME:
        if os.path.exists(checkpoint_file):
            logging.info(f'load the model from checkpoint {checkpoint_file}')
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            init_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # model = model.cuda()
            cnn_optimizer.load_state_dict(checkpoint['optimizer'])
            global_step = checkpoint['global_step']
            # validation part is the training itself
        else:
            logging.info(f'Checkpoint not exists')
            global_step, init_epoch = 0,0
    else:
        global_step, init_epoch = 0,0
    # start training
    logging.info('start training...')
    save_result = {} # initialization
    valid_pse_label = {}
            
    # save all the predicted results and labels index
    for epoch in range(init_epoch, config.TRAIN.END_EPOCH):
        if config.DISTRIBUTED:
            train_queue.sampler.set_epoch(global_step + config.SEED)
        logging.info(f'Epoch {epoch}:')
        global_step, loss_unsp = train(train_queue, model, cnn_optimizer, global_step, writer, logging, save_result, valid_pse_label)
        # validation
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step}, checkpoint_file)
        if loss_unsp < min_loss:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step}, best_file)
            logging.info('New best record !!! ,saving the best model')
            file = os.path.join(config.OUTPUT_DIR, 'thu_ct_transfer.pkl')
            with open(file,'wb') as f:
                pickle.dump(save_result, f)
            min_loss = loss_unsp
        writer.close()
    


def train(train_queue, model, cnn_optimizer, global_step, writer, logging, save_result, valid_pse_label):
    project_loss_c = utils.AvgrageMeter()
    prior_loss_c = utils.AvgrageMeter()
    max_length_loss_c = utils.AvgrageMeter()
    symmetric_loss_c = utils.AvgrageMeter()
    ref_loss_c = utils.AvgrageMeter()
    leg_angle_loss_c = utils.AvgrageMeter()
    total_loss_c = utils.AvgrageMeter()
    average_entropy_c = utils.AvgrageMeter()
    model.train()

    for step, data in enumerate(train_queue):
        if len(data) == 0:
            continue
        input3d, input_heatmap, pred_2d, projectionM, grid_centers, ped, time = data
        # ref_pose = ref_pose.cuda()
        input3d = input3d.cuda()
        pred_2d = [x.cuda() for x in pred_2d]
        input_heatmap = [x.cuda() for x in input_heatmap]
        grid_centers = grid_centers.cuda()
        # calculate loss with the pred_kp
        # B x J x 3
        pred_kp_en,_ = model(input3d, input_heatmap, projectionM, grid_centers)
        pred_kp = pred_kp_en[:, :, :3]
        pred_en = pred_kp_en[:, :, 3]
        batch_size = input3d.shape[0]
        # loss calculate
        # ref pose loss
        pred_pose_entro = torch.max(pred_en, dim = -1)[0] # only get the max value
        average_entropy_c.update(torch.mean(pred_pose_entro).item())
        ref_pose_loss = []

        valid_pose = torch.zeros(batch_size, 1).cuda()
        for b in range(batch_size):
            # judge the pse label
            if time[b].item() not in valid_pse_label.keys(): # create the time
                valid_pse_label[time[b].item()] = {}
            if ped[b].item() in valid_pse_label[time[b].item()].keys():
                if valid_pse_label[time[b].item()][ped[b].item()]:
                    ref_poss = torch.tensor(save_result[time[b].item()][ped[b].item()][:,:3]).cuda()
                    ref_pose_loss.append(torch.sum(torch.norm(pred_kp[b,:,:3] - ref_poss, dim=1), dim=0))
            # update the pse label
            if pred_pose_entro[b] < 6: # setting the entropy threshold
                valid_pse_label[time[b].item()][ped[b].item()] = True
                valid_pose[b] = 1
            else:
                valid_pse_label[time[b].item()][ped[b].item()] = False # reset the pse label
        if len(ref_pose_loss) > 0:
            ref_pose_loss = torch.stack(ref_pose_loss).mean()
        else:
            ref_pose_loss = torch.tensor(0).cuda()


        # valid_pose = ref_pose[:,:,3]
        # ref_pose_loss = torch.mean(torch.sum(torch.norm(pred_kp - ref_pose[:,:,:3], dim=2) * valid_pose, dim=1))
        project_loss = 0
        for c in range(len(input_heatmap)):
            cam = {}
            for k, v in projectionM[c].items():
                cam[k] = v[0] # it is not related to batch
            xy = cameras.project_pose(pred_kp.reshape(-1 , 3), cam)
            pred_2d_view = pred_2d[c]
            project_2d_view = xy.reshape(batch_size, -1, 2)
            pred_2d_kp = pred_2d_view[...,:2]
            pred_2d_conf = pred_2d_view[...,2]
            project_loss += torch.mean(torch.sum(torch.norm(pred_2d_kp - project_2d_view, dim=-1) * pred_2d_conf,  dim=-1)) #* valid_pose
        # add the bone symmetry loss, and the bone length loss (hyperparameter)
        bone_length = get_bone_length(pred_kp)
        max_loss = torch.mean(torch.sum(torch.clamp(bone_length - 0.7, 0), dim=-1)) # give the detailed length for each limb
        min_loss = torch.mean(torch.sum(torch.clamp(0.1 - bone_length, 0), dim=-1)) # limit the min length of each limb
        symmetric_loss = torch.mean(torch.sum(torch.abs(bone_length[:,symmetric_idx[:,0]] - bone_length[:,symmetric_idx[:,1]]), dim=-1))
        # add the angle loss for legs
        leg_angle_loss = cal_leg_angle_loss(pred_kp)
        prior_loss = max_loss + symmetric_loss + min_loss + leg_angle_loss
        # chamfer loss for pointcloud?
        if ref_pose_loss > 0:
            total_loss = 0.1 * project_loss  + ref_pose_loss + 10 * prior_loss
        else:
            total_loss = 0.1 * project_loss + 10 * prior_loss # decay weight of project loss
        if torch.isnan(total_loss):
            import pdb;pdb.set_trace() # for nan of inf debug
        ref_loss_c.update(ref_pose_loss.item())
        project_loss_c.update(project_loss.item())
        total_loss_c.update(total_loss.item())
        prior_loss_c.update(prior_loss.item())
        max_length_loss_c.update(max_loss.item())
        symmetric_loss_c.update(symmetric_loss.item())
        leg_angle_loss_c.update(leg_angle_loss.item())
        cnn_optimizer.zero_grad()
        total_loss.backward()
        cnn_optimizer.step()
        for b in range(batch_size):
            if time[b].item() not in save_result.keys():
                save_result[time[b].item()] = {}
            save_result[time[b].item()][ped[b].item()] = pred_kp_en[b].detach().cpu().numpy()

        if (global_step + 1) % config.PRINT_FREQ == 0:
            # writer
            writer.add_scalar('train/loss', total_loss, global_step)
            writer.add_scalar('train/project_loss', project_loss, global_step)
            logging.info(f'{global_step}: The total loss is {total_loss_c.avg}') 
            logging.info(f'The project_loss is {project_loss_c.avg}')
            logging.info(f'The prior_loss is {prior_loss_c.avg}')
            logging.info(f'The max_length_loss is {max_length_loss_c.avg}')
            logging.info(f'The symmetric_loss is {symmetric_loss_c.avg}')
            logging.info(f'The ref_pose_loss is {ref_loss_c.avg}')
            logging.info(f'The leg_angle_loss is {leg_angle_loss_c.avg}')
            logging.info(f'The entropy is {average_entropy_c.avg}')
            # logging.info(f'The rela loss is {rela_loss_c.avg}')
            pred_kp_plot = pred_kp[0].cpu().detach().numpy()
            gt_kp_plot = pred_kp[0].cpu().detach().numpy()
            file_name = os.path.join(config.OUTPUT_DIR, 'joints_fig', f'train_{global_step}_{step}.jpg')
            utils.kp_plot(pred_kp_plot, gt_kp_plot, file_name)

        global_step += 1

        del input3d
        del grid_centers
        del pred_2d
        del input_heatmap
        del projectionM

    return global_step, total_loss_c.avg

if __name__ == '__main__':
    args = parse_args()
    create_exp_dir(config.OUTPUT_DIR)
    size = config.DDP.NUM_PROCESS_PER_NODE
    main(args) 