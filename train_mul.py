from utils import utils
import torch
import numpy as np
import os
import argparse # prepare for the arg config
from config import config
from config import update_config
import torch.backends.cudnn as cudnn
import shutil
# model and dataset part

from datasets.BasketBallSync import SyntheticDataM
from model.voxel_fusion_net import VoxelFusionNet
from tqdm import tqdm

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
    train_dataset = SyntheticDataM(config, config.DATA_DIR, is_train=True) # changed the datadir
    valid_dataset = SyntheticDataM(config, config.DATA_DIR, is_train=False)
    
    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size = config.TRAIN.BATCH_SIZE* len(gpus), collate_fn=my_collate_fn,
        shuffle=config.TRAIN.SHUFFLE,pin_memory=True, num_workers=config.WORKERS, drop_last=True)
    
    valid_queue = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.TEST.BATCH_SIZE* len(gpus), collate_fn=my_collate_fn,
        shuffle=False, pin_memory=True, num_workers=1, drop_last=False)

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
    # no scheduler

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
            # validation
            min_loss = validate(valid_queue, model, logging, writer,args, init_epoch)
        else:
            logging.info(f'Checkpoint not exists')
            global_step, init_epoch = 0,0
    else:
        global_step, init_epoch = 0,0
    # start training
    logging.info('start training...')
    
    for epoch in range(init_epoch, config.TRAIN.END_EPOCH):
        logging.info(f'Epoch {epoch}:')
        global_step = train(train_queue, model, cnn_optimizer, global_step, writer, logging)
        # validation
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step}, checkpoint_file)
        output_re_loss = validate(valid_queue, model, logging, writer, args, epoch)
        if output_re_loss < min_loss:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step}, best_file)
            logging.info('New best record !!! ,saving the best model')
            min_loss = output_re_loss
        writer.close()
    

def train(train_queue, model, cnn_optimizer, global_step, writer, logging):
    reconstruction_loss_c = utils.AvgrageMeter()
    rela_loss_c = utils.AvgrageMeter()
    total_loss_c = utils.AvgrageMeter()
    model.train()

    for step, data in tqdm(enumerate(train_queue)):
        if len(data) == 0:
            continue
        input3d, kp_gt, grid_centers, input_heatmap, output_cam = data
        input3d = input3d.cuda()
        input_heatmap = [x.cuda() for x in input_heatmap]
        output_cam = [x.cuda() for x in output_cam]
        grid_centers = grid_centers.cuda()
        kp_gt = kp_gt.cuda()
        # import pdb;pdb.set_trace()
        pred_kp, _ = model(input3d,input_heatmap, output_cam, grid_centers)
        # calculate loss with the pred_kp
        # B x J x 3
        pred_kp = pred_kp[...,:3]
        # using the kp_idx to index the heatmap

        rec_loss = torch.mean(torch.sum(torch.norm((pred_kp - kp_gt), p=1, dim=-1),dim = -1)) # direct compute
        pred_kp_rela = pred_kp - pred_kp[:,2:3,:] # relative pose
        kp_gt_rela = kp_gt - kp_gt[:,2:3,:]
        rela_loss = torch.mean(torch.sum(torch.norm((pred_kp_rela - kp_gt_rela), p=1, dim=-1),dim = -1)) 
        total_loss = rela_loss + rec_loss  # add weight for the reconstruction loss? maybe?+ heatmap_loss
        reconstruction_loss_c.update(rec_loss.item())
        rela_loss_c.update(rela_loss.item())
        total_loss_c.update(total_loss.item())
        cnn_optimizer.zero_grad()
        total_loss.backward()
        cnn_optimizer.step()

        if (global_step + 1) % config.PRINT_FREQ == 0:
            # writer
            writer.add_scalar('train/loss', total_loss, global_step)
            writer.add_scalar('train/re_loss', rec_loss, global_step)
            writer.add_scalar('train/rela_loss', rela_loss, global_step)
            # writer.add_scalar('train/heatmap_loss', heatmap_loss, global_step)
            logging.info(f'{global_step}: The total loss is {total_loss_c.avg}') 
            logging.info(f'The reconc loss is {reconstruction_loss_c.avg}')
            logging.info(f'The rela loss is {rela_loss_c.avg}')
            # logging.info(f'The heatmap loss is {heatmap_loss_c.avg}')

        global_step += 1

        del input3d
        del grid_centers
        del kp_gt
        del input_heatmap
        del output_cam

    return global_step


def validate(valid_queue, model, logging, writer, args, epoch):
    reconstruction_loss_c = utils.AvgrageMeter()
    rela_loss_c = utils.AvgrageMeter()
    total_loss_c = utils.AvgrageMeter()
    MPJPE = utils.AvgrageMeter()
    RA_MPJPE = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(valid_queue):
            if len(data) == 0:
                continue
            input3d, kp_gt, grid_centers, input_heatmap, output_cam = data
            input3d = input3d.cuda()
            input_heatmap = [x.cuda() for x in input_heatmap]
            output_cam = [x.cuda() for x in output_cam]
            grid_centers = grid_centers.cuda()
            kp_gt = kp_gt.cuda()
            pred_kp,_ = model(input3d,input_heatmap, output_cam, grid_centers)
            pred_kp = pred_kp[...,:3]

            # calculate loss with the pred_kp
            # B x J x 3
            rec_loss = torch.mean(torch.sum(torch.norm((pred_kp - kp_gt), p='fro', dim=-1),dim = -1)) # direct compute
            rec_avg_loss = torch.mean(torch.norm((pred_kp - kp_gt), p='fro', dim=-1))
            MPJPE.update(rec_avg_loss.item())
            pred_kp_rela = pred_kp - pred_kp[:,2:3,:] # relative pose
            kp_gt_rela = kp_gt - kp_gt[:,2:3,:]
            rela_loss = torch.mean(torch.sum(torch.norm((pred_kp_rela - kp_gt_rela), p='fro', dim=-1),dim = -1)) 
            rela_avg_loss = torch.mean(torch.norm((pred_kp_rela - kp_gt_rela), p='fro', dim=-1)) 
            RA_MPJPE.update(rela_avg_loss.item())
            total_loss = rela_loss + rec_loss
            reconstruction_loss_c.update(rec_loss.item())
            rela_loss_c.update(rela_loss.item())
            total_loss_c.update(total_loss.item())

            if (step + 1) % config.PRINT_FREQ == 0:
                # writer
                logging.info(f'VAL {step}: The total loss is {total_loss_c.avg}') 
                logging.info(f'The reconc loss is {reconstruction_loss_c.avg}')
                logging.info(f'The rela loss is {rela_loss_c.avg}')
                # plot a kp figure
                pred_kp_plot = pred_kp[0].cpu().detach().numpy()
                gt_kp_plot = kp_gt[0].cpu().detach().numpy()
                file_name = os.path.join(config.OUTPUT_DIR, 'joints_fig', f'{epoch}_{step}.jpg')
                utils.kp_plot(pred_kp_plot, gt_kp_plot, file_name)
            del input3d
            del grid_centers
            del kp_gt
            del input_heatmap
            del output_cam

    logging.info(f'Final : The total loss is {total_loss_c.avg}, rec loss is {reconstruction_loss_c.avg}, rela loss {rela_loss_c.avg}')
    logging.info(f'MPJPE is {MPJPE.avg}, RA_MPJPE is {RA_MPJPE.avg}')
    return total_loss_c.avg


if __name__ == '__main__':
    args = parse_args()
    create_exp_dir(config.OUTPUT_DIR)
    size = config.DDP.NUM_PROCESS_PER_NODE
    main(args)