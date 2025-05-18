import pickle
from utils import utils
import utils.transforms as transforms
import torch
import numpy as np
import os
import argparse # prepare for the arg config
from config import config
from config import update_config
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import shutil
from datasets.panoptic import Panoptic # the new 
from torch.utils.data.dataloader import default_collate


compare = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] # total0,1,2,

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
    # DATASET SETTIN
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # warmup is option
    print('=> Constructing models ..')
    model = eval(config.MODEL)(config) # test the 
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    best_file = os.path.join('/path/to/panoptic', 'best_model.pt')
    # time seq
    valid_scene = ["160906_pizza1","160422_haggling1","160906_ian5", "160906_band2"] # + validate_scene
    valid_dataset = []
    for scene in valid_scene:
        valid_dataset.append(Panoptic(config, os.path.join(config.DATA_DIR, scene), is_transfer = True))
    valid_dataset = torch.utils.data.ConcatDataset(valid_dataset)
    valid_queue = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=config.TEST.BATCH_SIZE* len(gpus), collate_fn=my_collate_fn,
                    shuffle=False, pin_memory=True, num_workers=1, drop_last=False)
    save_result = {}
    for scene in valid_scene:  
        save_result[scene] = {}
    if os.path.exists(best_file):
        logging.info(f'load the model from checkpoint {best_file}')
        checkpoint = torch.load(best_file, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        # validation
        logging.info(f'Testing')
        mpjpe, ra_mpjpe, pa_mpjpe,entropy = validate(valid_queue, model, logging, writer,args, 'validating', save_result)
        # time_end = time.time()
        logging.info(f'Testing mpjpe is {mpjpe}m, rela_mpjpe is {ra_mpjpe}m, pa_mpjpe is {pa_mpjpe},entropy is {entropy}')
        file = os.path.join(config.OUTPUT_DIR, f'panoptic_transfer_validate.pkl')
        with open(file,'wb') as f:
            pickle.dump(save_result, f)
    else:
        logging.info(f'BestFile not exists')
    

def validate(valid_queue, model, logging, writer, args, epoch, save_result):
    MPJPE = utils.AvgrageMeter()
    RA_MPJPE = utils.AvgrageMeter()
    PA_MPJPE = utils.AvgrageMeter()
    entropy = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(valid_queue):
            if len(data) == 0:
                continue
            input3d, input_heatmap, _, _,projectionM, grid_centers, ped, time, scene, kpgt = data
            kpgt = kpgt.cuda()
            input3d = input3d.cuda()
            input_heatmap = [x.cuda() for x in input_heatmap]
            grid_centers = grid_centers.cuda()
            pred_kp, _ = model(input3d, input_heatmap, projectionM, grid_centers) # 
            # calculate loss with the pred_kp
            # B x J x 3
            kpgt_cord = kpgt[:, :, :3]
            kpgt_cord_vis = kpgt[:, :, 3:4] > 0.1
            kpgt_cord_vis = kpgt_cord_vis[:,compare,:]
            if torch.sum(kpgt_cord_vis) == 0:
                continue
            entropy_b = torch.mean(torch.max(pred_kp[...,-1], dim=-1)[0])
            entropy.update(entropy_b.item())
            pred_kp = pred_kp[...,:3] # except the entropy
            rec_loss = torch.sum(torch.norm((pred_kp[:,compare,:] - kpgt_cord[:,compare,:]) * kpgt_cord_vis, p='fro', dim=-1)) /  torch.sum(kpgt_cord_vis)
            
            # calculate the relative mpjpe
            pred_kp_rela = pred_kp - pred_kp[:,2:3,:] # relative pose
            kp_gt_rela = kpgt_cord  - kpgt_cord[:,2:3,:]
            rec_loss_rela = torch.sum(torch.norm((pred_kp_rela[:,compare,:] - kp_gt_rela[:,compare,:]) * kpgt_cord_vis, p='fro', dim=-1)) /  torch.sum(kpgt_cord_vis)
            RA_MPJPE.update(rec_loss_rela.item())

            MPJPE.update(rec_loss.item())
            batch_size = input3d.shape[0]
            
            for b in range(batch_size):
                # calculate the pa-mpjpe
                vis_label = kpgt_cord_vis[b,:,0]
                if torch.sum(vis_label) == 0:
                    continue
                align_pose = transforms.procrustes_transform(kpgt_cord[:,compare,:][b,vis_label,:].cpu().numpy(), pred_kp[:,compare,:][b,vis_label,:].cpu().numpy())
                pa_mpjpe = torch.mean(torch.norm(torch.from_numpy(align_pose).cuda() - kpgt_cord[:,compare,:][b,vis_label,:], p='fro', dim=-1))
                PA_MPJPE.update(pa_mpjpe.item())
                if time[b].item() not in save_result[scene[b]].keys():
                    save_result[scene[b]][time[b].item()] = {}
                save_result[scene[b]][time[b].item()][ped[b].item()] = pred_kp[b].cpu().numpy()
            if (step + 1) % config.PRINT_FREQ == 0:
                # writer
                logging.info(f'VAL {step}: The mpjpe is {MPJPE.avg}, the rela mpjpe is {RA_MPJPE.avg}, the pa mpjpe is {PA_MPJPE.avg}') 
                logging.info(f'Entropy is {entropy.avg}')
                # logging.info(f'The rela loss is {rela_loss_c.avg}')
                # plot a kp figure
                pred_kp_plot = pred_kp[0].cpu().detach().numpy()
                gt_kp_plot = kpgt_cord[0].cpu().detach().numpy()
                file_name = os.path.join(config.OUTPUT_DIR, 'joints_fig', f'{epoch}_{step}.jpg')
                utils.kp_plot(pred_kp_plot, gt_kp_plot, file_name)
    return MPJPE.avg, RA_MPJPE.avg, PA_MPJPE.avg, entropy.avg 


if __name__ == '__main__':
    args = parse_args()
    create_exp_dir(config.OUTPUT_DIR)
    size = config.DDP.NUM_PROCESS_PER_NODE
    main(args)