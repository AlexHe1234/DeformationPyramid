from model.geometry import *
import os
import torch
from tqdm import tqdm
import argparse
# from data._4DMatch import _4DMatch
from correspondence.datasets._4dmatch import _4DMatch
from model.registration import Registration
import yaml
from easydict import EasyDict as edict
from model.loss import compute_flow_metrics
from utils.benchmark_utils import setup_seed
from utils.utils import Logger, AverageMeter
from utils.tiktok import Timers
from ma_dataset import MixamoAMASS


def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)

setup_seed(0)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help= 'Path to the config file.')
    parser.add_argument('--visualize', action = 'store_true', help= 'visualize the registration results')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['folder'], config['exp_dir'])
    os.makedirs(config['snapshot_dir'], exist_ok=True)


    config = edict(config)


    # backup the experiment
    os.system(f'cp -r config {config.snapshot_dir}')
    os.system(f'cp -r data {config.snapshot_dir}')
    os.system(f'cp -r model {config.snapshot_dir}')
    os.system(f'cp -r utils {config.snapshot_dir}')


    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')


    model = Registration(config)
    timer = Timers()

    # splits = ['4DMatch-F', '4DLoMatch-F']

    # for benchmark in splits:

    # config.split['test'] = benchmark

    D = MixamoAMASS(split='test', root_dir='data/mixamo_cmu')  # TODO:
    # logger = Logger(  os.path.join( config.snapshot_dir, benchmark+".log" ))

    stats_meter = None

    for i in tqdm( range( len(D))):
        # src_pcd, tar_pcd, flow_gt = dat

        batch = D.__getitem__(i)
        frame = len(batch['points'])
        
        traj_pred = []
        
        for f in range(frame):
            if f == 0:
                src_pcd = batch['points_mesh']
                tgt_pcd = batch['points'][0]
                # flow_gt = batch['tracks'][0] - batch['points_mesh']
            else:
                src_pcd = warped_pcd.detach()
                tgt_pcd = batch['points'][f]
                # flow_gt = batch['tracks'][1] - 
                # TODO:

            """obtain overlap mask"""
            overlap = np.ones(len(src_pcd))

            model.load_pcds(src_pcd, tgt_pcd)

            timer.tic("registration")
            warped_pcd, iter_cnt, timer = model.register(visualize=args.visualize, timer = timer)
            timer.toc("registration")
            # flow = warped_pcd - model.src_pcd
            # metric_info = compute_flow_metrics(flow, flow_gt, overlap=overlap)
            traj_pred.append(warped_pcd.detach().clone())

            # if stats_meter is None:
            #     stats_meter = dict()
            #     for key, _ in metric_info.items():
            #         stats_meter[key] = AverageMeter()
            # for key, value in metric_info.items():
                # stats_meter[key].update(value)
        breakpoint()


    # # note down flow scores on a benchmark
    # message = f'{i}/{len(D)}: '
    # for key, value in stats_meter.items():
    #     message += f'{key}: {value.avg:.3f}\t'
    # logger.write(message + '\n')
    # print( "score on ", benchmark, '\n', message)


    # # note down average time cost
    # print('time cost average')
    # for ele in timer.get_strings():
    #     logger.write(ele + '\n')
    #     print(ele)
