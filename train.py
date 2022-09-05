import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import torch
# torch.backends.cudnn.benchmark=True
import torchvision
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import tqdm

import argparse
import time
from pathlib import Path
from math import sqrt
import os, tqdm
from util import save_id

from util import add_salt_pepper, init_distributed_mode, pprint
from dataset import get_dataset, get_dataloader
from trainer import AnomalyTrainer
from hps import HPS_VQVAE as HPS
from helper import get_device, get_parameter_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--root_path', type=str, default='/media/mw/DATA/dataset/FMD/FMD')
    parser.add_argument('--root_path', type=str, default='/home/matrixai/mw/dataset/FMD')
    parser.add_argument('--task', type=str, default='anomaly')
    parser.add_argument('--load-path', type=str, default='runs/anomaly-2022-07-28_21-06-00/checkpoints/anomaly-state-dict-0045.pt')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--syncBN', type=bool, default=False)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', type=bool, default=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    args = parser.parse_args()
    init_distributed_mode(args=args)
    cfg = HPS[args.task]

    
    pprint(f"> Initialising VQ-VAE-2 model")
    trainer = AnomalyTrainer(cfg, args)
    pprint(f"> Number of parameters: {get_parameter_count(trainer.net)}")
    if args.load_path:
        pprint(f"> Loading model parameters from checkpoint")
        trainer.load_checkpoint(args.load_path,args, args.device)
    if args.batch_size:
        cfg.batch_size = args.batch_size

    if args.evaluate:
        pprint(f"> Loading {cfg.display_name} dataset")
        _, val_loader = get_dataloader(args.root_path, cfg, shuffle_test=True)
        pprint(f"> Generating evaluation batch of reconstructions")
        file_name = f"./recon-{save_id}-eval.{'jpg' if args.save_jpg else 'png'}"
        nb_generated = 0
        imgs = []
        pb = tqdm(total=cfg.batch_size)
        for x, _ in val_loader:
            *_, y = trainer.eval(x)
            imgs.append(y.cpu())
            nb_generated += y.shape[0]
            pb.update(y.shape[0])
            if nb_generated >= cfg.batch_size:
                break
        pprint(f"> Assembling Image")
        save_image(torch.cat(imgs, dim=0), file_name, nrow=int(sqrt(cfg.batch_size)), normalize=True, value_range=(-1,1))
        pprint(f"> Saved to {file_name}")
        exit()
        
    if not args.no_save:
        runs_dir = Path(f"runs")
        root_dir = runs_dir / f"{args.task}-{save_id}"
        chk_dir = root_dir / "checkpoints"
        img_dir = root_dir / "images"
        log_dir = root_dir / "logs"

        runs_dir.mkdir(exist_ok=True)
        root_dir.mkdir(exist_ok=True)
        chk_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)

    pprint(f"> Loading {cfg.display_name} dataset")
    root_path = 'D:/dataset/FMD/FMD'
    if not args.no_save:
        runs_dir = Path(f"runs")
        root_dir = runs_dir / f"{args.task}-{save_id}"
        chk_dir = root_dir / "checkpoints"
        img_dir = root_dir / "images"
        log_dir = root_dir / "logs"

        runs_dir.mkdir(exist_ok=True)
        root_dir.mkdir(exist_ok=True)
        chk_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
    train_dataset, val_dataset = get_dataset(args.root_path, cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,shuffle=False)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, cfg.mini_batch_size, drop_last=True)
    nw = min([os.cpu_count(), cfg.mini_batch_size if cfg.mini_batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                            #  batch_sampler=val_sampler,
                                             pin_memory=True,
                                             shuffle=False,
                                             num_workers=nw)
    for eid in range(cfg.max_epochs):

        pprint(f"> Epoch {eid+1}/{cfg.max_epochs}:")
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        epoch_start_time = time.time()
        if args.gpu == 0:
            pb = tqdm.tqdm(train_loader)
        else:
            pb = train_loader
        for i, (yt, _) in enumerate(pb):
            # x = add_salt_pepper(yt,-1,1,0.05,0.5)
            loss, r_loss, l_loss, _ = trainer.train(yt,yt)
            epoch_loss += loss
            epoch_r_loss += r_loss
            epoch_l_loss += l_loss
            if args.gpu == 0:
                pb.set_description(f"training_loss: {epoch_loss / (i+1):.5f} [r_loss: {epoch_r_loss/ (i+1):.5f}, l_loss: {epoch_l_loss / (i+1):.5f}]")
        pprint(f"> Training loss: {epoch_loss / len(train_loader):.5f} [r_loss: {epoch_r_loss / len(train_loader):.5f}, l_loss: {epoch_l_loss / len(train_loader):.5f}]")
        
        if eid % 1000 == 999:
            epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
            if args.gpu == 0:
                pb = tqdm.tqdm(val_loader)
            else:
                pb = val_loader
            for i, (x, _) in enumerate(pb):
                x = x.to(trainer.device)
                loss, r_loss, l_loss, y = trainer.eval(x)
                epoch_loss += loss
                epoch_r_loss += r_loss
                epoch_l_loss += l_loss
                if args.gpu == 0:
                    pb.set_description(f"evaluation: {epoch_loss / (i+1):.5f} [r_loss: {epoch_r_loss/ (i+1):.5f}, l_loss: {epoch_l_loss / (i+1):.5f}]")
                if i == 0 and not args.no_save and eid % cfg.image_frequency == 0:
                    save_image(x, img_dir / f"ori-{str(eid).zfill(4)}.{'jpg' if args.save_jpg else 'png'}", nrow=int(sqrt(cfg.mini_batch_size)), normalize=True, value_range=(-1,1))
                    save_image(y, img_dir / f"recon-{str(eid).zfill(4)}.{'jpg' if args.save_jpg else 'png'}", nrow=int(sqrt(cfg.mini_batch_size)), normalize=True, value_range=(-1,1))
                    save_image(torch.abs(y-x), img_dir / f"res-{str(eid).zfill(4)}.{'jpg' if args.save_jpg else 'png'}", nrow=int(sqrt(cfg.mini_batch_size)), normalize=True, value_range=(0,1))
            pprint(f"> Evaluation loss: {epoch_loss / len(val_loader):.5f} [r_loss: {epoch_r_loss / len(val_loader):.5f}, l_loss: {epoch_l_loss / len(val_loader):.5f}]")

        if eid % cfg.checkpoint_frequency == 0 and not args.no_save:
            if args.rank == 0:
                trainer.save_checkpoint(chk_dir / f"{args.task}-state-dict-{str(eid).zfill(4)}.pt")

        pprint(f"> Epoch time taken: {time.time() - epoch_start_time:.2f} seconds.")
        pprint()
