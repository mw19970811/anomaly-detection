
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from dataset import AnomalyDataset

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tqdm

import argparse
import datetime
import time
from pathlib import Path
from math import sqrt
import os, tqdm

from util import init_distributed_mode, pprint
from dataset import get_dataloader
from trainer import AnomalyTrainer
from ssim import SSIM
from gms import GMS
from hps import HPS_VQVAE as HPS
from helper import get_device, get_parameter_count
from functools import reduce
from itertools import product

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--root_path', type=str, default='D:/dataset/FMD/full')
    parser.add_argument('--task', type=str, default='anomaly')
    parser.add_argument('--load-path', type=str, default='runs/anomaly-2022-07-20_14-22-16/checkpoints/anomaly-state-dict-0020.pt' )
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--syncBN', type=bool, default=False)
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    args = parser.parse_args()
    init_distributed_mode(args=args)
    cfg = HPS[args.task]

    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    pprint(f"> Initialising VQ-VAE-2 model")
    trainer = AnomalyTrainer(cfg, args)
    pprint(f"> Number of parameters: {get_parameter_count(trainer.net)}")
    ssim = SSIM(size_average=False)
    gms = GMS(size_average=False)
    if args.load_path:
        pprint(f"> Loading model parameters from checkpoint")
        trainer.load_checkpoint(args.load_path,args,args.device)
    if args.batch_size:
        cfg.batch_size = args.batch_size

    out_path = 'D:/dataset/FMD/full/fine_output'
    os.makedirs(out_path,exist_ok=True)

    nw = min([os.cpu_count(), cfg.mini_batch_size if cfg.mini_batch_size > 1 else 0, 8])  # number of workers
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_dataset = AnomalyDataset(args.root_path, cfg=cfg, mode='val',transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,
                                               num_workers=nw)
    for i,(img,name) in tqdm.tqdm(enumerate(test_loader)):
        _, _, h, w = img.shape
        img = img.cuda()
        out = torch.zeros_like(img)
        patch_width = 1024
        patch_h = int(np.ceil(h/patch_width))
        patch_w = int(np.ceil(w/patch_width))
        # pprint('img shape: ',img.shape)
        trainer.net.eval()
        with torch.no_grad():
            for i,j in product(range(patch_h), range(patch_w)):
                tmp = trainer.net(img[...,i*patch_width:(i+1)*patch_width,j*patch_width:(j+1)*patch_width])[0]
                out[...,i*patch_width:(i+1)*patch_width,j*patch_width:(j+1)*patch_width] = tmp
        save_image(out,Path(out_path)/f'{name[0][:-4]}_out.jpg', normalize=True, value_range=(-1,1))
        print(Path(out_path)/f'{name[0][:-4]}_out.jpg')
        res_score = torch.abs(img-out)
        save_image(res_score,Path(out_path)/f'{name[0][:-4]}_res.jpg', normalize=True, value_range=(0,1))
        ssim_score = 1 - ssim(img,out)
        save_image(ssim_score,Path(out_path)/f'{name[0][:-4]}_ssim.jpg', normalize=True, value_range=(0,1))
        gms_score = 1 - gms(img,out)
        save_image(gms_score,Path(out_path)/f'{name[0][:-4]}_gms.jpg', normalize=True, value_range=(0,1))
