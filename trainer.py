from msilib.schema import Error
import numpy as np
import torch
import torch.nn.functional as F
import math
from itertools import product

from vqvae import VQVAE
from util import pprint
from helper import get_device

class AnomalyTrainer:
    def __init__(self, cfg, args):
        # self.device = get_device(args.cpu)
        self.device = torch.device(args.device)
        self.net = VQVAE(in_channels=cfg.in_channels, 
                    hidden_channels=cfg.hidden_channels, 
                    embed_dim=cfg.embed_dim, 
                    nb_entries=cfg.nb_entries, 
                    nb_levels=cfg.nb_levels, 
                    scaling_rates=cfg.scaling_rates).to(self.device)
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[args.gpu], output_device=args.gpu)
        self.net = self.net.to(self.device)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.opt.zero_grad()

        self.beta = cfg.beta
        self.scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

        self.update_frequency = math.ceil(cfg.batch_size / cfg.mini_batch_size)
        self.train_steps = 0

    def _calculate_loss(self, x: torch.FloatTensor, yt: torch.FloatTensor):
        x = x.to(self.device)
        yt = yt.to(self.device)
        yi, d, _, _, _ = self.net(x)
        # 计算重建损失r_loss和聚集损失d_loss
        r_loss, l_loss = yt.sub(yi).pow(2).mean(), sum(d)
        loss = r_loss + self.beta*l_loss
        return loss, r_loss, l_loss, yi

    # another function can then call step
    def train(self, x: torch.FloatTensor, yt: torch.FloatTensor):
        self.net.train()
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            loss, r_loss, l_loss, y = self._calculate_loss(x,yt)
        self.scaler.scale(loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()

        return loss.item(), r_loss.item(), l_loss.item(), y

    """
        Use accumulated gradients to step `self.opt`, updating parameters.
    """
    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.opt.zero_grad()
        self.scaler.update()

    @torch.no_grad()
    def eval(self, x: torch.FloatTensor):
        self.net.eval()
        # self.opt.zero_grad()
        _, _, h, w = x.shape
        x = x.to(self.device)
        patch_width = 1024
        patch_h = int(np.ceil(h/patch_width))
        patch_w = int(np.ceil(w/patch_width))
        losses = []
        r_losses = []
        l_losses = []
        y = torch.zeros_like(x)
        for i,j in product(range(patch_h), range(patch_w)):
            patch_in = x[:,:,i*patch_width:(i+1)*patch_width,j*patch_width:(j+1)*patch_width]
            loss, r_loss, l_loss, yout = self._calculate_loss(patch_in,patch_in)
            losses.append(loss.item())
            r_losses.append(r_loss.item())
            l_losses.append(l_loss.item())
            y[:,:,i*patch_width:(i+1)*patch_width,j*patch_width:(j+1)*patch_width] = yout
        loss = np.array(losses).mean()
        r_loss = np.array(r_losses).mean()
        l_loss = np.array(l_losses).mean()
        return loss, r_loss, l_loss, y

    def save_checkpoint(self, path):
        torch.save(self.net.state_dict(), path)

    def load_checkpoint(self, path, args, device='cuda'):
        weights_dict = torch.load(path,map_location=device)
        load_weights_dict = {}

        # if args.gpu == 0:
        #     pprint('checkpoint dict keys :',len(list(weights_dict.keys())))
        #     pprint('net dict keys: ',len(list(self.net.state_dict().keys())))
        
        # for j in self.net.state_dict():
        #     try:
        #         print(j)
        #     except Error as e:
        #         print(e)
        for i in list(weights_dict.keys()):
            for j in list(self.net.state_dict().keys()):
                if i == j or i == j[7:] or i[7:] == j:
                    load_weights_dict[j] = weights_dict[i]
        print('>> weights length: ',len(load_weights_dict))
        print('>> net weights length: ',len(self.net.state_dict()))
        print('>> checkpoint weights length: ',len(weights_dict))
        self.net.load_state_dict(load_weights_dict)
        pprint('Loading weights dict success!')
        pprint(f'Loading state dict length is {len(load_weights_dict)}')
    