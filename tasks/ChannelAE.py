import os
import mlconfig
import numpy as np
import torch
import torch.nn as nn
import shutil
from tqdm import tqdm
from collections import defaultdict

from util import *
from dataset import *
from model import *
from metrics import *
from visual import *

from tasks.base import BaseTrainer

COVER_LABEL=0
ENCODE_LABEL=1

def grad_debug(name, xx, **kwargs):
    def hook_func(grad):
        if torch.isnan(grad).sum():
            print("---------------")
            print(f'{name}: {grad}')
            if 'reconstr_loss' in kwargs.keys():
                idxs = np.where(torch.isnan(kwargs['reconstr_loss']).detach().cpu())
                ind0 = idxs[0][0]
                ind1 = idxs[1][0]
                print(ind0, ind1)
                print(kwargs['reconstr_loss'][ind0][ind1])
                print(kwargs['x_hat'][ind0][ind1])
                print(kwargs['x'][ind0][ind1])
            print("---------------")
    xx.register_hook(hook_func)

class Vimco_loss(nn.Module):
    def __init__(self, vimco_samples):
        super().__init__()
        self.vimco_samples = vimco_samples

    def build_vimco_loss(self, l):
        K, B = l.shape # l={l1,l2,...,l_k}
        l_logsumexp = torch.logsumexp(l, dim=0, keepdim=True) # 操作A: (1, B)
        L_hat = l_logsumexp - torch.log(torch.tensor(K))

        s = l.sum(dim=0, keepdim=True)

        diag_mask = torch.diag(torch.ones((K)).float()).unsqueeze(-1).cuda()
        off_diag_mask = 1 - diag_mask

        diff = (s - l).unsqueeze(0)                 # diff[i]: 除去第i个vsample以外所有vsample的sum=\sum_{j!=i} l_j
        l_i_diag = 1 / (K - 1) * diff * diag_mask   # l_i_diag[i][i]: 除去第i个vsample的mean 非对角线上全0
        l_i_off_diag = off_diag_mask * torch.stack([l] * K) # 对于任意j!=i, l_i_off_diag[i][j]为第i个vsample的loss=l_i
        l_i = l_i_diag + l_i_off_diag               # (K, K, B) 相对于stack起来的l，对角线上的东西是k-1 mean而不是k mean
        L_hat_minus_i = torch.logsumexp(l_i, dim=1) - torch.log(torch.tensor(K)) # 和A完全一样的操作，只不过第i个元素被替换成了其他vsample的mean (K, B)

        w = torch.exp(l - l_logsumexp).detach()

        local_l = (L_hat - L_hat_minus_i).detach()  # (K, B)

        return local_l,  w, L_hat[0, :]

    def forward(self, x, x_hat, q, y):
        eps = 1e-12
        x = x.unsqueeze(0).repeat(self.vimco_samples, 1, 1)
        reconstr_loss =  (- x * torch.log((torch.sigmoid(x_hat)).clamp_min(eps)) - (1 - x) * torch.log((1 - torch.sigmoid(x_hat)).clamp_min(eps))).sum(-1) # (K, B)

        log_q_h_list = q.log_prob(y)
        log_q_h = log_q_h_list.sum(-1)

        local_l, w, full_loss = self.build_vimco_loss(reconstr_loss)

        theta_loss = w * reconstr_loss
        phi_loss = local_l * log_q_h + theta_loss.detach()
        
        return theta_loss.sum(0).mean(), phi_loss.sum(0).mean(), full_loss.mean()

@mlconfig.register
class ChannelAE(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.vimco_samples = self.cfg.vimco_samples

        self.train_loader = self.cfg.train_set()
        self.valid_loader = self.cfg.valid_set()

        self.encoder_decoder = self.cfg.encoder_decoder().to(self.device)
        self.encoder = self.encoder_decoder.encoder
        self.decoder = self.encoder_decoder.decoder

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.lr, weight_decay=5e-4)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=self.cfg.lr)
        
        self.encoder_scheduler = self.cfg.scheduler(self.encoder_opt)
        self.decoder_scheduler = self.cfg.scheduler(self.decoder_opt)

        self.L = Vimco_loss(self.vimco_samples)

        self.epoch = 0
        self.load_checkpoint(self.cfg.checkpoint)
    
    def get_checkpoint(self, epoch):
        return {
            "encoder_decoder": self.encoder_decoder.state_dict(),
            "encoder_opt": self.encoder_opt.state_dict(),
            "decoder_opt": self.decoder_opt.state_dict(),
            "epoch": epoch
        }
    
    def load_checkpoint(self, path):
        if path:
            checkpoint = torch.load(path)
            self.encoder_decoder.load_state_dict(checkpoint["encoder_decoder"])
            self.encoder_opt.load_state_dict(checkpoint["encoder_opt"])
            self.decoder_opt.load_state_dict(checkpoint["decoder_opt"])
            self.epoch = checkpoint["epoch"]

    def train_epoch(self, epoch):
        V().info(f"Epoch {epoch} Start: lr={self.encoder_opt.param_groups[0]['lr']}")
        self.encoder_scheduler.step(epoch)
        self.decoder_scheduler.step(epoch)
        metrics = defaultdict(Avg)
        # for n, params in self.encoder_decoder.named_parameters():
        #     def zxy(name):
        #         def fn(x):
        #             print(f'{name}: {x}')
        #         return fn
        #     params.register_hook(zxy(n))
        self.encoder_decoder.train()
        with torch.enable_grad():
            # progress = self.train_loader
            progress = tqdm(self.train_loader, desc=f"Train")
            for x in progress:
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()

                x = x.float().to(self.device)
                stacked_x = x.unsqueeze(0).repeat(self.vimco_samples, 1, 1)
                N, B, L = stacked_x.shape

                '''encoder: phi, decoder: theta'''
                combined_prob, y_hat, y, q = self.encoder(x)
                xx = self.decoder(y_hat)
                x_hat = self.decoder(y)
                theta_loss, phi_loss, reconstr_loss = self.L(x, x_hat, q, y)
                phi_loss.backward()
                theta_loss.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()
                
                acc = (x_hat.round().clip(0,1) == stacked_x).sum() / (N * B * L)
                acc_origin = (xx.round().clip(0,1) == stacked_x).sum() / (N * B * L)
                metrics['reconstr_loss'].update(reconstr_loss.item() * B, B)
                metrics['phi_loss'].update(phi_loss.item() * B, B)
                metrics['theta_loss'].update(theta_loss.item() * B, B)
                metrics['acc'].update(acc.item() * B, B)
                metrics['acc_origin'].update(acc_origin.item() * B, B)
                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )
        return metrics

    def evaluate(self):
        metrics = defaultdict(Avg)
        self.encoder_decoder.eval()
        with torch.no_grad():
            # progress = self.valid_loader
            progress = tqdm(self.valid_loader, desc=f"Evaluate")
            for x in progress:
                x = x.float().to(self.device)
                stacked_x = x.unsqueeze(0).repeat(self.vimco_samples, 1, 1)
                N, B, L = stacked_x.shape

                combined_prob, y_hat, y, q = self.encoder(x)
                xx = self.decoder(y_hat)
                x_hat = self.decoder(y)
                theta_loss, phi_loss, reconstr_loss = self.L(x, x_hat, q, y)
                
                acc = (x_hat.round().clip(0,1) == stacked_x).sum() / (N * B * L)
                acc_origin = (xx.round().clip(0,1) == stacked_x).sum() / (N * B * L)
                metrics['reconstr_loss'].update(reconstr_loss.item() * B, B)
                metrics['phi_loss'].update(phi_loss.item() * B, B)
                metrics['theta_loss'].update(theta_loss.item() * B, B)
                metrics['acc'].update(acc.item() * B, B)
                metrics['acc_origin'].update(acc_origin.item() * B, B)
                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )
    
        return metrics
    
    def train(self):
        eval_history = defaultdict(list)
        train_history = defaultdict(list)
        for ep in range(self.epoch, self.cfg.epoch):
            '''epoch training'''
            train_metrics = self.train_epoch(ep)
            eval_metrics = self.evaluate()

            V().info("train_metrics", **train_metrics)
            V().info("eval_metrics", **eval_metrics)

            self.save_checkpoint(ep, eval_metrics["reconstr_loss"].value())

            for k, v in train_metrics.items():
                train_history[k].append(v)
                
            for k, v in eval_metrics.items():
                eval_history[k].append(v)
            
        '''plot history'''
        self.save_history("train", **train_history)
        self.save_history("eval", **eval_history)
        

    def save_checkpoint(self, epoch, metrics):
        output_dir = os.path.join(self.cfg.output_dir, V().name)
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = self.get_checkpoint(epoch)
        # os.makedirs(os.path.join(output_dir, str(epoch)), exist_ok=True)
        # torch.save(checkpoint, os.path.join(output_dir, str(epoch), "checkpoint.pyt"))
        torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pyt"))
        if self.best > metrics:
            torch.save(checkpoint, os.path.join(self.cfg.output_dir, V().name, "best.pyt"))
            self.best = metrics
    
    def save_history(self, mode, **kwargs):
        filename = os.path.join(self.cfg.output_dir, V().name, f'{mode}_history.pyt')
        # for k, v in kwargs.items():
            # v = list(map(lambda x: x.value(), v))
            # plot(v, os.path.join(self.cfg.output_dir, V().name, f'{mode}_{k}_plot.png'))
        
        torch.save(kwargs, filename)
    
    def acc(self):
        metrics = defaultdict(Avg)
        self.encoder_decoder.eval()
        with torch.no_grad():
            # progress = self.valid_loader
            progress = tqdm(self.train_loader, desc=f"Evaluate")
            for x in progress:
                x = x.float().to(self.device)
                stacked_x = x.unsqueeze(0).repeat(self.vimco_samples, 1, 1)
                N, B, L = stacked_x.shape

                combined_prob, y_hat, y, q = self.encoder(x)
                x_hat = self.decoder(y_hat) # without noise
                x_hat = self.decoder(y)     # with noise
                theta_loss, phi_loss, reconstr_loss = self.L(x, x_hat, q, y)
                
                acc = (x_hat.round().clip(0,1) == stacked_x).sum() / (N * B * L)
                flip_ratio = (y_hat==y).sum()/(y==y).sum()
                metrics['reconstr_loss'].update(reconstr_loss.item() * B, B)
                metrics['acc'].update(acc.item() * B, B)
                metrics['flip_ratio'].update(flip_ratio.item() * B, B)
                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )
    
        return metrics
    
    def debug(self):
        print(self.epoch)
        self.acc()
    def check_dup(self):
        train = defaultdict(lambda: 0)
        eval = defaultdict(lambda: 0)
        for x in self.valid_loader:
            nums = list(map(lambda xx: int("".join(str(xxx) for xxx in xx), 2), x.int().tolist()))
            for n in nums:
                eval[n]+=1
        for x in self.train_loader:
            nums = list(map(lambda xx: int("".join(str(xxx) for xxx in xx), 2), x.int().tolist()))
            for n in nums:
                train[n]+=1
        tot = 0
        for tk, tv in eval.items():
            if tk in train.keys():
                tot+=1
                print(tk, tv)
        print(tot)