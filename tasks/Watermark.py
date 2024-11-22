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

class L22Loss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss()
    def forward(self, x, y):
        return (torch.abs(x-y)**2).sum(dim=-1).mean()


@mlconfig.register
class Watermark(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.fool_loss = self.cfg.fool_loss
        self.xyz_loss = self.cfg.xyz_loss
        self.normal_loss = self.cfg.normal_loss
        self.decoder_loss = self.cfg.decoder_loss
        '''dataset'''
        self.valid_loader = self.cfg.valid_set(parallel=False)
        self.train_loader = self.cfg.train_set(parallel=False)
        '''msg'''
        self.ydim = self.cfg.ydim
        self.msg_length = self.cfg.msg_length
        '''model'''
        self.encoder_decoder = self.cfg.encoder_decoder().to(self.device)
        self.discriminator = self.cfg.discriminator().to(self.device)
        '''opt'''
        self.enc_dec_opt = torch.optim.Adam(self.encoder_decoder.parameters(), lr=self.cfg.lr)
        self.dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr)
        '''scheduler'''
        self.enc_dec_scheduler = self.cfg.scheduler(self.enc_dec_opt)
        self.dis_scheduler = self.cfg.scheduler(self.dis_opt)

        '''loss'''
        self.l22_loss = L22Loss()
        self.p = len(self.encoder_decoder.noise_layer)
        self.grouping_func = V().cfg.grouping_strategy()
        
        '''channelae'''
        self.epoch = 0
        self.channelae_ED = self.cfg.channelae_ED().to(self.device)
        self.channelae_E = self.channelae_ED.encoder
        self.channelae_D = self.channelae_ED.decoder
        self.load_channelae_checkpoint(self.cfg.channelae_checkpoint)
        self.load_checkpoint(self.cfg.checkpoint)

    def load_channelae_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=f"cuda:{self.device}")
        self.channelae_ED.load_state_dict(checkpoint["encoder_decoder"])
        
    def load_checkpoint(self, path):
        if path:
            checkpoint = torch.load(path, map_location=f"cuda:{self.device}")
            self.encoder_decoder.load_state_dict(checkpoint["encoder_decoder"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.enc_dec_opt.load_state_dict(checkpoint["enc_dec_opt"])
            self.dis_opt.load_state_dict(checkpoint["dis_opt"])
            self.epoch = checkpoint["epoch"]

    def train_epoch(self, epoch):
        V().info(f"Epoch {epoch} Start: lr={self.enc_dec_opt.param_groups[0]['lr']}")
        self.enc_dec_scheduler.step(epoch)
        self.dis_scheduler.step(epoch)
        metrics = defaultdict(Avg)
        self.discriminator.train()
        self.encoder_decoder.train()
        with torch.enable_grad():
            # progress = self.train_loader
            progress = tqdm(self.train_loader, desc=f"Train")
            for xyz, faces, _, _ in progress:
                self.dis_opt.zero_grad()
                self.enc_dec_opt.zero_grad()

                xyz = xyz.to(self.device)
                faces = faces.to(self.device)
                idx = self.grouping_func(xyz, xyz, faces)
                normals = compute_vertex_normals(xyz, faces)
                
                B = xyz.shape[0]
                batched_msg = torch.Tensor(np.random.choice([0, 1], (B, self.msg_length))).to(self.device)
                _, h_msg, _, _ = self.channelae_E(batched_msg)

                cover_decision_truth = torch.full((B, 1), COVER_LABEL, dtype=torch.float).to(self.device)
                encoded_decision_truth = torch.full((B, 1), ENCODE_LABEL, dtype=torch.float).to(self.device)
                encoded_decision_false = torch.full((B, 1), COVER_LABEL, dtype=torch.float).to(self.device)

                '''discriminator cover loss'''
                cover_decision = self.discriminator(xyz, faces, idx)
                loss_bce_cover = self.bce_loss(cover_decision, cover_decision_truth)
                loss_bce_cover.backward()
                # metrics["discriminator_on_cover"].update(loss_bce_cover.item() * B, B)

                '''discriminator encode loss'''
                encoded_xyz, decoded_h_msg = self.encoder_decoder(xyz, faces, h_msg, idx)
                decoded_h_msg = torch.sigmoid(decoded_h_msg) # scale to (0,1)
                decoded_h_msg = decoded_h_msg - (decoded_h_msg - decoded_h_msg.round()).detach() # (round to 0/1)
                decoded_msg = self.channelae_D(decoded_h_msg)
                encoded_normals = compute_vertex_normals(encoded_xyz, faces)

                encoded_decision = self.discriminator(encoded_xyz.detach(), faces, idx)
                loss_bce_encoded = self.bce_loss(encoded_decision, encoded_decision_truth)
                loss_bce_encoded.backward()
                # metrics["discriminator_on_encode"].update(loss_bce_encoded.item() * B, B)

                self.dis_opt.step()

                '''train encoder'''
                g_encoded_decision = self.discriminator(encoded_xyz, faces, idx)
                g_loss_bce_encoded = self.bce_loss(g_encoded_decision, encoded_decision_false)

                '''backward'''
                batched_msg = torch.cat([batched_msg] * self.p)
                loss_mse_xyz = self.l22_loss(encoded_xyz, xyz)
                loss_mse_norm = self.l22_loss(encoded_normals, normals)
                loss_bce_msg = self.bce_loss(decoded_msg, batched_msg)

                total_loss = self.fool_loss * g_loss_bce_encoded + \
                        self.xyz_loss * loss_mse_xyz + \
                        self.normal_loss * loss_mse_norm + \
                        self.decoder_loss * loss_bce_msg
                
                assert not torch.isnan(total_loss)

                total_loss.backward()
                acc = (decoded_msg.round().clip(0,1) == batched_msg).sum()/(B*self.msg_length)/self.p
                metrics["loss"].update(total_loss.item() * B, B)
                metrics["xyz_loss"].update(loss_mse_xyz.item() * B, B)
                metrics["normal_loss"].update(loss_mse_norm.item() * B, B)
                metrics["msg_loss"].update(loss_bce_msg.item() * B, B)
                metrics['msg_acc'].update(acc.item() * B, B)
                metrics['l1_normal'].update(self.l1_loss(encoded_normals, normals).item() * B, B)
                
                self.enc_dec_opt.step()

                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )
        return metrics

    def evaluate(self):
        self.encoder_decoder.eval()
        self.discriminator.eval()
        metrics = defaultdict(Avg)
        with torch.no_grad():
            # progress = self.valid_loader
            progress = tqdm(self.valid_loader, desc="Evaluate")
            for xyz, faces, _, _ in progress:
                B = xyz.shape[0]
                batched_msg = torch.Tensor(np.random.choice([0, 1], (B, self.msg_length))).to(self.device)

                xyz = xyz.to(self.device)
                faces = faces.to(self.device)
                idx = self.grouping_func(xyz, xyz, faces)
                normals = compute_vertex_normals(xyz, faces)
                
                B = xyz.shape[0]
                batched_msg = torch.Tensor(np.random.choice([0, 1], (B, self.msg_length))).to(self.device)
                _, h_msg, _, _ = self.channelae_E(batched_msg)

                cover_decision_truth = torch.full((B, 1), COVER_LABEL, dtype=torch.float).to(self.device)
                encoded_decision_truth = torch.full((B, 1), ENCODE_LABEL, dtype=torch.float).to(self.device)
                encoded_decision_false = torch.full((B, 1), COVER_LABEL, dtype=torch.float).to(self.device)

                '''discriminator cover loss'''
                cover_decision = self.discriminator(xyz, faces, idx)
                loss_bce_cover = self.bce_loss(cover_decision, cover_decision_truth)
                # metrics["discriminator_on_cover"].update(loss_bce_cover.item() * B, B)

                '''discriminator encode loss'''
                encoded_xyz, decoded_h_msg = self.encoder_decoder(xyz, faces, h_msg, idx)
                decoded_h_msg = torch.sigmoid(decoded_h_msg) # scale to (0,1)
                decoded_h_msg = decoded_h_msg - (decoded_h_msg - decoded_h_msg.round()).detach() # (round to 0/1)
                decoded_msg = self.channelae_D(decoded_h_msg)
                encoded_normals = compute_vertex_normals(encoded_xyz, faces)

                encoded_decision = self.discriminator(encoded_xyz.detach(), faces, idx)
                loss_bce_encoded = self.bce_loss(encoded_decision, encoded_decision_truth)
                # metrics["discriminator_on_encode"].update(loss_bce_encoded.item() * B, B)

                '''encoder_decoder loss'''
                g_encoded_decision = self.discriminator(encoded_xyz, faces, idx)
                g_loss_bce_encoded = self.bce_loss(g_encoded_decision, encoded_decision_false)

                batched_msg = torch.cat([batched_msg] * self.p)
                loss_mse_xyz = self.l2_loss(encoded_xyz, xyz)
                loss_mse_norm = self.l22_loss(encoded_normals, normals)
                loss_bce_msg = self.bce_loss(decoded_msg, batched_msg)

                total_loss = self.fool_loss * g_loss_bce_encoded + \
                        self.xyz_loss * loss_mse_xyz + \
                        self.normal_loss * loss_mse_norm + \
                        self.decoder_loss * loss_bce_msg
                
                acc = (decoded_msg.round().clip(0,1) == batched_msg).sum()/(B*self.msg_length)/self.p
                metrics["loss"].update(total_loss.item() * B, B)
                metrics["xyz_loss"].update(loss_mse_xyz.item() * B, B)
                metrics["normal_loss"].update(loss_mse_norm.item() * B, B)
                metrics["msg_loss"].update(loss_bce_msg.item() * B, B)
                metrics['msg_acc'].update(acc.item() * B, B)
                metrics['l1_normal'].update(self.l1_loss(encoded_normals, normals).item() * B, B)

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

            self.save_checkpoint(ep, eval_metrics["loss"].value())

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
        checkpoint = {
            "encoder_decoder": self.encoder_decoder.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "enc_dec_opt": self.enc_dec_opt.state_dict(),
            "dis_opt": self.dis_opt.state_dict(),
            "epoch": epoch,
            "channelae_ED": self.channelae_ED.state_dict(),
        }
        # os.makedirs(os.path.join(output_dir, str(epoch)), exist_ok=True)
        # torch.save(checkpoint, os.path.join(output_dir, str(epoch), "checkpoint.pyt"))
        torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pyt"))
        if self.best > metrics:
            torch.save(checkpoint, os.path.join(self.cfg.output_dir, V().name, "best.pyt"))
            self.best = metrics
    
    def save_history(self, mode, **kwargs):
        filename = os.path.join(self.cfg.output_dir, V().name, f'{mode}_history.pyt')
        for k, v in kwargs.items():
            v = list(map(lambda x: x.value(), v))
            # plot(v, os.path.join(self.cfg.output_dir, V().name, f'{mode}_{k}_plot.png'))
        
        torch.save(kwargs, filename)