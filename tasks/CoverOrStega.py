import torch
import mlconfig
from collections import defaultdict
from tqdm import tqdm

from util import *
from metrics import *
from tasks.base import BaseTrainer

COVER_LABEL=0
ENCODE_LABEL=1

@mlconfig.register
class CoverOrStega(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.cos = self.cfg.cos().to(self.device)
        
        self.opt = torch.optim.Adam(self.cos.parameters(), lr=self.cfg.lr)
        
        self.scheduler = self.cfg.scheduler(self.opt)

        self.grouping_strategy = V().cfg.grouping_strategy()

        self.epoch = 0
        '''dataset'''
        self.valid_loader = self.cfg.valid_set(parallel=False)
        self.train_loader = self.cfg.train_set(parallel=False)
        '''msg'''
        self.ydim = self.cfg.ydim
        self.msg_length = self.cfg.msg_length

        '''wm'''
        self.encoder_decoder = self.cfg.encoder_decoder().to(self.device)
        self.discriminator = self.cfg.discriminator().to(self.device)

        '''channelae'''
        self.channelae_ED = self.cfg.channelae_ED().to(self.device)
        self.channelae_E = self.channelae_ED.encoder
        self.channelae_D = self.channelae_ED.decoder
        self.load_channelae_checkpoint(self.cfg.channelae_checkpoint)
        self.load_checkpoint(self.cfg.checkpoint)

        self.discriminator.eval()
        self.encoder_decoder.eval()
        self.channelae_ED.eval()
    
    def load_channelae_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        self.channelae_ED.load_state_dict(checkpoint["encoder_decoder"])
        
    def load_checkpoint(self, path):
        if path:
            checkpoint = torch.load(path, map_location="cuda:0")
            self.encoder_decoder.load_state_dict(checkpoint["encoder_decoder"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])

    def train_epoch(self, epoch):
        V().info(f"Epoch {epoch} Start: lr={self.opt.param_groups[0]['lr']}")
        self.cos.train()
        self.scheduler.step(epoch)
        metrics = defaultdict(Avg)
        with torch.enable_grad():
            progress = tqdm(self.train_loader, desc="Train")
            for xyz, faces, _, _ in progress:
                self.opt.zero_grad()
                
                xyz = xyz.to(self.device)
                faces = faces.to(self.device)
                idx = self.grouping_strategy(xyz, xyz, faces)
                
                B = xyz.shape[0]
                batched_msg = torch.Tensor(np.random.choice([0, 1], (B, self.msg_length))).to(self.device)
                _, h_msg, _, _ = self.channelae_E(batched_msg)

                cover_decision_truth = torch.full((B, 1), COVER_LABEL, dtype=torch.float).to(self.device)
                encoded_decision_truth = torch.full((B, 1), ENCODE_LABEL, dtype=torch.float).to(self.device)

                encoded_xyz, decoded_h_msg = self.encoder_decoder(xyz, faces, h_msg, idx)
                '''cos'''
                cover_decision = self.cos(xyz, faces, idx)
                encoded_decision = self.cos(encoded_xyz.detach(), faces, idx)
                
                loss_bce_cover = self.bce_loss(cover_decision, cover_decision_truth)
                loss_bce_encoded = self.bce_loss(encoded_decision, encoded_decision_truth)
                loss = (loss_bce_cover + loss_bce_encoded)
                loss.backward()
                self.opt.step()
                metrics['cos'].update(loss.item() * B, B)

                acc = (cover_decision.round().clip(0,1)==cover_decision_truth).sum() + (encoded_decision.round().clip(0,1)==encoded_decision_truth).sum()
                metrics['cos_acc'].update(acc.item(), 2*B)

                '''discriminator'''
                cover_decision = self.discriminator(xyz, faces, idx)
                encoded_decision = self.discriminator(encoded_xyz.detach(), faces, idx)
                loss_bce_cover = self.bce_loss(cover_decision, cover_decision_truth)
                loss_bce_encoded = self.bce_loss(encoded_decision, encoded_decision_truth)
                acc = (cover_decision.round().clip(0,1)==cover_decision_truth).sum() + (encoded_decision.round().clip(0,1)==encoded_decision_truth).sum()
                metrics['dis_acc'].update(acc.item(), 2*B)

                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )
        return metrics
    def evaluate(self):
        self.cos.eval()
        metrics = defaultdict(Avg)
        with torch.enable_grad():
            progress = tqdm(self.valid_loader, desc="Eval")
            for xyz, faces, _, _ in progress:
                xyz = xyz.to(self.device)
                faces = faces.to(self.device)
                idx = self.grouping_strategy(xyz, xyz, faces)
                
                B = xyz.shape[0]
                batched_msg = torch.Tensor(np.random.choice([0, 1], (B, self.msg_length))).to(self.device)
                _, h_msg, _, _ = self.channelae_E(batched_msg)

                cover_decision_truth = torch.full((B, 1), COVER_LABEL, dtype=torch.float).to(self.device)
                encoded_decision_truth = torch.full((B, 1), ENCODE_LABEL, dtype=torch.float).to(self.device)

                encoded_xyz, decoded_h_msg = self.encoder_decoder(xyz, faces, h_msg, idx)
                '''cos'''
                cover_decision = self.cos(xyz, faces, idx)
                encoded_decision = self.cos(encoded_xyz.detach(), faces, idx)
                
                loss_bce_cover = self.bce_loss(cover_decision, cover_decision_truth)
                loss_bce_encoded = self.bce_loss(encoded_decision, encoded_decision_truth)
                loss = (loss_bce_cover + loss_bce_encoded)
                metrics['cos'].update(loss.item() * B, B)

                acc = (cover_decision.round().clip(0,1)==cover_decision_truth).sum() + (encoded_decision.round().clip(0,1)==encoded_decision_truth).sum()
                metrics['cos_acc'].update(acc.item(), 2*B)

                '''discriminator'''
                cover_decision = self.discriminator(xyz, faces, idx)
                encoded_decision = self.discriminator(encoded_xyz.detach(), faces, idx)
                loss_bce_cover = self.bce_loss(cover_decision, cover_decision_truth)
                loss_bce_encoded = self.bce_loss(encoded_decision, encoded_decision_truth)
                acc = (cover_decision.round().clip(0,1)==cover_decision_truth).sum() + (encoded_decision.round().clip(0,1)==encoded_decision_truth).sum()
                metrics['dis_acc'].update(acc.item(), 2*B)

                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )
        return metrics

    def train(self):
        for ep in range(self.epoch, self.cfg.epoch):
            '''epoch training'''
            train_metrics = self.train_epoch(ep)
            eval_metrics = self.evaluate()

            V().info("train_metrics", **train_metrics)
            V().info("eval_metrics", **eval_metrics)

            self.save_checkpoint(ep, eval_metrics["cos"].value())
        
    def save_checkpoint(self, epoch, metrics):
        output_dir = os.path.join(self.cfg.output_dir, V().name)
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = {
            "opt": self.opt.state_dict(),
            "cos": self.cos.state_dict(),
            "epoch": epoch
        }
        # os.makedirs(os.path.join(output_dir, str(epoch)), exist_ok=True)
        # torch.save(checkpoint, os.path.join(output_dir, str(epoch), "checkpoint.pyt"))
        torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pyt"))
        if self.best > metrics:
            torch.save(checkpoint, os.path.join(self.cfg.output_dir, V().name, "best.pyt"))
            self.best = metrics