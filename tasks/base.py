import os
import time
import torch
import numpy as np
import collections
import matplotlib.pyplot as plt
import json
import pathlib
import trimesh
from hotelling.stats import hotelling_t2
from chamferdist import ChamferDistance
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from metrics import *
from visual import *
from util import *
from model import *

COVER_LABEL=0
ENCODE_LABEL=1

def quantization(xyz, bit_len):
    xyz *= (1 << bit_len)
    xyz = torch.round(xyz)
    return xyz / (1<<bit_len)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_L2_regular_loss(model):
    pass

class BaseTrainer:
    def __init__(self):
        pass
        self.device = V().device
        self.cfg = V().cfg
        self.best = np.inf

        '''loss'''
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        
    ###############################################################
    #                  acc under all scenario                     #
    ###############################################################

    def acc(self):
        self.encoder_decoder.eval()
        encoder = self.encoder_decoder.encoder
        decoder = self.encoder_decoder.decoder
        group_func = self.cfg.grouping_strategy()
        
        metrics = collections.defaultdict(Avg)
        with torch.no_grad():
            progress = tqdm(self.valid_loader)
            for xyz, faces, _, _ in progress:
                xyz, centroid, m = pc_normalize(xyz)
                B, N, _ = xyz.shape
                xyz = xyz.to(self.device)
                faces = faces.cuda()
                idx = group_func(xyz, xyz, faces)
                batched_msg = torch.Tensor(np.random.choice([0, 1], (B, self.msg_length))).cuda()
                _, h_msg, _, _ = self.channelae_E(batched_msg)

                '''distortion'''
                t0 = time.time()
                enc_xyz = encoder(xyz, h_msg, idx)
                metrics['encode_time'].update(time.time() - t0, 1)

                l1d = torch.abs(enc_xyz - xyz).sum(dim=-1).mean()
                hd = max(hausdorff(enc_xyz[0], xyz[0]), hausdorff(xyz[0], enc_xyz[0]))
                cd = ChamferDistance()(enc_xyz, xyz, point_reduction="mean")
                snr = SNR(enc_xyz, xyz)

                '''msg acc'''
                enc_xyz, _, _ = pc_normalize(enc_xyz)
                idx = group_func(enc_xyz, enc_xyz, faces)
                t0 = time.time()
                dec_h_msg = decoder(enc_xyz, idx)
                metrics['decode_time'].update(time.time() - t0, B)
                dec_h_msg = torch.sigmoid(dec_h_msg) # scale to (0,1)
                dec_h_msg = dec_h_msg - (dec_h_msg - dec_h_msg.round()).detach() # (round to 0/1)
                decoded_msg = self.channelae_D(dec_h_msg)

                metrics["l1d"].update(l1d.item() * B, B)
                metrics["snr"].update(snr.item() * B, B)
                metrics["hd"].update(hd * B, B)
                metrics['cd'].update(cd * B, B)
                acc = (decoded_msg.round().clip(0,1)==batched_msg).sum()/self.msg_length/B
                metrics["acc"].update(acc.item() * B, B)
                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )
        print(', '.join([f'{k}: {v.value()}' for k, v in metrics.items()]))


    def acc_robustness(self):
        attacks = [
            Rotation(0.1),
            Gauss(var=0.005),
            Quantization(5),
            Scaling(0.1),
        ]
        ########################################
        # update your modelnet40 off list here
        ########################################
        file_list = glob("/data/xingyu/ModelNet40/airplane/*/*.off")
        self.encoder_decoder.eval()
        decoder = self.encoder_decoder.decoder
        encoder = self.encoder_decoder.encoder
        metrics = collections.defaultdict(Avg)
        group_func = self.cfg.grouping_strategy()
        with torch.no_grad():
            # progress = tqdm(self.valid_loader)
            # for xyz, faces, centroid, m in progress:
            #     B, N, _ = xyz.shape
            #     xyz = xyz.to(self.device)
            #     faces = faces.cuda()
            #     centroid = centroid.cuda()
            #     m = m.cuda()
            progress = tqdm(file_list)
            for file in progress:
                xyz, faces = read_mesh(file)
                try:
                    B, N, _ = xyz.shape
                    if N > 5e4: continue
                except:
                    continue
                xyz, centroid, m = pc_normalize(xyz)

                idx = group_func(xyz, xyz, faces)
                batched_msg = torch.Tensor(np.random.choice([0, 1], (B, self.msg_length))).cuda()
                _, h_msg, _, _ = self.channelae_E(batched_msg)

                '''embed msg'''
                enc_xyz = encoder(xyz, h_msg, idx)

                '''msg acc'''
                enc_xyz, _, _ = pc_normalize(enc_xyz)
                dec_h_msg = decoder(enc_xyz, idx)
                dec_h_msg = torch.sigmoid(dec_h_msg) # scale to (0,1)
                dec_h_msg = dec_h_msg - (dec_h_msg - dec_h_msg.round()).detach() # (round to 0/1)
                decoded_msg = self.channelae_D(dec_h_msg)
                acc = (decoded_msg.round().clip(0,1)==batched_msg).sum()/self.msg_length/B

                '''robustness'''
                for id, attack in enumerate(attacks):
                    tmp = enc_xyz #* m.view(B, 1, 1) + centroid.view(B, 1, 3)
                    att_xyz = attack(tmp).contiguous()
                    att_xyz, _, _ = pc_normalize(att_xyz)
                    dec_h_msg = decoder(att_xyz, idx)
                    dec_h_msg = torch.sigmoid(dec_h_msg) # scale to (0,1)
                    dec_h_msg = dec_h_msg - (dec_h_msg - dec_h_msg.round()).detach() # (round to 0/1)
                    decoded_msg = self.channelae_D(dec_h_msg)
                    acc = (decoded_msg.round().clip(0,1)==batched_msg).sum()/self.msg_length/B
                    metrics[f"attacked_acc_{id}"].update(acc.item() * B, B)
                
                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )
        print( (sum(list(map(lambda x:x.value(), metrics.values()))) - metrics["acc"].value()) / 9)
        print('\n'.join([f'{k}: {v.value()}' for k, v in metrics.items()]))

    def debug(self):
        self.acc()              # reproduce results: SNR & accuracy
        self.acc_robustness()   # reproduce results: robustness
        self.acc_size()         # reproduce results: size adaptation


    def acc_compress(self):
        import DracoPy
        self.encoder_decoder.eval()
        encoder = self.encoder_decoder.encoder
        decoder = self.encoder_decoder.decoder
        group_func = self.cfg.grouping_strategy()
        metrics = collections.defaultdict(Avg)
        Nq = 15
        with torch.no_grad():
            progress = tqdm(self.valid_loader)
            for xyz, face, _, _ in progress:
                B, N, _ = xyz.shape
                xyz = xyz.to(self.device)
                face = face.to(self.device)
                batched_msg = torch.Tensor(np.random.choice([0, 1], (1, self.msg_length))).cuda()
                _, h_msg, _, _ = self.channelae_E(batched_msg)
                
                idx = group_func(xyz, xyz, face)
                enc_xyz = encoder(xyz, h_msg, idx)
                enc_xyz, _, _ = pc_normalize(enc_xyz)

                binary = DracoPy.encode(
                    to_numpy(enc_xyz[0]), to_numpy(face[0]),
                    quantization_bits=Nq,
                )
                print(f'Original Bytes: {4 * (math.prod(enc_xyz.shape) + math.prod(face.shape))}, Compressed Bytes: {len(binary)}')
                mesh = DracoPy.decode(binary)
                comp_xyz, comp_face = torch.tensor(mesh.points).cuda().unsqueeze(dim=0).float(), torch.tensor(mesh.faces).cuda().unsqueeze(dim=0).int()
                idx = group_func(comp_xyz, comp_xyz, comp_face)


                decoded_h_msg = decoder(comp_xyz, idx)
                decoded_h_msg = torch.sigmoid(decoded_h_msg) # scale to (0,1)
                decoded_h_msg = decoded_h_msg - (decoded_h_msg - decoded_h_msg.round()).detach() # (round to 0/1)
                decoded_msg = self.channelae_D(decoded_h_msg)
                acc = (decoded_msg.round().clip(0,1)==batched_msg).sum()/self.msg_length
                
                metrics["acc"].update(acc.item() * B, B)
                progress.set_postfix_str(
                    ', '.join([f'{k}: {v.value()}' for k, v in metrics.items()])
                )

    def acc_size(self):
        self.encoder_decoder.eval()
        encoder = self.encoder_decoder.encoder
        decoder = self.encoder_decoder.decoder
        step_size = 20000
        tot_acc, tot_norm, tot_hd, tot_snr = \
            collections.defaultdict(lambda:Avg()), collections.defaultdict(lambda:Avg()), \
            collections.defaultdict(lambda:Avg()), collections.defaultdict(lambda:Avg())
        
        tot_time = collections.defaultdict(lambda:Avg())
        group_func = self.cfg.grouping_strategy()
        with torch.no_grad():
            off_file_paths = glob(os.path.join("/data/xingyu/ModelNet40", "*", "*", "*.off"))
            progress = tqdm(off_file_paths)
            for path in progress:
                xyz, faces = read_mesh(path)
                try:
                    B, N, _ = xyz.shape
                    if N > 1e5: continue
                except:
                    continue
                batched_msg = torch.Tensor(np.random.choice([0, 1], (B, self.msg_length))).cuda()
                _, h_msg, _, _ = self.channelae_E(batched_msg)
                idx = group_func(xyz, xyz, faces)

                '''distortion'''
                xyz, _, _ = pc_normalize(xyz)
                t1 = time.time()
                enc_xyz = encoder(xyz, h_msg, idx)
                duration1 = time.time() - t1

                l1d = torch.abs(enc_xyz - xyz).sum(dim=-1).mean()

                hd = max(hausdorff(enc_xyz[0], xyz[0]), hausdorff(xyz[0], enc_xyz[0]))
                snr = SNR(enc_xyz, xyz)

                '''msg acc'''
                enc_xyz, _, _ = pc_normalize(enc_xyz)
                t2 = time.time()
                dec_h_msg = decoder(enc_xyz, idx)
                duration2 = time.time() - t2
                dec_h_msg = torch.sigmoid(dec_h_msg) # scale to (0,1)
                dec_h_msg = dec_h_msg - (dec_h_msg - dec_h_msg.round()).detach() # (round to 0/1)
                decoded_msg = self.channelae_D(dec_h_msg)
                acc = (decoded_msg.round().clip(0,1)==batched_msg).sum()/self.msg_length/B

                tot_norm[N//step_size].update(l1d.item() * B, B)
                tot_hd[N//step_size].update(hd * B, B)
                tot_snr[N//step_size].update(snr.item() * B, B)
                tot_acc[N//step_size].update(acc.item() * B, B)
                tot_time[N].update((duration2 + duration1) * B, B)

                progress.set_postfix_str(
                    f'l1d: {l1d.item()}, \
                    hd: {hd}, \
                    snr: {snr.item()}, \
                    acc: {acc.item()} '
                )
        for k, v in tot_acc.items():
            tot_acc[k] = v.value()
        with open("tot_acc.json", "w") as f:
            json.dump(tot_acc, f)

        for k, v in tot_norm.items():
            tot_norm[k] = v.value()
        with open("tot_norm.json", "w") as f:
            json.dump(tot_norm, f)

        for k, v in tot_snr.items():
            tot_snr[k] = v.value()
        with open("tot_snr.json", "w") as f:
            json.dump(tot_snr, f)

        for k, v in tot_hd.items():
            tot_hd[k] = v.value()
        with open("tot_hd.json", "w") as f:
            json.dump(tot_hd, f)
        
        for k, v in tot_time.items():
            tot_time[k] = v.value()
        with open("tot_time.json", 'w') as f:
            json.dump(tot_time, f)
    

    def distribution_shape(self):
        def learn_from_relation(v0, v1, v2):
            return [v0 - v1, v1 - v2, v2 - v0, v1 - v0, v2 - v1, v0 - v2]
        X = []
        tot = collections.defaultdict(lambda:0)
        label = []
        with torch.no_grad():
            off_file_paths = glob(os.path.join("/data/xingyu/modelnet40_processed2500", "*", "car", "*.off"))
            progress = tqdm(off_file_paths)
            for path in progress:
                # 2500
                filename = pathlib.Path(path).name
                cls = pathlib.Path(path).parent.name
                split = pathlib.Path(path).parent.parent.name
                if tot[cls] > 0: continue
                tot[cls]+=1
                xyz, faces = read_mesh(path)
                try:
                    B, N, _ = xyz.shape
                    if N > 2500: continue
                except:
                    continue
                xyz, _, _ = pc_normalize(xyz)
                for face in faces[0]:
                    v0, v1, v2 = xyz[0][face[0]].cpu().detach().numpy(), xyz[0][face[1]].cpu().detach().numpy(), xyz[0][face[2]].cpu().detach().numpy()
                    res = learn_from_relation(v0, v1, v2)
                    X.extend(res)
                    label.extend(["Decimated"] * 6)
                # ori
                path = f"/data/xingyu/ModelNet40/{cls}/{split}/{filename}"
                xyz, faces = read_mesh(path)
                xyz, _, _ = pc_normalize(xyz)
                for face in faces[0]:
                    v0, v1, v2 = xyz[0][face[0]].cpu().detach().numpy(), xyz[0][face[1]].cpu().detach().numpy(), xyz[0][face[2]].cpu().detach().numpy()
                    res = learn_from_relation(v0, v1, v2)
                    X.extend(res)
                    label.extend(["Origin"] * 6)

        X = np.stack(X)
        fig, ax = plt.subplots()
        from sklearn import manifold
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        color_set = []
        label_set = ['Decimated', 'Origin']
        cmap = get_cmap(len(label_set) + 5)
        for i, l in enumerate(label_set):
            print(l, cmap(i))
            color_set.append(cmap(i))
        Y = tsne.fit_transform(X)

        for l in ['Origin', 'Decimated']:
            y_for_current_label = []
            for y, ll in zip(Y, label):
                if ll == l:
                    y_for_current_label.append(y)
            YY = np.stack(y_for_current_label)
            color_idx = label_set.index(l)
            color = color_set[color_idx]
            ax.scatter(YY[:, 0], YY[:, 1], color=color, label=l)
        plt.legend()
        plt.savefig("2.png")


    def distribution_offset(self):
        min_max = defaultdict(lambda: [np.inf, 0])
        metrics = defaultdict(Avg)
        def learn_from_relation(v0, v1, v2):
            return [v0 - v1, v1 - v2, v2 - v0, v1 - v0, v2 - v1, v0 - v2]
        def learn_from_coord(v0, v1, v2):
            return [v0, v1, v2]
        with torch.no_grad():
            off_file_paths = glob(os.path.join("/data/xingyu/modelnet40_processed2500", "*", "*", "*.off"))
            progress = tqdm(off_file_paths)
            for path in progress:
                # 2500
                filename = pathlib.Path(path).name
                cls = pathlib.Path(path).parent.name
                split = pathlib.Path(path).parent.parent.name
                xyz, faces = read_mesh(path)
                try:
                    B, N, _ = xyz.shape
                    if N != 2500: continue
                except:
                    continue
                X1, X2 = [], []
                xyz, _, _ = pc_normalize(xyz)
                for face in faces[0]:
                    v0, v1, v2 = xyz[0][face[0]].cpu().detach().numpy(), xyz[0][face[1]].cpu().detach().numpy(), xyz[0][face[2]].cpu().detach().numpy()
                    res = learn_from_relation(v0, v1, v2)
                    X1.extend(res)
                # ori
                path = f"/data/xingyu/ModelNet40/{cls}/{split}/{filename}"
                xyz, faces = read_mesh(path)
                xyz, _, _ = pc_normalize(xyz)
                for face in faces[0]:
                    v0, v1, v2 = xyz[0][face[0]].cpu().detach().numpy(), xyz[0][face[1]].cpu().detach().numpy(), xyz[0][face[2]].cpu().detach().numpy()
                    res = learn_from_relation(v0, v1, v2)
                    X2.extend(res)
                
                X1_t, X2_t = np.stack(X1), np.stack(X2)
                try:
                    T_value, F_value, p_value, cov = hotelling_t2(X1_t, X2_t)
                    reject = F_value>2.6049
                    if reject:
                        print('hi')
                except:
                    continue
                min_max["T_value"] = [min(min_max["T_value"][0], T_value), max(min_max["T_value"][1], T_value)]
                min_max["F_value"] = [min(min_max["F_value"][0], F_value), max(min_max["F_value"][1], F_value)]
                min_max["p_value"] = [min(min_max["p_value"][0], p_value), max(min_max["p_value"][1], p_value)]
                
                metrics["T_value"].update(T_value, 1)
                metrics["F_value"].update(F_value, 1)
                metrics["p_value"].update(p_value, 1)
                metrics["reject"].update(reject, 1)
                progress.set_postfix_str(
                    f'T: {T_value}, F:{F_value}, P: {p_value}'
                )
        print('\n'.join([f'{k}={v}' for k, v in metrics.items()]))
        print('\n'.join([f'{k}={v}' for k, v in min_max.items()]))