import os
import numpy as np
import mlconfig
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from util import V

class RandomBitsDataset(Dataset):
    def __init__(self, root, split, **kwargs):
        super().__init__()
        msg_length = V().cfg.msg_length
        os.makedirs(root, exist_ok=True)
        self.save_path = os.path.join(root, f'random{msg_length}_{split}.dat')
        # self.save_path = f'/data_HDD/zhuxingyu/vscode/necst/data/random/random_bits_{split}.npy'
        nsample = dict(train=20000, test=4000)
        if not os.path.exists(self.save_path):
            V().info(f"Processing {self.save_path}...")
            tot, end = sum(nsample.values()), 0
            data = np.reshape(np.random.randint(0, 2, tot * msg_length), (tot, -1)).astype(np.float32)
            for split_1, length in nsample.items():
                save_path = os.path.join(root, f'random{msg_length}_{split}.dat')
                with open(save_path, 'wb') as f:
                    pickle.dump([data[end:end+length]], f)
                if split==split_1:
                    self.data = data[end:end+length]
                end+=length
        else:
            V().info(f"Load processed data from {self.save_path}...")
            with open(self.save_path, 'rb') as f:
                self.data = pickle.load(f)[0]
            # self.data = np.load(self.save_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


@mlconfig.register
class RandomBitsDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle, **dataset_kwargs):
        dataset = RandomBitsDataset(**dataset_kwargs)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10)