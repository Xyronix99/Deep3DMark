import GPUtil
import math
import os
import torch
import numpy as np
import mlconfig
import logging
from datetime import datetime


def debug(param):
    if V().d is not None:
        print(torch.sum(V().d!=param))
    else:
        V().d = param

class V():
    cfg: mlconfig.Config
    device: int
    logger: logging.Logger
    who_am_I: int
    d=None
    name: str
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(V, cls).__new__(cls)
        return cls.instance
    
    @staticmethod
    def info(msg, **kwargs):
        s = f'{datetime.now()} - <{V().who_am_I}>.INFO: {msg}'
        for k, v in kwargs.items(): s += f', {k}={v}'
        V().logger.info(s)

def pick_gpu():
    """
    Picks a GPU with the least memory load.
    :return:
    """
    gpu = GPUtil.getAvailable(order='memory', limit=math.inf, maxLoad=2, maxMemory=0.1, includeNan=False,
                                    excludeID=[], excludeUUID=[])
    return gpu

def reserve_gpu(gpu_id : list):
    """ Chooses a GPU.
    If None, uses the GPU with the least memory load.
    """
    if gpu_id is None:
        gpu_id = pick_gpu()
    
    if gpu_id:
        s = f'{gpu_id[0]}'
        # for gpu in gpu_id[1:]: s += f',{gpu}'
        os.environ["CUDA_VISIBLE_DEVICES"] = s
    print(f'CUDA_VISIBLE_DEVICES={s}')
    return gpu_id

def calculate_mean(X: np.ndarray):
    """
        X: (N, D) a sample of size N with dimention D
    """
    return X.mean(axis=0)
def calculate_cov_matrix(X: np.ndarray):
    """
        X: (N, D) a sample of size N with dimention D
    """
    N, D = X.shape
    X_bar = calculate_mean(X)
    X_diff = np.expand_dims(X - X_bar, axis=-1) # (N, D, 1)
    X_diff_trans = np.transpose(X_diff, (0, 2, 1)) # (N, 1, D)
    covs = np.matmul(X_diff, X_diff_trans)
    covs = covs.sum(axis=0) / (N - 1)
    return covs

def hotellingTTest(X1, X2):
    D = X1.shape[1]
    N1, N2 = X1.shape[0], X2.shape[0]
    X_bar1, X_bar2 = calculate_mean(X1), calculate_mean(X2)
    X_cov1, X_cov2 = calculate_cov_matrix(X1), calculate_cov_matrix(X2)
    pooled_cov = ((N1 - 1) * X_cov1 + (N2 - 1) * X_cov2) / (N1 + N2 - 2)
    bar_diff = np.expand_dims(X_bar1 - X_bar2, axis=-1) # (3, 1)
    bar_diff_trans = np.transpose(bar_diff) # (1, 3)
    T_square = N1*N2/(N1+N2) * \
        np.matmul(
            np.matmul(
                bar_diff_trans,
                np.linalg.inv(pooled_cov)
            ),
            bar_diff
        )
    F_value = (N1 + N2 - D - 1)/(D * (N1 + N2 - 2)) * T_square
    return T_square[0][0], F_value[0][0]

if __name__ == "__main__":
    X1 = np.array(
        [[1,2,3], [4,5,6], [7,8,9]],
    )
    X2 = np.array(
        [[3,2,1], [6,5,4], [9,8,7]]
    )
    print(X1.shape)
    T, F =hotellingTTest(X1, X2)
    print(T, F)
    V().device=1
    print(V().device)