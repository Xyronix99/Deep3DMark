import torch
import numpy as np
from scipy.spatial import KDTree

def to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.cpu().detach().numpy()
    return a

class Avg:
    def __init__(self):
        self.sum = 0
        self.cnt = 0
    def update(self, x, n):
        self.sum+=x
        self.cnt+=n
    def value(self):
        if self.cnt == 0:
            return 0
        return self.sum/self.cnt
    def __str__(self):
        return str(self.value())
    def __add__(self, other):
        res = Avg()
        res.sum = self.sum + other.sum
        res.cnt = self.cnt + other.cnt
        return res

def hausdorff(x, y):
    """
        x,y: (N, D)
    """
    x, y = to_numpy(x), to_numpy(y)
    xTree = KDTree(x)
    y_to_x_dist, _ = xTree.query(y, 1)
    yTree = KDTree(y)
    x_to_y_dist, _ = yTree.query(x, 1)
    y_to_x_max_dist = np.max(y_to_x_dist)
    x_to_y_max_dist = np.max(x_to_y_dist)
    return max(y_to_x_max_dist, x_to_y_max_dist)
        
def SNR(ori, wm):
    mean_v = ori.mean(dim=1, keepdim=True)
    son = ((ori - mean_v)**2).sum(dim=-1).sum(dim=-1)
    mother = ((wm - ori)**2).sum(dim=-1).sum(dim=-1)
    snr = son / mother
    return 10 * torch.log10(snr).mean()

if __name__ == "__main__":
    import torch
    B, N, D = 4, 10, 3
    a, b = torch.randn((B, N, D)), torch.randn((B, N, D))
    d = hausdorff(a, b)
    print(d)

    from chamferdist import ChamferDistance
    chamferDist = ChamferDistance()
    dist_forward = chamferDist(a, b)
    print(dist_forward)
    a,b=a*10,b*10
    dist_forward = chamferDist(a, b)
    print(dist_forward)