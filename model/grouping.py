import mlconfig
import torch

import model.backend as B

@mlconfig.register
def graph_neighbor_query_func(nsample):
    def grouping_strategy(new_xyz, xyz, faces):
        idx, num = B.graph_neighbor_query(xyz, faces, nsample)
        return torch.cat([idx, num.unsqueeze(-1)], dim=-1)
    return grouping_strategy

@mlconfig.register
def graph_neighbor_query_all_func(nsample):
    def grouping_strategy(new_xyz, xyz, faces):
        idx, num = B.graph_neighbor_query_all(xyz, faces)
        return torch.cat([idx, num.unsqueeze(-1)], dim=-1)
    return grouping_strategy

@mlconfig.register
def k_neighbor_query_func(nsample):
    def grouping_strategy(new_xyz, xyz, faces):
        idx = B.k_neighbor_query(new_xyz, xyz, nsample) # (B, N, 9)
        b, N, _ = idx.shape
        num = torch.full(size=(b, N, 1), fill_value=nsample).cuda().int()
        return torch.cat([idx, num], dim=-1)
    return grouping_strategy

@mlconfig.register
def ball_query_func(nsample):
    def grouping_strategy(new_xyz, xyz, faces):
        return B.ball_query(radius=0.2, nsample=nsample, xyz=xyz, new_xyz=new_xyz)
    return grouping_strategy

@mlconfig.register
def one_step_neighbor_func(nsample=None):
    def grouping_strategy(new_xyz, xyz, faces):
        return B.build_edge_matrix_from_triangle(xyz, faces)
    return grouping_strategy