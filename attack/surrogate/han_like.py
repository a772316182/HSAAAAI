import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import DenseGraphConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HeteGNN_Surrogate_Inner_Compose_DGL(nn.Module):
    def __init__(
        self, meta_paths, in_size, out_size, hidden_size=8, num_heads=1, device="cuda:0"
    ):
        super().__init__()
        self.device = device
        self.model_type = "hgcn"
        self.meta_paths = meta_paths
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                DenseGraphConv(
                    in_size,
                    hidden_size,
                    activation=torch.nn.functional.relu,
                    bias=False,
                )
            )
        in_size = hidden_size * num_heads
        self.semantic_attention = SemanticAttention(in_size)
        self.predict = nn.Linear(in_size, out_size)

    def forward(
        self,
        A_list,
        X,
        return_reachable_adj=False,
        hg: dgl.DGLHeteroGraph = None,
        check=False,
    ):
        mx_list = []
        meta_path_Z_list = []
        for i in range(len(self.meta_paths)):
            meta_path_reachable = None
            meta_path = self.meta_paths[i]
            for path in meta_path:
                adj_etype = A_list[path]
                if meta_path_reachable is None:
                    meta_path_reachable = adj_etype
                else:
                    meta_path_reachable = meta_path_reachable @ adj_etype
            # self loop
            mx = meta_path_reachable + torch.eye(meta_path_reachable.shape[0]).to(
                meta_path_reachable.device
            )
            mx[torch.where(mx > 1)] = 1
            if check:
                check_target = dgl.metapath_reachable_graph(
                    hg.to(self.device), metapath=meta_path
                ).adj().to_dense() + torch.eye(
                    meta_path_reachable.shape[0], device=meta_path_reachable.device
                )
                check_target[torch.where(check_target > 1)] = 1
                assert bool(torch.all(check_target == mx))
            mx_list.append(mx)
            mx_x__flatten = self.gat_layers[i].forward(mx, X).flatten(1)
            mx_x__flatten = torch.nn.functional.normalize(mx_x__flatten)
            meta_path_Z_list.append(mx_x__flatten)

        meta_path_Z = torch.stack(meta_path_Z_list, dim=1)  # 3025 * 2 * 8
        meta_path_Z_attention = self.semantic_attention(meta_path_Z)  # 3025 * 2 * 8
        Z2 = self.predict(meta_path_Z_attention)  # 3025 * 3
        Z2_softmax = torch.nn.functional.softmax(Z2 + 1e-8, dim=1)
        if return_reachable_adj:
            return Z2_softmax, mx_list
        else:
            return Z2_softmax


class HGCN_Outer_Compose_DGL(HeteGNN_Surrogate_Inner_Compose_DGL):
    def __init__(
        self,
        meta_paths,
        in_size,
        out_size,
        hidden_size=8,
        num_heads=1,
    ):
        super().__init__(meta_paths, in_size, out_size, hidden_size, num_heads)

    def forward(self, mx_list, X):
        meta_path_Z_list = []
        for i in range(len(self.meta_paths)):
            mx = mx_list[i]
            mx_x__flatten = self.gat_layers[i].forward(mx, X).flatten(1)
            mx_x__flatten = torch.nn.functional.normalize(mx_x__flatten)
            meta_path_Z_list.append(mx_x__flatten)
        meta_path_Z = torch.stack(meta_path_Z_list, dim=1)  # 3025 * 2 * 8
        meta_path_Z_attention = self.semantic_attention(meta_path_Z)  # 3025 * 2 * 8
        Z2 = self.predict(meta_path_Z_attention)  # 3025 * 3
        Z2_softmax = torch.nn.functional.softmax(Z2 + 1e-8, dim=1)
        return Z2_softmax
