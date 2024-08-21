import torch
import torch.nn as nn

from attack.surrogate.gcn_on_homog import GCN


class GCNOuter(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, hidden_size=16, device="cuda:0"):
        super().__init__()
        self.device = device
        self.meta_paths = meta_paths
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.gcn_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gcn_layers.append(
                GCN(device, in_size=in_size, out_size=out_size, hidden_size=hidden_size)
            )
        self.fuse = nn.Linear(len(meta_paths), 1)

    def forward(self, reachable_adj_list, X):
        z_list = []
        for i in range(len(reachable_adj_list)):
            item = reachable_adj_list[i]
            z_list.append(self.gcn_layers[i].forward(item, X))
        z = self.fuse(torch.stack(z_list, dim=-1))
        z = torch.squeeze(z)
        # z = softmax(z + 1e-8, dim=1)
        return z


class GCNInner(GCNOuter):
    def __init__(self, meta_paths, in_size, out_size, hidden_size=16, device="cuda:0"):
        super().__init__(meta_paths, in_size, out_size, hidden_size, device)

    def forward(self, adj_dict, X, return_reachable_adj=False):
        meta_paths = self.meta_paths
        reachable_adj_list = []
        for i in range(len(meta_paths)):
            meta_path_reachable = None
            meta_path = meta_paths[i]
            for etype in meta_path:
                adj_etype = adj_dict[etype]
                if meta_path_reachable is None:
                    meta_path_reachable = adj_etype
                else:
                    meta_path_reachable = meta_path_reachable @ adj_etype
            mx = meta_path_reachable
            mx[torch.where(mx > 1)] = 1
            reachable_adj_list.append(mx)

        z_list = []
        for i in range(len(reachable_adj_list)):
            item = reachable_adj_list[i]
            z_list.append(self.gcn_layers[i].forward(item, X))
        z = self.fuse(torch.stack(z_list, dim=-1))
        z = torch.squeeze(z)
        # z = softmax(z + 1e-8, dim=1)
        if return_reachable_adj:
            return z, reachable_adj_list
        else:
            return z
