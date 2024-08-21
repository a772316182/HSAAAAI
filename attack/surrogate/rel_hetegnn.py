import dgl
import torch
from torch import nn

from attack.attack_utils.hg_info_extract import get_filtered_etype
from attack.surrogate.gcn_on_homog import GCN_without_norm
from attack.surrogate.hgcn_blocks import GCNInner
from openhgnn import HeteroFeature, get_nodes_dict
from openhgnn.models.ATT_HGCN import ATT_HGCN


class RelationGCNs(nn.Module):
    def __init__(
        self,
        category,
        example_graph: dgl.DGLHeteroGraph,
        h_dict,
        in_size,
        out_size,
        device,
    ):
        super(RelationGCNs, self).__init__()
        self.in_size = in_size
        self.category = category
        self.example_graph = example_graph
        self.filtered_etype = get_filtered_etype(example_graph, example_graph.ntypes)
        self.h_dict = h_dict
        self.out_size = out_size
        self.device = device

        net_schema = dict(zip(example_graph.ntypes, [[] for _ in example_graph.ntypes]))
        adj_dict = dict(zip(example_graph.ntypes, [{} for _ in example_graph.ntypes]))
        for item in example_graph.canonical_etypes:
            source = item[0]
            target = item[2]
            net_schema[source].append(target)
            adj_dict[source][target] = (
                example_graph.adj(etype=item[1]).to_dense().to_sparse()
            )

        self.input_layer_shape = dict([(k, self.in_size) for k in self.h_dict.keys()])
        self.hidden_layer_shape = [
            dict.fromkeys(self.h_dict.keys(), l_hid) for l_hid in [16, 8]
        ]
        self.output_layer_shape = dict.fromkeys(self.h_dict.keys(), self.out_size)
        self.layer_shape = []
        self.layer_shape.append(self.input_layer_shape)
        self.layer_shape.extend(self.hidden_layer_shape)
        self.layer_shape.append(self.output_layer_shape)

        self.SHGP = ATT_HGCN(
            net_schema=net_schema, layer_shape=self.layer_shape, label_keys=[category]
        )

    def forward(self, etype_adj_dict):
        feature = self.h_dict
        logits, embedding_dict, attention_dict = self.SHGP.forward(
            feature, etype_adj_dict
        )
        return logits[self.category]


def get_start_and_end_nid_by_ntype(example_graph: dgl.DGLHeteroGraph, ntype):
    graph_ntypes = example_graph.ntypes
    graph_ntypes_index = graph_ntypes.index(ntype)
    start_nid = 0
    for i in range(graph_ntypes_index):
        start_nid += example_graph.num_nodes(graph_ntypes[i])
    end_nid = start_nid + example_graph.num_nodes(ntype)
    start_nid_end_nid_ = [start_nid, end_nid]
    return start_nid_end_nid_


class GCNFreeCompuse(torch.nn.Module):
    def __init__(
        self, category, example_graph: dgl.DGLHeteroGraph, h_dict, out_size, device
    ):
        super().__init__()
        self.example_graph = example_graph
        self.device = device
        self.weight_dict = {}
        self.etype_pos = {}
        self.adj_pos = {}
        filtered_etypes = get_filtered_etype(example_graph, example_graph.ntypes)
        for item in example_graph.canonical_etypes:
            source_ntype = item[0]
            range_of_src = get_start_and_end_nid_by_ntype(example_graph, source_ntype)
            target_ntype = item[2]
            range_of_tar = get_start_and_end_nid_by_ntype(example_graph, target_ntype)
            etype = item[1]

            self.weight_dict[etype] = torch.nn.Parameter(
                torch.randn(
                    example_graph.num_nodes(source_ntype),
                    example_graph.num_nodes(target_ntype),
                ).to(device)
            )
            self.adj_pos[etype] = [range_of_src, range_of_tar]

        self.zero_adj_example = torch.zeros(
            example_graph.num_nodes(), example_graph.num_nodes(), device=device
        )
        self.feature_build = HeteroFeature(
            {category: h_dict[category]}, get_nodes_dict(example_graph), 32
        ).to(self.device)
        self.zero_feat_example = torch.zeros(
            example_graph.num_nodes(), self.feature_build.embed_size, device=device
        )
        self.fuse_graph = torch.nn.Linear(len(example_graph.etypes), 1, bias=False)
        self.gcn = GCN_without_norm(
            self.device, in_size=32, out_size=out_size, hidden_size=16
        ).to(self.device)

    def adj_norm(self, adj, device):
        mx = adj + torch.eye(adj.shape[0]).to(device)
        mx[torch.where(mx > 1)] = 1
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

    def forward(self, etype_adj_dict):
        zero_adj_example = self.zero_adj_example.clone()
        zero_feat_example = self.zero_feat_example.clone()
        cat_adj_example = self.zero_adj_example.clone()
        feature = self.feature_build.forward()
        for k, v in etype_adj_dict.items():
            pos_str = self.adj_pos[k][0]
            pos_end = self.adj_pos[k][1]
            # todo
            zero_adj_example[pos_str[0] : pos_str[1], pos_end[0] : pos_end[1]] += (
                self.weight_dict[k] * v
            )
            cat_adj_example[pos_str[0] : pos_str[1], pos_end[0] : pos_end[1]] += v
        for ntype in self.example_graph.ntypes:
            and_end_nid_by_ntype = get_start_and_end_nid_by_ntype(
                self.example_graph, ntype
            )
            zero_feat_example[and_end_nid_by_ntype[0] : and_end_nid_by_ntype[1]] = (
                feature[ntype]
            )
        zero_adj_example = (zero_adj_example + zero_adj_example.T) / 2
        self_gcn_forward = self.gcn.forward(zero_adj_example, zero_feat_example)
        return self_gcn_forward


class HeteMetapahAndMulti(torch.nn.Module):
    def __init__(
        self,
        category,
        example_graph,
        meta_paths,
        h_dict,
        in_size,
        out_size,
        hidden_size=16,
        device="cuda:0",
    ):
        super().__init__()
        self.category = category
        self.example_graph = example_graph
        self.meta_paths = meta_paths
        self.h_dict = h_dict
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.device = device
        self.GCNInner = GCNInner(
            meta_paths, in_size, out_size, hidden_size=hidden_size, device=device
        )
        self.GCNFreeCompuse = GCNFreeCompuse(
            category=category,
            example_graph=example_graph,
            h_dict=h_dict,
            device=device,
            out_size=out_size,
        )
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, etype_adj_dict, X):
        gcn_inner_forward = self.GCNInner.forward(adj_dict=etype_adj_dict, X=X)

        gcn_free_compuse_forward = self.GCNFreeCompuse.forward(etype_adj_dict)
        start_end_nid_by_ntype = get_start_and_end_nid_by_ntype(
            self.example_graph, self.category
        )
        gcn_free_compuse_forward_part = gcn_free_compuse_forward[
            start_end_nid_by_ntype[0] : start_end_nid_by_ntype[1]
        ]

        cated_input = torch.stack(
            [gcn_inner_forward, gcn_free_compuse_forward_part], dim=-1
        )
        fused_input = self.linear(cated_input)
        fused_input = fused_input.squeeze_(-1)
        return fused_input


class JustHeteMetapah(HeteMetapahAndMulti):
    def __init__(
        self,
        category,
        example_graph,
        meta_paths,
        h_dict,
        in_size,
        out_size,
        hidden_size=16,
        device="cuda:0",
    ):
        super().__init__(
            category,
            example_graph,
            meta_paths,
            h_dict,
            in_size,
            out_size,
            hidden_size,
            device,
        )

    def forward(self, etype_adj_dict, X):
        gcn_inner_forward = self.GCNInner.forward(adj_dict=etype_adj_dict, X=X)
        return gcn_inner_forward
