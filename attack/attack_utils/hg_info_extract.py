import itertools
import random

import dgl
import numpy
import pandas as pd
import torch

from attack.attack_utils.attack_evaluator import get_pred_label
from openhgnn import extract_metapaths
from openhgnn.trainerflow import NodeClassification


def get_adj_dict_for_surrogate(g: dgl.DGLHeteroGraph, device):
    adj_dict_formated = dict(zip(g.ntypes, [{} for _ in g.ntypes]))
    for item in g.canonical_etypes:
        source = item[0]
        target = item[2]
        adj_dict_formated[source][target] = (
            g.adj(etype=item[1]).to_dense().to_sparse().to(device)
        )
    return adj_dict_formated


def get_transition(given_hete_adjs, metapath_info):
    # transition
    hete_adj_dict_tmp = {}
    for key in given_hete_adjs.keys():
        a = given_hete_adjs[key]
        deg = a.sum(1)
        b = torch.where(deg > 0, deg, 1)
        hete_adj_dict_tmp[key] = a / torch.unsqueeze(b, dim=-1)  # make sure deg>0
    homo_adj_list = []
    for i in range(len(metapath_info)):
        adj = hete_adj_dict_tmp[metapath_info[i][0]]
        for etype in metapath_info[i][1:]:
            adj = adj @ hete_adj_dict_tmp[etype]
        homo_adj_list.append(adj)
    return homo_adj_list


def extract_metapath_from_dataset_for_surrogate(flow: NodeClassification):
    category = flow.category
    feat = flow.hg.ndata["h"][category].cpu()
    dataset_name = flow.args.dataset
    if dataset_name == "acm4GTN":
        meta_paths = [
            ["paper-author", "author-paper"],
            ["paper-subject", "subject-paper"],
        ]
    elif dataset_name == "imdb4GTN":
        meta_paths = [
            ["movie-director", "director-movie"],
            ["movie-actor", "actor-movie"],
        ]
        # meta_paths = [['movie-director', 'director-movie'], ['movie-actor', 'actor-movie'],
        #               ['movie-director', 'director-movie', 'movie-director', 'director-movie']]
    elif dataset_name == "dblp4GTN":
        # meta_paths = [['author-paper', 'paper-author']]
        meta_paths = [
            ["author-paper", "paper-author"],
            # ['author-paper', 'paper-conference', 'conference-paper', 'paper-author'],
            # ['author-paper', 'paper-author', 'author-paper', 'paper-author']
        ]
    else:
        meta_paths = []
        for meta_path in list(
            extract_metapaths(category, flow.hg.canonical_etypes).values()
        ):
            meta_paths.append([paths[1] for paths in meta_path])
    return feat.to(flow.device), meta_paths


def generate_mata_path_reachable_adjs(
    flow: NodeClassification,
    hg: dgl.DGLHeteroGraph,
    numpy_format: bool = False,
    edge_index_format: bool = False,
):
    hg = hg.cpu()
    etype_adjs = {}
    for etype in hg.etypes:
        etype_adj = hg.adj(etype=etype).to_dense().cpu()
        etype_adjs[etype] = etype_adj
    feat, meta_paths = extract_metapath_from_dataset_for_surrogate(flow)

    metapath_reachable_graph_adj_dict = {}
    for i in range(len(meta_paths)):
        meta_path_reachable = None
        meta_path = meta_paths[i]
        for path in meta_path:
            adj_etype = etype_adjs[path]
            if meta_path_reachable is None:
                meta_path_reachable = adj_etype
            else:
                meta_path_reachable = meta_path_reachable @ adj_etype
        meta_path_reachable = (
            meta_path_reachable + torch.eye(meta_path_reachable.shape[0]).cpu()
        )
        meta_path_reachable[torch.where(meta_path_reachable > 1)] = 1

        check_target = dgl.metapath_reachable_graph(
            hg.cpu(), metapath=meta_path
        ).adj().to_dense() + torch.eye(meta_path_reachable.shape[0], device="cpu")
        check_target[torch.where(check_target > 1)] = 1
        assert bool(torch.all(check_target == meta_path_reachable.cpu()))
        if edge_index_format:
            metapath_reachable_graph_adj_dict[str(meta_path)] = (
                torch.nonzero(meta_path_reachable).long().T
            )
        if numpy_format:
            metapath_reachable_graph_adj_dict[str(meta_path)] = (
                meta_path_reachable.cpu().numpy()
            )
        else:
            metapath_reachable_graph_adj_dict[str(meta_path)] = (
                meta_path_reachable.cpu()
            )
    return metapath_reachable_graph_adj_dict


def get_connect_type_matrix(hg: dgl.DGLHeteroGraph, ntypes):
    connect_type_matrix = pd.DataFrame(index=ntypes, columns=ntypes, dtype=str)
    for item in hg.canonical_etypes:
        source = item[0]
        edge_type = item[1]
        target = item[2]
        connect_type_matrix[source][target] = edge_type
    connect_type_matrix = connect_type_matrix.fillna("")
    return connect_type_matrix


def get_node_num_dict(hg: dgl.DGLHeteroGraph):
    return dict(zip(hg.ntypes, [hg.num_nodes(ntype=ntype) for ntype in hg.ntypes]))


def get_edge_num_dict(hg: dgl.DGLHeteroGraph):
    return dict(zip(hg.etypes, [hg.num_edges(etype=etype) for etype in hg.etypes]))


def get_etype_dict(hg: dgl.DGLHeteroGraph):
    return dict(zip(hg.etypes, [(item[0], item[2]) for item in hg.canonical_etypes]))


def get_etype_adj_dict(hg: dgl.DGLHeteroGraph, device):
    return dict(
        zip(
            hg.etypes,
            [
                hg.adj(etype).to_dense().detach().clone().to(device)
                for etype in hg.etypes
            ],
        )
    )


def get_filtered_etype(hg: dgl.DGLHeteroGraph, ntypes: [str]):
    filtered_type = []
    filtered_source_target_ntype = []
    for item in list(itertools.permutations(ntypes, 2)):
        if [item[0], item[1]] not in filtered_source_target_ntype and [
            item[1],
            item[0],
        ] not in filtered_source_target_ntype:
            filtered_source_target_ntype.append([item[1], item[0]])
    connect_type_matrix = get_connect_type_matrix(hg, hg.ntypes)
    for item in filtered_source_target_ntype:
        source_ntype = item[0]
        target_ntype = item[1]
        etype = connect_type_matrix[source_ntype][target_ntype]
        if len(etype) > 0:
            filtered_type.append(etype)
    return filtered_type


def extract_target_nodes(flow: NodeClassification, num_of_target=40):
    pred_label = get_pred_label(flow)
    pred_label = pred_label.cpu().numpy()
    ground_truth_label = flow.labels.cpu().numpy()
    test_idx = flow.test_idx.to("cpu").numpy()
    pred_corrected_test_nodes = []
    to_cpu_test_idx__numpy = numpy.column_stack(
        [test_idx, pred_label, ground_truth_label[test_idx]]
    ).tolist()
    for i in range(len(to_cpu_test_idx__numpy)):
        if to_cpu_test_idx__numpy[i][1] == to_cpu_test_idx__numpy[i][2]:
            pred_corrected_test_nodes.append(to_cpu_test_idx__numpy[i][0])
    # 从中采样40个作为目标节点
    target_nodes = random.sample(pred_corrected_test_nodes, num_of_target)

    return target_nodes
