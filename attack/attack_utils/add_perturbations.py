import copy

import dgl
import numpy
import scipy
import torch

from attack.attack_utils.hg_info_extract import (
    get_connect_type_matrix,
    get_filtered_etype,
    get_etype_dict,
)


def normalize_grad_n2v(data: [dict], beta: float):
    print("z-score...")

    etypes = set([item["etype"] for item in data])
    grad_means_by_etypes = {}
    grad_stds_by_etypes = {}

    grad_data_splict_by_etypes = {}
    z_scored_grad_data_splict_by_etypes = {}

    n2v_data_splict_by_etypes = {}
    z_scored_n2v_data_splict_by_etypes = {}

    index_of_each_etypes = {}

    for etype in etypes:
        data_item = list(filter(lambda item: item["etype"] == etype, data))

        grad_data_splict_by_etypes[etype] = [item["grad_value"] for item in data_item]
        n2v_data_splict_by_etypes[etype] = [
            item["similarity_score"] for item in data_item
        ]

        grad_means_by_etypes[etype] = numpy.mean(grad_data_splict_by_etypes[etype])
        grad_stds_by_etypes[etype] = numpy.std(grad_data_splict_by_etypes[etype])

        z_scored_grad_data_splict_by_etypes[etype] = scipy.stats.zscore(
            grad_data_splict_by_etypes[etype]
        )
        z_scored_n2v_data_splict_by_etypes[etype] = scipy.stats.zscore(
            n2v_data_splict_by_etypes[etype]
        )

        index_of_each_etypes[etype] = 0

    for i in range(len(data)):
        data_i_etype_ = data[i]["etype"]
        by_etypes_data_i_etype__z_scored_grad_data_splict_by_etypes = (
            z_scored_grad_data_splict_by_etypes[data_i_etype_]
        )
        by_etypes_data_i_etype__z_scored_n2v_data_splict_by_etypes = (
            z_scored_n2v_data_splict_by_etypes[data_i_etype_]
        )

        data[i]["z_scored_grad_value"] = (
            by_etypes_data_i_etype__z_scored_grad_data_splict_by_etypes[
                index_of_each_etypes[data_i_etype_]
            ]
        )
        data[i]["z_scored_similarity_score"] = (
            by_etypes_data_i_etype__z_scored_n2v_data_splict_by_etypes[
                index_of_each_etypes[data_i_etype_]
            ]
        )

        if data[i]["has_edge"]:
            # 有边，新增元邻居，此时，新的元邻居要和原来的差距大，即similarity小
            data[i]["mix_value"] = grad_means_by_etypes[data_i_etype_] * (
                data[i]["z_scored_grad_value"]
                - beta
                * grad_stds_by_etypes[data_i_etype_]
                * data[i]["z_scored_similarity_score"]
            )
        else:
            # 无边，删除元邻居，此时，新的元邻居要和原来的差距小，即similarity大
            data[i]["mix_value"] = grad_means_by_etypes[data_i_etype_] * (
                data[i]["z_scored_grad_value"]
                + beta
                * grad_stds_by_etypes[data_i_etype_]
                * data[i]["z_scored_similarity_score"]
            )
        index_of_each_etypes[data_i_etype_] += 1
    return data


def remove_all_edges_by_etype(
    hg: dgl.DGLHeteroGraph, etype: str, etype_reverse: str, device: str
):
    hg = hg.clone()

    etyped_adj = hg.adjacency_matrix(etype=etype).to_dense()
    etyped_index = torch.nonzero(etyped_adj)
    etyped_eids = hg.edge_ids(u=etyped_index[:, 0], v=etyped_index[:, 1], etype=etype)
    etyped_eids_reverse = hg.edge_ids(
        u=etyped_index[:, 1], v=etyped_index[:, 0], etype=etype_reverse
    )

    hg.remove_edges(etype=etype, eids=etyped_eids)
    hg.remove_edges(etype=etype_reverse, eids=etyped_eids_reverse)

    return hg.to(device)


def from_adj_dict_to_dgl_graph(
    example_graph: dgl.DGLHeteroGraph, adj_dict: {str, torch.Tensor}
):
    etype_dict = get_etype_dict(example_graph)
    num_nodes_dict = {
        ntype: example_graph.number_of_nodes(ntype) for ntype in example_graph.ntypes
    }
    connect_type_matrix = get_connect_type_matrix(example_graph, example_graph.ntypes)
    filtered_etype = get_filtered_etype(example_graph, example_graph.ntypes)
    for etype in filtered_etype:
        source_ntype, target_ntype = etype_dict[etype]
        etype_reverse = connect_type_matrix[target_ntype][source_ntype]
        assert len(torch.where(adj_dict[etype] != adj_dict[etype_reverse].T)[0]) == 0

    etype_dict = {}
    for connect_type in example_graph.canonical_etypes:
        etype_dict[connect_type[1]] = connect_type

    data_dict = {}
    for k, v in etype_dict.items():
        adj_dict_modified_k_ = adj_dict[k]
        data_dict[v] = torch.where(adj_dict_modified_k_ == 1)
    g_modified = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    return g_modified


def random_flip_hg_auto_reverse(clean_hg: dgl.DGLHeteroGraph, radio=0.05, etype=None):
    hg = copy.deepcopy(clean_hg)
    num_nodes_dict = {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes}
    adj_dict = dict(zip(hg.etypes, [hg.adj(etype).to_dense() for etype in hg.etypes]))
    etype_dict = {}
    for connect_type in hg.canonical_etypes:
        etype_dict[connect_type[1]] = connect_type

    connect_type_matrix = get_connect_type_matrix(hg, hg.ntypes)
    filtered_type = get_filtered_etype(hg, hg.ntypes)
    if etype is None:
        adj_dict_modified = copy.deepcopy(adj_dict)
        for etype in filtered_type:
            source_type_, etype_, target_type_ = etype_dict[etype]
            num_purb = int(clean_hg.num_edges(etype_) * radio)
            etype_reverse = connect_type_matrix[target_type_][source_type_]
            sources = torch.randint(
                low=0, high=hg.num_nodes(source_type_) - 1, size=[num_purb]
            )
            targets = torch.randint(
                low=0, high=hg.num_nodes(target_type_) - 1, size=[num_purb]
            )

            etype_ori_adj = hg.adjacency_matrix(etype).to_dense().clone()

            etype_perturb_adj = torch.zeros_like(etype_ori_adj)
            etype_perturb_adj[sources, targets] = 1

            etype_modified_adj = etype_ori_adj - etype_perturb_adj
            etype_modified_adj = torch.where(
                etype_modified_adj == -1,
                torch.ones_like(etype_ori_adj),
                etype_modified_adj,
            )

            adj_dict_modified[etype] = etype_modified_adj
            adj_dict_modified[etype_reverse] = etype_modified_adj.T

        data_dict = {}
        for k, v in etype_dict.items():
            adj_dict_modified_k_ = adj_dict_modified[k]
            data_dict[v] = torch.where(adj_dict_modified_k_ == 1)
        g_modified = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        return g_modified
    else:
        source_type_, etype_, target_type_ = etype_dict[etype]
        num_purb = int(clean_hg.num_edges() * radio)
        etype_reverse = connect_type_matrix[target_type_][source_type_]
        sources = torch.randint(
            low=0, high=hg.num_nodes(source_type_) - 1, size=[num_purb]
        )
        targets = torch.randint(
            low=0, high=hg.num_nodes(target_type_) - 1, size=[num_purb]
        )

        etype_ori_adj = hg.adjacency_matrix(etype).to_dense().clone()

        etype_perturb_adj = torch.zeros_like(etype_ori_adj)
        etype_perturb_adj[sources, targets] = 1

        etype_modified_adj = etype_ori_adj - etype_perturb_adj
        etype_modified_adj = torch.where(
            etype_modified_adj == -1, torch.ones_like(etype_ori_adj), etype_modified_adj
        )

        adj_dict_modified = copy.deepcopy(adj_dict)
        adj_dict_modified[etype] = etype_modified_adj
        adj_dict_modified[etype_reverse] = etype_modified_adj.T

        data_dict = {}
        for k, v in etype_dict.items():
            adj_dict_modified_k_ = adj_dict_modified[k]
            data_dict[v] = torch.where(adj_dict_modified_k_ == 1)
        g_modified = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        return g_modified


def flip_on_hg(
    etype,
    graph_for_modify: dgl.DGLHeteroGraph,
    source,
    target,
    etype_reverse=None,
    auto_reverse=False,
):
    purb_counter = 0

    graph_for_modify = copy.deepcopy(graph_for_modify)

    if isinstance(etype, str) and isinstance(etype_reverse, str):
        for item in graph_for_modify.canonical_etypes:
            if item[1] == etype:
                etype_canonical_etypes = item
            if item[1] == etype_reverse:
                etype_reverse_canonical_etypes = item
    else:
        etype_canonical_etypes = etype
        etype_reverse_canonical_etypes = etype_reverse

    if graph_for_modify.has_edges_between(
        int(source), int(target), etype=etype_canonical_etypes
    ):
        # 有边
        graph_for_modify.remove_edges(
            eids=graph_for_modify.edge_ids(
                source, target, etype=etype_canonical_etypes
            ),
            etype=etype_canonical_etypes,
        )
        purb_counter += 1
        if auto_reverse:
            graph_for_modify.remove_edges(
                eids=graph_for_modify.edge_ids(
                    target, source, etype=etype_reverse_canonical_etypes
                ),
                etype=etype_reverse_canonical_etypes,
            )
            purb_counter += 1
    else:
        # 无边
        graph_for_modify.add_edges(source, target, etype=etype)
        purb_counter += 1
        if auto_reverse:
            graph_for_modify.add_edges(target, source, etype=etype_reverse)
            purb_counter += 1
    return purb_counter, graph_for_modify
