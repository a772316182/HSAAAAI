import copy

import torch as th
from loguru import logger
from torch.nn.functional import cross_entropy

from attack.attack_utils.hg_info_extract import *


def _transform_relation_graph_list(hg, category, identity=True):
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    g = dgl.to_homogeneous(hg, ndata="h")
    # find out the target node ids in g
    loc = (g.ndata[dgl.NTYPE] == category_id).to("cpu")
    category_idx = th.arange(g.num_nodes())[loc]

    edges = g.edges()
    etype = g.edata[dgl.ETYPE]
    ctx = g.device
    num_edge_type = th.max(etype).item()

    graph_list = []
    for i in range(num_edge_type + 1):
        e_ids = th.nonzero(etype == i).squeeze(-1)
        sg = dgl.graph((edges[0][e_ids], edges[1][e_ids]), num_nodes=g.num_nodes())
        sg.edata["w"] = th.ones(sg.num_edges(), device=ctx)
        adj__to_dense = sg.adj().to_dense()
        z = {"adj": adj__to_dense, "w": th.ones(sg.num_edges(), device=ctx)}
        graph_list.append(z)
    if identity == True:
        x = th.arange(0, g.num_nodes(), device=ctx)
        sg = dgl.graph((x, x))
        sg.edata["w"] = th.ones(g.num_nodes(), device=ctx)
        sg_adj__to_dense = sg.adj().to_dense()
        z = {"adj": sg_adj__to_dense, "w": th.ones(sg.num_edges(), device=ctx)}
        graph_list.append(z)
    return graph_list, g.ndata["h"], category_idx


def get_grad_GTN_dgl(
    flow, hg, surrogate_model_inner_compose, surrogate_model_outer_compose, mask=None
):
    surrogate_model_inner_compose.eval()
    surrogate_model_outer_compose.eval()

    if mask is None:
        mask = flow.test_idx
    mask = torch.flatten(mask).to(flow.device)

    hg: dgl.DGLHeteroGraph = copy.deepcopy(hg.cpu()).to(flow.device)
    device = flow.device
    g_homo = dgl.to_homogeneous(hg)

    edge_num_dict = get_edge_num_dict(hg)
    node_num_dict = get_node_num_dict(hg)
    etype_adj_dict = get_etype_adj_dict(hg, "cpu")
    etype_dict = get_etype_dict(hg)

    hg_eid_dict = {}
    for etype in hg.etypes:
        hg_eid_dict[etype] = torch.column_stack(list(hg.edges(etype=etype)))
    homog_heteg_nid_dict = {}
    for i, ntype in enumerate(hg.ntypes):
        torch_where = torch.where(g_homo.ndata[dgl.NTYPE] == i)[0]
        homog_heteg_nid_dict[ntype] = {}
        homog_heteg_nid_dict[ntype]["hete"] = g_homo.ndata[dgl.NID][torch_where]
        homog_heteg_nid_dict[ntype]["homo"] = g_homo.nodes()[torch_where]
    filtered_type = get_filtered_etype(hg, hg.ntypes)
    flow_category = flow.category
    graph_list, feat, category_idx = _transform_relation_graph_list(hg, flow_category)
    for i, v in enumerate(graph_list):
        graph_list[i]["adj"].requires_grad = True
    outer_compose_forward = surrogate_model_outer_compose.forward(
        graph_list, feat, category_idx
    )
    forward_flow_category_ = outer_compose_forward[flow_category]
    logits = forward_flow_category_[mask]
    pred_label = logits.argmax(dim=1)
    loss = cross_entropy(logits, pred_label)

    etype_grad_dict = {}
    grad_sum = 0
    etypes = hg.etypes
    for i, v in enumerate(graph_list):
        if i != len(etypes):
            k = etypes[i]
            etype_grad_dict[k] = torch.autograd.grad(loss, v["adj"], retain_graph=True)[
                0
            ]
            grad_sum += etype_grad_dict[k].sum()

    grad = list(etype_grad_dict.values())[0]

    assert grad_sum != 0
    logger.debug("grad_sum", grad_sum.item())

    typed_grad = {}
    for etype in filtered_type:
        source_ntype, target_ntype = etype_dict[etype]
        source_ntype_homo_nid = homog_heteg_nid_dict[source_ntype]["homo"]
        target_ntype_homo_nid = homog_heteg_nid_dict[target_ntype]["homo"]
        typed_grad[etype] = grad[source_ntype_homo_nid, :][:, target_ntype_homo_nid]

    return typed_grad


def get_logits_HGCN_dgl(
    flow, hg, surrogate_model_inner_compose, surrogate_model_outer_compose, mask=None
):
    surrogate_model_outer_compose.eval()
    surrogate_model_inner_compose.eval()

    if mask is None:
        mask = flow.test_idx
    mask = mask.to(flow.device)

    feat, meta_paths = extract_metapath_from_dataset_for_surrogate(flow)
    hg = copy.deepcopy(hg.cpu())
    device = flow.device

    etype_adj_dict = get_etype_adj_dict(hg, device)
    for k, v in etype_adj_dict.items():
        etype_adj_dict[k].requires_grad = True

    logits = surrogate_model_inner_compose.forward(etype_adj_dict, feat)[mask]
    return logits


def get_grad_SHGP_dgl(flow, hg, adj_dict, surrogate_model, mask=None, verbose=True):
    device = flow.device
    connect_type_matrix = get_connect_type_matrix(hg, hg.ntypes)
    surrogate_model.eval()
    for k, v in adj_dict.items():
        for kk, vv in v.items():
            vv.requires_grad = True
    if mask is None:
        mask = flow.test_idx
    mask = mask.to(flow.device)
    logits = surrogate_model(adj_dict)
    pred_label = get_pred_label(flow, hg, mask).to(device)
    loss = cross_entropy(logits[mask], pred_label.long().detach())

    grad_sum = 0.0
    etype_grad_dict = {}
    for k, v in adj_dict.items():
        for kk, vv in v.items():
            grad_ = torch.autograd.grad(loss, vv, retain_graph=True, allow_unused=True)[
                0
            ]
            source_ntype = k
            target_ntype = kk
            etype = connect_type_matrix[source_ntype][target_ntype]
            if grad_ is not None:
                etype_grad_dict[etype] = grad_
                grad_sum += etype_grad_dict[etype].sum()
            else:
                etype_grad_dict[etype] = torch.zeros_like(v, device=v.device)
    assert grad_sum != 0
    assert grad_sum != 0
    if verbose:
        logger.debug("grad_sum", grad_sum.item())
    filtered_type = get_filtered_etype(hg, hg.ntypes)
    etype_dict = get_etype_dict(hg)
    if verbose:
        logger.debug("===grad extracting===")
    etype_grads_fused = {}
    for etype in filtered_type:
        source_ntype = etype_dict[etype][0]
        target_ntype = etype_dict[etype][1]
        etype_reverse = connect_type_matrix[target_ntype][source_ntype]
        etype_grad = etype_grad_dict[etype] + etype_grad_dict[etype_reverse].T
        etype_grads_fused[etype] = etype_grad
        if verbose:
            logger.debug(
                f"grad fuse, the grad mean of [{etype}] is {etype_grad.mean().item()}"
            )

    return etype_grads_fused


def get_grad_HGCN_dgl(
    flow,
    hg,
    surrogate_model_inner_compose,
    surrogate_model_outer_compose,
    mask=None,
    verbose=True,
):
    surrogate_model_outer_compose.eval()
    surrogate_model_inner_compose.eval()

    if mask is None:
        mask = flow.test_idx
    mask = mask.to(flow.device)

    feat, meta_paths = extract_metapath_from_dataset_for_surrogate(flow)
    hg = copy.deepcopy(hg.cpu())
    device = flow.device

    etype_adj_dict = get_etype_adj_dict(hg, device)
    for k, v in etype_adj_dict.items():
        etype_adj_dict[k].requires_grad = True

    logits = surrogate_model_inner_compose.forward(etype_adj_dict, feat)[mask]
    pred_label = get_pred_label(flow, hg, mask).to(device)
    loss = cross_entropy(logits, pred_label)

    etype_grad_dict = {}
    grad_sum = 0
    for k, v in etype_adj_dict.items():
        grad_ = torch.autograd.grad(loss, v, retain_graph=True, allow_unused=True)[0]
        if grad_ is not None:
            etype_grad_dict[k] = grad_
            grad_sum += etype_grad_dict[k].sum()
        else:
            etype_grad_dict[k] = torch.zeros_like(v, device=v.device)
    assert grad_sum != 0
    if verbose:
        print("grad_sum", grad_sum.item())
    filtered_type = get_filtered_etype(hg, hg.ntypes)
    connect_type_matrix = get_connect_type_matrix(hg, hg.ntypes)
    etype_dict = get_etype_dict(hg)
    if verbose:
        print("===grad extracting===")
    etype_grads_fused = {}
    for etype in filtered_type:
        source_ntype = etype_dict[etype][0]
        target_ntype = etype_dict[etype][1]
        etype_reverse = connect_type_matrix[target_ntype][source_ntype]
        etype_grad = etype_grad_dict[etype] + etype_grad_dict[etype_reverse].T
        etype_grads_fused[etype] = etype_grad
        if verbose:
            logger.debug(
                f"grad fuse, the grad mean of [{etype}] is {etype_grad.mean().item()}"
            )

    return etype_grads_fused
