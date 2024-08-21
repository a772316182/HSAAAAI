import operator

from loguru import logger
from torch import optim
from torch.nn.functional import mse_loss, cross_entropy
from tqdm import tqdm

from attack.attack_utils.add_perturbations import *
from attack.attack_utils.attack_evaluator import *
from attack.attack_utils.grad_extract import get_grad_SHGP_dgl
from attack.attack_utils.hg_info_extract import *
from attack.surrogate.rel_hetegnn import RelationGCNs
from openhgnn import set_random_seed


def hsa_starter(
    flow: NodeClassification,
    seed: int,
    purb_limit_per_target: int,
    level_3_query_limit: int,
    level_1_training_epoch: int,
    level_2_query_limit: int,
    level_2_training_epoch: int,
    level_3_training_epoch: int,
    weight_decay_lv1: float,
    weight_decay_lv3: float,
    surrogate_lr_lv1: float,
    surrogate_lr_lv3: float,
    level_3_training_batch_size: int,
    lv3_attack_rate: float,
    weight_decay_lv2: float,
    surrogate_lr_lv2: float,
    level_2_training_batch_size: int,
):
    logger.warning(f"attack model {flow.model_name} with seed {seed}")
    logger.warning(
        f"dataset {flow.args.dataset_name}, attack budget {purb_limit_per_target}, query budget {level_2_query_limit + level_3_query_limit}"
    )
    logger.warning(
        f"alpha is {level_2_query_limit / (level_3_query_limit+ level_2_query_limit)}"
    )

    set_random_seed(seed)
    category_type = flow.category
    victim_nodes = extract_targets(flow)
    flow_test_idx = flow.test_idx
    lv1_mask = flow_test_idx
    lv2_mask = flow_test_idx

    victim_model = flow.model
    victim_model.eval()
    device = flow.args.device
    example_graph: dgl.DGLHeteroGraph = copy.deepcopy(flow.hg.clone())
    connect_type_matrix = get_connect_type_matrix(example_graph, example_graph.ntypes)
    etype_dict = get_etype_dict(example_graph)
    h_dict = flow.hg.ndata["h"]
    feat, _ = extract_metapath_from_dataset_for_surrogate(flow)
    feat = feat.to(device)
    in_size = feat.shape[1]
    out_size = flow.num_classes

    surrogate_model = RelationGCNs(
        category=category_type,
        example_graph=example_graph,
        h_dict=h_dict,
        in_size=in_size,
        out_size=out_size,
        device=device,
    ).to(device)

    optimizer_lv1 = optim.Adagrad(surrogate_model.parameters(), lr=surrogate_lr_lv1)
    for i in range(level_1_training_epoch):
        target_model_pred = get_pred_label(flow, example_graph, lv1_mask).to(device)
        surrogate_model_logits = surrogate_model.forward(
            get_adj_dict_for_surrogate(example_graph, device)
        )
        surrogate_model_pred = surrogate_model_logits[lv1_mask].argmax(dim=1)
        loss_train = cross_entropy(surrogate_model_logits[lv1_mask], target_model_pred)
        optimizer_lv1.zero_grad()
        loss_train.backward()
        optimizer_lv1.step()

    """
       lv2 training
    """
    # note this buffer trick
    lv2_buffer = []
    graph_canonical_etypes = copy.deepcopy(example_graph.canonical_etypes)
    for i in tqdm(range(level_2_query_limit)):
        _, etype, _ = random.choice(graph_canonical_etypes)
        hg_lv2 = random_flip_hg_auto_reverse(
            example_graph, 0.3, etype
        )
        target_model_pred = get_pred_label(flow, hg_lv2, lv1_mask).to(
            device
        )
        lv2_buffer.append(
            {
                "chosen_etype": etype,
                "hg": hg_lv2.cpu(),
                "pred": target_model_pred.cpu(),
            }
        )

    optimizer_lv2 = optim.Adagrad(surrogate_model.parameters(), lr=surrogate_lr_lv2)
    for i in range(level_2_training_epoch):
        surrogate_model.train()
        batchs = random.choices(lv2_buffer, k=level_2_training_batch_size)
        loss_train = 0
        for item in batchs:
            hg_lv2 = item["hg"].to(device)
            pred = item["pred"].to(device)
            surrogate_model_logits = surrogate_model.forward(
                get_adj_dict_for_surrogate(hg_lv2, device)
            )
            surrogate_model_pred = surrogate_model_logits[lv1_mask].argmax(dim=1)
            loss_train += cross_entropy(surrogate_model_logits[lv1_mask], pred)
        optimizer_lv2.zero_grad()
        loss_train.backward()
        optimizer_lv2.step()
        logger.info(f"PHASE 1 => In epoch {i}, loss: {loss_train.item():.3f}")

    etype_grads = get_grad_SHGP_dgl(
        flow,
        example_graph,
        get_adj_dict_for_surrogate(example_graph, device),
        surrogate_model
    )

    etype_grad_summary = dict(zip(
        etype_grads.keys(),
        torch.softmax(torch.tensor([torch.norm(item, p=2) for item in etype_grads.values()]), dim=-1).tolist()
    ))

    """
        lv3 training
    """
    lv3_buffer = []
    logger.info(
        "=====================decision boundary exploration====================="
    )
    for i in tqdm(range(level_3_query_limit)):
        hg_lv3 = example_graph.clone()
        hg_lv3 = random_flip_hg_auto_reverse(hg_lv3, radio=0.3)
        etype = random.choices(list(etype_grad_summary.keys()), weights=list(etype_grad_summary.values()), k=1)[0]
        hg_lv3 = preserve_one_etypes(hg_lv3, etype, device)

        target_model_pred = get_pred_label(flow, hg_lv3, lv2_mask).to(
            device
        )

        lv3_buffer.append(
            {
                "chosen_etype": etype,
                "hg": hg_lv3.cpu(),
                "pred": target_model_pred.cpu(),
            }
        )

    logger.info("=====================buffer statistics=====================")
    pred_mse_dicts = dict(zip(example_graph.etypes, [[] for _ in example_graph.etypes]))
    for item in lv3_buffer:
        if item["chosen_etype"] is not None:
            pred_mse_dicts[item["chosen_etype"]].append(
                mse_loss(item["pred"].float(), lv3_buffer[0]["pred"].float()).item()
            )
    for k, v in pred_mse_dicts.items():
        logger.info(k, numpy.mean(v))

    optimizer_lv3 = optim.NAdam(
        surrogate_model.parameters(), lr=surrogate_lr_lv3, weight_decay=weight_decay_lv3
    )
    for i in range(level_3_training_epoch):
        surrogate_model.train()
        batchs = random.choices(lv3_buffer, k=level_3_training_batch_size)
        loss_train = 0.0
        for item in batchs:
            hg = item["hg"].to(device)
            pred = item["pred"].to(device)

            surrogate_model_logits = surrogate_model.forward(
                get_adj_dict_for_surrogate(hg, device)
            )
            surrogate_model_pred = surrogate_model_logits[lv2_mask].argmax(dim=1)

            loss_train += cross_entropy(surrogate_model_logits[lv2_mask], pred)
        optimizer_lv3.zero_grad()
        loss_train.backward()
        optimizer_lv3.step()
        logger.debug(
            f"surrogate_model_pred sum {surrogate_model_pred.sum().item()}, surrogate_model_logits sum {surrogate_model_logits.sum().item()}"
        )
        logger.info(f"PHASE 2 => In epoch {i}, loss: {loss_train.item():.3f}")

    cnt = 0
    iii = 0
    filtered_etype = get_filtered_etype(example_graph, example_graph.ntypes)
    for target_victim_node in victim_nodes:
        logger.debug("=" * 100)
        logger.info(f"attacking node {iii} => total {len(victim_nodes)}")
        graph_modified = copy.deepcopy(example_graph).clone()
        graph_modified_adj_dict = get_etype_adj_dict(graph_modified, device)
        for i in range(purb_limit_per_target):
            etype_grads = get_grad_SHGP_dgl(
                flow,
                graph_modified,
                get_adj_dict_for_surrogate(graph_modified, device),
                surrogate_model,
                torch.tensor([target_victim_node]),
            )
            perturbations_on_each_etype = []
            for etype in filtered_etype:
                [source_ntype, target_ntype] = etype_dict[etype]
                etype_reverse = connect_type_matrix[target_ntype][source_ntype]
                modified_adj_etype = graph_modified_adj_dict[etype]
                grad = etype_grads[etype]
                # todo check!!!
                grad = grad * (-2 * modified_adj_etype + 1)

                if category_type == source_ntype:
                    torch_argmax_grad__item = torch.argmax(
                        grad[target_victim_node, :]
                    ).item()
                    source = target_victim_node
                    target = torch_argmax_grad__item
                    grad_max_value = torch.max(grad[target_victim_node, :])
                elif category_type == target_ntype:
                    torch_argmax_grad__item = torch.argmax(
                        grad[:, target_victim_node]
                    ).item()
                    source = torch_argmax_grad__item
                    target = target_victim_node
                    grad_max_value = torch.max(grad[:, target_victim_node])
                else:
                    continue

                perturbations_on_each_etype.append(
                    {
                        "has_edge": (
                            graph_modified_adj_dict[etype][source, target] == 1
                        ).item(),
                        "etype": etype,
                        "etype_reverse": etype_reverse,
                        "source": source,
                        "source_ntype": source_ntype,
                        "target": target,
                        "target_ntype": target_ntype,
                        "grad_value": grad_max_value.item(),
                        "mix_value": grad_max_value.item(),
                    }
                )
            perturbations_on_each_etype.sort(
                key=operator.itemgetter("mix_value"), reverse=True
            )
            chosen_perturbation = perturbations_on_each_etype[0]
            chosen_etype = chosen_perturbation["etype"]
            chosen_etype_reverse = chosen_perturbation["etype_reverse"]
            chosen_source = chosen_perturbation["source"]
            chosen_target = chosen_perturbation["target"]
            logger.debug(chosen_perturbation)
            modified_adj_etype = graph_modified_adj_dict[chosen_etype]
            modified_adj_etype_reverse = graph_modified_adj_dict[chosen_etype_reverse]
            value = -2 * modified_adj_etype[chosen_source][chosen_target] + 1
            value_reverse = (
                -2 * modified_adj_etype_reverse[chosen_target][chosen_source] + 1
            )

            # modified_adj.data[target_node][grad_argmax] += value
            # modified_adj.data[grad_argmax][target_node] += value

            graph_modified_adj_dict[chosen_etype][chosen_source, chosen_target] += value
            graph_modified_adj_dict[chosen_etype_reverse][
                chosen_target, chosen_source
            ] += value_reverse
        graph_modified = from_adj_dict_to_dgl_graph(
            example_graph=example_graph, adj_dict=graph_modified_adj_dict
        )
        if is_attack_success_on_single_node(flow, graph_modified, target_victim_node):
            cnt += 1
        iii += 1
    ASR = cnt / len(victim_nodes)
    logger.info("FINISH mis-classification rate (ASR): %s" % ASR)
    logger.info("FINISH mis-classification rate (ASR): %s" % ASR)
    logger.info("FINISH mis-classification rate (ASR): %s" % ASR)
    return ASR
