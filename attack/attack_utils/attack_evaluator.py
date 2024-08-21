import copy

import dgl
import numpy
import torch
from sklearn.metrics import f1_score, accuracy_score

from openhgnn import Experiment
from openhgnn.trainerflow.node_classification import NodeClassification


def get_logits(hg: dgl.DGLHeteroGraph, flow: NodeClassification):
    with torch.no_grad(), hg.local_scope():
        hg.ndata.clear()
        hg.edata.clear()
        hg.dstdata.clear()
        hg.srcdata.clear()
        h_dict = flow.model.input_feature()
        h_dict = {k: e.to(flow.device) for k, e in h_dict.items()}
        logits = flow.model(hg.to(flow.device), h_dict)[flow.category].cpu()
        return logits


def eval_transfer(hg: dgl.DGLHeteroGraph, flow: NodeClassification):
    print(f"ori model is {flow.args.model_name}")
    print("~transfer~" * 10)
    example_graph = copy.deepcopy(flow.hg.clone())
    dataset_name = flow.args.dataset_name
    all_models = ["HAN", "RGCN", "fastGTN"]
    micro_downs = []
    macro_downs = []
    for model_name in all_models:
        print(f"to {model_name}")
        victim_graph = copy.deepcopy(hg.clone())
        experiment = Experiment(
            model=model_name,
            dataset=dataset_name,
            task=flow.args.task,
            gpu=flow.new_args.gpu,
            use_best_config=True,
            load_from_pretrained=False,
        )

        new_result, new_flow = experiment.run()
        new_ori_micro_f1, new_ori_macro_f1, _, _ = eval_evasion(
            example_graph, new_flow, flow.test_idx
        )
        new_victim_micro_f1, new_victim_macro_f1, _, _ = eval_evasion(
            victim_graph, new_flow, flow.test_idx
        )
        micro_down = new_ori_micro_f1 - new_victim_micro_f1
        macro_down = new_ori_macro_f1 - new_victim_macro_f1

        print(f"ori_micro => {new_ori_micro_f1 * 100}")
        print(f"victim_mirco => {new_victim_micro_f1 * 100}")
        print(f"ori_macro => {new_ori_macro_f1 * 100}")
        print(f"victim_marco => {new_victim_macro_f1 * 100}")
        print(
            f"transfer to model {model_name}, micro_down is {micro_down}, macro_down is {macro_down}"
        )

        micro_downs.append(micro_down)
        macro_downs.append(macro_down)
    print(micro_downs)
    print(macro_downs)
    print("===")
    mean_micro_down = float(numpy.mean(micro_downs))
    mean_macro_down = float(numpy.mean(macro_downs))
    print("===")
    print(mean_micro_down)
    print(mean_macro_down)
    return mean_micro_down, mean_macro_down


def eval_evasion(
    hg: dgl.DGLHeteroGraph,
    flow: NodeClassification,
    target_nodes: [int] = None,
    return_acc: bool = False,
):
    with torch.no_grad(), hg.local_scope():
        hg.ndata.clear()
        hg.edata.clear()
        hg.dstdata.clear()
        hg.srcdata.clear()
        h_dict = flow.model.input_feature()
        h_dict = {k: e.to(flow.device) for k, e in h_dict.items()}
        logits = flow.model(hg.to(flow.device), h_dict)[flow.category].cpu()
        if target_nodes is None:
            mask = flow.test_idx
        else:
            mask = torch.tensor(target_nodes).cpu()
        pred = logits[mask].argmax(dim=1).cpu()
        label = flow.labels[mask].cpu()
        macro_f1 = f1_score(label, pred, average="macro")
        micro_f1 = f1_score(label, pred, average="micro")
        loss = flow.loss_fn(logits[mask], torch.squeeze(label)).item()
    if return_acc:
        acc = accuracy_score(label, pred)
        return acc, loss, logits
    else:
        return micro_f1, macro_f1, loss, logits


def get_pred_label(flow: NodeClassification, hg=None, mask=None):
    with torch.no_grad(), hg.local_scope():
        hg.ndata.clear()
        hg.edata.clear()
        hg.dstdata.clear()
        hg.srcdata.clear()
        h_dict = flow.input_feature()
        hg_ = hg if hg else flow.hg
        hg_ = hg_.clone().to(flow.device)
        logits = flow.model(hg_, h_dict)[flow.category]
        if mask is None:
            mask = flow.test_idx
        loss = flow.loss_fn(logits[mask], flow.labels[mask]).item()
        if flow.task.multi_label:
            pred = (logits[mask].cpu().numpy() > 0).astype(int)
        else:
            pred = logits[mask].argmax(dim=1).to("cpu")
        return pred


def extract_targets(flow: NodeClassification):
    idx_test = flow.test_idx
    output = get_logits(flow.hg, flow)
    labels = flow.labels
    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0:  # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    high = [x.item() for x, y in sorted_margins[:10]]
    low = [x.item() for x, y in sorted_margins[-10:]]
    other = [x for x, y in sorted_margins[10:-10]]
    other = numpy.random.choice(other, 20, replace=False).tolist()

    return high + low + other


def is_attack_success_on_single_node(
    flow: NodeClassification, hg: dgl.DGLHeteroGraph, target_node: int
):
    mask = torch.tensor([target_node], device=flow.device)
    pred = get_pred_label(flow=flow, hg=hg, mask=mask)
    if pred.item() == flow.labels[mask]:
        success = False
    else:
        success = True
    return success


def classification_margin(output, true_label):
    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()
