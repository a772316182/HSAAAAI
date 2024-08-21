import argparse
import warnings

import numpy
import sklearn.metrics
import torch
import torch_geometric
from loguru import logger

from attack.attack_utils.add_perturbations import random_flip_hg_auto_reverse
from attack.attack_utils.hg_info_extract import (
    get_filtered_etype,
    get_connect_type_matrix,
    get_etype_dict,
)
from openhgnn.experiment import Experiment


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="HAN", type=str, help="name of models")
    parser.add_argument(
        "--task", "-t", default="node_classification", type=str, help="name of task"
    )
    parser.add_argument(
        "--dataset", "-d", default="acm4GTN", type=str, help="name of datasets"
    )
    parser.add_argument("--gpu", "-g", default="0", type=int, help="-1 means cpu")
    parser.add_argument(
        "--use_best_config", action="store_true", help="will load utils.best_config"
    )
    parser.add_argument(
        "--load_from_pretrained",
        action="store_true",
        help="load model from the checkpoint",
    )
    args = parser.parse_args()

    seed = 0
    torch_geometric.seed_everything(seed)

    load_from_pretrained = False
    use_best_config = args.model != "MHNF" and args.dataset != "dblp4GTN"

    experiment = Experiment(
        model=args.model,
        dataset=args.dataset,
        task=args.task,
        gpu=args.gpu,
        use_best_config=use_best_config,
        load_from_pretrained=load_from_pretrained,
    )

    result, flow = experiment.run()
    flow.new_args = args

    hg = flow.hg.clone()
    new_hg = hg.clone()
    flow.model.eval()
    h_dict = flow.model.input_feature()
    feat = {k: e.to(flow.device) for k, e in h_dict.items()}
    logits = flow.model.forward(new_hg, feat)
    pred = logits[flow.category][flow.test_idx].argmax(dim=1).cpu()
    label = flow.labels[flow.test_idx].cpu()
    ori_mf1 = sklearn.metrics.f1_score(label, pred, average="micro")
    for purb_rate in [0.01, 0.03, 0.05]:
        etypes = get_filtered_etype(hg, hg.ntypes)
        connect_type_matrix = get_connect_type_matrix(hg, hg.ntypes)
        etype_dict = get_etype_dict(hg)
        for etype in etypes:

            microf1s = []
            macrof1s = []
            source_ntype, target_ntype = etype_dict[etype]
            etype_reverse = connect_type_matrix[target_ntype][source_ntype]
            for _ in range(5):
                with torch.no_grad():
                    hg = flow.hg.clone()
                    new_hg = random_flip_hg_auto_reverse(hg, purb_rate, etype)
                    flow.model.eval()
                    h_dict = flow.model.input_feature()
                    feat = {k: e.to(flow.device) for k, e in h_dict.items()}
                    logits = flow.model.forward(new_hg, feat)
                    pred = logits[flow.category][flow.test_idx].argmax(dim=1).cpu()
                    label = flow.labels[flow.test_idx].cpu()
                    acc1 = sklearn.metrics.f1_score(label, pred, average="micro")
                    acc2 = sklearn.metrics.f1_score(label, pred, average="macro")
                    microf1s.append(acc1)
                    macrof1s.append(acc2)
            logger.info(
                f"seed {seed}: etype => {etype}, purb rate {purb_rate} ==> micro-f1 down {ori_mf1 - numpy.mean(microf1s)}"
            )
