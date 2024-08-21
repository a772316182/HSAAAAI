import argparse
import warnings

import dgl
import numpy
from loguru import logger

from attack.attack_utils.attack_evaluator import eval_evasion
from openhgnn import HANNodeClassification
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
    flow: HANNodeClassification = flow
    flow._full_test_step(modes=["test"])
    hg = flow.hg.clone()
    micro_f1s = []
    for _ in range(10):
        # add 50% edges to each types
        trans = dgl.AddEdge(ratio=0.5)
        new_hg = trans(hg)

        micro_f1, macro_f1, loss, logits = eval_evasion(new_hg, flow)
        micro_f1s.append(micro_f1)
    logger.info("ADD 50% edges")
    logger.info(
        f'MICRO F1 DOWN: {result["metric"]["test"]["Micro_f1"] - numpy.average(micro_f1s, axis=0)}'
    )
