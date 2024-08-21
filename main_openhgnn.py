import argparse
import warnings

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

    experiment.run()
