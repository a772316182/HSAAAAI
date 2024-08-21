import argparse
import warnings

from attack.hsa.start_target import hsa_starter
from openhgnn.experiment import Experiment

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="HAN", type=str, help="name of models")
    parser.add_argument(
        "--task", "-t", default="node_classification", type=str, help="name of task"
    )
    parser.add_argument(
        "--dataset", "-d", default="imdb4GTN", type=str, help="name of datasets"
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

    # query budget = level_2_query_limit + level_3_query_limit
    hsa_starter(
        flow=flow,
        seed=0,
        purb_limit_per_target=1,
        # we don't use plain training anymore
        # note: set to -1 will cause error
        level_1_training_epoch=0,
        surrogate_lr_lv1=1e-2,
        weight_decay_lv1=1e-3,
        #
        level_2_training_batch_size=8,
        level_2_query_limit=50,
        level_2_training_epoch=8,
        surrogate_lr_lv2=1e-2,
        weight_decay_lv2=3e-5,
        #
        level_3_training_batch_size=8,
        level_3_query_limit=50,
        level_3_training_epoch=8,
        surrogate_lr_lv3=1e-3,
        weight_decay_lv3=5e-5,
        lv3_attack_rate=0.5,
    )
