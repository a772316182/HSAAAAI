from . import BaseFlow, register_flow
from ..models import build_model
from ..tasks import build_task


@register_flow("demo")
class Demo(BaseFlow):
    """Demo flows."""

    def __init__(self, args):
        super(Demo, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device

        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)

    def preprocess(self):
        return

    def train(self):
        pass

    def _mini_train_step(
        self,
    ):
        pass

    def loss_calculation(self, positive_graph, negative_graph, embedding):
        pass

    def _full_train_setp(self):
        pass

    def _test_step(self, split=None, logits=None):
        pass
