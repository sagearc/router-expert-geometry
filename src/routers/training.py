"""Trainer glue for the K-means router's learning-rate-coupled updates."""

from dataclasses import dataclass

from olmo_core.train.callbacks import Callback
from olmo_core.train.train_module import TransformerTrainModule
from torch import nn

from src.routers.centroid import PaperCentroidRouter


def set_router_learning_rate(model: nn.Module, learning_rate: float) -> int:
    """Give every K-means router the current learning rate; return how many there are."""
    routers = [module for module in model.modules() if isinstance(module, PaperCentroidRouter)]
    for router in routers:
        router.learning_rate = learning_rate
    return len(routers)


@dataclass
class RouterLearningRateCallback(Callback):
    """Read the optimizer's learning rate before each batch for the K-means routers.

    OLMo-core's scheduler sets the rate inside ``optim_step``, after the batch's
    ``post_batch`` centroid and bias updates, so those updates use the rate of the
    previous optimizer step. The original runs read the same value.
    """

    def pre_train(self) -> None:
        self._update()

    def pre_step(self, batch: dict[str, object]) -> None:
        del batch
        self._update()

    def _update(self) -> None:
        module = self.trainer.train_module
        if not isinstance(module, TransformerTrainModule):
            raise TypeError('RouterLearningRateCallback requires TransformerTrainModule')
        if not set_router_learning_rate(module.model, float(module.optim.param_groups[0]['lr'])):
            raise ValueError('RouterLearningRateCallback requires a K-means router')
