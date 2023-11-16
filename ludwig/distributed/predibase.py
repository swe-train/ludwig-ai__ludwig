from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, TYPE_CHECKING

import torch
from torch import nn
from torch.optim import Optimizer

from ludwig.distributed.base import DistributedStrategy
from ludwig.modules.optimization_modules import create_optimizer
from ludwig.utils.torch_utils import get_torch_device

if TYPE_CHECKING:
    from ray.train.backend import BackendConfig
    from ray.train.data_parallel_trainer import DataParallelTrainer

    from ludwig.models.base import BaseModel
    from ludwig.modules.lr_scheduler import LRScheduler
    from ludwig.schema.trainer import ECDTrainerConfig
    from ludwig.utils.checkpoint_utils import Checkpoint


class PredibaseStrategy(DistributedStrategy):
    def prepare(
        self,
        model: nn.Module,
        trainer_config: ECDTrainerConfig,
        base_learning_rate: float,
    ) -> tuple[nn.Module, Optimizer]:
        # TODO: <Alex>ALEX</Alex>
        # return model, create_optimizer(model, trainer_config.optimizer, base_learning_rate)
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def size(self) -> int:
        # TODO: <Alex>ALEX</Alex>
        # return 1
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def rank(self) -> int:
        # TODO: <Alex>ALEX</Alex>
        # return 0
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def local_size(self) -> int:
        # TODO: <Alex>ALEX</Alex>
        # return 0
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def local_rank(self) -> int:
        # TODO: <Alex>ALEX</Alex>
        # return 0
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def barrier(self):
        # TODO: <Alex>ALEX</Alex>
        # pass
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def allreduce(self, t: torch.Tensor) -> torch.Tensor:
        # TODO: <Alex>ALEX</Alex>
        # return t
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def broadcast(self, t: torch.Tensor) -> torch.Tensor:
        # TODO: <Alex>ALEX</Alex>
        # return t
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def sync_model(self, model: nn.Module):
        # TODO: <Alex>ALEX</Alex>
        # pass
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def sync_optimizer(self, optimizer: Optimizer):
        # TODO: <Alex>ALEX</Alex>
        # pass
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def broadcast_object(self, v: Any, name: str | None = None) -> Any:
        # TODO: <Alex>ALEX</Alex>
        # return v
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    def wait_optimizer_synced(self, optimizer: Optimizer):
        # TODO: <Alex>ALEX</Alex>
        # pass
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    @contextlib.contextmanager
    def prepare_model_update(self, model: nn.Module, should_step: bool):
        # TODO: <Alex>ALEX</Alex>
        # yield
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    @contextlib.contextmanager
    def prepare_optimizer_update(self, optimizer: Optimizer):
        # TODO: <Alex>ALEX</Alex>
        # yield
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    @classmethod
    def is_available(cls) -> bool:
        # While this strategy is always an option, it is not "distributed" which is the meaning of availability
        # in this context.
        # TODO: <Alex>ALEX</Alex>
        # return False
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    @classmethod
    def gather_all_tensors_fn(cls) -> Callable | None:
        # TODO: <Alex>ALEX</Alex>
        # return None
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    @classmethod
    def get_ray_trainer_backend(cls, **kwargs) -> Any | None:
        # TODO: <Alex>ALEX</Alex>
        # return None
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>

    @classmethod
    def get_trainer_cls(cls, backend_config: BackendConfig) -> tuple[type[DataParallelTrainer], dict[str, Any]]:
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Cannot construct a trainer from the Predibase strategy.")
        # TODO: <Alex>ALEX</Alex>

    def shutdown(self):
        # TODO: <Alex>ALEX</Alex>
        # pass
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        raise ValueError("Not used in Predibase backend")
        # TODO: <Alex>ALEX</Alex>
