from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Generator, TYPE_CHECKING

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm

from ludwig.api_annotations import DeveloperAPI
from ludwig.backend.base import Backend
from ludwig.backend.utils.storage import StorageManager
from ludwig.constants import MODEL_LLM
from ludwig.data.cache.manager import CacheManager
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.data.dataframe.pandas import PANDAS
from ludwig.data.dataset.base import DatasetManager
from ludwig.data.dataset.pandas import PandasDatasetManager
from ludwig.distributed import init_dist_strategy
from ludwig.distributed.base import DistributedStrategy
from ludwig.models.base import BaseModel
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.types import HyperoptConfigDict
from ludwig.utils.audio_utils import read_audio_from_path
from ludwig.utils.batch_size_tuner import BatchSizeEvaluator
from ludwig.utils.dataframe_utils import from_batches, to_batches
from ludwig.utils.fs_utils import get_bytes_obj_from_path
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.system_utils import Resources
from ludwig.utils.torch_utils import initialize_pytorch
from ludwig.utils.types import DataFrame, Series

if TYPE_CHECKING:
    from ludwig.trainers.base import BaseTrainer


# TODO: <Alex>ALEX</Alex>
@DeveloperAPI
class PredibaseBackend(Backend):
    BACKEND_TYPE = "predibase"

    _shared_instance: PredibaseBackend

    @classmethod
    def shared_instance(cls) -> PredibaseBackend:
        """Returns a shared singleton PredibaseBackend instance."""
        if not hasattr(cls, "_shared_instance"):
            cls._shared_instance = cls()
        return cls._shared_instance

    def __init__(self, **kwargs) -> None:
        super().__init__(dataset_manager=PandasDatasetManager(self), **kwargs)

    @staticmethod
    def initialize():
        init_dist_strategy("predibase")

    @staticmethod
    def initialize_pytorch(*args, **kwargs):
        # TODO: <Alex>ALEX</Alex>
        # initialize_pytorch(*args, **kwargs)
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        pass
        # TODO: <Alex>ALEX</Alex>

    def create_trainer(
        self,
        config: BaseTrainerConfig,
        model: BaseModel,
        **kwargs,
    ) -> BaseTrainer:  # type: ignore[override]
        # TODO: <Alex>ALEX</Alex>
        # from ludwig.trainers.registry import get_llm_trainers_registry, get_trainers_registry
        # TODO: <Alex>ALEX</Alex>
        from ludwig.trainers.registry import get_llm_trainers_registry

        trainer_cls: type
        if model.type() == MODEL_LLM:
            print(
                f"\n[ALEX_TEST] [PredibaseBackend.create_trainer()] CONFIG.TYPE:\n{config.type} ; TYPE: {str(type(config.type))}"
            )
            trainer_cls = get_from_registry(config.type, get_llm_trainers_registry())
            print(
                f"\n[ALEX_TEST] [PredibaseBackend.create_trainer()] TRAINER_CLS:\n{trainer_cls} ; TYPE: {str(type(trainer_cls))}"
            )
        else:
            raise ValueError(f"Only {MODEL_LLM} model type is currently supported.")

        return trainer_cls(config=config, model=model, **kwargs)

    # batch_transform, df_engine, get_available_resources, is_coordinator, max_concurrent_trials, num_nodes, num_training_workers, read_binary_files, supports_multiprocessing, tune_batch_size

    def sync_model(self, model):
        raise ValueError("Not used in LLM")

    def broadcast_return(self, fn):
        return fn()

    @staticmethod
    def is_coordinator() -> bool:
        return True

    @property
    def df_engine(self):
        return PANDAS

    @property
    def supports_multiprocessing(self):
        return True

    @staticmethod
    def read_binary_files(self, column: Series, map_fn: Callable | None = None) -> Series:
        raise ValueError("Not used in LLM")

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def num_training_workers(self) -> int:
        return 1

    def get_available_resources(self) -> Resources:
        raise ValueError("Not used in LLM")

    def max_concurrent_trials(self, hyperopt_config: HyperoptConfigDict) -> int | None:
        raise ValueError("Not used in LLM")

    @staticmethod
    def tune_batch_size(evaluator_cls: type[BatchSizeEvaluator], dataset_len: int) -> int:
        raise ValueError("Not used in LLM")

    @staticmethod
    def batch_transform(df: DataFrame, batch_size: int, transform_fn: Callable, name: str | None = None) -> DataFrame:
        raise ValueError("Not used in LLM")


# TODO: <Alex>ALEX</Alex>
