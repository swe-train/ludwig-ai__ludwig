from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from ludwig.constants import IMAGE, TEXT

from ludwig.features.base_feature import ModuleWrapper


if TYPE_CHECKING:
    from ludwig.features.base_feature import InputFeature
    from ludwig.models.llm import LLM


class Projection(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class TextProjection(Projection):
    def __init__(self, model: "LLM", input_feature: "InputFeature"):
        super().__init__()
        self.input_embeddings = ModuleWrapper(model.model.get_input_embeddings())

    def forward(self, input_ids: torch.Tensor):
        return self.input_embeddings.module(input_ids)


class ImageProjection(Projection):
    def __init__(self, model: "LLM", input_feature: "InputFeature"):
        super().__init__()
        print("input feature shape", input_feature.output_shape)
        self.projector = torch.nn.Linear(input_feature.output_shape[0], model.model.config.hidden_size)

    def forward(self, inputs: torch.Tensor):
        print("input feature", inputs.shape)
        t = self.projector(inputs)
        if len(t.shape) == 2:
            t = t.reshape(t.shape[0], 1, t.shape[1])
        print("projected feature", t.shape)
        return t


projection_registry = {
    TEXT: TextProjection,
    IMAGE: ImageProjection,
}


def create_projection(model: "LLM", input_feature: "InputFeature") -> torch.nn.Module:
    return projection_registry[input_feature.type()](model, input_feature)
