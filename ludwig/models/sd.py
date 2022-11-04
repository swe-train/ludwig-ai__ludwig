import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from diffusers import DiffusionPipeline, StableDiffusionPipeline

from ludwig.constants import IMAGE, MODEL_STABLE_DIFFUSION, TEXT, TYPE
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.schema.model_config import ModelConfig
from ludwig.utils.torch_utils import get_torch_device
from ludwig.utils.types import TorchDevice


class StableDiffusion(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_STABLE_DIFFUSION

    def __init__(
        self,
        config_obj: ModelConfig,
        random_seed: int = None,
        **_kwargs,
    ):
        self.config_obj = config_obj
        self._random_seed = random_seed

        super().__init__(random_seed=self._random_seed)

        input_features = self.config_obj.input_features.to_list()
        if len(input_features) > 1 or input_features[0][TYPE] != TEXT:
            raise ValueError(
                f"StableDiffusion only supports one input text feature, you specified: {input_features}"
            )
        # output_features = self.config_obj.output_features.to_list()
        # if len(output_features) > 1 or output_features[0][TYPE] != IMAGE:
        #     raise ValueError(
        #         f"StableDiffusion only supports one output image feature, you specified: {output_features}"
        #     )

        # ================ Inputs ================
        try:
            self.input_features.update(self.build_inputs(input_feature_configs=self.config_obj.input_features))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # # ================ Outputs ================
        # self.output_features.update(
        #     self.build_outputs(output_feature_configs=self.config_obj.output_features, input_size=self.input_shape[-1])
        # )

        self.diffuser_pipeline: DiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            # TODO: add to config
            # self.config_obj.input_features[0].pretrained_model_name_or_path
            "/home/ray/stable-diffusion-v1-5"
        )
        self.diffuser_pipeline = self.diffuser_pipeline.to("cuda")

        self.num_inference_steps = 2


    def forward(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        mask=None,
    ) -> Dict[str, torch.Tensor]:
        assert list(inputs.keys()) == self.input_features.keys()
        # Convert dict to list of str
        prompts = next(iter(inputs.values()))
        output_predicted_images = self.diffuser_pipeline(prompts, num_inference_steps=self.num_inference_steps).images
        return output_predicted_images

    def save(self, save_path):
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        self.diffuser_pipeline.save(weights_save_path)

    def load(self, save_path):
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        self.diffuser_pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
            weights_save_path
        )

        device = torch.device(get_torch_device())
        self.diffuser_pipeline.to(device)

    def to_torchscript(self, device: Optional[TorchDevice] = None):
        """Converts the ECD model as a TorchScript model."""

        # Disable gradient calculation for hummingbird Parameter nodes.
        self.diffuser_pipeline.requires_grad_(False)

        return super().to_torchscript(device)

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return self.config_obj.input_features.to_list(), self.config_obj.output_features.to_list(), self._random_seed
