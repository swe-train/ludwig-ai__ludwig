import contextlib
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

# TODO: <Alex>ALEX</Alex>
from predibase import PredibaseClient
from predibase.pql import get_session
from predibase.resource.model import Model as PredibaseCloudModel
from predibase.resource.model import ModelRepo as PredibaseModelRepo
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel

from ludwig.constants import IGNORE_INDEX_TOKEN_ID, LOGITS, MODEL_LLM, PREDICTIONS, TEXT
from ludwig.features.base_feature import ModuleWrapper, OutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.features.text_feature import TextOutputFeature
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.schema.features.base import BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.model_types.llm import LLMModelConfig
from ludwig.utils.augmentation_utils import AugmentationPipelines
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.error_handling_utils import default_retry
from ludwig.utils.llm_utils import (
    add_left_padding,
    generate_merged_ids,
    get_context_len,
    get_realigned_target_and_prediction_tensors_for_inference,
    pad_target_tensor_for_fine_tuning,
    remove_left_padding,
    set_pad_token,
)
from ludwig.utils.logging_utils import log_once
from ludwig.utils.output_feature_utils import set_output_feature_tensor
from ludwig.utils.torch_utils import reg_loss

# TODO: <Alex>ALEX</Alex>

logger = logging.getLogger(__name__)


# TODO: <Alex>ALEX</Alex>
# TODO: <Alex>ALEX</Alex>


class PredibaseModel:
    @staticmethod
    def type() -> str:
        return MODEL_LLM

    def __init__(
        self,
        config_obj: LLMModelConfig = None,
        predibase_config: dict = None,
        **_kwargs,
    ):
        self.config_obj = config_obj

        self.predibase_config = predibase_config

        # Get the api token, and set the serving endpoint for staging
        token = os.getenv(
            "PREDIBASE_API_TOKEN",
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyVVVJRCI6IjRjNjQwYjM5LTE4M2UtNDdlZS04NjEwLTQ3YWU0MjJhNGFjMyIsInRlbmFudFVVSUQiOiIiLCJlbmdpbmVVVUlEIjoiIiwic2NvcGUiOiJVU0VSIiwiZXhwIjoxNzMxNjE0MzM2LCJpYXQiOjE2OTk5OTE5MzYsImlzcyI6InByZWRpYmFzZSIsIm5iZiI6MTY5OTk5MTkzNiwic3ViIjoiN2M3ZWZhIn0.pXfd6GdpVkZ2Mzk9oKdL9DhueIBTJry9lQZqeARLsPKOzfpQggNFYkbeR9DQibqsplPLyumUZLZzlR9iTDPe9iLKrZpYllC8sbF3RkhJHAWOz9MPU82crfRCtR75DMBZMHe78zt_19KkHWDyyU9v0j6TkR6sb6XA_pHbU1gyE6apTtvrvYxwSSpxaDEDIxHFNmM-2lgpHiyt92O46-PfGdnffX3xOWW-NMdZDnzxFyKVdeJGPK4iwHq9jfVRs6eHlO_YA5NUzBuyXZeByozoW0KsVD12uOKQr5e6lJz5Ogp8w2nVPpxPv5Zw5n4Yp4ltq-t7mGW5Ueiux08f8xb1eA",
        )
        session = get_session(token=token, gateway="https://api.staging.predibase.com/v1")

        # Get current user to output session tenant
        client = PredibaseClient(session)
        print(
            f"\n[ALEX_TEST] [PredibaseModel.__INIT__()] CURRENT_ENGINE_NAME:\n{client.get_current_engine().name} ; TYPE: {str(type(client.get_current_engine().name))}"
        )

        self._client = client

        self._model_repo = None
        self._model = None
        # TODO: <Alex>ALEX</Alex>
        self._adapter_deployment = None
        # TODO: <Alex>ALEX</Alex>

    @property
    def model_repo(self) -> PredibaseModelRepo:
        return self._model_repo

    @property
    def model(self) -> PredibaseCloudModel:
        return self._model

    def _create_dataset(
        self, dataset: Optional[Union[str, dict, pd.DataFrame]] = None, experiment_name="my_experiment"
    ):
        try:
            ds = self._client.get_dataset(experiment_name, "file_uploads")
            print(
                f"\n[ALEX_TEST] [PredibaseModel._create_dataset()] GOT_EXISTING_DATASET:\n{ds.name} ; TYPE: {str(type(ds))}"
            )
        except:
            if isinstance(dataset, str):
                ds = self._client.upload_dataset(dataset, name=experiment_name)
            elif isinstance(dataset, pd.DataFrame):
                ds = self._client.create_dataset_from_df(dataset, name=experiment_name)
            else:
                raise ValueError(f'Unsupported dataset type "{dataset}" encounted.')
            print(
                f"\n[ALEX_TEST] [PredibaseModel._create_dataset()] CREATED_NEW_DATASET:\n{ds.name} ; TYPE: {str(type(ds))}"
            )

        return ds

    def load_model_repo(
        self,
        model_name: str = "run",
    ) -> None:
        model_repo = self._client.get_model_repo(model_name)
        # assert model_repo is not None
        self._model_repo = model_repo
        print(
            f"\n[ALEX_TEST] [PredibaseModel.load_model_repo()] LOADED_MODEL_REPO:\n{self._model_repo} ; TYPE: {str(type(self._model_repo))}"
        )

    def load_model(
        self,
        model_name: str = "run",
        load_model_repo: bool = False,
    ) -> None:
        model = self._client.get_model(model_name)
        self._model = model
        print(
            f"\n[ALEX_TEST] [PredibaseModel.load_model()] LOADED_MODEL:\n{self._model} ; TYPE: {str(type(self._model))}"
        )

        if load_model_repo:
            self.load_model_repo(model_name=model_name)
        print(
            f"\n[ALEX_TEST] [PredibaseModel.load_model()] LOADED_MODEL_REPO:\n{self.model_repo} ; TYPE: {str(type(self.model_repo))}"
        )
        # TODO: <Alex>ALEX</Alex>
        ft_model = self._client.get_model(model_name)
        base_deployment = self._client.LLM("pb://deployments/llama-2-7b")
        adapter_deployment = base_deployment.with_adapter(ft_model)
        self._adapter_deployment = adapter_deployment
        # TODO: <Alex>ALEX</Alex>

    def train(
        self,
        dataset: Optional[Union[str, dict, pd.DataFrame]] = None,
        experiment_name: str = "api_experiment",
        model_name: str = "run",
        **kwargs,
    ) -> None:
        print(f"\n[ALEX_TEST] [PredibaseModel.train()] DATASET:\n{dataset} ; TYPE: {str(type(dataset))}")
        ds = self._create_dataset(dataset=dataset, experiment_name=experiment_name)
        # TODO: <Alex>ALEX</Alex>
        # try:
        #     model = self._client.get_model(model_name)
        #     print("Got latest model", model.name)
        # except:
        #     print("Creating model")
        #     model = self._client.create_model(repository_name=model_name, dataset=ds, config=self.predibase_config)
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        print(f"\n[ALEX_TEST] [PredibaseModel.train()] CREATING_MODEL:\n{model_name} ; TYPE: {str(type(model_name))}")
        model = self._client.create_model(repository_name=model_name, dataset=ds, config=self.predibase_config)
        # TODO: <Alex>ALEX</Alex>

        self._model = model

    # TODO: <Alex>ALEX</Alex>
    # def predict(
    #     self,
    #     targets: str,
    #     dataset: Optional[Union[str, dict, pd.DataFrame]] = None,
    #     **kwargs,
    # ):
    #     eng = self._client.get_engine("arnav_gpu_small")
    #     print(f'\n[ALEX_TEST] [PredibaseModel.predict()] ENGINE:\n{eng} ; TYPE: {str(type(eng))}')
    #     results = self._model.predict(targets=targets, source=dataset, engine=eng)
    #     print(f'\n[ALEX_TEST] [PredibaseModel.predict()] RESULTS:\n{results} ; TYPE: {str(type(results))}')
    #     return results
    # TODO: <Alex>ALEX</Alex>
    # TODO: <Alex>ALEX</Alex>
    def predict(
        self,
        dataset: Optional[Union[str, dict, pd.DataFrame]] = None,
        **kwargs,
    ):
        # print(f'\n[ALEX_TEST] [PredibaseModel.predict()] SELF.CONFIG_OBJ:\n{self.config_obj} ; TYPE: {str(type(self.config_obj))}')
        input_column_name: str = self.config_obj.input_features[0].column
        input = dataset[input_column_name].tolist()[0]
        result = self._adapter_deployment.prompt(
            {
                input_column_name: input,
            },
            max_new_tokens=256,
        )
        print(f"\n[ALEX_TEST] [PredibaseModel.predict()] RESULT:\n{result} ; TYPE: {str(type(result))}")
        return result

    # TODO: <Alex>ALEX</Alex>
