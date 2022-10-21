from copy import deepcopy
from typing import Dict

import pytest
import torch

from ludwig.features.number_feature import NumberInputFeature
from ludwig.utils.torch_utils import get_torch_device

BATCH_SIZE = 2
DEVICE = get_torch_device()


@pytest.fixture(scope="module")
def number_config():
    return {"name": "number_column_name", "type": "number"}


def test_number_input_feature(
    number_config: Dict,
) -> None:
    # setup image input feature definition
    number_def = deepcopy(number_config)

    # pickup any other missing parameters
    NumberInputFeature.populate_defaults(number_def)

    # ensure no exceptions raised during build
    input_feature_obj = NumberInputFeature(number_def).to(DEVICE)

    # check one forward pass through input feature
    input_tensor = input_feature_obj.create_sample_input(batch_size=BATCH_SIZE)
    assert input_tensor.shape == torch.Size((BATCH_SIZE, 1))
    assert input_tensor.dtype == torch.float32

    encoder_output = input_feature_obj(input_tensor)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)
