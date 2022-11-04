#!/usr/bin/env python

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/latest/examples/titanic/).

# Import required libraries
import logging
import os
import shutil

import pandas as pd

from ludwig.api import LudwigModel

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

df = pd.DataFrame({
    'text': ['this is a test', 'this is another test'] * 10,
    'image': ['None', 'None'] * 10,
    'split': [0] * 10 + [1] * 7 + [2] * 3
})

# Define Ludwig model object that drive model training
model = LudwigModel(config="./config.yaml", logging_level=logging.INFO, backend="local")
model.model = model.create_model(model.config_obj)

# batch prediction
model.training_set_metadata = {}
model.predict(df, skip_save_predictions=False)
