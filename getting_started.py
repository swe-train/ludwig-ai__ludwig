import logging

import pandas as pd
import yaml

from ludwig.api import LudwigModel

df = pd.read_csv("rotten_tomatoes.csv")

config = yaml.safe_load(
    """
input_features:
    # - name: genres
    #   type: set
    #   preprocessing:
    #       tokenizer: comma
    - name: content_rating
      type: category
    - name: top_critic
      type: binary
    - name: runtime
      type: number
    # - name: review_content
    #   type: text
    #   encoder: embed
output_features:
    - name: recommended
      type: binary
model_type: gbm
backend:
    type: ray
"""
)

model = LudwigModel(config=config, logging_level=logging.INFO)
results = model.train(dataset=df)

# model = LudwigModel.load(
#     "results/api_experiment_run/model"
# )  ## results/experiment_run/model -> results/api_experiment_run/model

# model = LudwigModel.load("results/api_experiment_run_1/model")
# model = LudwigModel.load("results/api_experiment_run/model")

# predictions, _ = model.predict(dataset="rotten_tomatoes_test_2.csv")
# predictions, _ = model.predict(dataset=df)
# print(predictions)


# from ludwig.automl import create_auto_config

# df = pd.read_csv("rotten_tomatoes.csv")
# config = create_auto_config(df, "recommended", time_limit_s=3600, tune_for_memory=False)
# del config["hyperopt"]

# model = LudwigModel(config=config, logging_level=logging.INFO)

# results = model.train(dataset=df)

# from pprint import pprint

# pprint(config)
