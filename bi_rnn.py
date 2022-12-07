import logging

import yaml

from ludwig.api import LudwigModel

config = yaml.safe_load(
    """
input_features:
  - name: content
    type: text
    level: word
    encoder: parallel_cnn
  - name: internal_communication
    type: binary
  - name: external_communication
    type: binary
  - name: time_of_day
    type: numerical
  - name: n_recipients
    type: numerical
  - name: n_cc
    type: numerical
  - name: sender_address
    type: text
    level: char
    encoder: rnn
#    cell_type: lstm
    bidirectional: true
  - name: message_id
    type: text
    level: char
    encoder: rnn
output_features:
  - name: spam
    type: binary
training:
  decay: False
  learning_rate: 0.001
  validation_field: spam
  validation_metric: accuracy
"""
)

model = LudwigModel(config=config, logging_level=logging.INFO)
results = model.train(dataset="spam_assassin.parquet")
