import logging

import yaml

from ludwig.api import LudwigModel

config = yaml.safe_load("""
input_features:
    -
        name: utterance
        type: text
        encoder: 
            type: rnn
            cell_type: lstm
            bidirectional: true
            num_layers: 2
            reduce_output: null
        preprocessing:
            tokenizer: space

output_features:
    -
        name: intent
        type: category
        reduce_input: sum
        decoder:
            num_fc_layers: 1
            output_size: 64
    -
        name: slots
        type: sequence
        decoder:
            type: tagger

""")

# Define Ludwig model object that drive model training
model = LudwigModel(config=config, logging_level=logging.INFO)

# initiate model training
(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(dataset="data.tsv", experiment_name="simple_experiment", model_name="simple_model")
