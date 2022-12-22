import logging

from ludwig.api import LudwigModel

config = {
    "input_features": [
        {
            "name": "source_code",
            "type": "text",
            "encoder": {
                "type": "auto_transformer",
                "pretrained_model_name_or_path": "facebook/bart-base",
            }
        },
    ],
    "output_features": [
        {
            "name": "test_code",
            "type": "text",
        }
    ],
}

# Define Ludwig model object that drive model training
model = LudwigModel(config=config, logging_level=logging.INFO)

# initiate model training
(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(dataset="data.csv", experiment_name="simple_experiment", model_name="simple_model")
