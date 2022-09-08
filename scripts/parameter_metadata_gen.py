# flake8: noqa
# fmt: off
import collections
import os
from ast import literal_eval
from dataclasses import dataclass
from enum import Enum
from pprint import pformat
from typing import Any, List, Union
from ludwig.schema.metadata.parameter_metadata import ParameterMetadata

import pandas as pd


excel = pd.read_excel("sprint5.xlsx")
excel.fillna("None", inplace=True)
mask = excel["class"] == "#/definitions/TrainerConfig"
trainer_excel = excel[mask].iloc[:, 3:-2]
trainer_excel["allow_none"] = trainer_excel["allow_none"].astype(bool)

trainer_dict = trainer_excel.set_index("parameter_name").T.to_dict()


def safe_eval(expr):
    try:
        return literal_eval(expr)
    except Exception:
        return expr


# Metadata dicts
# print(__file__)
# print(Path(__file__).parent)
# print(os.path.dirname(__file__))
# print(os.path.dirname(os.path.dirname(__file__)))

all_metadata_dict = {}

for param in trainer_dict:
    # print(param)
    # Clear NaNs:
    metadata_dict = collections.defaultdict(str)
    all_metadata_dict[param] = metadata_dict
    for k, v in trainer_dict[param].items():
        # Fields that are captured by the Marshmallow field definition.
        if k == "default_value":
            continue
        if k == "type_annotation":
            continue
        if k == "allow_none":
            continue
        if k == "schema":
            continue
        if k == "description":
            continue

        if k == "expected_impact":
            metadata_dict[k] = v.upper()
        if v == "None":
            continue
        else:
            value = safe_eval(v)
            if isinstance(value, str):
                metadata_dict[k] = value.strip()
            else:
                metadata_dict[k] = value

parameter_metadata_objects = {}
for parameter_name, metadata_dict in all_metadata_dict.items():
    parameter_metadata = ParameterMetadata(**metadata_dict)
    parameter_metadata_objects[parameter_name] = parameter_metadata


path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ludwig/schema/metadata/trainer_combined.py"
)
with open(path, "w") as f:
    print(pformat(parameter_metadata_objects["batch_size"], width=120))
    f.write(pformat(parameter_metadata_objects, width=120))
