from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry
from dataclasses import field
from marshmallow_dataclass import dataclass
from typing import List
from ludwig.constants import TIED, NAME
from ludwig.schema.utils import BaseMarshmallowConfig
from ludwig.schema.features import BaseInputFeatureConfig
from marshmallow import fields, ValidationError

input_type_registry = Registry()
input_config_registry = Registry()
output_type_registry = Registry()
output_config_registry = Registry()


def register_input_feature(name: str):
    def wrap(cls):
        input_type_registry[name] = cls
        return cls

    return wrap


def register_output_feature(name: str):
    def wrap(cls):
        output_type_registry[name] = cls
        return cls

    return wrap

@dataclass
class InputFeaturesList(BaseMarshmallowConfig):
    """AudioFeatureInputFeature is a dataclass that configures the parameters used for an audio input feature."""

    input_features: List[BaseInputFeatureConfig] = schema_utils.InputFeatureList()

    # @validates_schema(pass_original=True)
    # def validate_tied():
    #     pass


def InputFeatureListDataclassField(features_list: list = []):
    class InputFeaturesListMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, dict):
                feature_name = value[NAME]
                tied = value[TIED]
                if isinstance(tied, str) and str == feature_name:
                    raise ValidationError("You are really stupid")
            raise ValidationError("Field should be dict")

        @staticmethod
        def _jsonschema_type_mapping():
        # def get_input_feature_jsonschema():
            input_feature_types = sorted(list(input_type_registry.keys()))
            return {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": input_feature_types},
                        "column": {"type": "string"},
                    },
                    "additionalProperties": True,
                    "allOf": get_input_feature_conds(),
                    "required": ["name", "type"],
                },
            }
    
    return field(
        metadata={
            "marshmallow_field": InputFeaturesListMarshmallowField(
                allow_none=False,
                dump_default=features_list,
                load_default=features_list,
            )
        },
        default_factory=lambda: features_list,
    )


def get_input_feature_jsonschema():
    """This function returns a JSON schema structured to only requires a `type` key and then conditionally applies
    a corresponding input feature's field constraints.

    Returns: JSON Schema
    """
    input_feature_types = sorted(list(input_type_registry.keys()))
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": input_feature_types},
                "column": {"type": "string"},
            },
            "additionalProperties": True,
            "allOf": get_input_feature_conds(),
            "required": ["name", "type"],
        },
    }


def get_input_feature_conds():
    """This function returns a list of if-then JSON clauses for each input feature type along with their properties
    and constraints.

    Returns: List of JSON clauses
    """
    input_feature_types = sorted(list(input_type_registry.keys()))
    conds = []
    for feature_type in input_feature_types:
        feature_cls = input_type_registry[feature_type]
        schema_cls = feature_cls.get_schema_cls()
        feature_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        feature_props = feature_schema["properties"]
        feature_cond = schema_utils.create_cond({"type": feature_type}, feature_props)
        conds.append(feature_cond)
    return conds


def get_output_feature_jsonschema():
    """This function returns a JSON schema structured to only requires a `type` key and then conditionally applies
    a corresponding output feature's field constraints.

    Returns: JSON Schema
    """
    output_feature_types = sorted(list(output_type_registry.keys()))
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": output_feature_types},
                "column": {"type": "string"},
            },
            "additionalProperties": True,
            "allOf": get_output_feature_conds(),
            "required": ["name", "type"],
        },
    }


def get_output_feature_conds():
    """This function returns a list of if-then JSON clauses for each output feature type along with their
    properties and constraints.

    Returns: List of JSON clauses
    """
    output_feature_types = sorted(list(output_type_registry.keys()))
    conds = []
    for feature_type in output_feature_types:
        feature_cls = output_type_registry[feature_type]
        schema_cls = feature_cls.get_schema_cls()
        feature_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        feature_props = feature_schema["properties"]
        feature_cond = schema_utils.create_cond({"type": feature_type}, feature_props)
        conds.append(feature_cond)
    return conds
