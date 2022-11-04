from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.combiner_metadata import COMBINER_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json
from ludwig.utils.registry import Registry

from dataclasses import field
from marshmallow_dataclass import dataclass
from ludwig.constants import TYPE
from ludwig.schema.utils import BaseMarshmallowConfig
from ludwig.schema.combiners.base import BaseCombinerConfig
from marshmallow import fields, ValidationError

combiner_registry = Registry()


def register_combiner(name: str):
    def wrap(cls):
        combiner_registry[name] = cls
        return cls

    return wrap


def CombinerOptionsDataclassField(default: dict = {"default": "concat"}):
    class CombinerOptionsMarshmallowField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            combiner = None
            if isinstance(value, dict):
                combiner_type = value[TYPE]
                if combiner_type == "comparator":
                    # Required params for this type:
                    entity_1 = set(value["entity_1"])
                    entity_2 = set(value["entity_2"])

                    entity_intersection = entity_1.intersection(entity_2)
                    if len(entity_intersection) > 0:
                        raise ValidationError("You are really really stupid")

                    try:
                        combiner_cls = combiner_registry[combiner_type]
                        return combiner_cls.Schema().load(value)

                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid combiner params: {value}, see `{combiner_type}` definition. Error: {error}"
                        )
            raise ValidationError("Field should be dict")

        @staticmethod
        def _jsonschema_type_mapping():
            combiner_types = sorted(list(combiner_registry.keys()))
            parameter_metadata = convert_metadata_to_json(COMBINER_METADATA[TYPE])
            return {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": combiner_types,
                        "default": "concat",
                        "title": "combiner_options",
                        "description": "Select the combiner type.",
                        "parameter_metadata": parameter_metadata,
                    },
                },
                "allOf": get_combiner_conds(),
                "required": ["type"],
            }

    try:
        combiner_type = default[TYPE]
        combiner_cls = combiner_registry[combiner_type]
        load_default = combiner_cls.Schema().load(default)
        dump_default = combiner_cls.Schema().dump(default)

    except (TypeError, ValidationError) as error:
        raise ValidationError(
            f"Invalid default combiner params: {default}, see `{combiner_type}` definition. Error: {error}"
        )

    # This field by design has no default dump/load - it is a required parameter:
    return field(
        metadata={
            "marshmallow_field": CombinerOptionsMarshmallowField(
                allow_none=False, load_default=load_default, dump_default=dump_default
            )
        },
    )


# def get_combiner_jsonschema():
#     """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
#     combiner's field constraints."""
#     combiner_types = sorted(list(combiner_registry.keys()))
#     parameter_metadata = convert_metadata_to_json(COMBINER_METADATA[TYPE])
#     return {
#         "type": "object",
#         "properties": {
#             "type": {
#                 "type": "string",
#                 "enum": combiner_types,
#                 "default": "concat",
#                 "title": "combiner_options",
#                 "description": "Select the combiner type.",
#                 "parameter_metadata": parameter_metadata,
#             },
#         },
#         "allOf": get_combiner_conds(),
#         "required": ["type"],
#     }


def get_combiner_conds():
    """Returns a list of if-then JSON clauses for each combiner type in `combiner_registry` and its properties'
    constraints."""
    combiner_types = sorted(list(combiner_registry.keys()))
    conds = []
    for combiner_type in combiner_types:
        combiner_cls = combiner_registry[combiner_type]
        schema_cls = combiner_cls.get_schema_cls()
        combiner_schema = schema_utils.unload_jsonschema_from_marshmallow_class(schema_cls)
        combiner_props = combiner_schema["properties"]
        schema_utils.remove_duplicate_fields(combiner_props)
        combiner_cond = schema_utils.create_cond({"type": combiner_type}, combiner_props)
        conds.append(combiner_cond)
    return conds


@dataclass
class CombinerOptions(BaseMarshmallowConfig):
    combiner: BaseCombinerConfig = CombinerOptionsDataclassField()

    # @validates_schema(pass_original=True)
    # def validate_tied():
    #     pass
