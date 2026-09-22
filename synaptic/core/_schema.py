"""Python annotation schemas and validated, fully defaulted call arguments."""

import copy
import inspect
from typing import Any, get_type_hints

from pydantic import ConfigDict, Field, TypeAdapter, create_model
from jsonschema import Draft202012Validator


def infer(function, descriptions):
    hints = get_type_hints(function)
    fields: dict[str, Any] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in ("self", "cls"):
            continue
        if parameter.kind in (
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
            parameter.POSITIONAL_ONLY,
        ):
            raise TypeError(
                "autotool requires named parameters, not variadic/positional-only arguments"
            )
        annotation = hints.get(name, Any)
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (
            annotation,
            Field(default=default, description=descriptions.get(name, "")),
        )
    try:
        model = create_model(
            function.__name__ + "Arguments", __config__=ConfigDict(extra="forbid"), **fields
        )
        schema = model.model_json_schema()
        output = TypeAdapter(hints["return"]) if "return" in hints else None
        if output:
            output.json_schema()
    except Exception as exc:
        raise TypeError(f"Unsupported annotation for tool {function.__name__}: {exc}") from exc
    return model, schema, output


def effective(function, defaults, arguments, schema, model=None):
    values = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            values[name] = copy.deepcopy(parameter.default)
    # JSON Schema defaults are annotations; apply top-level parameter defaults deliberately.
    for name, definition in schema.get("properties", {}).items():
        if "default" in definition:
            values.setdefault(name, copy.deepcopy(definition["default"]))
    values.update(copy.deepcopy(defaults))
    values.update(copy.deepcopy(dict(arguments)))
    if model:
        validated = model.model_validate(values)
        values = {name: getattr(validated, name) for name in model.model_fields}
        json_values = validated.model_dump(mode="json")
    else:
        json_values = copy.deepcopy(values)
    inspect.signature(function).bind(**values)
    Draft202012Validator(schema).validate(json_values)
    return values, json_values
