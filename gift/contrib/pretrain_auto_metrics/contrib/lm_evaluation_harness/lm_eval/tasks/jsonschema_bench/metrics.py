import ipaddress
import json
import logging
import math
import uuid
from typing import Any, Dict, List, Union


# check if jsonschema is installed
try:
    import jsonschema
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError
except ImportError as e:
    raise ImportError(
        "jsonschema is not installed. Please install it using 'pip install jsonschema[format]'"
    ) from e

eval_logger = logging.getLogger(__name__)


def is_json_schema_valid(schema: dict):
    """
    Check if a JSON schema is valid.

    :param schema: A JSON schema.
    :return: True if the schema is valid, False otherwise.
    """
    try:
        # Check if the schema is valid
        jsonschema.Draft202012Validator.check_schema(schema)
        return True
    except jsonschema.SchemaError:
        return False


# Initialize the FormatChecker
format_checker = FormatChecker()


# Add custom format checkers
@format_checker.checks("ipv4")
def ipv4_check(value):
    ipaddress.IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    ipaddress.IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    uuid.UUID(value)


def schema_conform_with_format_checker(
    instance: Dict[str, Any], schema: Dict[str, Any]
) -> bool:
    """
    Validate a JSON instance against a schema with enhanced format checking.

    :param schema: The JSON schema to validate against.
    :param instance: The JSON instance to validate.
    :raises ValidationError: If the validation fails.
    """
    # first check if the schema is valid
    if not is_json_schema_valid(schema):
        raise ValidationError("The JSON schema is invalid.")
    validator = Draft202012Validator(schema, format_checker=format_checker)
    try:
        validator.validate(instance)
    except ValidationError as e:
        raise ValidationError(e.message)
    return True


def schema_compliance(references: list[str], predictions: list[str]) -> bool:
    assert len(references) == 1, (
        "We only have one reference for this task, which is the JSON schema."
    )
    assert len(predictions) == 1, (
        "schema_compliance expects a single prediction; use schema_compliance_pass_at_k for pass@k."
    )
    reference = references[0]
    prediction = predictions[0]  # Since predictions is a list of lists

    json_schema = json.loads(reference.strip())
    try:
        json_obj = json.loads(prediction.strip().strip("```").strip("json"))
    except json.JSONDecodeError:
        return False

    try:
        schema_conform = schema_conform_with_format_checker(json_obj, json_schema)
    except Exception as e:
        eval_logger.error(f"Error: {e}")
        return False

    return schema_conform


def json_validity(references: list[str], predictions: list[str]) -> bool:
    assert len(predictions) == 1, (
        "json_validity expects a single prediction; use json_validity_pass_at_k for pass@k."
    )
    prediction = predictions[0]  # Since predictions is a list of lists
    try:
        json.loads(prediction.strip().strip("```").strip("json").strip())
    except json.JSONDecodeError:
        return False
    return True


def _estimator_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _is_schema_compliant(reference: str, prediction: str) -> bool:
    json_schema = json.loads(reference.strip())
    try:
        json_obj = json.loads(prediction.strip().strip("```").strip("json"))
    except (ValueError, RecursionError):
        return False

    try:
        return schema_conform_with_format_checker(json_obj, json_schema)
    except Exception as e:
        eval_logger.error(f"Error: {e}")
        return False


def _is_json_valid(prediction: str) -> bool:
    try:
        json.loads(prediction.strip().strip("```").strip("json").strip())
    except (ValueError, RecursionError):
        return False
    return True


def schema_compliance_pass_at_k(
    references: Union[str, List[str]],
    predictions: Union[str, List[List[str]]],
    k: Union[int, List[int]] = None,
) -> dict:
    assert k is not None
    if isinstance(k, int):
        k = [k]
    if isinstance(predictions, str):
        predictions = [[predictions]]
    if not predictions or not predictions[0]:
        return {f"schema_compliance_pass@{ki}": 0.0 for ki in k}

    reference = references[0]
    outputs = predictions[0]

    n = len(outputs)
    c = sum(1 for prediction in outputs if _is_schema_compliant(reference, prediction))

    return {f"schema_compliance_pass@{ki}": _estimator_pass_at_k(n, c, ki) for ki in k}


def json_validity_pass_at_k(
    references: Union[str, List[str]],
    predictions: Union[str, List[List[str]]],
    k: Union[int, List[int]] = None,
) -> dict:
    assert k is not None
    if isinstance(k, int):
        k = [k]
    if isinstance(predictions, str):
        predictions = [[predictions]]
    if not predictions or not predictions[0]:
        return {f"json_validity_pass@{ki}": 0.0 for ki in k}

    outputs = predictions[0]

    n = len(outputs)
    c = sum(1 for prediction in outputs if _is_json_valid(prediction))

    return {f"json_validity_pass@{ki}": _estimator_pass_at_k(n, c, ki) for ki in k}
