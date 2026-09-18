"""Validate identity independently of result payloads before changing any state."""

import copy
from typing import Any

ResultInput = list[Any] | dict[str, Any]


def call_ids(memory):
    ids = [call.id for call in memory.tool_calls]
    if any(not isinstance(key, str) or not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Tool call IDs must be nonempty and unique")
    return ids


def indexed_results(memory, results, *, complete=True):
    ids = call_ids(memory)
    if isinstance(results, dict):
        indexed = dict(results)
        for key, result in indexed.items():
            if isinstance(result, dict) and "call_id" in result and result["call_id"] != key:
                raise ValueError("Mapping key contradicts embedded call_id")
    elif isinstance(results, list):
        tagged = [isinstance(item, dict) and "call_id" in item for item in results]
        if any(tagged):
            if not all(tagged):
                raise ValueError("Cannot mix tagged and positional results")
            indexed = {}
            for result in results:
                key = result["call_id"]
                if not isinstance(key, str) or not key or key in indexed:
                    raise ValueError("Invalid or duplicate result call_id")
                indexed[key] = result
        else:
            if len(results) != len(ids):
                if not complete and not results:
                    return {}
                raise ValueError("Every positional result requires exactly one tool call")
            indexed = dict(zip(ids, results))
    else:
        raise ValueError("Results must be a mapping or list")
    if any(not isinstance(key, str) or key not in ids for key in indexed):
        raise ValueError("Unexpected result call_id")
    if complete and set(indexed) != set(ids):
        raise ValueError("External results must match the complete set of call IDs")
    return indexed


def correlate(memory, results):
    indexed = indexed_results(memory, results)
    return [copy.deepcopy(indexed[key]) for key in call_ids(memory)]
