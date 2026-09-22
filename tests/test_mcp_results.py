import json
from types import SimpleNamespace

import pytest

from synaptic.mcp import extract_text, legacy_result, normalize_result


@pytest.mark.parametrize("value", [None, False, 0, [], {}, {"nested": [1, None, True, "é"]}])
def test_structured_values_round_trip(value):
    result = normalize_result({"structuredContent": value, "content": []})
    assert json.loads(json.dumps(result))["structured_content"] == value
    assert result["has_structured_content"]


def test_all_modalities_and_extensions_preserved_without_sdk_objects():
    mcp = pytest.importorskip("mcp.types")
    blocks = [
        {"type": "text", "text": "caption", "annotations": {"audience": ["user"]}},
        {"type": "image", "data": "AA==", "mimeType": "image/png", "_meta": {"source": "camera"}},
        {"type": "audio", "data": "AA==", "mimeType": "audio/wav"},
        {"type": "resource", "resource": {"uri": "file:///fixture", "text": "embedded"}},
        {"type": "resource_link", "uri": "file:///asset", "name": "asset"},
    ]
    sdk = mcp.CallToolResult.model_validate(
        {
            "content": blocks,
            "structuredContent": {"x": 1},
            "_meta": {"trace": "abc"},
            "vendor": {"x": False},
        }
    )
    result = normalize_result(sdk, provenance={"server": "fixture", "tool": "mixed"})
    assert result["content"] == blocks
    assert result["extensions"] == {"vendor": {"x": False}}
    assert result["metadata"] == {"trace": "abc"}
    assert extract_text(result) == "caption"
    assert legacy_result(result) == {"x": 1}
    assert json.loads(json.dumps(result)) == result
    result["content"][0]["text"] = "edited"
    assert sdk.content[0].text == "caption"


def test_tool_failure_keeps_details_and_legacy_convenience():
    result = normalize_result(
        {
            "isError": True,
            "structuredContent": {"reason": "missing"},
            "content": [{"type": "text", "text": "not found"}],
        }
    )
    assert result["is_error"]
    assert result["structured_content"] == {"reason": "missing"}
    assert legacy_result(result) == {"error": "not found"}


@pytest.mark.parametrize("value", [object(), {1: "bad key"}, float("nan")])
def test_opaque_or_non_json_data_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        normalize_result(SimpleNamespace(structuredContent=value, content=[]))
