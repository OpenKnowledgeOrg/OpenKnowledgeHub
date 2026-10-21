from typing import Any

from openknowledge_contracts.search import PAPERS, SearchField, index_body


def mapped(fields: tuple[SearchField, ...], properties: dict[str, Any]) -> None:
    assert set(properties) == {field.name for field in fields}
    for field in fields:
        if field.type == "object":
            assert properties[field.name]["type"] == "object"
            mapped(field.fields, properties[field.name]["properties"])
        else:
            assert properties[field.name] == {"type": field.type}


def test_every_field_is_mapped_and_the_contract_is_at_version_1() -> None:
    assert PAPERS.version == 1
    body = index_body(PAPERS, shards=3)
    properties = body.mappings["properties"]
    mapped(PAPERS.fields, properties)
    assert (body.mappings["dynamic"], body.mappings["_source"]) == (False, {"enabled": True})
    assert properties["externalids"]["dynamic"] is False
    assert properties[PAPERS.id_field] == {"type": "keyword"}
    assert properties["contract_version"] == {"type": "integer"}
    assert body.settings == {
        "index": {"number_of_shards": 3, "number_of_replicas": 0, "codec": "zstd"}
    }
