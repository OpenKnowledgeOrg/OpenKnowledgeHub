"""``okp search index`` against ClickHouse and OpenSearch in Docker, which CI must have."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
from click.testing import CliRunner

from openknowledge_pipelines import cli
from tests import test_load_e2e as load_e2e
from tests.test_load_e2e import BUCKET, FIXTURE, archive, archive_diff, ran

# The same compose stack, started once more for this module, and the same skip without Docker.
ports = load_e2e.ports
pytestmark = load_e2e.pytestmark

PAPERS = FIXTURE.with_name("papers.jsonl")
ABSTRACTS = FIXTURE.with_name("abstracts.jsonl")
INDEX = "papers"
BASE, NEXT = "2026-10-13", "2026-10-20"


def okp_search(ports: dict[str, str], *args: str) -> tuple[int, str]:
    env = {
        "OKP_CLICKHOUSE_URL": f"http://127.0.0.1:{ports['CLICKHOUSE_PORT']}",
        "CLICKHOUSE_PASSWORD": "openknowledge",
        "OKP_CLICKHOUSE_RAW_BUCKET": BUCKET,
        "OKP_OPENSEARCH_URL": f"http://127.0.0.1:{ports['OPENSEARCH_PORT']}",
        "OPENSEARCH_PASSWORD": "unused-without-the-security-plugin",
        "OKP_INDEX_SHARDS": "1",
    }
    result = CliRunner().invoke(cli.main, ["search", "index", "--index", INDEX, *args], env=env)
    return result.exit_code, result.output


def indexed(ports: dict[str, str], *args: str) -> dict[str, Any]:
    code, output = okp_search(ports, *args)
    assert code == 0, output
    return json.loads(output.splitlines()[-1])


def document(ports: dict[str, str], corpusid: str) -> dict[str, Any] | None:
    """The stored document with its ``_version``, or None once the index no longer has it."""
    response = httpx.get(f"http://127.0.0.1:{ports['OPENSEARCH_PORT']}/{INDEX}/_doc/{corpusid}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def search(ports: dict[str, str], query: dict[str, Any]) -> list[dict[str, Any]]:
    response = httpx.post(
        f"http://127.0.0.1:{ports['OPENSEARCH_PORT']}/{INDEX}/_search", json={"query": query}
    )
    response.raise_for_status()
    return response.json()["hits"]["hits"]


def version_of(release: str) -> int:
    """The table's ``version`` of a live row loaded at ``release``."""
    return (date.fromisoformat(release) - date(1970, 1, 1)).days * 2


def test_a_base_pass_then_a_diff_pass_keep_the_index_current(ports: dict[str, str]) -> None:
    papers = PAPERS.read_text().splitlines()
    abstracts = ABSTRACTS.read_text().splitlines()
    archive(ports, BASE, papers, "papers")
    archive(ports, BASE, abstracts, "abstracts")
    ran(ports, "load", "--release", BASE, "--dataset", "papers")
    ran(ports, "load", "--release", BASE, "--dataset", "abstracts")

    result = indexed(ports, "--release", BASE)
    assert (result["created"], result["written"], result["deleted"]) == (True, 5, 0)
    assert (result["requests"], result["documents"]) == (1, 5)
    hits = search(ports, {"match_phrase": {"title": "A Synthetic Paper"}})
    assert [hit["_source"]["corpusid"] for hit in hits] == ["101"]
    expected = {
        str(json.loads(line)["corpusid"]): json.loads(line)["abstract"] for line in abstracts
    }
    for line in papers:
        corpusid = str(json.loads(line)["corpusid"])
        found = document(ports, corpusid)
        assert found is not None, corpusid
        assert found["_source"].get("abstract") == expected.get(corpusid), corpusid
        assert (found["_version"], found["_source"]["contract_version"]) == (version_of(BASE), 1)

    retitled = {**json.loads(papers[4]), "title": "A Retitled Paper"}
    assert retitled["corpusid"] == 104
    archive_diff(ports, BASE, NEXT, [json.dumps(retitled)], ['{"corpusid": 103}'], dataset="papers")
    ran(ports, "load-diff", "--from-release", BASE, "--release", NEXT, "--dataset", "papers")
    for _ in range(2):  # the second pass repeats every write, which external_gte accepts
        result = indexed(ports, "--release", NEXT, "--from-release", BASE)
        assert (result["created"], result["written"], result["deleted"]) == (False, 1, 1)
        assert result["documents"] == 4
    assert document(ports, "103") is None
    found = document(ports, "104")
    assert found is not None
    assert (found["_source"]["title"], found["_version"]) == ("A Retitled Paper", version_of(NEXT))
    assert all(document(ports, key) is not None for key in ("101", "102", "18446744073709551615"))
