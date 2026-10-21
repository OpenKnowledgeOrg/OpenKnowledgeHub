"""Search contracts: the fields of one search document and the index that holds them."""

from openknowledge_contracts.search.contract import (
    CONTRACT_VERSION,
    IndexBody,
    SearchContract,
    SearchField,
    index_body,
)
from openknowledge_contracts.search.papers import PAPERS

__all__ = ["CONTRACT_VERSION", "PAPERS", "IndexBody", "SearchContract", "SearchField", "index_body"]
