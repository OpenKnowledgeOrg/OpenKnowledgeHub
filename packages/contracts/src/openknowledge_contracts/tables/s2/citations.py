"""``citations``: one record per citation, keyed by ``citationid``.

Delete records carry only ``citationid``, so it is the key, not the endpoint pair: a pair can
have several citations, and its endpoints can change. ``citedcorpusid`` is null when S2 could
not resolve the cited paper. ``contexts`` and ``intents`` often differ in length and are kept
as delivered, never zipped. An ``intents`` entry is null when S2 classified nothing. It is
stored as an empty list, so the load has to fail if S2 ever sends an empty one.

Within one release, null contexts may yield to a non-null contexts list only when every other
stored value agrees and all non-null contexts lists agree. Empty is a non-null list. The table's
version orders release, deletion, then context presence, so replay order cannot lose enrichment.
"""

import pyarrow as pa

from openknowledge_contracts.tables.contract import DatasetContract
from openknowledge_contracts.tables.fields import list_of, optional, required

CITATIONS = DatasetContract(
    name="citations",
    table="citations",
    key=("citationid",),
    source=pa.schema(
        [
            required("citationid", pa.uint64()),
            required("citingcorpusid", pa.uint64()),
            optional("citedcorpusid", pa.uint64()),
            required("isinfluential", pa.bool_()),
            optional("contexts", list_of(pa.string())),
            optional("intents", list_of(list_of(pa.string()), nullable_items=True)),
        ]
    ),
    null_as_empty=frozenset({"intents.item"}),
    enriched="contexts",
)
