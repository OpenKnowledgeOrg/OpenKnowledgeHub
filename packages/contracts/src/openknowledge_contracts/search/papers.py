"""``papers``: one search document per live paper, with its abstract when the paper has one.

The document is built from the current rows of the ``papers`` and ``abstracts`` tables. Its
counts are ``long`` because the tables hold them as ``UInt32``, which an ``integer`` field, a
signed 32-bit number, cannot take at its top. ``externalids`` lists the identifier kinds the
index knows; Semantic Scholar adds kinds without notice, and those stay out of the index.
"""

from openknowledge_contracts.search.contract import CONTRACT_VERSION, SearchContract, SearchField

PAPERS = SearchContract(
    name="papers",
    version=1,
    id_field="corpusid",
    fields=(
        SearchField("corpusid", "keyword"),
        SearchField("title", "text"),
        SearchField("abstract", "text"),
        SearchField(
            "authors",
            "object",
            (SearchField("name", "text"), SearchField("authorId", "keyword")),
        ),
        SearchField("venue", "keyword"),
        SearchField("publicationvenueid", "keyword"),
        SearchField("year", "integer"),
        SearchField("publicationdate", "keyword"),
        SearchField("journal", "object", (SearchField("name", "keyword"),)),
        SearchField(
            "externalids",
            "object",
            tuple(
                SearchField(kind, "keyword")
                for kind in ("DOI", "ArXiv", "PubMed", "PubMedCentral", "MAG", "ACL", "DBLP")
            ),
            dynamic=False,
        ),
        SearchField("s2fieldsofstudy", "object", (SearchField("category", "keyword"),)),
        SearchField("publicationtypes", "keyword"),
        SearchField("referencecount", "long"),
        SearchField("citationcount", "long"),
        SearchField("influentialcitationcount", "long"),
        SearchField("isopenaccess", "boolean"),
        SearchField("url", "keyword"),
        SearchField(CONTRACT_VERSION, "integer"),
    ),
)
