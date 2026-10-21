"""Search failures, on the ingestion error base so that one command line reports both."""

from openknowledge_core.ingestion.errors import IngestionError, TransientError


class SearchUnavailable(TransientError):
    """OpenSearch cannot be reached, or answered with a status a retry may fix."""


class SearchRejected(IngestionError):
    """OpenSearch refused a call in a way a retry will not fix."""


class BulkRefused(IngestionError):
    """A bulk request was answered, but at least one of its items was refused."""
