"""The ``okp`` command line, where only the commands that run flows load Prefect."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Coroutine, Mapping
from contextlib import aclosing
from pathlib import Path
from typing import Any, Literal, TextIO

import click
from pydantic import BaseModel, ValidationError

from openknowledge_core.ingestion import (
    DownloadDataset,
    DownloadDiff,
    DownloadShard,
    IngestionError,
    LoadDataset,
    LoadDatasetResult,
    LoadDiff,
    LoadDiffResult,
    PinRelease,
)
from openknowledge_core.ingestion.families import ArchiveFamily, archive_family
from openknowledge_core.search import IndexPapers
from openknowledge_pipelines import bootstrap
from openknowledge_pipelines.settings import (
    BatchSettings,
    CommonSettings,
    LoadSettings,
    PipelineSettings,
    SearchSettings,
    WatchdogSettings,
)

PIN_HELP = "Archive a release's metadata and print it as JSON, with the explicit release_id."
DOWNLOAD_HELP = "Archive a whole dataset through the Prefect flow."
DOWNLOAD_SHARD_HELP = "Archive one shard of a dataset, or one file of its diff, without Prefect."
LOAD_HELP = "Load an archived dataset into its ClickHouse table through the Prefect flow."
LOAD_DIFF_HELP = "Load an archived diff into its ClickHouse table through the Prefect flow."
SERVE_HELP = "Serve the flows the control VM runs, until stopped."
INDEX_HELP = "Write one corpusid slice of the papers into an OpenSearch index, without Prefect."
RELEASE_HELP = "The release_id that okp s2 pin printed, such as 2026-10-06."
DATASET_HELP = "Dataset name, such as publication-venues."
PREFIX_HELP = "Folder in the bucket that every object goes under."
OPTION_RULES = {
    "release": "--release must look like 2026-10-06, the release_id that okp s2 pin printed",
    "dataset": (
        "--dataset must look like publication-venues: lowercase letters, digits, hyphens "
        "and underscores"
    ),
    "shard": "--shard must be a whole number like 0, counting the shards in dataset.json from 0",
    "bucket": (
        "--bucket must be a bucket name of 3 to 63 lowercase letters, digits, dots, hyphens "
        "and underscores, without gs://"
    ),
    "prefix": "--prefix must look like semantic-scholar, with no empty, '.' or '..' part",
    "from_release": "--from-release must look like 2026-09-29, the release before --release",
}
PIN_RULES = {**OPTION_RULES, "release": "--release must be latest or look like 2026-10-06"}
RANGE_RULE = "--range must look like 1/4: the part to write and the number of equal corpusid parts"
SEARCH_RULES = {
    **OPTION_RULES,
    "index": (
        "--index must be an OpenSearch index name: lowercase letters, digits, hyphens, "
        "underscores and dots"
    ),
    "part": RANGE_RULE,
    "parts": RANGE_RULE,
}
SETTING_RULES = {
    "OKP_IMAGE_DIGEST": "OKP_IMAGE_DIGEST must be sha256: followed by 64 lowercase hex digits",
    "OKP_S2_API_URL": "OKP_S2_API_URL must be an http or https URL",
    "OKP_CLICKHOUSE_URL": "OKP_CLICKHOUSE_URL must be an http or https URL",
    "OKP_OPENSEARCH_URL": (
        "OKP_OPENSEARCH_URL must be an http or https URL without a user or password"
    ),
}
MISSING_KEY = (
    "S2_API_KEY is not set; keep it in ~/.config/openknowledge/.env and run okp through "
    "uv run --env-file ~/.config/openknowledge/.env"
)


def _default(kind: type[BaseModel], field: str) -> object:
    return kind.model_fields[field].default


@click.group()
def main() -> None:
    """OpenKnowledge pipelines."""


@main.group()
def s2() -> None:
    """Semantic Scholar datasets."""


@s2.command(help=PIN_HELP, short_help=PIN_HELP)
@click.option(
    "--release",
    default=_default(PinRelease, "release"),
    show_default=True,
    help="Release id, or latest.",
)
@click.option("--bucket", required=True, help="Target bucket name, without gs://.")
@click.option(
    "--prefix", default=_default(PinRelease, "prefix"), show_default=True, help=PREFIX_HELP
)
def pin(release: str, bucket: str, prefix: str) -> None:
    command = _command(PinRelease, PIN_RULES, release=release, bucket=bucket, prefix=prefix)
    settings = _settings(PipelineSettings)

    async def run() -> BaseModel:
        async with bootstrap.open_service(settings) as service:
            return await service.pin_release(command)

    _run(run)


@s2.command(help=DOWNLOAD_HELP, short_help=DOWNLOAD_HELP)
@click.option("--release", required=True, help=RELEASE_HELP)
@click.option("--dataset", required=True, help=DATASET_HELP)
@click.option("--bucket", required=True, help="Target bucket name, without gs://.")
@click.option(
    "--prefix", default=_default(DownloadDataset, "prefix"), show_default=True, help=PREFIX_HELP
)
def download(release: str, dataset: str, bucket: str, prefix: str) -> None:
    command = _command(
        DownloadDataset,
        OPTION_RULES,
        release=release,
        dataset=dataset,
        bucket=bucket,
        prefix=prefix,
    )
    settings = _settings(PipelineSettings)

    async def run() -> BaseModel:
        async with bootstrap.open_service(settings) as service:
            return await bootstrap.download_flow(service, settings)(command)

    _run(run)


def _archive(command: DownloadDataset | DownloadDiff) -> None:
    try:
        family = archive_family(command.dataset)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from None
    from openknowledge_pipelines import deployments

    _require_managed_server((deployments.S2_LIMIT,))
    _settings(BatchSettings)

    async def run() -> BaseModel:
        if isinstance(command, DownloadDiff):
            diff_routes = {
                ArchiveFamily.CORE: deployments.s2_batch_download_diff,
                ArchiveFamily.FULLTEXT: deployments.s2_batch_download_fulltext_diff,
                ArchiveFamily.EMBEDDINGS: deployments.s2_batch_download_embeddings_diff,
            }
            return await diff_routes[family](command)
        routes = {
            ArchiveFamily.CORE: deployments.s2_batch_download,
            ArchiveFamily.FULLTEXT: deployments.s2_batch_download_fulltext,
            ArchiveFamily.EMBEDDINGS: deployments.s2_batch_download_embeddings,
        }
        return await routes[family](command)

    _run_managed(run)


def _require_managed_server(required_limits: tuple[str, ...]) -> None:
    from httpx import ConnectError, TimeoutException
    from prefect.client.orchestration import get_client
    from prefect.exceptions import ObjectNotFound
    from prefect.settings import PREFECT_API_URL

    if not PREFECT_API_URL.value():
        raise click.ClickException(
            "Managed commands require the shared control VM Prefect server; configure "
            "PREFECT_API_URL in the environment or Prefect profile before retrying."
        )

    # Validate configuration before a decorated flow can log a missing-limit traceback.
    # This does not reserve capacity; strict acquisition in the flow remains authoritative.
    with get_client(sync_client=True) as client:
        for name in required_limits:
            try:
                client.read_global_concurrency_limit_by_name(name)
            except (ConnectError, TimeoutException):
                raise click.ClickException(
                    "Cannot reach the configured shared control API. Check network access, "
                    "the server and the effective PREFECT_API_URL setting, then retry."
                ) from None
            except ObjectNotFound:
                raise click.ClickException(
                    f"Required shared Prefect limit {name!r} is missing. Check that "
                    "PREFECT_API_URL selects the shared control VM server and run okp serve "
                    "there to create its limits before retrying."
                ) from None


def _run_managed(run: Callable[[], Coroutine[Any, Any, BaseModel]]) -> None:
    from prefect.concurrency.asyncio import ConcurrencySlotAcquisitionError

    try:
        _run(run)
    except ConcurrencySlotAcquisitionError as exc:
        raise click.ClickException(
            "Cannot acquire the required shared Prefect limits. Check the shared control VM "
            "Prefect server and that its limits exist; if occupied, retry when free. "
            f"{exc}"
        ) from None


def _archive_options(function: Callable[..., None]) -> Callable[..., None]:
    for option in (
        click.option("--release", required=True, help=RELEASE_HELP),
        click.option("--from-release", help="Archive this explicit diff instead of a base."),
        click.option("--bucket", required=True),
        click.option("--prefix", default="semantic-scholar", show_default=True),
    ):
        function = option(function)
    return function


def _archive_dataset(
    dataset: str, release: str, from_release: str | None, bucket: str, prefix: str
) -> None:
    values = dict(release=release, dataset=dataset, bucket=bucket, prefix=prefix)
    command = (
        _command(DownloadDiff, OPTION_RULES, from_release=from_release, **values)
        if from_release is not None
        else _command(DownloadDataset, OPTION_RULES, **values)
    )
    _archive(command)


@s2.command("archive-fulltext", help="Archive s2orc_v2 base or explicit diff through Batch.")
@_archive_options
def archive_fulltext(release: str, from_release: str | None, bucket: str, prefix: str) -> None:
    _archive_dataset("s2orc_v2", release, from_release, bucket, prefix)


@s2.command(
    "archive-specter2", help="Archive embeddings-specter_v2 base or explicit diff through Batch."
)
@_archive_options
def archive_specter2(release: str, from_release: str | None, bucket: str, prefix: str) -> None:
    _archive_dataset("embeddings-specter_v2", release, from_release, bucket, prefix)


@s2.command(
    "adopt-archive-base", help="Validate an existing base and create its immutable completion."
)
@click.option("--dataset", required=True)
@click.option("--release", required=True)
@click.option("--bucket", required=True)
@click.option("--prefix", default="semantic-scholar", show_default=True)
def adopt_archive_base(dataset: str, release: str, bucket: str, prefix: str) -> None:
    command = _command(
        DownloadDataset,
        OPTION_RULES,
        dataset=dataset,
        release=release,
        bucket=bucket,
        prefix=prefix,
    )
    if archive_family(dataset) == ArchiveFamily.CORE:
        raise click.UsageError("core progress uses applied releases")
    settings = _settings(PipelineSettings)

    async def run() -> BaseModel:
        async with bootstrap.open_archive_progress(settings) as progress:
            return await progress.finalize(command)

    _run(run)


@s2.command(
    "verify-base-manifest",
    help="Validate a completed base and print its pinned manifest reference.",
)
@click.option("--dataset", required=True)
@click.option("--release", required=True)
@click.option("--bucket", required=True)
@click.option("--prefix", default="semantic-scholar", show_default=True)
def verify_base_manifest(dataset: str, release: str, bucket: str, prefix: str) -> None:
    command = _command(
        DownloadDataset,
        OPTION_RULES,
        dataset=dataset,
        release=release,
        bucket=bucket,
        prefix=prefix,
    )
    settings = _settings(PipelineSettings)

    async def run() -> BaseModel:
        from openknowledge_core.ingestion.models import dataset_file_path

        async with bootstrap.open_archive_progress(settings) as progress:
            path = dataset_file_path(
                command.bucket, command.prefix, command.release, command.dataset, "manifest.json"
            )
            reference = await progress.reference(path)
            await progress.baseline_manifest(reference, dataset, bucket, prefix)
            return reference

    _run(run)


@s2.command("sync", help="Run one coordinator cycle using OKP_SYNC configuration.")
def sync() -> None:
    from openknowledge_pipelines import deployments

    _require_managed_server((deployments.S2_LIMIT, deployments.LOAD_LIMIT))
    _run_managed(deployments.s2_sync)


@s2.command("download-shard", help=DOWNLOAD_SHARD_HELP, short_help=DOWNLOAD_SHARD_HELP)
@click.option("--release", required=True, help=RELEASE_HELP)
@click.option("--dataset", required=True, help=DATASET_HELP)
@click.option(
    "--shard",
    required=True,
    envvar="OKP_SHARD",
    metavar="N",
    help="Index into the dataset's archived inventory, dataset.json; Batch tasks set OKP_SHARD.",
)
@click.option("--bucket", required=True, help="Target bucket name, without gs://.")
@click.option(
    "--prefix", default=_default(DownloadShard, "prefix"), show_default=True, help=PREFIX_HELP
)
@click.option(
    "--from-release",
    help="For a diff file: the release the diff starts from, right before --release.",
)
@click.option(
    "--part",
    type=click.Choice(["updates", "deletes"]),
    help="For a diff file: whether --shard counts the diff's update or delete files.",
)
def download_shard(
    release: str,
    dataset: str,
    shard: str,
    bucket: str,
    prefix: str,
    from_release: str | None,
    part: str | None,
) -> None:
    command = _command(
        DownloadShard,
        OPTION_RULES,
        release=release,
        dataset=dataset,
        shard=shard,
        bucket=bucket,
        prefix=prefix,
        from_release=from_release,
        part=part,
    )
    settings = _settings(PipelineSettings)

    async def run() -> BaseModel:
        async with bootstrap.open_service(settings) as service:
            return await service.archive_shard_at(command)

    _run(run)


@s2.command(help=LOAD_HELP, short_help=LOAD_HELP)
@click.option("--release", required=True, help=RELEASE_HELP)
@click.option("--dataset", required=True, help=DATASET_HELP)
@click.option("--bucket", required=True, help="Bucket holding the archive, without gs://.")
@click.option(
    "--prefix", default=_default(LoadDataset, "prefix"), show_default=True, help=PREFIX_HELP
)
@click.option(
    "--measurement-report",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write measurements for this one load to a new JSON file.",
)
@click.option(
    "--measurement-kind",
    type=click.Choice(["explicit-load", "synthetic-validation"]),
    default="explicit-load",
    show_default=True,
)
def load(
    release: str,
    dataset: str,
    bucket: str,
    prefix: str,
    measurement_report: Path | None = None,
    measurement_kind: Literal["explicit-load", "synthetic-validation"] = "explicit-load",
) -> None:
    command = _command(
        LoadDataset,
        OPTION_RULES,
        release=release,
        dataset=dataset,
        bucket=bucket,
        prefix=prefix,
    )
    settings = _settings(LoadSettings)

    async def run() -> LoadDatasetResult:
        async with bootstrap.open_load_service(settings) as service:
            return await bootstrap.load_flows(service, settings).dataset(command)

    if measurement_report is None:
        _run(run)
    else:
        from openknowledge_pipelines.flows.measure_load import measured_load

        async def measured(report_file: TextIO) -> LoadDatasetResult:
            async with bootstrap.open_measured_load(settings) as (service, observation, telemetry):
                return await measured_load(
                    lambda: bootstrap.load_flows(service, settings).dataset(command),
                    command=command,
                    observation=observation,
                    telemetry=telemetry,
                    report_file=report_file,
                    validation_kind=measurement_kind,
                    warn=lambda message: click.echo(message, err=True),
                )

        path = measurement_report.resolve()
        try:
            report_file = path.open("x", encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                f"Cannot create measurement report ({type(exc).__name__}); "
                "choose a new writable file"
            ) from None
        try:
            _run(lambda: measured(report_file))
        finally:
            try:
                report_file.close()
            except OSError as exc:
                click.echo(f"Measurement report close failed ({type(exc).__name__}).", err=True)


@s2.command("load-diff", help=LOAD_DIFF_HELP, short_help=LOAD_DIFF_HELP)
@click.option(
    "--from-release",
    required=True,
    help="The release the diff starts from, right before --release.",
)
@click.option("--release", required=True, help=RELEASE_HELP)
@click.option("--dataset", required=True, help=DATASET_HELP)
@click.option("--bucket", required=True, help="Bucket holding the archive, without gs://.")
@click.option("--prefix", default=_default(LoadDiff, "prefix"), show_default=True, help=PREFIX_HELP)
def load_diff(from_release: str, release: str, dataset: str, bucket: str, prefix: str) -> None:
    command = _command(
        LoadDiff,
        OPTION_RULES,
        from_release=from_release,
        release=release,
        dataset=dataset,
        bucket=bucket,
        prefix=prefix,
    )
    settings = _settings(LoadSettings)

    async def run() -> LoadDiffResult:
        async with bootstrap.open_load_service(settings) as service:
            return await bootstrap.load_flows(service, settings).diff(command)

    _run(run)


@main.group()
def search() -> None:
    """The paper search index."""


@search.command("index", help=INDEX_HELP, short_help=INDEX_HELP)
@click.option(
    "--index",
    required=True,
    help=(
        "Index name, such as papers-v1; a base pass creates it with the contract's mapping "
        "when it is missing, a diff pass needs it to exist."
    ),
)
@click.option("--release", required=True, help=RELEASE_HELP)
@click.option(
    "--from-release",
    help="Write only the papers changed since this release, which the index already holds.",
)
@click.option(
    "--range",
    "corpus_range",
    default="1/1",
    show_default=True,
    metavar="I/N",
    help="Part I of the corpusid range split into N equal parts.",
)
def index(index: str, release: str, from_release: str | None, corpus_range: str) -> None:
    found = re.fullmatch(r"(\d+)/(\d+)", corpus_range)
    if found is None:
        raise click.UsageError(RANGE_RULE)
    command = _command(
        IndexPapers,
        SEARCH_RULES,
        index=index,
        release=release,
        from_release=from_release,
        part=int(found[1]),
        parts=int(found[2]),
    )
    settings = _settings(SearchSettings)

    async def run() -> BaseModel:
        async with bootstrap.open_search_service(settings) as service:
            return await service.index(command)

    _run(run)


@main.command(help=SERVE_HELP, short_help=SERVE_HELP)
def serve() -> None:
    _settings(BatchSettings)
    _settings(WatchdogSettings)
    from openknowledge_pipelines import deployments

    deployments.serve()


def _command[C: BaseModel](kind: type[C], rules: Mapping[str, str], **values: object) -> C:
    try:
        return kind.model_validate(values)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_url=False)
        options = dict.fromkeys(str(error["loc"][0]) for error in errors if error["loc"])
        together = [
            error["msg"].removeprefix("Value error, ") for error in errors if not error["loc"]
        ]
    messages = [rules.get(option, f"--{option} is not valid") for option in options]
    raise click.UsageError("; ".join(messages + together) or "these options do not fit together")


def _settings[S: CommonSettings](kind: type[S]) -> S:
    try:
        return kind()  # pyright: ignore[reportCallIssue]  # fields come from the environment
    except ValidationError as exc:
        problems = [_setting_problem(error) for error in exc.errors(include_input=False)]
    raise click.ClickException("; ".join(problems))


def _setting_problem(error: Any) -> str:  # noqa: ANN401 - pydantic's ErrorDetails
    field = ".".join(str(part) for part in error["loc"])
    prefix = PipelineSettings.model_config.get("env_prefix", "")
    name = field if field.isupper() else f"{prefix}{field}".upper()
    if error["type"] == "missing":
        return MISSING_KEY if name == "S2_API_KEY" else f"{name} is not set"
    return SETTING_RULES.get(name, f"{name}: {str(error['msg']).removeprefix('Value error, ')}")


def _run(run: Callable[[], Coroutine[Any, Any, BaseModel]]) -> None:
    try:
        result = asyncio.run(run())
    except IngestionError as exc:
        message = str(exc)
    else:
        click.echo(result.model_dump_json())
        return
    raise click.ClickException(message)


@s2.command("read-base", help="Stream a bounded JSONL prefix of a completed raw base with lineage.")
@click.option("--dataset", type=click.Choice(["s2orc_v2", "embeddings-specter_v2"]), required=True)
@click.option("--release", required=True)
@click.option("--bucket", required=True)
@click.option("--prefix", default="semantic-scholar", show_default=True)
@click.option("--max-records", type=click.IntRange(1, 10000), default=10, show_default=True)
@click.option("--max-record-bytes", type=click.IntRange(1, 67108864), default=8388608)
def read_base(
    dataset: str,
    release: str,
    bucket: str,
    prefix: str,
    max_records: int,
    max_record_bytes: int,
) -> None:
    from openknowledge_core.ingestion.consumers import ConsumeBase

    command = _command(
        ConsumeBase,
        OPTION_RULES,
        dataset=dataset,
        release=release,
        bucket=bucket,
        prefix=prefix,
        max_records=max_records,
        max_record_bytes=max_record_bytes,
    )
    settings = _settings(CommonSettings)

    async def run() -> None:
        async with bootstrap.open_archive_consumer(settings) as consumer:
            async with aclosing(consumer.stream(command)) as records:
                async for record in records:
                    click.echo(record.model_dump_json())

    try:
        asyncio.run(run())
    except IngestionError as exc:
        raise click.ClickException(str(exc)) from None
