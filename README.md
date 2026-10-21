# OpenKnowledgeHub

## Semantic Scholar download, locally

Needs `uv` and `gcloud auth application-default login` with an account that can create and read objects in the bucket.
Keep the API key in `~/.config/openknowledge/.env` as `S2_API_KEY=...` and let uv load it.
Run only `publication-venues` locally. Every other dataset downloads through Batch.
Direct local downloads do not participate in the shared managed-download budget.

```sh
uv sync
uv run --env-file ~/.config/openknowledge/.env okp s2 pin --bucket openknowledge-508216-raw --prefix test
uv run --env-file ~/.config/openknowledge/.env okp s2 download --release <release_id> --dataset publication-venues --bucket openknowledge-508216-raw --prefix test
```

`okp s2 pin` resolves `latest` and archives the release metadata.
It prints the explicit `release_id`, which every download of one import passes as `--release`.
That way datasets never mix releases.

`okp s2 download-shard --shard N` archives shard N of the pinned inventory without Prefect.
It lists the dataset once per process to find the shard's link, and the one request per second limit holds only within that process.

`--prefix` defaults to `semantic-scholar`. Local runs use `test`, so the folder can be deleted afterwards.

Every object is created once and never overwritten. Under `<prefix>/<release>/`:

- `release.json`: the release metadata with its README.
- `<dataset>/dataset.json`: the dataset metadata and the pinned inventory of its shards.
- `<dataset>/<file>`: the shard as served.
- `<dataset>/<file>.receipt.json`: its checksums and line count, written after the gzip check passed.
- `<dataset>/manifest.json`: every object that has a receipt, written by `okp s2 download` after the last receipt.

A dataset's diff from one release to the next goes under `<prefix>/diffs/<from>/<release>/<dataset>/`.
There `diff.json` pins its update and delete files, and `updates/` and `deletes/` each hold their files, receipts and `manifest.json`.

A rerun skips a shard whose receipt matches the stored object.
It rehashes an object that has no receipt.
It refreshes a signed link that has expired.

Settings come from the environment.
The key is `S2_API_KEY`, and every other setting is `OKP_` plus a field name from `pipelines/src/openknowledge_pipelines/settings.py`.

Unmanaged local flows can use Prefect's temporary local server when no API URL is configured.
`okp` turns that server's telemetry off.

## Downloading through Batch

On the control VM, `okp serve` also serves the deployment `s2-batch-download/control-vm`.
It pins the dataset, runs each shard without a receipt as a task of one Cloud Batch job, and writes the manifest.
Each task runs `okp s2 download-shard` from the same image with its shard in `OKP_SHARD`.
Up to `OKP_BATCH_PARALLELISM` tasks run at once, 1 by default.
Each task makes one listing call with the key, and S2 retries wait a random part of their backoff, so tasks that start together spread out.
A run makes up to `OKP_TASK_RETRIES` more rounds for shards still without a receipt, and a new run resumes the same way.

`s2-batch-download-diff/control-vm` does the same for one dataset's diff to the next release, with a job per part that has files.
Its tasks add `--from-release` and `--part` to `okp s2 download-shard`.

Batch VMs use Spot by default; `OKP_BATCH_PROVISIONING_MODEL=STANDARD` selects standard VMs.
Each submission keeps the same operation ID and job name across task retries, so a lost creation reply resumes that job rather than creating another one.
`OKP_BATCH_WAIT_TIMEOUT_SECONDS` bounds each job's whole wait, including queued time and API outages, to 24 hours by default, separately from each task's `OKP_BATCH_MAX_RUN_SECONDS`.
If that wait expires, the run fails and names the job.

All six managed download deployments share the `semantic-scholar` concurrency limit of 1.
One dataset job runs at a time, with up to `OKP_BATCH_PARALLELISM` parallel shards (8 when configured).
Busy requests fail promptly. Recovery handles queued/running jobs of the same dataset family;
active jobs from another family or legacy jobs without a known core dataset label block new work.
Legacy jobs labelled with one of the seven core datasets remain recoverable by core controllers.

## Full-text and SPECTER2 archives

Dedicated managed base/diff flows archive `s2orc_v2` and `embeddings-specter_v2` in the existing
raw GCS bucket. They reuse pinned inventories, receipts, manifests and resumable shard downloads.
Each dataset uses its own explicit release; these archives do not load into ClickHouse.
Managed archive commands require the shared Prefect server on the control VM, configured via
`PREFECT_API_URL` or the active Prefect profile, with the `semantic-scholar` limit created by
`okp serve`; they never fall back to a temporary local server. Both accept `--from-release`
for an explicit diff.

| Command | Deployments (base; diff) |
| --- | --- |
| `okp s2 archive-fulltext` | `s2-batch-download-fulltext/control-vm`; `s2-batch-download-fulltext-diff/control-vm` |
| `okp s2 archive-specter2` | `s2-batch-download-embeddings/control-vm`; `s2-batch-download-embeddings-diff/control-vm` |

`okp s2 read-base` emits a bounded JSONL prefix of a completed base for either archive
family, with generation-pinned source lineage. It needs storage read access only.
`ArchiveConsumer.stream` supports full traversal with `ConsumeBase(max_records=None)`;
close the async generator when stopping early. Prefix records are explicitly partial:
whole-shard checksums are verified on exhaustion. Archived diffs are not materialized.

## Coordinated sync

The Prefect deployment `s2-sync/control-vm` (or `okp s2 sync` on the shared server) updates
seven core datasets in ClickHouse, then archives full text and SPECTER2 in GCS.
It selects per-dataset releases and resumes contiguous diffs from verified progress.
Core failures, including a missing base, block both archives; initial bases require manual setup.
Core diffs above 30% of the configured full-base size are not loaded. This reference stays fixed
as diffs advance; refresh `core_baselines` after installing a replacement base.

`OKP_SYNC` is JSON with `bucket`, optional `prefix` (default `semantic-scholar`), and three
initially empty fields: `core_baselines`, `archives`, and `enabled_archives`.
`core_baselines` maps datasets to manifest references returned by `okp s2 verify-base-manifest`.
`archives` maps `s2orc_v2` or `embeddings-specter_v2` to `{base, completion}`: `base` contains
`dataset`, `release`, `bucket`, and optional `prefix`; `completion` is the verified reference
returned by `okp s2 adopt-archive-base`. Both helpers validate existing downloaded bases.
`enabled_archives` lists which archive stages run; a baseline alone does not enable a stage.
Manual archive commands with `--from-release` also require the matching `archives` entry.

The daily 01:30 Europe/Zurich schedule defaults disabled; enable it with
`OKP_SYNC_SCHEDULE_ENABLED=true` in the serving process environment.
`CANCEL_NEW` cancels overlapping deployed runs before execution, without a backlog;
these cancellations do not trigger the `run_failed` log alert.

## Loading into ClickHouse

```sh
uv run okp s2 load --release <release_id> --dataset publication-venues --bucket openknowledge-508216-raw
```

It loads every object in the dataset's `manifest.json` into the dataset's table.
ClickHouse reads each object itself through its `raw` named collection and checks every record against the dataset's contract.
It needs `OKP_CLICKHOUSE_URL`, `CLICKHOUSE_PASSWORD` and `OKP_CLICKHOUSE_RAW_BUCKET`, the bucket that collection reads, but no S2 key.

`okp s2 load --measurement-report` optionally writes a new JSON report for that single
load. It distinguishes acknowledged inserts from skipped checkpoints and unknown partial
writes; memory values are snapshot gauges, not peaks. Use
`--measurement-kind synthetic-validation` for disposable development loads. Measurement errors are reported
separately and preserve the load outcome.

Each loaded object leaves a row in `loads`.
A rerun skips an object whose row matches its generation, its checksum, the table and the load SQL, so a table created again or emptied loads everything again.
Apply migrations 00011 to 00014 before starting loads with the new image. Inserts reject NULL into non-nullable columns. The shared insert safety setting changes upsert and delete SQL hashes for all seven supported datasets. Eligible base files for all seven and diff update/delete files for the six datasets with diffs replay when next loaded, including files already loaded with the previous image. Allow an idle window for the 00012 journal rewrite and these replays.
Migration 00014 orders citation versions by release, then deletion, then whether `contexts` is present, so a record with contexts beats the same record without.
A citations load refuses until 00014 is applied, and 00014 refuses while `citations` holds rows or checkpoints from the old order, whose merges may have dropped contexts.
To get past it, stop citation loads, run `RENAME TABLE citations TO citations_legacy`, run the Up statement of 00007 to create an empty table with a new UUID, apply the migrations and load the base release again from all its raw objects.
The old checkpoints no longer count and stay in place. Drop `citations_legacy` once that load passes its checks.
`tests/test_load_e2e.py` runs it against ClickHouse and a fake GCS in Docker.
On the control VM, `okp serve` serves the same flow as the deployment `s2-load/control-vm`, so a run can start from the Prefect UI.

`okp s2 load-diff --from-release <release_id> --release <release_id> --dataset <name> --bucket <bucket>` loads one archived diff, its update files as upserts and then its delete files as tombstones, and `okp serve` serves it as `s2-load-diff/control-vm`.
Both load deployments take the Prefect global concurrency limit `clickhouse`, which `okp serve` creates with a limit of 1, so no two served loads overlap, and each load deployment runs one run at a time, so a second run of the same load waits on the server and holds no slot.
The `applied` table records each base load and diff that passed its checks, and a diff loads only into a table that holds the release it starts from or the one it brings, which is a rerun.
When lineage is recorded, a base loads only into an empty table or one holding the same release, and `publication-venues` also into one holding an older release.
A load of `publication-venues`, which has no diffs, flags the venues its release no longer lists in `removed`.

`okp serve` also serves `watchdog/control-vm`, which runs every 6 hours.
Each run writes `watchdog heartbeat`, then checks every table that follows diffs.
It writes one `watchdog stale` line per table whose last load was applied more than `OKP_WATCHDOG_MAX_AGE_DAYS` days ago, 14 by default.
A table never loaded does not count.
Stale data never fails a run, but a ClickHouse error does.
Both texts are fixed so that a log alert can match them.

Load recovery still needs persisted ownership of a begun load and shared coordination per table.
Until those are implemented, do not load a base over a populated table with missing lineage or overlap base and diff loads for the same dataset.

## Indexing papers into OpenSearch

```sh
uv run okp search index --index papers-v1 --release <release_id> [--from-release <release_id>] [--range I/N]
```

It reads the live rows of `papers` and `abstracts` from ClickHouse and writes one search document per paper into the index, with the abstract when the paper has one.
A base pass creates the index with the mapping of `packages/contracts/src/openknowledge_contracts/search/papers.py` when it does not exist, and writes into it when it does.
A diff pass needs the index to exist and stops with exit 1 otherwise.
Without `--from-release`, every live paper at the release is written.
With it, only the papers with a newer row in `papers` or `abstracts` are written, and a paper deleted since is deleted from the index.
`--range I/N` writes part I of the corpusid range split into N equal parts, 1/1 by default.
The range is the smallest to the largest corpusid at the release, read again by every part, so run all parts of one pass before the next load.
The rows of a part are read into memory whole, so pick N so that a part fits, and keep 1/1 for small tables.
The release must be the latest applied. The tables keep no history, so a base pass at an older release does not rebuild that release, it drops every paper whose latest row is newer, can carry an older abstract, and still reports success.
Each document carries the larger `version` of its two rows, and the writes use `external_gte`, which accepts an equal version again.
A part can be run again at the same or a newer release. A write below the version the index holds is refused and the pass stops with exit 1.
It needs `OKP_CLICKHOUSE_URL`, `CLICKHOUSE_PASSWORD`, `OKP_CLICKHOUSE_RAW_BUCKET`, `OKP_OPENSEARCH_URL` and `OPENSEARCH_PASSWORD`.
Unless set, `OKP_OPENSEARCH_USER` is `hub-indexer`, `OKP_OPENSEARCH_TIMEOUT_SECONDS` is 60 and `OKP_INDEX_SHARDS` is 5.
An OpenSearch or ClickHouse that cannot be reached ends the run with exit 1 before anything is written.
`tests/test_search_e2e.py` runs it against ClickHouse and OpenSearch in Docker.
The flow and the deployment come with the next pull request.

Before a pull request:

```sh
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run lint-imports
```

## Pipelines image

`pipelines/Dockerfile` builds one image for Batch download tasks, loads and the Prefect worker.
Its entry point is `okp`; the worker runs `prefect` from the same image.
Build it from the repository root:

```sh
docker build -f pipelines/Dockerfile -t pipelines .
docker run --rm pipelines s2 download-shard --help
```

Pull requests build and smoke-test it.
Every push to `main` publishes it as `europe-west4-docker.pkg.dev/openknowledge-508216/openknowledge/pipelines:<commit>` and writes the digest to the job summary.
Deployments pin that digest, never a tag, and pass it as `OKP_IMAGE_DIGEST`: receipts record it, and an image cannot know its own digest.
