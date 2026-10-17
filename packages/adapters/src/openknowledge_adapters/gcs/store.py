"""Create-only objects in Google Cloud Storage over gcloud-aio-storage."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import math
import os
import threading
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any, cast

import aiohttp
from gcloud.aio.storage import Storage
from gcloud.aio.storage.storage import StreamResponse

from openknowledge_contracts.manifests import ObjectPath, ObjectRef
from openknowledge_core.ingestion.errors import (
    TRANSIENT_STATUSES,
    IncompleteTransfer,
    IngestionError,
    ObjectChanged,
    StoreRejected,
    StoreUnavailable,
)
from openknowledge_core.ingestion.models import ByteStream, CheckedStream, StoredObject

NO_CREDENTIALS = (
    "could not get Google credentials; run gcloud auth application-default login, "
    "or give the VM a service account"
)
UNUSABLE_CREDENTIALS_FILE = (
    "the credentials file named by GOOGLE_APPLICATION_CREDENTIALS cannot be used; fix the "
    "file or unset the variable, then rerun"
)
STORAGE_PATHS = ("/storage/v1/", "/upload/storage/v1/")
TOKEN_REFUSED_STATUS = 401  # GCS refuses an expired or revoked token with it
READ_SIZE = 1024 * 1024
QUEUE_DEPTH = 8  # chunks waiting for the upload: 8 MiB with the S2 downloader's 1 MiB chunks
IDLE_TIMEOUT_SECONDS = 10.0
IDENTITY = {"accept-encoding": "identity"}


class GcsArtifactStore:
    def __init__(self, storage: Storage, *, request_timeout: int, min_transfer_rate: int) -> None:
        self._storage = storage
        self._request_timeout = request_timeout
        self._min_transfer_rate = min_transfer_rate

    def _timeout(self, size: int) -> int:
        """Total and per-read limits, as a ClientTimeout that gcloud-aio passes on unchanged."""
        total = transfer_timeout(size, self._request_timeout, self._min_transfer_rate)
        # aiohttp starts the read timer once the body is sent, so a slow upload is not cut.
        return cast(int, aiohttp.ClientTimeout(total=total, sock_read=self._request_timeout))

    async def list_objects(self, prefix: ObjectPath) -> tuple[StoredObject, ...]:
        params = {"prefix": prefix.name + "/"}
        objects: list[StoredObject] = []
        seen: set[str] = set()
        try:
            while True:
                page = await self._storage.list_objects(
                    prefix.bucket, params=params.copy(), timeout=self._timeout(0)
                )
                for item in page.get("items", []):
                    path = ObjectPath(bucket=prefix.bucket, name=item["name"])
                    if not path.name.startswith(prefix.name + "/"):
                        raise ValueError("listing escaped prefix")
                    objects.append(_stored(path, item))
                token = page.get("nextPageToken")
                if not token:
                    return tuple(objects)
                if not isinstance(token, str) or token in seen:
                    raise ValueError("invalid pagination token")
                seen.add(token)
                params["pageToken"] = token
        except aiohttp.ClientResponseError as exc:
            error = _failure("list", prefix, exc)
        except Exception as exc:
            error = _unreachable("list", prefix, exc)
        raise error

    async def stat(self, path: ObjectPath) -> StoredObject | None:
        try:
            metadata = await self._storage.download_metadata(
                path.bucket, path.name, timeout=self._timeout(0)
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status == 404 and _from_storage(exc):
                return None
            error = _failure("look up", path, exc)
        except Exception as exc:
            error = _unreachable("look up", path, exc)
        else:
            return _stored(path, metadata)
        raise error

    async def create(
        self, path: ObjectPath, data: bytes, *, content_type: str
    ) -> StoredObject | None:
        try:
            metadata = await self._storage.upload(
                path.bucket,
                path.name,
                data,
                content_type=content_type,
                parameters={"ifGenerationMatch": "0"},
                force_resumable_upload=False,
                timeout=self._timeout(len(data)),
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status == 412 and _from_storage(exc):
                return None
            error = _failure("create", path, exc)
        except Exception as exc:
            error = _unreachable("create", path, exc)
        else:
            return _stored(path, metadata)
        raise error

    async def create_from_stream(
        self, path: ObjectPath, body: CheckedStream, *, content_type: str
    ) -> StoredObject | None:
        reader = _StreamReader(asyncio.get_running_loop(), body.size)
        pump = asyncio.create_task(reader.fill(body))
        try:
            metadata = await self._storage.upload(
                path.bucket,
                path.name,
                reader,
                content_type=content_type,
                parameters={"ifGenerationMatch": "0"},
                force_resumable_upload=True,
                timeout=self._timeout(body.size),
            )
        except Exception as exc:
            # aiohttp's error holds the session URI and token, so no local keeps it.
            exists, rejected, kind = *_upload_outcome(path, exc), type(exc).__name__
        except BaseException:
            await _stop(reader, pump)
            raise
        else:
            if pump.done():
                await pump
            else:
                await _stop(reader, pump)  # core sees the body was not completed and refuses
            return _stored(path, metadata)
        failure = await _stop(reader, pump)
        if failure is not None:
            raise failure
        if exists:
            return None
        if rejected is not None:
            raise rejected
        raise StoreUnavailable(f"GCS could not upload {path} ({kind}); rerun later")

    async def open_read(self, ref: ObjectRef) -> ByteStream:
        before = await self.stat(ref.path)
        if before is None or before.ref.generation != ref.generation:
            raise ObjectChanged(
                f"{ref.path} no longer has generation {ref.generation}; rerun to check it as it is"
            )
        try:
            response = await self._storage.download_stream(
                ref.path.bucket,
                ref.path.name,
                headers=dict(IDENTITY),
                timeout=self._timeout(before.size),
            )
        except aiohttp.ClientResponseError as exc:
            error = _failure("read", ref.path, exc)
        except Exception as exc:
            error = _unreachable("read", ref.path, exc)
        else:
            chunks = self._chunks(response, ref)
            await anext(chunks)  # started, so closing it unread still releases the response
            return ByteStream(expected_size=before.size, chunks=chunks)
        raise error

    async def _chunks(
        self, response: StreamResponse, ref: ObjectRef
    ) -> AsyncGenerator[bytes, None]:
        """A generation change by the end of the read means the bytes cannot be trusted."""
        error: IngestionError | None = None
        async with response:
            yield b""  # taken by open_read, which starts the generator
            while True:
                try:
                    chunk = await response.read(READ_SIZE)
                except (aiohttp.ClientError, TimeoutError) as exc:
                    error = StoreUnavailable(
                        f"GCS stopped sending {ref.path} ({type(exc).__name__}); rerun later"
                    )
                    break
                if not chunk:
                    break
                yield chunk
        if error is not None:
            raise error
        after = await self.stat(ref.path)
        if after is None or after.ref.generation != ref.generation:
            raise ObjectChanged(
                f"{ref.path} changed generation while it was read; rerun to check it as it is"
            )


def transfer_timeout(size: int, request_timeout: int, min_transfer_rate: int) -> int:
    """Aiohttp's timeout covers a whole transfer, so it grows with the object's size."""
    return request_timeout + math.ceil(size / min_transfer_rate)


class _Aborted(OSError):
    """Raised into the executor thread so aiohttp abandons the PUT before its last byte."""


class _StreamReader(io.RawIOBase):
    """The file object aiohttp reads on executor threads during an upload, fed from the loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, size: int) -> None:
        super().__init__()
        self._loop = loop
        self._size = size
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(QUEUE_DEPTH)
        self._buffer = bytearray()
        self._position = 0
        self._ended = False
        self._lock = threading.Lock()
        self._pending: concurrent.futures.Future[bytes | None] | None = None
        self._aborted = False
        self._idle = threading.Event()
        self._idle.set()
        self.failure: IngestionError | None = None

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        # gcloud-aio seeks to the end and back only to learn the length, so any other seek retries.
        if whence == os.SEEK_END and offset == 0:
            return self._size
        if whence == os.SEEK_SET and offset == self._position:
            return offset
        raise io.UnsupportedOperation("a streamed upload cannot rewind")

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise ValueError("read() needs a size; the body is streamed")
        self._idle.clear()
        try:
            while len(self._buffer) < size and not self._ended:
                self._take()
            take = min(size, len(self._buffer))
            if self._position + take == self._size:
                while not self._ended:
                    self._take()  # the last bytes wait for the end of the checked body
            elif self._ended and self._position + len(self._buffer) < self._size:
                self._fail(IncompleteTransfer("the body ended before its declared size"))
            out = bytes(self._buffer[:take])
            del self._buffer[:take]
            self._position += take
            return out
        finally:
            self._idle.set()

    def _take(self) -> None:
        item = self._next()
        if item is None:
            self._ended = True
            return
        self._buffer += item
        if self._position + len(self._buffer) > self._size:
            self._fail(IncompleteTransfer("the body is longer than its declared size"))

    def _next(self) -> bytes | None:
        with self._lock:
            if self._aborted:
                raise _Aborted("the upload was abandoned")
            future = asyncio.run_coroutine_threadsafe(self._queue.get(), self._loop)
            self._pending = future
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise _Aborted("the upload was abandoned") from None
        finally:
            with self._lock:
                self._pending = None

    def _fail(self, failure: IngestionError) -> None:
        self.failure = failure
        raise _Aborted(str(failure))

    def abort(self) -> None:
        """Called on the event loop so a read that needs another chunk raises instead of waiting."""
        with self._lock:
            self._aborted = True
            if self._pending is not None:
                self._pending.cancel()

    async def fill(self, body: CheckedStream) -> None:
        try:
            async with aclosing(body.chunks) as chunks:
                async for chunk in chunks:
                    if chunk:
                        await self._queue.put(chunk)
            if not body.completed:
                raise IncompleteTransfer("the body ended before its checks completed")
            await self._queue.put(None)
        except BaseException:
            self.abort()
            raise

    async def wait_idle(self) -> None:
        """Wait until no executor thread is inside ``read``, which is immediate after ``abort``."""
        await asyncio.shield(asyncio.to_thread(self._idle.wait, IDLE_TIMEOUT_SECONDS))


async def _stop(reader: _StreamReader, pump: asyncio.Task[None]) -> BaseException | None:
    reader.abort()
    pump.cancel()
    await asyncio.wait([pump])
    await reader.wait_idle()
    if not pump.cancelled() and pump.exception() is not None:
        return pump.exception()
    return reader.failure


def _stored(path: ObjectPath, metadata: dict[str, Any]) -> StoredObject:
    try:
        stored = StoredObject(
            ref=ObjectRef(path=path, generation=int(metadata["generation"])),
            size=int(metadata["size"]),
            md5=metadata.get("md5Hash"),
            crc32c=metadata["crc32c"],
        )
        named = metadata["name"] == path.name and metadata["bucket"] == path.bucket
    except (KeyError, TypeError, ValueError):
        stored, named = None, False
    if stored is None or not named:
        raise StoreUnavailable(
            f"GCS described {path} without a usable name, generation, size or checksum; rerun later"
        )
    return stored


def _upload_outcome(path: ObjectPath, exc: BaseException) -> tuple[bool, IngestionError | None]:
    """Whether the object already exists, else the error for the HTTP answer behind ``exc``."""
    seen: BaseException | None = exc
    while seen is not None and not isinstance(seen, aiohttp.ClientResponseError):
        seen = seen.__cause__ or seen.__context__  # the library's retry can mask the answer
    if seen is None:
        return False, None
    if seen.status == 412 and _from_storage(seen):
        return True, None
    return False, _failure("upload", path, seen)


def _from_storage(exc: aiohttp.ClientResponseError) -> bool:
    """False when the error came from a token refresh gcloud-aio ran inside the call, not GCS."""
    return exc.request_info.real_url.path.startswith(STORAGE_PATHS)


def _failure(action: str, path: ObjectPath, exc: aiohttp.ClientResponseError) -> IngestionError:
    if not _from_storage(exc):
        return token_failure(exc.status)
    if exc.status in TRANSIENT_STATUSES:
        return StoreUnavailable(
            f"GCS answered HTTP {exc.status} when asked to {action} {path}; rerun later"
        )
    if exc.status == TOKEN_REFUSED_STATUS:
        return no_credentials()
    if exc.status == 403:
        return StoreRejected(
            f"GCS refused to {action} {path} with HTTP 403; give this account access "
            f"to create and read objects in {path.bucket}, then rerun"
        )
    if exc.status == 404 and action in ("create", "upload"):
        return StoreRejected(f"GCS has no bucket {path.bucket}; check --bucket, then rerun")
    return StoreRejected(
        f"GCS refused to {action} {path} with HTTP {exc.status}; rerun, and check the bucket "
        "if it repeats"
    )


def token_failure(status: int) -> IngestionError:
    if status in TRANSIENT_STATUSES:
        return StoreUnavailable(f"Google's token service answered HTTP {status}; rerun later")
    return no_credentials()


def no_credentials() -> StoreRejected:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return StoreRejected(UNUSABLE_CREDENTIALS_FILE)
    return StoreRejected(NO_CREDENTIALS)


def _unreachable(action: str, path: ObjectPath, exc: Exception) -> StoreUnavailable:
    return StoreUnavailable(f"GCS could not {action} {path} ({type(exc).__name__}); rerun later")
