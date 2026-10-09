"""The helper thread that writes a run's finished tables to disk.

A finished scenario is turned into tables on the main thread, which needs the
scenario's device arrays. The main thread then hands each table and its path
to `BackgroundWriter` and moves on to the next scenario while the file is
written.

Everything is on disk before the run says it is finished: `finish()` waits for
the queue to empty and joins the thread, and `run/output_directory.finish_run`
calls it before writing the run manifest, so the manifest cannot name a file
that is not there. A failed write fails the run: the thread stops on the first
error, and the next submission or `finish()` raises it as `FileWriteFailed`.

There is one thread, since the goal is overlapping writing with the rest of the
run rather than writing several files at once. The thread is registered to
drain at interpreter exit, so a caller that does not call `finish()` still
gets its files.

At most `TABLES_WAITING_TO_BE_WRITTEN` finished tables wait at a time. Once
that many are queued the main thread waits for the disk before handing over
another, which keeps the run's memory flat.
"""

from __future__ import annotations

import atexit
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

# Queue item that tells the thread there is no more work.
_NO_MORE_WORK = object()

# The number of finished tables that may wait to be written before the main
# thread stops and waits for the disk. Two keeps the disk busy, since while one
# file is being written another is queued behind it. A household-detail file is
# one row per household per year and runs to gigabytes, so an unbounded queue
# could hold several of those at once and run out of memory.
TABLES_WAITING_TO_BE_WRITTEN = 2


class FileWriteFailed(RuntimeError):
    """A file the run meant to write could not be written."""


class BackgroundWriter:
    """Writes files handed to it, on its own thread, in the order given."""

    def __init__(self) -> None:
        self._work: queue.Queue[Any] = queue.Queue(
            maxsize=TABLES_WAITING_TO_BE_WRITTEN
        )
        self._failure: BaseException | None = None
        self._failed_path: Path | None = None
        self._busy_seconds = 0.0
        self._thread = threading.Thread(
            target=self._keep_writing, name="run-file-writer", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)

    # -- submitting files ---------------------------------------------------

    def write_csv(self, path: Path, frame: pd.DataFrame, missing: str) -> None:
        """Write one table as CSV, with `missing` standing for a missing number."""
        self._submit(path, lambda: frame.to_csv(path, index=False, na_rep=missing))

    def write_parquet(self, path: Path, frame: pd.DataFrame) -> None:
        """Write one table as Parquet."""
        self._submit(path, lambda: frame.to_parquet(path, index=False))

    def write_with(self, path: Path, write: Callable[[], None]) -> None:
        """Write something that is not a table (a manifest) with the caller's own writer."""
        self._submit(path, write)

    # -- waiting for files --------------------------------------------------

    def finish(self) -> float:
        """Wait for every file handed over so far, then stop the thread.

        Returns the seconds the thread spent writing. Raises `FileWriteFailed`
        if any write failed.
        """
        self.stop()
        self.raise_any_failure()
        return self._busy_seconds

    def stop(self) -> None:
        """Finish the queue and end the thread, without raising.

        A run that is already failing calls this on its way out, so that a
        write failure is not raised on top of the error that stopped the run.
        Calling it twice is harmless.
        """
        if self._thread.is_alive():
            self._work.put(_NO_MORE_WORK)
            self._thread.join()
        atexit.unregister(self.stop)

    def wait_until_written(self) -> None:
        """Wait until everything handed over so far is on disk.

        The household-detail files are one row per household per year and too
        large for the run to hold two of. The run builds one world's table,
        hands it over, waits here, and then builds the next world's.
        """
        self._work.join()
        self.raise_any_failure()

    def raise_any_failure(self) -> None:
        """Raise the first write failure, if there was one."""
        if self._failure is not None:
            raise FileWriteFailed(
                f"the run could not write {self._failed_path}"
            ) from self._failure

    # -- the thread itself -------------------------------------------------

    def _submit(self, path: Path, write: Callable[[], None]) -> None:
        # A failed write surfaces at the next file the run hands over.
        self.raise_any_failure()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._work.put((Path(path), write))

    def _keep_writing(self) -> None:
        while True:
            item = self._work.get()
            if item is _NO_MORE_WORK:
                self._work.task_done()
                return
            path, write = item
            if self._failure is not None:
                # Once one file has failed the run is over, and the remaining
                # files are skipped.
                self._work.task_done()
                continue
            started = time.perf_counter()
            try:
                write()
            except BaseException as error:  # noqa: BLE001 - re-raised on the main thread
                self._failure = error
                self._failed_path = path
            self._busy_seconds += time.perf_counter() - started
            # Counted once the file is on disk, so that `wait_until_written`
            # returns only when the files are there.
            self._work.task_done()
