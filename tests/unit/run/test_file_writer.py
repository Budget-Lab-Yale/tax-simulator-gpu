"""The helper thread that writes a run's files: it finishes, and it fails loudly."""

import threading
import time

import pandas as pd
import pytest

from taxmodel.run import file_writer


def test_every_file_handed_over_is_on_disk_once_the_writer_is_finished(tmp_path):
    writer = file_writer.BackgroundWriter()
    frame = pd.DataFrame({"year": [2027, 2028], "receipts": [1.5, None]})
    for number in range(20):
        writer.write_csv(tmp_path / f"table_{number}.csv", frame, missing="NA")
        writer.write_parquet(tmp_path / f"table_{number}.parquet", frame)
    writer.finish()

    for number in range(20):
        written = pd.read_csv(tmp_path / f"table_{number}.csv")
        assert list(written["year"]) == [2027, 2028]
        assert pd.read_parquet(tmp_path / f"table_{number}.parquet").equals(frame)


def test_a_missing_column_is_written_as_NA_in_the_csv(tmp_path):
    writer = file_writer.BackgroundWriter()
    writer.write_csv(
        tmp_path / "receipts.csv",
        pd.DataFrame({"receipts": [None]}),
        missing="NA",
    )
    writer.finish()
    assert (tmp_path / "receipts.csv").read_text().splitlines() == ["receipts", "NA"]


def test_a_write_that_fails_ends_the_run_rather_than_passing_quietly(tmp_path):
    writer = file_writer.BackgroundWriter()

    def cannot_be_written() -> None:
        raise OSError("the disk is full")

    writer.write_with(tmp_path / "totals.csv", cannot_be_written)
    with pytest.raises(file_writer.FileWriteFailed) as failure:
        writer.finish()
    assert "totals.csv" in str(failure.value)
    assert isinstance(failure.value.__cause__, OSError)


def test_the_failure_reaches_the_run_at_the_next_file_it_hands_over(tmp_path):
    writer = file_writer.BackgroundWriter()

    def cannot_be_written() -> None:
        raise OSError("the disk is full")

    writer.write_with(tmp_path / "totals.csv", cannot_be_written)
    # Wait for the thread to have taken the failing write, then hand over
    # another file: the run should hear about the failure here, not only at
    # the end.
    writer.stop()
    with pytest.raises(file_writer.FileWriteFailed):
        writer.write_csv(tmp_path / "later.csv", pd.DataFrame({"a": [1]}), missing="NA")


def test_the_run_waits_for_the_disk_rather_than_piling_up_finished_tables(tmp_path):
    """Only a couple of tables may wait at a time, so memory cannot grow."""
    writer = file_writer.BackgroundWriter()
    let_the_first_write_finish = threading.Event()
    handed_over = []

    def slow_write() -> None:
        let_the_first_write_finish.wait(timeout=10)

    writer.write_with(tmp_path / "slow", slow_write)

    def hand_over_five() -> None:
        for number in range(5):
            writer.write_with(tmp_path / str(number), lambda: None)
            handed_over.append(number)

    handing_over = threading.Thread(target=hand_over_five)
    handing_over.start()
    # With the first write stuck, the run gets no further than filling the
    # queue: one table in the writer's hand and the queue's own limit behind it.
    time.sleep(0.5)
    assert len(handed_over) <= file_writer.TABLES_WAITING_TO_BE_WRITTEN

    let_the_first_write_finish.set()
    handing_over.join(timeout=10)
    writer.finish()
    assert handed_over == [0, 1, 2, 3, 4]


def test_files_are_written_in_the_order_they_are_handed_over(tmp_path):
    writer = file_writer.BackgroundWriter()
    order = []
    for name in ("first", "second", "third"):
        writer.write_with(tmp_path / name, lambda name=name: order.append(name))
    writer.finish()
    assert order == ["first", "second", "third"]
