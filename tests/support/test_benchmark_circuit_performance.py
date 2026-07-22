from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.support.benchmark_circuit_performance import (
    _paired_order,
    _summarize_nsys_sqlite,
)


def test_paired_order_alternates_abba_and_baab() -> None:
    assert _paired_order(0) == ("baseline", "circuit", "circuit", "baseline")
    assert _paired_order(1) == ("circuit", "baseline", "baseline", "circuit")
    assert _paired_order(2) == ("baseline", "circuit", "circuit", "baseline")
    with pytest.raises(ValueError, match="non-negative"):
        _paired_order(-1)


def test_nsys_summary_embeds_graph_sync_copy_kernel_and_api_counts(tmp_path: Path) -> None:
    database = tmp_path / "profile.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
            (start INTEGER, end INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
            (start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY
            (start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER, copyCount INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
            (start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE
            (start INTEGER, end INTEGER, graphExecId INTEGER);
        INSERT INTO StringIds VALUES (1, 'cudaGraphLaunch_v10000');
        INSERT INTO StringIds VALUES (2, 'cudaDeviceSynchronize_v3020');
        INSERT INTO StringIds VALUES (3, 'cudaLaunchKernel_v7000');
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 10, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (10, 30, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (30, 35, 3);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 7);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (7, 16);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0, 5, 64, 1, NULL);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (5, 11, 32, 2, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (0, 20);
        INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (0, 12, 7);
        INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (12, 24, 7);
        """
    )
    connection.commit()
    connection.close()

    summary = _summarize_nsys_sqlite(database)

    assert summary["cuda_api"] == {"count": 3, "total_duration_ns": 35}
    assert summary["kernels"] == {"count": 2, "total_duration_ns": 16}
    assert summary["cuda_graph"]["launch_api_calls"] == 1
    assert summary["cuda_graph"]["trace_records"] == 2
    assert summary["cuda_graph"]["distinct_graph_exec_ids"] == 1
    assert summary["synchronization"]["api_calls"] == 1
    assert summary["synchronization"]["activity_records"] == 1
    assert summary["memcpy"]["host_to_device"]["bytes"] == 64
    assert summary["memcpy"]["device_to_host"]["operations"] == 2
