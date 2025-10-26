"""Tests for the log analysis utility."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from scripts.analyze_extraction_log import analyze_log_file


@pytest.fixture()
def sample_log(tmp_path: Path) -> Path:
    """Create a sample log file to analyze."""

    content = dedent(
        """
        2025-10-26 16:31:07,858 - INFO - [RapidOCR] download_file.py:68: Initiating download: https://example.com/model.det
        2025-10-26 16:31:18,296 - INFO - [RapidOCR] download_file.py:95: Successfully saved to: /tmp/model.det
        2025-10-26 16:33:52,770 - INFO - Processing document biology2012.pdf
        2025-10-26 16:35:05,453 - INFO - Finished converting document biology2012.pdf in 238.35 sec.
        2025-10-26 16:35:13,418 - INFO - Document biology2012 timings | parse=238.37s domain=0.01s inventory=0.00s extraction=7.94s co_mentions=0.00s | processed=140 skipped=57 errors=0
        2025-10-26 16:35:20,603 - INFO - Extraction completed for paper biology2012 in 253.51s (processed=140, skipped=57, nodes=80, edges=69, errors=0)
        """
    ).strip()
    log_path = tmp_path / "sample.log"
    log_path.write_text(content, encoding="utf-8")
    return log_path


def test_parse_lines_extracts_download_and_document_timings(sample_log: Path) -> None:
    """Ensure the parser extracts downloads and per-document timings."""

    summary = analyze_log_file(sample_log)

    assert len(summary.downloads) == 1
    download = summary.downloads[0]
    assert download.duration_seconds is not None
    assert abs(download.duration_seconds - 10.438) < 0.001

    assert len(summary.documents) == 1
    document = summary.documents[0]
    assert document.conversion_seconds == pytest.approx(238.35)
    assert document.parse_seconds == pytest.approx(238.37)
    assert document.extraction_seconds == pytest.approx(7.94)
    assert document.total_seconds == pytest.approx(253.51)
    assert document.dominant_stage() == "parse"


def test_format_report_includes_dominant_stage(sample_log: Path) -> None:
    """Verify the formatted report mentions the dominant stage."""

    summary = analyze_log_file(sample_log)
    report = summary.format_report()

    assert "Download durations" in report
    assert "dominant: parse" in report
