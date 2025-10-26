"""Utilities for extracting performance insights from extraction logs."""

from __future__ import annotations

import argparse
import dataclasses
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
DOWNLOAD_START_PATTERN = re.compile(r"Initiating download: (?P<url>\S+)")
DOWNLOAD_END_PATTERN = re.compile(r"Successfully saved to: (?P<target>.+)")
DOCUMENT_FINISHED_PATTERN = re.compile(
    r"Finished converting document (?P<doc>.+?) in (?P<seconds>[\d.]+) sec"
)
DOCUMENT_TIMINGS_PATTERN = re.compile(
    r"Document (?P<doc>.+?) timings \| parse=(?P<parse>[\d.]+)s .* extraction="
    r"(?P<extraction>[\d.]+)s"
)
PIPELINE_TOTAL_PATTERN = re.compile(
    r"Extraction completed for paper (?P<paper>.+?) in (?P<seconds>[\d.]+)s"
)


@dataclasses.dataclass
class DownloadRecord:
    """Represents a single download event tracked in the log."""

    url: str
    start: datetime
    finish: Optional[datetime] = None
    target: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Return the duration of the download in seconds if complete."""

        if self.finish is None:
            return None
        return (self.finish - self.start).total_seconds()


@dataclasses.dataclass
class DocumentTiming:
    """Contains timing information for a processed document."""

    name: str
    parse_seconds: Optional[float] = None
    extraction_seconds: Optional[float] = None
    conversion_seconds: Optional[float] = None
    total_seconds: Optional[float] = None

    def dominant_stage(self) -> Optional[str]:
        """Return the stage that consumed the most time."""

        stages = {
            "parse": self.parse_seconds,
            "extraction": self.extraction_seconds,
            "conversion": self.conversion_seconds,
        }
        filtered = {key: value for key, value in stages.items() if value is not None}
        if not filtered:
            return None
        dominant = max(filtered, key=filtered.get)
        return dominant


@dataclasses.dataclass
class LogSummary:
    """Aggregated insights extracted from a log."""

    downloads: List[DownloadRecord]
    documents: List[DocumentTiming]

    def format_report(self) -> str:
        """Generate a human-readable report describing the log summary."""

        lines: List[str] = []

        if self.downloads:
            lines.append("Download durations (longest first):")
            completed = [d for d in self.downloads if d.duration_seconds is not None]
            completed.sort(key=lambda rec: rec.duration_seconds or 0, reverse=True)
            for record in completed:
                duration = record.duration_seconds
                lines.append(
                    f"  - {record.url} → {record.target} took {duration:.2f}s"
                )
            incomplete = [d for d in self.downloads if d.duration_seconds is None]
            if incomplete:
                lines.append("  - Incomplete downloads detected:")
                for record in incomplete:
                    lines.append(f"      * {record.url} (started {record.start})")
            lines.append("")

        if self.documents:
            lines.append("Document stage timings:")
            for timing in self.documents:
                dominant = timing.dominant_stage()
                components: List[str] = []
                if timing.parse_seconds is not None:
                    components.append(f"parse={timing.parse_seconds:.2f}s")
                if timing.extraction_seconds is not None:
                    components.append(
                        f"extraction={timing.extraction_seconds:.2f}s"
                    )
                if timing.conversion_seconds is not None:
                    components.append(
                        f"conversion={timing.conversion_seconds:.2f}s"
                    )
                if timing.total_seconds is not None:
                    components.append(f"total={timing.total_seconds:.2f}s")
                if dominant is not None:
                    dominant_fragment = f" (dominant: {dominant})"
                else:
                    dominant_fragment = ""
                lines.append(
                    f"  - {timing.name}: {', '.join(components)}{dominant_fragment}"
                )
        else:
            lines.append("No document timing information found.")

        return "\n".join(lines)


def _parse_timestamp(line: str) -> Optional[datetime]:
    """Extract a timestamp from a log line if present."""

    match = TIMESTAMP_PATTERN.search(line)
    if not match:
        return None
    raw_timestamp = match.group(1)
    return datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S,%f")


def _resolve_document(
    documents: "OrderedDict[str, DocumentTiming]", name: str
) -> DocumentTiming:
    """Retrieve a ``DocumentTiming`` instance using a flexible name match."""

    if name in documents:
        return documents[name]

    normalized = name.removesuffix(".pdf")
    for key, timing in documents.items():
        if key.removesuffix(".pdf") == normalized:
            return timing

    timing = DocumentTiming(name=name)
    documents[name] = timing
    return timing


def _parse_lines(lines: Iterable[str]) -> LogSummary:
    """Parse log lines into a structured summary."""

    downloads: "OrderedDict[str, DownloadRecord]" = OrderedDict()
    documents: "OrderedDict[str, DocumentTiming]" = OrderedDict()

    for line in lines:
        timestamp = _parse_timestamp(line)

        if timestamp is None:
            continue

        download_start_match = DOWNLOAD_START_PATTERN.search(line)
        if download_start_match:
            url = download_start_match.group("url")
            downloads[url] = DownloadRecord(url=url, start=timestamp)
            continue

        download_end_match = DOWNLOAD_END_PATTERN.search(line)
        if download_end_match:
            target = download_end_match.group("target").strip()
            pending_records = [record for record in downloads.values() if record.finish is None]
            if pending_records:
                record = pending_records[-1]
                record.finish = timestamp
                record.target = target
            continue

        document_finished_match = DOCUMENT_FINISHED_PATTERN.search(line)
        if document_finished_match:
            name = document_finished_match.group("doc")
            seconds = float(document_finished_match.group("seconds"))
            timing = _resolve_document(documents, name)
            timing.conversion_seconds = seconds
            continue

        document_timings_match = DOCUMENT_TIMINGS_PATTERN.search(line)
        if document_timings_match:
            name = document_timings_match.group("doc")
            parse_seconds = float(document_timings_match.group("parse"))
            extraction_seconds = float(document_timings_match.group("extraction"))
            timing = _resolve_document(documents, name)
            timing.parse_seconds = parse_seconds
            timing.extraction_seconds = extraction_seconds
            continue

        pipeline_total_match = PIPELINE_TOTAL_PATTERN.search(line)
        if pipeline_total_match:
            name = pipeline_total_match.group("paper")
            seconds = float(pipeline_total_match.group("seconds"))
            timing = _resolve_document(documents, name)
            timing.total_seconds = seconds

    return LogSummary(downloads=list(downloads.values()), documents=list(documents.values()))


def analyze_log_file(path: Path) -> LogSummary:
    """Load and analyze a log file located at ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        return _parse_lines(handle)


def main() -> None:
    """Entry point for the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect an extraction log to highlight slow downloads and document"
            " stages."
        )
    )
    parser.add_argument("logfile", type=Path, help="Path to the log file to analyze")
    args = parser.parse_args()

    summary = analyze_log_file(args.logfile)
    print(summary.format_report())


if __name__ == "__main__":
    main()
