"""Connectivity guard for package index availability during Docker builds."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_INDEX_URL = "https://pypi.org/simple/"


def _normalize_url(url: str) -> str:
    """Ensure the given index URL is properly normalized."""

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid package index URL: {url}")
    normalized = url if url.endswith("/") else f"{url}/"
    return normalized


def verify_package_index(url: str, timeout: float = 5.0) -> None:
    """Verify that the configured package index is reachable."""

    normalized = _normalize_url(url)
    request = urllib.request.Request(normalized, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            if response.status >= 400:
                raise RuntimeError(
                    f"Package index responded with HTTP {response.status}"
                )
    except urllib.error.URLError as exc:  # pragma: no cover - exercised via RuntimeError
        raise RuntimeError(f"Failed to reach package index: {normalized}") from exc


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the Python package index configured for the build is accessible."
        )
    )
    parser.add_argument(
        "--url",
        help=(
            "Package index URL to check. Defaults to the PIP_INDEX_URL environment"
            " variable or the public PyPI instance."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds when probing the package index (default: 5 seconds).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for command-line execution."""

    parser = _build_cli()
    args = parser.parse_args(argv)

    url = args.url or os.environ.get("PIP_INDEX_URL") or DEFAULT_INDEX_URL
    try:
        verify_package_index(url, timeout=args.timeout)
    except (RuntimeError, ValueError) as exc:
        message = (
            "Unable to reach the Python package index ("
            f"{url}). If you are behind a proxy or require an internal mirror, "
            "set PIP_INDEX_URL or supply --url when building."
        )
        print(message, file=sys.stderr)
        print(f"Detailed error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
