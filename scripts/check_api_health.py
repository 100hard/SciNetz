#!/usr/bin/env python3
"""CLI utility to verify that the SciNets API is reachable."""
from __future__ import annotations

import argparse
import logging
import sys

from backend.app.utils.api_health import check_api_health


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the health check utility.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_url",
        nargs="?",
        default="http://localhost:8000",
        help="Base URL for the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Request timeout in seconds (default: 5.0)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the CLI health check utility.

    Returns:
        int: Exit status code where ``0`` indicates success.
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    result = check_api_health(args.base_url, timeout=args.timeout)

    if result.ok:
        payload_repr = result.payload if result.payload is not None else "<no payload>"
        print(
            "API health check succeeded",
            f"status={result.status_code}",
            f"latency_ms={result.latency_ms:.2f}" if result.latency_ms is not None else "latency_ms=unknown",
            f"payload={payload_repr}",
        )
        return 0

    print("API health check failed:", result.detail, file=sys.stderr)
    if result.status_code is not None:
        print(f"Status code: {result.status_code}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
