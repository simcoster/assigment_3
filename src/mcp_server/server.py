"""FastMCP server for Bitext dataset tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP

from src.data.loader import load_cached_dataset
from src.data.store import DatasetStore
from src.tools import dataset_tools as dt

import src.data.store as store_module

mcp = FastMCP("bitext-analyst")


def init_dataset() -> None:
    """Load the cached dataset into the singleton store."""
    if store_module._store is None:
        store_module._store = DatasetStore(load_cached_dataset())


@mcp.tool
def list_categories() -> str:
    """List all unique category values in the Bitext customer service dataset."""
    return dt.list_categories()


@mcp.tool
def list_intents(category: str | None = None) -> str:
    """List intent names in the dataset, optionally filtered by category (e.g. ACCOUNT)."""
    return dt.list_intents(category)


@mcp.tool
def filter_by_category(category: str) -> str:
    """Filter the working subset to rows matching a category (e.g. SHIPPING, ACCOUNT)."""
    return dt.filter_by_category(category)


@mcp.tool
def count_rows() -> str:
    """Count rows in the active filtered subset, or the full dataset if no filter is set."""
    return dt.count_rows()


@mcp.tool
def sample_rows(n: int = 3, offset: int = 0) -> str:
    """Return example instruction/response pairs from the active filtered subset."""
    return dt.sample_rows(n=n, offset=offset)


@mcp.tool
def reset_filter() -> str:
    """Clear the active filter and use the full dataset for subsequent tools."""
    return dt.reset_filter()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitext analyst MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="MCP transport (default: stdio for Cursor/Claude Desktop)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port when --transport http (default: 8000)",
    )
    args = parser.parse_args()

    try:
        init_dataset()
    except FileNotFoundError as exc:
        print(
            "Dataset not found. Run: uv run python scripts/download_data.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if args.transport == "http":
        mcp.run(transport="http", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
