"""Smoke tests for MCP server tool wrappers (no live transport)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_cached_dataset
from src.data.store import DatasetStore
import src.data.store as store_module
from src.mcp_server.server import (
    count_rows,
    filter_by_category,
    init_dataset,
    list_categories,
    list_intents,
    mcp,
    reset_filter,
    sample_rows,
)


@pytest.fixture(autouse=True)
def setup_store():
    store_module._store = DatasetStore(load_cached_dataset())
    yield
    store_module._store = None


def test_init_dataset_loads_store():
    store_module._store = None
    init_dataset()
    assert store_module._store is not None
    assert len(store_module._store.full_dataframe) > 0


def test_list_categories_wrapper():
    result = json.loads(list_categories())
    assert "categories" in result
    assert result["count"] >= 1


def test_list_intents_wrapper():
    result = json.loads(list_intents("ACCOUNT"))
    assert "intents" in result
    assert result["category_filter"] == "ACCOUNT"


def test_filter_count_sample_chain():
    reset_filter()
    json.loads(filter_by_category("SHIPPING"))
    counted = json.loads(count_rows())
    assert counted["scope"] == "filtered_subset"
    assert counted["count"] > 0
    sampled = json.loads(sample_rows(2, 0))
    assert len(sampled["examples"]) == 2


def test_mcp_registers_at_least_three_tools():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert "list_categories" in names
    assert "filter_by_category" in names
    assert "count_rows" in names
    assert len(names) >= 5
