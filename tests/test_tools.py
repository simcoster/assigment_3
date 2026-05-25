"""Unit tests for dataset tools (no LLM required)."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_cached_dataset
from src.data.store import DatasetStore
import src.data.store as store_module
from src.tools.dataset_tools import (
    count_rows,
    filter_by_category,
    filter_by_intent,
    intent_distribution,
    list_categories,
    list_intents,
    reset_filter,
    sample_rows,
    search_instructions,
)


@pytest.fixture(autouse=True)
def setup_store():
    store_module._store = DatasetStore(load_cached_dataset())
    yield
    store_module._store = None


def test_list_categories():
    result = json.loads(list_categories())
    assert "ACCOUNT" in result["categories"]
    assert result["count"] >= 1


def test_refund_count_chain():
    reset_filter()
    intents = json.loads(list_intents())
    assert "get_refund" in intents["intents"]

    filtered = json.loads(filter_by_intent("get_refund"))
    assert filtered["matching_rows"] > 0

    counted = json.loads(count_rows())
    assert counted["scope"] == "filtered_subset"
    assert counted["count"] == filtered["matching_rows"]


def test_category_filter_and_sample():
    reset_filter()
    json.loads(filter_by_category("SHIPPING"))
    examples = json.loads(sample_rows(2))
    assert len(examples["examples"]) == 2
    assert examples["examples"][0]["category"] == "SHIPPING"


def test_intent_distribution():
    result = json.loads(intent_distribution("ACCOUNT"))
    assert result["category"] == "ACCOUNT"
    assert result["total_rows"] > 0
    assert isinstance(result["intent_distribution"], dict)


def test_search_instructions():
    reset_filter()
    result = json.loads(search_instructions("money back"))
    assert result["matching_rows"] >= 0


def test_sample_rows_pagination():
    reset_filter()
    json.loads(filter_by_category("ACCOUNT"))
    page1 = json.loads(sample_rows(2, offset=0))
    page2 = json.loads(sample_rows(2, offset=2))
    assert page1["next_offset"] == 2
    assert page2["offset"] == 2
    ids1 = {(e["instruction"], e["response"]) for e in page1["examples"]}
    ids2 = {(e["instruction"], e["response"]) for e in page2["examples"]}
    assert ids1.isdisjoint(ids2)
