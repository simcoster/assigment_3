"""LangChain tools for analyzing the Bitext customer service dataset."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import StructuredTool

from src.data.store import get_dataset_store
from src.tools.schemas import (
    CategoryInput,
    EmptyInput,
    GetConversationTextsInput,
    IntentInput,
    ListIntentsInput,
    SampleRowsInput,
    SearchInstructionsInput,
)


def _normalize_category(value: str) -> str:
    return value.strip().upper()


def _normalize_intent(value: str) -> str:
    return value.strip().lower()


def list_categories() -> str:
    """List all unique category values in the full Bitext dataset.

    Use this when the user asks what categories exist or wants an overview of
    top-level dataset groupings. Does not require a prior filter.
    """
    store = get_dataset_store()
    categories = sorted(store.full_dataframe["category"].dropna().unique().tolist())
    return json.dumps({"categories": categories, "count": len(categories)})


def list_intents(category: str | None = None) -> str:
    """List intent names in the dataset, optionally filtered by category.

    Use this to discover valid intent identifiers before calling filter_by_intent.
    Helpful when the user mentions a topic like 'refund' but not the exact intent key.
    """
    store = get_dataset_store()
    df = store.full_dataframe
    if category:
        df = df[df["category"] == _normalize_category(category)]
    intents = sorted(df["intent"].dropna().unique().tolist())
    return json.dumps(
        {
            "category_filter": _normalize_category(category) if category else None,
            "intents": intents,
            "count": len(intents),
        }
    )


def filter_by_category(category: str) -> str:
    """Filter the working dataset subset to rows matching a category.

    Use before count_rows, sample_rows, or get_conversation_texts when the user
    mentions a category such as SHIPPING or ACCOUNT. Resets any previous filter.
    """
    store = get_dataset_store()
    normalized = _normalize_category(category)
    mask = store.full_dataframe["category"] == normalized
    count = store.filter_mask(mask)
    return json.dumps(
        {
            "filter": "category",
            "value": normalized,
            "matching_rows": count,
            "message": "Active filter updated. Use count_rows or sample_rows next.",
        }
    )


def filter_by_intent(intent: str) -> str:
    """Filter the working dataset subset to rows matching an intent.

    Use after list_intents when you know the exact intent key (e.g. get_refund).
    Typical chain: filter_by_intent -> count_rows for 'how many' questions.
    """
    store = get_dataset_store()
    normalized = _normalize_intent(intent)
    mask = store.full_dataframe["intent"] == normalized
    count = store.filter_mask(mask)
    return json.dumps(
        {
            "filter": "intent",
            "value": normalized,
            "matching_rows": count,
            "message": "Active filter updated. Use count_rows or sample_rows next.",
        }
    )


def search_instructions(query: str, max_rows: int = 500) -> str:
    """Search customer instructions for a keyword/phrase and set the working filter.

    Use when the user describes a topic in natural language (e.g. 'money back',
    'complaint') and the exact intent name is unknown. Searches the instruction column.
    """
    store = get_dataset_store()
    pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
    mask = store.full_dataframe["instruction"].astype(str).str.contains(
        pattern, regex=True, na=False
    )
    indices = store.full_dataframe.index[mask].tolist()[:max_rows]
    count = store.set_working_indices(indices)
    sample_intents = (
        store.working_dataframe["intent"].value_counts().head(10).to_dict()
        if count
        else {}
    )
    return json.dumps(
        {
            "query": query,
            "matching_rows": count,
            "top_intents_in_matches": sample_intents,
            "message": "Active filter updated from instruction search.",
        }
    )


def count_rows() -> str:
    """Count rows in the active filtered subset, or the full dataset if no filter is set.

    Use after filter_by_category, filter_by_intent, or search_instructions for
    'how many' questions. If no prior filter was applied, counts the entire dataset.
    """
    store = get_dataset_store()
    count = len(store.working_dataframe)
    scope = "filtered_subset" if store.has_active_filter else "full_dataset"
    return json.dumps({"scope": scope, "count": count})


def sample_rows(n: int = 3, offset: int = 0) -> str:
    """Return example instruction/response pairs from the active filtered subset.

    Use after a filter tool when the user asks to see examples. Requires a prior
    filter unless sampling from the full dataset is acceptable. Use offset when
    the user asks for more examples after a previous sample_rows call.
    """
    store = get_dataset_store()
    subset_len = len(store.working_dataframe)
    df = store.working_dataframe.iloc[offset : offset + n]
    examples = [
        {
            "category": row["category"],
            "intent": row["intent"],
            "instruction": row["instruction"],
            "response": row["response"],
        }
        for _, row in df.iterrows()
    ]
    next_offset = offset + len(examples)
    return json.dumps(
        {
            "examples": examples,
            "returned": len(examples),
            "offset": offset,
            "next_offset": next_offset,
            "available_in_subset": subset_len,
        }
    )


def intent_distribution(category: str) -> str:
    """Return counts of each intent within a category.

    Use for questions like 'distribution of intents in ACCOUNT'. Does not modify
    the active working filter.
    """
    store = get_dataset_store()
    normalized = _normalize_category(category)
    subset = store.full_dataframe[store.full_dataframe["category"] == normalized]
    distribution = subset["intent"].value_counts().to_dict()
    return json.dumps(
        {
            "category": normalized,
            "total_rows": len(subset),
            "intent_distribution": distribution,
        }
    )


def get_conversation_texts(max_rows: int = 50) -> str:
    """Return instruction/response pairs from the active subset for summarization.

    Use on unstructured questions after filtering to the relevant slice. Returns at
    most max_rows pairs to keep context manageable.
    """
    store = get_dataset_store()
    df = store.working_dataframe.head(max_rows)
    conversations = [
        {
            "instruction": row["instruction"],
            "response": row["response"],
            "intent": row["intent"],
            "category": row["category"],
        }
        for _, row in df.iterrows()
    ]
    return json.dumps(
        {
            "conversations": conversations,
            "returned": len(conversations),
            "available_in_subset": len(store.working_dataframe),
            "note": "Summarize only from these texts; do not invent examples.",
        }
    )


def reset_filter() -> str:
    """Clear the active filter and revert to the full dataset for subsequent tools."""
    store = get_dataset_store()
    store.reset_filter()
    return json.dumps({"message": "Filter cleared.", "full_dataset_rows": len(store.full_dataframe)})


def _make_tool(func: Any, name: str, args_schema: type) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=(func.__doc__ or "").strip(),
        args_schema=args_schema,
    )


ALL_TOOLS = [
    _make_tool(list_categories, "list_categories", EmptyInput),
    _make_tool(list_intents, "list_intents", ListIntentsInput),
    _make_tool(filter_by_category, "filter_by_category", CategoryInput),
    _make_tool(filter_by_intent, "filter_by_intent", IntentInput),
    _make_tool(search_instructions, "search_instructions", SearchInstructionsInput),
    _make_tool(count_rows, "count_rows", EmptyInput),
    _make_tool(sample_rows, "sample_rows", SampleRowsInput),
    _make_tool(intent_distribution, "intent_distribution", CategoryInput),
    _make_tool(get_conversation_texts, "get_conversation_texts", GetConversationTextsInput),
    _make_tool(reset_filter, "reset_filter", EmptyInput),
]
