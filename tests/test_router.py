"""Tests for query routing logic."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.router import QueryClassification, classify_query
from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        nebius_api_key="test-key",
        router_model="test-router",
        agent_model="test-agent",
    )


@patch("src.agent.router.build_router_llm")
def test_classify_structured(mock_build_llm, settings: Settings):
    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = QueryClassification(
        route="structured",
        reasoning="Count question about dataset.",
    )

    result = classify_query("How many refund requests?", settings)
    assert result.route == "structured"


@patch("src.agent.router.build_router_llm")
def test_classify_out_of_scope(mock_build_llm, settings: Settings):
    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = QueryClassification(
        route="out_of_scope",
        reasoning="General knowledge question.",
    )

    result = classify_query("Who is the president of France?", settings)
    assert result.route == "out_of_scope"


@patch("src.agent.router.build_router_llm")
def test_classify_recommendation(mock_build_llm, settings: Settings):
    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = QueryClassification(
        route="recommendation",
        reasoning="User wants a suggested next dataset question.",
    )

    result = classify_query("What should I query next?", settings)
    assert result.route == "recommendation"
