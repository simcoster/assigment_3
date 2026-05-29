"""Tests for query recommendation helper."""

import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.recommender import (  # noqa: E402
    QueryRecommendation,
    format_recommendation_answer,
    format_refinement_answer,
    pending_from_state,
    recommend_next_query,
    refine_suggestion,
)
from src.agent.profile import UserProfile  # noqa: E402
from src.config import Settings  # noqa: E402


def test_format_recommendation_answer_includes_suggestion() -> None:
    rec = QueryRecommendation(
        suggested_query="How many SHIPPING conversations mention delays?",
        reasoning="Shipping was discussed earlier.",
    )
    answer = format_recommendation_answer(rec)
    assert "SHIPPING" in answer
    assert "not** running it yet" in answer


def test_format_refinement_answer_asks_for_confirmation() -> None:
    rec = QueryRecommendation(
        suggested_query="Show 5 examples from the REFUND category.",
        reasoning="Examples match the user's refinement.",
    )
    answer = format_refinement_answer(rec)
    assert "Show 5 examples from the REFUND category." in answer
    assert "Should I go ahead?" in answer


def test_pending_from_state_parses_recommendation() -> None:
    pending = pending_from_state(
        {
            "suggested_query": "How many SHIPPING conversations are there?",
            "reasoning": "Shipping was discussed earlier.",
        }
    )
    assert pending is not None
    assert "SHIPPING" in pending.suggested_query


@patch("src.agent.recommender.build_recommender_llm")
def test_refine_suggestion_uses_llm(mock_build_llm: object) -> None:
    settings = Settings(
        nebius_api_key="test-key",
        router_model="test-router",
        agent_model="test-agent",
    )
    pending = QueryRecommendation(
        suggested_query="Show intent distribution in the REFUND category.",
        reasoning="Follow-up on refunds.",
    )
    messages = [
        HumanMessage(content="What should I query next?"),
        AIMessage(content="Suggested distribution query."),
        HumanMessage(content="I'd rather see examples instead."),
    ]

    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = QueryRecommendation(
        suggested_query="Show 5 examples from the REFUND category.",
        reasoning="User asked for examples instead of distribution.",
    )

    rec = refine_suggestion(
        pending,
        "I'd rather see examples instead.",
        messages,
        UserProfile(user_id="alice"),
        settings,
    )
    assert "examples" in rec.suggested_query.lower()


@patch("src.agent.recommender.build_recommender_llm")
def test_recommend_next_query_uses_llm(mock_build_llm: object) -> None:
    settings = Settings(
        nebius_api_key="test-key",
        router_model="test-router",
        agent_model="test-agent",
    )
    user_profile = UserProfile(user_id="alice", topics_of_interest=["refunds"])
    messages = [
        HumanMessage(content="How many refund requests did we get?"),
        AIMessage(content="There were 42 refund requests."),
    ]

    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = QueryRecommendation(
        suggested_query="Show me examples of REFUND conversations.",
        reasoning="They just asked about refund counts, so examples are a natural follow-up.",
    )

    rec = recommend_next_query(messages, user_profile, settings)
    assert isinstance(rec, QueryRecommendation)
    assert "REFUND" in rec.suggested_query.upper()

