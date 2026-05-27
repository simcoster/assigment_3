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
    is_recommendation_request,
    recommend_next_query,
)
from src.agent.profile import UserProfile  # noqa: E402
from src.config import Settings  # noqa: E402


def test_is_recommendation_request_matches_key_phrases() -> None:
    assert is_recommendation_request("What should I query next?")
    assert is_recommendation_request("Can you suggest a query for me?")
    assert not is_recommendation_request("How many refund requests?")


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

