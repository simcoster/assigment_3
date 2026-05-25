"""Tests for user profile storage and merging (no LLM)."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.graph import get_graph_config
from src.agent.profile import (
    PreferenceReconciliation,
    ProfileStore,
    ProfileUpdate,
    TurnExtraction,
    UserProfile,
    apply_pending_profile_update,
    apply_profile_update,
    format_profile_recall_answer,
    reconcile_preferences,
    routing_question_from_extraction,
    to_prompt_block,
)
from src.config import Settings


def test_profile_store_round_trip(tmp_path: Path):
    store = ProfileStore(tmp_path)
    profile = UserProfile(user_id="alice", name="Alice")
    store.save(profile)

    loaded = store.load("alice")
    assert loaded.name == "Alice"
    assert loaded.user_id == "alice"


def test_profile_store_isolates_users(tmp_path: Path):
    store = ProfileStore(tmp_path)
    store.save(UserProfile(user_id="alice", name="Alice"))
    store.save(UserProfile(user_id="bob", name="Bob"))

    assert store.load("alice").name == "Alice"
    assert store.load("bob").name == "Bob"


def test_apply_profile_update_adds_preference():
    profile = UserProfile(user_id="alice")
    update = ProfileUpdate(
        name="Alice",
        add_preferences=["When calling sample_rows, use n=2 at most"],
        add_topics=["refunds"],
    )
    merged = apply_profile_update(profile, update)
    assert merged.name == "Alice"
    assert "refunds" in merged.topics_of_interest
    assert any("n=2" in p for p in merged.preferences)


def test_apply_profile_update_skips_when_should_update_false():
    profile = UserProfile(user_id="alice", name="Alice")
    update = ProfileUpdate(should_update=False, add_preferences=["ignored"])
    merged = apply_profile_update(profile, update)
    assert merged.preferences == []


def test_routing_question_from_extraction_uses_dataset_part():
    extraction = TurnExtraction(
        dataset_query="Show examples of the SHIPPING category",
        profile_update=ProfileUpdate(
            should_update=True,
            add_preferences=["When calling sample_rows, use n=5 at most"],
        ),
    )
    user_message = "Show me some examples of the SHIPPING category. I like 5 example max"
    assert routing_question_from_extraction(user_message, extraction) == (
        "Show examples of the SHIPPING category"
    )


@patch("src.agent.profile.build_profile_llm")
def test_apply_pending_profile_update_from_graph_state(mock_build_llm):
    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        PreferenceReconciliation(
            resolved_preferences=["When calling sample_rows, use n=5 at most"],
            replaced=[],
        )
    )
    profile = UserProfile(user_id="alice")
    state_values = {
        "turn_profile_update": ProfileUpdate(
            should_update=True,
            add_preferences=["When calling sample_rows, use n=5 at most"],
        ).model_dump()
    }
    settings = Settings(nebius_api_key="test")
    merged = apply_pending_profile_update(profile, state_values, settings)
    assert any("n=5" in p for p in merged.preferences)


def test_apply_pending_profile_update_noop_when_missing():
    profile = UserProfile(user_id="alice", name="Alice")
    settings = Settings(nebius_api_key="test")
    merged = apply_pending_profile_update(profile, {}, settings)
    assert merged.name == "Alice"


@patch("src.agent.profile.build_profile_llm")
def test_reconcile_preferences_replaces_more_specific(mock_build_llm):
    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        PreferenceReconciliation(
            resolved_preferences=["I like 2 examples"],
            replaced=["I like 5 examples"],
        )
    )
    settings = Settings(nebius_api_key="test")
    profile = UserProfile(
        user_id="alice",
        preferences=["I like 5 examples"],
    )
    update = ProfileUpdate(
        should_update=True,
        add_preferences=["I like 2 examples"],
    )
    merged = apply_profile_update(profile, update, settings=settings)
    assert merged.preferences == ["I like 2 examples"]


@patch("src.agent.profile.build_profile_llm")
def test_reconcile_preferences_expands_without_dropping(mock_build_llm):
    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = (
        PreferenceReconciliation(
            resolved_preferences=["I like cats and dogs"],
            replaced=[],
        )
    )
    settings = Settings(nebius_api_key="test")
    resolved, replaced = reconcile_preferences(
        ["I only like cats"],
        ["Actually I also like dogs"],
        settings,
    )
    assert resolved == ["I like cats and dogs"]
    assert replaced == []


@patch("src.agent.profile.build_profile_llm")
def test_extract_turn_parts_mixed_message(mock_build_llm):
    mock_llm = mock_build_llm.return_value
    mock_llm.with_structured_output.return_value.invoke.return_value = TurnExtraction(
        dataset_query="Show examples of the SHIPPING category",
        profile_update=ProfileUpdate(
            should_update=True,
            add_preferences=["When calling sample_rows, use n=5 at most"],
        ),
    )
    settings = Settings(nebius_api_key="test")
    from src.agent.profile import extract_turn_parts

    result = extract_turn_parts(
        "Show me some examples of the SHIPPING category. I like 5 example max",
        settings,
    )
    assert result.dataset_query == "Show examples of the SHIPPING category"
    assert result.profile_update.should_update is True
    assert result.profile_update.add_preferences


def test_to_prompt_block_includes_preference():
    profile = UserProfile(
        user_id="alice",
        name="Alice",
        preferences=["When calling sample_rows, use n=2 at most"],
    )
    block = to_prompt_block(profile)
    assert "Alice" in block
    assert "n=2" in block


def test_format_profile_recall_answer():
    profile = UserProfile(
        user_id="alice",
        name="Alice",
        preferences=["Max 2 examples per sample"],
    )
    answer = format_profile_recall_answer(profile)
    assert "Alice" in answer
    assert "2 examples" in answer


def test_get_graph_config_includes_user_id():
    config = get_graph_config(session_id="s1", user_id="alice")
    assert config["configurable"]["thread_id"] == "s1"
    assert config["configurable"]["user_id"] == "alice"


def test_profile_json_persisted_on_disk(tmp_path: Path):
    store = ProfileStore(tmp_path)
    store.save(
        UserProfile(
            user_id="alice",
            preferences=["concise answers"],
        )
    )
    raw = json.loads((tmp_path / "alice.json").read_text(encoding="utf-8"))
    assert "concise answers" in raw["preferences"]
