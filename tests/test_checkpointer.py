"""Tests for session-scoped LangGraph checkpoint persistence (no LLM)."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.checkpointer import create_checkpointer
from src.agent.graph import get_graph_config
from src.agent.state import AgentState
from src.data.loader import load_cached_dataset
from src.data.store import DatasetStore
import src.data.store as store_module
from src.tools.dataset_tools import filter_by_category, reset_filter, sample_rows


def _close_checkpointer(checkpointer) -> None:
    if hasattr(checkpointer, "conn"):
        checkpointer.conn.close()


def _build_echo_graph(checkpointer):
    def echo_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        return {"messages": [AIMessage(content=f"seen:{last.content}")]}

    graph = StateGraph(AgentState)
    graph.add_node("echo", echo_node)
    graph.add_edge(START, "echo")
    graph.add_edge("echo", END)
    return graph.compile(checkpointer=checkpointer)


def test_get_graph_config_maps_session_id_to_thread_id():
    config = get_graph_config(session_id="my_session")
    assert config["configurable"]["thread_id"] == "my_session"
    assert config["recursion_limit"] >= 4


def test_different_session_ids_are_isolated():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.sqlite"
        checkpointer = create_checkpointer(db_path)
        compiled = _build_echo_graph(checkpointer)

        config_a = get_graph_config(session_id="session_a")
        config_b = get_graph_config(session_id="session_b")

        compiled.invoke({"messages": [HumanMessage(content="hello")]}, config=config_a)
        compiled.invoke({"messages": [HumanMessage(content="world")]}, config=config_b)

        state_a = compiled.get_state(config_a)
        state_b = compiled.get_state(config_b)

        assert len(state_a.values["messages"]) == 2
        assert len(state_b.values["messages"]) == 2
        assert "hello" in state_a.values["messages"][0].content
        assert "world" in state_b.values["messages"][0].content
        _close_checkpointer(checkpointer)


def test_same_session_id_accumulates_messages_across_invokes():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.sqlite"
        checkpointer = create_checkpointer(db_path)
        compiled = _build_echo_graph(checkpointer)

        config = get_graph_config(session_id="persist_test")
        compiled.invoke({"messages": [HumanMessage(content="turn1")]}, config=config)
        compiled.invoke({"messages": [HumanMessage(content="turn2")]}, config=config)

        state = compiled.get_state(config)
        human_messages = [
            m for m in state.values["messages"] if isinstance(m, HumanMessage)
        ]
        assert len(human_messages) == 2
        assert human_messages[0].content == "turn1"
        assert human_messages[1].content == "turn2"
        _close_checkpointer(checkpointer)


def test_session_restored_after_reopening_checkpointer():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.sqlite"
        config = get_graph_config(session_id="restart_test")

        checkpointer1 = create_checkpointer(db_path)
        compiled1 = _build_echo_graph(checkpointer1)
        compiled1.invoke({"messages": [HumanMessage(content="before_restart")]}, config=config)
        _close_checkpointer(checkpointer1)

        checkpointer2 = create_checkpointer(db_path)
        compiled2 = _build_echo_graph(checkpointer2)

        state = compiled2.get_state(config)
        human_messages = [
            m for m in state.values["messages"] if isinstance(m, HumanMessage)
        ]
        assert len(human_messages) == 1
        assert human_messages[0].content == "before_restart"
        _close_checkpointer(checkpointer2)


@pytest.fixture(autouse=True)
def setup_store():
    store_module._store = DatasetStore(load_cached_dataset())
    yield
    store_module._store = None


def test_sample_rows_offset_returns_disjoint_examples():
    reset_filter()
    json.loads(filter_by_category("SHIPPING"))
    first = json.loads(sample_rows(3, offset=0))
    second = json.loads(sample_rows(3, offset=3))

    assert first["offset"] == 0
    assert first["next_offset"] == 3
    assert second["offset"] == 3
    assert len(first["examples"]) == 3
    assert len(second["examples"]) == 3

    first_instructions = {e["instruction"] for e in first["examples"]}
    second_instructions = {e["instruction"] for e in second["examples"]}
    assert first_instructions.isdisjoint(second_instructions)
