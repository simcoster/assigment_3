"""Streamlit chat UI for the Bitext data analyst agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.checkpointer import create_checkpointer
from src.agent.cli import checkpoint_messages_to_chat, format_agent_update, truncate_text
from src.agent.graph import build_graph, get_graph_config
from src.agent.profile import ProfileStore, UserProfile, apply_pending_profile_update
from src.agent.recommender import (
    is_recommendation_request,
    recommend_next_query,
)
from src.config import Settings, get_settings
from src.data.loader import load_cached_dataset
from src.data.store import DatasetStore


def _ensure_dataset() -> None:
    import src.data.store as store_module

    if store_module._store is None:
        store_module._store = DatasetStore(load_cached_dataset())


@st.cache_resource
def _load_agent() -> tuple[Settings, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = get_settings()
    _ensure_dataset()
    checkpointer = create_checkpointer(settings.checkpoint_path)
    graph = build_graph(settings, checkpointer=checkpointer)
    return settings, graph


def _render_reasoning_steps(steps: list[dict[str, Any]]) -> None:
    if not steps:
        st.caption("No reasoning steps recorded for this turn.")
        return

    for step in steps:
        step_type = step.get("type")
        if step_type == "router":
            st.markdown(f"**Router** → `{step.get('route')}`")
            if step.get("reasoning"):
                st.caption(step["reasoning"])
        elif step_type == "tool_call":
            st.markdown("**Tool call**")
            st.code(step.get("label", step.get("name", "unknown")), language="text")
        elif step_type == "observation":
            st.markdown("**Observation**")
            st.text(step.get("display") or truncate_text(str(step.get("content", ""))))
        elif step_type == "assistant" and step.get("content"):
            st.markdown("**Assistant (intermediate)**")
            st.markdown(step["content"])
        elif step_type == "split_turn":
            st.markdown("**Turn split**")
            st.json(
                {
                    "dataset_query": step.get("dataset_query"),
                    "profile_update": step.get("profile_update"),
                }
            )
        elif step_type == "verbose":
            st.markdown(f"**{step.get('node', 'debug')}**")
            st.write(step.get("message") or step.get("update"))


def _load_session_messages(graph: Any, config: dict) -> list[dict[str, Any]]:
    try:
        state = graph.get_state(config)
        prior = state.values.get("messages", []) if state else []
        return checkpoint_messages_to_chat(prior)
    except Exception:
        return []


def _sync_session_from_checkpoint(settings: Settings, graph: Any) -> None:
    config = get_graph_config(
        settings,
        session_id=st.session_state.session_id,
        user_id=st.session_state.user_id,
    )
    st.session_state.messages = _load_session_messages(graph, config)
    profile_store = ProfileStore(settings.profile_dir)
    st.session_state.user_profile = profile_store.load(st.session_state.user_id).model_dump()
    st.session_state._loaded_session_key = (
        st.session_state.session_id,
        st.session_state.user_id,
    )


def _init_session_state(settings: Settings, graph: Any) -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    session_key = (st.session_state.session_id, st.session_state.user_id)
    if st.session_state.get("_loaded_session_key") != session_key:
        _sync_session_from_checkpoint(settings, graph)


def main() -> None:
    st.set_page_config(page_title="Bitext Data Analyst", page_icon="💬", layout="wide")
    st.title("Bitext Data Analyst")
    st.caption("Ask questions about the customer service dataset. Reasoning steps appear under each reply.")

    try:
        settings, graph = _load_agent()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    _init_session_state(settings, graph)
    profile_store = ProfileStore(settings.profile_dir)

    with st.sidebar:
        st.header("Session")
        st.text_input(
            "Session ID",
            key="session_id",
            help="Use the same ID to resume a conversation across restarts. Press Enter to reload.",
        )
        st.text_input(
            "User ID",
            key="user_id",
            help="Persistent profile (preferences, name) is stored per user.",
        )
        st.caption("Changing session or user ID reloads checkpoint history on the next run.")

        st.divider()
        restored = len(st.session_state.messages)
        if restored:
            st.success(f"Restored {restored} message(s) from checkpoint.")
        else:
            st.info("No prior messages for this session.")

        profile = UserProfile.model_validate(st.session_state.user_profile)
        if profile.name or profile.preferences or profile.topics_of_interest:
            st.markdown("**User profile loaded**")
            if profile.name:
                st.write(f"Name: {profile.name}")
            if profile.preferences:
                st.write("Preferences:", ", ".join(profile.preferences))

    config = get_graph_config(
        settings,
        session_id=st.session_state.session_id,
        user_id=st.session_state.user_id,
    )
    user_profile = UserProfile.model_validate(st.session_state.user_profile)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            steps = message.get("steps")
            if message["role"] == "assistant" and steps:
                with st.expander("Reasoning steps", expanded=False):
                    _render_reasoning_steps(steps)

    if prompt := st.chat_input("Ask about the dataset…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if is_recommendation_request(prompt):
            try:
                state = graph.get_state(config)
                messages = state.values.get("messages", []) if state else []
            except Exception:
                messages = []

            try:
                rec = recommend_next_query(messages, user_profile, settings)
                with st.chat_message("assistant"):
                    st.markdown(
                        "Here is a suggested next question about the dataset "
                        "(I am **not** running it yet):"
                    )
                    st.markdown(f"> {rec.suggested_query}")
                    st.caption(rec.reasoning)
                    st.info(
                        "You can refine this suggestion or type a different question, "
                        "and I'll run that next."
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Suggested next question (not executed):\n\n"
                            f"> {rec.suggested_query}\n\n"
                            f"{rec.reasoning}"
                        ),
                        "steps": [],
                    }
                )
            except Exception as exc:
                with st.chat_message("assistant"):
                    st.error(f"Failed to generate a query suggestion: {exc}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"[recommender] Failed to generate suggestion: {exc}",
                        "steps": [],
                    }
                )
            return

        reasoning_steps: list[dict[str, Any]] = []
        final_answer: str | None = None

        with st.chat_message("assistant"):
            status = st.status("Running agent…", expanded=True)
            try:
                for chunk in graph.stream(
                    {
                        "messages": [HumanMessage(content=prompt)],
                        "user_profile": user_profile.model_dump(),
                    },
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, update in chunk.items():
                        new_steps = format_agent_update(node_name, update)
                        reasoning_steps.extend(new_steps)
                        for step in new_steps:
                            if step.get("type") == "router":
                                status.write(f"Router → `{step.get('route')}`")
                            elif step.get("type") == "tool_call":
                                status.write(f"Tool: `{step.get('name')}`")
                            elif step.get("type") == "observation":
                                status.write("Tool result received")

                        if node_name in {"agent", "decline", "profile_answer"}:
                            messages = update.get("messages", [])
                            if messages and hasattr(messages[-1], "content"):
                                content = messages[-1].content
                                if content and not getattr(messages[-1], "tool_calls", None):
                                    final_answer = str(content)

                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Error", state="error")
                st.error(f"Agent error: {exc}")
                final_answer = None

            if final_answer:
                st.markdown(final_answer)
                with st.expander("Reasoning steps", expanded=True):
                    _render_reasoning_steps(reasoning_steps)
            elif not final_answer:
                st.warning("No answer was produced for this turn.")

        if final_answer:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "steps": reasoning_steps,
                }
            )

            try:
                state = graph.get_state(config)
                if state and state.values:
                    user_profile = apply_pending_profile_update(
                        user_profile, state.values, settings
                    )
                    st.session_state.user_profile = user_profile.model_dump()
                    profile_store.save(user_profile)
            except Exception as exc:
                st.sidebar.warning(f"Profile update failed: {exc}")


if __name__ == "__main__":
    main()
