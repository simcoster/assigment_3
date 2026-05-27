"""Interactive CLI for the Bitext data analyst agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Ensure project root is on sys.path when running as `python main.py`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.checkpointer import create_checkpointer
from src.agent.cli import print_agent_update
from src.agent.graph import build_graph, get_graph_config
from src.agent.profile import ProfileStore, apply_pending_profile_update
from src.config import get_settings
from src.data.loader import load_cached_dataset
from src.data.store import DatasetStore


def _ensure_dataset() -> None:
    """Load dataset into the singleton store."""
    import src.data.store as store_module

    if store_module._store is None:
        store_module._store = DatasetStore(load_cached_dataset())


def run_repl(
    verbose: bool = False,
    session_id: str = "default",
    user_id: str = "default",
) -> None:
    """Run the interactive question loop."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = get_settings()
    _ensure_dataset()

    checkpointer = create_checkpointer(settings.checkpoint_path)
    graph = build_graph(settings, checkpointer=checkpointer)
    config = get_graph_config(settings, session_id=session_id, user_id=user_id)

    profile_store = ProfileStore(settings.profile_dir)
    user_profile = profile_store.load(user_id)

    print("Bitext Data Analyst Agent")
    print(f"Session: {session_id}")
    print(f"User: {user_id}")
    print("Ask questions about the customer service dataset. Type 'exit' or 'quit' to stop.\n")

    try:
        state = graph.get_state(config)
        prior_messages = state.values.get("messages", []) if state else []
        if prior_messages:
            print(
                f"Restored {len(prior_messages)} message(s) from this session.\n"
            )
    except Exception:
        pass

    if user_profile.name or user_profile.preferences or user_profile.topics_of_interest:
        print("Loaded user profile from disk.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            final_answer: str | None = None

            for chunk in graph.stream(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_profile": user_profile.model_dump(),
                },
                config=config,
                stream_mode="updates",
            ):
                for node_name, update in chunk.items():
                    print_agent_update(node_name, update, verbose=verbose)
                    if node_name in {"agent", "decline", "profile_answer", "recommendation"}:
                        messages = update.get("messages", [])
                        if messages and hasattr(messages[-1], "content"):
                            content = messages[-1].content
                            if content and not getattr(messages[-1], "tool_calls", None):
                                final_answer = str(content)

            if final_answer:
                print(f"\n=== Final Answer ===\n{final_answer}\n")

            try:
                state = graph.get_state(config)
                if state and state.values:
                    user_profile = apply_pending_profile_update(
                        user_profile, state.values, settings
                    )
                    profile_store.save(user_profile)
            except Exception as exc:
                print(f"[profile] Warning: could not update profile: {exc}")
    finally:
        profile_store.save(user_profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitext customer service data analyst")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full graph update payloads",
    )
    parser.add_argument(
        "--session",
        "-s",
        default="default",
        help="Session ID for conversation memory (restored across restarts)",
    )
    parser.add_argument(
        "--user",
        "-u",
        default="default",
        help="User ID for persistent profile (separate from session history)",
    )
    args = parser.parse_args()
    run_repl(verbose=args.verbose, session_id=args.session, user_id=args.user)


if __name__ == "__main__":
    main()
