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

from src.agent.cli import print_agent_update
from src.agent.graph import build_graph, get_graph_config
from src.config import get_settings
from src.data.loader import load_cached_dataset
from src.data.store import DatasetStore, get_dataset_store


def _ensure_dataset() -> None:
    """Load dataset into the singleton store."""
    import src.data.store as store_module

    if store_module._store is None:
        store_module._store = DatasetStore(load_cached_dataset())
    

def run_repl(verbose: bool = False) -> None:
    """Run the interactive question loop."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = get_settings()
    _ensure_dataset()
    graph = build_graph(settings)
    config = get_graph_config(settings)

    print("Bitext Data Analyst Agent")
    print("Ask questions about the customer service dataset. Type 'exit' or 'quit' to stop.\n")

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

        get_dataset_store().reset_filter()
        final_answer: str | None = None

        for chunk in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                print_agent_update(node_name, update, verbose=verbose)
                if node_name in {"agent", "decline"}:
                    messages = update.get("messages", [])
                    if messages and hasattr(messages[-1], "content"):
                        content = messages[-1].content
                        if content and not getattr(messages[-1], "tool_calls", None):
                            final_answer = str(content)

        if final_answer:
            print(f"\n=== Final Answer ===\n{final_answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitext customer service data analyst")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full graph update payloads",
    )
    args = parser.parse_args()
    run_repl(verbose=args.verbose)


if __name__ == "__main__":
    main()
