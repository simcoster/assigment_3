"""CLI helpers for printing agent reasoning steps."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


def _truncate(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def print_router_decision(route: str | None, reasoning: str | None) -> None:
    print(f"\n[router] route={route}")
    if reasoning:
        print(f"[router] reasoning: {reasoning}")


def print_agent_update(node_name: str, update: dict[str, Any], verbose: bool = False) -> None:
    if node_name == "split_turn":
        if verbose:
            print(f"\n[split_turn] dataset_query={update.get('turn_dataset_query')!r}")
            if update.get("turn_profile_update"):
                print(f"[split_turn] profile_update={update.get('turn_profile_update')}")
        return

    if node_name == "router":
        print_router_decision(update.get("route"), update.get("router_reasoning"))
        return

    if node_name in {"decline", "profile_answer"}:
        messages = update.get("messages", [])
        for message in messages:
            if isinstance(message, AIMessage):
                print(f"\n[assistant] {message.content}")
        return

    if node_name == "agent":
        messages = update.get("messages", [])
        for message in messages:
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    for call in message.tool_calls:
                        args = call.get("args", {})
                        print(f"\n[tool_call] {call['name']}({json.dumps(args, ensure_ascii=False)})")
                elif message.content:
                    print(f"\n[assistant] {message.content}")
                elif verbose:
                    print(f"\n[agent] {message}")
        return

    if node_name == "tools":
        messages = update.get("messages", [])
        for message in messages:
            if isinstance(message, ToolMessage):
                print(f"[observation] {_truncate(str(message.content))}")
        return

    if verbose:
        print(f"\n[{node_name}] {update}")
