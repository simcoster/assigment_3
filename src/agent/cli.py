"""CLI helpers for formatting and printing agent reasoning steps."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

ReasoningStep = dict[str, Any]
ReasoningStepType = Literal[
    "router",
    "tool_call",
    "observation",
    "assistant",
    "split_turn",
    "verbose",
]


def truncate_text(text: str, limit: int = 1200) -> str:
    """Truncate long tool observations for display."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def format_agent_update(
    node_name: str, update: dict[str, Any], verbose: bool = False
) -> list[ReasoningStep]:
    """Turn a graph stream update into structured reasoning steps for UI or logging."""
    steps: list[ReasoningStep] = []

    if node_name == "split_turn":
        if verbose:
            steps.append(
                {
                    "type": "split_turn",
                    "dataset_query": update.get("turn_dataset_query"),
                    "profile_update": update.get("turn_profile_update"),
                }
            )
        return steps

    if node_name == "router":
        steps.append(
            {
                "type": "router",
                "route": update.get("route"),
                "reasoning": update.get("router_reasoning"),
            }
        )
        return steps

    if node_name in {"decline", "profile_answer", "recommendation"}:
        for message in update.get("messages", []):
            if isinstance(message, AIMessage) and message.content:
                steps.append({"type": "assistant", "content": str(message.content)})
        return steps

    if node_name == "agent":
        for message in update.get("messages", []):
            if not isinstance(message, AIMessage):
                continue
            if message.tool_calls:
                for call in message.tool_calls:
                    args = call.get("args", {})
                    steps.append(
                        {
                            "type": "tool_call",
                            "name": call["name"],
                            "args": args,
                            "label": f"{call['name']}({json.dumps(args, ensure_ascii=False)})",
                        }
                    )
            elif message.content:
                steps.append({"type": "assistant", "content": str(message.content)})
            elif verbose:
                steps.append({"type": "verbose", "node": "agent", "message": repr(message)})
        return steps

    if node_name == "tools":
        for message in update.get("messages", []):
            if isinstance(message, ToolMessage):
                content = str(message.content)
                steps.append(
                    {
                        "type": "observation",
                        "content": content,
                        "display": truncate_text(content),
                    }
                )
        return steps

    if verbose:
        steps.append({"type": "verbose", "node": node_name, "update": update})
    return steps


def checkpoint_messages_to_chat(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Build chat history (user + final assistant turns) from checkpoint messages."""
    history: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
            history.append({"role": "assistant", "content": str(message.content), "steps": []})
    return history


def print_router_decision(route: str | None, reasoning: str | None) -> None:
    print(f"\n[router] route={route}")
    if reasoning:
        print(f"[router] reasoning: {reasoning}")


def _print_reasoning_step(step: ReasoningStep) -> None:
    step_type = step.get("type")
    if step_type == "router":
        print_router_decision(step.get("route"), step.get("reasoning"))
    elif step_type == "tool_call":
        print(f"\n[tool_call] {step.get('label')}")
    elif step_type == "observation":
        print(f"[observation] {step.get('display')}")
    elif step_type == "assistant":
        print(f"\n[assistant] {step.get('content')}")
    elif step_type == "split_turn":
        print(f"\n[split_turn] dataset_query={step.get('dataset_query')!r}")
        if step.get("profile_update"):
            print(f"[split_turn] profile_update={step.get('profile_update')}")
    elif step_type == "verbose":
        node = step.get("node", "?")
        if "message" in step:
            print(f"\n[{node}] {step['message']}")
        else:
            print(f"\n[{node}] {step.get('update')}")


def print_agent_update(node_name: str, update: dict[str, Any], verbose: bool = False) -> None:
    for step in format_agent_update(node_name, update, verbose=verbose):
        _print_reasoning_step(step)
