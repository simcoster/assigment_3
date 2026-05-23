"""LangGraph assembly for the Bitext data analyst agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.prompts import (
    DECLINE_MESSAGE,
    MAX_ITERATIONS_MESSAGE,
    STRUCTURED_SYSTEM_PROMPT,
    UNSTRUCTURED_SYSTEM_PROMPT,
)
from src.agent.router import classify_query
from src.agent.state import AgentState
from src.config import Settings, get_settings
from src.data.store import get_dataset_store
from src.tools import ALL_TOOLS


def build_agent_llm(settings: Settings) -> ChatOpenAI:
    """Create the main ReAct agent LLM."""
    return ChatOpenAI(
        model=settings.agent_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
    )


def _latest_user_question(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return str(messages[-1].content) if messages else ""


def build_graph(settings: Settings | None = None):
    """Build and compile the analyst agent graph."""
    settings = settings or get_settings()
    if not settings.nebius_api_key:
        raise ValueError(
            "NEBIUS_API_KEY is required. Copy .env.example to .env and set your API key."
        )
    agent_llm = build_agent_llm(settings).bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    def router_node(state: AgentState) -> dict:
        question = _latest_user_question(state["messages"])
        get_dataset_store().reset_filter()
        classification = classify_query(question, settings)
        return {
            "route": classification.route,
            "router_reasoning": classification.reasoning,
            "iteration_count": 0,
        }

    def decline_node(_: AgentState) -> dict:
        return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}

    def agent_node(state: AgentState) -> dict:
        iteration = state.get("iteration_count", 0)
        if iteration >= settings.max_iterations:
            return {
                "messages": [AIMessage(content=MAX_ITERATIONS_MESSAGE)],
                "iteration_count": iteration,
            }

        route = state.get("route") or "structured"
        system_prompt = (
            STRUCTURED_SYSTEM_PROMPT
            if route == "structured"
            else UNSTRUCTURED_SYSTEM_PROMPT
        )
        prompt_messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = agent_llm.invoke(prompt_messages)
        return {
            "messages": [response],
            "iteration_count": iteration + 1,
        }

    def route_after_router(state: AgentState) -> str:
        if state.get("route") == "out_of_scope":
            return "decline"
        return "agent"

    def route_after_agent(state: AgentState) -> str:
        iteration = state.get("iteration_count", 0)
        if iteration >= settings.max_iterations:
            return END

        messages = state["messages"]
        if not messages:
            return END

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("decline", decline_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"decline": "decline", "agent": "agent"},
    )
    graph.add_edge("decline", END)
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def get_graph_config(settings: Settings | None = None) -> dict:
    """Return invoke/stream config including recursion limit."""
    settings = settings or get_settings()
    return {"recursion_limit": settings.max_iterations + 4}
