"""LangGraph assembly for the Bitext data analyst agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.profile import (
    UserProfile,
    extract_turn_parts,
    format_profile_recall_answer,
    profile_from_state_dict,
    routing_question_from_extraction,
    to_prompt_block,
)
from src.agent.prompts import (
    DECLINE_MESSAGE,
    MAX_ITERATIONS_MESSAGE,
    STRUCTURED_SYSTEM_PROMPT,
    UNSTRUCTURED_SYSTEM_PROMPT,
)
from src.agent.recommender import format_recommendation_answer, recommend_next_query
from src.agent.router import classify_query
from src.agent.state import AgentState
from src.config import Settings, get_settings
from src.tools import ALL_TOOLS

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite import SqliteSaver


def build_agent_llm(settings: Settings) -> ChatOpenAI:
    """Create the main ReAct agent LLM."""
    return ChatOpenAI(
        model=settings.agent_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
        model_kwargs={"parallel_tool_calls": False},
    )


def _latest_user_question(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return str(messages[-1].content) if messages else ""


def _conversation_context(messages: list, max_turns: int = 4, max_chars: int = 2000) -> str | None:
    """Build a short summary of recent turns for router follow-up classification."""
    if len(messages) <= 1:
        return None

    lines: list[str] = []
    for message in messages[:-1]:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                calls = ", ".join(c["name"] for c in message.tool_calls)
                lines.append(f"Assistant: [called tools: {calls}]")
            elif message.content:
                text = str(message.content)
                lines.append(f"Assistant: {text[:400]}{'...' if len(text) > 400 else ''}")
        elif isinstance(message, ToolMessage):
            text = str(message.content)
            lines.append(f"Tool result: {text[:300]}{'...' if len(text) > 300 else ''}")

    if not lines:
        return None

    context = "\n".join(lines[-max_turns * 3 :])
    if len(context) > max_chars:
        context = context[-max_chars:]
    return context


def _system_prompt_with_profile(base_prompt: str, state: AgentState) -> str:
    profile = profile_from_state_dict(state.get("user_profile"))
    if not profile:
        return base_prompt
    block = to_prompt_block(profile)
    if not block:
        return base_prompt
    return f"{base_prompt}\n\n{block}"


def _has_tool_results(messages: list) -> bool:
    """Return True if the conversation already contains tool observations."""
    return any(isinstance(message, ToolMessage) for message in messages)


def _enforce_single_tool_call(response: AIMessage) -> AIMessage:
    """Keep only the first tool call; Nebius Llama models reject multiple per turn."""
    if not response.tool_calls or len(response.tool_calls) <= 1:
        return response
    return response.model_copy(update={"tool_calls": [response.tool_calls[0]]})


def _bind_agent_llm(settings: Settings, messages: list):
    """Bind tools with dynamic tool_choice: force tools first, then allow text answers."""
    tool_choice = "auto" if _has_tool_results(messages) else "any"
    return build_agent_llm(settings).bind_tools(ALL_TOOLS, tool_choice=tool_choice, parallel_tool_calls=False)


def build_graph(
    settings: Settings | None = None,
    checkpointer: SqliteSaver | None = None,
):
    """Build and compile the analyst agent graph."""
    settings = settings or get_settings()
    if not settings.nebius_api_key:
        raise ValueError(
            "NEBIUS_API_KEY is required. Copy .env.example to .env and set your API key."
        )
    tool_node = ToolNode(ALL_TOOLS)

    def split_turn_node(state: AgentState) -> dict:
        """Separate dataset question from profile facts before routing."""
        user_message = _latest_user_question(state["messages"])
        extraction = extract_turn_parts(user_message, settings)
        routing_q = routing_question_from_extraction(user_message, extraction)
        pending = (
            extraction.profile_update.model_dump()
            if extraction.profile_update.should_update
            else None
        )
        return {
            "turn_dataset_query": routing_q,
            "turn_profile_update": pending,
        }

    def router_node(state: AgentState) -> dict:
        messages = state["messages"]
        question = state.get("turn_dataset_query") or _latest_user_question(messages)
        context = _conversation_context(messages)
        classification = classify_query(question, settings, context=context)
        return {
            "route": classification.route,
            "router_reasoning": classification.reasoning,
            "iteration_count": 0,
        }

    def decline_node(_: AgentState) -> dict:
        return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}

    def profile_answer_node(state: AgentState) -> dict:
        profile = profile_from_state_dict(state.get("user_profile"))
        if profile is None:
            profile = UserProfile(user_id="default")
        content = format_profile_recall_answer(profile)
        return {"messages": [AIMessage(content=content)]}

    def recommendation_node(state: AgentState) -> dict:
        profile = profile_from_state_dict(state.get("user_profile"))
        if profile is None:
            profile = UserProfile(user_id="default")
        recommendation = recommend_next_query(state["messages"], profile, settings)
        content = format_recommendation_answer(recommendation)
        return {"messages": [AIMessage(content=content)]}

    def agent_node(state: AgentState) -> dict:
        iteration = state.get("iteration_count", 0)
        if iteration >= settings.max_iterations:
            return {
                "messages": [AIMessage(content=MAX_ITERATIONS_MESSAGE)],
                "iteration_count": iteration,
            }

        route = state.get("route") or "structured"
        base_prompt = (
            STRUCTURED_SYSTEM_PROMPT
            if route == "structured"
            else UNSTRUCTURED_SYSTEM_PROMPT
        )
        system_prompt = _system_prompt_with_profile(base_prompt, state)
        prompt_messages = [SystemMessage(content=system_prompt), *state["messages"]]
        agent_llm = _bind_agent_llm(settings, state["messages"])
        response = agent_llm.invoke(prompt_messages)
        response = _enforce_single_tool_call(response)
        return {
            "messages": [response],
            "iteration_count": iteration + 1,
        }

    def route_after_router(state: AgentState) -> str:
        route = state.get("route")
        if route == "out_of_scope":
            return "decline"
        if route == "profile_recall":
            return "profile_answer"
        if route == "recommendation":
            return "recommendation"
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
    graph.add_node("split_turn", split_turn_node)
    graph.add_node("router", router_node)
    graph.add_node("decline", decline_node)
    graph.add_node("profile_answer", profile_answer_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "split_turn")
    graph.add_edge("split_turn", "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "decline": "decline",
            "profile_answer": "profile_answer",
            "recommendation": "recommendation",
            "agent": "agent",
        },
    )
    graph.add_edge("decline", END)
    graph.add_edge("profile_answer", END)
    graph.add_edge("recommendation", END)
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


def get_graph_config(
    settings: Settings | None = None,
    session_id: str = "default",
    user_id: str = "default",
) -> dict:
    """Build invoke/stream config. session_id is the assignment's session ID."""
    settings = settings or get_settings()
    return {
        "recursion_limit": settings.max_iterations + 4,
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        },
    }
