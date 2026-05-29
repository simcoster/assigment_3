"""Query recommendation helper based on episodic memory and user profile."""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agent.profile import UserProfile, to_prompt_block
from src.config import Settings


class QueryRecommendation(BaseModel):
    """Structured output for a suggested follow-up dataset question."""

    suggested_query: str = Field(
        description="One concrete dataset question the user might ask next."
    )
    reasoning: str = Field(
        description="Brief explanation of why this query is relevant."
    )


def build_recommender_llm(settings: Settings) -> ChatOpenAI:
    """Create the LLM used for query recommendations."""
    return ChatOpenAI(
        model=settings.router_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0.3,
    )


def _conversation_summary(messages: Iterable[BaseMessage], max_chars: int = 2000) -> str:
    """Summarize recent conversation turns for the recommender prompt."""
    lines: list[str] = []
    for message in messages:
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

    summary = "\n".join(lines[-40:])
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary


def format_recommendation_answer(recommendation: QueryRecommendation) -> str:
    """Natural-language answer for a suggested follow-up dataset question."""
    return (
        "Here is a suggested next question about the dataset "
        "(I am **not** running it yet):\n\n"
        f"> {recommendation.suggested_query}\n\n"
        f"{recommendation.reasoning}\n\n"
        "You can refine this suggestion or ask a different question in the next turn."
    )


def recommend_next_query(
    messages: list[BaseMessage],
    user_profile: UserProfile,
    settings: Settings,
) -> QueryRecommendation:
    """Suggest a follow-up dataset query without executing it."""
    from langchain_core.messages import HumanMessage as LCHumanMessage, SystemMessage

    convo = _conversation_summary(messages)
    profile_block = to_prompt_block(user_profile)

    system_prompt = (
        "You help a data analyst decide what to ask NEXT about the Bitext customer "
        "service dataset.\n\n"
        "You are given:\n"
        "- A short summary of the recent conversation (previous questions, answers, and tool results)\n"
        "- A distilled user profile with their interests and preferences\n\n"
        "Your job is to propose ONE concrete follow-up QUESTION about the dataset that would be a "
        "useful next step. The question should be something the existing tools could answer "
        "(counts, filters, examples, or summaries of the Bitext customer service conversations).\n\n"
        "Do not answer the question. Only propose the next question."
    )

    user_parts: list[str] = []
    if convo:
        user_parts.append(f"Recent conversation:\n{convo}")
    if profile_block:
        user_parts.append(f"\nUser profile:\n{profile_block}")
    if not user_parts:
        user_parts.append(
            "There is no prior conversation. Propose a good first question about this dataset."
        )

    user_content = "\n\n".join(user_parts)

    llm = build_recommender_llm(settings).with_structured_output(QueryRecommendation)
    result = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            LCHumanMessage(content=user_content),
        ],
        max_tokens=1000
    )
    if isinstance(result, QueryRecommendation):
        return result
    return QueryRecommendation.model_validate(result)

