"""Query classification router."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agent.prompts import ROUTER_SYSTEM_PROMPT
from src.config import Settings


class QueryClassification(BaseModel):
    """Structured output for query routing."""

    route: Literal["structured", "unstructured", "profile_recall", "out_of_scope"] = Field(
        description="Query type: structured, unstructured, profile_recall, or out_of_scope."
    )
    reasoning: str = Field(description="Brief justification for the chosen route.")


def build_router_llm(settings: Settings) -> ChatOpenAI:
    """Create the router LLM with structured classification output."""
    return ChatOpenAI(
        model=settings.router_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
    )


def classify_query(
    question: str,
    settings: Settings,
    *,
    context: str | None = None,
) -> QueryClassification:
    """Classify a user question before tool selection."""
    llm = build_router_llm(settings).with_structured_output(QueryClassification)
    if context:
        user_content = (
            f"Recent conversation:\n{context}\n\n"
            f"Latest user message to classify:\n{question}"
        )
    else:
        user_content = question
    result = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
    )
    if isinstance(result, QueryClassification):
        return result
    return QueryClassification.model_validate(result)
