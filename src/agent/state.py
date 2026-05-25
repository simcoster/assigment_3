"""LangGraph state definitions."""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

QueryRoute = Literal["structured", "unstructured", "profile_recall", "out_of_scope"]


class AgentState(TypedDict):
    """State carried through the analyst agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    route: QueryRoute | None
    router_reasoning: str | None
    iteration_count: int
    user_profile: dict | None
    turn_dataset_query: str | None
    turn_profile_update: dict | None
