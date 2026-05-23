"""Pydantic input schemas for dataset analysis tools."""

from pydantic import BaseModel, Field


class EmptyInput(BaseModel):
    """No parameters."""


class CategoryInput(BaseModel):
    category: str = Field(
        description="Dataset category name, e.g. ACCOUNT, SHIPPING, FEEDBACK."
    )


class IntentInput(BaseModel):
    intent: str = Field(
        description="Dataset intent name in snake_case, e.g. get_refund, track_order."
    )


class ListIntentsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional category to restrict intent listing.",
    )


class SearchInstructionsInput(BaseModel):
    query: str = Field(
        description="Natural-language or keyword phrase to search in customer instructions."
    )
    max_rows: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum rows to include in the filtered working set.",
    )


class SampleRowsInput(BaseModel):
    n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of example rows to return from the active filtered subset.",
    )


class GetConversationTextsInput(BaseModel):
    max_rows: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum instruction/response pairs to return for summarization.",
    )
