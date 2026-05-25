"""Per-user distilled profile storage and updates."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agent.prompts import (
    PREFERENCE_RECONCILE_SYSTEM_PROMPT,
    TURN_SPLIT_SYSTEM_PROMPT,
)
from src.config import Settings


class UserProfile(BaseModel):
    """Distilled facts about a user (not conversation history)."""

    user_id: str
    name: str | None = None
    topics_of_interest: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    updated_at: str = ""

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


class ProfileUpdate(BaseModel):
    """Structured deltas from distilling personal facts in a user turn."""

    name: str | None = Field(
        default=None,
        description="Set or replace display name; omit if unchanged.",
    )
    add_topics: list[str] = Field(default_factory=list)
    remove_topics: list[str] = Field(default_factory=list)
    add_preferences: list[str] = Field(default_factory=list)
    remove_preferences: list[str] = Field(default_factory=list)
    add_notes: list[str] = Field(default_factory=list)
    remove_notes: list[str] = Field(default_factory=list)
    should_update: bool = Field(
        default=True,
        description="False if this turn has no personal facts to store.",
    )


class PreferenceReconciliation(BaseModel):
    """Merged preference list after resolving conflicts."""

    resolved_preferences: list[str] = Field(
        default_factory=list,
        description="Full preference list after this turn.",
    )
    replaced: list[str] = Field(
        default_factory=list,
        description="Existing preferences removed as superseded or contradictory.",
    )


class TurnExtraction(BaseModel):
    """LLM split of one user message into dataset query vs profile facts."""

    dataset_query: str | None = Field(
        default=None,
        description="Dataset-related question only; null if none.",
    )
    profile_update: ProfileUpdate = Field(
        default_factory=ProfileUpdate,
        description="Personal facts to merge into the user profile.",
    )


class ProfileStore:
    """JSON file store: one profile per user_id."""

    def __init__(self, profile_dir: Path) -> None:
        self._profile_dir = profile_dir

    def _path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^\w\-]", "_", user_id) or "default"
        return self._profile_dir / f"{safe_id}.json"

    def load(self, user_id: str) -> UserProfile:
        path = self._path_for(user_id)
        if not path.exists():
            profile = UserProfile(user_id=user_id)
            profile.touch()
            return profile
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserProfile.model_validate(data)

    def save(self, profile: UserProfile) -> None:
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        profile.touch()
        path = self._path_for(profile.user_id)
        path.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )


def to_prompt_block(profile: UserProfile) -> str:
    """Format profile for injection into system prompts."""
    if not profile.name and not profile.topics_of_interest and not profile.preferences and not profile.notes:
        return ""

    lines = ["User profile (distilled facts, not the conversation log):"]
    if profile.name:
        lines.append(f"- Name: {profile.name}")
    if profile.topics_of_interest:
        lines.append(f"- Often asks about: {', '.join(profile.topics_of_interest)}")
    if profile.preferences:
        for pref in profile.preferences:
            lines.append(f"- Preference: {pref}")
    if profile.notes:
        for note in profile.notes:
            lines.append(f"- Note: {note}")
    lines.append(
        "Follow user preferences when choosing tool arguments (e.g. sample_rows n)."
    )
    return "\n".join(lines)


def format_profile_recall_answer(profile: UserProfile) -> str:
    """Natural-language answer for 'What do you remember about me?'"""
    if not profile.name and not profile.topics_of_interest and not profile.preferences and not profile.notes:
        return (
            "I don't have any saved information about you yet. "
            "Tell me your name or what you like to explore in the dataset and I'll remember it."
        )

    parts: list[str] = ["Here's what I remember about you:"]
    if profile.name:
        parts.append(f"- Your name is {profile.name}.")
    if profile.topics_of_interest:
        parts.append(
            f"- You often ask about: {', '.join(profile.topics_of_interest)}."
        )
    if profile.preferences:
        for pref in profile.preferences:
            parts.append(f"- {pref}")
    if profile.notes:
        for note in profile.notes:
            parts.append(f"- {note}")
    return "\n".join(parts)


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def apply_profile_update(profile: UserProfile, update: ProfileUpdate) -> UserProfile:
    """Merge structured deltas into a profile (pure function for tests)."""
    if not update.should_update:
        return profile

    data = profile.model_dump()
    if update.name is not None:
        data["name"] = update.name.strip() or None

    for field, add_key, remove_key in (
        ("topics_of_interest", "add_topics", "remove_topics"),
        ("preferences", "add_preferences", "remove_preferences"),
        ("notes", "add_notes", "remove_notes"),
    ):
        current = list(data[field])
        remove_set = {x.strip().lower() for x in getattr(update, remove_key)}
        current = [x for x in current if x.strip().lower() not in remove_set]
        current.extend(getattr(update, add_key))
        data[field] = _dedupe_preserve(current)

    merged = UserProfile.model_validate(data)
    merged.touch()
    return merged


def build_profile_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.router_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
    )


def extract_turn_parts(user_message: str, settings: Settings) -> TurnExtraction:
    """Split a user message into dataset query vs profile facts (LLM, no regex)."""
    llm = build_profile_llm(settings).with_structured_output(TurnExtraction)
    result = llm.invoke(
        [
            SystemMessage(content=TURN_SPLIT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    if isinstance(result, TurnExtraction):
        return result
    return TurnExtraction.model_validate(result)


def routing_question_from_extraction(
    user_message: str, extraction: TurnExtraction
) -> str:
    """Question text for the router after splitting."""
    if extraction.dataset_query and extraction.dataset_query.strip():
        return extraction.dataset_query.strip()
    return user_message


def profile_update_from_state(state_values: dict) -> ProfileUpdate | None:
    """Read pending profile update produced by split_turn node."""
    raw = state_values.get("turn_profile_update")
    if not raw:
        return None
    return ProfileUpdate.model_validate(raw)


def apply_pending_profile_update(
    profile: UserProfile, state_values: dict
) -> UserProfile:
    """Apply turn_profile_update from graph state if present."""
    update = profile_update_from_state(state_values)
    if update is None:
        return profile
    return apply_profile_update(profile, update)


def profile_from_state_dict(data: dict | None) -> UserProfile | None:
    if not data:
        return None
    return UserProfile.model_validate(data)
