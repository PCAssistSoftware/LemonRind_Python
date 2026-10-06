"""Asking the model which lasting facts about the user a conversation turn revealed.

This is a **separate, tool-free request** made after the reply has been shown, with its own tiny message
list (not the chat history). The server is told to answer in a fixed JSON shape (a *JSON schema*), so the
reply can be read by a program instead of being free text that has to be guessed at.

The prompt matters more than the code. A model happily "extracts" things that are not facts about the user:
a name found *inside a document it was shown*, the results of a web search, its own earlier reply. Those
would then be remembered as if the user had said them, so the prompt spells this out.

Python ideas used here:

* ``pydantic`` models to *parse and validate* what the model returned (it is untrusted text, even when
  the server constrains its shape).
* Catching ``ValidationError`` (which pydantic also raises for text that is not JSON at all): a bad
  answer means "no facts this turn", never a crash.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

SCHEMA_NAME = "extracted_facts"

# Written out by hand rather than generated from the pydantic class: it is small, and servers differ in how
# well they understand the ``$ref`` links pydantic would produce.
SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "pinned": {"type": "boolean"},
                },
                "required": ["content", "pinned"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You extract durable, worth-remembering facts about the USER from one exchange of a \
conversation with an AI assistant.

Only extract facts that would still be true and useful in future, unrelated conversations: stable \
preferences, identity details (name, role, location), ongoing projects, working style. Write each fact as \
one short, self-contained sentence about the user, for example "The user's name is Darren." or "The user \
lives in Manchester." Do not extract facts about the assistant, one-off requests, or transient details. If \
nothing durable was said, return an empty list.

Critical: only extract something the user stated in their own words, about themselves. Never extract \
information that merely appeared in an attached file, a knowledge base, a web search or page, a tool \
result, or the assistant's own reply, even if the reply repeats it. A name, company or other detail found \
INSIDE a document or web page is NOT the user's own unless the user explicitly claims it ("my name is X").

Set pinned=true only for core identity or preference facts worth having in every conversation (the user's \
name, role, home town, strong preferences). Set pinned=false for more specific facts that are only worth \
recalling when relevant."""


class ExtractedFact(BaseModel):
    content: str
    pinned: bool = False


class FactExtraction(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


def build_messages(user_text: str, reply_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'User said: "{user_text}"\nAssistant replied: "{reply_text}"',
        },
    ]


def parse_facts(text: str) -> list[ExtractedFact]:
    """Read the model's JSON answer. Anything unusable gives an empty list; blank facts are dropped."""
    try:
        extraction = FactExtraction.model_validate_json(text)
    except ValidationError:  # covers invalid JSON as well as a wrong shape
        return []
    return [fact for fact in extraction.facts if fact.content.strip()]
