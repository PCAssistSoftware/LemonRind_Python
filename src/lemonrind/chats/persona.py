"""The persona: who the assistant is, who it is talking to, and how it should write.

Three kinds of settings (``PersonaSettings`` in ``config.py``) are turned into the *base* system prompt:

* **Identity**: who the assistant is ("You are Max, a dry-witted assistant with a dash of sarcasm.");
* **About you**: what it should know about the person it talks to ("I am Darren, a developer in Manchester.");
* **Writing style**: a tone, a length, how much emoji, a vocabulary level, formatting and habits to avoid (switches), and
  any free-text custom instructions.

The order matters, because a model reads top to bottom and later text can override earlier text:

    Identity
    About you ("About the person you are talking to: ...")
    the base system prompt (how to behave)
    Writing style (bullet points)

"Who am I, and who am I talking to" reads naturally before "how should I behave", and the style notes come last so they
are the freshest instruction. Anything left at its default is **left out entirely**: an unconfigured persona adds nothing
to the prompt (and so costs no tokens). The rest of the system prompt (pinned memories, the summary of a long chat) is
added after this by the ``Conversation``.

This is a plain function with no settings file, model or screen involved, so it is easy to test and every front end
builds the prompt the same way. A scheduled job leaves the writing style out (``include_style=False``): a report should
not come back in chat slang, or in one sentence, because that is how you like to chat.

Python ideas used here:

* ``dict`` lookups with ``.get`` for turning a choice into a sentence; building a list and joining it, skipping empties.
* Reading a pydantic model's fields; ``Literal`` types for a fixed set of choices (in ``config.py``).
"""

from __future__ import annotations

from lemonrind.config import PersonaSettings, Settings

# What each choice means, written for the model. A bare word like "Warm" is open to interpretation; a short
# description is much more reliable, especially with small local models.
TONE_HINTS = {
    "Casual": "relaxed and conversational, like talking to a friend",
    "Formal": "professional, polite and precise",
    "Warm": "friendly, kind and encouraging",
    "Concise": "brisk and businesslike, with no small talk",
    "Direct": "straightforward: say what matters first, with no padding or hedging",
    "Chat-like": "short, informal messages, as if texting on a messaging app",
}
VERBOSITY_HINTS = {
    "Concise": "keep answers short; give the essential answer and stop",
    "Detailed": "give thorough answers with the reasoning and relevant detail",
    "One paragraph": "answer in a single paragraph",
    "One sentence": "answer in one sentence",
}
EMOJI_HINTS = {
    "None": "do not use emoji",
    "Sparing": "use an emoji only occasionally, where it adds something",
    "Frequent": "use emoji freely to add personality",
}
VOCABULARY_HINTS = {
    "Simple": "use simple, everyday words and short sentences",
    "Academic": "use precise, academic vocabulary",
}


def writing_style(persona: PersonaSettings) -> str:
    """The "Writing style" block, or ``""`` if every field is at its default."""
    lines = []
    if hint := TONE_HINTS.get(persona.tone):
        lines.append(f"- Tone: {persona.tone}: {hint}.")
    if hint := VERBOSITY_HINTS.get(persona.verbosity):
        lines.append(f"- Length: {persona.verbosity}: {hint}.")
    if hint := EMOJI_HINTS.get(persona.emoji_usage):
        lines.append(f"- Emoji: {persona.emoji_usage}: {hint}.")
    if hint := VOCABULARY_HINTS.get(persona.vocabulary):
        lines.append(f"- Vocabulary: {persona.vocabulary}: {hint}.")
    if formatting := _formatting_rule(persona):
        lines.append(f"- Formatting: {formatting}.")
    if persona.no_flattery:
        lines.append(
            '- Do not open with praise or agreement such as "You\'re absolutely right" or "Great question".'
        )
    if persona.no_em_dashes:
        lines.append("- Do not use em dashes (the long dash); use commas or full stops instead.")
    if custom := persona.custom_instructions.strip():
        lines.append(f"- Also: {custom}")
    return ("Writing style:\n" + "\n".join(lines)) if lines else ""


def _formatting_rule(persona: PersonaSettings) -> str:
    """One sentence about formatting, or ``""``. "No Markdown" already covers headers, tables and lists."""
    if persona.no_markdown:
        return "do not use Markdown (no bold, headers, tables or lists); write plain text"
    parts = []
    if persona.no_headers:
        parts.append("do not use headers")
    if persona.no_tables:
        parts.append("do not use tables")
    if persona.no_lists:
        parts.append("avoid bullet and numbered lists; write in sentences")
    return "; ".join(parts)


def build_system_prompt(settings: Settings, *, include_style: bool = True) -> str:
    """The base system prompt: persona, behaviour prompt and writing style, in that order.

    ``include_style=False`` leaves the writing style out (scheduled jobs, which are not a chat with you).
    """
    persona = settings.persona
    sections = []
    if identity := persona.identity.strip():
        sections.append(identity)
    if about := persona.about_user.strip():
        sections.append(f"About the person you are talking to:\n{about}")
    sections.append(settings.assistant.system_prompt.strip())
    if include_style and (style := writing_style(persona)):
        sections.append(style)
    return "\n\n".join(section for section in sections if section)
