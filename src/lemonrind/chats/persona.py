"""The persona: who the assistant is, who it is talking to, and how it should write.

Three kinds of settings (``PersonaSettings`` in ``config.py``) are turned into the *base* system prompt:

* **Identity**: who the assistant is ("You are Max, a dry-witted assistant with a dash of sarcasm.");
* **About you**: what it should know about the person it talks to ("I am Darren, a developer in Manchester.");
* **Writing style**: a tone, a length (verbosity), how much emoji, and any free-text custom instructions.

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
(terminal, web, scheduled jobs) builds the prompt the same way.

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
}
VERBOSITY_HINTS = {
    "Concise": "keep answers short; give the essential answer and stop",
    "Detailed": "give thorough answers with the reasoning and relevant detail",
}
EMOJI_HINTS = {
    "None": "do not use emoji",
    "Sparing": "use an emoji only occasionally, where it adds something",
    "Frequent": "use emoji freely to add personality",
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
    if custom := persona.custom_instructions.strip():
        lines.append(f"- Also: {custom}")
    return ("Writing style:\n" + "\n".join(lines)) if lines else ""


def build_system_prompt(settings: Settings) -> str:
    """The base system prompt: persona, behaviour prompt and writing style, in that order."""
    persona = settings.persona
    sections = []
    if identity := persona.identity.strip():
        sections.append(identity)
    if about := persona.about_user.strip():
        sections.append(f"About the person you are talking to:\n{about}")
    sections.append(settings.assistant.system_prompt.strip())
    if style := writing_style(persona):
        sections.append(style)
    return "\n\n".join(section for section in sections if section)
