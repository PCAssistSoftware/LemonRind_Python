"""The persona: who the assistant is and how it writes, composed into the system prompt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nicegui.testing import User
from pydantic import ValidationError

from lemonrind.chats import ChatRepository, Conversation, build_system_prompt
from lemonrind.chats.persona import writing_style
from lemonrind.config import PersonaSettings, Settings
from lemonrind.modules import build_registry
from lemonrind.storage import Database
from lemonrind.terminal.chatloop import ChatLoop
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.test_chatloop import FakeClient, ScriptedConsole
from tests.test_scheduler import FakeLemonade, a_job, make_repo, make_runner, reply
from tests.test_webui import FakeLemonade as WebLemonade
from tests.test_webui import make_context

BASE = "You are a helpful local AI assistant running on the user's own machine."


def settings_with(**persona) -> Settings:
    settings = Settings()
    for name, value in persona.items():
        setattr(settings.persona, name, value)
    return settings


# --- composing the prompt -----------------------------------------------------------------------------------------------


def test_an_unconfigured_persona_adds_nothing():
    assert build_system_prompt(Settings()) == BASE
    assert writing_style(PersonaSettings()) == ""


def test_identity_and_about_you_come_first_then_the_base_prompt_then_the_style():
    settings = settings_with(
        identity="You are Max, a dry-witted assistant.",
        about_user="I am Darren, a developer in Manchester.",
        tone="Casual",
        emoji_usage="None",
    )

    prompt = build_system_prompt(settings)

    assert (
        prompt.index("You are Max")
        < prompt.index("About the person you are talking to")
        < prompt.index(BASE)
    )
    assert prompt.index(BASE) < prompt.index("Writing style:")
    assert "I am Darren, a developer in Manchester." in prompt
    assert prompt.count("\n\n") == 3  # four blocks, each separated by one blank line


def test_each_style_choice_becomes_a_sentence_the_model_can_follow():
    style = writing_style(
        PersonaSettings(
            tone="Warm",
            verbosity="Detailed",
            emoji_usage="Sparing",
            custom_instructions="  Answer in British English.  ",
        )
    )

    assert style == (
        "Writing style:\n"
        "- Tone: Warm: friendly, kind and encouraging.\n"
        "- Length: Detailed: give thorough answers with the reasoning and relevant detail.\n"
        "- Emoji: Sparing: use an emoji only occasionally, where it adds something.\n"
        "- Also: Answer in British English."
    )


def test_default_choices_and_blank_text_are_left_out_entirely():
    settings = settings_with(
        identity="   ", about_user="\n", tone="Default", custom_instructions=" "
    )

    assert build_system_prompt(settings) == BASE
    assert "Tone" in writing_style(
        PersonaSettings(tone="Formal")
    ) and "Length" not in writing_style(PersonaSettings(tone="Formal"))


def test_the_prompt_is_built_from_the_current_base_prompt_setting():
    settings = settings_with(identity="You are Max.")
    settings.assistant.system_prompt = "Be brief."

    assert build_system_prompt(settings) == "You are Max.\n\nBe brief."


# --- validation ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["tone", "verbosity", "emoji_usage"])
def test_a_choice_outside_the_allowed_list_is_rejected(field):
    with pytest.raises(ValidationError):
        setattr(PersonaSettings(), field, "Pirate")


def test_a_hand_edited_settings_file_with_a_typo_fails_clearly_and_a_good_one_loads(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"persona": {"tone": "Casul"}}), encoding="utf-8")
    with pytest.raises(ValidationError, match="tone"):
        Settings.load(path)

    path.write_text(
        json.dumps({"persona": {"tone": "Casual", "identity": "You are Max."}}), encoding="utf-8"
    )
    assert Settings.load(path).persona.tone == "Casual"


# --- the conversation, the terminal and the web ------------------------------------------------------------------------------


class Quiet:
    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    async def stream_chat(self, messages, model, **kwargs):
        from lemonrind.lemonade import Finished, TextDelta
        from lemonrind.lemonade.events import RequestStats

        self.requests.append([dict(m) for m in messages])
        yield TextDelta("ok")
        yield Finished(RequestStats(), "stop")


async def test_the_model_is_sent_the_persona_and_the_full_prompt_can_be_inspected():
    settings = settings_with(identity="You are Max.", tone="Direct")
    client = Quiet()
    conversation = Conversation(
        client=client,
        repo=ChatRepository(Database(":memory:")),
        system_prompt=build_system_prompt(settings),
        model="m",
    )

    await conversation.send("hi")
    shown = await conversation.current_system_prompt()

    assert (
        client.requests[0][0]["content"].startswith("You are Max.\n\n")
        and "Tone: Direct" in client.requests[0][0]["content"]
    )
    assert shown == client.requests[0][0]["content"]


async def test_the_terminal_shows_and_changes_the_persona_and_saves_it(tmp_path: Path):
    settings = Settings()
    settings_file = tmp_path / "settings.json"
    console = ScriptedConsole(
        [
            "/persona",
            "/persona identity You are Max, dry-witted.",
            "/persona tone Casual",
            "/persona tone Pirate",
            "/persona bogus x",
            "/prompt",
            "/persona identity",
            "/persona tone",
            "/quit",
        ]
    )
    loop = ChatLoop(
        client=FakeClient(),  # type: ignore[arg-type]
        settings=settings,
        settings_file=settings_file,
        repo=ChatRepository(Database(":memory:")),
        console=console,
        model="m",
    )

    await loop.run(resume=False)

    output = console.output
    assert "identity     (not set)" in output
    assert "You are Max, dry-witted." in output and "Tone: Casual" in output  # the /prompt output
    assert "not one of the allowed values" in output and "Usage: /persona FIELD VALUE" in output
    assert (
        settings.persona.identity == "" and settings.persona.tone == "Default"
    )  # cleared again at the end
    assert Settings.load(settings_file).persona.tone == "Default"  # and saved each time


async def test_the_web_page_sends_the_persona_and_shows_it_in_the_prompt_viewer(
    user: User, tmp_path: Path
):
    lemonade = WebLemonade()
    context = make_context(tmp_path, lemonade)
    context.settings.persona.identity = "You are Max, a dry-witted assistant."
    context.modules = build_registry(context.settings, tmp_path, db=context.db, client=lemonade)
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        user.find(marker="view-prompt").click()

        await user.should_see("You are Max, a dry-witted assistant.")
        await user.should_see("System prompt")

        user.find(marker="message-input").type("hello")
        user.find(marker="send").click()
        await user.should_see("Paris.")
        assert lemonade.requests[0][0]["content"].startswith("You are Max, a dry-witted assistant.")
    finally:
        set_context(None)


async def test_scheduled_jobs_use_the_persona_too():
    client = FakeLemonade(reply("done"))
    runner, _, settings = make_runner(client)
    settings.persona.identity = "You are Max."

    await runner.run(a_job(make_repo()))

    system = client.requests[0]["messages"][0]["content"]
    assert system.startswith("You are Max.") and "scheduled job" in system
