"""While a scheduled job runs: the page says so, asks before switching to another model, and shows the job's model."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nicegui.testing import User

from lemonrind.lemonade.models import ModelInfo
from lemonrind.modules.scheduler.live import LiveRuns
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.conftest import elements
from tests.test_scheduler import SlowLemonade, a_job, make_repo, make_runner
from tests.test_scheduler_ui import scheduler
from tests.test_settings_sections import settle
from tests.test_webui import FakeLemonade, make_context


class TwoModels(FakeLemonade):
    """Two chat models to choose between, and a record of which ones the page asked Lemonade to load."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded: list[str] = []

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]:
        return [
            ModelInfo(id="Fake-Chat", labels=["chat"], downloaded=True, context_length=8000),
            ModelInfo(id="Other-Chat", labels=["chat"], downloaded=True, context_length=8000),
        ]

    async def load_model(self, name: str) -> None:
        self.loaded.append(name)


# --- the record of a run -----------------------------------------------------------------------------------------------


def test_a_run_knows_its_jobs_name_and_the_runs_going_now_can_be_listed():
    runs = LiveRuns()
    first = runs.start("chat-1", "prompt", "Nightly job")
    first.model = "Qwen-8B"
    runs.start("chat-2", "prompt")

    assert (first.name, first.model) == ("Nightly job", "Qwen-8B")
    assert [r.session_id for r in runs.running()] == ["chat-1", "chat-2"]
    runs.end("chat-1")
    assert [r.session_id for r in runs.running()] == [
        "chat-2"
    ]  # a finished run is no longer "going now"


async def test_the_runner_records_which_model_the_run_chose():
    client = SlowLemonade()
    runner, chats, _ = make_runner(client)
    task = asyncio.create_task(runner.run(a_job(make_repo())))
    await client.started.wait()

    (session,) = chats.list_sessions()
    live = runner.live.get(session.id)
    assert live is not None and live.name == "Weekly digest"
    assert live.model != ""  # chosen before the first request was sent

    client.go.set()
    await task


# --- the page --------------------------------------------------------------------------------------------------------------


async def test_the_page_says_which_job_is_running_and_with_which_model(user: User, tmp_path: Path):
    context = make_context(tmp_path, TwoModels())
    module = scheduler(context.modules)  # type: ignore[arg-type]
    run = module.live.start("job-chat", "Write the report", "Nightly job")
    run.model = "Other-Chat"
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await asyncio.sleep(1.8)  # the note is refreshed every second and a half

        (note,) = elements(user, "job-running-note")
        assert note.text == "Job running: Nightly job (Other-Chat)"
        module.live.end("job-chat")
        await asyncio.sleep(1.8)
        assert not note.visible  # and it goes away when the job ends
    finally:
        set_context(None)


async def test_picking_another_model_while_a_job_runs_asks_first_and_cancel_changes_nothing(
    user: User, tmp_path: Path
):
    lemonade = TwoModels()
    context = make_context(tmp_path, lemonade)
    module = scheduler(context.modules)  # type: ignore[arg-type]
    module.live.start("job-chat", "Write the report", "Nightly job").model = "Fake-Chat"
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        (picker,) = elements(user, "model-select")
        assert picker.value == "Fake-Chat"

        picker.value = "Other-Chat"
        await user.should_see("A scheduled job is running")
        await user.should_see("Nightly job is using Fake-Chat")
        user.find(marker="dialog-cancel").click()
        await settle()

        assert picker.value == "Fake-Chat"  # put back
        assert lemonade.loaded == []  # and Lemonade was never asked to load anything
    finally:
        module.live.end("job-chat")
        set_context(None)


async def test_confirming_the_question_switches_the_model(user: User, tmp_path: Path):
    lemonade = TwoModels()
    context = make_context(tmp_path, lemonade)
    module = scheduler(context.modules)  # type: ignore[arg-type]
    module.live.start("job-chat", "Write the report", "Nightly job").model = "Fake-Chat"
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        (picker,) = elements(user, "model-select")

        picker.value = "Other-Chat"
        await user.should_see("A scheduled job is running")
        user.find(marker="dialog-ok").click()
        await settle()

        assert picker.value == "Other-Chat" and lemonade.loaded == ["Other-Chat"]
    finally:
        module.live.end("job-chat")
        set_context(None)


async def test_no_question_when_nothing_is_running_or_when_the_job_uses_that_model(
    user: User, tmp_path: Path
):
    lemonade = TwoModels()
    context = make_context(tmp_path, lemonade)
    module = scheduler(context.modules)  # type: ignore[arg-type]
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        (picker,) = elements(user, "model-select")

        picker.value = "Other-Chat"  # no job is running
        await settle()
        assert picker.value == "Other-Chat" and lemonade.loaded == ["Other-Chat"]

        module.live.start("job-chat", "Write the report", "Nightly job").model = "Fake-Chat"
        picker.value = "Fake-Chat"  # a job is running, but it uses exactly this model
        await settle()
        await user.should_not_see("A scheduled job is running")
        assert picker.value == "Fake-Chat" and lemonade.loaded == ["Other-Chat", "Fake-Chat"]
    finally:
        module.live.end("job-chat")
        set_context(None)


async def test_watching_a_running_job_shows_its_model_read_only_and_gives_the_picker_back(
    user: User, tmp_path: Path
):
    lemonade = TwoModels()
    context = make_context(tmp_path, lemonade)
    module = scheduler(context.modules)  # type: ignore[arg-type]
    running = context.repo.create_session("Nightly job")
    context.repo.add_tag(running.id, "scheduled")
    context.repo.add_tag(running.id, "running")
    live = module.live.start(running.id, "Write the nightly report", "Nightly job")
    live.model = "Other-Chat"
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()
        (picker,) = elements(user, "model-select")
        assert picker.value == "Fake-Chat" and picker.enabled  # your own model, before watching

        user.find(marker="chat-item").click()
        await user.should_see("Write the nightly report")
        await settle()
        assert picker.value == "Other-Chat" and not picker.enabled  # the job's model, and read-only

        module.live.end(running.id)  # the job finishes
        await asyncio.sleep(1.5)
        assert picker.value == "Fake-Chat" and picker.enabled  # yours again

        # and watching never changed your own model, saved it as your choice, or asked Lemonade to load the job's
        assert context.settings.lemonade.chat_model != "Other-Chat"
        assert lemonade.loaded == []
    finally:
        module.live.end(running.id)
        set_context(None)


async def test_a_removed_default_model_gives_one_notice_and_the_automatic_choice(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, TwoModels())
    context.settings.lemonade.chat_model = (
        "Removed-Model-GGUF"  # saved long ago, since removed from Lemonade
    )
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see("Lemonade healthy")
        await settle()

        (picker,) = elements(user, "model-select")
        assert picker.value == "Fake-Chat"  # the model Lemonade has loaded: the page works as usual
        assert user.notify.contains("Your default model 'Removed-Model-GGUF' is no longer on")
        assert not user.notify.contains("is not downloaded on this Lemonade")  # and no error banner
        assert (
            context.settings.lemonade.chat_model == "Removed-Model-GGUF"
        )  # the saved setting is left alone
    finally:
        set_context(None)
