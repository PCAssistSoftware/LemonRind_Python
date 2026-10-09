"""The scheduler's screens: the terminal /jobs commands and the web Scheduled jobs dialog."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nicegui.testing import User

from lemonrind.chats import ChatRepository
from lemonrind.config import Settings
from lemonrind.modules import SchedulerModule, build_registry
from lemonrind.storage import Database
from lemonrind.terminal.chatloop import ChatLoop
from lemonrind.webui.context import set_context
from lemonrind.webui.page import register_pages
from tests.conftest import elements, open_section
from tests.test_chatloop import ScriptedConsole
from tests.test_scheduler import FakeLemonade, reply
from tests.test_webui import make_context


def scheduler(modules) -> SchedulerModule:
    return next(m for m in modules.modules if isinstance(m, SchedulerModule))


async def test_jobs_commands_add_list_switch_run_and_remove(tmp_path: Path):
    settings = Settings()
    db = Database(":memory:")
    lemonade = FakeLemonade(reply("Here is the digest."))
    modules = build_registry(settings, tmp_path, db=db, client=lemonade)
    chats = ChatRepository(db)
    console = ScriptedConsole(
        [
            "/jobs",
            "/jobs add Digest | 0 9 * * 1 | Summarise the news",
            "/jobs add Bad | nonsense | x",
            "/jobs add Digest | 0 9 * * 1 | again",
            "/jobs add missing parts",
            "/jobs allow digest on",
            "/jobs off digest",
            "/jobs on digest",
            "/jobs run digest",
            "/jobs",
            "/jobs remove digest",
            "/jobs on nobody",
            "/quit",
        ]
    )
    loop = ChatLoop(
        client=lemonade,  # type: ignore[arg-type]
        settings=settings,
        repo=chats,
        console=console,
        model="Chat",
        modules=modules,
    )

    await loop.run(resume=False)

    output = console.output
    assert "No jobs yet" in output
    assert "Added 'Digest'. Next runs:" in output
    assert "5 fields" in output  # the bad schedule is explained
    assert "already exists" in output
    assert "Usage: /jobs add NAME | CRON | PROMPT" in output
    assert "Unattended tools for 'Digest': on." in output
    assert "'Digest' is now off." in output and "'Digest' is now on." in output
    assert "Finished: ok." in output
    assert "may run tools unattended" in output and "ok" in output
    assert "Removed 'Digest'." in output and "No job with that name" in output
    assert scheduler(modules).repo.list() == []
    (session,) = [s for s in chats.list_sessions() if "scheduled" in s.tags]  # the run's chat
    assert session.title == "Digest"


async def test_jobs_commands_explain_when_the_module_is_off(tmp_path: Path):
    settings = Settings()
    db = Database(":memory:")
    modules = build_registry(settings, tmp_path, db=db, client=FakeLemonade())
    console = ScriptedConsole(["/module scheduler off", "/jobs", "/quit"])
    loop = ChatLoop(
        client=FakeLemonade(),  # type: ignore[arg-type]
        settings=settings,
        repo=ChatRepository(db),
        console=console,
        model="Chat",
        modules=modules,
    )

    await loop.run(resume=False)

    assert "The Scheduler module is off" in console.output


async def test_the_web_screen_creates_a_job_and_explains_a_bad_schedule(user: User, tmp_path: Path):
    context = make_context(tmp_path, FakeLemonade())
    module = scheduler(context.modules)  # type: ignore[arg-type]
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "scheduler")
        await user.should_see("No jobs yet")
        user.find(marker="job-new").click()
        await user.should_see("What should the assistant do each time?")
        user.find(marker="job-name").type("Friday note")
        user.find(marker="job-schedule").clear()
        user.find(marker="job-schedule").type("nonsense")
        user.find(marker="job-prompt").type("Say hello")
        await user.should_see("5 fields")  # the live preview already complains

        user.find(marker="job-save").click()
        await user.should_see("5 fields")
        assert module.repo.list() == []  # nothing saved, and the form stays open
        await asyncio.sleep(0.2)  # let the form go back to waiting for the next press of Save

        user.find(marker="job-schedule").clear()
        user.find(marker="job-schedule").type("0 19 * * 5")
        await user.should_see("Next runs:")
        user.find(marker="job-save").click()

        for _ in range(40):  # saving happens just after the click
            if module.repo.list():
                break
            await asyncio.sleep(0.05)
        (job,) = module.repo.list()
        await user.should_see("next:")  # and the saved job now shows in the list
        assert (job.cron, job.prompt, job.allow_unattended_tools) == (
            "0 19 * * 5",
            "Say hello",
            False,
        )
    finally:
        set_context(None)


async def test_the_job_form_has_pickers_that_follow_the_schedule_text_and_back(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, FakeLemonade())
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "scheduler")
        user.find(marker="job-new").click()
        await user.should_see("What should the assistant do each time?")

        def element(marker: str):
            (found,) = user.find(marker=marker).elements
            return found

        assert element("job-schedule").value == "0 9 * * *"  # a new job starts as daily at 09:00
        assert element("job-frequency").value == "daily"

        element("job-frequency").value = "weekly"  # choosing Weekly shows the day picker
        element("job-weekday").value = 5
        element("job-hour").value = 19
        element("job-minute").value = 30
        assert element("job-schedule").value == "30 19 * * 5"
        await user.should_see(marker="job-weekday")
        await user.should_not_see(marker="job-day")

        element("job-frequency").value = "monthly"
        element("job-day").value = 15
        assert element("job-schedule").value == "30 19 15 * *"
        await user.should_see(marker="job-day")
        await user.should_not_see(marker="job-weekday")

        element("job-schedule").value = "*/15 * * * *"  # typed text the pickers cannot show: Custom
        assert element("job-frequency").value == "custom"
        await user.should_not_see(marker="job-hour")
        assert element("job-schedule").value == "*/15 * * * *"  # and the text is left alone

        element("job-schedule").value = "0 7 * * 2"  # typed text they can show: the pickers follow
        assert element("job-frequency").value == "weekly" and element("job-weekday").value == 2
        assert (element("job-hour").value, element("job-minute").value) == (7, 0)
    finally:
        set_context(None)


async def test_run_now_from_the_web_screen_makes_a_tagged_chat(user: User, tmp_path: Path):
    context = make_context(tmp_path, FakeLemonade())
    context.modules = build_registry(
        context.settings, tmp_path, db=context.db, client=FakeLemonade(reply("Done."))
    )
    module = scheduler(context.modules)
    module.add_job("Digest", "0 9 * * 1", "Summarise")
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "scheduler")
        await user.should_see("Digest")
        user.find(marker="job-run").click()

        for _ in range(50):  # the run happens in the background
            if module.runs_finished:
                break
            await asyncio.sleep(0.05)
        assert module.runs_finished == 1
        (session,) = [s for s in context.repo.list_sessions() if "scheduled" in s.tags]
        assert session.title == "Digest"
        await user.should_see("Last run: ok", retries=60)  # the list redraws on a 1.5 second timer
    finally:
        set_context(None)


async def test_a_running_job_shows_a_stop_button_instead_of_run_now(user: User, tmp_path: Path):
    context = make_context(tmp_path, FakeLemonade())
    module = scheduler(context.modules)  # type: ignore[arg-type]
    job = module.add_job("Slow one", "0 9 * * 1", "Do it")
    module.repo.mark_running(job.id)  # the job is mid-run when the screen is drawn
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "scheduler")

        await user.should_see(marker="job-stop")
        await user.should_not_see(marker="job-run")
    finally:
        set_context(None)


async def test_a_running_jobs_chat_says_so_and_is_not_opened_at_start_up(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, FakeLemonade())
    older = context.repo.create_session("An ordinary chat")
    context.repo.add_message(older.id, "user", "hello there")
    context.repo.add_message(older.id, "assistant", "General Kenobi")
    running = context.repo.create_session("Nightly job")  # newer, empty, still running
    context.repo.add_tag(running.id, "scheduled")
    context.repo.add_tag(running.id, "running")
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await user.should_see(
            "General Kenobi"
        )  # the ordinary chat opened, not the empty running one
        await user.should_not_see("This scheduled job is still running")
    finally:
        set_context(None)


async def test_opening_a_running_jobs_chat_explains_that_it_is_still_working(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, FakeLemonade())
    running = context.repo.create_session("Nightly job")
    context.repo.add_tag(running.id, "scheduled")
    context.repo.add_tag(running.id, "running")
    set_context(context)
    register_pages()
    try:
        await user.open("/")  # the only chat is the running one, so there is nothing else to open
        await user.should_see("Start a conversation")
        user.find(marker="chat-item").click()
        await user.should_see("This scheduled job is still running")
    finally:
        set_context(None)


async def test_the_time_limit_for_a_run_is_shown_changed_and_saved_at_once(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, FakeLemonade())
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "scheduler")

        (box,) = elements(user, "job-time-limit")
        assert box.value == 30  # the default: 1,800 seconds

        box.value = 120  # two hours
        for _ in range(40):
            if context.settings.modules.scheduler.job_timeout_seconds == 7200:
                break
            await asyncio.sleep(0.05)

        assert context.settings.modules.scheduler.job_timeout_seconds == 7200
        assert '"job_timeout_seconds": 7200.0' in context.settings_file.read_text(encoding="utf-8")

        box.value = 0  # outside the allowed range: ignored
        box.value = 99999
        await asyncio.sleep(0.2)
        assert context.settings.modules.scheduler.job_timeout_seconds == 7200
    finally:
        set_context(None)


async def test_the_job_forms_buttons_are_outside_the_part_that_scrolls(user: User, tmp_path: Path):
    from tests.test_dialog_footer import ancestors

    context = make_context(tmp_path, FakeLemonade())
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "scheduler")
        user.find(marker="job-new").click()
        await user.should_see(marker="job-save")

        (save,) = elements(user, "job-save")
        (prompt,) = elements(user, "job-prompt")

        # a very long prompt must scroll inside the form's own box, never behind or below the buttons
        assert [a for a in ancestors(save) if "overflow-auto" in a.classes] == []
        scrolling = [a for a in ancestors(prompt) if "overflow-auto" in a.classes]
        assert len(scrolling) == 1 and "lr-form-body" in scrolling[0].classes
    finally:
        set_context(None)


def test_a_jobs_model_is_kept_in_the_list_even_when_lemonade_no_longer_has_it():
    from lemonrind.webui.scheduler_dialog import model_choices

    listed = model_choices(["Qwen3-8B-GGUF"], "Qwen3-8B-GGUF")
    assert list(listed) == ["", "Qwen3-8B-GGUF"]  # still listed: nothing added

    gone = model_choices(["Qwen3-8B-GGUF"], "Old-Model-GGUF")
    assert gone["Old-Model-GGUF"] == "Old-Model-GGUF (not listed by Lemonade now)"
    assert "Qwen3-8B-GGUF" in gone and gone[""] == "(whatever model is selected)"

    assert list(model_choices([], "")) == [
        ""
    ]  # no model chosen and none listed: just the blank choice
    assert (
        "(not listed" in model_choices([], "Anything")["Anything"]
    )  # Lemonade unreachable: still editable


async def test_a_job_whose_model_has_been_removed_can_still_be_opened_and_saved(
    user: User, tmp_path: Path
):
    context = make_context(tmp_path, FakeLemonade())
    module = scheduler(context.modules)  # type: ignore[arg-type]
    module.add_job("Digest", "0 9 * * 1", "Summarise", model="Removed-Model-GGUF")
    set_context(context)
    register_pages()
    try:
        await user.open("/")
        await open_section(user, "scheduler")
        user.find(marker="job-edit").click()

        await user.should_see(
            marker="job-save"
        )  # the form opened (it used to fail with "Invalid value")
        user.find(marker="job-save").click()
        for _ in range(40):
            await asyncio.sleep(0.05)
            if module.repo.find("Digest") is not None:
                break
        assert module.repo.find("Digest").model == "Removed-Model-GGUF"  # type: ignore[union-attr]
    finally:
        set_context(None)
