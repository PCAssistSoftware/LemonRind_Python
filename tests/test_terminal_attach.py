"""The terminal chat's /attach and /detach."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from lemonrind.attachments import has_image, text_of
from lemonrind.chats import ChatRepository
from lemonrind.config import Settings
from lemonrind.lemonade.models import ModelInfo
from lemonrind.storage import Database
from lemonrind.terminal.chatloop import ChatLoop
from tests.test_chatloop import FakeClient, ScriptedConsole


class SeeingClient(FakeClient):
    """A fake whose only model ("m", the one the loop uses) can see pictures."""

    async def list_models(self, **kwargs) -> list:
        return [ModelInfo(id="m", labels=["chat", "vision"], downloaded=True)]


class BlindClient(FakeClient):
    async def list_models(self, **kwargs) -> list:
        return [ModelInfo(id="m", labels=["chat"], downloaded=True)]


def loop_for(lines: list[str], client: FakeClient, tmp_path: Path):
    console = ScriptedConsole(lines)
    repo = ChatRepository(Database(":memory:"))
    loop = ChatLoop(
        client=client,  # type: ignore[arg-type]
        settings=Settings(),
        settings_file=tmp_path / "settings.json",
        repo=repo,
        console=console,
        model="m",
    )
    return loop, console, repo


def picture(tmp_path: Path) -> Path:
    path = tmp_path / "photo.png"
    Image.new("RGB", (120, 80), (0, 100, 200)).save(path)
    return path


async def test_an_attached_document_goes_with_the_next_message_only(tmp_path: Path):
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"Otters hold hands.")
    client = FakeClient()
    loop, console, repo = loop_for([f"/attach {notes}", "summarise", "and again"], client, tmp_path)

    await loop.run()

    assert "Attached notes.txt" in console.output
    first, second = (request[-1]["content"] for request in client.sent)
    assert "Otters hold hands." in first and first.endswith("summarise")
    assert second.endswith("and again") and "Otters" not in second  # the attachment was used up


async def test_a_picture_is_sent_to_a_model_that_can_see_and_refused_for_one_that_cannot(
    tmp_path: Path,
):
    seeing = SeeingClient()
    loop, console, repo = loop_for(
        [f"/attach {picture(tmp_path)}", "what is this?"], seeing, tmp_path
    )
    await loop.run()
    assert has_image(seeing.sent[0][-1]) and text_of(seeing.sent[0][-1]).endswith("what is this?")  # type: ignore[arg-type]
    (session,) = repo.list_sessions()
    assert repo.list_messages(session.id)[0].content.startswith("[Attached image: ")

    blind = BlindClient()
    loop, console, repo = loop_for(
        [f"/attach {picture(tmp_path)}", "what is this?"], blind, tmp_path
    )
    await loop.run()
    assert "does not support image input" in console.output
    assert not has_image(blind.sent[0][-1])  # type: ignore[arg-type]  # the message went, without the picture


async def test_detach_removes_the_attachment(tmp_path: Path):
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"secret")
    client = FakeClient()
    loop, console, _ = loop_for([f"/attach {notes}", "/detach", "hello"], client, tmp_path)

    await loop.run()

    assert "Attachment removed" in console.output and client.sent[0][-1]["content"].endswith(
        "hello"
    )
    assert "secret" not in client.sent[0][-1]["content"]


async def test_bad_attachments_are_explained(tmp_path: Path):
    program = tmp_path / "tool.exe"
    program.write_bytes(b"MZ")
    loop, console, _ = loop_for(
        ["/attach", f"/attach {tmp_path / 'missing.txt'}", f"/attach {program}"],
        FakeClient(),
        tmp_path,
    )

    await loop.run()

    assert "Usage: /attach PATH" in console.output
    assert "No such file" in console.output
    assert "is not a type that can be attached" in console.output


async def test_the_attachment_is_kept_when_the_message_fails(tmp_path: Path):
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"keep me")
    client = FakeClient(fail=True)
    loop, console, repo = loop_for([f"/attach {notes}", "first try"], client, tmp_path)

    await loop.run()

    assert repo.list_sessions() == []  # nothing was saved
    assert (
        loop._attachment is not None and loop._attachment.name == "notes.txt"
    )  # ready for the next try
