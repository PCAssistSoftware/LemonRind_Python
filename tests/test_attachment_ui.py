"""Attaching a picture or a document in the web page: the preview, the vision check, sending, and reopening a chat."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from nicegui import ui
from nicegui.elements.upload_files import SmallFileUpload
from nicegui.testing import User
from PIL import Image

from lemonrind.attachments import attachments_dir, has_image, text_of
from lemonrind.lemonade.models import ModelInfo
from lemonrind.webui.context import AppContext
from tests.test_model_ui import box
from tests.test_settings_sections import settle


def picture_bytes(tmp_path: Path, size=(300, 200)) -> bytes:
    path = tmp_path / "source.png"
    Image.new("RGB", size, (20, 120, 220)).save(path)
    return path.read_bytes()


async def attach(user: User, name: str, data: bytes) -> None:
    """Do what the browser does after the person picks a file: hand it to the page's upload element."""
    (upload,) = user.find(kind=ui.upload).elements
    await upload.handle_uploads([SmallFileUpload(name, "application/octet-stream", data)])  # type: ignore[attr-defined]
    await settle()


def with_vision_model(web: AppContext, monkeypatch: pytest.MonkeyPatch) -> None:
    async def models(*, downloaded_only: bool = True) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="Fake-Chat", labels=["chat", "vision"], downloaded=True, context_length=8000
            )
        ]

    monkeypatch.setattr(web.client, "list_models", models)


async def open_page(user: User) -> None:
    await user.open("/")
    await user.should_see("Lemonade healthy")
    await settle()


# --- documents -----------------------------------------------------------------------------------------------------


async def test_an_attached_document_shows_a_chip_goes_inline_with_the_message_and_is_then_cleared(
    user: User, web: AppContext
):
    await open_page(user)

    await attach(user, "notes.txt", b"Otters hold hands.")
    await user.should_see("notes.txt")
    await user.should_see(marker="attach-remove")
    user.find(marker="message-input").type("summarise this")
    user.find(marker="send").click()

    await user.should_see("Paris.")  # the fake model's reply
    sent = web.client.requests[0][-1]["content"]  # type: ignore[attr-defined]
    assert (
        "[Attached file: notes.txt]" in sent
        and "Otters hold hands." in sent
        and sent.endswith("summarise this")
    )
    await user.should_not_see(marker="attach-remove")  # one attachment per message
    await user.should_see(
        "Attached: notes.txt"
    )  # the bubble keeps the document as a collapsible block
    (session,) = web.repo.list_sessions()
    assert session.title == "summarise this"  # the title is the question, not the document
    saved = web.repo.list_messages(session.id)[0].content
    assert saved.startswith("[Attached file: notes.txt]") and saved.endswith("summarise this")


async def test_removing_the_attachment_sends_a_plain_message(user: User, web: AppContext):
    await open_page(user)
    await attach(user, "notes.txt", b"secret notes")

    user.find(marker="attach-remove").click()
    await user.should_not_see(marker="attach-remove")
    user.find(marker="message-input").type("hello")
    user.find(marker="send").click()

    await user.should_see("Paris.")
    sent = cast(Any, web.client).requests  # the fake records every request
    assert sent[0][-1]["content"].endswith("hello") and "secret notes" not in sent[0][-1]["content"]


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("tool.exe", b"MZ", "is not a type that can be attached"),
        ("broken.png", b"this is not a picture", "could not be read as a picture"),
        ("empty.txt", b"   ", "no readable text"),
    ],
)
async def test_a_file_that_cannot_be_attached_is_explained_under_the_box(
    user: User, web: AppContext, name, data, message
):
    await open_page(user)

    await attach(user, name, data)

    await user.should_see(message)
    await user.should_not_see(marker="attach-remove")


# --- pictures ------------------------------------------------------------------------------------------------------


async def test_a_picture_for_a_model_that_cannot_see_is_refused_until_it_is_removed(
    user: User, web: AppContext, tmp_path: Path
):
    await open_page(user)

    await attach(user, "photo.png", picture_bytes(tmp_path))
    await user.should_see("doesn't support image input")
    user.find(marker="message-input").type("what is this?")
    user.find(marker="send").click()
    await settle()
    assert web.client.requests == []  # type: ignore[attr-defined]  # nothing was sent

    user.find(marker="attach-remove").click()
    await user.should_not_see("doesn't support image input")


async def test_a_picture_for_a_vision_model_is_shrunk_stored_and_sent_as_an_image_part(
    user: User, web: AppContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with_vision_model(web, monkeypatch)
    await open_page(user)

    await attach(user, "photo.png", picture_bytes(tmp_path, size=(3000, 1000)))
    await user.should_not_see("doesn't support image input")
    stored = list(attachments_dir(tmp_path).glob("img_*.jpg"))
    assert len(stored) == 1
    with Image.open(stored[0]) as shrunk:
        assert max(shrunk.size) == 1568  # the stored copy is the shrunk one
    user.find(marker="message-input").type("what colour is it?")
    user.find(marker="send").click()

    await user.should_see("Paris.")
    request = web.client.requests[0][-1]  # type: ignore[attr-defined]
    assert has_image(request) and text_of(request).endswith("what colour is it?")
    (session,) = web.repo.list_sessions()
    assert (
        web.repo.list_messages(session.id)[0].content
        == f"[Attached image: {stored[0]}]\n\nwhat colour is it?"
    )
    assert any(
        str(image.source).startswith("/attached/img_")
        for image in user.find(kind=ui.image).elements
    )  # type: ignore[attr-defined]


async def test_a_failed_send_puts_the_message_and_its_attachment_back(user: User, web: AppContext):
    web.client.fail = True  # type: ignore[attr-defined]
    await open_page(user)
    await attach(user, "notes.txt", b"keep me")
    user.find(marker="message-input").type("please read")
    user.find(marker="send").click()
    await settle()

    await user.should_see(marker="attach-remove")  # the chip is back
    assert box(user, "message-input").value == "please read"  # type: ignore[attr-defined]
    assert web.repo.list_sessions() == []  # nothing was saved


# --- reopening a chat ----------------------------------------------------------------------------------------------


async def test_reopening_a_chat_shows_its_picture_and_its_document(
    user: User, web: AppContext, tmp_path: Path
):
    folder = attachments_dir(tmp_path)
    folder.mkdir(parents=True)
    Image.new("RGB", (40, 40), (1, 2, 3)).save(folder / "img_old.jpg")
    chat = web.repo.create_session(title="Old chat")
    web.repo.add_message(
        chat.id, "user", f"[Attached image: {folder / 'img_old.jpg'}]\n\nwhat is this?"
    )
    web.repo.add_message(chat.id, "assistant", "A small square.")
    web.repo.add_message(
        chat.id,
        "user",
        "[Attached file: report.pdf]\n\nRevenue rose.\n\n[End of attached file]\n\nsummarise",
    )
    web.repo.add_message(chat.id, "assistant", "It went up.")
    web.repo.add_message(
        chat.id, "user", f"[Attached image: {folder / 'img_gone.jpg'}]\n\nand this one?"
    )
    web.repo.add_message(chat.id, "assistant", "I cannot see it.")

    await open_page(user)

    await user.should_see("what is this?")  # the typed text, without the marker
    await user.should_see("Attached: report.pdf")
    await user.should_see("summarise")
    await user.should_see("(the attached picture is no longer on disk)")
    await user.should_not_see("[Attached image:")
    await user.should_not_see("[End of attached file]")
    assert any(
        str(image.source) == "/attached/img_old.jpg" for image in user.find(kind=ui.image).elements
    )  # type: ignore[attr-defined]


async def test_the_next_message_after_reopening_still_carries_the_old_picture_only_if_it_is_the_newest(
    user: User, web: AppContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with_vision_model(web, monkeypatch)
    folder = attachments_dir(tmp_path)
    folder.mkdir(parents=True)
    Image.new("RGB", (40, 40), (9, 9, 9)).save(folder / "img_old.jpg")
    chat = web.repo.create_session(title="Old chat")
    web.repo.add_message(
        chat.id, "user", f"[Attached image: {folder / 'img_old.jpg'}]\n\nwhat is this?"
    )
    web.repo.add_message(chat.id, "assistant", "A square.")

    await open_page(user)
    user.find(marker="message-input").type("and what colour?")
    user.find(marker="send").click()
    await user.should_see("Paris.")
    await asyncio.sleep(0.1)

    earlier = web.client.requests[0][1]  # type: ignore[attr-defined]
    assert (
        has_image(earlier) and text_of(earlier) == "what is this?"
    )  # no newer picture, so this one is still sent
