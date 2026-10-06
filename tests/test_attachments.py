"""Attaching pictures and documents: resizing, reading, the stored marker, and what the model is sent."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from lemonrind.attachment_markers import (
    FILE_END,
    FILE_PREFIX,
    IMAGE_PREFIX,
    parse_file_attachment,
    parse_file_marker,
    parse_image_marker,
    title_source,
)
from lemonrind.attachments import (
    MAX_FILE_CHARACTERS,
    MAX_LONGEST_EDGE,
    Attachment,
    AttachmentError,
    has_image,
    prepare,
    read_document,
    save_resized_image,
    stored_text,
    text_of,
    wire_content,
)
from lemonrind.chats import ChatRepository, Conversation
from lemonrind.chats.conversation import wire_message
from lemonrind.storage import Database
from tests.test_conversation import ScriptedStreamer

# --- pictures ------------------------------------------------------------------------------------------------------


def make_picture(path: Path, size=(400, 300), mode="RGB", colour=(200, 30, 30)) -> Path:
    Image.new(mode, size, colour).save(path)
    return path


def test_a_big_picture_is_shrunk_to_the_longest_edge_and_saved_as_a_jpeg(tmp_path: Path):
    source = make_picture(tmp_path / "big.png", size=(3136, 1000))

    saved = save_resized_image(source, tmp_path / "out")

    with Image.open(saved) as result:
        assert result.format == "JPEG" and result.size == (
            MAX_LONGEST_EDGE,
            round(1000 * MAX_LONGEST_EDGE / 3136),
        )
    assert (
        saved.parent == tmp_path / "out"
        and saved.name.startswith("img_")
        and saved.suffix == ".jpg"
    )


def test_a_small_picture_keeps_its_size_and_every_save_gets_its_own_name(tmp_path: Path):
    source = make_picture(tmp_path / "small.png", size=(400, 300))

    first = save_resized_image(source, tmp_path / "out")
    second = save_resized_image(source, tmp_path / "out")

    with Image.open(first) as result:
        assert result.size == (400, 300)
    assert first != second


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "L", "CMYK"])
def test_pictures_in_any_colour_mode_become_plain_rgb_jpegs(tmp_path: Path, mode: str):
    source = tmp_path / f"{mode}.png" if mode not in ("CMYK",) else tmp_path / "cmyk.tif"
    Image.new(mode, (50, 50)).save(source)

    saved = save_resized_image(source, tmp_path / "out")

    with Image.open(saved) as result:
        assert result.mode == "RGB"


def test_a_transparent_picture_is_put_on_white_not_black(tmp_path: Path):
    source = tmp_path / "clear.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(source)  # fully transparent

    saved = save_resized_image(source, tmp_path / "out")

    with Image.open(saved) as result:
        assert min(cast(tuple[int, ...], result.getpixel((5, 5)))) > 240  # white, not black


def test_a_phones_rotation_flag_is_applied(tmp_path: Path):
    source = tmp_path / "phone.jpg"
    picture = Image.new("RGB", (200, 100), (10, 10, 200))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to view upright"
    picture.save(source, exif=exif)

    saved = save_resized_image(source, tmp_path / "out")

    with Image.open(saved) as result:
        assert result.size == (100, 200)


def test_something_that_is_not_a_picture_is_reported_clearly(tmp_path: Path):
    fake = tmp_path / "not.png"
    fake.write_text("this is text, not a picture", encoding="utf-8")

    with pytest.raises(AttachmentError, match=r"not\.png.*could not be read as a picture"):
        save_resized_image(fake, tmp_path / "out")


# --- documents -----------------------------------------------------------------------------------------------------


def test_a_text_document_is_read(tmp_path: Path):
    notes = tmp_path / "notes.md"
    notes.write_bytes(
        b"# Plan\nBuy otters."
    )  # bytes, so Windows does not turn the line break into two characters

    assert read_document(notes) == "# Plan\nBuy otters."


def test_a_very_long_document_is_cut_with_a_note(tmp_path: Path):
    big = tmp_path / "big.txt"
    big.write_text("x" * (MAX_FILE_CHARACTERS + 500), encoding="utf-8")

    text = read_document(big)

    assert text.startswith("x" * MAX_FILE_CHARACTERS) and "truncated" in text and "50,000" in text
    assert len(text) < MAX_FILE_CHARACTERS + 200


def test_an_empty_document_and_a_scanned_pdf_get_specific_messages(tmp_path: Path):
    from pypdf import PdfWriter

    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(AttachmentError, match="no readable text"):
        read_document(empty)

    scanned = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # a page with no text layer, like a scan
    with scanned.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(AttachmentError, match="no text layer.*without OCR"):
        read_document(scanned)


def test_prepare_chooses_the_right_treatment_and_refuses_other_types(tmp_path: Path):
    picture = make_picture(tmp_path / "p.png")
    document = tmp_path / "d.txt"
    document.write_text("hello", encoding="utf-8")
    program = tmp_path / "tool.exe"
    program.write_bytes(b"MZ")

    image_attachment = prepare(picture, tmp_path / "out")
    file_attachment = prepare(document, tmp_path / "out")

    assert (
        image_attachment.is_image
        and image_attachment.path is not None
        and image_attachment.name == "p.png"
    )
    assert (file_attachment.kind, file_attachment.text) == ("file", "hello")
    with pytest.raises(AttachmentError, match=r"tool\.exe.*Supported"):
        prepare(program, tmp_path / "out")


# --- the marker ----------------------------------------------------------------------------------------------------


def test_an_image_marker_round_trips_even_with_a_windows_path_with_spaces(tmp_path: Path):
    path = tmp_path / "my folder" / "img_1.jpg"
    attachment = Attachment("image", "holiday.png", path=path)

    saved = stored_text("what is this?", attachment)

    assert saved.startswith(IMAGE_PREFIX) and saved.endswith("\n\nwhat is this?")
    assert parse_image_marker(saved) == (path, "what is this?")


def test_text_without_a_marker_or_with_a_broken_one_is_left_alone():
    assert parse_image_marker("just a question") == (None, "just a question")
    assert parse_image_marker("[Attached image: no closing bracket") == (
        None,
        "[Attached image: no closing bracket",
    )
    assert (
        parse_file_marker("[Attached image: x]\n\nq") is None and parse_file_marker("plain") is None
    )


def test_a_document_marker_holds_the_name_the_text_and_the_question():
    attachment = Attachment("file", "report.pdf", text="Revenue rose.")

    saved = stored_text("summarise", attachment)

    assert saved == f"{FILE_PREFIX}report.pdf]\n\nRevenue rose.\n\n{FILE_END}\n\nsummarise"
    assert parse_file_marker(saved) == "report.pdf"
    assert stored_text("no attachment", None) == "no attachment"


def test_what_the_model_is_sent_is_an_image_part_for_a_picture_and_inline_text_for_a_document(
    tmp_path: Path,
):
    picture = save_resized_image(make_picture(tmp_path / "p.png"), tmp_path / "out")
    image = Attachment("image", "p.png", path=picture)
    document = Attachment("file", "d.txt", text="body")

    parts = wire_content("describe", image)
    assert isinstance(parts, list) and parts[0] == {"type": "text", "text": "describe"}
    url = parts[1]["image_url"]["url"]
    assert (
        url.startswith("data:image/jpeg;base64,")
        and base64.b64decode(url.split(",", 1)[1]) == picture.read_bytes()
    )
    assert (
        wire_content("summarise", document)
        == "[Attached file: d.txt]\n\nbody\n\n[End of attached file]\n\nsummarise"
    )
    assert wire_content("plain", None) == "plain"


def test_helpers_tell_pictures_apart_and_read_text_from_either_shape():
    plain = {"role": "user", "content": "hi"}
    pictured = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "u"}},
        ],
    }

    assert not has_image(plain) and has_image(pictured)
    assert (
        text_of(plain) == "hi" and text_of(pictured) == "look" and text_of({"role": "user"}) == ""
    )


# --- the conversation ----------------------------------------------------------------------------------------------


def conversation_with(streamer: ScriptedStreamer) -> tuple[Conversation, ChatRepository]:
    repo = ChatRepository(Database(":memory:"))
    return Conversation(client=streamer, repo=repo, system_prompt="s", model="m"), repo


async def test_a_picture_is_sent_as_an_image_part_and_saved_with_a_marker(tmp_path: Path):
    picture = save_resized_image(make_picture(tmp_path / "p.png"), tmp_path / "out")
    streamer = ScriptedStreamer()
    conversation, repo = conversation_with(streamer)

    await conversation.send("what is this?", attachment=Attachment("image", "p.png", path=picture))

    sent = streamer.requests[0][-1]
    assert has_image(sent) and text_of(sent) == "what is this?"
    (session,) = repo.list_sessions()
    saved = repo.list_messages(session.id)[0].content
    assert saved == f"{IMAGE_PREFIX}{picture}]\n\nwhat is this?"
    assert session.title == "what is this?"  # the title comes from the question, not the marker


async def test_a_document_goes_inline_and_is_saved_whole(tmp_path: Path):
    streamer = ScriptedStreamer()
    conversation, repo = conversation_with(streamer)

    await conversation.send(
        "summarise", attachment=Attachment("file", "r.txt", text="Revenue rose.")
    )

    assert (
        streamer.requests[0][-1]["content"]
        == "[Attached file: r.txt]\n\nRevenue rose.\n\n[End of attached file]\n\nsummarise"
    )
    (session,) = repo.list_sessions()
    assert (
        repo.list_messages(session.id)[0].content
        == "[Attached file: r.txt]\n\nRevenue rose.\n\n[End of attached file]\n\nsummarise"
    )


async def test_only_the_newest_picture_is_ever_sent(tmp_path: Path):
    first = save_resized_image(
        make_picture(tmp_path / "a.png", colour=(255, 0, 0)), tmp_path / "out"
    )
    second = save_resized_image(
        make_picture(tmp_path / "b.png", colour=(0, 0, 255)), tmp_path / "out"
    )
    streamer = ScriptedStreamer()
    conversation, _ = conversation_with(streamer)

    await conversation.send("first picture", attachment=Attachment("image", "a.png", path=first))
    await conversation.send("and a text only question")
    await conversation.send("second picture", attachment=Attachment("image", "b.png", path=second))

    second_request = streamer.requests[1]  # the first picture is still the newest here: it is sent
    assert [has_image(m) for m in second_request if m["role"] == "user"] == [True, False]
    third_request = streamer.requests[2]
    users = [m for m in third_request if m["role"] == "user"]
    assert [has_image(m) for m in users] == [False, False, True]  # the old picture is now text only
    assert text_of(users[0]) == "first picture"  # its question is kept
    assert (
        has_image(conversation.history[1]) is True
    )  # (the stored history itself is not changed by sending)


async def test_background_text_goes_in_front_of_the_questions_text_part_when_there_is_a_picture(
    tmp_path: Path,
):
    picture = save_resized_image(make_picture(tmp_path / "p.png"), tmp_path / "out")
    streamer = ScriptedStreamer()
    conversation, _ = conversation_with(streamer)
    conversation.time_awareness = True

    await conversation.send("describe it", attachment=Attachment("image", "p.png", path=picture))

    parts = cast(list[dict[str, Any]], streamer.requests[0][-1]["content"])
    assert parts[0]["text"].startswith("(Background for the assistant") and parts[0][
        "text"
    ].endswith("describe it")
    assert parts[1]["type"] == "image_url"


async def test_reopening_a_chat_brings_its_picture_back_and_copes_when_the_file_is_gone(
    tmp_path: Path,
):
    picture = save_resized_image(make_picture(tmp_path / "p.png"), tmp_path / "out")
    streamer = ScriptedStreamer()
    conversation, repo = conversation_with(streamer)
    await conversation.send("what is this?", attachment=Attachment("image", "p.png", path=picture))
    (session,) = repo.list_sessions()

    conversation.new()
    conversation.open(session)
    assert (
        has_image(conversation.history[1]) and text_of(conversation.history[1]) == "what is this?"
    )

    picture.unlink()  # the stored copy has been deleted
    conversation.new()
    conversation.open(session)
    assert conversation.history[1] == {
        "role": "user",
        "content": "what is this?",
    }  # the question survives, no marker


def test_wire_message_leaves_ordinary_messages_alone():
    from datetime import UTC, datetime

    from lemonrind.chats.models import StoredMessage

    message = StoredMessage(
        id=1, session_id="s", role="user", content="hello", created_at=datetime.now(UTC)
    )

    assert wire_message(message) == {"role": "user", "content": "hello"}


def test_a_stored_document_message_is_split_into_name_text_and_what_was_typed():
    saved = stored_text("what changed?", Attachment("file", "r.txt", text="line one\n\nline two"))

    parts = parse_file_attachment(saved)

    assert parts is not None
    assert (parts.name, parts.text, parts.typed) == (
        "r.txt",
        "line one\n\nline two",
        "what changed?",
    )  # blank lines inside are fine
    assert parse_file_attachment("ordinary") is None


def test_a_document_message_saved_without_the_end_line_is_all_document_text():
    old_style = "[Attached file: old.txt]\n\nbody text"

    parts = parse_file_attachment(old_style)

    assert parts is not None and (parts.name, parts.text, parts.typed) == (
        "old.txt",
        "body text",
        "",
    )


def test_the_title_comes_from_what_was_typed():
    assert (
        title_source(stored_text("summarise the year", Attachment("file", "r.txt", text="x")))
        == "summarise the year"
    )
    assert (
        title_source(stored_text("", Attachment("file", "r.txt", text="x"))) == "r.txt"
    )  # nothing typed: the file name
    assert title_source(f"{IMAGE_PREFIX}C:/x.jpg]\n\nwhat is this?") == "what is this?"
    assert title_source("plain") == "plain"
