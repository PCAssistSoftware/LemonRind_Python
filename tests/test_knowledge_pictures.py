"""Pictures in a knowledge base: read by the loaded vision model, searchable as text, readable and re-readable."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from lemonrind.config import Settings
from lemonrind.embeddings import Embedder
from lemonrind.lemonade.client import LemonadeError
from lemonrind.lemonade.models import Health, LoadedModel, ModelInfo
from lemonrind.modules.knowledge import KnowledgeIngestor, KnowledgeRepository
from lemonrind.modules.knowledge.pictures import (
    PictureError,
    is_picture,
    loaded_vision_model,
)
from tests.test_memory import FakeMemoryClient


class VisionClient(FakeMemoryClient):
    """A Lemonade with chat models of both kinds, one of them loaded, that answers a picture with a set text."""

    def __init__(
        self, loaded: str | None = "seer", answer: str = "A red bicycle. The sign says OPEN."
    ) -> None:
        super().__init__()
        self.loaded = loaded
        self.answer = answer
        self.asked: list[str] = []

    async def health(self) -> Health:
        models = [LoadedModel(model_name=self.loaded, type="llm")] if self.loaded else []
        return Health(status="ok", model_loaded=self.loaded, all_models_loaded=models)

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]:
        return [
            ModelInfo(id="embed", labels=["embeddings"], downloaded=True),
            ModelInfo(id="blind", labels=["chat"], downloaded=True),
            ModelInfo(id="seer", labels=["chat", "vision"], downloaded=True),
        ]

    async def describe_image(self, model, image, prompt, *, max_tokens):
        self.asked.append(model)
        if self.answer == "FAIL":
            raise LemonadeError("the model fell over")
        return self.answer


def make_picture(folder: Path, name: str = "bike.png") -> Path:
    path = folder / name
    Image.new("RGB", (40, 30), (200, 30, 30)).save(path)
    return path


def world(tmp_path: Path, client: VisionClient):
    settings = Settings()
    settings.modules.knowledge.chunk_words = 50
    settings.modules.knowledge.chunk_overlap = 10
    repo = KnowledgeRepository(tmp_path / "knowledge")
    embedder = Embedder(settings, client)  # type: ignore[arg-type]
    ingestor = KnowledgeIngestor(settings, repo, embedder, client=client)  # type: ignore[arg-type]
    return repo, ingestor


async def test_the_loaded_model_must_be_able_to_see():
    assert await loaded_vision_model(VisionClient("seer")) == "seer"  # type: ignore[arg-type]
    try:
        await loaded_vision_model(VisionClient("blind"))  # type: ignore[arg-type]
    except PictureError as error:
        assert "'blind' cannot read pictures" in str(error) and "seer" in str(error)
    else:
        raise AssertionError("a model that cannot see was accepted")
    try:
        await loaded_vision_model(VisionClient(None))  # type: ignore[arg-type]
    except PictureError as error:
        assert "No chat model is loaded" in str(error)
    else:
        raise AssertionError("nothing loaded was accepted")


async def test_a_picture_becomes_searchable_text_kept_on_its_source(tmp_path: Path):
    repo, ingestor = world(tmp_path, VisionClient())
    kb = repo.create_kb("Pics")
    source = await ingestor.add_file(kb.id, make_picture(tmp_path))

    assert source.status == "ready" and source.chunk_count >= 1
    assert source.picture_text == "A red bicycle. The sign says OPEN."
    assert repo.picture_image(source.id) is not None  # the shrunk copy, for reading again
    assert any("OPEN" in chunk.content for chunk, _ in repo.searchable_chunks(kb.id))
    assert is_picture(Path("x.JPG")) and not is_picture(Path("x.pdf"))


async def test_a_picture_the_loaded_model_cannot_read_fails_with_a_reason_and_can_be_read_again(
    tmp_path: Path,
):
    client = VisionClient("blind")
    repo, ingestor = world(tmp_path, client)
    kb = repo.create_kb("Pics")
    failed = await ingestor.add_file(kb.id, make_picture(tmp_path))
    assert failed.status == "failed" and "cannot read pictures" in (failed.error or "")

    client.loaded = "seer"  # the person loads a model that can see
    fixed = await ingestor.reread_picture(kb.id, failed.id)
    assert fixed is not None and fixed.status == "ready" and fixed.picture_text
    assert client.asked == ["seer"]


async def test_read_again_works_after_the_original_file_is_gone_and_replaces_the_old_text(
    tmp_path: Path,
):
    client = VisionClient(answer="First reading about a bicycle.")
    repo, ingestor = world(tmp_path, client)
    kb = repo.create_kb("Pics")
    picture = make_picture(tmp_path)
    source = await ingestor.add_file(kb.id, picture)
    picture.unlink()

    client.answer = "Second reading: a red rectangle."
    again = await ingestor.reread_picture(kb.id, source.id)
    assert again is not None and again.status == "ready"
    assert again.picture_text == "Second reading: a red rectangle."
    contents = [chunk.content for chunk, _ in repo.searchable_chunks(kb.id)]
    assert all("First reading" not in text for text in contents) and any(
        "Second" in t for t in contents
    )


async def test_a_model_error_is_recorded_on_the_source_not_raised(tmp_path: Path):
    repo, ingestor = world(tmp_path, VisionClient(answer="FAIL"))
    kb = repo.create_kb("Pics")
    source = await ingestor.add_file(kb.id, make_picture(tmp_path))
    assert source.status == "failed" and "fell over" in (source.error or "")


async def test_a_picture_that_is_not_a_picture_is_refused_politely(tmp_path: Path):
    repo, ingestor = world(tmp_path, VisionClient())
    kb = repo.create_kb("Pics")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not an image")
    source = await ingestor.add_file(kb.id, broken)
    assert source.status == "failed" and "could not be read as a picture" in (source.error or "")


# --- pictures inside a folder ------------------------------------------------------------------------------------------------


def make_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    (folder / "notes.txt").write_text("The meeting is on Friday at noon in the long room.")
    make_picture(folder / "sub", "sign.png")
    return folder


async def test_a_folder_reads_its_pictures_with_the_loaded_model(tmp_path: Path):
    repo, ingestor = world(tmp_path, VisionClient())
    kb = repo.create_kb("Docs")
    source = await ingestor.add_folder(kb.id, make_folder(tmp_path))

    assert source.status == "ready" and source.error is None
    contents = [chunk.content for chunk, _ in repo.searchable_chunks(kb.id)]
    assert any("Friday" in text for text in contents) and any("OPEN" in text for text in contents)


async def test_a_folder_keeps_its_documents_when_the_loaded_model_cannot_see_and_says_so(
    tmp_path: Path,
):
    client = VisionClient("blind")
    repo, ingestor = world(tmp_path, client)
    kb = repo.create_kb("Docs")
    source = await ingestor.add_folder(kb.id, make_folder(tmp_path))

    assert source.status == "ready" and source.chunk_count >= 1  # the text file is in
    assert "1 picture not read" in (source.error or "") and "cannot read pictures" in (
        source.error or ""
    )
    assert client.asked == []  # no vision request was made for a model that cannot see


async def test_a_folder_of_only_pictures_fails_with_the_reason_when_no_model_can_see(
    tmp_path: Path,
):
    repo, ingestor = world(tmp_path, VisionClient("blind"))
    kb = repo.create_kb("Docs")
    folder = tmp_path / "pics"
    folder.mkdir()
    make_picture(folder, "a.png")
    source = await ingestor.add_folder(kb.id, folder)
    assert source.status == "failed" and "cannot read pictures" in (source.error or "")
