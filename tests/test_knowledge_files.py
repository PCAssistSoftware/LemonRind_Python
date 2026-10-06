"""A knowledge base is one file: creating, reopening, copying to another computer, and the embedding-model rules."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lemonrind.config import Settings
from lemonrind.embeddings import Embedder
from lemonrind.lemonade.models import ModelInfo
from lemonrind.modules.base import ChatContext
from lemonrind.modules.knowledge import (
    OPTION_KEY,
    DuplicateNameError,
    KnowledgeFileError,
    KnowledgeIngestor,
    KnowledgeModule,
    KnowledgeRepository,
    ModelMismatchError,
)
from lemonrind.modules.knowledge.repository import KB_MIGRATIONS
from tests.test_memory import FakeMemoryClient


class TwoModelClient(FakeMemoryClient):
    """A Lemonade whose one embedding model can be swapped for another (a different model, different numbers)."""

    def __init__(self, model: str = "embed-a") -> None:
        super().__init__()
        self.model = model

    async def list_models(self, *, downloaded_only: bool = True) -> list[ModelInfo]:
        return [ModelInfo(id=self.model, labels=["embeddings"], downloaded=True)]

    async def embed(self, texts, model):
        assert model == self.model
        vectors = await super().embed(texts, "embed")
        # model B gives different numbers for everything, as a real second model would
        return vectors if self.model == "embed-a" else [list(reversed(v)) for v in vectors]


def world(folder: Path, client: TwoModelClient | None = None):
    settings = Settings()
    settings.modules.knowledge.chunk_words = 50
    settings.modules.knowledge.chunk_overlap = 10
    settings.modules.knowledge.min_similarity = 0.6
    repo = KnowledgeRepository(folder)
    embedder = Embedder(settings, client or TwoModelClient())  # type: ignore[arg-type]
    ingestor = KnowledgeIngestor(settings, repo, embedder)
    return repo, ingestor, KnowledgeModule(settings, repo, embedder, ingestor)


# --- a file per knowledge base ----------------------------------------------------------------------------------


def test_each_knowledge_base_is_its_own_readable_file_that_survives_a_restart(tmp_path: Path):
    folder = tmp_path / "knowledge"
    repo = KnowledgeRepository(folder)
    work = repo.create_kb("Work docs!", "everything for the job")
    repo.create_kb("Recipes")
    source = repo.create_source(work.id, "text", None, "note")
    repo.add_chunks(work.id, source.id, 0, ["pizza on friday"], [[1.0, 0.0]])
    repo.mark_ready(source.id, 1)
    repo.close()

    names = sorted(p.name for p in folder.iterdir())
    assert len(names) == 2 and all(n.endswith(".kb") for n in names)
    assert any(
        n.startswith("work-docs-") for n in names
    )  # readable, unique, and no stray characters

    again = KnowledgeRepository(folder)  # a new start finds both
    assert [kb.name for kb in again.list_kbs()] == ["Recipes", "Work docs!"]
    found = again.find_kb("WORK DOCS!")
    assert (
        found is not None and found.description == "everything for the job" and found.id == work.id
    )
    assert [c.content for c, _ in again.searchable_chunks(work.id)] == ["pizza on friday"]
    assert again.stats(work.id).chunks == 1


def test_a_knowledge_base_file_is_a_single_file_with_no_companions(tmp_path: Path):
    repo = KnowledgeRepository(tmp_path / "knowledge")
    kb = repo.create_kb("Work")
    source = repo.create_source(kb.id, "text", None, "n")
    repo.add_chunks(kb.id, source.id, 0, ["x"], [[1.0, 0.0]])

    assert [p.name for p in (tmp_path / "knowledge").iterdir()] == [kb.path.name]  # type: ignore[union-attr]
    repo.close()


def test_a_file_dropped_into_the_folder_is_noticed_without_a_restart(tmp_path: Path):
    mine = KnowledgeRepository(tmp_path / "mine")
    kb = mine.create_kb("Shared")
    other = KnowledgeRepository(tmp_path / "knowledge")
    assert other.list_kbs() == []

    mine.export_copy(kb.id, tmp_path / "knowledge" / "shared.kb")  # copied in by hand

    assert [k.name for k in other.list_kbs()] == ["Shared"]
    other.close()


def test_deleting_a_knowledge_base_deletes_its_file(tmp_path: Path):
    repo = KnowledgeRepository(tmp_path / "knowledge")
    kb = repo.create_kb("Gone")
    path = kb.path
    assert path is not None and path.exists()

    repo.delete_kb(kb.id)

    assert not path.exists() and repo.get_kb(kb.id) is None and repo.list_sources(kb.id) == []


def test_names_are_still_unique_across_files_ignoring_case(tmp_path: Path):
    repo = KnowledgeRepository(tmp_path / "knowledge")
    repo.create_kb("Work")
    with pytest.raises(DuplicateNameError):
        repo.create_kb("WORK")


def test_files_in_the_folder_that_are_not_knowledge_bases_are_left_alone(tmp_path: Path):
    folder = tmp_path / "knowledge"
    folder.mkdir()
    (folder / "notes.kb").write_text("this is not a database", encoding="utf-8")
    stranger = sqlite3.connect(folder / "other.kb")  # a real SQLite file, but not ours
    stranger.execute("CREATE TABLE something_else (a)")
    stranger.commit()
    stranger.close()

    repo = KnowledgeRepository(folder)

    assert repo.list_kbs() == []
    check = sqlite3.connect(folder / "other.kb")  # and the stranger's file was not changed
    assert [r[0] for r in check.execute("SELECT name FROM sqlite_master")] == ["something_else"]
    check.close()


# --- copying a knowledge base to another computer ---------------------------------------------------------------


def filled(repo: KnowledgeRepository, name: str = "Work"):
    kb = repo.create_kb(name, "for the move")
    source = repo.create_source(kb.id, "folder", "C:/docs", "docs")
    repo.add_chunks(
        kb.id,
        source.id,
        0,
        ["alpha", "beta"],
        [[1.0, 0.0], [0.0, 1.0]],
        label="a.md",
        model="embed-a",
    )
    repo.mark_ready(source.id, 2)
    return kb


def test_a_knowledge_base_exported_from_one_place_can_be_added_in_another(tmp_path: Path):
    here = KnowledgeRepository(tmp_path / "here")
    original = filled(here)
    exported = here.export_copy(original.id, tmp_path / "carry" / "work.kb")

    there = KnowledgeRepository(tmp_path / "there")
    added = there.import_file(exported)

    assert added.id == original.id and added.name == "Work" and added.description == "for the move"
    assert (added.embedding_model, added.embedding_dim) == ("embed-a", 2)
    assert [c.content for c, _ in there.searchable_chunks(added.id)] == ["alpha", "beta"]
    assert [s.display_name for s in there.list_sources(added.id)] == ["docs"]
    assert there.file_labels(there.list_sources(added.id)[0].id) == ["a.md"]
    assert (
        added.path is not None and added.path.parent == tmp_path / "there"
    )  # a copy in the data folder
    assert exported.exists()  # the original file was left where it was


def test_adding_the_same_knowledge_base_twice_is_refused_and_a_name_clash_is_renamed(
    tmp_path: Path,
):
    here = KnowledgeRepository(tmp_path / "here")
    original = filled(here)
    exported = here.export_copy(original.id, tmp_path / "work.kb")

    there = KnowledgeRepository(tmp_path / "there")
    there.import_file(exported)
    with pytest.raises(KnowledgeFileError, match="already added"):
        there.import_file(exported)

    # a different knowledge base that happens to be called the same
    clash = KnowledgeRepository(tmp_path / "other")
    clash_kb = filled(clash)  # also "Work", but a different id
    clash_file = clash.export_copy(clash_kb.id, tmp_path / "work2.kb")
    renamed = there.import_file(clash_file)
    assert renamed.name == "Work (2)"
    assert sorted(kb.name for kb in there.list_kbs()) == ["Work", "Work (2)"]


def test_a_file_that_is_not_a_knowledge_base_or_is_too_new_is_refused_with_a_reason(tmp_path: Path):
    repo = KnowledgeRepository(tmp_path / "knowledge")

    text = tmp_path / "text.kb"
    text.write_text("hello", encoding="utf-8")
    with pytest.raises(KnowledgeFileError, match="not a knowledge base"):
        repo.import_file(text)

    stranger = tmp_path / "other.kb"
    connection = sqlite3.connect(stranger)
    connection.execute("CREATE TABLE x (a)")
    connection.commit()
    connection.close()
    with pytest.raises(KnowledgeFileError, match="not a knowledge base"):
        repo.import_file(stranger)

    other = KnowledgeRepository(tmp_path / "elsewhere")
    kb = filled(other)
    newer = other.export_copy(kb.id, tmp_path / "newer.kb")
    connection = sqlite3.connect(newer)
    connection.execute(f"PRAGMA user_version = {len(KB_MIGRATIONS) + 1}")
    connection.close()
    with pytest.raises(KnowledgeFileError, match="newer version"):
        repo.import_file(newer)
    assert repo.list_kbs() == []  # nothing was half-added


# --- the embedding model travels with the knowledge base ---------------------------------------------------------


async def test_the_model_is_recorded_when_the_first_chunks_are_added(tmp_path: Path):
    repo, ingestor, _ = world(tmp_path / "knowledge")
    kb = repo.create_kb("Work")
    assert (kb.embedding_model, kb.embedding_dim) == ("", 0)

    source = await ingestor.add_text(kb.id, "note", "pizza food on fridays")

    assert source.status == "ready"
    stored = repo.get_kb(kb.id)
    assert stored is not None and stored.embedding_model == "embed-a" and stored.embedding_dim > 0


async def test_another_model_is_refused_for_new_sources_with_a_message_that_says_what_to_do(
    tmp_path: Path,
):
    client = TwoModelClient("embed-a")
    repo, ingestor, _ = world(tmp_path / "knowledge", client)
    kb = repo.create_kb("Work")
    await ingestor.add_text(kb.id, "first", "pizza food on fridays")

    client.model = "embed-b"
    other_embedder = Embedder(Settings(), client)  # type: ignore[arg-type]  # a fresh lookup finds the new model
    second = await KnowledgeIngestor(Settings(), repo, other_embedder).add_text(
        kb.id, "second", "more pizza"
    )

    assert second.status == "failed"
    assert (
        "embed-a" in (second.error or "")
        and "embed-b" in (second.error or "")
        and "Re-embed" in (second.error or "")
    )
    with pytest.raises(ModelMismatchError):
        repo.add_chunks(kb.id, second.id, 0, ["x"], [[1.0, 0.0]], model="embed-b")


async def test_a_knowledge_base_from_another_model_is_not_searched_and_says_why(tmp_path: Path):
    client = TwoModelClient("embed-a")
    repo, ingestor, module = world(tmp_path / "knowledge", client)
    kb = repo.create_kb("Work")
    await ingestor.add_text(kb.id, "menu", "pizza food on fridays")
    chat = ChatContext(options={OPTION_KEY: kb.id})
    assert "pizza food" in await module.turn_context(
        "any pizza food?", chat=chat
    )  # fine with its own model

    client.model = "embed-b"
    module._embedder.reset()  # the embedding model is looked up again
    context = await module.turn_context("any pizza food?", chat=chat)

    assert (
        "cannot be searched" in context
        and "Re-embed" in context
        and "pizza food on fridays" not in context
    )
    assert await module.search(kb.id, "pizza", limit=3) == []


async def test_re_embedding_makes_the_knowledge_base_work_with_the_new_model(tmp_path: Path):
    client = TwoModelClient("embed-a")
    repo, ingestor, module = world(tmp_path / "knowledge", client)
    kb = repo.create_kb("Work")
    await ingestor.add_text(kb.id, "menu", "pizza food on fridays")
    chat = ChatContext(options={OPTION_KEY: kb.id})
    before = repo.searchable_chunks(kb.id)[0][1].tolist()

    client.model = "embed-b"
    module._embedder.reset()
    count = await module.reembed(kb.id)

    assert count == 1
    stored = repo.get_kb(kb.id)
    assert stored is not None and stored.embedding_model == "embed-b"
    assert repo.searchable_chunks(kb.id)[0][1].tolist() != before  # new numbers, same text
    assert "pizza food on fridays" in await module.turn_context(
        "pizza food?", chat=chat
    )  # and it searches again
    assert module.reembedding == set()


async def test_a_failed_re_embedding_leaves_the_knowledge_base_as_it_was(tmp_path: Path):
    client = TwoModelClient("embed-a")
    repo, ingestor, module = world(tmp_path / "knowledge", client)
    kb = repo.create_kb("Work")
    await ingestor.add_text(kb.id, "menu", "pizza food on fridays")
    before = repo.searchable_chunks(kb.id)[0][1].tolist()

    async def broken(texts, model):
        raise RuntimeError("Lemonade fell over")

    client.embed = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await module.reembed(kb.id)

    assert repo.searchable_chunks(kb.id)[0][1].tolist() == before
    assert repo.get_kb(kb.id).embedding_model == "embed-a"  # type: ignore[union-attr]
    assert module.reembedding == set()  # and it is not stuck as "being re-embedded"
