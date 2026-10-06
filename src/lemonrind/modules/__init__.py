"""Capability modules: self-contained features that can each be switched on or off.

``build_registry`` is the one place that lists them. To add a feature: write a ``Module`` subclass in
its own file and add one line here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lemonrind.config import Settings
from lemonrind.embeddings import Embedder
from lemonrind.modules.backup import AutoBackupModule
from lemonrind.modules.base import Module
from lemonrind.modules.coder import CoderModule
from lemonrind.modules.filesystem import FileSystemModule
from lemonrind.modules.images import ImagesModule
from lemonrind.modules.knowledge import (
    FOLDER_NAME,
    KnowledgeIngestor,
    KnowledgeModule,
    KnowledgeRepository,
)
from lemonrind.modules.mcp import McpModule, McpServerRepository
from lemonrind.modules.memory import MemoryModule, MemoryRepository
from lemonrind.modules.registry import ModuleRegistry
from lemonrind.modules.scheduler import JobRunner, SchedulerModule, SchedulerRepository
from lemonrind.modules.tool import Tool, ToolError, ToolResult, tool_from_function
from lemonrind.modules.utilities import UtilitiesModule
from lemonrind.modules.web_reader import WebReaderModule
from lemonrind.modules.web_search import WebSearchModule
from lemonrind.storage import Database


def build_registry(
    settings: Settings, data_dir: Path, *, db: Database, client: Any
) -> ModuleRegistry:
    """Create every module and the registry that manages them.

    Modules receive what they need (the settings, the database, the Lemonade client) as constructor
    arguments, so each can be tested with fakes.
    """
    from lemonrind.chats import (
        ChatRepository,  # here, not at the top: chats imports this package (circular)
    )

    # A scheduled job runs through the very registry that contains the scheduler, so the runner is given a
    # function that looks the registry up when a job runs, after everything has been built.
    registries: list[ModuleRegistry] = []
    runner = JobRunner(
        settings, ChatRepository(db), client, lambda: registries[0] if registries else None
    )
    embedder = Embedder(
        settings, client
    )  # one embedding-model lookup, shared by memory and knowledge
    knowledge_repo = KnowledgeRepository(data_dir / FOLDER_NAME)  # one .kb file per knowledge base
    knowledge_repo.migrate_from_main_database(
        db
    )  # ones kept in the main database by earlier versions move out
    files = FileSystemModule(settings, data_dir)
    modules: list[Module] = [
        UtilitiesModule(settings),
        WebSearchModule(settings),
        WebReaderModule(settings),
        files,
        CoderModule(settings, files),  # works in the same sandbox folder as the file system module
        MemoryModule(settings, MemoryRepository(db), client, embedder),
        KnowledgeModule(
            settings,
            knowledge_repo,
            embedder,
            KnowledgeIngestor(settings, knowledge_repo, embedder),
        ),
        McpModule(settings, McpServerRepository(db), log_dir=data_dir / "logs"),
        SchedulerModule(settings, SchedulerRepository(db), runner),
        ImagesModule(settings, data_dir, client),
        AutoBackupModule(settings, data_dir),
    ]
    registry = ModuleRegistry(modules, max_output_chars=settings.modules.max_tool_output_chars)
    registries.append(registry)  # now the scheduler's runner can find it
    return registry


__all__ = [
    "AutoBackupModule",
    "CoderModule",
    "FileSystemModule",
    "ImagesModule",
    "KnowledgeModule",
    "McpModule",
    "MemoryModule",
    "Module",
    "ModuleRegistry",
    "SchedulerModule",
    "Tool",
    "ToolError",
    "ToolResult",
    "UtilitiesModule",
    "WebReaderModule",
    "WebSearchModule",
    "build_registry",
    "tool_from_function",
]
