"""Knowledge bases: search over your own documents, attached per chat."""

from lemonrind.modules.knowledge.ingestion import KnowledgeIngestor
from lemonrind.modules.knowledge.module import OPTION_KEY, KnowledgeModule
from lemonrind.modules.knowledge.repository import (
    FOLDER_NAME,
    DuplicateNameError,
    ImportPlan,
    KnowledgeBase,
    KnowledgeFileError,
    KnowledgeRepository,
    ModelMismatchError,
    Source,
    copy_database,
)

__all__ = [
    "FOLDER_NAME",
    "ImportPlan",
    "OPTION_KEY",
    "DuplicateNameError",
    "copy_database",
    "KnowledgeBase",
    "KnowledgeFileError",
    "KnowledgeIngestor",
    "KnowledgeModule",
    "KnowledgeRepository",
    "ModelMismatchError",
    "Source",
]
