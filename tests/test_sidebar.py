"""The chat list's grouping rule, tested without any widgets."""

from lemonrind.chats import ChatRepository
from lemonrind.storage import Database
from lemonrind.webui.sidebar import group_by_folder


def make_repo() -> ChatRepository:
    return ChatRepository(Database(":memory:"))


def test_folders_come_first_by_name_then_unfiled_chats():
    repo = make_repo()
    zoo = repo.get_or_create_folder("Zoo")
    work = repo.get_or_create_folder("work")
    a = repo.create_session(title="a")
    b = repo.create_session(title="b")
    c = repo.create_session(title="c")
    repo.move_to_folder(a.id, zoo.id)
    repo.move_to_folder(b.id, work.id)

    groups = group_by_folder(repo.list_sessions(), repo.list_folders(), include_empty_folders=True)

    assert [(f.name if f else None, [s.title for s in chats]) for f, chats in groups] == [
        ("work", ["b"]),
        ("Zoo", ["a"]),
        (None, [c.title]),
    ]


def test_empty_folders_are_listed_only_when_asked_for():
    repo = make_repo()
    repo.get_or_create_folder("Empty")
    repo.create_session(title="loose")

    shown = group_by_folder(repo.list_sessions(), repo.list_folders(), include_empty_folders=True)
    hidden = group_by_folder(repo.list_sessions(), repo.list_folders(), include_empty_folders=False)

    assert [f.name if f else None for f, _ in shown] == ["Empty", None]
    assert [f.name if f else None for f, _ in hidden] == [None]


def test_a_folder_goes_when_its_last_chat_is_moved_out_or_deleted():
    repo = make_repo()
    work = repo.get_or_create_folder("Work")
    one = repo.create_session(title="one", folder_id=work.id)
    two = repo.create_session(title="two", folder_id=work.id)

    repo.move_to_folder(one.id, None)
    assert [f.name for f in repo.list_folders()] == ["Work"]  # "two" is still in it

    repo.delete_session(two.id)
    assert repo.list_folders() == []  # the last chat left, so the folder went

    other = repo.get_or_create_folder("Home")  # a folder made ahead of time is not touched
    kept = repo.create_session(title="kept")
    repo.move_to_folder(kept.id, other.id)
    assert [f.name for f in repo.list_folders()] == ["Home"]


def test_moving_a_chat_to_the_folder_it_is_already_in_keeps_the_folder():
    repo = make_repo()
    work = repo.get_or_create_folder("Work")
    chat = repo.create_session(title="x", folder_id=work.id)

    repo.move_to_folder(chat.id, work.id)

    assert [f.name for f in repo.list_folders()] == ["Work"]


def test_pruning_removes_every_empty_folder_and_only_those():
    repo = make_repo()
    repo.get_or_create_folder("Empty one")
    repo.get_or_create_folder("Empty two")
    full = repo.get_or_create_folder("Full")
    repo.create_session(title="x", folder_id=full.id)

    repo.prune_empty_folders()

    assert [f.name for f in repo.list_folders()] == ["Full"]
