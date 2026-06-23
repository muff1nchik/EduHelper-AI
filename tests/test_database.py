import asyncio

from app.database import Database


async def add_document(database: Database, user_id: int, filename: str) -> int:
    return await database.add_document(
        user_id=user_id,
        filename=filename,
        file_path=f"data/uploads/{filename}",
    )


async def add_document_with_chunks(
    database: Database,
    user_id: int,
    filename: str,
    contents: list[str],
) -> int:
    chunks = [
        {
            "content": content,
            "embedding": [float(index + 1), 0.0],
            "chunk_index": index,
        }
        for index, content in enumerate(contents)
    ]
    return await database.add_document_with_chunks(
        user_id=user_id,
        filename=filename,
        file_path=f"data/uploads/{filename}",
        chunks=chunks,
    )


def test_get_user_documents_returns_only_current_user_documents(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        own_id = await add_document(database, 123, "own.pdf")
        await add_document(database, 456, "other.pdf")

        documents = await database.get_user_documents(123)

        assert documents == [{"id": own_id, "filename": "own.pdf"}]

    asyncio.run(run_test())


def test_add_document_with_chunks_makes_new_document_active(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        document_id = await add_document_with_chunks(database, 123, "lesson.pdf", ["one"])

        active_document = await database.get_active_document(123)

        assert active_document == {"id": document_id, "filename": "lesson.pdf"}

    asyncio.run(run_test())


def test_latest_uploaded_document_becomes_active(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        await add_document_with_chunks(database, 123, "first.pdf", ["one"])
        second_id = await add_document_with_chunks(database, 123, "second.pdf", ["two"])

        active_document = await database.get_active_document(123)

        assert active_document == {"id": second_id, "filename": "second.pdf"}

    asyncio.run(run_test())


def test_set_active_document_switches_user_document(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        first_id = await add_document(database, 123, "first.pdf")
        await add_document(database, 123, "second.pdf")

        selected = await database.set_active_document(123, first_id)
        active_document = await database.get_active_document(123)

        assert selected == {"id": first_id, "filename": "first.pdf"}
        assert active_document == {"id": first_id, "filename": "first.pdf"}

    asyncio.run(run_test())


def test_set_active_document_rejects_other_user_document(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        other_id = await add_document(database, 456, "other.pdf")

        selected = await database.set_active_document(123, other_id)

        assert selected is None

    asyncio.run(run_test())


def test_set_active_document_returns_none_for_missing_document(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()

        selected = await database.set_active_document(123, 999)

        assert selected is None

    asyncio.run(run_test())


def test_get_active_document_returns_selected_document(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        first_id = await add_document(database, 123, "first.pdf")
        await add_document(database, 123, "second.pdf")
        await database.set_active_document(123, first_id)

        active_document = await database.get_active_document(123)

        assert active_document == {"id": first_id, "filename": "first.pdf"}

    asyncio.run(run_test())


def test_get_active_document_falls_back_to_newest_document(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        await add_document(database, 123, "first.pdf")
        second_id = await add_document(database, 123, "second.pdf")

        active_document = await database.get_active_document(123)

        assert active_document == {"id": second_id, "filename": "second.pdf"}

    asyncio.run(run_test())


def test_get_document_chunks_returns_only_selected_document_chunks(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        first_id = await add_document_with_chunks(database, 123, "first.pdf", ["a1", "a2"])
        await add_document_with_chunks(database, 123, "second.pdf", ["b1"])

        chunks = await database.get_document_chunks(123, first_id)

        assert [chunk["content"] for chunk in chunks] == ["a1", "a2"]
        assert {chunk["filename"] for chunk in chunks} == {"first.pdf"}

    asyncio.run(run_test())


def test_get_document_chunks_excludes_other_document_chunks(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        first_id = await add_document_with_chunks(database, 123, "first.pdf", ["active"])
        await add_document_with_chunks(database, 123, "second.pdf", ["inactive"])

        chunks = await database.get_document_chunks(123, first_id)

        assert "inactive" not in [chunk["content"] for chunk in chunks]

    asyncio.run(run_test())


def test_get_document_chunks_excludes_other_user_chunks(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        other_id = await add_document_with_chunks(database, 456, "other.pdf", ["secret"])

        chunks = await database.get_document_chunks(123, other_id)

        assert chunks == []

    asyncio.run(run_test())


def test_get_user_documents_sorts_by_id_desc(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        first_id = await add_document(database, 123, "first.pdf")
        second_id = await add_document(database, 123, "second.pdf")
        third_id = await add_document(database, 123, "third.pdf")

        documents = await database.get_user_documents(123)

        assert [document["id"] for document in documents] == [
            third_id,
            second_id,
            first_id,
        ]

    asyncio.run(run_test())


def test_get_user_documents_contains_id_and_filename(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        document_id = await add_document(database, 123, "lesson.pdf")

        documents = await database.get_user_documents(123)

        assert documents == [{"id": document_id, "filename": "lesson.pdf"}]

    asyncio.run(run_test())
