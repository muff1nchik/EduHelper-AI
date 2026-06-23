import asyncio

import aiosqlite
import pytest

from app.database import Database
from app.services import EduHelperService, _format_sources


class FixedSplitter:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    def split(self, text: str) -> list[str]:
        return self.chunks


class FakeOllamaClient:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("Ошибка Ollama")
        return [float(self.calls), 1.0]


async def count_rows(database_path: str, table: str) -> int:
    async with aiosqlite.connect(database_path) as db:
        cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
        row = await cursor.fetchone()
    return int(row[0])


def make_service(database: Database, splitter: FixedSplitter, ollama_client: FakeOllamaClient):
    return EduHelperService(
        settings=None,
        database=database,
        ollama_client=ollama_client,
        splitter=splitter,
        search_engine=None,
    )


def test_process_file_does_not_save_partial_document_on_embedding_error(tmp_path):
    async def run_test() -> None:
        database_path = tmp_path / "eduhelper.db"
        file_path = tmp_path / "lesson.txt"
        file_path.write_text("Учебный текст", encoding="utf-8")

        database = Database(str(database_path))
        await database.init()
        service = make_service(
            database=database,
            splitter=FixedSplitter(["Первый чанк", "Второй чанк"]),
            ollama_client=FakeOllamaClient(fail_on_call=2),
        )

        with pytest.raises(RuntimeError, match="Ошибка Ollama"):
            await service.process_file(
                user_id=123,
                file_path=str(file_path),
                filename="lesson.txt",
            )

        assert await count_rows(str(database_path), "documents") == 0
        assert await count_rows(str(database_path), "chunks") == 0

    asyncio.run(run_test())


def test_process_file_saves_document_and_chunks_after_all_embeddings(tmp_path):
    async def run_test() -> None:
        database_path = tmp_path / "eduhelper.db"
        file_path = tmp_path / "lesson.txt"
        file_path.write_text("Учебный текст", encoding="utf-8")

        database = Database(str(database_path))
        await database.init()
        service = make_service(
            database=database,
            splitter=FixedSplitter(["Первый чанк", "Второй чанк", "Третий чанк"]),
            ollama_client=FakeOllamaClient(),
        )

        chunks_count = await service.process_file(
            user_id=123,
            file_path=str(file_path),
            filename="lesson.txt",
        )

        assert chunks_count == 3
        assert await count_rows(str(database_path), "documents") == 1
        assert await count_rows(str(database_path), "chunks") == 3

    asyncio.run(run_test())


def test_add_document_with_chunks_rolls_back_chunk_write_error(tmp_path):
    async def run_test() -> None:
        database_path = tmp_path / "eduhelper.db"
        database = Database(str(database_path))
        await database.init()
        chunks = [
            {
                "content": "Первый чанк",
                "embedding": [1.0, 0.0],
                "chunk_index": 0,
            },
            {
                "content": "Второй чанк",
                "chunk_index": 1,
            },
        ]

        with pytest.raises(KeyError, match="embedding"):
            await database.add_document_with_chunks(
                user_id=123,
                filename="lesson.txt",
                file_path="lesson.txt",
                chunks=chunks,
            )

        assert await count_rows(str(database_path), "documents") == 0
        assert await count_rows(str(database_path), "chunks") == 0

    asyncio.run(run_test())


def test_format_sources_formats_single_source():
    results = [{"filename": "file.pdf"}]

    assert _format_sources(results) == "Источник: file.pdf"


def test_format_sources_removes_duplicate_sources():
    results = [
        {"filename": "file.pdf"},
        {"filename": "file.pdf"},
    ]

    assert _format_sources(results) == "Источник: file.pdf"


def test_format_sources_formats_multiple_sources_as_list():
    results = [
        {"filename": "lecture.pdf"},
        {"filename": "notes.txt"},
    ]

    assert _format_sources(results) == "Источники:\n- lecture.pdf\n- notes.txt"


def test_format_sources_preserves_first_seen_order():
    results = [
        {"filename": "b.txt"},
        {"filename": "a.txt"},
        {"filename": "b.txt"},
    ]

    assert _format_sources(results) == "Источники:\n- b.txt\n- a.txt"
