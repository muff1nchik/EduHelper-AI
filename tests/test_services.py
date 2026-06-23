import asyncio
from types import SimpleNamespace

import aiosqlite
import pytest

from app.database import Database
from app.messages import INSUFFICIENT_INFORMATION_MESSAGE
from app.search import VectorSearch
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


class FakeAnswerOllamaClient:
    def __init__(self, answer: str = "Ответ по материалам") -> None:
        self.answer = answer
        self.generate_calls = 0
        self.context_chunks: list[str] = []

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        self.generate_calls += 1
        self.context_chunks = context_chunks
        return self.answer


class FakeQuestionDatabase:
    def __init__(
        self,
        documents: list[dict] | None = None,
        active_document: dict | None = None,
        chunks: list[dict] | None = None,
    ) -> None:
        self.documents = documents or []
        self.active_document = active_document
        self.chunks = chunks or [
            {
                "chunk_id": 1,
                "document_id": 1,
                "filename": "lesson.pdf",
                "content": "Учебный фрагмент",
                "embedding": [1.0, 0.0],
            }
        ]

    async def user_has_chunks(self, user_id: int) -> bool:
        return True

    async def get_user_chunks(self, user_id: int) -> list[dict]:
        return self.chunks

    async def get_active_document(self, user_id: int) -> dict | None:
        if self.active_document is not None:
            return self.active_document
        if self.documents:
            return self.documents[0]
        return {"id": 1, "filename": "lesson.pdf"}

    async def get_document_chunks(self, user_id: int, document_id: int) -> list[dict]:
        return [
            chunk for chunk in self.chunks
            if chunk.get("document_id") == document_id
        ]

    async def get_user_documents(self, user_id: int) -> list[dict]:
        return self.documents

    async def set_active_document(self, user_id: int, document_id: int) -> dict | None:
        for document in self.documents:
            if document["id"] == document_id:
                self.active_document = document
                return document
        return None


class FixedSearchEngine:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def search(self, chunks: list[dict], query_embedding: list[float], top_k: int) -> list[dict]:
        return self.results


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


def make_question_service(search_results: list[dict], ollama_client: FakeAnswerOllamaClient):
    return EduHelperService(
        settings=SimpleNamespace(top_k=4),
        database=FakeQuestionDatabase(),
        ollama_client=ollama_client,
        splitter=None,
        search_engine=FixedSearchEngine(search_results),
    )


def make_document_service(documents: list[dict], active_document: dict | None = None):
    return EduHelperService(
        settings=None,
        database=FakeQuestionDatabase(documents, active_document=active_document),
        ollama_client=None,
        splitter=None,
        search_engine=None,
    )


async def add_document_with_chunks(
    database: Database,
    user_id: int,
    filename: str,
    chunks: list[dict],
) -> int:
    return await database.add_document_with_chunks(
        user_id=user_id,
        filename=filename,
        file_path=f"data/uploads/{filename}",
        chunks=chunks,
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


def test_answer_question_returns_refusal_when_search_results_are_empty():
    async def run_test() -> None:
        ollama_client = FakeAnswerOllamaClient()
        service = make_question_service([], ollama_client)

        answer = await service.answer_question(user_id=123, question="Как приготовить борщ?")

        assert answer == INSUFFICIENT_INFORMATION_MESSAGE

    asyncio.run(run_test())


def test_answer_question_does_not_call_ollama_generation_when_results_are_empty():
    async def run_test() -> None:
        ollama_client = FakeAnswerOllamaClient()
        service = make_question_service([], ollama_client)

        await service.answer_question(user_id=123, question="Как приготовить борщ?")

        assert ollama_client.generate_calls == 0

    asyncio.run(run_test())


def test_answer_question_calls_ollama_generation_when_result_exists():
    async def run_test() -> None:
        ollama_client = FakeAnswerOllamaClient()
        service = make_question_service(
            [
                {
                    "filename": "lesson.pdf",
                    "content": "Учебный фрагмент",
                    "score": 0.9,
                }
            ],
            ollama_client,
        )

        await service.answer_question(user_id=123, question="Что в материале?")

        assert ollama_client.generate_calls == 1

    asyncio.run(run_test())


def test_answer_question_adds_source_when_result_exists():
    async def run_test() -> None:
        ollama_client = FakeAnswerOllamaClient()
        service = make_question_service(
            [
                {
                    "filename": "lesson.pdf",
                    "content": "Учебный фрагмент",
                    "score": 0.9,
                }
            ],
            ollama_client,
        )

        answer = await service.answer_question(user_id=123, question="Что в материале?")

        assert answer == "Ответ по материалам\n\nИсточник: lesson.pdf"

    asyncio.run(run_test())


def test_answer_question_returns_model_refusal_without_source():
    async def run_test() -> None:
        ollama_client = FakeAnswerOllamaClient(
            answer=f"\n{INSUFFICIENT_INFORMATION_MESSAGE}  "
        )
        service = make_question_service(
            [
                {
                    "filename": "lesson.pdf",
                    "content": "Учебный фрагмент",
                    "score": 0.9,
                }
            ],
            ollama_client,
        )

        answer = await service.answer_question(user_id=123, question="Что в материале?")

        assert answer == INSUFFICIENT_INFORMATION_MESSAGE
        assert "Источник:" not in answer
        assert "Источники:" not in answer

    asyncio.run(run_test())


def test_answer_question_does_not_add_source_to_refusal():
    async def run_test() -> None:
        ollama_client = FakeAnswerOllamaClient()
        service = make_question_service([], ollama_client)

        answer = await service.answer_question(user_id=123, question="Как приготовить борщ?")

        assert "Источник:" not in answer
        assert "Источники:" not in answer

    asyncio.run(run_test())


def test_list_documents_formats_multiple_documents_with_real_ids():
    async def run_test() -> None:
        service = make_document_service(
            [
                {"id": 7, "filename": "mechanics.pdf"},
                {"id": 3, "filename": "voprosy.pdf"},
            ],
            active_document={"id": 7, "filename": "mechanics.pdf"},
        )

        answer = await service.list_documents(user_id=123)

        assert answer == (
            "Ваши загруженные материалы:\n\n"
            "7. mechanics.pdf — активный\n"
            "3. voprosy.pdf"
        )

    asyncio.run(run_test())


def test_use_document_returns_success_message():
    async def run_test() -> None:
        service = make_document_service([{"id": 7, "filename": "mechanics.pdf"}])

        answer = await service.use_document(user_id=123, document_id=7)

        assert answer == "Активный материал: mechanics.pdf"

    asyncio.run(run_test())


def test_use_document_returns_not_found_message_for_invalid_id():
    async def run_test() -> None:
        service = make_document_service([{"id": 7, "filename": "mechanics.pdf"}])

        answer = await service.use_document(user_id=123, document_id=999)

        assert answer == "Документ с таким ID не найден."

    asyncio.run(run_test())


def test_list_documents_marks_active_document_once():
    async def run_test() -> None:
        service = make_document_service(
            [
                {"id": 11, "filename": "code.txt"},
                {"id": 10, "filename": "voprosy.pdf"},
            ],
            active_document={"id": 11, "filename": "code.txt"},
        )

        answer = await service.list_documents(user_id=123)

        assert "11. code.txt — активный" in answer
        assert answer.count("— активный") == 1

    asyncio.run(run_test())


def test_answer_question_passes_only_active_document_chunks_to_ollama(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        active_id = await add_document_with_chunks(
            database,
            123,
            "active.pdf",
            [
                {
                    "content": "Активный материал",
                    "embedding": [1.0, 0.0],
                    "chunk_index": 0,
                }
            ],
        )
        await add_document_with_chunks(
            database,
            123,
            "inactive.pdf",
            [
                {
                    "content": "Другой материал",
                    "embedding": [1.0, 0.0],
                    "chunk_index": 0,
                }
            ],
        )
        await database.set_active_document(123, active_id)
        ollama_client = FakeAnswerOllamaClient()
        service = EduHelperService(
            settings=SimpleNamespace(top_k=4),
            database=database,
            ollama_client=ollama_client,
            splitter=None,
            search_engine=VectorSearch(min_similarity=-1.0),
        )

        await service.answer_question(user_id=123, question="Что в материале?")

        assert ollama_client.context_chunks == ["Активный материал"]

    asyncio.run(run_test())


def test_answer_question_source_matches_active_document(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        active_id = await add_document_with_chunks(
            database,
            123,
            "active.pdf",
            [
                {
                    "content": "Активный материал",
                    "embedding": [1.0, 0.0],
                    "chunk_index": 0,
                }
            ],
        )
        await add_document_with_chunks(
            database,
            123,
            "inactive.pdf",
            [
                {
                    "content": "Другой материал",
                    "embedding": [1.0, 0.0],
                    "chunk_index": 0,
                }
            ],
        )
        await database.set_active_document(123, active_id)
        service = EduHelperService(
            settings=SimpleNamespace(top_k=4),
            database=database,
            ollama_client=FakeAnswerOllamaClient(),
            splitter=None,
            search_engine=VectorSearch(min_similarity=-1.0),
        )

        answer = await service.answer_question(user_id=123, question="Что в материале?")

        assert answer.endswith("Источник: active.pdf")

    asyncio.run(run_test())


def test_changing_active_document_changes_source_and_context(tmp_path):
    async def run_test() -> None:
        database = Database(str(tmp_path / "eduhelper.db"))
        await database.init()
        first_id = await add_document_with_chunks(
            database,
            123,
            "first.pdf",
            [
                {
                    "content": "Первый материал",
                    "embedding": [1.0, 0.0],
                    "chunk_index": 0,
                }
            ],
        )
        second_id = await add_document_with_chunks(
            database,
            123,
            "second.pdf",
            [
                {
                    "content": "Второй материал",
                    "embedding": [1.0, 0.0],
                    "chunk_index": 0,
                }
            ],
        )
        ollama_client = FakeAnswerOllamaClient()
        service = EduHelperService(
            settings=SimpleNamespace(top_k=4),
            database=database,
            ollama_client=ollama_client,
            splitter=None,
            search_engine=VectorSearch(min_similarity=-1.0),
        )

        await database.set_active_document(123, first_id)
        first_answer = await service.answer_question(user_id=123, question="Что в материале?")
        first_context = ollama_client.context_chunks
        await database.set_active_document(123, second_id)
        second_answer = await service.answer_question(user_id=123, question="Что в материале?")

        assert first_context == ["Первый материал"]
        assert first_answer.endswith("Источник: first.pdf")
        assert ollama_client.context_chunks == ["Второй материал"]
        assert second_answer.endswith("Источник: second.pdf")

    asyncio.run(run_test())


def test_list_documents_returns_empty_message_when_no_documents():
    async def run_test() -> None:
        service = make_document_service([])

        answer = await service.list_documents(user_id=123)

        assert answer == "У вас пока нет загруженных материалов."

    asyncio.run(run_test())
