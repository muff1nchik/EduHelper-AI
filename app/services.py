from pathlib import Path

from app.loaders import get_loader
from app.messages import INSUFFICIENT_INFORMATION_MESSAGE


class EduHelperService:
    def __init__(
        self,
        settings,
        database,
        ollama_client,
        splitter,
        search_engine,
    ) -> None:
        self.settings = settings
        self.database = database
        self.ollama_client = ollama_client
        self.splitter = splitter
        self.search_engine = search_engine

    async def process_file(self, user_id: int, file_path: str, filename: str) -> int:
        saved = False
        try:
            loader = get_loader(file_path)
            text = loader.load_text(file_path)
            chunks = self.splitter.split(text)
            if not chunks:
                raise ValueError("Документ пустой после очистки текста.")

            prepared_chunks = []
            for index, chunk in enumerate(chunks):
                embedding = await self.ollama_client.embed(chunk)
                prepared_chunks.append(
                    {
                        "content": chunk,
                        "embedding": embedding,
                        "chunk_index": index,
                    }
                )

            await self.database.add_document_with_chunks(
                user_id=user_id,
                filename=filename,
                file_path=file_path,
                chunks=prepared_chunks,
            )
            saved = True
            return len(chunks)
        except Exception:
            if not saved:
                _delete_file(Path(file_path))
            raise

    async def answer_question(self, user_id: int, question: str) -> str:
        active_document = await self.database.get_active_document(user_id)
        if active_document is None:
            return "Сначала загрузите учебный файл."

        query_embedding = await self.ollama_client.embed(question)
        chunks = await self.database.get_document_chunks(user_id, active_document["id"])
        results = self.search_engine.search(chunks, query_embedding, self.settings.top_k)
        if not results:
            return INSUFFICIENT_INFORMATION_MESSAGE

        context_chunks = [result["content"] for result in results if result.get("content")]
        if not context_chunks:
            return INSUFFICIENT_INFORMATION_MESSAGE
        answer = await self.ollama_client.generate_answer(question, context_chunks)
        if answer.strip() == INSUFFICIENT_INFORMATION_MESSAGE:
            return INSUFFICIENT_INFORMATION_MESSAGE

        sources = _format_sources(results)
        if sources:
            return f"{answer}\n\n{sources}"
        return answer

    async def use_document(self, user_id: int, document_id: int) -> str:
        document = await self.database.set_active_document(user_id, document_id)
        if document is None:
            return "Документ с таким ID не найден."
        return f"Активный материал: {document['filename']}"

    async def delete_document(self, user_id: int, document_id: int) -> str:
        document = await self.database.delete_document(user_id, document_id)
        if document is None:
            return "Документ с таким ID не найден."

        if not _delete_file(Path(document["file_path"])):
            return (
                f"Материал удалён: {document['filename']}. "
                "Но файл на диске удалить не удалось."
            )
        return f"Материал удалён: {document['filename']}"

    async def clear_user_data(self, user_id: int) -> str:
        file_paths = await self.database.clear_user_data(user_id)
        if _delete_files(file_paths):
            return "Ваши загруженные материалы очищены."
        return "Материалы очищены, но не удалось удалить некоторые файлы с диска."

    async def list_documents(self, user_id: int) -> str:
        documents = await self.database.get_user_documents(user_id)
        if not documents:
            return "У вас пока нет загруженных материалов."

        active_document = await self.database.get_active_document(user_id)
        active_document_id = active_document["id"] if active_document else None
        lines = ["Ваши загруженные материалы:", ""]
        for document in documents:
            line = f"{document['id']}. {document['filename']}"
            if document["id"] == active_document_id:
                line = f"{line} — активный"
            lines.append(line)
        return "\n".join(lines)


def _format_sources(results: list[dict]) -> str:
    filenames = []
    for result in results:
        filename = result.get("filename")
        if filename and filename not in filenames:
            filenames.append(filename)

    if not filenames:
        return ""
    if len(filenames) == 1:
        return f"Источник: {filenames[0]}"

    sources = "\n".join(f"- {filename}" for filename in filenames)
    return f"Источники:\n{sources}"


def _delete_file(file_path: Path) -> bool:
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
    except OSError:
        return False
    return True


def _delete_files(file_paths: list[str]) -> bool:
    success = True
    seen_paths = set()
    for file_path in file_paths:
        if file_path in seen_paths:
            continue
        seen_paths.add(file_path)
        if not _delete_file(Path(file_path)):
            success = False
    return success
