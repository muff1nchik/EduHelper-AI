from app.loaders import get_loader


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
        loader = get_loader(file_path)
        text = loader.load_text(file_path)
        chunks = self.splitter.split(text)
        if not chunks:
            raise ValueError("Документ пустой после очистки текста.")

        document_id = await self.database.add_document(user_id, filename, file_path)
        for index, chunk in enumerate(chunks):
            embedding = await self.ollama_client.embed(chunk)
            await self.database.add_chunk(
                document_id=document_id,
                user_id=user_id,
                content=chunk,
                embedding=embedding,
                chunk_index=index,
            )

        return len(chunks)

    async def answer_question(self, user_id: int, question: str) -> str:
        if not await self.database.user_has_chunks(user_id):
            return "Сначала загрузите учебный файл."

        query_embedding = await self.ollama_client.embed(question)
        chunks = await self.database.get_user_chunks(user_id)
        results = self.search_engine.search(chunks, query_embedding, self.settings.top_k)
        context_chunks = [result["content"] for result in results if result.get("content")]
        if not context_chunks:
            return "Не удалось найти релевантные фрагменты в загруженных материалах."
        return await self.ollama_client.generate_answer(question, context_chunks)
