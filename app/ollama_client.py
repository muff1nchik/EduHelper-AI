import httpx


OLLAMA_ERROR = (
    "Не удалось подключиться к Ollama. Проверьте, что Ollama запущена "
    "и модели установлены."
)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "input": text}
        data = await self._post_json("/api/embed", payload)

        if "embeddings" in data and data["embeddings"]:
            embedding = data["embeddings"][0]
        elif "embedding" in data:
            embedding = data["embedding"]
        else:
            raise RuntimeError("Ollama вернула неизвестный формат embedding-ответа.")

        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Ollama вернула пустой embedding.")
        return [float(value) for value in embedding]

    async def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(
            f"Фрагмент {index + 1}:\n{chunk}"
            for index, chunk in enumerate(context_chunks)
        )
        system_prompt = (
            "Ты образовательный ассистент EduHelper AI. Отвечай на русском языке, "
            "понятно и по шагам. Опирайся только на предоставленные фрагменты. "
            "Если точного ответа нет в материалах, честно скажи, что он не найден. "
            "Не выдумывай факты, не делай ответ слишком длинным. При необходимости "
            "приведи простой пример."
        )
        user_prompt = (
            f"Контекст из учебных материалов:\n{context}\n\n"
            f"Вопрос пользователя:\n{question}"
        )
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        data = await self._post_json("/api/chat", payload)

        message = data.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            raise RuntimeError("Ollama вернула пустой ответ.")
        return str(content).strip()

    async def _post_json(self, path: str, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}{path}", json=payload)
                response.raise_for_status()
                return response.json()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            raise RuntimeError(OLLAMA_ERROR) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(OLLAMA_ERROR) from exc
