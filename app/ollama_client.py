import httpx

from app.messages import INSUFFICIENT_INFORMATION_MESSAGE


OLLAMA_ERROR = (
    "Не удалось подключиться к Ollama. Проверьте, что Ollama запущена "
    "и модели установлены."
)

SYSTEM_PROMPT = """
Ты образовательный ассистент EduHelper AI.
Отвечай только на основе переданного контекста из загруженных материалов.
Не используй собственные знания, если ответа нет в контексте.
Не добавляй факты, формулы, примеры или объяснения, которых нет в материалах.
Если пользователь просит найти, назвать, перечислить или переписать конкретный
пункт документа, воспроизведи его максимально близко к исходному тексту.
Если пользователь просит только назвать пункт, не добавляй объяснение и примеры.
Если пользователь отдельно просит объяснить материал или привести пример,
пояснение разрешено, но оно должно основываться только на переданном контексте.
Если информации недостаточно, ответь ровно так:
{insufficient_information_message}
Ответ должен быть обычным текстом для Telegram.
Не используй Markdown-жирный текст через **, заголовки с #, обратные кавычки,
Markdown-таблицы и HTML-разметку.
""".format(
    insufficient_information_message=INSUFFICIENT_INFORMATION_MESSAGE
).strip()


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        temperature: float = 0.2,
        num_ctx: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.num_ctx = num_ctx
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
        user_prompt = (
            f"Контекст из учебных материалов:\n{context}\n\n"
            f"Вопрос пользователя:\n{question}"
        )
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
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
