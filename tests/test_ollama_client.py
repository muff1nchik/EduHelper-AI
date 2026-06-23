import asyncio

from app.ollama_client import OllamaClient


class CapturingOllamaClient(OllamaClient):
    def __init__(self) -> None:
        super().__init__(
            base_url="http://ollama.test",
            chat_model="qwen3:8b",
            embedding_model="nomic-embed-text",
            temperature=0.35,
            num_ctx=2048,
        )
        self.payload = None

    async def _post_json(self, path: str, payload: dict) -> dict:
        self.payload = payload
        return {"message": {"content": "Ответ по контексту"}}


def test_generate_answer_prompt_restricts_answer_to_context():
    async def run_test() -> None:
        client = CapturingOllamaClient()

        await client.generate_answer("Вопрос", ["Контекст"])

        prompt = client.payload["messages"][0]["content"]
        assert "Отвечай только на основе переданного контекста" in prompt
        assert "Не используй собственные знания" in prompt

    asyncio.run(run_test())


def test_generate_answer_prompt_forbids_unsupported_additions():
    async def run_test() -> None:
        client = CapturingOllamaClient()

        await client.generate_answer("Вопрос", ["Контекст"])

        prompt = client.payload["messages"][0]["content"]
        assert "Не добавляй факты, формулы, примеры или объяснения" in prompt
        assert "которых нет в материалах" in prompt

    asyncio.run(run_test())


def test_generate_answer_prompt_requires_close_reproduction_of_items():
    async def run_test() -> None:
        client = CapturingOllamaClient()

        await client.generate_answer("Назови третий пункт", ["Контекст"])

        prompt = client.payload["messages"][0]["content"]
        assert "воспроизведи его максимально близко к исходному тексту" in prompt
        assert "не добавляй объяснение и примеры" in prompt

    asyncio.run(run_test())


def test_generate_answer_prompt_forbids_markdown_markup():
    async def run_test() -> None:
        client = CapturingOllamaClient()

        await client.generate_answer("Вопрос", ["Контекст"])

        prompt = client.payload["messages"][0]["content"]
        assert "Markdown-жирный текст через **" in prompt
        assert "заголовки с #" in prompt
        assert "обратные кавычки" in prompt
        assert "Markdown-таблицы" in prompt
        assert "HTML-разметку" in prompt

    asyncio.run(run_test())


def test_generate_answer_sends_generation_options():
    async def run_test() -> None:
        client = CapturingOllamaClient()

        await client.generate_answer("Вопрос", ["Контекст"])

        assert client.payload["options"] == {
            "temperature": 0.35,
            "num_ctx": 2048,
        }

    asyncio.run(run_test())
