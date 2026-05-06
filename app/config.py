from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    bot_token: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    database_path: str
    uploads_dir: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Настройка {name} должна быть целым числом.") from exc


def load_settings() -> Settings:
    load_dotenv()

    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError(
            "BOT_TOKEN не задан. Создайте .env на основе .env.example "
            "и укажите токен Telegram-бота."
        )

    return Settings(
        bot_token=bot_token,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        database_path=os.getenv("DATABASE_PATH", "data/eduhelper.db"),
        uploads_dir=os.getenv("UPLOADS_DIR", "data/uploads"),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        top_k=_get_int("TOP_K", 4),
    )
