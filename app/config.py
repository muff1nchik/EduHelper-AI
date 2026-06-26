"""Загружает настройки приложения из переменных окружения."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Хранит настройки, с которыми запускается приложение."""

    bot_token: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    ollama_temperature: float
    ollama_num_ctx: int
    database_path: str
    uploads_dir: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_similarity: float


def _get_int(name: str, default: int) -> int:
    """Читает целочисленную настройку из окружения."""
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Настройка {name} должна быть целым числом.") from exc


def _get_float(name: str, default: float) -> float:
    """Читает числовую настройку из окружения."""
    value = os.getenv(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Настройка {name} должна быть числом.") from exc


def _get_min_similarity() -> float:
    """Читает минимальную похожесть и проверяет её диапазон."""
    value = _get_float("MIN_SIMILARITY", 0.35)
    if not -1.0 <= value <= 1.0:
        raise ValueError("Настройка MIN_SIMILARITY должна быть от -1.0 до 1.0.")
    return value


def load_settings() -> Settings:
    """Загружает настройки и проверяет обязательный токен бота."""
    load_dotenv()

    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError(
            "BOT_TOKEN не задан. Создайте .env на основе .env.example "
            "и укажите токен Telegram-бота."
        )

    return Settings(
        bot_token=bot_token,
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ).rstrip("/"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_temperature=_get_float("OLLAMA_TEMPERATURE", 0.2),
        ollama_num_ctx=_get_int("OLLAMA_NUM_CTX", 4096),
        database_path=os.getenv("DATABASE_PATH", "data/eduhelper.db"),
        uploads_dir=os.getenv("UPLOADS_DIR", "data/uploads"),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        top_k=_get_int("TOP_K", 4),
        min_similarity=_get_min_similarity(),
    )
