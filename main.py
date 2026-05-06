import asyncio
import logging
from pathlib import Path
import sys

from app.bot import create_dispatcher, run_bot
from app.config import load_settings
from app.database import Database
from app.ollama_client import OllamaClient
from app.search import VectorSearch
from app.services import EduHelperService
from app.splitter import TextSplitter


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    settings = load_settings()
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)

    database = Database(settings.database_path)
    await database.init()

    ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        chat_model=settings.ollama_chat_model,
        embedding_model=settings.ollama_embedding_model,
    )
    splitter = TextSplitter(settings.chunk_size, settings.chunk_overlap)
    search_engine = VectorSearch()
    service = EduHelperService(
        settings=settings,
        database=database,
        ollama_client=ollama_client,
        splitter=splitter,
        search_engine=search_engine,
    )
    dispatcher = create_dispatcher(service, database, settings.uploads_dir)
    await run_bot(settings.bot_token, dispatcher)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
