import logging
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def create_dispatcher(service, database, uploads_dir: str) -> Dispatcher:
    router = Router()
    uploads_path = Path(uploads_dir)

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        await message.answer(
            "Привет! Я EduHelper AI — локальный образовательный ИИ-ассистент. "
            "Загрузи PDF, TXT или MD файл, а потом задай вопрос по материалу."
        )

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        await message.answer(
            "Как пользоваться:\n"
            "1. Отправьте PDF, TXT или MD файл.\n"
            "2. Дождитесь обработки.\n"
            "3. Задайте вопрос по материалу.\n"
            "Команды:\n"
            "/start — начало работы\n"
            "/help — помощь\n"
            "/clear — очистить загруженные материалы\n"
            "/documents — список загруженных материалов\n"
            "/use ID — выбрать активный материал\n"
            "/delete ID — удалить материал"
        )

    @router.message(Command("clear"))
    async def clear(message: Message) -> None:
        await database.clear_user_data(message.from_user.id)
        await message.answer("Ваши загруженные материалы очищены.")

    @router.message(Command("documents"))
    async def documents(message: Message) -> None:
        if message.from_user is None:
            await message.answer("Не удалось определить пользователя.")
            return

        answer = await service.list_documents(message.from_user.id)
        await message.answer(answer)

    @router.message(Command("use"))
    async def use_document(message: Message) -> None:
        if message.from_user is None:
            await message.answer("Не удалось определить пользователя.")
            return

        parts = (message.text or "").split()
        if len(parts) != 2:
            await message.answer("Использование: /use ID")
            return

        try:
            document_id = int(parts[1])
        except ValueError:
            await message.answer("Использование: /use ID")
            return

        if document_id <= 0:
            await message.answer("Использование: /use ID")
            return

        answer = await service.use_document(message.from_user.id, document_id)
        await message.answer(answer)

    @router.message(Command("delete"))
    async def delete_document(message: Message) -> None:
        if message.from_user is None:
            await message.answer("Не удалось определить пользователя.")
            return

        parts = (message.text or "").split()
        if len(parts) != 2:
            await message.answer("Использование: /delete ID")
            return

        try:
            document_id = int(parts[1])
        except ValueError:
            await message.answer("Использование: /delete ID")
            return

        if document_id <= 0:
            await message.answer("Использование: /delete ID")
            return

        answer = await service.delete_document(message.from_user.id, document_id)
        await message.answer(answer)

    @router.message(F.document)
    async def handle_document(message: Message, bot: Bot) -> None:
        document = message.document
        original_name = document.file_name or "document"
        extension = Path(original_name).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            await message.answer("Поддерживаются только файлы PDF, TXT и MD.")
            return

        uploads_path.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(original_name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        local_path = uploads_path / f"{message.from_user.id}_{timestamp}_{safe_name}"

        try:
            async with ChatActionSender.upload_document(bot=bot, chat_id=message.chat.id):
                await bot.download(document, destination=local_path)
            await message.answer("Файл получен. Обрабатываю материал...")
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                chunks_count = await service.process_file(
                    user_id=message.from_user.id,
                    file_path=str(local_path),
                    filename=original_name,
                )
            await message.answer(
                f"Файл обработан. Добавлено {chunks_count} фрагментов. "
                "Теперь можно задавать вопросы."
            )
        except ValueError as exc:
            await message.answer(str(exc))
        except RuntimeError as exc:
            await message.answer(str(exc))
        except Exception:
            logging.exception("Ошибка обработки документа")
            await message.answer("Не удалось обработать файл. Проверьте формат и попробуйте еще раз.")

    @router.message(F.text)
    async def handle_question(message: Message, bot: Bot) -> None:
        text = (message.text or "").strip()
        if not text or text.startswith("/"):
            return

        try:
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                answer = await service.answer_question(message.from_user.id, text)
            await message.answer(answer)
        except RuntimeError as exc:
            await message.answer(str(exc))
        except ValueError as exc:
            await message.answer(str(exc))
        except Exception:
            logging.exception("Ошибка ответа на вопрос")
            await message.answer("Не удалось подготовить ответ. Попробуйте позже.")

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


async def run_bot(token: str, dispatcher: Dispatcher) -> None:
    bot = Bot(token=token)
    await dispatcher.start_polling(bot)


def _safe_filename(filename: str) -> str:
    allowed = []
    for char in filename:
        if char.isalnum() or char in {".", "-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("._") or "document"
