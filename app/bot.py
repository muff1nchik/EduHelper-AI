import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.quiz import (
    QUIZ_DEFAULT_QUESTION_COUNT,
    QUIZ_EVALUATION_ERROR_MESSAGE,
    QUIZ_MAX_QUESTION_COUNT,
    QUIZ_MIN_QUESTION_COUNT,
    QuizSession,
    format_question,
    format_quiz_result,
    format_quiz_start,
    is_skip_answer,
)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
SEMANTIC_ROUTER_MAX_LENGTH = 180
GREETING_RESPONSE = (
    "Привет! Я помогу разобраться с учебными материалами. Загрузите документ "
    "или задайте вопрос по активному материалу. Команда /help покажет все возможности."
)
CAPABILITIES_RESPONSE = (
    "Я могу принимать PDF, TXT, MD и DOCX, отвечать на вопросы по активному "
    "материалу, создавать конспекты, показывать список документов, переключать "
    "и удалять материалы. Команда /help покажет все доступные команды."
)
USAGE_RESPONSE = (
    "Сначала загрузите PDF, TXT, MD или DOCX. Затем задавайте вопросы по "
    "активному материалу. Для списка команд используйте /help."
)
THANKS_RESPONSE = "Пожалуйста! Можете продолжать задавать вопросы по материалу."
GOODBYE_RESPONSE = (
    "До встречи! Загруженные материалы останутся доступны при следующем запуске бота."
)
INTENT_RESPONSES = {
    "greeting": GREETING_RESPONSE,
    "capabilities": CAPABILITIES_RESPONSE,
    "usage": USAGE_RESPONSE,
    "thanks": THANKS_RESPONSE,
    "farewell": GOODBYE_RESPONSE,
}
GREETING_PHRASES = {
    "привет",
    "приветик",
    "здравствуй",
    "здравствуйте",
    "доброе утро",
    "добрый день",
    "добрый вечер",
    "хай",
    "привет бот",
}
THANKS_PHRASES = {
    "спасибо",
    "спасибо большое",
    "благодарю",
    "большое спасибо",
}
FAREWELL_PHRASES = {
    "пока",
    "до свидания",
    "до встречи",
    "всего доброго",
}
USAGE_PHRASES = {
    "как пользоваться ботом",
    "покажи инструкцию",
    "какие команды есть",
    "с чего начать",
}
CAPABILITIES_PHRASES = {
    "что ты умеешь",
    "чем ты можешь помочь",
}


def create_dispatcher(service, database, uploads_dir: str) -> Dispatcher:
    router = Router()
    uploads_path = Path(uploads_dir)
    quiz_sessions: dict[int, QuizSession] = {}
    quiz_starting: set[int] = set()

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        await message.answer(
            "Привет! Я EduHelper AI — локальный образовательный ИИ-ассистент. "
            "Загрузи PDF, TXT, MD или DOCX файл, задай вопрос по материалу "
            "или запусти викторину командой /quiz."
        )

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        await message.answer(
            "Как пользоваться:\n"
            "1. Отправьте PDF, TXT, MD или DOCX файл.\n"
            "2. Дождитесь обработки.\n"
            "3. Задайте вопрос по материалу.\n"
            "Команды:\n"
            "/start — начало работы\n"
            "/help — помощь\n"
            "/clear — очистить загруженные материалы\n"
            "/documents — список загруженных материалов\n"
            "/use ID — выбрать активный материал\n"
            "/delete ID — удалить материал\n"
            "/summary — создать конспект активного материала\n"
            "/quiz [N] — начать викторину по активному документу, по умолчанию 5 вопросов\n"
            "/stopquiz — остановить текущую викторину"
        )

    @router.message(Command("clear"))
    async def clear(message: Message) -> None:
        answer = await service.clear_user_data(message.from_user.id)
        _clear_quiz_state(message.from_user.id, quiz_sessions, quiz_starting)
        await message.answer(answer)

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
        if answer.startswith("Активный материал:"):
            _clear_quiz_state(message.from_user.id, quiz_sessions, quiz_starting)
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

        session = quiz_sessions.get(message.from_user.id)
        answer = await service.delete_document(message.from_user.id, document_id)
        if answer.startswith("Материал удалён:") and session and session.document_id == document_id:
            _clear_quiz_state(message.from_user.id, quiz_sessions, quiz_starting)
        await message.answer(answer)

    @router.message(Command("quiz"))
    async def quiz(message: Message, bot: Bot) -> None:
        if message.from_user is None:
            await message.answer("Не удалось определить пользователя.")
            return

        user_id = message.from_user.id
        question_count = _parse_quiz_question_count(message.text or "")
        if question_count is None:
            await message.answer("Использование: /quiz [количество вопросов от 1 до 10].")
            return
        if user_id in quiz_starting:
            await message.answer("Викторина уже создаётся. Подождите немного.")
            return
        if user_id in quiz_sessions:
            session = quiz_sessions[user_id]
            await message.answer(
                "Викторина уже идёт. Сейчас вопрос "
                f"{session.current_index + 1}/{session.total_questions}.\n"
                "Чтобы завершить её, используйте /stopquiz."
            )
            return

        quiz_starting.add(user_id)
        try:
            await message.answer("Готовлю викторину по активному документу…")
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                result = await service.generate_quiz(user_id, question_count)
            if isinstance(result, QuizSession):
                quiz_sessions[user_id] = result
                await send_long_message(message, format_quiz_start(result))
            else:
                await send_long_message(message, result)
        finally:
            quiz_starting.discard(user_id)

    @router.message(Command("stopquiz"))
    async def stop_quiz(message: Message) -> None:
        if message.from_user is None:
            await message.answer("Не удалось определить пользователя.")
            return

        session = quiz_sessions.pop(message.from_user.id, None)
        quiz_starting.discard(message.from_user.id)
        if session is None:
            await message.answer("Активной викторины нет.")
            return
        logging.info("Викторина остановлена: user_id=%s", message.from_user.id)
        await message.answer(format_quiz_result(session, stopped=True))

    @router.message(Command("summary"))
    async def summary(message: Message, bot: Bot) -> None:
        if message.from_user is None:
            await message.answer("Не удалось определить пользователя.")
            return

        if len((message.text or "").split()) != 1:
            await message.answer("Использование: /summary")
            return

        try:
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                answer = await service.summarize_document(message.from_user.id)
            await send_long_message(message, answer)
        except RuntimeError as exc:
            await message.answer(str(exc))
        except ValueError as exc:
            await message.answer(str(exc))
        except Exception:
            logging.exception("Ошибка создания конспекта")
            await message.answer("Не удалось подготовить ответ. Попробуйте позже.")

    @router.message(F.document)
    async def handle_document(message: Message, bot: Bot) -> None:
        document = message.document
        original_name = document.file_name or "document"
        extension = Path(original_name).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            await message.answer("Поддерживаются только файлы PDF, TXT, MD и DOCX.")
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
            stopped_quiz = _clear_quiz_state(message.from_user.id, quiz_sessions, quiz_starting)
            suffix = "\nПредыдущая викторина остановлена." if stopped_quiz else ""
            await message.answer(
                f"Файл обработан. Добавлено {chunks_count} фрагментов. "
                f"Теперь можно задавать вопросы.{suffix}"
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

        if message.from_user.id in quiz_starting:
            await message.answer("Викторина ещё создаётся. Подождите немного.")
            return
        if message.from_user.id in quiz_sessions:
            await _handle_quiz_answer(
                message,
                service,
                quiz_sessions,
                message.from_user.id,
                text,
            )
            return

        builtin_response = get_builtin_response(text)
        if builtin_response is not None:
            await send_long_message(message, builtin_response)
            return
        query_embedding = None
        if is_builtin_intent_candidate(text):
            route = await service.route_text_semantic(text)
            query_embedding = route.query_embedding
            builtin_response = INTENT_RESPONSES.get(route.intent)
            if builtin_response is not None:
                await send_long_message(message, builtin_response)
                return

        try:
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                answer = await service.answer_question(
                    message.from_user.id,
                    text,
                    query_embedding=query_embedding,
                )
            await send_long_message(message, answer)
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


async def _handle_quiz_answer(
    message: Message,
    service,
    quiz_sessions: dict[int, QuizSession],
    user_id: int,
    text: str,
) -> None:
    session = quiz_sessions.get(user_id)
    if session is None:
        return
    if session.is_processing:
        await message.answer("Предыдущий ответ ещё проверяется. Подождите немного.")
        return

    question_index = session.current_index
    document_id = session.document_id
    question = session.current_question

    if is_skip_answer(text):
        _apply_quiz_verdict(session, "incorrect")
        answer = (
            "Вопрос пропущен.\n\n"
            "Правильный ответ:\n"
            f"{question.reference_answer}"
        )
        await _send_quiz_progress(message, quiz_sessions, user_id, session, answer)
        return

    session.is_processing = True
    try:
        result = await service.evaluate_quiz_answer(
            question.question,
            question.reference_answer,
            question.source_context,
            text,
        )
    finally:
        current_session = quiz_sessions.get(user_id)
        if current_session is session:
            session.is_processing = False

    current_session = quiz_sessions.get(user_id)
    if (
        current_session is not session
        or session.document_id != document_id
        or session.current_index != question_index
    ):
        return
    if result == QUIZ_EVALUATION_ERROR_MESSAGE:
        await message.answer(QUIZ_EVALUATION_ERROR_MESSAGE)
        return

    verdict, feedback = result
    _apply_quiz_verdict(session, verdict)
    await _send_quiz_progress(
        message,
        quiz_sessions,
        user_id,
        session,
        _format_quiz_feedback(verdict, feedback, question.reference_answer),
    )


async def _send_quiz_progress(
    message: Message,
    quiz_sessions: dict[int, QuizSession],
    user_id: int,
    session: QuizSession,
    prefix: str,
) -> None:
    if session.current_index >= session.total_questions:
        quiz_sessions.pop(user_id, None)
        await send_long_message(message, f"{prefix}\n\n{format_quiz_result(session)}")
        return
    await send_long_message(message, f"{prefix}\n\n{format_question(session)}")


def _apply_quiz_verdict(session: QuizSession, verdict: str) -> None:
    if verdict == "correct":
        session.earned_points += 1.0
        session.correct_count += 1
    elif verdict == "partial":
        session.earned_points += 0.5
        session.partial_count += 1
    else:
        session.incorrect_count += 1
    session.answered_count += 1
    session.current_index += 1


def _format_quiz_feedback(verdict: str, feedback: str, reference_answer: str) -> str:
    if verdict == "correct":
        return f"Верно!\n\n{feedback}"
    if verdict == "partial":
        return (
            f"Частично верно.\n\n{feedback}\n\n"
            "Эталонный ответ:\n"
            f"{reference_answer}"
        )
    return (
        f"Неверно.\n\n{feedback}\n\n"
        "Правильный ответ:\n"
        f"{reference_answer}"
    )


def _parse_quiz_question_count(text: str) -> int | None:
    parts = text.split()
    if len(parts) == 1:
        return QUIZ_DEFAULT_QUESTION_COUNT
    if len(parts) != 2:
        return None
    try:
        question_count = int(parts[1])
    except ValueError:
        return None
    if not QUIZ_MIN_QUESTION_COUNT <= question_count <= QUIZ_MAX_QUESTION_COUNT:
        return None
    return question_count


def _clear_quiz_state(
    user_id: int,
    quiz_sessions: dict[int, QuizSession],
    quiz_starting: set[int],
) -> bool:
    existed = user_id in quiz_sessions or user_id in quiz_starting
    quiz_sessions.pop(user_id, None)
    quiz_starting.discard(user_id)
    return existed


async def run_bot(token: str, dispatcher: Dispatcher) -> None:
    bot = Bot(token=token)
    await dispatcher.start_polling(bot)


async def send_long_message(message: Message, text: str) -> None:
    for part in split_message(text):
        await message.answer(part)


def split_message(text: str, max_length: int = 4000) -> list[str]:
    remaining = text.strip()
    if not remaining:
        return []

    parts: list[str] = []
    while remaining:
        if len(remaining) <= max_length:
            parts.append(remaining)
            break

        split_at = _find_message_split(remaining, max_length)
        part = remaining[:split_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].strip()

    return parts


def _find_message_split(text: str, max_length: int) -> int:
    min_boundary = max(1, max_length // 2)
    window = text[:max_length]

    for separator in ("\n\n", "\n"):
        index = window.rfind(separator, min_boundary)
        if index != -1:
            return index + len(separator)

    for index in range(max_length - 1, min_boundary - 1, -1):
        if window[index] in ".!?…":
            next_index = index + 1
            if next_index == len(text) or text[next_index].isspace():
                return next_index

    index = window.rfind(" ", min_boundary)
    if index != -1:
        return index

    return max_length


def get_builtin_response(text: str) -> str | None:
    intent = detect_builtin_intent(text)
    if intent is None:
        return None
    return INTENT_RESPONSES[intent]


def detect_builtin_intent(text: str) -> str | None:
    normalized = _normalize_builtin_text(text)
    if not normalized:
        return None
    if normalized in GREETING_PHRASES:
        return "greeting"
    if normalized in THANKS_PHRASES:
        return "thanks"
    if normalized in FAREWELL_PHRASES:
        return "farewell"
    if normalized in USAGE_PHRASES:
        return "usage"
    if normalized in CAPABILITIES_PHRASES:
        return "capabilities"
    return None


def is_builtin_intent_candidate(text: str) -> bool:
    normalized = _normalize_builtin_text(text)
    return bool(normalized) and len(normalized) <= SEMANTIC_ROUTER_MAX_LENGTH


def _normalize_builtin_text(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _safe_filename(filename: str) -> str:
    allowed = []
    for char in filename:
        if char.isalnum() or char in {".", "-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("._") or "document"
