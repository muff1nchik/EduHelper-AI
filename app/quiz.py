from dataclasses import dataclass
import re


QUIZ_DEFAULT_QUESTION_COUNT = 5
QUIZ_MIN_QUESTION_COUNT = 1
QUIZ_MAX_QUESTION_COUNT = 10
QUIZ_CONTEXT_MAX_CHARS = 8000
QUIZ_CONTEXT_MAX_CHUNKS = 10
QUIZ_MIN_CONTEXT_CHARS = 300
QUIZ_QUESTION_MAX_LENGTH = 500
QUIZ_REFERENCE_ANSWER_MAX_LENGTH = 1200
QUIZ_FEEDBACK_MAX_LENGTH = 600
QUIZ_SKIP_ANSWERS = {"пропустить", "не знаю", "дальше"}
QUIZ_INSUFFICIENT_MESSAGE = (
    "В активном документе недостаточно содержательного материала для викторины. "
    "Нужен документ, в котором есть определения, объяснения или факты, а не только список тем."
)
QUIZ_GENERATION_ERROR_MESSAGE = (
    "Не удалось корректно составить викторину. Попробуйте ещё раз."
)
QUIZ_OLLAMA_ERROR_MESSAGE = (
    "Не удалось создать викторину. Проверьте, что Ollama запущена, и попробуйте ещё раз."
)
QUIZ_EVALUATION_ERROR_MESSAGE = "Не удалось проверить ответ. Попробуйте отправить его ещё раз."


class QuizValidationError(ValueError):
    pass


@dataclass(frozen=True)
class QuizQuestion:
    question: str
    reference_answer: str
    source_chunk_ids: tuple[int, ...]
    source_context: str


@dataclass
class QuizSession:
    document_id: int
    document_name: str
    questions: list[QuizQuestion]
    current_index: int = 0
    earned_points: float = 0.0
    correct_count: int = 0
    partial_count: int = 0
    incorrect_count: int = 0
    answered_count: int = 0
    is_processing: bool = False

    @property
    def current_question(self) -> QuizQuestion:
        return self.questions[self.current_index]

    @property
    def total_questions(self) -> int:
        return len(self.questions)


def normalize_quiz_text(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def is_skip_answer(text: str) -> bool:
    return normalize_quiz_text(text) in QUIZ_SKIP_ANSWERS


def select_quiz_chunks(
    chunks: list[dict],
    max_chunks: int = QUIZ_CONTEXT_MAX_CHUNKS,
    max_chars: int = QUIZ_CONTEXT_MAX_CHARS,
) -> list[dict]:
    non_empty_chunks = [chunk for chunk in chunks if chunk.get("content")]
    if not non_empty_chunks:
        return []

    total_length = sum(len(chunk["content"]) for chunk in non_empty_chunks)
    if len(non_empty_chunks) <= max_chunks and total_length <= max_chars:
        return non_empty_chunks

    count = min(max_chunks, len(non_empty_chunks))
    if count == 1:
        candidate_indices = [0]
    else:
        candidate_indices = [
            round(index * (len(non_empty_chunks) - 1) / (count - 1))
            for index in range(count)
        ]

    selected = []
    used_indices = set()
    current_length = 0
    for index in candidate_indices:
        if index in used_indices:
            continue
        chunk = non_empty_chunks[index]
        next_length = current_length + len(chunk["content"])
        if selected and next_length > max_chars:
            continue
        if not selected and len(chunk["content"]) > max_chars:
            chunk = {**chunk, "content": chunk["content"][:max_chars]}
            next_length = len(chunk["content"])
        selected.append(chunk)
        used_indices.add(index)
        current_length = next_length
    return selected


def format_quiz_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(f"[CHUNK {index}]\n{chunk['content']}")
    return "\n\n".join(parts)


def validate_quiz_payload(
    payload: dict,
    question_count: int,
    selected_chunks: list[dict],
) -> list[QuizQuestion] | str:
    status = payload.get("status")
    if status not in {"ok", "insufficient_material"}:
        raise QuizValidationError("Некорректный status викторины.")
    if status == "insufficient_material":
        return "insufficient_material"

    questions = payload.get("questions")
    if not isinstance(questions, list) or len(questions) != question_count:
        raise QuizValidationError("Некорректное количество вопросов.")

    valid_chunk_ids = set(range(1, len(selected_chunks) + 1))
    normalized_questions = set()
    result = []
    for item in questions:
        if not isinstance(item, dict):
            raise QuizValidationError("Вопрос должен быть объектом.")
        question = str(item.get("question", "")).strip()
        reference_answer = str(item.get("reference_answer", "")).strip()
        source_chunk_ids = item.get("source_chunk_ids")
        if not question or len(question) > QUIZ_QUESTION_MAX_LENGTH:
            raise QuizValidationError("Некорректный текст вопроса.")
        if not reference_answer or len(reference_answer) > QUIZ_REFERENCE_ANSWER_MAX_LENGTH:
            raise QuizValidationError("Некорректный эталонный ответ.")
        if not isinstance(source_chunk_ids, list) or not source_chunk_ids:
            raise QuizValidationError("Не указаны фрагменты-источники.")
        if len(source_chunk_ids) != len(set(source_chunk_ids)):
            raise QuizValidationError("Повторяются фрагменты-источники.")
        if any(not isinstance(chunk_id, int) or chunk_id not in valid_chunk_ids for chunk_id in source_chunk_ids):
            raise QuizValidationError("Указан неизвестный фрагмент-источник.")

        normalized_question = normalize_quiz_text(question)
        if normalized_question in normalized_questions:
            raise QuizValidationError("Вопросы дублируются.")
        normalized_questions.add(normalized_question)

        source_context = "\n\n".join(
            selected_chunks[chunk_id - 1]["content"]
            for chunk_id in source_chunk_ids
            if selected_chunks[chunk_id - 1].get("content")
        )
        if not source_context:
            raise QuizValidationError("Пустой контекст источника.")
        result.append(
            QuizQuestion(
                question=question,
                reference_answer=reference_answer,
                source_chunk_ids=tuple(source_chunk_ids),
                source_context=source_context,
            )
        )
    return result


def validate_evaluation_payload(payload: dict) -> tuple[str, str]:
    verdict = payload.get("verdict")
    feedback = str(payload.get("feedback", "")).strip()
    if verdict not in {"correct", "partial", "incorrect"}:
        raise QuizValidationError("Некорректный verdict.")
    if not feedback or len(feedback) > QUIZ_FEEDBACK_MAX_LENGTH:
        raise QuizValidationError("Некорректный feedback.")
    return verdict, feedback


def format_question(session: QuizSession) -> str:
    return (
        f"Вопрос {session.current_index + 1}/{session.total_questions}:\n"
        f"{session.current_question.question}"
    )


def format_quiz_start(session: QuizSession) -> str:
    return (
        f"Викторина по документу «{session.document_name}».\n"
        f"Вопросов: {session.total_questions}.\n\n"
        f"{format_question(session)}"
    )


def format_quiz_result(session: QuizSession, stopped: bool = False) -> str:
    if stopped:
        return (
            "Викторина остановлена.\n"
            f"Результат: {format_points(session.earned_points)}/{session.total_questions}.\n"
            f"Отвечено на вопросов: {session.answered_count}."
        )

    percent = round(session.earned_points / session.total_questions * 100)
    return (
        "Викторина завершена!\n\n"
        f"Результат: {format_points(session.earned_points)}/{session.total_questions}\n"
        f"Процент: {percent}%\n\n"
        f"Правильных ответов: {session.correct_count}\n"
        f"Частично правильных: {session.partial_count}\n"
        f"Неправильных: {session.incorrect_count}"
    )


def format_points(points: float) -> str:
    return str(int(points)) if points.is_integer() else str(points)
