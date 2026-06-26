"""Хранит структуры и проверки для интерактивной викторины."""

import re
import unicodedata
from dataclasses import dataclass

from app.text_utils import clean_model_output, tokenize_for_search


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
    "Нужен документ, в котором есть определения, объяснения или факты, "
    "а не только список тем."
)
QUIZ_GENERATION_ERROR_MESSAGE = (
    "Не удалось корректно составить викторину. Попробуйте ещё раз."
)
QUIZ_OLLAMA_ERROR_MESSAGE = (
    "Не удалось создать викторину. Проверьте, что Ollama запущена, "
    "и попробуйте ещё раз."
)
QUIZ_EVALUATION_ERROR_MESSAGE = (
    "Не удалось проверить ответ. Попробуйте отправить его ещё раз."
)


class QuizValidationError(ValueError):
    """Возникает, если вопрос викторины не проходит проверку."""

    pass


@dataclass(frozen=True)
class QuizPayloadValidation:
    """Хранит результат частичной проверки вопросов викторины."""

    questions: list["QuizQuestion"]
    rejected_reasons: list[str]
    insufficient_material: bool = False


@dataclass(frozen=True)
class QuizQuestion:
    """Хранит один вопрос викторины и его источник."""

    question: str
    reference_answer: str
    source_chunk_ids: tuple[int, ...]
    source_context: str
    evidence_quote: str = ""


@dataclass
class QuizSession:
    """Хранит состояние текущей викторины пользователя."""

    document_id: int
    document_name: str
    questions: list[QuizQuestion]
    requested_question_count: int = 0
    current_index: int = 0
    earned_points: float = 0.0
    correct_count: int = 0
    partial_count: int = 0
    incorrect_count: int = 0
    answered_count: int = 0
    is_processing: bool = False

    @property
    def current_question(self) -> QuizQuestion:
        """Возвращает текущий вопрос викторины."""
        return self.questions[self.current_index]

    @property
    def total_questions(self) -> int:
        """Возвращает количество вопросов в текущей викторине."""
        return len(self.questions)


def normalize_quiz_text(text: str) -> str:
    """Нормализует короткий текст для команд викторины."""
    normalized = text.lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def is_skip_answer(text: str) -> bool:
    """Проверяет, хочет ли пользователь пропустить вопрос."""
    return normalize_quiz_text(text) in QUIZ_SKIP_ANSWERS


def select_quiz_chunks(
    chunks: list[dict],
    max_chunks: int = QUIZ_CONTEXT_MAX_CHUNKS,
    max_chars: int = QUIZ_CONTEXT_MAX_CHARS,
) -> list[dict]:
    """Выбирает содержательные фрагменты документа для генерации викторины."""
    strict_candidates = [
        (index, chunk, score_quiz_chunk(chunk.get("content", "")))
        for index, chunk in enumerate(chunks)
        if chunk.get("content")
    ]
    candidates = [item for item in strict_candidates if item[2] > 0]
    if len(candidates) < min(2, max_chunks):
        candidates = _fallback_quiz_candidates(chunks)
    if not candidates:
        return []
    non_empty_chunks = [chunk for _index, chunk, _score in candidates]
    total_length = sum(len(chunk["content"]) for chunk in non_empty_chunks)
    if len(non_empty_chunks) <= max_chunks and total_length <= max_chars:
        return non_empty_chunks

    selected_indices: set[int] = set()
    selected: list[dict] = []
    ranges = min(max_chunks, max(1, len(candidates)))
    for range_index in range(ranges):
        start = range_index * len(chunks) / ranges
        end = (range_index + 1) * len(chunks) / ranges
        range_candidates = [
            item
            for item in candidates
            if start <= item[0] < end and item[0] not in selected_indices
        ]
        if range_index == ranges - 1:
            range_candidates.sort(key=lambda item: (item[2], item[0]), reverse=True)
        else:
            range_candidates.sort(key=lambda item: (-item[2], item[0]))
        for index, chunk, _score in range_candidates:
            if _is_near_duplicate(chunk, selected):
                continue
            selected.append(chunk)
            selected_indices.add(index)
            break
        if len(selected) >= max_chunks:
            break

    if len(selected) < min(max_chunks, len(candidates)):
        for index, chunk, _score in sorted(
            candidates,
            key=lambda item: item[2],
            reverse=True,
        ):
            if len(selected) >= max_chunks:
                break
            if index in selected_indices or _is_near_duplicate(chunk, selected):
                continue
            selected.append(chunk)
            selected_indices.add(index)

    selected = _add_quiz_neighbors(chunks, selected, max_chunks, max_chars)
    selected.sort(
        key=lambda chunk: chunk.get(
            "chunk_index",
            chunks.index(chunk) if chunk in chunks else 0,
        )
    )
    return _limit_quiz_context(selected, max_chars)


def score_quiz_chunk(content: str) -> float:
    """Оценивает, насколько фрагмент подходит для вопросов."""
    text = (content or "").strip()
    tokens = tokenize_for_search(text)
    if len(text) < 80 or len(tokens) < 8:
        return 0.0
    if _looks_like_heading_only(text) or _looks_like_table_of_contents(text):
        return 0.0
    symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    if symbol_count / max(1, len(text)) > 0.45:
        return 0.0

    sentences = re.split(r"(?<=[.!?…])\s+", text)
    full_sentences = [
        sentence
        for sentence in sentences
        if len(tokenize_for_search(sentence)) >= 5
    ]
    score = len(tokens) / 40
    score += min(3, len(full_sentences)) * 0.8
    lowered = text.casefold()
    for marker in (
        "называется", "является", "определ", "правило", "услов", "если",
        "потому", "поэтому", "следовательно", "пример", "использ", "позвол",
    ):
        if marker in lowered:
            score += 0.7
    if "\n\n" in text:
        score += 0.5
    return score


def format_quiz_context(chunks: list[dict]) -> str:
    """Формирует SOURCE-контекст для генерации вопросов."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(f"[SOURCE {index}]\n{chunk['content']}")
    return "\n\n".join(parts)


def validate_quiz_payload(
    payload: dict,
    question_count: int,
    selected_chunks: list[dict],
) -> list[QuizQuestion] | str:
    """Проверяет полный ответ модели для старого сценария генерации."""
    validation = validate_quiz_payload_partial(payload, selected_chunks)
    if validation.insufficient_material:
        return "insufficient_material"
    if len(validation.questions) != question_count or validation.rejected_reasons:
        reason = (
            validation.rejected_reasons[0]
            if validation.rejected_reasons
            else "invalid_schema"
        )
        raise QuizValidationError(reason)
    return validation.questions


def validate_quiz_payload_partial(
    payload: dict,
    selected_chunks: list[dict],
    existing_questions: list[QuizQuestion] | None = None,
) -> QuizPayloadValidation:
    """Проверяет вопросы по одному и сохраняет валидные."""
    status = payload.get("status")
    if status not in {"ok", "insufficient_material"}:
        return QuizPayloadValidation([], ["invalid_schema"])
    if status == "insufficient_material":
        return QuizPayloadValidation([], [], insufficient_material=True)

    questions = payload.get("questions")
    if not isinstance(questions, list):
        return QuizPayloadValidation([], ["invalid_schema"])

    normalized_questions = {
        normalize_quiz_text(question.question)
        for question in (existing_questions or [])
    }
    rejected_reasons: list[str] = []
    result = []
    for item in questions:
        question, reason = _validate_quiz_item(item, selected_chunks)
        if reason is not None:
            rejected_reasons.append(reason)
            continue
        assert question is not None
        normalized_question = normalize_quiz_text(question.question)
        if normalized_question in normalized_questions:
            rejected_reasons.append("duplicate_question")
            continue
        normalized_questions.add(normalized_question)
        result.append(question)
    return QuizPayloadValidation(result, rejected_reasons)


def _validate_quiz_item(
    item: object,
    selected_chunks: list[dict],
) -> tuple[QuizQuestion | None, str | None]:
    """Проверяет один вопрос и собирает его источник."""
    if not isinstance(item, dict):
        return None, "invalid_schema"

    question = str(item.get("question", "")).strip()
    reference_answer = clean_model_output(str(item.get("reference_answer", "")))
    if not question or len(question) > QUIZ_QUESTION_MAX_LENGTH:
        return None, "empty_question"
    if not reference_answer or len(reference_answer) > QUIZ_REFERENCE_ANSWER_MAX_LENGTH:
        return None, "empty_reference_answer"

    parsed_source_ids = _parse_source_ids(
        item.get("source_ids", item.get("source_chunk_ids")),
        len(selected_chunks),
    )
    resolved_source_ids = _resolve_source_ids(
        question,
        reference_answer,
        selected_chunks,
    )
    source_ids = parsed_source_ids
    if resolved_source_ids and (
        not parsed_source_ids
        or _source_overlap_score(
            resolved_source_ids,
            question,
            reference_answer,
            selected_chunks,
        )
        > _source_overlap_score(
            parsed_source_ids,
            question,
            reference_answer,
            selected_chunks,
        )
    ):
        source_ids = resolved_source_ids
    if not source_ids:
        return None, "source_unresolved"

    source_context = "\n\n".join(
        selected_chunks[source_id - 1]["content"]
        for source_id in source_ids
        if selected_chunks[source_id - 1].get("content")
    )
    if not source_context:
        return None, "source_unresolved"
    evidence_quote = _build_evidence_quote(question, reference_answer, source_context)
    if not evidence_quote:
        return None, "evidence_unresolved"
    reference_reason = _reference_answer_rejection_reason(
        question,
        reference_answer,
        evidence_quote,
        source_context,
    )
    if reference_reason is not None:
        return None, reference_reason
    return (
        QuizQuestion(
            question=question,
            reference_answer=reference_answer,
            source_chunk_ids=tuple(source_ids),
            source_context=source_context,
            evidence_quote=evidence_quote,
        ),
        None,
    )


def validate_evaluation_payload(payload: dict) -> tuple[str, str]:
    """Проверяет structured-ответ оценки пользовательского ответа."""
    verdict = payload.get("verdict")
    feedback = str(payload.get("feedback", "")).strip()
    if verdict not in {"correct", "partial", "incorrect"}:
        raise QuizValidationError("Некорректный verdict.")
    if not feedback or len(feedback) > QUIZ_FEEDBACK_MAX_LENGTH:
        raise QuizValidationError("Некорректный feedback.")
    return verdict, clean_model_output(feedback)


def format_question(session: QuizSession) -> str:
    """Форматирует текущий вопрос викторины."""
    return (
        f"Вопрос {session.current_index + 1}/{session.total_questions}:\n"
        f"{session.current_question.question}"
    )


def format_quiz_start(session: QuizSession) -> str:
    """Форматирует первое сообщение после создания викторины."""
    prefix = ""
    if (
        session.requested_question_count
        and session.total_questions < session.requested_question_count
    ):
        prefix = (
            f"Удалось подготовить {session.total_questions} "
            "качественных вопросов.\n\n"
        )
    return (
        f"{prefix}"
        f"Викторина по документу «{session.document_name}».\n"
        f"Вопросов: {session.total_questions}.\n\n"
        f"{format_question(session)}"
    )


def format_quiz_result(session: QuizSession, stopped: bool = False) -> str:
    """Форматирует итог или остановку викторины."""
    if stopped:
        return (
            "Викторина остановлена.\n"
            f"Результат: {format_points(session.earned_points)}/"
            f"{session.total_questions}.\n"
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
    """Показывает целые баллы без лишней дробной части."""
    return str(int(points)) if points.is_integer() else str(points)


def _looks_like_heading_only(text: str) -> bool:
    """Проверяет, похож ли фрагмент только на заголовок."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (
        len(lines) == 1
        and len(lines[0]) < 90
        and not lines[0].endswith((".", "!", "?", "…"))
    )


def _looks_like_table_of_contents(text: str) -> bool:
    """Проверяет, похож ли фрагмент на оглавление."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    short_lines = sum(1 for line in lines if len(line) < 90)
    dotted = sum(1 for line in lines if re.search(r"\.{3,}\s*\d+$", line))
    return short_lines / len(lines) > 0.8 and (dotted or len(lines) > 8)


def _fallback_quiz_candidates(chunks: list[dict]) -> list[tuple[int, dict, float]]:
    """Выбирает запасные фрагменты, если строгая оценка слишком жёсткая."""
    candidates = []
    for index, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        if not _is_usable_quiz_context(content):
            continue
        tokens = tokenize_for_search(content)
        sentence_count = len(
            [
                sentence
                for sentence in re.split(r"(?<=[.!?…])\s+", content)
                if len(tokenize_for_search(sentence)) >= 4
            ]
        )
        score = max(
            score_quiz_chunk(content),
            len(tokens) / 80 + min(sentence_count, 3) * 0.4,
        )
        candidates.append((index, chunk, score))
    return candidates


def _is_usable_quiz_context(content: str) -> bool:
    """Проверяет, можно ли использовать фрагмент как контекст викторины."""
    text = (content or "").strip()
    if len(text) < 60:
        return False
    tokens = tokenize_for_search(text)
    if len(tokens) < 5:
        return False
    if re.fullmatch(r"\d+([.\-–—]\d+)*", text):
        return False
    if _looks_like_heading_only(text) or _looks_like_table_of_contents(text):
        return False
    symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return symbol_count / max(1, len(text)) <= 0.55


def _is_near_duplicate(chunk: dict, selected: list[dict]) -> bool:
    """Проверяет, похож ли фрагмент на уже выбранные."""
    content = _normalize_space(chunk.get("content", ""))
    tokens = set(tokenize_for_search(chunk.get("content", "")))
    if not tokens:
        return True
    for item in selected:
        if content == _normalize_space(item.get("content", "")):
            return True
        other = set(tokenize_for_search(item.get("content", "")))
        if other and len(tokens & other) / len(tokens | other) > 0.95:
            return True
    return False


def _add_quiz_neighbors(
    chunks: list[dict],
    selected: list[dict],
    max_chunks: int,
    max_chars: int,
) -> list[dict]:
    """Добавляет соседние фрагменты к выбранному quiz-контексту."""
    by_index = {
        chunk.get("chunk_index", index): chunk
        for index, chunk in enumerate(chunks)
    }
    result = list(selected)
    used = {id(chunk) for chunk in result}
    for chunk in list(selected):
        if len(result) >= max_chunks:
            break
        chunk_index = chunk.get("chunk_index")
        if chunk_index is None:
            continue
        for neighbor_index in (chunk_index - 1, chunk_index + 1):
            neighbor = by_index.get(neighbor_index)
            if not neighbor or id(neighbor) in used:
                continue
            if _is_near_duplicate(neighbor, result):
                continue
            if (
                sum(len(item.get("content", "")) for item in result)
                + len(neighbor.get("content", ""))
                > max_chars
            ):
                continue
            result.append(neighbor)
            used.add(id(neighbor))
            if len(result) >= max_chunks:
                break
    return result


def _limit_quiz_context(chunks: list[dict], max_chars: int) -> list[dict]:
    """Ограничивает quiz-контекст по общей длине."""
    result = []
    total = 0
    for chunk in chunks:
        length = len(chunk.get("content", ""))
        if result and total + length > max_chars:
            continue
        result.append(chunk)
        total += length
    return result


def _parse_source_ids(value: object, source_count: int) -> tuple[int, ...]:
    """Читает локальные номера SOURCE из ответа модели."""
    if not isinstance(value, list):
        return ()
    result = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            source_id = item
        elif isinstance(item, str) and item.strip().isdigit():
            source_id = int(item.strip())
        else:
            continue
        if 1 <= source_id <= source_count and source_id not in result:
            result.append(source_id)
    return tuple(result)


def _resolve_source_ids(
    question: str,
    reference_answer: str,
    selected_chunks: list[dict],
) -> tuple[int, ...]:
    """Подбирает источник вопроса по пересечению значимых слов."""
    query_tokens = _significant_quiz_tokens(f"{question} {reference_answer}")
    if not query_tokens:
        return ()
    scored = []
    for source_id, chunk in enumerate(selected_chunks, start=1):
        source_tokens = _significant_quiz_tokens(chunk.get("content", ""))
        overlap = query_tokens & source_tokens
        if overlap:
            scored.append((len(overlap), source_id))
    if not scored:
        return ()
    scored.sort(reverse=True)
    best_score = scored[0][0]
    if best_score < 1:
        return ()
    return tuple(source_id for score, source_id in scored if score == best_score)[:2]


def _source_overlap_score(
    source_ids: tuple[int, ...],
    question: str,
    reference_answer: str,
    selected_chunks: list[dict],
) -> int:
    """Считает совпадение вопроса и ответа с указанными SOURCE."""
    query_tokens = _significant_quiz_tokens(f"{question} {reference_answer}")
    score = 0
    for source_id in source_ids:
        if not 1 <= source_id <= len(selected_chunks):
            continue
        source_tokens = _significant_quiz_tokens(
            selected_chunks[source_id - 1].get("content", "")
        )
        score += len(query_tokens & source_tokens)
    return score


def _build_evidence_quote(
    question: str,
    reference_answer: str,
    source_context: str,
) -> str:
    """Выбирает подтверждающий фрагмент из связанного SOURCE."""
    query_tokens = _significant_quiz_tokens(f"{question} {reference_answer}")
    candidates = _evidence_candidates(source_context)
    if not candidates:
        return ""
    scored = []
    for candidate in candidates:
        candidate_tokens = _significant_quiz_tokens(candidate)
        overlap = len(query_tokens & candidate_tokens) if query_tokens else 0
        scored.append((overlap, len(candidate_tokens), candidate))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_overlap, _token_count, best_candidate = scored[0]
    if query_tokens and best_overlap == 0:
        return ""
    return _trim_evidence(best_candidate, 600)


def _evidence_candidates(text: str) -> list[str]:
    """Делит source-текст на кандидаты для evidence."""
    candidates = []
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text or "")
        if part.strip()
    ]
    for paragraph in paragraphs or [text.strip()]:
        if not paragraph:
            continue
        if len(paragraph) <= 600:
            candidates.append(paragraph)
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?…])\s+", paragraph)
            if sentence.strip()
        ]
        candidates.extend(sentences or [paragraph])
    return [
        candidate
        for candidate in candidates
        if len(_significant_quiz_tokens(candidate)) >= 2
    ]


def _trim_evidence(text: str, limit: int) -> str:
    """Обрезает evidence без разрыва слова, когда это возможно."""
    stripped = _normalize_space(text)
    if len(stripped) <= limit:
        return stripped
    truncated = stripped[:limit].rstrip()
    sentence_end = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
        truncated.rfind("…"),
    )
    if sentence_end >= limit // 2:
        return truncated[:sentence_end + 1].strip()
    word_end = truncated.rfind(" ")
    if word_end >= limit // 2:
        return truncated[:word_end].strip()
    return truncated


def _quote_in_context(quote: str, context: str) -> bool:
    """Проверяет наличие цитаты в контексте после мягкой нормализации."""
    return _normalize_evidence_text(quote) in _normalize_evidence_text(context)


def _is_substantive_reference_answer(
    question: str,
    answer: str,
    evidence_quote: str,
    source_context: str,
) -> bool:
    """Проверяет, что эталонный ответ не является заглушкой."""
    return (
        _reference_answer_rejection_reason(
            question,
            answer,
            evidence_quote,
            source_context,
        )
        is None
    )


def _reference_answer_rejection_reason(
    question: str,
    answer: str,
    evidence_quote: str,
    source_context: str,
) -> str | None:
    """Возвращает причину отклонения эталона или None."""
    answer_tokens = tokenize_for_search(answer)
    if len(answer_tokens) <= 2:
        question_tokens = set(tokenize_for_search(question))
        if question_tokens & {
            "название",
            "называется",
            "термин",
            "обозначение",
            "код",
            "значение",
        }:
            return (
                None
                if _has_source_overlap(answer_tokens, evidence_quote, source_context)
                else "reference_unrelated"
            )
        if re.search(r"\d|[()=<>+\-*/]|O\(", answer):
            return None
        return "reference_unrelated"

    lowered = answer.casefold()
    if re.fullmatch(
        r"(формул\w*|определени\w*|теорем\w*|свойств\w*)"
        r"(\s+\w+){0,3}",
        lowered,
    ):
        return "meta_reference_answer"
    if (
        "согласно документу" in lowered
        or "согласно материалу" in lowered
        or "содержится в материале" in lowered
        or "содержится в документ" in lowered
        or "смотрите источник" in lowered
    ):
        return "meta_reference_answer"
    if len(answer_tokens) <= 3 and any(
        token.startswith(("формул", "определени", "теорем", "свойств"))
        for token in answer_tokens
    ):
        return "meta_reference_answer"
    if "формул" in question.casefold() and not re.search(
        r"\d|[()=<>+\-*/]|равн|отнош|предел|сумм",
        lowered,
    ):
        return "meta_reference_answer"
    if "определ" in question.casefold() and not any(
        marker in lowered
        for marker in ("это", "является", "называется", "понимается")
    ):
        return "meta_reference_answer"
    if not _has_source_overlap(answer_tokens, evidence_quote, source_context):
        question_tokens = set(tokenize_for_search(question))
        if len(set(answer_tokens) & question_tokens) < 1:
            return "reference_unrelated"
    return None


def _has_source_overlap(
    tokens: list[str],
    evidence_quote: str,
    source_context: str,
) -> bool:
    """Проверяет минимальную связь ответа с источником."""
    if not tokens:
        return False
    source_tokens = set(tokenize_for_search(f"{evidence_quote} {source_context}"))
    matched = set(tokens) & source_tokens
    return len(matched) / len(set(tokens)) >= 0.2


def _significant_quiz_tokens(text: str) -> set[str]:
    """Выделяет значимые токены для проверки вопросов."""
    return {
        token
        for token in tokenize_for_search(text)
        if len(token) > 2 or any(char.isdigit() for char in token)
    }


def _normalize_space(text: str) -> str:
    """Схлопывает повторяющиеся пробелы в одну строку."""
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_evidence_text(text: str) -> str:
    """Нормализует текст для проверки evidence-фрагмента."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00ad", "")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.translate(
        str.maketrans(
            {
                "«": '"',
                "»": '"',
                "“": '"',
                "”": '"',
                "„": '"',
                "‘": "'",
                "’": "'",
            }
        )
    )
    return _normalize_space(normalized)
