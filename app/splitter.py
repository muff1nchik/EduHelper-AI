"""Делит очищенный текст документа на удобные фрагменты."""

import re

from app.text_utils import normalize_document_text


class TextSplitter:
    """Разбивает документ на чанки с небольшим перекрытием."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP должен быть меньше CHUNK_SIZE.")
        if chunk_size <= 0:
            raise ValueError("CHUNK_SIZE должен быть больше нуля.")
        if overlap < 0:
            raise ValueError("CHUNK_OVERLAP не может быть отрицательным.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text: str) -> str:
        """Очищает текст перед разбиением на фрагменты."""
        return normalize_document_text(text)

    def split(self, text: str) -> list[str]:
        """Делит текст на чанки не длиннее заданного размера."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        blocks = self._build_blocks(cleaned)
        if not blocks:
            return []

        chunks: list[str] = []
        current = ""
        for block in blocks:
            pieces = self._split_long_block(block)
            for piece in pieces:
                candidate = f"{current}\n\n{piece}".strip() if current else piece
                if len(candidate) <= self.chunk_size:
                    current = candidate
                    continue
                if current:
                    chunks.append(current)
                overlap = self._make_overlap(current)
                candidate = f"{overlap}\n\n{piece}".strip() if overlap else piece
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    chunks.extend(self._split_hard_with_overlap(piece))
                    current = ""

        if current:
            chunks.append(current)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return self._deduplicate_nearby(chunks)

    def _build_blocks(self, text: str) -> list[str]:
        """Собирает абзацы и присоединяет короткие заголовки к тексту."""
        paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        blocks: list[str] = []
        pending_heading = ""
        for index, paragraph in enumerate(paragraphs):
            if self._is_heading(paragraph) and index + 1 < len(paragraphs):
                pending_heading = f"{pending_heading}\n{paragraph}".strip()
                continue
            if pending_heading:
                paragraph = f"{pending_heading}\n\n{paragraph}"
                pending_heading = ""
            blocks.append(paragraph)
        if pending_heading:
            if blocks and len(f"{blocks[-1]}\n\n{pending_heading}") <= self.chunk_size:
                blocks[-1] = f"{blocks[-1]}\n\n{pending_heading}"
            else:
                blocks.append(pending_heading)
        return blocks

    def _is_heading(self, block: str) -> bool:
        """Проверяет, похож ли блок на короткий заголовок."""
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) != 1:
            return False
        line = lines[0]
        if len(line) > 90 or line.endswith((".", "!", "?", "…", ";", ",")):
            return False
        if re.match(r"^[-*+]\s+", line) or re.match(r"^\d+[.)]\s+\S.+[.!?]$", line):
            return False
        if line.startswith("```") or any(char in line for char in "{}=;"):
            return False
        return bool(
            line.startswith("#")
            or re.match(r"^(глава|раздел|chapter|section)\s+\d+", line, re.IGNORECASE)
            or re.match(r"^\d+(?:\.\d+)*\.?\s+\S+", line)
            or (len(line.split()) <= 7 and not line.endswith(":"))
        )

    def _split_long_block(self, block: str) -> list[str]:
        """Делит длинный абзац по предложениям или жёстко по длине."""
        if len(block) <= self.chunk_size:
            return [block]
        if "```" in block:
            return self._split_hard_with_overlap(block)

        sentences = self._split_sentences(block)
        if len(sentences) <= 1:
            return self._split_hard_with_overlap(block)
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                pieces.append(current)
            current = sentence
        if current:
            pieces.append(current)
        result: list[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                result.append(piece)
            else:
                result.extend(self._split_hard_with_overlap(piece))
        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Разбивает текст по очевидным границам предложений."""
        parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9])", text)
        return [part.strip() for part in parts if part.strip()]

    def _split_hard_with_overlap(self, text: str) -> list[str]:
        """Делит длинную строку с учётом overlap."""
        chunks: list[str] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = self._find_chunk_end(text, start)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            next_start = self._find_next_start(text, start, end)
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    def _find_chunk_end(self, text: str, start: int) -> int:
        """Ищет естественную границу конца чанка."""
        hard_end = min(start + self.chunk_size, len(text))
        if hard_end == len(text):
            return hard_end

        min_boundary = start + max(1, self.chunk_size // 2)
        paragraph_end = text.rfind("\n\n", start, hard_end)
        if paragraph_end >= min_boundary:
            return paragraph_end + 2

        sentence_end = self._find_sentence_end(text, start, hard_end, min_boundary)
        if sentence_end is not None:
            return sentence_end

        whitespace_end = self._find_whitespace_end(text, start, hard_end, min_boundary)
        if whitespace_end is not None:
            return whitespace_end

        return hard_end

    def _find_sentence_end(
        self,
        text: str,
        start: int,
        hard_end: int,
        min_boundary: int,
    ) -> int | None:
        """Ищет ближайший конец предложения в окне чанка."""
        for index in range(hard_end - 1, min_boundary - 1, -1):
            if text[index] not in ".!?…":
                continue
            next_index = index + 1
            if next_index == len(text) or text[next_index].isspace():
                return next_index
        return None

    def _find_whitespace_end(
        self,
        text: str,
        start: int,
        hard_end: int,
        min_boundary: int,
    ) -> int | None:
        """Ищет пробел для мягкого разрыва чанка."""
        for index in range(hard_end - 1, min_boundary - 1, -1):
            if text[index].isspace():
                return index
        return None

    def _find_next_start(self, text: str, start: int, end: int) -> int:
        """Выбирает начало следующего чанка с учётом overlap."""
        if self.overlap == 0:
            next_start = end
        else:
            next_start = max(start + 1, end - self.overlap)
            while (
                next_start > start + 1
                and next_start < end
                and text[next_start - 1].isalnum()
                and text[next_start].isalnum()
            ):
                next_start -= 1

        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        return next_start

    def _make_overlap(self, previous: str) -> str:
        """Берёт короткий хвост предыдущего чанка для контекста."""
        if self.overlap <= 0 or not previous:
            return ""
        tail = previous[-self.overlap:].strip()
        sentence_match = re.search(r"([^.!?…]{1,%d}[.!?…])\s*$" % self.overlap, previous)
        if sentence_match:
            tail = sentence_match.group(1).strip()
        if len(tail) > self.chunk_size // 3:
            return ""
        return tail

    def _deduplicate_nearby(self, chunks: list[str]) -> list[str]:
        """Удаляет почти одинаковые соседние чанки."""
        result: list[str] = []
        for chunk in chunks:
            if result and (chunk == result[-1] or chunk in result[-1] or result[-1] in chunk):
                shorter = min(len(chunk), len(result[-1]))
                longer = max(len(chunk), len(result[-1]))
                if shorter / longer > 0.85:
                    continue
            result.append(chunk)
        return result
