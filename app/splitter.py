import re


class TextSplitter:
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
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        cleaned_lines: list[str] = []
        previous_empty = False

        for line in lines:
            is_empty = not line
            if is_empty and previous_empty:
                continue
            cleaned_lines.append(line)
            previous_empty = is_empty

        return "\n".join(cleaned_lines).strip()

    def split(self, text: str) -> list[str]:
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        chunks: list[str] = []
        start = 0
        text_length = len(cleaned)

        while start < text_length:
            end = self._find_chunk_end(cleaned, start)
            chunk = cleaned[start:end].strip()

            if chunk:
                chunks.append(chunk)
            if end == text_length:
                break

            next_start = self._find_next_start(cleaned, start, end)
            if next_start <= start:
                raise RuntimeError("Не удалось разделить текст: некорректный шаг чанков.")
            start = next_start

        return chunks

    def _find_chunk_end(self, text: str, start: int) -> int:
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
        for index in range(hard_end - 1, min_boundary - 1, -1):
            if text[index].isspace():
                return index
        return None

    def _find_next_start(self, text: str, start: int, end: int) -> int:
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
