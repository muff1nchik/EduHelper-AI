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
            end = min(start + self.chunk_size, text_length)
            chunk = cleaned[start:end].strip()

            if chunk:
                chunks.append(chunk)
            if end == text_length:
                break

            next_start = end - self.overlap
            if next_start <= start:
                raise RuntimeError("Не удалось разделить текст: некорректный шаг чанков.")
            start = next_start

        return chunks
