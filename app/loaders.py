from abc import ABC, abstractmethod
from pathlib import Path

import fitz


class BaseLoader(ABC):
    @abstractmethod
    def load_text(self, file_path: str) -> str:
        """Извлечь текст из файла."""

    def _ensure_not_empty(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Документ пустой или не содержит текстового слоя.")
        return text


class TextLoader(BaseLoader):
    def load_text(self, file_path: str) -> str:
        try:
            text = Path(file_path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Не удалось прочитать текстовый файл в кодировке UTF-8.") from exc
        return self._ensure_not_empty(text)


class PdfLoader(BaseLoader):
    def load_text(self, file_path: str) -> str:
        try:
            with fitz.open(file_path) as document:
                text = "\n".join(page.get_text("text") for page in document)
        except Exception as exc:
            raise ValueError("Не удалось прочитать PDF-файл.") from exc
        return self._ensure_not_empty(text)


def get_loader(file_path: str) -> BaseLoader:
    extension = Path(file_path).suffix.lower()
    if extension in {".txt", ".md"}:
        return TextLoader()
    if extension == ".pdf":
        return PdfLoader()
    raise ValueError("Поддерживаются только файлы PDF, TXT и MD.")
