import re
import unicodedata


SOFT_HYPHEN = "\u00ad"
NBSP_CHARS = ("\u00a0", "\u202f")
WORD_RE = re.compile(r"[a-zа-яё]+(?:_[a-zа-яё0-9]+)*|\d+(?:\.\d+)*", re.IGNORECASE)
TECH_IDENTIFIER_RE = re.compile(r"[a-z_][a-z0-9_]*", re.IGNORECASE)
STRUCTURAL_REF_RE = re.compile(
    r"\b(?:теорем\w*|theorem|раздел\w*|section|глав\w*|chapter)\s+\d+(?:\.\d+)*\b|\b\d+\.\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
STOP_WORDS = {
    "а", "в", "во", "и", "или", "к", "ко", "на", "о", "об", "от", "по", "с",
    "со", "у", "что", "это", "как", "какие", "какой", "какая", "какое", "про",
    "для", "из", "за", "the", "a", "an", "is", "are", "of", "to", "in", "on",
    "about", "me", "tell", "explain", "what",
}
QUERY_PREFIXES = (
    "что такое",
    "расскажи про",
    "объясни",
    "дай определение",
    "what is",
    "explain",
    "tell me about",
)


def normalize_document_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(SOFT_HYPHEN, "")
    for char in NBSP_CHARS:
        text = text.replace(char, " ")

    raw_lines = text.split("\n")
    lines = []
    index = 0
    while index < len(raw_lines):
        line = _normalize_line_spaces(raw_lines[index])
        if _can_join_hyphenated_line(line, raw_lines, index):
            next_index = _next_non_empty_line_index(raw_lines, index + 1)
            next_line = _normalize_line_spaces(raw_lines[next_index])
            line = f"{line[:-1]}{next_line}"
            index = next_index + 1
        else:
            index += 1
        lines.append(line)

    cleaned_lines = []
    empty_count = 0
    for line in lines:
        if not line:
            empty_count += 1
            if empty_count <= 1:
                cleaned_lines.append("")
            continue
        empty_count = 0
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def tokenize_for_search(text: str) -> list[str]:
    normalized = _normalize_search_text(text)
    return [
        token
        for token in WORD_RE.findall(normalized)
        if token not in STOP_WORDS
    ]


def clean_model_output(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?<!\$)\$\s*([^$\n]+?)\s*\$(?!\$)", r"\1", cleaned)
    return cleaned.strip()


def significant_query_phrase(query: str) -> str:
    normalized = _normalize_search_text(query)
    for prefix in QUERY_PREFIXES:
        if normalized.startswith(prefix + " "):
            normalized = normalized[len(prefix):].strip()
            break
    tokens = [token for token in tokenize_for_search(normalized) if token not in STOP_WORDS]
    return " ".join(tokens)


def structural_refs(text: str) -> set[str]:
    normalized = _normalize_search_text(text)
    refs = {match.group(0).strip() for match in STRUCTURAL_REF_RE.finditer(normalized)}
    refs.update(re.findall(r"\b\d+\.\d+(?:\.\d+)*\b", normalized))
    return refs


def technical_identifiers(text: str) -> set[str]:
    return {
        token
        for token in tokenize_for_search(text)
        if TECH_IDENTIFIER_RE.fullmatch(token) and ("_" in token or any(char.isdigit() for char in token))
    }


def _normalize_line_spaces(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).rstrip().strip()


def _can_join_hyphenated_line(line: str, raw_lines: list[str], index: int) -> bool:
    if not re.search(r"[^\W\d_]-$", line, re.UNICODE):
        return False
    if _looks_like_list_or_code(line):
        return False
    next_index = _next_non_empty_line_index(raw_lines, index + 1)
    if next_index is None or next_index != index + 1:
        return False
    next_line = _normalize_line_spaces(raw_lines[next_index])
    if _looks_like_list_or_code(next_line):
        return False
    before = line[:-1].split()[-1]
    first = next_line.split()[0] if next_line.split() else ""
    return bool(
        before.isalpha()
        and first.isalpha()
        and first[0].islower()
    )


def _next_non_empty_line_index(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
        return None
    return None


def _looks_like_list_or_code(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(r"^[-*+]\s+", stripped)
        or re.match(r"^\d+[.)]\s+", stripped)
        or stripped.startswith((">>>", "$", "```"))
    )


def _normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "").casefold().replace("ё", "е")
    normalized = normalized.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", normalized).strip()
