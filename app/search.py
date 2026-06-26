import math

from app.text_utils import (
    significant_query_phrase,
    structural_refs,
    technical_identifiers,
    tokenize_for_search,
)


NEIGHBOR_CONTEXT_MAX_CHARS = 6000
LEXICAL_EXACT_PHRASE_BONUS = 0.18
LEXICAL_STRONG_BONUS = 0.12
LEXICAL_WEAK_BONUS = 0.05


class VectorSearch:
    def __init__(self, min_similarity: float = 0.35) -> None:
        self.min_similarity = min_similarity

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            raise ValueError("Embeddings должны быть непустыми векторами одинаковой длины.")

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(
        self,
        chunks: list[dict],
        query_embedding: list[float],
        top_k: int,
        query_text: str = "",
    ) -> list[dict]:
        if not query_embedding:
            raise ValueError("Embedding вопроса пустой.")

        query_features = _build_query_features(query_text)
        results: list[dict] = []
        for position, chunk in enumerate(chunks):
            embedding = chunk.get("embedding")
            try:
                score = self.cosine_similarity(embedding, query_embedding)
            except (TypeError, ValueError) as exc:
                raise ValueError("В чанках найден некорректный embedding.") from exc
            lexical = _lexical_evidence(query_features, chunk.get("content", ""))
            accepted = score >= self.min_similarity or lexical["strong"]

            if accepted:
                results.append(
                    {
                        "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                        "document_id": chunk.get("document_id"),
                        "chunk_index": chunk.get("chunk_index", position),
                        "filename": chunk.get("filename"),
                        "content": chunk.get("content", ""),
                        "score": score,
                        "combined_score": score + lexical["bonus"],
                        "strong_lexical": lexical["strong"],
                        "_position": position,
                    }
                )

        main_results = sorted(
            results,
            key=lambda item: (
                item["strong_lexical"],
                item["combined_score"],
                item["score"],
            ),
            reverse=True,
        )[:top_k]
        return _with_neighbor_context(chunks, main_results)


def _build_query_features(query: str) -> dict:
    tokens = tokenize_for_search(query)
    return {
        "tokens": tokens,
        "token_set": set(tokens),
        "phrase": significant_query_phrase(query),
        "structural_refs": structural_refs(query),
        "technical_ids": technical_identifiers(query),
    }


def _lexical_evidence(query: dict, content: str) -> dict:
    if not query["tokens"]:
        return {"strong": False, "bonus": 0.0}

    content_tokens = tokenize_for_search(content)
    content_token_set = set(content_tokens)
    content_text = " ".join(content_tokens)
    matched_tokens = query["token_set"] & content_token_set
    coverage = len(matched_tokens) / max(1, len(query["token_set"]))

    phrase = query["phrase"]
    exact_phrase = bool(
        phrase
        and len(phrase.split()) >= 2
        and phrase in content_text
    )
    technical_match = bool(query["technical_ids"] & content_token_set)
    structural_match = bool(query["structural_refs"] & structural_refs(content))
    heading_match = _heading_matches(query["token_set"], content)
    majority_match = len(query["token_set"]) >= 2 and coverage >= 0.66

    strong = exact_phrase or technical_match or structural_match or heading_match or majority_match
    if exact_phrase or structural_match:
        bonus = LEXICAL_EXACT_PHRASE_BONUS
    elif strong:
        bonus = LEXICAL_STRONG_BONUS
    elif len(matched_tokens) >= 2:
        bonus = LEXICAL_WEAK_BONUS
    else:
        bonus = 0.0
    return {"strong": strong, "bonus": bonus}


def _heading_matches(query_tokens: set[str], content: str) -> bool:
    if not query_tokens or len(query_tokens) < 2:
        return False
    first_block = content.split("\n\n", 1)[0]
    first_lines = [line.strip() for line in first_block.splitlines() if line.strip()]
    heading = first_lines[0] if first_lines else ""
    if not heading or len(heading) > 120 or heading.endswith((".", "!", "?")):
        return False
    heading_tokens = set(tokenize_for_search(heading))
    if not heading_tokens:
        return False
    return len(query_tokens & heading_tokens) / len(query_tokens) >= 0.66


def _with_neighbor_context(chunks: list[dict], main_results: list[dict]) -> list[dict]:
    if not main_results:
        return []

    if not any(chunk.get("chunk_index") is not None for chunk in chunks):
        return main_results

    by_key = {
        (chunk.get("document_id"), chunk.get("chunk_index")): (position, chunk)
        for position, chunk in enumerate(chunks)
        if chunk.get("chunk_index") is not None
    }
    selected: dict[tuple, dict] = {}
    main_keys = {
        (result.get("document_id"), result.get("chunk_index"))
        for result in main_results
    }
    current_length = 0

    def add_chunk(chunk: dict, base: dict | None = None, neighbor: bool = False) -> None:
        nonlocal current_length
        key = (chunk.get("document_id"), chunk.get("chunk_index"))
        if key in selected:
            return
        content = chunk.get("content", "")
        if current_length + len(content) > NEIGHBOR_CONTEXT_MAX_CHARS and selected:
            return
        selected[key] = {
            "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
            "document_id": chunk.get("document_id"),
            "chunk_index": chunk.get("chunk_index"),
            "filename": chunk.get("filename"),
            "content": content,
            "score": base.get("score", 0.0) if base else 0.0,
            "strong_lexical": base.get("strong_lexical", False) if base else False,
            "is_neighbor": neighbor,
        }
        current_length += len(content)

    for result in main_results:
        add_chunk(result, result, neighbor=False)
    for result in main_results:
        document_id = result.get("document_id")
        chunk_index = result.get("chunk_index")
        if chunk_index is None:
            continue
        for neighbor_index in (chunk_index - 1, chunk_index + 1):
            key = (document_id, neighbor_index)
            if key in main_keys or key not in by_key:
                continue
            _, chunk = by_key[key]
            add_chunk(chunk, result, neighbor=True)

    return sorted(
        selected.values(),
        key=lambda item: (item.get("document_id") or 0, item.get("chunk_index") or 0),
    )
