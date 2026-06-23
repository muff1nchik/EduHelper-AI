import math


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
    ) -> list[dict]:
        if not query_embedding:
            raise ValueError("Embedding вопроса пустой.")

        results: list[dict] = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            try:
                score = self.cosine_similarity(embedding, query_embedding)
            except (TypeError, ValueError) as exc:
                raise ValueError("В чанках найден некорректный embedding.") from exc

            if score >= self.min_similarity:
                results.append(
                    {
                        "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                        "document_id": chunk.get("document_id"),
                        "filename": chunk.get("filename"),
                        "content": chunk.get("content", ""),
                        "score": score,
                    }
                )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
