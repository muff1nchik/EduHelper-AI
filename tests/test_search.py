import pytest

from app.search import VectorSearch


def test_cosine_similarity_same_vectors_is_close_to_one():
    search = VectorSearch()

    assert search.cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_close_to_zero():
    search = VectorSearch()

    assert search.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_search_returns_top_k_results():
    search = VectorSearch()
    chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Про производную",
            "embedding": [1, 0],
        },
        {
            "chunk_id": 2,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Про интеграл",
            "embedding": [0.8, 0.2],
        },
        {
            "chunk_id": 3,
            "document_id": 2,
            "filename": "b.txt",
            "content": "Про историю",
            "embedding": [0, 1],
        },
    ]

    results = search.search(chunks, [1, 0], top_k=2)

    assert len(results) == 2
    assert [result["chunk_id"] for result in results] == [1, 2]


def test_search_keeps_result_above_min_similarity():
    search = VectorSearch(min_similarity=0.35)
    chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Релевантный фрагмент",
            "embedding": [1, 0],
        }
    ]

    results = search.search(chunks, [1, 0], top_k=1)

    assert [result["chunk_id"] for result in results] == [1]


def test_search_removes_result_below_min_similarity():
    search = VectorSearch(min_similarity=0.35)
    chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Нерелевантный фрагмент",
            "embedding": [0, 1],
        }
    ]

    assert search.search(chunks, [1, 0], top_k=1) == []


def test_search_keeps_result_equal_to_min_similarity():
    search = VectorSearch(min_similarity=0.5)
    chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Пороговый фрагмент",
            "embedding": [1, 1],
        }
    ]

    results = search.search(chunks, [1, 0], top_k=1)

    assert results[0]["chunk_id"] == 1


def test_search_applies_min_similarity_before_top_k():
    search = VectorSearch(min_similarity=0.5)
    chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Высокая похожесть",
            "embedding": [1, 0],
        },
        {
            "chunk_id": 2,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Низкая похожесть",
            "embedding": [0, 1],
        },
        {
            "chunk_id": 3,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Средняя похожесть",
            "embedding": [1, 1],
        },
    ]

    results = search.search(chunks, [1, 0], top_k=3)

    assert [result["chunk_id"] for result in results] == [1, 3]


def test_search_returns_empty_list_when_no_results_match_threshold():
    search = VectorSearch(min_similarity=0.9)
    chunks = [
        {
            "chunk_id": 1,
            "document_id": 1,
            "filename": "a.txt",
            "content": "Слабый фрагмент",
            "embedding": [1, 1],
        }
    ]

    assert search.search(chunks, [1, 0], top_k=1) == []
