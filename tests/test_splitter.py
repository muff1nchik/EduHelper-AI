import pytest

from app.splitter import TextSplitter


def test_short_text_returns_one_chunk():
    splitter = TextSplitter(chunk_size=100, overlap=20)

    assert splitter.split("Короткий учебный текст.") == ["Короткий учебный текст."]


def test_long_text_splits_into_multiple_chunks():
    splitter = TextSplitter(chunk_size=50, overlap=10)
    text = " ".join(["математика"] * 30)

    chunks = splitter.split(text)

    assert len(chunks) > 1
    assert all(chunks)


def test_empty_text_returns_empty_list():
    splitter = TextSplitter(chunk_size=100, overlap=20)

    assert splitter.split("   \n\n\t ") == []


def test_overlap_greater_or_equal_chunk_size_raises_value_error():
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=100, overlap=100)
