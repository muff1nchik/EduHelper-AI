import pytest

from app.config import load_settings


def test_load_settings_converts_ollama_temperature_to_float(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.45")

    settings = load_settings()

    assert settings.ollama_temperature == 0.45
    assert isinstance(settings.ollama_temperature, float)


def test_load_settings_converts_ollama_num_ctx_to_int(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")

    settings = load_settings()

    assert settings.ollama_num_ctx == 8192
    assert isinstance(settings.ollama_num_ctx, int)


def test_load_settings_converts_min_similarity_to_float(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("MIN_SIMILARITY", "0.42")

    settings = load_settings()

    assert settings.min_similarity == 0.42
    assert isinstance(settings.min_similarity, float)


def test_load_settings_rejects_min_similarity_below_minus_one(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("MIN_SIMILARITY", "-1.01")

    with pytest.raises(ValueError, match="MIN_SIMILARITY"):
        load_settings()


def test_load_settings_rejects_min_similarity_above_one(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("MIN_SIMILARITY", "1.01")

    with pytest.raises(ValueError, match="MIN_SIMILARITY"):
        load_settings()
