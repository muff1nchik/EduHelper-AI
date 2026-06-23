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
