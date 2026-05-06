# EduHelper AI

## Описание

EduHelper AI — локальный образовательный Telegram-бот, который отвечает на вопросы по загруженным учебным материалам с использованием RAG-подхода и локальной модели Ollama.

## Возможности MVP

- загрузка PDF, TXT, MD
- извлечение текста
- разбиение текста на фрагменты
- создание embeddings через Ollama
- хранение материалов в SQLite
- semantic search по материалам
- генерация ответа через локальную LLM
- команды /start, /help, /clear

## Что не входит в первую версию

- OCR
- DOCX
- веб-интерфейс
- личный кабинет
- облачный деплой
- генерация тестов
- генерация конспектов

## Почему проект готов к масштабированию

Загрузчики вынесены в отдельные классы, поэтому позже можно добавить `DocxLoader` или `ImageLoader` без переписывания бота. База данных отделена от Telegram-слоя, а основная бизнес-логика находится в `EduHelperService`. `OllamaClient` можно заменить на другого LLM-провайдера, `VectorSearch` — на более мощное векторное хранилище, а команды бота можно расширять отдельными обработчиками.

## Архитектура

Пайплайн MVP:

Telegram → файл → loader → splitter → embeddings → SQLite → vector search → Ollama → ответ

Основной поток:

1. Пользователь отправляет PDF, TXT или MD файл.
2. Бот сохраняет файл в `data/uploads`.
3. Loader извлекает текст.
4. `TextSplitter` очищает текст и делит его на фрагменты.
5. `OllamaClient` создает embeddings через `nomic-embed-text`.
6. Документ, фрагменты и embeddings сохраняются в SQLite.
7. Пользователь задает вопрос.
8. Вопрос превращается в embedding, затем `VectorSearch` ищет релевантные фрагменты.
9. Фрагменты и вопрос отправляются в `qwen3:8b`.
10. Бот возвращает учебный ответ.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Установка Ollama-моделей

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Также должна быть запущена Ollama:

```bash
ollama serve
```

## Настройка .env

```bash
cp .env.example .env
```

В `.env` нужно вставить `BOT_TOKEN`.

## Запуск

```bash
python main.py
```

## Тесты

```bash
pytest
```

## Структура проекта

- `main.py` — точка входа, сборка зависимостей и запуск polling.
- `app/config.py` — загрузка настроек из `.env`.
- `app/bot.py` — Telegram-обработчики aiogram.
- `app/database.py` — работа с SQLite через `aiosqlite`.
- `app/loaders.py` — загрузчики TXT, MD и PDF.
- `app/splitter.py` — очистка и разбиение текста.
- `app/ollama_client.py` — запросы к Ollama для embeddings и генерации ответа.
- `app/search.py` — cosine similarity и выбор релевантных фрагментов.
- `app/services.py` — бизнес-логика обработки файлов и вопросов.
- `tests/` — тесты без зависимости от Telegram и Ollama.
# EduHelper-AI
