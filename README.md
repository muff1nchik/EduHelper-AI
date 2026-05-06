# EduHelper AI

EduHelper AI — локальный образовательный Telegram-бот, который помогает работать с учебными материалами. Пользователь загружает файл, бот извлекает из него текст, разбивает материал на фрагменты, находит релевантную информацию и формирует ответ с помощью локальной модели Ollama.

Проект использует RAG-подход: перед генерацией ответа бот сначала ищет подходящие фрагменты в загруженных материалах, а затем передает их языковой модели.

## Возможности

- загрузка учебных материалов в форматах PDF, TXT и MD
- извлечение текста из файлов
- очистка и разбиение текста на фрагменты
- создание embeddings через Ollama
- хранение документов и фрагментов в SQLite
- семантический поиск по материалам
- генерация ответа через локальную LLM
- работа через Telegram-бота
- команды `/start`, `/help`, `/clear`

## Архитектура

Основной пайплайн работы:

```text
Telegram → файл → loader → splitter → embeddings → SQLite → vector search → Ollama → ответ
```

Логика работы:

1. Пользователь отправляет PDF, TXT или MD файл.
2. Бот сохраняет файл в `data/uploads`.
3. Загрузчик извлекает текст из файла.
4. `TextSplitter` очищает текст и делит его на фрагменты.
5. `OllamaClient` создает embeddings через модель `nomic-embed-text`.
6. Документ, фрагменты и embeddings сохраняются в SQLite.
7. Пользователь задает вопрос по материалу.
8. Вопрос преобразуется в embedding.
9. `VectorSearch` ищет наиболее релевантные фрагменты.
10. Найденные фрагменты и вопрос отправляются в модель `qwen3:8b`.
11. Бот возвращает учебный ответ пользователю.

## Технологии

- Python
- aiogram
- SQLite
- aiosqlite
- PyMuPDF
- Ollama
- qwen3:8b
- nomic-embed-text
- pytest

## Установка

Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Установка моделей Ollama

Перед запуском проекта установите модели:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Ollama должна быть запущена локально:

```bash
ollama serve
```

## Настройка окружения

Создайте файл `.env` на основе примера:

```bash
cp .env.example .env
```

В `.env` нужно указать токен Telegram-бота:

```env
BOT_TOKEN=your_telegram_bot_token
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
DATABASE_PATH=data/eduhelper.db
UPLOADS_DIR=data/uploads
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=4
```

## Запуск

```bash
python main.py
```

После запуска откройте Telegram-бота и отправьте команду:

```text
/start
```

## Команды бота

- `/start` — начать работу с ботом
- `/help` — показать инструкцию
- `/clear` — удалить загруженные материалы пользователя

## Пример использования

1. Запустите Ollama.
2. Запустите проект командой `python main.py`.
3. Откройте Telegram-бота.
4. Отправьте PDF, TXT или MD файл.
5. Дождитесь сообщения об успешной обработке.
6. Задайте вопрос по загруженному материалу.

## Тесты

```bash
pytest
```

Тесты проверяют работу разбиения текста и векторного поиска. Они не требуют запущенного Telegram-бота или Ollama.

## Структура проекта

```text
EduHelper-AI/
├── app/
│   ├── bot.py
│   ├── config.py
│   ├── database.py
│   ├── loaders.py
│   ├── ollama_client.py
│   ├── search.py
│   ├── services.py
│   └── splitter.py
├── data/
│   └── uploads/
├── tests/
│   ├── test_search.py
│   └── test_splitter.py
├── main.py
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

## Основные файлы

- `main.py` — точка входа, сборка зависимостей и запуск бота
- `app/config.py` — загрузка настроек из `.env`
- `app/bot.py` — обработчики Telegram-бота
- `app/database.py` — работа с SQLite
- `app/loaders.py` — загрузчики TXT, MD и PDF
- `app/splitter.py` — очистка и разбиение текста
- `app/ollama_client.py` — запросы к Ollama
- `app/search.py` — семантический поиск по embeddings
- `app/services.py` — основная бизнес-логика
- `tests/` — тесты проекта