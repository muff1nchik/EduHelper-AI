from datetime import datetime, timezone
import json
from pathlib import Path

import aiosqlite


class Database:
    def __init__(self, path: str) -> None:
        self.path = path

    async def init(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            await db.commit()

    async def add_document(self, user_id: int, filename: str, file_path: str) -> int:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                """
                INSERT INTO documents (user_id, filename, file_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, filename, file_path, self._now()),
            )
            await db.commit()
            return int(cursor.lastrowid)

    async def add_chunk(
        self,
        document_id: int,
        user_id: int,
        content: str,
        embedding: list[float],
        chunk_index: int,
    ) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO chunks
                    (document_id, user_id, chunk_index, content, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    user_id,
                    chunk_index,
                    content,
                    json.dumps(embedding),
                    self._now(),
                ),
            )
            await db.commit()

    async def add_document_with_chunks(
        self,
        user_id: int,
        filename: str,
        file_path: str,
        chunks: list[dict],
    ) -> int:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            try:
                await db.execute("BEGIN")
                cursor = await db.execute(
                    """
                    INSERT INTO documents (user_id, filename, file_path, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, filename, file_path, self._now()),
                )
                document_id = int(cursor.lastrowid)

                for chunk in chunks:
                    await db.execute(
                        """
                        INSERT INTO chunks
                            (document_id, user_id, chunk_index, content, embedding, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            user_id,
                            chunk["chunk_index"],
                            chunk["content"],
                            json.dumps(chunk["embedding"]),
                            self._now(),
                        ),
                    )

                await db.commit()
                return document_id
            except Exception:
                await db.rollback()
                raise

    async def get_user_chunks(self, user_id: int) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT
                    chunks.id AS chunk_id,
                    chunks.document_id,
                    documents.filename,
                    chunks.content,
                    chunks.embedding
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.user_id = ?
                ORDER BY chunks.id
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()

        chunks = []
        for row in rows:
            item = dict(row)
            try:
                item["embedding"] = json.loads(item["embedding"])
            except json.JSONDecodeError as exc:
                raise RuntimeError("В базе найден некорректный embedding.") from exc
            chunks.append(item)
        return chunks

    async def clear_user_data(self, user_id: int) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("DELETE FROM chunks WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
            await db.commit()

    async def user_has_chunks(self, user_id: int) -> bool:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM chunks WHERE user_id = ? LIMIT 1",
                (user_id,),
            )
            row = await cursor.fetchone()
        return row is not None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
