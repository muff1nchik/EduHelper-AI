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
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_state (
                    user_id INTEGER PRIMARY KEY,
                    active_document_id INTEGER,
                    FOREIGN KEY(active_document_id)
                        REFERENCES documents(id)
                        ON DELETE SET NULL
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

                await db.execute(
                    """
                    INSERT INTO user_state (user_id, active_document_id)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        active_document_id = excluded.active_document_id
                    """,
                    (user_id, document_id),
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

    async def get_document_chunks(self, user_id: int, document_id: int) -> list[dict]:
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
                    AND chunks.document_id = ?
                    AND documents.user_id = ?
                ORDER BY chunks.chunk_index
                """,
                (user_id, document_id, user_id),
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

    async def get_user_documents(self, user_id: int) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT id, filename
                FROM documents
                WHERE user_id = ?
                ORDER BY id DESC
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def set_active_document(self, user_id: int, document_id: int) -> dict | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            cursor = await db.execute(
                """
                SELECT id, filename
                FROM documents
                WHERE id = ? AND user_id = ?
                """,
                (document_id, user_id),
            )
            row = await cursor.fetchone()
            if row is None:
                return None

            await db.execute(
                """
                INSERT INTO user_state (user_id, active_document_id)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    active_document_id = excluded.active_document_id
                """,
                (user_id, document_id),
            )
            await db.commit()
        return dict(row)

    async def get_active_document(self, user_id: int) -> dict | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT documents.id, documents.filename
                FROM user_state
                JOIN documents ON documents.id = user_state.active_document_id
                WHERE user_state.user_id = ?
                    AND documents.user_id = ?
                """,
                (user_id, user_id),
            )
            row = await cursor.fetchone()
            if row is not None:
                return dict(row)

            cursor = await db.execute(
                """
                SELECT id, filename
                FROM documents
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            )
            row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def delete_document(self, user_id: int, document_id: int) -> dict | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            try:
                await db.execute("BEGIN")
                cursor = await db.execute(
                    """
                    SELECT id, filename, file_path
                    FROM documents
                    WHERE id = ? AND user_id = ?
                    """,
                    (document_id, user_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    await db.rollback()
                    return None

                cursor = await db.execute(
                    """
                    SELECT active_document_id
                    FROM user_state
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
                state_row = await cursor.fetchone()
                is_active = (
                    state_row is not None
                    and state_row["active_document_id"] == document_id
                )

                await db.execute(
                    "DELETE FROM chunks WHERE user_id = ? AND document_id = ?",
                    (user_id, document_id),
                )
                await db.execute(
                    "DELETE FROM documents WHERE id = ? AND user_id = ?",
                    (document_id, user_id),
                )

                if is_active:
                    cursor = await db.execute(
                        """
                        SELECT id
                        FROM documents
                        WHERE user_id = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (user_id,),
                    )
                    next_document = await cursor.fetchone()
                    if next_document is None:
                        await db.execute(
                            "DELETE FROM user_state WHERE user_id = ?",
                            (user_id,),
                        )
                    else:
                        await db.execute(
                            """
                            INSERT INTO user_state (user_id, active_document_id)
                            VALUES (?, ?)
                            ON CONFLICT(user_id) DO UPDATE SET
                                active_document_id = excluded.active_document_id
                            """,
                            (user_id, next_document["id"]),
                        )

                await db.commit()
                return dict(row)
            except Exception:
                await db.rollback()
                raise

    async def clear_user_data(self, user_id: int) -> list[str]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            try:
                await db.execute("BEGIN")
                cursor = await db.execute(
                    """
                    SELECT file_path
                    FROM documents
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
                rows = await cursor.fetchall()

                await db.execute("DELETE FROM chunks WHERE user_id = ?", (user_id,))
                await db.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
                await db.execute("DELETE FROM user_state WHERE user_id = ?", (user_id,))
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        return [row["file_path"] for row in rows]

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
