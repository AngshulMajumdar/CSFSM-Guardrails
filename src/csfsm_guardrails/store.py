"""Atomic session stores with tenant scoping, expiry, revisions and idempotency."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Protocol

from .errors import CapacityExceeded, IdempotencyConflict, RevisionConflict, SessionNotFound
from .types import SessionState, canonical_json


class SessionStore(Protocol):
    def create(self, state: SessionState) -> None: ...
    def get(self, session_id: str, owner: str) -> SessionState: ...
    def transact(self, session_id: str, owner: str, fn: Callable, **kwargs) -> dict: ...
    def delete(self, session_id: str, owner: str) -> None: ...


def _check(state, owner, clock, expected_revision=None):
    if state is None or state.owner != owner or state.expires_at <= clock():
        raise SessionNotFound("Session not found or expired")
    if expected_revision is not None and state.revision != expected_revision:
        raise RevisionConflict("Session changed; fetch its current revision")


class MemoryStore:
    def __init__(self, max_sessions=10000, clock=time.time):
        self.max_sessions, self.clock = max_sessions, clock
        self._states, self._cache = {}, {}
        self._lock = threading.RLock()

    def create(self, state):
        with self._lock:
            expired = [k for k, s in self._states.items() if s.expires_at <= self.clock()]
            for k in expired:
                self._states.pop(k)
                self._cache.pop(k, None)
            if len(self._states) >= self.max_sessions:
                raise CapacityExceeded("Session capacity reached")
            if state.id in self._states:
                raise ValueError("Session already exists")
            self._states[state.id] = state.model_copy(deep=True)

    def get(self, session_id, owner):
        with self._lock:
            state = self._states.get(session_id)
            _check(state, owner, self.clock)
            return state.model_copy(deep=True)

    def transact(
        self,
        session_id,
        owner,
        fn,
        *,
        expected_revision=None,
        idempotency_key=None,
        request_digest=None,
    ):
        with self._lock:
            original = self._states.get(session_id)
            _check(original, owner, self.clock)
            cache = self._cache.setdefault(session_id, {})
            if idempotency_key and idempotency_key in cache:
                digest, response = cache[idempotency_key]
                if digest != request_digest:
                    raise IdempotencyConflict("Key was used for another request")
                return json.loads(response)
            _check(original, owner, self.clock, expected_revision)
            if idempotency_key and len(cache) >= 20000:
                raise CapacityExceeded("Session idempotency capacity reached")
            state = original.model_copy(deep=True)
            response = fn(state)
            encoded = canonical_json(response)
            self._states[session_id] = state
            if idempotency_key:
                cache[idempotency_key] = (request_digest, encoded)
            return json.loads(encoded)

    def delete(self, session_id, owner):
        with self._lock:
            _check(self._states.get(session_id), owner, self.clock)
            self._states.pop(session_id)
            self._cache.pop(session_id, None)


class SQLiteStore:
    """Single-host, multi-process transactions. Use a local persistent volume."""

    def __init__(self, path, max_sessions=10000, clock=time.time):
        self.path, self.max_sessions, self.clock = str(path), max_sessions, clock
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL, data TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS responses (sid TEXT NOT NULL, key TEXT NOT NULL, digest TEXT, data TEXT NOT NULL, PRIMARY KEY(sid,key), FOREIGN KEY(sid) REFERENCES sessions(id) ON DELETE CASCADE)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS session_expiry ON sessions(expires)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def create(self, state):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM sessions WHERE expires<=?", (self.clock(),))
            if db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] >= self.max_sessions:
                raise CapacityExceeded("Session capacity reached")
            db.execute(
                "INSERT INTO sessions VALUES(?,?,?,?)",
                (state.id, state.owner, state.expires_at, state.model_dump_json()),
            )

    def _read(self, db, sid, owner):
        row = db.execute(
            "SELECT data FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        state = SessionState.model_validate_json(row[0]) if row else None
        _check(state, owner, self.clock)
        return state

    def get(self, session_id, owner):
        with self._connect() as db:
            return self._read(db, session_id, owner)

    def transact(
        self,
        session_id,
        owner,
        fn,
        *,
        expected_revision=None,
        idempotency_key=None,
        request_digest=None,
    ):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._read(db, session_id, owner)
            if idempotency_key:
                row = db.execute(
                    "SELECT digest,data FROM responses WHERE sid=? AND key=?",
                    (session_id, idempotency_key),
                ).fetchone()
                if row:
                    if row[0] != request_digest:
                        raise IdempotencyConflict("Key was used for another request")
                    return json.loads(row[1])
                if (
                    db.execute(
                        "SELECT COUNT(*) FROM responses WHERE sid=?", (session_id,)
                    ).fetchone()[0]
                    >= 20000
                ):
                    raise CapacityExceeded("Session idempotency capacity reached")
            _check(state, owner, self.clock, expected_revision)
            response = fn(state)
            encoded = canonical_json(response)
            db.execute(
                "UPDATE sessions SET data=? WHERE id=?", (state.model_dump_json(), session_id)
            )
            if idempotency_key:
                db.execute(
                    "INSERT INTO responses VALUES(?,?,?,?)",
                    (session_id, idempotency_key, request_digest, encoded),
                )
            return json.loads(encoded)

    def delete(self, session_id, owner):
        with self._connect() as db:
            self._read(db, session_id, owner)
            db.execute("DELETE FROM sessions WHERE id=? AND owner=?", (session_id, owner))
