"""SQLite-backed dynamic task pools with strict round barriers.

This module is deliberately independent from the legacy CSV shard runners.
SQLite is the authoritative scheduler state; CSV files are only snapshots
managed by a coordinator that integrates this pool with the existing queue.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

ROUND_NAMES = (
    "same_city_matching",
    "same_city_gsc",
    "same_city_mc",
    "cross_city_matching",
    "cross_city_gsc",
    "cross_city_mc",
)

TERMINAL_STATUSES = frozenset(
    {"matched_labelled", "gsc_labelled", "mc_labelled", "skipped"}
)

STATUS_TO_ROUND = {
    "pending": "same_city_matching",
    "matching_running": "same_city_matching",
    "gsc_pending": "same_city_gsc",
    "gsc_running": "same_city_gsc",
    "mc_pending": "same_city_mc",
    "mc_running": "same_city_mc",
    "cross_matching_running": "cross_city_matching",
    "cross_gsc_running": "cross_city_gsc",
    "cross_mc_running": "cross_city_mc",
}


@dataclass(frozen=True)
class RunIdentity:
    """Inputs that make a scheduler run distinct and reproducible."""

    run_id: str
    specification_fingerprint: str
    receipt_digest: str
    sample_digest: str
    queue_namespace: str


@dataclass(frozen=True)
class PoolTask:
    treatment_order: int
    outcome_family: str
    round_name: str
    queue_status: str
    attempt_count: int
    lease_token: str
    lease_until: float


@dataclass(frozen=True)
class PoolRun:
    run_id: str
    specification_fingerprint: str
    receipt_digest: str
    sample_digest: str
    queue_namespace: str
    status: str


def digest_json(value: object) -> str:
    """Return a stable SHA-256 digest for run metadata."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def new_run_id() -> str:
    return uuid.uuid4().hex


class RoundPool:
    """A small transactional scheduler for one causal-label run."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        lease_seconds: float = 3600.0,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        self.db_path = Path(db_path)
        self.lease_seconds = float(lease_seconds)
        self.busy_timeout_ms = int(busy_timeout_ms)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.db_path,
            timeout=max(self.busy_timeout_ms / 1000.0, 1.0),
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    specification_fingerprint TEXT NOT NULL,
                    receipt_digest TEXT NOT NULL,
                    sample_digest TEXT NOT NULL,
                    queue_namespace TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'completed', 'failed')),
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rounds (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'open', 'closed')),
                    opened_at REAL,
                    closed_at REAL,
                    PRIMARY KEY (run_id, name),
                    UNIQUE (run_id, ordinal)
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    treatment_order INTEGER NOT NULL,
                    outcome_family TEXT NOT NULL,
                    round_name TEXT,
                    queue_status TEXT NOT NULL,
                    pool_status TEXT NOT NULL CHECK (pool_status IN ('pending', 'running', 'completed', 'failed')),
                    worker_id TEXT,
                    lease_token TEXT,
                    lease_until REAL,
                    heartbeat_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    selected_method TEXT,
                    failure_reason TEXT,
                    last_error TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (run_id, treatment_order, outcome_family),
                    FOREIGN KEY (run_id, round_name) REFERENCES rounds(run_id, name)
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_claimable
                    ON tasks(run_id, round_name, pool_status, lease_until, treatment_order, outcome_family);

                CREATE TABLE IF NOT EXISTS resources (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    resource_key TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    lease_token TEXT NOT NULL,
                    lease_until REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    PRIMARY KEY (run_id, resource_key)
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    treatment_order INTEGER,
                    outcome_family TEXT,
                    event_type TEXT NOT NULL,
                    worker_id TEXT,
                    lease_token TEXT,
                    detail TEXT,
                    created_at REAL NOT NULL
                );
                """
            )

    def create_run(
        self,
        identity: RunIdentity,
        tasks: Iterable[Mapping[str, object]],
    ) -> None:
        """Create a run from queue rows and open its first non-empty Round.

        A row with a terminal queue status is recorded as completed. Any
        running legacy status is safely normalized to pending in its current
        Round because its old process/lease cannot be trusted after restart.
        """

        self.initialize()
        task_rows = list(tasks)
        now = time.time()
        normalized: list[
            tuple[int, str, str | None, str, str, str | None, str | None]
        ] = []
        seen: set[tuple[int, str]] = set()
        round_has_tasks = {name: False for name in ROUND_NAMES}
        for row in task_rows:
            order = int(row["treatment_order"])  # type: ignore[call-overload]
            family = str(row["outcome_family"])
            status = str(row.get("status", "pending"))
            key = (order, family)
            if key in seen:
                raise ValueError(f"duplicate task key: {key}")
            seen.add(key)
            if status in TERMINAL_STATUSES:
                selected_method = row.get("selected_method")
                failure_reason = row.get("failure_reason")
                if selected_method is not None and str(selected_method) == "<NA>":
                    selected_method = None
                if failure_reason is not None and str(failure_reason) == "<NA>":
                    failure_reason = None
                normalized.append(
                    (
                        order,
                        family,
                        None,
                        status,
                        "completed",
                        None if selected_method is None else str(selected_method),
                        None if failure_reason is None else str(failure_reason),
                    )
                )
                continue
            round_name = STATUS_TO_ROUND.get(status)
            if round_name is None:
                raise ValueError(f"unknown queue status {status!r} for task {key}")
            round_has_tasks[round_name] = True
            selected_method = row.get("selected_method")
            failure_reason = row.get("failure_reason")
            if selected_method is not None and str(selected_method) == "<NA>":
                selected_method = None
            if failure_reason is not None and str(failure_reason) == "<NA>":
                failure_reason = None
            normalized.append(
                (
                    order,
                    family,
                    round_name,
                    status,
                    "pending",
                    None if selected_method is None else str(selected_method),
                    None if failure_reason is None else str(failure_reason),
                )
            )

        first_open_ordinal = next(
            (i for i, name in enumerate(ROUND_NAMES) if round_has_tasks[name]),
            None,
        )
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    """
                    INSERT INTO runs (
                        run_id, specification_fingerprint, receipt_digest,
                        sample_digest, queue_namespace, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (
                        identity.run_id,
                        identity.specification_fingerprint,
                        identity.receipt_digest,
                        identity.sample_digest,
                        identity.queue_namespace,
                        now,
                    ),
                )
                for ordinal, name in enumerate(ROUND_NAMES):
                    has_tasks = round_has_tasks[name]
                    state = "closed" if not has_tasks or ordinal < (first_open_ordinal or 0) else "pending"
                    if first_open_ordinal == ordinal:
                        state = "open"
                    db.execute(
                        "INSERT INTO rounds (run_id, ordinal, name, state, opened_at) VALUES (?, ?, ?, ?, ?)",
                        (identity.run_id, ordinal, name, state, now if state == "open" else None),
                    )
                for (
                    order,
                    family,
                    round_name,
                    status,
                    pool_status,
                    selected_method,
                    failure_reason,
                ) in normalized:
                    db.execute(
                        """
                        INSERT INTO tasks (
                            run_id, treatment_order, outcome_family, round_name,
                            queue_status, pool_status, selected_method, failure_reason, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identity.run_id,
                            order,
                            family,
                            round_name,
                            status,
                            pool_status,
                            selected_method,
                            failure_reason,
                            now,
                        ),
                    )
                self._event(
                    db,
                    identity.run_id,
                    "run_created",
                    detail={"task_count": len(normalized), "first_open_ordinal": first_open_ordinal},
                )
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise

    def get_run(self, run_id: str) -> PoolRun | None:
        self.initialize()
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            return PoolRun(
                run_id=str(row["run_id"]),
                specification_fingerprint=str(row["specification_fingerprint"]),
                receipt_digest=str(row["receipt_digest"]),
                sample_digest=str(row["sample_digest"]),
                queue_namespace=str(row["queue_namespace"]),
                status=str(row["status"]),
            )

    def _event(
        self,
        db: sqlite3.Connection,
        run_id: str,
        event_type: str,
        *,
        treatment_order: int | None = None,
        outcome_family: str | None = None,
        worker_id: str | None = None,
        lease_token: str | None = None,
        detail: object | None = None,
    ) -> None:
        db.execute(
            """
            INSERT INTO events (
                run_id, treatment_order, outcome_family, event_type,
                worker_id, lease_token, detail, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                treatment_order,
                outcome_family,
                event_type,
                worker_id,
                lease_token,
                None if detail is None else json.dumps(detail, sort_keys=True, default=str),
                time.time(),
            ),
        )

    def _round_ordinal(self, db: sqlite3.Connection, run_id: str, round_name: str) -> int:
        row = db.execute(
            "SELECT ordinal FROM rounds WHERE run_id = ? AND name = ?",
            (run_id, round_name),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown Round {round_name!r} for run {run_id!r}")
        return int(row["ordinal"])

    def claim_next(
        self,
        run_id: str,
        round_name: str,
        worker_id: str,
        *,
        lease_seconds: float | None = None,
    ) -> PoolTask | None:
        """Atomically claim one task, excluding tasks whose order is locked."""

        lease_duration = self.lease_seconds if lease_seconds is None else float(lease_seconds)
        now = time.time()
        until = now + lease_duration
        token = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                round_row = db.execute(
                    "SELECT ordinal, state FROM rounds WHERE run_id = ? AND name = ?",
                    (run_id, round_name),
                ).fetchone()
                if round_row is None:
                    raise KeyError(f"unknown Round {round_name!r}")
                ordinal = int(round_row["ordinal"])
                if round_row["state"] != "open":
                    db.execute("COMMIT")
                    return None
                earlier = db.execute(
                    "SELECT COUNT(*) AS n FROM rounds WHERE run_id = ? AND ordinal < ? AND state != 'closed'",
                    (run_id, ordinal),
                ).fetchone()
                if int(earlier["n"]) != 0:
                    db.execute("COMMIT")
                    return None
                db.execute("DELETE FROM resources WHERE run_id = ? AND lease_until <= ?", (run_id, now))
                row = db.execute(
                    """
                    SELECT t.*
                    FROM tasks AS t
                    WHERE t.run_id = ?
                      AND t.round_name = ?
                      AND t.pool_status = 'pending'
                      AND NOT EXISTS (
                          SELECT 1 FROM resources AS r
                          WHERE r.run_id = t.run_id
                            AND r.resource_key = 'order:' || CAST(t.treatment_order AS TEXT)
                            AND r.lease_until > ?
                      )
                    ORDER BY t.treatment_order, t.outcome_family
                    LIMIT 1
                    """,
                    (run_id, round_name, now),
                ).fetchone()
                if row is None:
                    db.execute("COMMIT")
                    return None
                order = int(row["treatment_order"])
                family = str(row["outcome_family"])
                updated = db.execute(
                    """
                    UPDATE tasks
                    SET pool_status = 'running', worker_id = ?, lease_token = ?,
                        lease_until = ?, heartbeat_at = ?, attempt_count = attempt_count + 1,
                        updated_at = ?
                    WHERE run_id = ? AND treatment_order = ? AND outcome_family = ?
                      AND pool_status = 'pending'
                    """,
                    (worker_id, token, until, now, now, run_id, order, family),
                ).rowcount
                if updated != 1:
                    db.execute("ROLLBACK")
                    return None
                db.execute(
                    """
                    INSERT INTO resources (run_id, resource_key, worker_id, lease_token, lease_until, heartbeat_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, f"order:{order}", worker_id, token, until, now),
                )
                self._event(
                    db,
                    run_id,
                    "task_claimed",
                    treatment_order=order,
                    outcome_family=family,
                    worker_id=worker_id,
                    lease_token=token,
                    detail={"round_name": round_name, "lease_until": until},
                )
                db.execute("COMMIT")
                return PoolTask(
                    treatment_order=order,
                    outcome_family=family,
                    round_name=round_name,
                    queue_status=str(row["queue_status"]),
                    attempt_count=int(row["attempt_count"]) + 1,
                    lease_token=token,
                    lease_until=until,
                )
            except Exception:
                db.execute("ROLLBACK")
                raise

    def heartbeat(self, run_id: str, lease: PoolTask, worker_id: str, *, lease_seconds: float | None = None) -> bool:
        duration = self.lease_seconds if lease_seconds is None else float(lease_seconds)
        now = time.time()
        until = now + duration
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                """
                UPDATE tasks
                SET lease_until = ?, heartbeat_at = ?, updated_at = ?
                WHERE run_id = ? AND treatment_order = ? AND outcome_family = ?
                  AND pool_status = 'running' AND worker_id = ? AND lease_token = ?
                """,
                (
                    until,
                    now,
                    now,
                    run_id,
                    lease.treatment_order,
                    lease.outcome_family,
                    worker_id,
                    lease.lease_token,
                ),
            ).rowcount
            if updated:
                db.execute(
                    """
                    UPDATE resources
                    SET lease_until = ?, heartbeat_at = ?
                    WHERE run_id = ? AND resource_key = ? AND worker_id = ? AND lease_token = ?
                    """,
                    (until, now, run_id, f"order:{lease.treatment_order}", worker_id, lease.lease_token),
                )
            db.execute("COMMIT")
            return updated == 1

    def finish(
        self,
        run_id: str,
        lease: PoolTask,
        worker_id: str,
        *,
        queue_status: str,
        next_round: str | None = None,
        error: str | None = None,
        selected_method: str | None = None,
        failure_reason: str | None = None,
    ) -> bool:
        """Commit only the current lease; stale workers cannot fence it."""

        if queue_status in TERMINAL_STATUSES:
            if next_round is not None:
                raise ValueError("terminal task cannot have next_round")
            pool_status = "completed"
            target_round = None
        else:
            target_round = next_round or STATUS_TO_ROUND.get(queue_status)
            if target_round not in ROUND_NAMES:
                raise ValueError(f"nonterminal queue status needs a valid next Round: {queue_status!r}")
            current_ordinal = ROUND_NAMES.index(lease.round_name)
            if ROUND_NAMES.index(target_round) <= current_ordinal:
                raise ValueError("task may only move forward to a later Round")
            pool_status = "pending"
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                updated = db.execute(
                    """
                    UPDATE tasks
                    SET round_name = ?, queue_status = ?, pool_status = ?,
                        worker_id = NULL, lease_token = NULL, lease_until = NULL,
                        heartbeat_at = NULL, selected_method = ?, failure_reason = ?,
                        last_error = ?, updated_at = ?
                    WHERE run_id = ? AND treatment_order = ? AND outcome_family = ?
                      AND pool_status = 'running' AND worker_id = ? AND lease_token = ?
                    """,
                    (
                        target_round,
                        queue_status,
                        pool_status,
                        selected_method,
                        failure_reason,
                        error,
                        now,
                        run_id,
                        lease.treatment_order,
                        lease.outcome_family,
                        worker_id,
                        lease.lease_token,
                    ),
                ).rowcount
                if updated != 1:
                    db.execute("ROLLBACK")
                    return False
                db.execute(
                    "DELETE FROM resources WHERE run_id = ? AND resource_key = ? AND worker_id = ? AND lease_token = ?",
                    (run_id, f"order:{lease.treatment_order}", worker_id, lease.lease_token),
                )
                self._event(
                    db,
                    run_id,
                    "task_finished",
                    treatment_order=lease.treatment_order,
                    outcome_family=lease.outcome_family,
                    worker_id=worker_id,
                    lease_token=lease.lease_token,
                    detail={"queue_status": queue_status, "next_round": target_round, "error": error},
                )
                db.execute("COMMIT")
                return True
            except Exception:
                db.execute("ROLLBACK")
                raise

    def fail(self, run_id: str, lease: PoolTask, worker_id: str, error: str, *, retry: bool = True) -> bool:
        """Retry a task or record a hard failure without silently skipping it."""

        if retry:
            next_round = lease.round_name
            # finish() intentionally forbids same-round transitions, so this
            # method performs the retry transition directly.
            now = time.time()
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                updated = db.execute(
                    """
                    UPDATE tasks
                    SET pool_status = 'pending', worker_id = NULL, lease_token = NULL,
                        lease_until = NULL, heartbeat_at = NULL, last_error = ?, updated_at = ?
                    WHERE run_id = ? AND treatment_order = ? AND outcome_family = ?
                      AND pool_status = 'running' AND worker_id = ? AND lease_token = ?
                    """,
                    (
                        error,
                        now,
                        run_id,
                        lease.treatment_order,
                        lease.outcome_family,
                        worker_id,
                        lease.lease_token,
                    ),
                ).rowcount
                if updated == 1:
                    db.execute(
                        "DELETE FROM resources WHERE run_id = ? AND resource_key = ? AND worker_id = ? AND lease_token = ?",
                        (run_id, f"order:{lease.treatment_order}", worker_id, lease.lease_token),
                    )
                    self._event(
                        db,
                        run_id,
                        "task_retry",
                        treatment_order=lease.treatment_order,
                        outcome_family=lease.outcome_family,
                        worker_id=worker_id,
                        lease_token=lease.lease_token,
                        detail={"round_name": next_round, "error": error},
                    )
                db.execute("COMMIT")
                return updated == 1
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                """
                UPDATE tasks
                SET pool_status = 'failed', worker_id = NULL, lease_token = NULL,
                    lease_until = NULL, heartbeat_at = NULL, last_error = ?, updated_at = ?
                WHERE run_id = ? AND treatment_order = ? AND outcome_family = ?
                  AND pool_status = 'running' AND worker_id = ? AND lease_token = ?
                """,
                (
                    error,
                    now,
                    run_id,
                    lease.treatment_order,
                    lease.outcome_family,
                    worker_id,
                    lease.lease_token,
                ),
            ).rowcount
            if updated == 1:
                db.execute(
                    "DELETE FROM resources WHERE run_id = ? AND resource_key = ? AND worker_id = ? AND lease_token = ?",
                    (run_id, f"order:{lease.treatment_order}", worker_id, lease.lease_token),
                )
                self._event(
                    db,
                    run_id,
                    "task_failed",
                    treatment_order=lease.treatment_order,
                    outcome_family=lease.outcome_family,
                    worker_id=worker_id,
                    lease_token=lease.lease_token,
                    detail={"error": error},
                )
            db.execute("COMMIT")
            return updated == 1

    def requeue_worker(self, run_id: str, worker_id: str, reason: str = "worker exited") -> int:
        """Fence and requeue every lease owned by a failed worker."""

        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT treatment_order, outcome_family, lease_token FROM tasks WHERE run_id = ? AND pool_status = 'running' AND worker_id = ?",
                (run_id, worker_id),
            ).fetchall()
            for row in rows:
                db.execute(
                    """
                    UPDATE tasks
                    SET pool_status = 'pending', worker_id = NULL, lease_token = NULL,
                        lease_until = NULL, heartbeat_at = NULL, last_error = ?, updated_at = ?
                    WHERE run_id = ? AND treatment_order = ? AND outcome_family = ?
                    """,
                    (reason, now, run_id, row["treatment_order"], row["outcome_family"]),
                )
                db.execute(
                    "DELETE FROM resources WHERE run_id = ? AND resource_key = ? AND lease_token = ?",
                    (run_id, f"order:{row['treatment_order']}", row["lease_token"]),
                )
                self._event(
                    db,
                    run_id,
                    "worker_requeue",
                    treatment_order=int(row["treatment_order"]),
                    outcome_family=str(row["outcome_family"]),
                    worker_id=worker_id,
                    lease_token=row["lease_token"],
                    detail={"reason": reason},
                )
            db.execute("COMMIT")
            return len(rows)

    def requeue_expired(self, run_id: str) -> int:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT treatment_order, outcome_family, worker_id, lease_token FROM tasks WHERE run_id = ? AND pool_status = 'running' AND lease_until <= ?",
                (run_id, now),
            ).fetchall()
            for row in rows:
                db.execute(
                    """
                    UPDATE tasks
                    SET pool_status = 'pending', worker_id = NULL, lease_token = NULL,
                        lease_until = NULL, heartbeat_at = NULL, last_error = 'lease expired', updated_at = ?
                    WHERE run_id = ? AND treatment_order = ? AND outcome_family = ?
                    """,
                    (now, run_id, row["treatment_order"], row["outcome_family"]),
                )
                db.execute(
                    "DELETE FROM resources WHERE run_id = ? AND resource_key = ? AND lease_token = ?",
                    (run_id, f"order:{row['treatment_order']}", row["lease_token"]),
                )
                self._event(
                    db,
                    run_id,
                    "lease_expired",
                    treatment_order=int(row["treatment_order"]),
                    outcome_family=str(row["outcome_family"]),
                    worker_id=row["worker_id"],
                    lease_token=row["lease_token"],
                )
            db.execute("DELETE FROM resources WHERE run_id = ? AND lease_until <= ?", (run_id, now))
            db.execute("COMMIT")
            return len(rows)

    def requeue_failed(
        self,
        run_id: str,
        *,
        round_name: str | None = None,
        reason: str = "operator requeued failed task",
    ) -> int:
        """Return hard-failed tasks to pending so a run can be resumed.

        ``fail(..., retry=False)`` is intentionally fail-closed, but without an
        explicit repair path a single hard failure would make the run
        permanently unresumable under the same run_id.
        """

        if round_name is not None and round_name not in ROUND_NAMES:
            raise ValueError(f"unknown Round {round_name!r}")
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            sql = (
                "UPDATE tasks SET pool_status = 'pending', worker_id = NULL, "
                "lease_token = NULL, lease_until = NULL, heartbeat_at = NULL, "
                "selected_method = NULL, failure_reason = NULL, last_error = ?, "
                "updated_at = ? WHERE run_id = ? AND pool_status = 'failed'"
            )
            parameters: list[object] = [reason, now, run_id]
            if round_name is not None:
                sql += " AND round_name = ?"
                parameters.append(round_name)
            updated = db.execute(sql, parameters).rowcount
            if updated:
                self._event(
                    db,
                    run_id,
                    "failed_tasks_requeued",
                    detail={"count": int(updated), "round_name": round_name},
                )
            db.execute("COMMIT")
            return int(updated)

    def close_round_if_drained(self, run_id: str, round_name: str) -> bool:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT ordinal, state FROM rounds WHERE run_id = ? AND name = ?",
                (run_id, round_name),
            ).fetchone()
            if row is None:
                raise KeyError(round_name)
            if row["state"] == "closed":
                db.execute("COMMIT")
                return True
            active = db.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE run_id = ? AND round_name = ? AND pool_status IN ('pending', 'running')",
                (run_id, round_name),
            ).fetchone()
            if int(active["n"]) != 0:
                db.execute("COMMIT")
                return False
            db.execute(
                "UPDATE rounds SET state = 'closed', closed_at = ? WHERE run_id = ? AND name = ?",
                (now, run_id, round_name),
            )
            next_row = db.execute(
                "SELECT name FROM rounds WHERE run_id = ? AND ordinal > ? AND state = 'pending' ORDER BY ordinal LIMIT 1",
                (run_id, row["ordinal"]),
            ).fetchone()
            if next_row is not None:
                db.execute(
                    """
                    UPDATE rounds
                    SET state = 'open', opened_at = ?
                    WHERE run_id = ? AND name = ? AND state = 'pending'
                      AND NOT EXISTS (
                          SELECT 1 FROM rounds AS prior
                          WHERE prior.run_id = rounds.run_id
                            AND prior.ordinal < rounds.ordinal
                            AND prior.state != 'closed'
                      )
                    """,
                    (now, run_id, next_row["name"]),
                )
            self._event(db, run_id, "round_closed", detail={"round_name": round_name})
            db.execute("COMMIT")
            return True

    def counts(self, run_id: str) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT pool_status, COUNT(*) AS n FROM tasks WHERE run_id = ? GROUP BY pool_status",
                (run_id,),
            ).fetchall()
            return {str(row["pool_status"]): int(row["n"]) for row in rows}

    def round_states(self, run_id: str) -> list[dict[str, object]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT ordinal, name, state, opened_at, closed_at FROM rounds WHERE run_id = ? ORDER BY ordinal",
                (run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def task_records(self, run_id: str) -> list[dict[str, object]]:
        """Return the authoritative task state for coordinator snapshots."""

        with self._connect() as db:
            rows = db.execute(
                """
                SELECT treatment_order, outcome_family, round_name, queue_status,
                       pool_status, selected_method, failure_reason, last_error,
                       attempt_count, updated_at
                FROM tasks
                WHERE run_id = ?
                ORDER BY treatment_order, outcome_family
                """,
                (run_id,),
            ).fetchall()
            return [dict(row) for row in rows]
