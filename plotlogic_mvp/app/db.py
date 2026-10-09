from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH, ensure_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dumps(value: Any) -> str:
    # Event payloads routinely carry Path objects; never let logging raise and mask the real error.
    return json.dumps(value, ensure_ascii=True, sort_keys=False, default=str)


def loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            create table if not exists jobs (
                job_id text primary key,
                paper_id integer,
                status text not null,
                phase text not null,
                progress integer not null default 0,
                source_pdf_name text not null,
                title text,
                error text,
                chart_count integer not null default 0,
                target_count integer not null default 0,
                skipped_count integer not null default 0,
                created_at text not null,
                updated_at text not null
            );

            create table if not exists job_events (
                id integer primary key autoincrement,
                job_id text not null,
                level text not null,
                message text not null,
                payload_json text,
                created_at text not null
            );

            create table if not exists charts (
                chart_id text primary key,
                job_id text not null,
                paper_id integer not null,
                image_name text not null,
                image_rel_path text not null,
                image_abs_path text not null,
                status text not null,
                panel_type text,
                extraction_family text,
                chart_family text,
                is_chart integer not null default 0,
                is_target_dual_y integer not null default 0,
                confidence real not null default 0,
                skip_reason text,
                result_dir text,
                overlay_path text,
                csv_path text,
                series_count integer not null default 0,
                issues_json text not null default '[]',
                created_at text not null,
                updated_at text not null
            );

            create table if not exists paper_extractions (
                paper_id integer primary key,
                latest_job_id text,
                status text not null,
                phase text not null,
                image_count integer not null default 0,
                target_count integer not null default 0,
                skipped_count integer not null default 0,
                error text,
                created_at text not null,
                updated_at text not null
            );
            """
        )
        chart_columns = {row[1] for row in conn.execute("pragma table_info(charts)").fetchall()}
        if "panel_type" not in chart_columns:
            conn.execute("alter table charts add column panel_type text")
        if "extraction_family" not in chart_columns:
            conn.execute("alter table charts add column extraction_family text")
        job_columns = {row[1] for row in conn.execute("pragma table_info(jobs)").fetchall()}
        if "prompt_variant" not in job_columns:
            # Jobs from before prompt selection existed all used the baseline.
            conn.execute("alter table jobs add column prompt_variant text")
        if "method" not in job_columns:
            conn.execute("alter table jobs add column method text")
            conn.execute("update jobs set method='baseline' where method is null")
        if "method" not in chart_columns:
            conn.execute("alter table charts add column method text")
            conn.execute("update charts set method='baseline' where method is null")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def create_job(
    job_id: str,
    paper_id: int,
    source_pdf_name: str,
    title: str,
    prompt_variant: str | None = None,
    method: str | None = None,
) -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            insert into jobs (
                job_id, paper_id, status, phase, progress,
                source_pdf_name, title, prompt_variant, method, created_at, updated_at
            ) values (?, ?, 'queued', 'queued', 0, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, paper_id, source_pdf_name, title, prompt_variant, method, now, now),
        )
        conn.execute(
            """
            insert into paper_extractions (
                paper_id, latest_job_id, status, phase, created_at, updated_at
            ) values (?, ?, 'queued', 'queued', ?, ?)
            on conflict(paper_id) do update set
                latest_job_id=excluded.latest_job_id,
                status=excluded.status,
                phase=excluded.phase,
                error=null,
                updated_at=excluded.updated_at
            """,
            (paper_id, job_id, now, now),
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{key}=?" for key in fields)
    values = list(fields.values())
    values.append(job_id)
    with connect() as conn:
        conn.execute(f"update jobs set {assignments} where job_id=?", values)


def update_paper_extraction(paper_id: int, **fields: Any) -> None:
    now = utc_now()
    fields.setdefault("updated_at", now)
    fields.setdefault("created_at", now)
    existing_columns = [
        "paper_id",
        "latest_job_id",
        "status",
        "phase",
        "image_count",
        "target_count",
        "skipped_count",
        "error",
        "created_at",
        "updated_at",
    ]
    payload = {"paper_id": paper_id, **fields}
    columns = [column for column in existing_columns if column in payload]
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [column for column in columns if column not in {"paper_id", "created_at"}]
    updates = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
    with connect() as conn:
        conn.execute(
            f"""
            insert into paper_extractions ({", ".join(columns)})
            values ({placeholders})
            on conflict(paper_id) do update set {updates}
            """,
            [payload[column] for column in columns],
        )


def add_event(job_id: str, message: str, *, level: str = "info", payload: Any = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            insert into job_events (job_id, level, message, payload_json, created_at)
            values (?, ?, ?, ?, ?)
            """,
            (job_id, level, message, dumps(payload) if payload is not None else None, utc_now()),
        )


def upsert_chart(**fields: Any) -> None:
    now = utc_now()
    fields.setdefault("created_at", now)
    fields["updated_at"] = now
    fields["issues_json"] = dumps(fields.get("issues", []))
    fields.pop("issues", None)
    columns = list(fields)
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [col for col in columns if col != "chart_id"]
    updates = ", ".join(f"{col}=excluded.{col}" for col in update_columns)
    values = [fields[col] for col in columns]
    with connect() as conn:
        conn.execute(
            f"""
            insert into charts ({", ".join(columns)})
            values ({placeholders})
            on conflict(chart_id) do update set {updates}
            """,
            values,
        )


def delete_job(job_id: str) -> bool:
    """Delete a job row and everything that references it (events, charts).

    Returns False if the job did not exist. Does not touch paper_extractions.latest_job_id — a
    paper whose only job gets deleted just keeps pointing at a job_id that no longer exists, the
    same as it would for any other job that failed to leave data behind.
    """
    with connect() as conn:
        cursor = conn.execute("delete from jobs where job_id=?", (job_id,))
        deleted = cursor.rowcount > 0
        conn.execute("delete from job_events where job_id=?", (job_id,))
        conn.execute("delete from charts where job_id=?", (job_id,))
    return deleted


def get_job(job_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        return row_to_dict(conn.execute("select * from jobs where job_id=?", (job_id,)).fetchone())


def list_jobs(limit: int = 30) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "select * from jobs order by created_at desc limit ?",
            (max(1, min(limit, 100)),),
        ).fetchall()
    return [dict(row) for row in rows]


def list_jobs_for_paper(paper_id: int, limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "select * from jobs where paper_id=? order by created_at desc limit ?",
            (paper_id, max(1, min(limit, 100))),
        ).fetchall()
    return [dict(row) for row in rows]


def get_paper_extraction(paper_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        return row_to_dict(
            conn.execute(
                "select * from paper_extractions where paper_id=?",
                (paper_id,),
            ).fetchone()
        )


def list_paper_extractions() -> dict[int, dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("select * from paper_extractions").fetchall()
    return {int(row["paper_id"]): dict(row) for row in rows}


def list_events(job_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "select * from job_events where job_id=? order by id asc",
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_charts(job_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "select * from charts where job_id=? order by image_name asc",
            (job_id,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["issues"] = loads(item.pop("issues_json", "[]"), [])
        out.append(item)
    return out


def list_charts_for_paper(paper_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            select * from charts
            where paper_id=?
            order by image_name asc, updated_at desc
            """,
            (paper_id,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["issues"] = loads(item.pop("issues_json", "[]"), [])
        out.append(item)
    return out


def get_chart(job_id: str, chart_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "select * from charts where job_id=? and chart_id=?",
            (job_id, chart_id),
        ).fetchone()
    item = row_to_dict(row)
    if item:
        item["issues"] = loads(item.pop("issues_json", "[]"), [])
    return item


def safe_relative(path: str | Path, root: str | Path) -> Path:
    root_path = Path(root).resolve()
    target = Path(path).resolve()
    if target != root_path and root_path not in target.parents:
        raise ValueError(f"path is outside root: {target}")
    return target.relative_to(root_path)
