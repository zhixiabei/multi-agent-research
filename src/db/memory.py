"""长期记忆存储 —— SQLite，存对话历史 + 研究报告 + Agent 执行记录"""
import os
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Any

# 数据库文件路径：项目根目录 / data / memory.db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── 建表 ──

def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '未命名对话',
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            topic TEXT NOT NULL,
            summary TEXT,
            sections TEXT NOT NULL DEFAULT '[]',
            conclusion TEXT,
            unresolved TEXT NOT NULL DEFAULT '[]',
            agent_trace TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS agent_traces (
            id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            input_data TEXT,
            output_data TEXT,
            duration_ms INTEGER,
            success INTEGER NOT NULL DEFAULT 1,
            error_msg TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (report_id) REFERENCES reports(id)
        );

        CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_reports_session ON reports(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_traces_report ON agent_traces(report_id);
    """)
    conn.commit()
    conn.close()


# ── 会话管理 ──

def create_session(title: str = "未命名对话") -> str:
    session_id = str(uuid.uuid4())[:8]
    conn = _get_conn()
    conn.execute("INSERT INTO sessions (id, title) VALUES (?, ?)", (session_id, title))
    conn.commit()
    conn.close()
    return session_id


def list_sessions(limit: int = 20) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def touch_session(session_id: str):
    conn = _get_conn()
    conn.execute("UPDATE sessions SET updated_at=datetime('now','localtime') WHERE id=?", (session_id,))
    conn.commit()
    conn.close()


# ── 对话历史 ──

def save_message(session_id: str, role: str, content: str, metadata: dict | None = None):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO conversations (id, session_id, role, content, metadata) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4())[:8], session_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
    )
    touch_session(session_id)
    conn.commit()
    conn.close()


def get_history(session_id: str, limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, metadata, created_at FROM conversations WHERE session_id=? ORDER BY created_at ASC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_context_for_llm(session_id: str, max_messages: int = 20) -> list[dict[str, str]]:
    """取最近 N 条对话，格式化为 LLM 可用的 messages 列表"""
    history = get_history(session_id, limit=max_messages)
    return [{"role": h["role"], "content": h["content"]} for h in history]


# ── 研究报告 ──

def save_report(
    topic: str,
    summary: str,
    sections: list[dict],
    conclusion: str,
    unresolved: list[str],
    session_id: str | None = None,
    agent_trace: list[dict] | None = None,
) -> str:
    report_id = str(uuid.uuid4())[:8]
    conn = _get_conn()
    conn.execute(
        """INSERT INTO reports (id, session_id, topic, summary, sections, conclusion, unresolved, agent_trace)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            report_id, session_id, topic, summary,
            json.dumps(sections, ensure_ascii=False),
            conclusion,
            json.dumps(unresolved, ensure_ascii=False),
            json.dumps(agent_trace or [], ensure_ascii=False),
        ),
    )
    if session_id:
        touch_session(session_id)
    conn.commit()
    conn.close()
    return report_id


def get_report(report_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["sections"] = json.loads(d["sections"])
    d["unresolved"] = json.loads(d.get("unresolved", "[]"))
    d["agent_trace"] = json.loads(d.get("agent_trace", "[]"))
    return d


def list_reports(session_id: str | None = None, limit: int = 20) -> list[dict]:
    conn = _get_conn()
    if session_id:
        rows = conn.execute(
            "SELECT id, session_id, topic, summary, created_at FROM reports WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, session_id, topic, summary, created_at FROM reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_reports_by_session(session_id: str) -> list[dict]:
    return list_reports(session_id=session_id)


# ── Agent 执行记录 ──

def save_trace(report_id: str, agent_name: str, input_data: Any, output_data: Any,
               duration_ms: int, success: bool = True, error_msg: str | None = None):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO agent_traces (id, report_id, agent_name, input_data, output_data, duration_ms, success, error_msg) VALUES (?,?,?,?,?,?,?,?)",
        (
            str(uuid.uuid4())[:8], report_id, agent_name,
            json.dumps(input_data, ensure_ascii=False, default=str),
            json.dumps(output_data, ensure_ascii=False, default=str),
            duration_ms, int(success), error_msg,
        ),
    )
    conn.commit()
    conn.close()


def get_traces(report_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM agent_traces WHERE report_id=? ORDER BY created_at ASC", (report_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── 初始化 ──
init_db()
