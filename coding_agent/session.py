from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class SessionError(RuntimeError):
    """会话文件无法读取或格式非法。"""


class Session:
    """JSONL 会话持久化：首行 header + 每行一条消息，原子全量重写。

    - 崩溃安全：临时文件 + fsync + os.replace，断电/杀进程不损坏上次快照；
    - 加载容错：header 缺失/非法抛 SessionError，撕裂的尾行直接丢弃。
    """

    def __init__(
        self,
        path: Path,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        created_at: str | None = None,
    ):
        self.path = Path(path)
        self.session_id = session_id or _new_session_id()
        self.cwd = cwd or str(Path.cwd())
        self.created_at = created_at or datetime.now().isoformat(timespec="seconds")
        self.messages: list[dict[str, str]] = []

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def add(self, role: str, content: str) -> None:
        """追加一条消息并立即原子落盘。"""
        if not isinstance(content, str):
            raise TypeError("content 必须是字符串")
        self.messages.append({"role": role, "content": content})
        self.save()

    def save(self) -> None:
        """原子全量重写：临时文件 + fsync + os.replace。失败不破坏上次快照。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "id": self.session_id,
            "created_at": self.created_at,
            "cwd": self.cwd,
            "message_count": self.message_count,
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(header, ensure_ascii=False) + "\n")
                for message in self.messages:
                    f.write(json.dumps(message, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.path)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    @classmethod
    def load(cls, path: Path) -> Session:
        """从 JSONL 恢复会话；header 缺失/非法抛 SessionError，撕裂尾行丢弃。"""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        if not lines:
            raise SessionError(f"会话文件为空: {path}")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise SessionError(f"会话文件 header 非法: {path}") from exc
        if not isinstance(header, dict) or not isinstance(header.get("id"), str) or not isinstance(
            header.get("created_at"), str
        ):
            raise SessionError(f"会话文件 header 缺少 id/created_at: {path}")

        body = lines[1:]
        if body:
            try:
                json.loads(body[-1])
            except json.JSONDecodeError:
                body = body[:-1]  # 崩溃残留的半截尾行，丢弃
        messages: list[dict[str, str]] = []
        for line_no, line in enumerate(body, start=2):
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionError(f"会话文件第 {line_no} 行非法: {exc}") from exc
            if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(
                message.get("content"), str
            ):
                raise SessionError(f"会话文件第 {line_no} 行不是合法消息")
            messages.append({"role": message["role"], "content": message["content"]})

        session = cls(
            path,
            session_id=header["id"],
            cwd=header.get("cwd"),
            created_at=header["created_at"],
        )
        session.messages = messages
        return session


def sessions_dir(workspace: Path) -> Path:
    """会话文件目录：<workspace>/.coding_agent/sessions。"""
    return Path(workspace) / ".coding_agent" / "sessions"


def results_dir(workspace: Path) -> Path:
    """L3 大结果落盘目录：<workspace>/.coding_agent/results。"""
    return Path(workspace) / ".coding_agent" / "results"


def _new_session_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"


def new_session_path(workspace: Path) -> Path:
    """生成新会话文件路径（文件名带微秒时间戳，按名排序即时间序）。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return sessions_dir(workspace) / f"{stamp}-{uuid4().hex[:8]}.jsonl"


def new_session(workspace: Path) -> Session:
    """创建新会话：会话 id 与文件名保持一致（/status 展示更清晰）。"""
    session_id = _new_session_id()
    path = sessions_dir(workspace) / f"{session_id}.jsonl"
    return Session(path, session_id=session_id)


def latest_session_path(workspace: Path) -> Path | None:
    """最近一次会话文件；无任何会话返回 None。"""
    directory = sessions_dir(workspace)
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.jsonl"))
    return files[-1] if files else None
