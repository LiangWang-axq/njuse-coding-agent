from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


SESSION_PREVIEW_CHARS = 48


class SessionError(RuntimeError):
    """会话文件无法读取或格式非法。"""


@dataclass(frozen=True)
class SessionInfo:
    """会话列表中的一条记录；损坏文件也会保留以便删除。"""

    path: Path
    session_id: str
    created_at: str | None
    cwd: str | None
    message_count: int | None
    preview: str
    valid: bool
    error: str | None = None


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


def list_sessions(workspace: Path) -> list[SessionInfo]:
    """扫描工作区会话，按创建时间最新优先返回，损坏文件不会被隐藏。"""
    directory = sessions_dir(workspace)
    if not directory.is_dir():
        return []
    entries = [_session_info(path) for path in directory.glob("*.jsonl") if path.is_file()]
    return sorted(entries, key=_session_sort_key, reverse=True)


def resolve_session(
    workspace: Path,
    selector: str,
    *,
    require_valid: bool = True,
) -> SessionInfo:
    """按最新优先序号、完整 ID 或唯一 ID 前缀选择会话。"""
    selector = selector.strip()
    if not selector:
        raise SessionError("会话选择器不能为空")
    entries = list_sessions(workspace)
    if selector.isdigit():
        index = int(selector)
        if index < 1 or index > len(entries):
            raise SessionError(f"会话序号超出范围: {selector}")
        selected = entries[index - 1]
    else:
        exact = [
            entry
            for entry in entries
            if selector in {entry.session_id, entry.path.stem}
        ]
        if len(exact) == 1:
            selected = exact[0]
        elif len(exact) > 1:
            raise SessionError(f"会话 ID 不唯一: {selector}")
        else:
            matches = [
                entry
                for entry in entries
                if entry.session_id.startswith(selector) or entry.path.stem.startswith(selector)
            ]
            if not matches:
                raise SessionError(f"未找到会话: {selector}")
            if len(matches) > 1:
                ids = "、".join(entry.session_id for entry in matches[:3])
                raise SessionError(f"会话前缀不唯一: {selector}（匹配 {ids}）")
            selected = matches[0]
    if require_valid and not selected.valid:
        raise SessionError(f"会话已损坏，无法恢复: {selected.session_id}（{selected.error}）")
    return selected


def delete_session(workspace: Path, selector: str) -> SessionInfo:
    """永久删除选中的会话 JSONL 文件并返回被删除的记录。"""
    selected = resolve_session(workspace, selector, require_valid=False)
    directory = sessions_dir(workspace).resolve()
    if selected.path.parent.resolve() != directory or selected.path.suffix.lower() != ".jsonl":
        raise SessionError("拒绝删除会话目录之外的文件")
    try:
        selected.path.unlink()
    except FileNotFoundError as exc:
        raise SessionError(f"会话文件已不存在: {selected.session_id}") from exc
    except OSError as exc:
        raise SessionError(f"删除会话失败: {selected.session_id}（{exc}）") from exc
    return selected


def _session_info(path: Path) -> SessionInfo:
    try:
        session = Session.load(path)
    except (OSError, SessionError) as exc:
        return SessionInfo(
            path=path,
            session_id=path.stem,
            created_at=None,
            cwd=None,
            message_count=None,
            preview="无法读取会话内容",
            valid=False,
            error=str(exc),
        )
    preview = next(
        (_message_preview(message["content"]) for message in session.messages if message["role"] == "user"),
        "(无用户消息)",
    )
    return SessionInfo(
        path=path,
        session_id=session.session_id,
        created_at=session.created_at,
        cwd=session.cwd,
        message_count=session.message_count,
        preview=preview,
        valid=True,
    )


def _message_preview(content: str) -> str:
    text = " ".join(content.split())
    if not text:
        return "(空消息)"
    if len(text) <= SESSION_PREVIEW_CHARS:
        return text
    return text[:SESSION_PREVIEW_CHARS] + "…"


def _session_sort_key(info: SessionInfo) -> tuple[float, str]:
    if info.created_at:
        try:
            created = datetime.fromisoformat(info.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return created.timestamp(), info.path.name
        except ValueError:
            pass
    try:
        return info.path.stat().st_mtime, info.path.name
    except OSError:
        return 0.0, info.path.name
