"""Workspace selection and path validation helpers."""

from __future__ import annotations

from pathlib import Path


class WorkspaceError(ValueError):
    """Raised when a workspace selection cannot be used."""


def resolve_workspace(value: str | Path, *, base: Path) -> Path:
    """Resolve an existing directory from an absolute or base-relative path."""
    raw = str(value).strip()
    if not raw:
        raise WorkspaceError("工作区路径不能为空")
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base) / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise WorkspaceError(f"工作区不存在: {value}") from exc
    except OSError as exc:
        raise WorkspaceError(f"无法访问工作区: {value}") from exc
    if not resolved.is_dir():
        raise WorkspaceError(f"工作区不是目录: {value}")
    return resolved
