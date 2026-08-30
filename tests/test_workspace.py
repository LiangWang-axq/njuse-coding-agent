from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent.workspace import WorkspaceError, resolve_workspace


class WorkspaceResolutionTests(unittest.TestCase):
    def test_resolves_relative_and_absolute_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            child.mkdir()
            self.assertEqual(resolve_workspace("child", base=root), child.resolve())
            self.assertEqual(resolve_workspace(child, base=root), child.resolve())

    def test_rejects_missing_path_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "file.txt"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(WorkspaceError, "工作区不存在"):
                resolve_workspace("missing", base=root)
            with self.assertRaisesRegex(WorkspaceError, "不是目录"):
                resolve_workspace(file_path, base=root)

    def test_accepts_quoted_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "project folder"
            child.mkdir()
            self.assertEqual(resolve_workspace(f'"{child}"', base=root), child.resolve())


if __name__ == "__main__":
    unittest.main()
