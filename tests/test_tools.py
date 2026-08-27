from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from coding_agent.tools import ToolError, WorkspaceTools


class NewToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "keep.txt").write_text("hello", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_delete_file(self):
        tools = WorkspaceTools(self.root)
        (self.root / "junk.txt").write_text("x", encoding="utf-8")
        result = tools.delete_file("junk.txt")
        self.assertTrue(result["deleted"])
        self.assertFalse((self.root / "junk.txt").exists())

    def test_delete_rejects_env_and_directory(self):
        tools = WorkspaceTools(self.root)
        with self.assertRaises(ValueError):
            tools.delete_file(".env")
        with self.assertRaises(ToolError):
            tools.delete_file(".")

    def test_move_file_and_reject_overwrite(self):
        tools = WorkspaceTools(self.root)
        (self.root / "sub").mkdir()
        result = tools.move_file("keep.txt", "sub/renamed.txt")
        self.assertTrue(result["moved"])
        self.assertFalse((self.root / "keep.txt").exists())
        self.assertTrue((self.root / "sub" / "renamed.txt").is_file())
        with self.assertRaises(ToolError):
            tools.move_file("sub/renamed.txt", "sub/renamed.txt")

    def test_move_rejects_escape_and_env(self):
        tools = WorkspaceTools(self.root)
        with self.assertRaises(ValueError):
            tools.move_file("keep.txt", "../out.txt")
        with self.assertRaises(ValueError):
            tools.move_file(".env", "x.txt")

    def test_git_status_reads_repo(self):
        tools = WorkspaceTools(self.root)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        result = tools.git_status()
        self.assertIn("status", result)
        self.assertIn("recent_commits", result)

    def test_git_status_rejects_non_repo(self):
        tools = WorkspaceTools(self.root)
        with self.assertRaises(ToolError):
            tools.git_status()


if __name__ == "__main__":
    unittest.main()
