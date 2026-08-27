from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from coding_agent.tools import ToolError, WorkspaceTools, tool_schemas


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

    def test_run_tests_with_cwd_runs_in_subdirectory(self):
        tools = WorkspaceTools(self.root)
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        tests = sub / "tests"
        tests.mkdir()
        (tests / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        result = tools.run_tests("python -m unittest discover -s tests -v", cwd="sub")
        self.assertTrue(result["passed"])
        self.assertEqual(result["cwd"], "sub")

    def test_run_tests_rejects_bad_cwd(self):
        tools = WorkspaceTools(self.root)
        with self.assertRaises(ToolError):
            tools.run_tests("python -m unittest", cwd="missing")
        with self.assertRaises(ValueError):
            tools.run_tests("python -m unittest", cwd="..")

    def test_safe_command_strips_surrounding_quotes(self):
        args = WorkspaceTools._safe_test_command('python -m unittest discover -s tests -p "test_*.py"')
        self.assertEqual(args, ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])

    def test_list_files_uses_forward_slashes(self):
        tools = WorkspaceTools(self.root)
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("x = 1\n", encoding="utf-8")
        files = tools.list_files()["files"]
        self.assertIn("sub/nested.py", files)
        self.assertFalse(any("\\" in name for name in files))

    def test_search_code_uses_forward_slashes(self):
        tools = WorkspaceTools(self.root)
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("target_value = 1\n", encoding="utf-8")
        matches = tools.search_code("target_value")["matches"]
        self.assertTrue(any(match.startswith("sub/nested.py:") for match in matches))

    def test_apply_diff_is_registered_in_schema(self):
        names = [schema["name"] for schema in tool_schemas()]
        self.assertIn("apply_diff", names)
        self.assertEqual(len(names), 10)

    def test_apply_diff_basic_edit(self):
        tools = WorkspaceTools(self.root)
        target = self.root / "greet.py"
        target.write_text('def greet():\n    return "hi"\n', encoding="utf-8")
        result = tools.apply_diff(
            "greet.py",
            '--- a/greet.py\n+++ b/greet.py\n@@ -1,2 +1,2 @@\n def greet():\n-    return "hi"\n+    return "hello"\n',
        )
        self.assertEqual(result["hunks_applied"], 1)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["removed"], 1)
        self.assertIn('return "hello"', target.read_text(encoding="utf-8"))

    def test_apply_diff_multiple_hunks(self):
        tools = WorkspaceTools(self.root)
        target = self.root / "data.txt"
        target.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
        diff = (
            "--- a/data.txt\n"
            "+++ b/data.txt\n"
            "@@ -1,2 +1,2 @@\n"
            "-line1\n"
            "+one\n"
            " line2\n"
            "@@ -3,2 +3,2 @@\n"
            " line3\n"
            "-line4\n"
            "+four\n"
        )
        result = tools.apply_diff("data.txt", diff)
        self.assertEqual(result["hunks_applied"], 2)
        self.assertEqual(result["added"], 2)
        self.assertEqual(result["removed"], 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "one\nline2\nline3\nfour\n")

    def test_apply_diff_insert_only(self):
        tools = WorkspaceTools(self.root)
        target = self.root / "data.txt"
        target.write_text("a\nb\n", encoding="utf-8")
        result = tools.apply_diff("data.txt", "--- a/data.txt\n+++ b/data.txt\n@@ -2,0 +2,1 @@\n+inserted\n")
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["removed"], 0)
        self.assertEqual(target.read_text(encoding="utf-8"), "a\ninserted\nb\n")

    def test_apply_diff_remove_only(self):
        tools = WorkspaceTools(self.root)
        target = self.root / "data.txt"
        target.write_text("a\nb\n", encoding="utf-8")
        result = tools.apply_diff("data.txt", "--- a/data.txt\n+++ b/data.txt\n@@ -2,1 +1,0 @@\n-b\n")
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["removed"], 1)
        self.assertEqual(target.read_text(encoding="utf-8"), "a\n")

    def test_apply_diff_rejects_malformed_diff(self):
        tools = WorkspaceTools(self.root)
        (self.root / "data.txt").write_text("x\n", encoding="utf-8")
        with self.assertRaises(ToolError):
            tools.apply_diff("data.txt", "--- a/data.txt\n+++ b/data.txt\n-x\n+x\n")
        with self.assertRaises(ToolError):
            tools.apply_diff("data.txt", "--- a/data.txt\n+++ b/data.txt\n@@ -1,1 +1,1 @@\n\\ No newline at end of file\n")
        with self.assertRaises(ToolError):
            tools.apply_diff("data.txt", "")

    def test_apply_diff_mismatch_is_atomic(self):
        tools = WorkspaceTools(self.root)
        target = self.root / "data.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        diff = (
            "--- a/data.txt\n+++ b/data.txt\n"
            "@@ -1,2 +1,2 @@\n alpha\n-beta\n+first\n"
            "@@ -2,2 +2,2 @@\n gamma\n-delta\n+second\n"
        )
        with self.assertRaises(ToolError):
            tools.apply_diff("data.txt", diff)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\nbeta\ngamma\n")

    def test_apply_diff_finds_unique_match_when_line_numbers_off(self):
        tools = WorkspaceTools(self.root)
        target = self.root / "data.txt"
        target.write_text("x\ny\nz\n", encoding="utf-8")
        result = tools.apply_diff("data.txt", "--- a/data.txt\n+++ b/data.txt\n@@ -99,2 +99,2 @@\n y\n-z\n+Z\n")
        self.assertEqual(result["hunks_applied"], 1)
        self.assertEqual(target.read_text(encoding="utf-8"), "x\ny\nZ\n")

    def test_apply_diff_ambiguous_match_fails(self):
        tools = WorkspaceTools(self.root)
        (self.root / "data.txt").write_text("a\nb\na\nb\n", encoding="utf-8")
        with self.assertRaises(ToolError):
            tools.apply_diff("data.txt", "--- a/data.txt\n+++ b/data.txt\n@@ -1,1 +1,1 @@\n-b\n")

    def test_apply_diff_rejects_bad_paths(self):
        tools = WorkspaceTools(self.root)
        diff = "--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-x\n+y\n"
        with self.assertRaises(ValueError):
            tools.apply_diff("../out.txt", diff)
        with self.assertRaises(ValueError):
            tools.apply_diff(".env", diff)
        with self.assertRaises(ToolError):
            tools.apply_diff("missing.txt", diff)

    def test_apply_diff_rejects_oversized_result(self):
        tools = WorkspaceTools(self.root)
        target = self.root / "data.txt"
        target.write_text("x\n", encoding="utf-8")
        huge = "y" * (tools.MAX_WRITE_CHARS + 10)
        with self.assertRaises(ToolError):
            tools.apply_diff("data.txt", f"--- a/data.txt\n+++ b/data.txt\n@@ -1,1 +1,1 @@\n-x\n+{huge}\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "x\n")

    def test_apply_diff_normalizes_crlf(self):
        tools = WorkspaceTools(self.root)
        target = self.root / "data.txt"
        target.write_bytes(b"a\r\nb\r\n")
        tools.apply_diff("data.txt", "--- a/data.txt\n+++ b/data.txt\n@@ -2,1 +2,1 @@\n-b\n+B\n")
        self.assertEqual(target.read_bytes(), b"a\nB\n")


if __name__ == "__main__":
    unittest.main()
