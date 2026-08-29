from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent.session import (
    Session,
    SessionError,
    latest_session_path,
    new_session_path,
    sessions_dir,
)


class SessionTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = new_session_path(root)
            session = Session(path, session_id="session-1", cwd=str(root))
            session.add("user", "任务")
            session.add("assistant", "完成")
            loaded = Session.load(path)
            self.assertEqual(loaded.session_id, "session-1")
            self.assertEqual(loaded.cwd, str(root))
            self.assertEqual(
                loaded.messages,
                [
                    {"role": "user", "content": "任务"},
                    {"role": "assistant", "content": "完成"},
                ],
            )

    def test_save_leaves_no_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = new_session_path(Path(directory))
            session = Session(path)
            session.add("user", "任务")
            leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".")]
            self.assertEqual(leftovers, [])
            self.assertTrue(path.is_file())

    def test_load_drops_torn_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = new_session_path(Path(directory))
            session = Session(path)
            session.add("user", "任务")
            with open(path, "a", encoding="utf-8") as f:
                f.write('{"role":"assistant","content":"半截')
            loaded = Session.load(path)
            self.assertEqual(loaded.messages, [{"role": "user", "content": "任务"}])

    def test_load_rejects_missing_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"role":"user","content":"x"}\n', encoding="utf-8")
            with self.assertRaises(SessionError):
                Session.load(path)

    def test_load_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(SessionError):
                Session.load(path)

    def test_latest_session_path_picks_newest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions_dir(root).mkdir(parents=True, exist_ok=True)
            old = new_session_path(root)
            old.write_text('{"id":"old","created_at":"2026-01-01"}\n', encoding="utf-8")
            new = new_session_path(root)
            new.write_text('{"id":"new","created_at":"2026-01-02"}\n', encoding="utf-8")
            self.assertEqual(latest_session_path(root), new)

    def test_latest_session_path_none_when_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(latest_session_path(Path(directory)))

    def test_sessions_dir_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(sessions_dir(Path(directory)), Path(directory) / ".coding_agent" / "sessions")


if __name__ == "__main__":
    unittest.main()
