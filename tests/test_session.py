from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent.session import (
    Session,
    SessionError,
    delete_session,
    latest_session_path,
    list_sessions,
    new_session_path,
    resolve_session,
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

    def test_list_sessions_newest_first_with_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = Session(
                sessions_dir(root) / "old.jsonl",
                session_id="old-session",
                created_at="2026-01-01T10:00:00",
            )
            old.add("user", "第一行\n  第二行")
            new = Session(
                sessions_dir(root) / "new.jsonl",
                session_id="new-session",
                created_at="2026-01-02T10:00:00",
            )
            new.add("user", "最新任务")

            entries = list_sessions(root)

            self.assertEqual([entry.session_id for entry in entries], ["new-session", "old-session"])
            self.assertEqual(entries[1].preview, "第一行 第二行")
            self.assertEqual(entries[1].message_count, 1)

    def test_list_sessions_empty_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(list_sessions(Path(directory)), [])

    def test_resolve_session_by_index_exact_id_and_unique_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = Session(
                sessions_dir(root) / "alpha-one.jsonl",
                session_id="alpha-one",
                created_at="2026-01-01T10:00:00",
            )
            older.save()
            newer = Session(
                sessions_dir(root) / "beta-two.jsonl",
                session_id="beta-two",
                created_at="2026-01-02T10:00:00",
            )
            newer.save()

            self.assertEqual(resolve_session(root, "1").session_id, "beta-two")
            self.assertEqual(resolve_session(root, "alpha-one").path, older.path)
            self.assertEqual(resolve_session(root, "alpha").session_id, "alpha-one")

    def test_resolve_session_rejects_ambiguous_and_missing_selectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, session_id in enumerate(("alpha-one", "alpha-two"), start=1):
                Session(
                    sessions_dir(root) / f"{session_id}.jsonl",
                    session_id=session_id,
                    created_at=f"2026-01-0{index}T10:00:00",
                ).save()
            with self.assertRaisesRegex(SessionError, "前缀不唯一"):
                resolve_session(root, "alpha")
            with self.assertRaisesRegex(SessionError, "未找到"):
                resolve_session(root, "missing")
            with self.assertRaisesRegex(SessionError, "超出范围"):
                resolve_session(root, "3")

    def test_damaged_session_is_listed_rejected_for_resume_and_can_be_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = sessions_dir(root) / "broken.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("not-json\n", encoding="utf-8")

            entry = list_sessions(root)[0]
            self.assertFalse(entry.valid)
            self.assertEqual(entry.session_id, "broken")
            with self.assertRaisesRegex(SessionError, "已损坏"):
                resolve_session(root, "broken")
            self.assertEqual(resolve_session(root, "broken", require_valid=False), entry)

            deleted = delete_session(root, "broken")
            self.assertEqual(deleted.session_id, "broken")
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(SessionError, "未找到"):
                delete_session(root, "broken")


if __name__ == "__main__":
    unittest.main()
