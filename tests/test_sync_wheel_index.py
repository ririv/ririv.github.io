"""Exercise compatibility publication and invalid upstream boundaries."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sync_wheel_index import sync_index


class SyncIndexTests(unittest.TestCase):
    def prepare(self, root):
        source = root / "source"
        (source / "example").mkdir(parents=True)
        (source / "index.html").write_text('<a href="example/">example</a>')
        (source / "example" / "index.html").write_text(
            '<a href="https://github.com/ririv/py-extra-wheels/raw/wheels/example/example-1-py3-none-any.whl">wheel</a>'
        )
        destination = root / "site" / "android-wheels"
        destination.mkdir(parents=True)
        (destination / "stale.html").write_text("stale")
        (destination.parent / "README.md").write_text("现有首页")
        return source, destination

    def test_replaces_only_compatibility_subtree(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = self.prepare(Path(tmp))
            sync_index(source, destination, "ririv/py-extra-wheels", "a" * 40, "b" * 40)
            self.assertFalse((destination / "stale.html").exists())
            self.assertEqual((destination.parent / "README.md").read_text(), "现有首页")
            self.assertEqual(
                (destination / "index.html").read_bytes(),
                (source / "index.html").read_bytes(),
            )
            self.assertEqual(
                (destination / "sync-revision.txt").read_text(), "a" * 40 + "\n"
            )
            self.assertEqual(json.loads((destination / "sync-state.json").read_text()), {"tree": "b" * 40})
            sync_index(source, destination, "ririv/py-extra-wheels", "a" * 40, "b" * 40)
            self.assertEqual(len(list(destination.rglob("*.html"))), 2)

    def test_rejects_incomplete_index_before_replacing_published_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = self.prepare(Path(tmp))
            (source / "example" / "index.html").unlink()
            with self.assertRaises(FileNotFoundError):
                sync_index(source, destination, "ririv/py-extra-wheels", "a" * 40, "b" * 40)
            self.assertEqual((destination / "stale.html").read_text(), "stale")

    def test_rejects_wrong_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = self.prepare(Path(tmp))
            (source / "example" / "index.html").write_text(
                '<a href="https://example.com/wheel.whl">wheel</a>'
            )
            with self.assertRaises(ValueError):
                sync_index(source, destination, "ririv/py-extra-wheels", "a" * 40, "b" * 40)
            self.assertTrue((destination / "stale.html").is_file())

    def test_rejects_empty_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = self.prepare(Path(tmp))
            (source / "index.html").write_text("没有包")
            with self.assertRaises(ValueError):
                sync_index(source, destination, "ririv/py-extra-wheels", "a" * 40, "b" * 40)
            self.assertTrue((destination / "stale.html").is_file())

    def test_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = self.prepare(Path(tmp))
            (source / "external.html").symlink_to(destination / "stale.html")
            with self.assertRaises(ValueError):
                sync_index(source, destination, "ririv/py-extra-wheels", "a" * 40, "b" * 40)
            self.assertTrue((destination / "stale.html").is_file())


if __name__ == "__main__":
    unittest.main()
