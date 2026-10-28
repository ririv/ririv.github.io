"""Validate and mirror a Python simple index into a compatibility path."""

import argparse
import json
import re
import shutil
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href" and value)


def read_links(path: Path) -> list[str]:
    parser = Links()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.hrefs


def collect_index_files(source: Path, wheel_repository: str) -> list[Path]:
    """Reject incomplete indexes and wheel links outside the canonical repository."""
    files = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"索引不允许符号链接：{relative}")
        if path.is_file() and path.suffix == ".html":
            files.append(relative)

    projects = read_links(source / "index.html")
    if not projects:
        raise ValueError("上游索引没有包，拒绝发布空索引")
    expected_prefix = f"https://github.com/{wheel_repository}/raw/wheels/"
    for project in projects:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*/", project):
            raise ValueError(f"包路径不是规范的相对路径：{project}")
        wheels = read_links(source / project / "index.html")
        if not wheels:
            raise ValueError(f"包索引没有 wheel：{project}")
        for wheel in wheels:
            if not wheel.startswith(expected_prefix) or not urlsplit(
                wheel
            ).path.endswith(".whl"):
                raise ValueError(f"wheel 链接未指向指定仓库：{wheel}")
    return files


def sync_index(
    source: Path, destination: Path, wheel_repository: str, revision: str, tree: str
) -> None:
    """Replace only the compatibility subtree after validating the complete source."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"必须提供上游的完整提交 SHA：{revision}")
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ValueError("必须提供上游的完整 Git 树 SHA")
    files = collect_index_files(source, wheel_repository)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wheel-index-", dir=destination.parent
    ) as tmp:
        staged = Path(tmp) / "index"
        staged.mkdir()
        for relative in files:
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        (staged / "sync-revision.txt").write_text(revision + "\n", encoding="utf-8")
        (staged / "sync-state.json").write_text(json.dumps({"tree": tree}) + "\n", encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)
    print(f"已同步 {len(files)} 个索引页面，上游提交：{revision}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--wheel-repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tree", required=True)
    args = parser.parse_args()
    sync_index(args.source, args.destination, args.wheel_repository, args.revision, args.tree)


if __name__ == "__main__":
    main()
