"""Check a blog post's Project tab against the library.

Every ```python block in the tab must appear, word for word, in one file of
neural_network/ (only the indent may differ). A line of just ... skips lines,
so a block can show a class line and then a method further down. The tab
explains the library only, so test code is not allowed.

    uv run python tools/check_article.py <post index.mdx>
"""

import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def project_tab(article: Path) -> str:
    body = article.read_text(encoding="utf-8")
    start = body.find("<ProjectVersion")
    if start < 0:
        sys.exit(f"No <ProjectVersion> in {article}")
    return body[start:body.index("</ProjectVersion>", start)]


def source_files() -> list[list[str]]:
    """Each .py file of the library, as its lines."""
    files = sorted((REPO / "neural_network").rglob("*.py"))
    return [[line.rstrip() for line in f.read_text(encoding="utf-8").splitlines()] for f in files]


def find_run(piece: list[str], lines: list[str], start: int) -> int:
    """Where the piece starts as a run of lines at or after start, ignoring a shared indent; -1 if nowhere."""
    want = textwrap.dedent("\n".join(piece)).splitlines()
    for at in range(start, len(lines) - len(want) + 1):
        window = textwrap.dedent("\n".join(lines[at:at + len(want)]))
        if window.splitlines() == want:
            return at
    return -1


def in_file(snippet: str, lines: list[str]) -> bool:
    pieces, piece = [], []
    for line in snippet.rstrip().splitlines():
        if line.strip() == "...":
            pieces.append(piece)
            piece = []
        else:
            piece.append(line)
    pieces.append(piece)
    start = 0
    for piece in filter(None, pieces):
        at = find_run(piece, lines, start)
        if at < 0:
            return False
        start = at + len(piece)
    return True


def main() -> None:
    tab = project_tab(Path(sys.argv[1]))
    blocks = re.findall(r"```python\n(.*?)```", tab, re.S)
    files = source_files()
    failures = 0
    for code in blocks:
        ok = any(in_file(code, lines) for lines in files)
        failures += not ok
        print(f"{'OK  ' if ok else 'MISS'}  {code.strip().splitlines()[0][:70]}")
    print(f"\n{len(blocks)} code blocks checked, {failures} missing")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
