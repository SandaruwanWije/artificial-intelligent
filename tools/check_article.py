"""Check a blog post's Project tab against the project it shows.

Every ```python block in the tab must appear in the project's .py files
(only the indent may differ), and every ```text block must appear in what
`python -m <project>` prints.

    uv run python tools/check_article.py <post index.mdx> <project folder>
"""

import re
import subprocess
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


def source_lines(project: Path) -> list[str]:
    files = sorted(project.rglob("*.py"))
    return [line.rstrip() for f in files for line in f.read_text(encoding="utf-8").splitlines()]


def in_sources(snippet: str, lines: list[str]) -> bool:
    """True when the snippet is a run of lines in the sources, ignoring a shared indent."""
    want = textwrap.dedent(snippet).rstrip().splitlines()
    for start in range(len(lines) - len(want) + 1):
        window = textwrap.dedent("\n".join(lines[start:start + len(want)]))
        if window.splitlines() == want:
            return True
    return False


def main() -> None:
    article, project = Path(sys.argv[1]), REPO / sys.argv[2]
    blocks = re.findall(r"```(\w+)\n(.*?)```", project_tab(article), re.S)
    lines = source_lines(project)
    printed = subprocess.run([sys.executable, "-m", project.name], cwd=REPO, check=True,
                             capture_output=True, text=True, encoding="utf-8").stdout

    failures = 0
    for lang, code in blocks:
        if lang == "python":
            ok = in_sources(code, lines)
        elif lang == "text":
            ok = code.rstrip() in printed
        else:
            continue
        failures += not ok
        print(f"{'OK  ' if ok else 'MISS'}  {lang:6}  {code.strip().splitlines()[0][:70]}")
    print(f"\n{len(blocks)} blocks checked, {failures} missing")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
