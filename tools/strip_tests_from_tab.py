"""Remove every mention of tests from a post's Project tab; the tests stay in the repo.

    uv run python tools/strip_tests_from_tab.py <post index.mdx> [...]
"""

import re
import sys
from pathlib import Path

TESTS_ROW = re.compile(r"^\| `tests/` \|[^\n]*\n", re.M)
TESTS_SECTION = re.compile(r"^## The note['’]s numbers, checked\n.*\Z", re.M | re.S)
TESTED_PHRASE = re.compile(r"be read, and tested, on its own")
PYTEST_TAG = re.compile(r",\s*'pytest'")


def strip(tab: str) -> str:
    tab = TESTS_ROW.sub("", tab)
    tab = TESTS_SECTION.sub("", tab)
    tab = TESTED_PHRASE.sub("be read on its own", tab)
    return PYTEST_TAG.sub("", tab)


def main() -> None:
    for name in sys.argv[1:]:
        path = Path(name)
        raw = path.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        start = text.index("<ProjectVersion")
        end = text.index("</ProjectVersion>", start)
        stripped = text[:start] + strip(text[start:end]) + text[end:]
        if crlf:
            stripped = stripped.replace("\n", "\r\n")
        path.write_bytes(stripped.encode("utf-8"))
        left = re.findall(r"pytest|`tests/`|numbers, checked", strip(text[start:end]))
        print(f"{path.parent.name}: {len(text) - len(stripped.replace(chr(13), ''))} chars removed, "
              f"mentions left: {left or 'none'}")


if __name__ == "__main__":
    main()
