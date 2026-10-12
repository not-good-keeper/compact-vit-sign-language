"""Assemble islvit/ui/index.html from islvit/ui/src/.

The page ships as one self-contained file -- no external requests, so it works
offline in a review room and can be served as a single response -- but it is
edited as one file per screen, so each screen can change without touching the
others.

    python islvit/ui/build.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
SCREENS = ("capture", "notsure", "sentence", "settings", "diagnostics", "vocabulary", "enrolment")


def read(path: Path) -> str:
    if not path.exists():
        print(f"  missing {path.relative_to(HERE)} -- left empty")
        return ""
    return path.read_text(encoding="utf-8")


def main() -> None:
    page = read(SRC / "shell.html")
    css = [read(SRC / "base.css")]
    js = [read(SRC / "core.js")]
    for name in SCREENS:
        folder = SRC / "screens"
        page = page.replace(f"<!-- @screen {name} -->", read(folder / f"{name}.html"))
        css.append(f"\n/* ===================================================== {name} */\n" + read(folder / f"{name}.css"))
        js.append(f"\n/* ===================================================== {name} */\n" + read(folder / f"{name}.js"))

    script = "\n".join(js) + "\nApp.boot();\n"
    if "</script" in script.lower():
        raise SystemExit("a script contains '</script' and would end the inline <script> early")
    page = page.replace("/* @css */", "\n".join(css)).replace("/* @js */", script)
    out = HERE / "index.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out.relative_to(HERE.parent.parent)} ({len(page.encode('utf-8')) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
