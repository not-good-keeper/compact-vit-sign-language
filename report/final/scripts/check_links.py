"""Extract every hyperlink from the report PDF and check that it resolves.

GitHub blob/tree links pinned to the audited commit are checked against the
local git object store (the commit is on origin/main, so a path present in the
commit is present on GitHub); their #L anchors are checked against the file's
line count. All other links (DOI, arXiv, repository root) are requested over
HTTP. Output: report/final/data/link_check.json.

Usage (repository root):
    python report/final/scripts/check_links.py report/final/latex/main.pdf
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PIN = "ea092f3e25124fcd8929929900e53879d7084943"
PREFIXES = (f"https://github.com/not-good-keeper/compact-vit-sign-language/blob/{PIN}/",
            f"https://github.com/not-good-keeper/compact-vit-sign-language/tree/{PIN}/")


def uris(pdf: Path) -> list[str]:
    data = pdf.read_bytes()
    found = re.findall(rb"/URI\s*\((.*?)\)\s*>>", data, re.S)
    out = []
    for raw in found:
        s = raw.replace(rb"\\", b"\\").replace(rb"\(", b"(").replace(rb"\)", b")").decode("latin-1")
        out.append(s)
    return sorted(set(out))


def git_check(url: str) -> dict:
    kind = "tree" if "/tree/" in url else "blob"
    rest = url.split(PIN + "/", 1)[1]
    path, _, anchor = rest.partition("#")
    typ = subprocess.run(["git", "cat-file", "-t", f"{PIN}:{path}"], cwd=ROOT, capture_output=True, text=True)
    ok = typ.returncode == 0 and typ.stdout.strip() == ("tree" if kind == "tree" else "blob")
    res = {"url": url, "kind": "git", "ok": ok}
    if ok and anchor:
        m = re.fullmatch(r"L(\d+)(?:-L(\d+))?", anchor)
        lines = subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", errors="replace").stdout.count("\n")
        hi = int(m.group(2) or m.group(1)) if m else -1
        res["anchor_ok"] = bool(m) and hi <= lines + 1
        res["lines"] = lines
        res["ok"] = res["ok"] and res["anchor_ok"]
    return res


def http_check(url: str) -> dict:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "Mozilla/5.0 (link check)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {"url": url, "kind": "http", "ok": r.status < 400, "status": r.status, "final": r.geturl()}
    except urllib.error.HTTPError as e:
        # Publisher sites often refuse scripted requests (403) after a valid DOI redirect.
        return {"url": url, "kind": "http", "ok": e.code in (403, 418, 429), "status": e.code,
                "final": e.geturl(), "note": "refused scripted access" if e.code in (403, 418, 429) else ""}
    except Exception as e:  # noqa: BLE001
        return {"url": url, "kind": "http", "ok": False, "error": str(e)}


def main() -> None:
    pdf = Path(sys.argv[1])
    results = [git_check(u) if u.startswith(PREFIXES) else http_check(u) for u in uris(pdf)]
    out = ROOT / "report" / "final" / "data" / "link_check.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    bad = [r for r in results if not r["ok"]]
    print(f"{len(results)} unique links; {sum(r['kind'] == 'git' for r in results)} repository links; "
          f"{len(bad)} failing")
    for r in bad:
        print("  FAIL", r)
    for r in results:
        if r.get("note"):
            print("  note", r["status"], r["url"])


if __name__ == "__main__":
    main()
