#!/usr/bin/env bash
# Build main.pdf with pdflatex + BibTeX (MiKTeX or TeX Live).
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1 || true
bibtex main >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
grep -E "^!|undefined|Overfull" main.log || true
grep -E "Output written" main.log
