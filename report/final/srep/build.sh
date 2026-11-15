#!/usr/bin/env bash
# Build main.pdf and supp.pdf; each reads the other's .aux for cross-references.
set -uo pipefail
cd "$(dirname "$0")"
: > build.log
for d in main supp; do pdflatex -interaction=nonstopmode $d.tex >> build.log 2>&1; done
for d in main supp; do bibtex $d >> build.log 2>&1; done
for i in 1 2; do for d in main supp; do pdflatex -interaction=nonstopmode $d.tex >> build.log 2>&1; done; done
grep -E "^!|undefined|Overfull" main.log supp.log || true
grep -E "Output written" main.log supp.log
