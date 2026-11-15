"""Build the Scientific Reports version of the paper from report/final/latex/main.tex.

Writes report/final/srep/main.tex (manuscript in the supervisor's order: Introduction,
Related work, Proposed methodology, Results and discussion, Conclusion and future work;
at most 8 display items) and report/final/srep/supp.tex (Supplementary
Information with every other figure and table, copied verbatim so no diagram is
rescaled). The two documents cross-reference each other through xr-hyper.

    python report/final/scripts/make_srep_version.py
    bash report/final/srep/build.sh
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "latex"
OUT = ROOT / "srep"
B = "\\"

# Display items that move to the Supplementary Information, in SI order.
SI_ITEMS = ["tab:lit", "fig:exdata", "fig:excrops", "fig:model", "tab:setup",
            "fig:variants", "fig:ablation", "fig:perclass",
            "fig:gan", "fig:exgan", "tab:gan", "fig:exdetect", "tab:contrib"]
SI_LABELS = SI_ITEMS + ["tab:trace", "tab:borrow", "tab:iter", "app:trace", "app:design"]
ENVS = ("widefigure", "widetable", "figure*", "table*", "figure", "table")


def once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    assert n == 1, f"expected one match, found {n}: {old[:70]!r}"
    return text.replace(old, new)


def cut_block(text: str, label: str) -> tuple[str, str]:
    """Remove the float environment that carries \\label{label}; return (text, block)."""
    pos = text.index(B + "label{" + label + "}")
    best = None
    for env in ENVS:
        start = text.rfind(B + "begin{" + env + "}", 0, pos)
        if start >= 0 and (best is None or start > best[0]):
            best = (start, env)
    start, env = best
    end = text.index(B + "end{" + env + "}", pos) + len(B + "end{" + env + "}")
    while end < len(text) and text[end] == "\n":
        end += 1
    return text[:start] + text[end:], text[start:end].rstrip("\n") + "\n"


def section(text: str, head: str, nxt: str) -> str:
    a = text.index(head)
    b = text.index(nxt, a + len(head))
    return text[a:b]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name in ("sn-jnl.cls", "sn-mathphys-num.bst", "references.bib"):
        shutil.copy(SRC / name, OUT / name)
    if (OUT / "figures").exists():
        shutil.rmtree(OUT / "figures")
    shutil.copytree(SRC / "figures", OUT / "figures")

    tex = (SRC / "main.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
    preamble = tex[:tex.index(B + "begin{document}")]
    # Table footnotes become legend text (the journal does not take footnotes).
    doc = tex.index(B + "begin{document}")
    tex = tex[:doc] + tex[doc:].replace(B + "footnotetext{", B + "src{")
    for cmd in ("footnotetext{", "footnote{", "footnotemark"):
        assert B + cmd not in tex[doc:], cmd

    # Six keywords at most.
    tex = once(tex, "spatio-temporal modelling, ", "")

    # ---- move display items and appendices to the SI ----
    si_blocks = {}
    for label in SI_ITEMS:
        tex, si_blocks[label] = cut_block(tex, label)
    a = tex.index(B + "begin{appendices}")
    b = tex.index(B + "end{appendices}") + len(B + "end{appendices}")
    appendices = tex[a + len(B + "begin{appendices}"):b - len(B + "end{appendices}")]
    tex = tex[:a] + tex[b:]

    # The pipeline figure opens the methodology, where it is first cited, so the
    # main figures are numbered in the order the text cites them.
    tex, f_pipe = cut_block(tex, "fig:archpipe")
    tex = once(tex, B + "section{Methodology}" + B + "label{sec:method}\n",
               B + "section{Proposed methodology}" + B + "label{sec:method}\n"
               "Figure~" + B + "ref{fig:archpipe} shows the end-to-end pipeline. For each clip, 32 frames are sampled, "
               "a landmark detector locates the hands and face, six 16-frame views are drawn, and the 4-bit classifier "
               "scores every view; pretraining, supervised training and quantisation-aware training run once, offline, "
               "before deployment. Sections~" + B + "ref{sec:method:data} to~" + B + "ref{sec:method:metrics} "
               "describe these stages in turn.\n\n" + f_pipe + "\n")

    # ---- citations: every display item is cited, main items in order ----
    tex = once(tex, "all from the original 1080p sources.",
               "all from the original 1080p sources (Supplementary Fig.~" + B + "ref{fig:exdata}).")
    tex = once(tex, "One code path builds the training cache and the test cache (",
               "One code path builds the training cache and the test cache (Supplementary Fig.~" + B + "ref{fig:excrops}; ")
    tex = once(tex, "Figure~" + B + "ref{fig:model} shows the network (",
               "Supplementary Fig.~" + B + "ref{fig:model} and Fig.~" + B + "ref{fig:archmodel} show the network (")
    tex = once(tex, "Figure~" + B + "ref{fig:gan} shows the comparison design.",
               "Supplementary Fig.~" + B + "ref{fig:gan} shows the comparison design, and Supplementary Fig.~"
               + B + "ref{fig:exgan} shows what each front end does to the hand region of a 180p clip.")
    tex = once(tex, "Going from 8 to 16 frames raised single-view top-1",
               "Going from 8 to 16 frames (Supplementary Fig.~" + B + "ref{fig:ablation}a) raised single-view top-1")
    tex = once(tex, "Crop size is set at 64 pixels:",
               "Crop size is set at 64 pixels (Supplementary Fig.~" + B + "ref{fig:ablation}b):")
    tex = once(tex, "Twenty-seven of the 50 words reached at least 90" + B + "% recall.",
               "Twenty-seven of the 50 words reached at least 90" + B + "% recall (Supplementary Fig.~" + B + "ref{fig:perclass}).")
    tex = once(tex, "Every learned upscaler beat bicubic at every resolution;",
               "Every learned upscaler beat bicubic at every resolution (Fig.~" + B + "ref{fig:ganres});")
    tex = once(tex, "for SR-L1).", "for SR-L1; Supplementary Fig.~" + B + "ref{fig:exdetect}).")
    tex = once(tex, "The matched 50-word comparison gave",
               "The matched 50-word comparison (Fig.~" + B + "ref{fig:protocol}a) gave")
    tex = once(tex, "With the early 8-frame recipe on 262 words, accuracy fell",
               "With the early 8-frame recipe on 262 words (Fig.~" + B + "ref{fig:protocol}b), accuracy fell")

    # References to items now in the SI.
    for label in SI_LABELS:
        r = B + "ref{" + label + "}"
        for word in ("Table~", "Tab.~"):
            tex = tex.replace(word + r, "Supplementary Table~" + r)
        for word in ("Figure~", "Fig.~"):
            tex = tex.replace("Supplementary " + word + r, "@@" + r)  # already converted
            tex = tex.replace(word + r, "Supplementary Fig.~" + r)
            tex = tex.replace("@@" + r, "Supplementary Fig.~" + r)
        tex = tex.replace("Appendix~" + r, "Supplementary Section~" + r)
    tex = tex.replace("Supplementary Supplementary", "Supplementary")

    # ---- section order requested by the supervisor ----
    # 1 Introduction, 2 Related work, 3 Proposed methodology (data, technical
    # details, experimental setup), 4 Results and discussion, 5 Conclusion and
    # future work. The journal suggests but does not require IMRaD order.
    tex = once(tex, B + "section{Experimental setup}", B + "subsection{Experimental setup}")
    tex = once(tex, B + "section{Results}" + B + "label{sec:results}", B + "section{Results and discussion}" + B + "label{sec:results}")
    tex = once(tex, B + "section{Discussion and originality}", B + "subsection{Discussion and originality}")

    # Declarations the journal requires.
    tail = tex[tex.index(B + "backmatter"):]
    tail = once(tail, B + "bmhead{Use of AI assistance}",
                B + "bmhead{Author contributions}\n"
                "[To be completed by the authors: the contribution of each author.]\n\n"
                + B + "bmhead{Competing interests}\nThe authors declare no competing interests.\n\n"
                + B + "bmhead{Funding}\nThis work received no external funding.\n\n"
                + B + "bmhead{Use of AI assistance}")

    xr = "\n" + B + "usepackage{xr-hyper}\n"
    main_tex = preamble + xr + B + "externaldocument{supp}\n\n" + tex[tex.index(B + "begin{document}"):tex.index(B + "backmatter")] + tail
    (OUT / "main.tex").write_bytes(main_tex.replace("\n", "\r\n").encode("utf-8"))

    # ---- Supplementary Information ----
    title_block = tex[tex.index(B + "begin{document}"):tex.index(B + "abstract{")]
    title_block = re.sub(r"\\title\[([^\]]*)\]\{([^}]*)\}",
                         lambda m: B + "title[" + m.group(1) + "]{Supplementary Information for ``" + m.group(2) + "''}",
                         title_block, count=1)
    appendices = appendices.replace("this appendix adds no new measurement", "this section adds no new measurement")
    for label in SI_LABELS:
        appendices = appendices.replace("Appendix~" + B + "ref{" + label + "}", "Section~" + B + "ref{" + label + "}")
    S = B + "section"
    def items(*labels: str) -> str:
        return "\n".join(si_blocks[l] for l in labels) + "\n"
    # One section introduces every supplementary figure and table; the floats
    # follow and pack together. Separate one-sentence sections per group left
    # their headings stacked on one page, away from their figures.
    def entry(head: str, text: str) -> str:
        return B + "item " + B + "textit{" + head + "} " + text + "\n"
    supp_body = (
        S + "{Supplementary figures and tables}" + B + "label{supp:items}\n"
        "Each supplementary figure and table extends a section of the main text, as follows.\n"
        + B + "begin{itemize}" + B + "setlength{" + B + "itemsep}{1pt}" + B + "setlength{" + B + "parskip}{0pt}\n"
        + entry("Closest prior work.", "Supplementary Table~" + B + "ref{tab:lit} places the closest works beside this project; "
                "numbers are as reported by each paper (main text, Section~" + B + "ref{sec:related}).")
        + entry("Data and region extraction.", "Supplementary Fig.~" + B + "ref{fig:exdata} shows frames from INCLUDE and CISLR, and "
                "Supplementary Fig.~" + B + "ref{fig:excrops} shows the hand and face crops that region extraction produces "
                "(Sections~" + B + "ref{sec:method:data} and~" + B + "ref{sec:method:pre}).")
        + entry("Model overview.", "Supplementary Fig.~" + B + "ref{fig:model} summarises the network of Section~"
                + B + "ref{sec:method:model} in one diagram; Fig.~" + B + "ref{fig:archmodel} of the main text gives the detailed architecture.")
        + entry("Experimental setup.", "Supplementary Table~" + B + "ref{tab:setup} lists, for each principal experiment, the models "
                "compared, the test set, the number of seeds and the metric (Section~" + B + "ref{sec:setup}).")
        + entry("Additional results.", "Supplementary Figs.~" + B + "ref{fig:variants} to~" + B + "ref{fig:perclass} show the variant "
                "comparison, the temporal-length and crop-size ablations, and per-word recall (Sections~" + B + "ref{sec:res:variants}, "
                + B + "ref{sec:res:ablation} and~" + B + "ref{sec:res:perclass}).")
        + entry("Super-resolution study.", "Supplementary Fig.~" + B + "ref{fig:gan} shows the design of the study, Supplementary Figs.~"
                + B + "ref{fig:exgan} and~" + B + "ref{fig:exdetect} show what each front end does to a 180p clip, and Supplementary "
                "Table~" + B + "ref{tab:gan} gives the results of every arm (Sections~" + B + "ref{sec:method:sr} and~"
                + B + "ref{sec:res:gan}).")
        + entry("Claimed contributions.", "Supplementary Table~" + B + "ref{tab:contrib} ties each claim of Section~"
                + B + "ref{sec:discussion} to its evidence.")
        + B + "end{itemize}\n\n"
        + items("tab:lit", "fig:exdata", "fig:excrops", "fig:model", "tab:setup", "fig:variants", "fig:ablation",
                "fig:perclass", "fig:gan", "fig:exgan", "tab:gan", "fig:exdetect", "tab:contrib")
        + "\n"
        + appendices.strip("\n") + "\n\n")
    # The supplement is set in one column and is mostly large figures with a
    # sentence or two of text each, so floats are free to pack together
    # (several per page, at the top, bottom or on float pages) rather than being
    # held behind section breaks, which left half-empty pages. Only the
    # placement options change; float contents and sizes are untouched.
    supp_body = re.sub(r"\\begin\{(figure\*|table\*|widefigure|widetable)\}(\[[a-z!]*\])?",
                       lambda m: B + "begin{" + m.group(1) + "}[!htbp]", supp_body)
    supp_body = supp_body.replace(B + "clearpage\n\n" + B + "section{Experiment traceability}", B + "section{Experiment traceability}")
    one_col = once(preamble, "[pdflatex,sn-mathphys-num,twocolumn]{sn-jnl}", "[pdflatex,sn-mathphys-num,onecolumn]{sn-jnl}")
    one_col = once(one_col, "footskip=8mm,twocolumn,columnsep=6.3mm}", "footskip=8mm,onecolumn}")
    one_col = once(one_col, B + "usepackage{stfloats}%", B + "usepackage{stfloats}%\n" + B + "usepackage{placeins}%")
    one_col = once(one_col, B + "renewcommand" + B + "textfraction{0.1}",
                   B + "renewcommand" + B + "textfraction{0.05}\n" + B + "renewcommand" + B + "topfraction{0.95}\n"
                   + B + "renewcommand" + B + "bottomfraction{0.9}\n" + B + "renewcommand" + B + "floatpagefraction{0.85}\n"
                   + B + "setcounter{topnumber}{3}\n" + B + "setcounter{bottomnumber}{2}\n" + B + "setcounter{totalnumber}{5}\n"
                   # Float-only pages start at the top instead of being centred vertically.
                   + B + "makeatletter\n" + B + "setlength" + B + "@fptop{0pt}\n" + B + "setlength" + B + "@fpsep{18pt}\n"
                   + B + "setlength" + B + "@fpbot{0pt plus 1fil}\n" + B + "makeatother")
    supp_pre = (one_col + xr + B + "externaldocument{main}\n"
                + B + "renewcommand{" + B + "thefigure}{S" + B + "arabic{figure}}\n"
                + B + "renewcommand{" + B + "thetable}{S" + B + "arabic{table}}\n"
                + B + "renewcommand{" + B + "thesection}{S" + B + "arabic{section}}\n\n")
    supp_tex = (supp_pre + title_block + B + "maketitle\n\n" + supp_body
                + B + "bibliography{references}\n\n" + B + "end{document}\n")
    (OUT / "supp.tex").write_bytes(supp_tex.replace("\n", "\r\n").encode("utf-8"))
    (OUT / "build.sh").write_text(
        "#!/usr/bin/env bash\n# Build main.pdf and supp.pdf; each reads the other's .aux for cross-references.\n"
        "set -uo pipefail\ncd \"$(dirname \"$0\")\"\n: > build.log\n"
        "for d in main supp; do pdflatex -interaction=nonstopmode $d.tex >> build.log 2>&1; done\n"
        "for d in main supp; do bibtex $d >> build.log 2>&1; done\n"
        "for i in 1 2; do for d in main supp; do pdflatex -interaction=nonstopmode $d.tex >> build.log 2>&1; done; done\n"
        "grep -E \"^!|undefined|Overfull\" main.log supp.log || true\n"
        "grep -E \"Output written\" main.log supp.log\n", encoding="utf-8", newline="\n")
    print("wrote", OUT / "main.tex", "and", OUT / "supp.tex")


if __name__ == "__main__":
    main()
