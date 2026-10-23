# Validation checklist

What was checked before delivery, how, and the result. Items marked "not done" were not checked and should not be assumed to pass.

| Check | How it was checked | Result |
|---|---|---|
| Headline numbers recomputed | `scripts/audit_results.py` reads every result file named in the report; output in `data/audit_summary.txt` | Pass: all values in Tables 3–6 and Figures 5–9 match the files |
| Headline reproduced from the release files | `islvit.mask50` on `release/*.pt` with the clean test cache (`audit/rescore_release_s*.json`) | Pass: per-clip outcomes identical for all three seeds |
| Paired statistics | Independent exact McNemar implementation vs the values in earlier documents and `islvit/gan_report.py` verdicts | Pass: all 30 GAN pairs and the landmark/QAT pairs agree |
| Test-set identity | `_test472_paths.csv` vs split test rows vs `video_paths` in every headline file | Pass |
| Split leakage | Session blocks rebuilt from file names; take-groups and blocks crossing train/test counted | Pass: 0 for the session-disjoint training and deployed splits |
| Numerical consistency across abstract, text, tables, figures | Manual pass plus search for stale values (22.9, 91.1, 3.01, 24 ms, "at chance") | Pass: no hits remain in the paper |
| Scope of the paper's text | At the authors' request the paper states measurement conditions as scope and carries no limitations section; the audit notes on open items stay in this folder (`discrepancy_register.md`, `rubric_compliance.md`, this file) | Done |
| Code and result links | `scripts/check_links.py` on the PDF: 100 repository links checked against commit `3330b87` (path type and `#L` range); 38 external links requested | Pass: 138/138 resolve (one DOI failed in Python's certificate store and was confirmed with `curl`; three publishers refuse scripted requests after a valid DOI redirect) |
| Citation metadata | Crossref API for the 34 DOIs; arXiv API for the entries cited by arXiv identifier; reported results checked in the papers' own text (INCLUDE abstract, OpenHands Table 2/4, CISLR Tables 2–4, SPOTER, WLASL, AUTSL, SAM-SLR, TinyMSLR, Kannada SLR, Nedungadi et al., Shawon et al.) | Pass; author given names and page ranges corrected from Crossref |
| Figure and table numbering, cross-references | LaTeX log: no undefined references or citations | Pass |
| LaTeX compilation | `latex/build.sh` (pdflatex + BibTeX) | Pass: 25 pages; four overfull lines of 1.3 pt in Table A1 |
| PDF visual inspection | Every page rendered at 80 dpi and inspected; page 1 compared with the official `sn-article.pdf` sample compiled with the same abstract | Pass: page geometry identical to the sample (A4, 372 pt text block); keywords on page 1; all tables upright; every figure within two pages of its first reference |
| Figure readability | Plots regenerated at the template text width (372 pt) with 7.2 pt text; checked at print size | Pass |
| Architecture diagram matches code | Fig. 1 compared with `ISLViT.__init__`, `temporal`, `landmark_features` | Pass |
| Rubric coverage | `rubric_compliance.md` | Two items need the team's confirmation (one model per member; author spelling); poster missing |
| Plagiarism check | Not done (no access to the institution's checker); prose was written from the audit, not copied | Not done |
| Model-only latency | Measured twice; first run (80.7 ms) contaminated by a background job and discarded | Pass on idle machine (20.9 ms) |
| Browser-level UI test | Not done in this audit (server route tested instead) | Not done |
| Wearable or phone hardware | Not available | Not done |

## Open items before submission

1. Confirm the spelling "Dhruseek" vs "Dhrusheek" (D13) and the author order.
2. Confirm how the "one model per team member" requirement is to be read (D14).
3. Run the plagiarism check.
4. If the Review 3 slides are reused, update the CISLR statement, latency figures and leakage gap (D08, D12, D01).
