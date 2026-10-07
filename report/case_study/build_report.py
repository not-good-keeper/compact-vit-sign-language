"""Build the case-study report as a self-contained HTML page, a PDF and a DOCX.

Run make_diagrams.py first; the report embeds its PNGs.

    python report/case_study/build_report.py
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = "ISL_ViT_Tiny_Case_Study.md"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# Single-column journal layout in the spirit of the Springer Nature template.
CSS = """
@page { size: A4; margin: 20mm 20mm 22mm 20mm;
        @bottom-center { content: counter(page); font: 9pt 'Times New Roman', serif; color: #444; } }
html { background: #fff; }
body { font: 10.5pt/1.45 'Times New Roman', Times, serif; color: #111; max-width: 170mm;
       margin: 0 auto; text-align: justify; hyphens: auto; }
header { text-align: center; margin-bottom: 4mm; }
h1.title { font-size: 18pt; line-height: 1.25; font-weight: bold; margin: 0 0 5mm; text-align: center; }
.authors { text-align: center; font-size: 11pt; margin-bottom: 2mm; }
.affiliation { text-align: center; font-size: 9.5pt; font-style: italic; color: #333; margin-bottom: 6mm; }
.keywords { font-size: 9.5pt; margin: 3mm 0 6mm; }
h1 { font-size: 12.5pt; margin: 6mm 0 2mm; text-align: left; break-after: avoid; }
h2 { font-size: 11pt; font-style: italic; font-weight: bold; margin: 4mm 0 1.5mm; break-after: avoid; }
h1#abstract { margin-top: 2mm; }
#abstract + p { font-size: 9.8pt; }
p { margin: 0 0 2.2mm; }
ol, ul { margin: 0 0 2.5mm; padding-left: 7mm; }
figure { margin: 4mm 0 5mm; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 9pt; text-align: left; margin-top: 2mm; line-height: 1.35; }
table { border-collapse: collapse; margin: 2mm auto 5mm; font-size: 9pt; break-inside: avoid;
        border-top: 1.2px solid #000; border-bottom: 1.2px solid #000; }
caption { caption-side: top; font-size: 9pt; text-align: left; margin-bottom: 1.5mm; }
th { border-bottom: 0.8px solid #000; text-align: left; padding: 1.2mm 3mm; }
td { padding: 1mm 3mm; text-align: left; vertical-align: top; }
.references ol { font-size: 9pt; padding-left: 6mm; text-align: left; }
.references li { margin-bottom: 1.2mm; overflow-wrap: anywhere; }
"""


def main():
    (HERE / "report.css").write_text(CSS, encoding="utf-8")
    html = HERE / "ISL_ViT_Tiny_Case_Study.html"
    subprocess.run(["pandoc", SOURCE, "--standalone", "--self-contained", "--number-sections",
                    "--css", "report.css", "--metadata", "pagetitle=ISL-ViT-Tiny case study",
                    "-o", html.name], cwd=HERE, check=True)

    pdf = HERE / "ISL_ViT_Tiny_Case_Study.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf-no-header", f"--print-to-pdf={pdf}", html.as_uri()],
                   check=True, capture_output=True)

    subprocess.run(["pandoc", SOURCE, "--number-sections", "-o", "ISL_ViT_Tiny_Case_Study.docx"],
                   cwd=HERE, check=True)
    (HERE / "report.css").unlink()
    for path in (html, pdf, HERE / "ISL_ViT_Tiny_Case_Study.docx"):
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
