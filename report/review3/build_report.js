// Build the Review 3 case-study report (.docx) from report_content.js.
//   NODE_PATH=<dir with docx installed> node report/review3/build_report.js
// Layout follows the course format: Title, Authors with affiliation, Abstract, Keywords,
// numbered sections, References; single-column journal style in the spirit of Springer Nature.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, HeadingLevel, PageNumber, Footer, LevelFormat, TableLayoutType,
} = require("docx");

const HERE = __dirname;
const C = require(path.join(HERE, "report_content.js"));
const SIZES = JSON.parse(fs.readFileSync(path.join(HERE, "assets", "sizes.json"), "utf8"));

const FONT = "Times New Roman";
const BODY = 20;            // half-points: 10 pt
const SMALL = 18;           // 9 pt
const PAGE_W = 11906, MARGIN = 1134;   // A4, 2 cm margins (twips)
const TEXT_W = PAGE_W - 2 * MARGIN;    // 9638 twips
const TEXT_PX = Math.round(TEXT_W / 1440 * 96);   // usable width in px at 96 dpi

const run = (r, size = BODY) => {
  if (typeof r === "string") return new TextRun({ text: r, font: FONT, size });
  if (r.b) return new TextRun({ text: r.b, bold: true, font: FONT, size });
  if (r.i) return new TextRun({ text: r.i, italics: true, font: FONT, size });
  throw new Error("bad run " + JSON.stringify(r));
};
const runs = (content, size) => (Array.isArray(content) ? content : [content]).map((r) => run(r, size));

let h1 = 0, h2 = 0, fig = 0;
const out = [];

// ---- front matter
out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: C.title, bold: true, font: FONT, size: 32 })] }));
out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [new TextRun({ text: C.authors.join(" · "), font: FONT, size: 22 })] }));
out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: C.affiliation, italics: true, font: FONT, size: SMALL })] }));
out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
  children: [new TextRun({ text: C.guide, italics: true, font: FONT, size: SMALL })] }));
out.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Abstract", bold: true, font: FONT, size: 22 })] }));
for (const para of C.abstract) {
  out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 264 }, children: runs(para, SMALL + 1) }));
}
out.push(new Paragraph({ spacing: { before: 60, after: 280 }, children: [
  new TextRun({ text: "Keywords: ", bold: true, font: FONT, size: SMALL + 1 }),
  new TextRun({ text: C.keywords, font: FONT, size: SMALL + 1 })] }));

// ---- body
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const thick = { style: BorderStyle.SINGLE, size: 10, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function cell(content, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    borders: { top: opts.top || none, bottom: opts.bottom || none, left: none, right: none },
    children: [new Paragraph({ children: runs(content, SMALL).map((r) => r), spacing: { after: 0 },
      ...(opts.head ? {} : {}) })],
  });
}

for (const block of C.body) {
  if (block.h1) {
    h1 += 1; h2 = 0;
    out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 300, after: 120 },
      children: [new TextRun({ text: `${h1} ${block.h1}`, bold: true, font: FONT, size: 24 })] }));
  } else if (block.h2) {
    h2 += 1;
    out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 200, after: 80 },
      children: [new TextRun({ text: `${h1}.${h2} ${block.h2}`, bold: true, italics: true, font: FONT, size: 21 })] }));
  } else if (block.p) {
    out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 264 }, children: runs(block.p) }));
  } else if (block.list) {
    for (const item of block.list) {
      out.push(new Paragraph({ numbering: { reference: "contrib", level: 0 }, alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 80, line: 264 }, children: runs(item) }));
    }
  } else if (block.fig) {
    fig += 1;
    const [pw, ph] = SIZES[block.fig];
    const w = Math.round(TEXT_PX * block.w);
    const h = Math.round(w * ph / pw);
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(HERE, "assets", block.fig)),
        transformation: { width: w, height: h }, altText: { title: `Fig. ${fig}`, description: block.caption, name: block.fig } })] }));
    out.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 200 }, children: [
      new TextRun({ text: `Fig. ${fig} `, bold: true, font: FONT, size: SMALL }),
      new TextRun({ text: block.caption, font: FONT, size: SMALL })] }));
  } else if (block.table) {
    const t = block.table;
    const widths = t.widths.map((f) => Math.round(TEXT_W * f));
    widths[widths.length - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
    out.push(new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 }, children: [
      new TextRun({ text: t.caption.split(" ").slice(0, 2).join(" ") + " ", bold: true, font: FONT, size: SMALL }),
      new TextRun({ text: t.caption.split(" ").slice(2).join(" "), font: FONT, size: SMALL })] }));
    const rows = [new TableRow({ tableHeader: true, children: t.head.map((hd, i) =>
      cell({ b: hd }, widths[i], { top: thick, bottom: thin })) })];
    t.rows.forEach((r, ri) => rows.push(new TableRow({ cantSplit: true, children: r.map((c, i) =>
      cell(c, widths[i], { bottom: ri === t.rows.length - 1 ? thick : undefined })) })));
    out.push(new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, rows }));
    out.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  } else {
    throw new Error("unknown block " + JSON.stringify(block).slice(0, 80));
  }
}

// ---- references
out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 300, after: 120 },
  children: [new TextRun({ text: "References", bold: true, font: FONT, size: 24 })] }));
C.references.forEach((ref, i) => {
  out.push(new Paragraph({ spacing: { after: 60 }, indent: { left: 400, hanging: 400 },
    children: [new TextRun({ text: `${i + 1}. `, font: FONT, size: SMALL }), new TextRun({ text: ref, font: FONT, size: SMALL })] }));
});

const doc = new Document({
  creator: "Group 03, 23CSE473",
  title: "ISL Reader: Review 3 case-study report",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 21, bold: true, italics: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "contrib", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SMALL })] })] }) },
    children: out,
  }],
});

const target = path.join(HERE, "ISL_Reader_Review3_Case_Study.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(target, buf); console.log("wrote", target, `(${fig} figures)`); });
