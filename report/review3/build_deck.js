// Review 3 presentation (.pptx). Numbers come from assets/chart_data.json (written by make_charts.py
// from runs/) and from the same verified figures as the report.
//   NODE_PATH=<dir with pptxgenjs> node report/review3/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HERE = __dirname;
const A = (f) => path.join(HERE, "assets", f);
const SIZES = JSON.parse(fs.readFileSync(A("sizes.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(A("chart_data.json"), "utf8"));

const THEME = {
  name: "ISL Reader",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2430", lt1: "FFFFFF", dk2: "12161F", lt2: "EEF1F5",
    accent1: "2E9E6E", accent2: "C065B0", accent3: "5B7BD6", accent4: "D98C2B", accent5: "A3AAB8", accent6: "8E5BB5",
    hlink: "5B7BD6", folHlink: "8E5BB5",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "ISL Reader - Review 3";
pres.author = "Group 03";
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.55, w: 7.0, h: 1.6, fontFace: THEME.headFontFace, fontSize: 48, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.25, w: 6.9, h: 1.5, fontSize: 18, color: "C9CFDA", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "ISL Reader · Review 3 · Group 03", options: { x: 0.6, y: 7.05, w: 6, h: 0.3, fontSize: 10, color: "8A93A3", margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.03, w: 0.5, h: 0.3, fontSize: 10, color: "8A93A3", align: "right" },
});

// ---------------------------------------------------------------- helpers
let slideNo = 0;
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  slideNo += 1;
  return s;
}
// Image fitted inside a box, preserving aspect ratio; optional light card behind it.
function img(s, file, x, y, w, h, opts = {}) {
  const [pw, ph] = SIZES[file];
  const pad = opts.card ? 0.12 : 0;
  const bw = w - 2 * pad, bh = h - 2 * pad;
  let iw = bw, ih = bw * ph / pw;
  if (ih > bh) { ih = bh; iw = bh * pw / ph; }
  if (opts.card) s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "card " + file });
  s.addImage({ path: A(file), x: x + pad + (bw - iw) / 2, y: y + pad + (bh - ih) / 2, w: iw, h: ih, objectName: file,
    altText: opts.alt || file, rounding: false });
}
function card(s, x, y, w, h, title, body, accent = C.accent1) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "card " + title });
  s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.24, w: 0.22, h: 0.22, fill: { color: accent }, line: { color: accent }, objectName: "dot " + title });
  s.addText(title, { x: x + 0.56, y: y + 0.12, w: w - 0.75, h: 0.46, fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  s.addText(body, { x: x + 0.24, y: y + 0.66, w: w - 0.46, h: h - 0.8, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, paraSpaceAfter: 4 });
}
function stat(s, x, y, w, value, label, color = C.accent1) {
  s.addText(value, { x, y, w, h: 0.85, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color, margin: 0, isTextBox: true });
  s.addText(label, { x, y: y + 0.85, w, h: 0.5, fontSize: 13, color: "5B6474", margin: 0, isTextBox: true, valign: "top" });
}
function bullets(s, items, x, y, w, h, size = 15) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontSize: size, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true });
}
const chartBase = (title) => ({
  showTitle: !!title, title, titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  catAxisLabelColor: "4B5563", valAxisLabelColor: "4B5563", catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  valGridLine: { color: "E5E7EB", size: 0.75 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.0",
});

// ================================================================ 1 title
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  slideNo += 1;
  s.addText("ISL Reader", { placeholder: "title" });
  s.addText("A 2 MB Vision Transformer, a super-resolution GAN front end and an integrated interface for Indian Sign Language", { placeholder: "body" });
  s.addText([
    { text: "Yalamanchi Kushal · Nethi Kushala Kumar · Dhruseek Rishi Menon", options: { breakLine: true } },
    { text: "Guide: Dr. T. Bagyammal · 23CSE473 NNDL · Group 03 · Review 3" },
  ], { x: 0.7, y: 5.45, w: 6.9, h: 0.9, fontSize: 14, color: "9AA3B2", margin: 0, isTextBox: true });
  img(s, "ui_capture_crop.png", 7.9, 1.45, 4.95, 4.6, { alt: "The ISL Reader capture screen reading a held-out clip" });
  s.addNotes("ISL Reader recognises isolated Indian Sign Language words from short video. This review covers the full system: the classifier, the super-resolution GAN we added in front of the hand detector, and the web interface that ties them together. The screenshot on the right is the live app reading a held-out clip of 'Hello' at 97 % confidence.");
}

// ================================================================ 2 goal
pres.addSection({ title: "Problem and data" });
{
  const s = content("What we set out to build", "Problem and data");
  card(s, 0.6, 1.45, 3.95, 1.8, "A model of about 2 MB", "It has to run on smart glasses, so the classifier must fit in roughly two megabytes.", C.accent1);
  card(s, 4.69, 1.45, 3.95, 1.8, "≥ 75 % on unseen people", "Measured only on recording sessions the model never saw, on the 50 deployed words.", C.accent3);
  card(s, 8.78, 1.45, 3.95, 1.8, "A working app", "Camera or video in, word out, with a confidence check and a way to correct mistakes.", C.accent2);
  s.addText("Why it is hard", { x: 0.6, y: 3.65, w: 6, h: 0.45, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
  bullets(s, [
    "Signs are two-handed: handshape, motion and location together decide the word.",
    "Little labelled data: about 15 clips per word in INCLUDE.",
    "A wearable camera gives smaller, compressed video than the 1080p studio data.",
    "Random splits leak recording sessions, so published accuracies overstate real use.",
  ], 0.6, 4.2, 12.1, 2.6, 18);
  s.addNotes("Three targets: about 2 MB, at least 75 % top-1 on recording sessions held out from training, and a working app. Four difficulties: two-handed signs, scarce data, low-resolution wearable video, and leaky evaluation in the literature.");
}

// ================================================================ 3 dataset
{
  const s = content("Dataset details", "Problem and data");
  stat(s, 0.6, 1.35, 2.3, "4,257", "INCLUDE clips used");
  stat(s, 2.95, 1.35, 2.1, "262", "words, 15 categories", C.accent3);
  stat(s, 5.05, 1.35, 2.4, "1080p", "25 fps, one studio", C.accent2);
  stat(s, 7.45, 1.35, 1.9, "13", "recording sessions", C.accent4);
  const rows = [
    ["Split", "Clips", "Use"],
    ["Training sessions", "2,845", "Classifier and GAN training"],
    ["Held-out test sessions", "1,010", "472 of them = the 50 deployed words"],
    ["Excluded", "402", "Words with < 6 training clips"],
    ["Validation sessions", "888", "GAN development only"],
    ["iSign (unlabelled)", "18,000", "Self-supervised pretraining"],
    ["CISLR (300 x 300 px)", "609", "Cross-corpus test"],
  ];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: {
    bold: i === 0, color: i === 0 ? HEX.lt1 : HEX.dk1, fill: { color: i === 0 ? HEX.dk2 : (i % 2 ? HEX.lt1 : HEX.lt2) } } }))),
    { x: 0.6, y: 2.9, w: 8.75, colW: [2.8, 1.25, 4.7], fontSize: 13, fontFace: THEME.bodyFontFace, border: { type: "solid", pt: 0.5, color: "D5DAE2" }, margin: 0.06, rowH: 0.42 });
  const posters = ["01_loud.jpg", "04_hello.jpg", "08_dress.jpg", "06_i.jpg"];
  const dir = path.join(HERE, "..", "..", "demo_samples");
  const avail = fs.readdirSync(dir).filter((f) => f.endsWith(".jpg"));
  avail.slice(0, 4).forEach((f, i) => {
    s.addImage({ path: path.join(dir, f), x: 9.75 + (i % 2) * 1.55, y: 1.45 + Math.floor(i / 2) * 1.0, w: 1.45, h: 0.9, objectName: "frame " + f, altText: "INCLUDE frame" });
  });
  s.addText("Frames from held-out INCLUDE clips (Sridhar et al., ACM MM 2020, CC-BY-4.0)", { x: 9.75, y: 3.5, w: 3.0, h: 0.5, fontSize: 10, color: "6B7280", margin: 0, isTextBox: true });
  s.addText("We train on all 262 words and answer from the 50 deployed words: per-word data is fixed, so a smaller training vocabulary only removes data.", { x: 9.75, y: 4.2, w: 3.0, h: 2.4, fontSize: 13, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
  s.addNotes("INCLUDE is the labelled corpus: 4,257 usable clips of 262 words, recorded at 1080p in one studio. We rebuilt 13 recording sessions from the camera's clip numbers because the dataset has no signer labels. 2,845 clips train the model; 1,010 clips from held-out sessions test it, and 472 of those are the 50 deployed words used in every result. iSign gives 18,000 unlabelled clips for pretraining; CISLR is a second corpus for a transfer test.");
}

// ================================================================ 4 protocol
{
  const s = content("Evaluation: hold out whole recording sessions", "Problem and data");
  img(s, "fig3_evaluation_split.png", 0.6, 1.35, 5.4, 4.6, { card: true, alt: "Building the session split" });
  s.addChart(pres.charts.BAR, [{ name: "Top-1 accuracy (%)", labels: ["Random video split", "Held-out sessions"], values: [D.leakage.random, D.leakage.sessions] }], {
    x: 6.4, y: 1.35, w: 6.3, h: 3.7, barDir: "col", chartColors: [HEX.accent2, HEX.accent1], valAxisMinVal: 0, valAxisMaxVal: 100,
    ...chartBase("One pixel-only model, two splits"), showLegend: false, barGapWidthPct: 60,
  });
  bullets(s, [
    "Camera clip numbers run in sequence; big jumps mark a new session.",
    "Whole sessions are held out; a check refuses any overlap.",
    "The 22.9-point gap is what the model gains from seeing the same person and room.",
  ], 6.4, 5.2, 6.3, 1.7, 14);
  s.addNotes("INCLUDE was recorded as back-to-back takes. A random split puts near-identical takes on both sides, so the same pixel-only model scores 98.5 % there and 75.6 % on unseen sessions. Every result after this slide is on unseen sessions, with three seeds and paired McNemar tests.");
}

// ================================================================ 5 architecture
pres.addSection({ title: "Architecture and algorithms" });
{
  const s = content("System architecture of the integrated app", "Architecture and algorithms");
  img(s, "d1.png", 0.6, 1.3, 12.1, 5.55, { alt: "Browser, server, front end and classifier" });
  s.addNotes("The browser posts a clip to the Python server. The server decodes 32 frames, upscales low-resolution video with either bicubic or the SR-GAN, runs MediaPipe Holistic, builds three crops and hand landmarks, and runs the classifier on six views. The answer goes back with the top five words and how each crop was found. The server can load any trained checkpoint and either front end at start-up, and it restricts answers to the 50 deployed words exactly as the evaluation does.");
}

// ================================================================ 6 pipeline
{
  const s = content("From video to model inputs", "Architecture and algorithms");
  img(s, "fig0_how_it_works.png", 0.6, 1.3, 6.4, 5.55, { card: true, alt: "How a sign becomes a word" });
  bullets(s, [
    "32 frames sampled uniformly, letterboxed to 1280 x 720.",
    "MediaPipe Holistic finds hands, face and pose; a missed hand gets a second search in an upscaled forearm region, else the nearest box is reused and flagged.",
    "Three crops per frame (left hand, right hand, face), 64 px.",
    "21 joints per hand: shape relative to the wrist, location relative to the shoulders.",
    "Detectors are rebuilt for every video; otherwise repeated runs disagreed on 3–6 % of frames.",
  ], 7.35, 1.45, 5.35, 5.3, 15);
  s.addNotes("This is the input pipeline with real frames from a held-out clip. The landmark features are the biggest single lever in the project: the detector was already computing them, and adding them raised accuracy from 73.7 to 86.5 percent.");
}

// ================================================================ 7 classifier
{
  const s = content("Classifier: ISL-ViT-Tiny", "Architecture and algorithms");
  img(s, "fig2_model_architecture.png", 0.6, 1.3, 5.2, 5.55, { card: true, alt: "Model architecture" });
  const rows = [
    ["Component", "Setting"],
    ["Input", "16 frames x 3 crops x 64 x 64, box geometry, detection flags, landmarks"],
    ["Spatial stage", "4 blocks, width 192, 3 heads, shared over 48 crops"],
    ["Temporal stage", "4 blocks over 48 crop tokens + class token"],
    ["Output", "262 words, restricted to the 50 deployed"],
    ["Parameters", "3,894,214 (91 k for landmarks)"],
  ];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? HEX.lt1 : HEX.dk1, fill: { color: i === 0 ? HEX.dk2 : (i % 2 ? HEX.lt1 : HEX.lt2) } } }))),
    { x: 6.15, y: 1.4, w: 6.55, colW: [1.9, 4.65], fontSize: 13, fontFace: THEME.bodyFontFace, border: { type: "solid", pt: 0.5, color: "D5DAE2" }, margin: 0.06 });
  s.addText([
    { text: "Why factorised: ", options: { bold: true } },
    { text: "joint attention over every patch would be 816 tokens; the model attends over 17 tokens per crop, then 49 across the clip. Box geometry puts back the location that cropping removes." },
  ], { x: 6.15, y: 4.75, w: 6.55, h: 1.9, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("A spatial ViT reads each crop on its own with shared weights; each crop's token is tagged with its stream, frame, box position, a reliability flag and the landmark features; a temporal transformer then attends across all 48 crop tokens. Width 192 and three heads match DeiT-Tiny so pretrained weights load directly.");
}

// ================================================================ 8 training
{
  const s = content("Training algorithm and compression", "Architecture and algorithms");
  img(s, "fig4_training_pipeline.png", 0.6, 1.2, 12.1, 2.85, { alt: "Training pipeline" });
  card(s, 0.6, 4.3, 3.95, 2.45, "1. Self-supervised pretraining", "Masked image modelling (SimMIM) on 18,000 unlabelled iSign clips: 75 % of patches hidden and reconstructed.", C.accent3);
  card(s, 4.69, 4.3, 3.95, 2.45, "2. Supervised training", "1,000 epochs on 2,845 clips, AdamW 5e-4, mixup, stream dropout, resolution jitter, EMA. The final epoch is kept: no selection on test.", C.accent1);
  card(s, 8.78, 4.3, 3.95, 2.45, "3. 4-bit quantisation-aware", "Weights at 4 bits in groups of 128 with 16-bit scales; straight-through estimator; 300 epochs. Post-training rounding lost 3.1 points, QAT only 0.6.", C.accent2);
  s.addNotes("Pretraining by masked image modelling on unlabelled iSign video, then supervised training for a fixed 1,000 epochs keeping the final weights, then 300 epochs of quantisation-aware training. Rounding after training cost 3.1 points; training with the rounding in the loop cost 0.6, at a seventh of the file size.");
}

// ================================================================ 9 classifier comparison
pres.addSection({ title: "Comparisons" });
{
  const s = content("Comparing the classifier variants", "Comparisons");
  s.addChart(pres.charts.BAR, [{ name: "Top-1 (%)", labels: D.models.map((m) => `${m.label} (${m.size})`), values: D.models.map((m) => m.top1) }], {
    x: 0.6, y: 1.3, w: 7.6, h: 5.45, barDir: "col", chartColors: [HEX.accent5, HEX.accent3, HEX.accent2, HEX.accent1],
    valAxisMinVal: 0, valAxisMaxVal: 100, ...chartBase("Top-1 on 472 held-out clips, mean of 3 seeds"), showLegend: false, barGapWidthPct: 50,
  });
  stat(s, 8.6, 1.45, 4.1, "85.9 %", "top-1, shipped 4-bit model (seeds 84.5 / 87.5 / 85.8)");
  stat(s, 8.6, 2.95, 4.1, "96.4 %", "top-5: the right word is usually in the shortlist", C.accent3);
  stat(s, 8.6, 4.45, 4.1, "1.97 MiB", "3.89 M parameters, 4-bit", C.accent2);
  s.addText("Answering from all 262 words instead of the 50 gives 72.0 %, so the app restricts answers exactly as the evaluation does.", { x: 8.6, y: 5.95, w: 4.1, h: 0.85, fontSize: 12, color: "4B5563", margin: 0, isTextBox: true });
  s.addNotes("Landmarks are the big step: 73.7 to 86.5 percent. Post-training 4-bit rounding drops to 83.4; quantisation-aware training holds 85.9 at 1.97 MiB, 10.9 points above the 75 percent target. The same model answering from all 262 words scores 72.0, which is why the app restricts to the 50 deployed words.");
}

// ================================================================ 10 why a GAN
{
  const s = content("Where a GAN can help: low-resolution video", "Comparisons");
  // flow diagram in shapes
  const steps = [["Camera video", C.background2], ["Below 720p?", C.background2], ["Bicubic  |  SR-GAN", C.accent1], ["MediaPipe hands", C.background2], ["ISL-ViT-Tiny", C.background2]];
  steps.forEach(([t, fill], i) => {
    const x = 0.6 + i * 2.48;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.45, w: 2.1, h: 0.9, rectRadius: 0.1, fill: { color: fill }, line: { color: i === 2 ? HEX.accent1 : "C9CFDA", width: 1 }, objectName: "step " + t });
    s.addText(t, { x, y: 1.45, w: 2.1, h: 0.9, fontSize: 14, bold: true, align: "center", valign: "middle", color: i === 2 ? C.background1 : C.text1, margin: 0, isTextBox: true });
    if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + 2.12, y: 1.9, w: 0.34, h: 0, line: { color: "6B7280", width: 1.5, endArrowType: "triangle" }, objectName: "arrow " + i });
  });
  card(s, 0.6, 2.7, 5.95, 2.35, "Why here", "Every clip reaches the hand detector through one resize to 720p, which was bicubic. A wearable camera would send small, compressed video through exactly this step. The GAN replaces only that resize; 1080p video never passes through it, so clean results cannot change.", C.accent1);
  card(s, 6.75, 2.7, 5.95, 2.35, "How we tested it", "472 test clips degraded to 360p, 240p, 180p with JPEG. Front ends: bicubic, SR-L1 (same generator, no adversarial loss), SR-GAN v1, v2 and two v3 arms. Frozen 4-bit and 32-bit models, plus models fine-tuned on 360p. Paired McNemar tests; a rule fixed before the runs.", C.accent3);
  stat(s, 0.6, 5.3, 2.9, "3", "levels: 360p, 240p, 180p, each with JPEG");
  stat(s, 3.65, 5.3, 2.9, "6", "front ends: bicubic, SR-L1, SR-GAN v1, v2, two v3 arms", C.accent3);
  stat(s, 6.7, 5.3, 2.9, "472", "test clips, identical in every run", C.accent2);
  stat(s, 9.75, 5.3, 2.95, "p < 0.05", "and all 3 seeds agree, fixed in advance", C.accent4);
  s.addNotes("We looked for the place where a GAN can make a measurable difference. The only upscaling step in the pipeline is the resize to 720p before hand detection, and it is exactly where low-resolution wearable video would go. SR-L1 is the key control: same network, no adversarial loss, so it separates what learned upscaling gives from what the GAN adds.");
}

// ================================================================ 11 GAN architecture
{
  const s = content("The super-resolution GAN", "Comparisons");
  img(s, "d2.png", 0.6, 1.3, 12.1, 5.55, { alt: "Generator, discriminator, losses and schedule" });
  s.addNotes("Real-ESRGAN-style design. Generator: compact VGG-style network working at low resolution with a pixel shuffle and a residual upsample. Discriminator: U-Net with spectral normalisation, scoring each pixel. Training patches come only from training sessions and are degraded by random blur, resize, noise and JPEG applied twice. Losses: L1, VGG perceptual and adversarial. Version 3 adds a hand-landmark loss from a network we distilled from MediaPipe.");
}

// ================================================================ 12 GAN results
{
  const s = content("GAN results on degraded test video", "Comparisons");
  s.addChart(pres.charts.BAR, D.frozen_int4.arms.map((a) => ({ name: a.label, labels: D.frozen_int4.levels.map((l) => l + " + JPEG"), values: a.values })), {
    x: 0.6, y: 1.3, w: 8.0, h: 5.5, barDir: "col", chartColors: [HEX.accent5, HEX.accent3, HEX.accent1, "1E7A5A", HEX.accent6],
    valAxisMinVal: 50, valAxisMaxVal: 90, ...chartBase("Shipped 4-bit model, frozen (clean 1080p: 85.9 %)"), dataLabelFontSize: 8,
    showLegend: true, legendPos: "b", legendFontSize: 10, barGapWidthPct: 40,
  });
  stat(s, 9.0, 1.45, 3.7, "+14.5", "points at 180p, SR-GAN v2 vs bicubic (p = 7 × 10⁻³²)");
  stat(s, 9.0, 2.95, 3.7, "+1.5", "the adversarial loss's own gain over SR-L1, at 240p only", C.accent4);
  stat(s, 9.0, 4.45, 3.7, "0", "reliable gains for v3 over v2 or SR-L1, at any level", C.accent2);
  s.addText("Fine-tuned on 360p: +5.2 at 240p, +12.5 at 180p over bicubic.", { x: 9.0, y: 5.95, w: 3.7, h: 0.8, fontSize: 12, color: "4B5563", margin: 0, isTextBox: true });
  s.addNotes("Learned super-resolution helps at every degraded level: up to 14.5 points at 180p, and it still helps after the classifier is fine-tuned. But the adversarial part adds little: SR-GAN beats the non-adversarial SR-L1 only at 240p. More GAN training (v2) and the task-aware v3 did not improve on that.");
}

// ================================================================ 13 diagnosis
{
  const s = content("Where the remaining loss sits", "Comparisons");
  s.addChart(pres.charts.BAR, D.decomposition.arms.map((a) => ({ name: a.label, labels: D.decomposition.categories, values: a.values })), {
    x: 0.6, y: 1.3, w: 7.4, h: 5.5, barDir: "col", chartColors: [HEX.accent5, HEX.accent1], valAxisMinVal: 50, valAxisMaxVal: 90,
    ...chartBase("180p + JPEG 40, frozen 4-bit (clean 85.9 %)"), showLegend: true, legendPos: "b", legendFontSize: 11, barGapWidthPct: 50,
  });
  card(s, 8.35, 1.35, 4.35, 1.75, "Landmarks, not pixels", "With SR-GAN, 10.4 points are still missing at 180p: about 7.7 from the landmarks and 2.8 from the pixels.", C.accent1);
  card(s, 8.35, 3.25, 4.35, 1.95, "Detector tuning fails", "Lower thresholds and wider searches found up to 99 % of hand frames, but accuracy moved at most 0.8 points: misplaced hands rose to 14 %.", C.accent4);
  card(s, 8.35, 5.35, 4.35, 1.45, "No invented hands", "Hands found only in degraded video: 1.9 % with bicubic, 2.6 % with SR-GAN, at 180p.", C.accent3);
  s.addNotes("We swapped clean and degraded inputs. With clean pixels and degraded landmarks the model scores 78.2; with degraded pixels and clean landmarks 83.1. So after super-resolution the problem is where the detector places the joints. Tuning the detector finds more hands but more of them are misplaced.");
}

// ================================================================ 14 v3
{
  const s = content("A task-aware GAN (v3): what it changed", "Comparisons");
  card(s, 0.6, 1.35, 4.0, 3.75, "Hand-landmark proxy", "ResNet-18 with heatmap head, trained on 43,682 verified hand crops to read MediaPipe's 21 joints.\n\nError on validation crops: 0.0215 of the crop, 91.4 % of points within 0.05 (MediaPipe vs itself: 0.041).\n\nThe generator is penalised when the proxy reads different joints from its output than from the original.", C.accent6);
  s.addChart(pres.charts.BAR, [{ name: "180p", labels: D.v3_validation.labels, values: D.v3_validation.present_L3 }, { name: "240p", labels: D.v3_validation.labels, values: D.v3_validation.present_L2 }], {
    x: 4.85, y: 1.3, w: 7.85, h: 3.9, barDir: "col", chartColors: [HEX.accent1, "9BD3BB"], valAxisMinVal: 30, valAxisMaxVal: 95,
    ...chartBase("Validation: hand frames found (%), clean video 88.3"), dataLabelFontSize: 8, catAxisLabelFontSize: 9, showLegend: true, legendPos: "b", legendFontSize: 10,
  });
  s.addText([
    { text: "On validation it found more hands (73.3 % vs 63.3 % for v2 at 180p). ", options: { bold: true } },
    { text: "On the test set it was not reliably better than v2 or SR-L1, and the version without the adversarial loss was 3.1 points worse at 240p. Better detection did not become better classification." },
  ], { x: 4.85, y: 5.35, w: 7.85, h: 1.45, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  stat(s, 0.6, 5.3, 4.0, "−3.1", "points: v3 without the adversarial loss vs v2 at 240p (p = 5 × 10⁻⁶)", C.accent2);
  s.addNotes("Because the loss is in the landmarks, version 3 trains the generator against a differentiable stand-in for the hand detector. It did what it was designed to do at detection level on validation sessions, but not at the classifier on the test set. One untested explanation: the frozen classifier was trained on clean-video landmarks; fine-tuning it on the v3 front end would test that.");
}

// ================================================================ 15 interface
pres.addSection({ title: "Integration" });
{
  const s = content("The interface, connected to the models", "Integration");
  img(s, "ui_capture_crop.png", 0.6, 1.3, 7.3, 3.4, { card: true, alt: "Capture screen" });
  img(s, "ui_notsure_crop.png", 0.6, 4.85, 3.6, 1.95, { card: true, alt: "Not sure screen" });
  img(s, "ui_sentence_crop.png", 4.3, 4.85, 3.6, 1.95, { card: true, alt: "Sentence screen" });
  s.addText("Seven screens", { x: 8.25, y: 1.35, w: 4.45, h: 0.45, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
  bullets(s, [
    "Capture: camera, video file or one of 10 held-out demo clips",
    "Not sure: shortlist when confidence < threshold",
    "Sentence: accepted words, speak, copy, export",
    "Settings: threshold and views",
    "Detection: where each crop came from",
    "Vocabulary: per-word recall on unseen sessions",
    "Enrolment: record new signers for training",
  ], 8.25, 1.9, 4.45, 3.9, 14);
  s.addText("Runs any trained checkpoint and either upscaling front end; reached over an SSH tunnel, which also enables the camera.", { x: 8.25, y: 5.9, w: 4.45, h: 0.9, fontSize: 12, color: "4B5563", margin: 0, isTextBox: true });
  s.addNotes("The interface is one web page served by the recognition server. These are live screenshots: Capture reading a held-out clip, the Not-sure shortlist for a look-alike word, and the sentence built from accepted words. Corrections are logged as labelled examples, and the Enrolment screen records new signers into the training folder.");
}

// ================================================================ 16 tests
{
  const s = content("Testing the interface with the models", "Integration");
  const rows = [
    ["Test", "Method", "Result"],
    ["Screens render", "Chromium, 7 screens at 1366 and 1920 px", "No console errors"],
    ["Live answers", "10 demo clips on the live server; 4 also via the Capture screen", "9/9 reliable clips correct (66–97 %); look-alike → Not sure"],
    ["Same answers as the evaluation", "App vs offline result, same clips and model", "8 of 8 clips agree"],
    ["Vocabulary restriction", "50-word vs 262-word answers", "Matches the 85.9 % setting (262 words: 72.0 %)"],
    ["GAN front end in the app", "180p copy of a demo clip, SR-GAN on", "Read correctly"],
    ["Latency", "Desktop CPU, 6 views, end to end", "5–9 s per clip"],
  ];
  s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? HEX.lt1 : HEX.dk1, fill: { color: i === 0 ? HEX.dk2 : (i % 2 ? HEX.lt1 : HEX.lt2) } } }))),
    { x: 0.6, y: 1.4, w: 12.1, colW: [2.9, 4.6, 4.6], fontSize: 14, fontFace: THEME.bodyFontFace, border: { type: "solid", pt: 0.5, color: "D5DAE2" }, margin: 0.08 });
  img(s, "ui_capture_samples.png", 0.6, 4.55, 6.0, 2.3, { card: true, alt: "Demo clips with verdicts" });
  img(s, "ui_diagnostics.png", 6.7, 4.55, 6.0, 2.3, { card: true, alt: "Detection screen" });
  s.addNotes("Browser tests: Playwright driving Chromium. Live tests: the running server on held-out clips. Consistency test: the server's classification code against the offline evaluation's per-clip results. We tested at three levels: automated browser tests on every screen, live end-to-end tests through the running server, and a consistency test that the app gives the same answer as the offline evaluation for the same clip and model. It matched on all 8 clips, including the 2 the model gets wrong.");
}

// ================================================================ 17 conclusion
pres.addSection({ title: "Close" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Close" });
  slideNo += 1;
  s.addText("What we found", { placeholder: "title" });
  s.addText([
    { text: "85.9 % top-1 on unseen sessions from a 1.97 MiB model; random splits overstate accuracy by over 20 points.", options: { bullet: true, breakLine: true } },
    { text: "Learned super-resolution recovers up to 14.5 points on 180p video; the adversarial loss adds little beyond it.", options: { bullet: true, breakLine: true } },
    { text: "After upscaling, the remaining error is in landmark placement, not pixels.", options: { bullet: true, breakLine: true } },
    { text: "The app reproduces the evaluation's answers clip for clip.", options: { bullet: true } },
  ], { x: 0.7, y: 3.3, w: 7.2, h: 3.0, fontSize: 17, color: "E5E9F0", margin: 0, paraSpaceAfter: 10, isTextBox: true, valign: "top" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.35, y: 1.6, w: 4.35, h: 4.7, rectRadius: 0.12, fill: { color: "1E2430" }, line: { color: "2A3140" }, objectName: "next steps card" });
  s.addText("Next steps", { x: 8.65, y: 1.85, w: 3.8, h: 0.5, fontSize: 18, bold: true, color: C.background1, margin: 0, isTextBox: true });
  s.addText([
    { text: "Fine-tune the classifier on the task-aware front end", options: { bullet: true, breakLine: true } },
    { text: "Re-measure the confidence curve for the shipped model", options: { bullet: true, breakLine: true } },
    { text: "Fix hand detection for tightly framed video (CISLR)", options: { bullet: true, breakLine: true } },
    { text: "Record new signers through Enrolment; test on wearable hardware", options: { bullet: true } },
  ], { x: 8.65, y: 2.45, w: 3.8, h: 3.6, fontSize: 15, color: "C9CFDA", margin: 0, paraSpaceAfter: 8, isTextBox: true, valign: "top" });
  s.addNotes("Limitations: one studio, inferred sessions, no successful cross-corpus transfer yet, synthetic degradation, latency measured on a desktop CPU, isolated words only. Thank you; we can run the live demo now.");
}

(async () => {
  const file = path.join(HERE, "ISL_Reader_Review3.pptx");
  await pres.writeFile({ fileName: file });
  await applyTheme(file, THEME);
  console.log("wrote", file, slideNo, "slides");
})();
