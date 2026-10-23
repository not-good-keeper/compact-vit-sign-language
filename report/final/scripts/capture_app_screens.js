// Screenshots of the running application (python -m islvit.demo), driven in a real Chrome.
//
//   node report/final/scripts/capture_app_screens.js tour   OUT_DIR LOG.json [port]
//   node report/final/scripts/capture_app_screens.js lowres OUT_DIR LOG.json PORT CLIP.mkv NAME
//
// "tour" walks the seven screens with the built-in demo clips. "lowres" uploads one stored 180p clip through the
// Capture screen's file input. Every screenshot is logged next to the server's own answer for the clip that
// produced it (top candidates, confidence, detection share, seconds), so each image can be traced to a file,
// a clip and a model. Nothing here changes the app; the corrections log is not written (no candidate is picked).
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const [mode, outDir, logPath, arg3, arg4, arg5] = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const port = mode === "tour" ? (arg3 || 7860) : arg3;
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new", args: ["--no-sandbox"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 940, deviceScaleFactor: 1 });
  const log = fs.existsSync(logPath) ? JSON.parse(fs.readFileSync(logPath, "utf8")) : { shots: [] };
  const answers = [];
  page.on("response", async (response) => {
    if (!response.url().endsWith("/predict")) return;
    try {
      const j = await response.json();
      answers.push({
        at: new Date().toISOString(), top: j.candidates && j.candidates[0], confidence: j.confidence,
        detection_rate: j.detection_rate, frames: j.frames, views: j.views, seconds: j.seconds,
        candidates: (j.candidates || []).map((c) => [c.word, +c.p.toFixed(4)]),
      });
    } catch (e) { /* not JSON */ }
  });

  const shot = async (name, note, extra = {}, fullPage = true) => {
    const file = path.join(outDir, name + ".png");
    await page.evaluate(() => { window.scrollTo(0, 0); document.querySelectorAll("*").forEach((e) => { if (e.scrollTop) e.scrollTop = 0; }); });
    await sleep(500);
    await page.screenshot({ path: file, fullPage });
    const sha = crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
    log.shots = log.shots.filter((s) => s.file !== name + ".png");
    const ui = await page.evaluate(() => {
      const g = (q) => { const e = document.querySelector(q); return e ? e.textContent.trim().replace(/\s+/g, " ") : null; };
      return { screen_title: g("#screen-title"), kicker: g('[data-ref="kicker"]'), word: g('[data-ref="word"]'), conf: g('[data-ref="conf"]'),
               status: g('[data-ref="status"]'), warning: g('[data-ref="warn"]'), tau: g("#tau-text"), views: g("#views-text") };
    });
    log.shots.push({ file: name + ".png", sha256: sha, note, port: +port, on_screen: ui, server_answer: answers[answers.length - 1] || null, ...extra });
    console.log("shot", name);
  };
  const go = async (screen) => {
    await page.click(`button[data-go="${screen}"]`);
    await sleep(900);
  };
  const waitConnected = async () => {
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "networkidle2" });
    await page.waitForFunction(() => !/connecting/i.test(document.getElementById("mode-text").textContent), { timeout: 60000 });
    await sleep(1200);
  };
  const playSample = async (i) => {
    await page.click(`.cap-sample[data-sample="${i}"]`);
  };
  const sampleDone = async (i) => {
    await page.waitForFunction((k) => {
      const s = document.querySelector(`.cap-sample[data-sample="${k}"]`);
      return s && ["right", "wrong", "unsure"].includes(s.dataset.state);
    }, { timeout: 240000 }, i);
    await sleep(1800);
  };

  await waitConnected();
  const info = await page.evaluate(() => ({ run: document.getElementById("rail-run").textContent, spec: document.getElementById("rail-spec").textContent }));
  log.model_shown_in_app = info;

  if (mode === "tour") {
    const manifest = await (await fetch(`http://127.0.0.1:${port}/samples`)).json();
    log.manifest = manifest.samples.map((s, i) => ({ index: i, file: s.file, word: s.word, source: s.source, server_p_at_build: s.server_p }));
    await shot("app_01_capture_idle", "Capture screen after load, before any clip", { screen: "capture" });

    await playSample(4);                       // Brother, caught mid-read
    await sleep(2800);
    await shot("app_02_capture_reading", "Capture screen while the server reads the Brother clip", { screen: "capture", clip: manifest.samples[4].file });
    await sampleDone(4);

    await playSample(3);                       // Hello: confident
    await sampleDone(3);
    await shot("app_03_capture_confident", "Capture screen after a confident answer (Hello)", { screen: "capture", clip: manifest.samples[3].file });
    await page.click('[data-act="accept"]');
    await sleep(600);

    for (const i of [0, 7]) {                  // loud, Dress (the lowest-confidence reliable clip)
      await playSample(i);
      await sampleDone(i);
      if (i === 7) await shot("app_04_capture_lower_confidence", "Capture screen after the Dress clip, the lowest-confidence reliable demo clip", { screen: "capture", clip: manifest.samples[i].file });
      await page.click('[data-act="accept"]');
      await sleep(600);
    }

    await playSample(9);                       // wide: the look-alike hard case
    await sampleDone(9);
    await shot("app_05_capture_unsure", "Capture screen after the look-alike clip (wide), below tau", { screen: "capture", clip: manifest.samples[9].file });
    await page.click('[data-act="shortlist"]');
    await sleep(1500);
    await shot("app_06_notsure", "Not sure screen: the five candidates for the look-alike clip", { screen: "notsure", clip: manifest.samples[9].file });

    await go("sentence");
    await shot("app_07_sentence", "Sentence screen with the three accepted words and the clip timeline", { screen: "sentence" });
    await go("settings");
    await shot("app_08_settings", "Settings screen at the default tau and view count", { screen: "settings" });
    await go("diagnostics");
    await sleep(900);
    await shot("app_09_detection", "Detection screen after the session's clips", { screen: "diagnostics" });
    await go("vocabulary");
    await sleep(900);
    await shot("app_10_vocabulary", "Vocabulary screen: per-word recall of the shipped model", { screen: "vocabulary" });
    await go("enrolment");
    await shot("app_11_enrolment", "Enrolment screen: the take queue", { screen: "enrolment" });
    await go("capture");
    await page.click("#help-btn");
    await sleep(900);
    await shot("app_12_shortcuts", "Keyboard-shortcuts dialog", { screen: "help" }, false);
  } else {
    const clip = arg4, name = arg5;
    const input = await page.$('input[type="file"]');
    await input.uploadFile(clip);
    await page.waitForFunction(() => {
      const lens = document.querySelector('[data-ref="lens"]');
      return lens && ["sure", "unsure"].includes(lens.dataset.state);
    }, { timeout: 240000 });
    await sleep(2200);
    await shot(name, `Capture screen after uploading ${path.basename(path.dirname(clip))}/${path.basename(clip)} (180p, JPEG 40) to the server on port ${port}`, { screen: "capture", clip: clip.replace(/\\/g, "/") });
  }

  fs.writeFileSync(logPath, JSON.stringify(log, null, 1));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
