/* ISL Reader -- core: state, server calls, routing and shared interaction.

   Screens register themselves on App.screens and talk to each other only through
   App.S, the functions below and the event bus. The server API is the one
   islvit/serve.py already exposes: GET /info, POST /predict, POST /correct,
   POST /enrol. Opened without that server (file://, or ?preview in the URL) the
   page runs in preview mode and says so; predictions are then simulated. */
"use strict";

const App = (window.App = { screens: {} });

(function () {
  /* ------------------------------------------------------------------ data --
     Numbers here are read from files already in the repository, not typed in:
     deployed words and per-word recall from runs/f16_262w_lm_s{0,1,2}_qat4/
     masked50_clean.json scored against splits/vocab50clean__session-disjoint.csv,
     and the gating curve from runs/gate_mixed.json. */
  const DATA = {"deployed":["1. loud","2. quiet","3. happy","37. Hat","38. Dress","40. I","41. Shirt","41. you","42. T-Shirt","42. he","43. Pant","43. she","44. Shoes","44. it","45. Pocket","45. we","46. you (plural)","47. they","48. Hello","49. How are you","50. Alright","51. Brown","51. Good Morning","52. Good afternoon","53. Good evening","56. Grey","66. Brother","77. Boy","78. long","79. short","80. tall","81. wide","82. narrow","83. big large","84. small little","85. slow","86. fast","87. hot","88. cold","89. warm","90. cool","91. new","92. old","93. young","94. good","95. bad","96. wet","97. dry","98. sick","99. healthy"],"recall":{"isl_vit_tiny_lm_int4_s0.pt":{"source":"runs/f16_262w_lm_s0_qat4/masked50_clean.json","top1":0.8453,"words":{"1. loud":[9,9],"2. quiet":[9,9],"3. happy":[9,9],"37. Hat":[10,10],"38. Dress":[6,10],"40. I":[6,9],"41. Shirt":[10,10],"41. you":[6,6],"42. T-Shirt":[10,10],"42. he":[10,10],"43. Pant":[10,10],"43. she":[10,10],"44. Shoes":[8,10],"44. it":[9,10],"45. Pocket":[4,10],"45. we":[10,10],"46. you (plural)":[8,10],"47. they":[10,10],"48. Hello":[10,10],"49. How are you":[10,10],"50. Alright":[9,10],"51. Brown":[9,11],"51. Good Morning":[7,10],"52. Good afternoon":[10,11],"53. Good evening":[11,11],"56. Grey":[10,10],"66. Brother":[10,10],"77. Boy":[9,10],"78. long":[5,9],"79. short":[9,9],"80. tall":[9,9],"81. wide":[4,9],"82. narrow":[6,9],"83. big large":[6,9],"84. small little":[4,9],"85. slow":[4,9],"86. fast":[4,9],"87. hot":[6,9],"88. cold":[8,8],"89. warm":[9,9],"90. cool":[9,9],"91. new":[7,9],"92. old":[9,9],"93. young":[9,9],"94. good":[5,9],"95. bad":[7,9],"96. wet":[3,9],"97. dry":[9,9],"98. sick":[9,9],"99. healthy":[9,9]}},"isl_vit_tiny_lm_int4_s1.pt":{"source":"runs/f16_262w_lm_s1_qat4/masked50_clean.json","top1":0.875,"words":{"1. loud":[9,9],"2. quiet":[9,9],"3. happy":[9,9],"37. Hat":[10,10],"38. Dress":[7,10],"40. I":[6,9],"41. Shirt":[10,10],"41. you":[6,6],"42. T-Shirt":[10,10],"42. he":[10,10],"43. Pant":[10,10],"43. she":[9,10],"44. Shoes":[10,10],"44. it":[10,10],"45. Pocket":[8,10],"45. we":[9,10],"46. you (plural)":[10,10],"47. they":[10,10],"48. Hello":[6,10],"49. How are you":[9,10],"50. Alright":[8,10],"51. Brown":[8,11],"51. Good Morning":[9,10],"52. Good afternoon":[8,11],"53. Good evening":[11,11],"56. Grey":[10,10],"66. Brother":[10,10],"77. Boy":[9,10],"78. long":[7,9],"79. short":[9,9],"80. tall":[7,9],"81. wide":[3,9],"82. narrow":[9,9],"83. big large":[6,9],"84. small little":[4,9],"85. slow":[9,9],"86. fast":[2,9],"87. hot":[8,9],"88. cold":[8,8],"89. warm":[9,9],"90. cool":[9,9],"91. new":[9,9],"92. old":[9,9],"93. young":[9,9],"94. good":[8,9],"95. bad":[9,9],"96. wet":[4,9],"97. dry":[7,9],"98. sick":[9,9],"99. healthy":[9,9]}},"isl_vit_tiny_lm_int4_s2.pt":{"source":"runs/f16_262w_lm_s2_qat4/masked50_clean.json","top1":0.8581,"words":{"1. loud":[9,9],"2. quiet":[9,9],"3. happy":[9,9],"37. Hat":[10,10],"38. Dress":[6,10],"40. I":[6,9],"41. Shirt":[10,10],"41. you":[6,6],"42. T-Shirt":[9,10],"42. he":[10,10],"43. Pant":[10,10],"43. she":[10,10],"44. Shoes":[8,10],"44. it":[10,10],"45. Pocket":[9,10],"45. we":[10,10],"46. you (plural)":[10,10],"47. they":[10,10],"48. Hello":[10,10],"49. How are you":[8,10],"50. Alright":[9,10],"51. Brown":[10,11],"51. Good Morning":[7,10],"52. Good afternoon":[10,11],"53. Good evening":[11,11],"56. Grey":[9,10],"66. Brother":[10,10],"77. Boy":[7,10],"78. long":[8,9],"79. short":[8,9],"80. tall":[7,9],"81. wide":[3,9],"82. narrow":[9,9],"83. big large":[6,9],"84. small little":[4,9],"85. slow":[6,9],"86. fast":[5,9],"87. hot":[4,9],"88. cold":[8,8],"89. warm":[9,9],"90. cool":[9,9],"91. new":[7,9],"92. old":[6,9],"93. young":[9,9],"94. good":[3,9],"95. bad":[9,9],"96. wet":[6,9],"97. dry":[9,9],"98. sick":[9,9],"99. healthy":[9,9]}}},"gate":{"source":"runs/gate_mixed.json","runs":["runs/f16_clean_s0","runs/f16_clean_s1","runs/f32_clean_s0"],"curve":[{"threshold":0.0,"coverage":1.0,"accuracy":0.75,"yield":0.75},{"threshold":0.2,"coverage":0.964,"accuracy":0.7648,"yield":0.7373},{"threshold":0.3,"coverage":0.8602,"accuracy":0.8276,"yield":0.7119},{"threshold":0.4,"coverage":0.7521,"accuracy":0.8873,"yield":0.6674},{"threshold":0.5,"coverage":0.6716,"accuracy":0.9243,"yield":0.6208},{"threshold":0.6,"coverage":0.5678,"accuracy":0.9552,"yield":0.5424},{"threshold":0.7,"coverage":0.4852,"accuracy":0.9563,"yield":0.464},{"threshold":0.8,"coverage":0.3581,"accuracy":0.9704,"yield":0.3475},{"threshold":0.9,"coverage":0.1547,"accuracy":0.9589,"yield":0.1483}]}};
  // Look-alike groups from the validation error analysis (commit 42080f0): about
  // 70 % of the remaining errors fall inside these.
  DATA.lookalikes = [
    ["41. you", "46. you (plural)"],
    ["83. big large", "81. wide", "82. narrow"],
    ["79. short", "84. small little"],
    ["94. good", "95. bad", "96. wet"],
    ["38. Dress", "42. T-Shirt"],
  ];
  // Genuine detection rates, report section 11.11.
  DATA.reference = { include: { left: 0.911, right: 0.852, face: 0.999 }, cislr: { left: 0.254, right: 0.496 } };
  App.DATA = DATA;

  const SCREENS = {
    capture: { title: "Capture", sub: "Sign one word. The lens clears when the model is sure of it." },
    notsure: { title: "Not sure", sub: "Clips the model wouldn't commit to, with its shortlist." },
    sentence: { title: "Sentence", sub: "Accepted words, in the order they were signed." },
    settings: { title: "Settings", sub: "Each control trades one measured quantity for another." },
    diagnostics: { title: "Detection", sub: "Where the hand and face crops came from, clip by clip." },
    vocabulary: { title: "Vocabulary", sub: "How reliably each word is read on sessions the model never saw." },
    enrolment: { title: "Enrolment", sub: "Record new signers, three takes per word." },
  };
  const ORDER = Object.keys(SCREENS);

  /* ------------------------------------------------------------------ utils -- */
  const ICONS = {
    camera: '<path d="M4 8.5h3.2L9 6.2h6l1.8 2.3H20V19H4z"/><circle cx="12" cy="13.4" r="3.4"/>',
    upload: '<path d="M12 15V4.5m0 0L8 8.5m4-4 4 4"/><path d="M5 14v5.5h14V14"/>',
    record: '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="3.3" fill="currentColor" stroke="none"/>',
    stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="2.2"/>',
    check: '<path d="M5 12.5l4.2 4.2L19 7"/>',
    x: '<path d="M6.5 6.5l11 11m0-11-11 11"/>',
    undo: '<path d="M9 7 5 11l4 4"/><path d="M5 11h9.5a4.5 4.5 0 0 1 0 9H12"/>',
    speak: '<path d="M4.5 10v4h3.2l4.3 3.6V6.4L7.7 10z"/><path d="M15.3 9.2a4 4 0 0 1 0 5.6M17.8 6.6a7.6 7.6 0 0 1 0 10.8"/>',
    copy: '<rect x="8.5" y="8.5" width="10.5" height="10.5" rx="2"/><path d="M15.5 8.5V6a1 1 0 0 0-1-1H6a1 1 0 0 0-1 1v8.5a1 1 0 0 0 1 1h2.5"/>',
    download: '<path d="M12 4.5v10.5m0 0 4-4m-4 4-4-4"/><path d="M5 17v3h14v-3"/>',
    question: '<circle cx="12" cy="12" r="8.5"/><path d="M9.6 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1.1.9-1.1 1.6v.5"/><circle cx="12" cy="16.8" r=".7" fill="currentColor" stroke="none"/>',
    "arrow-right": '<path d="M5 12h14m-5-5 5 5-5 5"/>',
    hand: '<path d="M8.5 12.5V6.3a1.4 1.4 0 0 1 2.8 0v5m0-5.8V4.9a1.4 1.4 0 0 1 2.8 0v6.4m0-5a1.4 1.4 0 0 1 2.8 0v7.2c0 4.2-2.6 7-6.4 7-2.7 0-4.2-1.4-5.3-3.8L4 12.8a1.3 1.3 0 0 1 2.3-1.2l2.2 2.9"/>',
    face: '<circle cx="12" cy="12" r="8.5"/><path d="M9.2 10h.01M14.8 10h.01"/><path d="M8.8 14.3a4 4 0 0 0 6.4 0"/>',
    sliders: '<path d="M4.5 7.5h9m4 0h2M4.5 16.5h2.5m4 0h8.5"/><circle cx="15.5" cy="7.5" r="2"/><circle cx="9" cy="16.5" r="2"/>',
    grid: '<rect x="4.5" y="4.5" width="6" height="6" rx="1.5"/><rect x="13.5" y="4.5" width="6" height="6" rx="1.5"/><rect x="4.5" y="13.5" width="6" height="6" rx="1.5"/><rect x="13.5" y="13.5" width="6" height="6" rx="1.5"/>',
    user: '<circle cx="12" cy="8.5" r="3.5"/><path d="M5 19.5c1.2-3.3 3.8-5 7-5s5.8 1.7 7 5"/>',
    "user-plus": '<circle cx="10" cy="8.5" r="3.5"/><path d="M3.5 19.5c1.1-3.2 3.5-5 6.5-5 1.6 0 3 .5 4.1 1.4"/><path d="M18 13.5v6m-3-3h6"/>',
    refresh: '<path d="M19 12a7 7 0 1 1-2.1-5"/><path d="M19.2 4.5V9h-4.5"/>',
    alert: '<path d="M12 4.2 3.6 19h16.8z"/><path d="M12 10v4"/><circle cx="12" cy="16.6" r=".7" fill="currentColor" stroke="none"/>',
    lens: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.4"/><path d="M9.3 9.7a3.2 3.2 0 0 1 2.3-1.3"/>',
    clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    trash: '<path d="M5 7h14M10 7V5h4v2M7 7l1 12.5h8L17 7"/>',
    "chevron-right": '<path d="m10 6 6 6-6 6"/>',
    play: '<path d="M8 5.5v13l10-6.5z"/>',
    activity: '<path d="M3 12h4l2.5-6.5 5 13 2.5-6.5h4"/>',
    text: '<path d="M5 7h14M5 12h14M5 17h9"/>',
  };
  const SYNONYMS = { "big large": "big / large", "small little": "small / little" };

  const U = {
    $: (selector, root = document) => root.querySelector(selector),
    $$: (selector, root = document) => Array.from(root.querySelectorAll(selector)),
    esc: (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])),
    clean(label) {
      const word = String(label ?? "").replace(/^\s*\d+\.\s*/, "");
      return SYNONYMS[word.toLowerCase()] || word;
    },
    pct: (x, digits = 0) => (x == null || Number.isNaN(+x) ? "—" : `${(x * 100).toFixed(digits)}%`),
    time: (date) => new Date(date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }),
    provbar(p) {
      if (!p) return '<div class="provbar"></div>';
      const part = (cls, v) => `<i class="${cls}" style="width:${(Math.max(0, v || 0) * 100).toFixed(1)}%"></i>`;
      const label = `direct ${U.pct(p.direct)}, ROI-rescued ${U.pct(p.roi)}, interpolated ${U.pct(p.interp)}, missing ${U.pct(p.missing)}`;
      return `<div class="provbar" role="img" aria-label="${label}">${part("d", p.direct)}${part("r", p.roi)}${part("p", p.interp)}${part("m", p.missing)}</div>`;
    },
    icon: (name, cls = "") =>
      `<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`,
  };
  App.U = U;
  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let nextId = 1;
  const newId = () => `c${Date.now().toString(36)}${(nextId++).toString(36)}`;

  /* ------------------------------------------------------------------ events -- */
  const listeners = {};
  App.on = (event, fn) => { (listeners[event] ||= []).push(fn); };
  App.emit = (event, payload) => {
    for (const fn of listeners[event] || []) {
      try { fn(payload); } catch (error) { console.error(`[${event}]`, error); }
    }
  };

  /* ------------------------------------------------------------------ state -- */
  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private window */ } },
  };
  const S = (App.S = {
    mode: "live",
    info: null,
    settings: Object.assign({ tau: 0.4, views: 6, floor: 0.25 }, store.get("isl.settings", {})),
    clips: [],
    pending: [],
    sentence: [],
    corrections: 0,
    enrol: { signer: store.get("isl.signer", ""), word: null, counts: {} },
    busy: false,
  });
  let current = null;
  let previewNext = "sure";

  App.settings = {
    set(partial) {
      Object.assign(S.settings, partial);
      store.set("isl.settings", S.settings);
      renderChrome();
      App.emit("settings:change", S.settings);
      updateBadges();
    },
  };

  App.isAnswered = (clip) => !!clip && clip.res.confidence >= S.settings.tau;
  App.typing = () => {
    const el = document.activeElement;
    if (!el) return false;
    if (el.isContentEditable || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return true;
    return el.tagName === "INPUT" && !["range", "checkbox", "radio", "button"].includes(el.type);
  };
  App.active = () => current;

  /* ------------------------------------------------------------------ vocabulary -- */
  let deployedSet = null;
  App.deployed = () => {
    const classes = S.info?.classes || [];
    const have = new Set(classes);
    if (classes.length > DATA.deployed.length && DATA.deployed.every((w) => have.has(w))) return DATA.deployed.slice();
    return classes.slice();
  };
  App.inDeployed = (label) => {
    if (!deployedSet) deployedSet = new Set(App.deployed());
    return (S.info?.classes || []).length <= DATA.deployed.length || deployedSet.has(label);
  };
  App.lookalikes = (label) => {
    const group = DATA.lookalikes.find((g) => g.includes(label));
    return group ? group.filter((w) => w !== label) : [];
  };
  let recallCache;
  App.recall = () => {
    if (recallCache !== undefined) return recallCache;
    const info = S.info || {};
    if (Array.isArray(info.per_class) && info.per_class.length) {
      const rows = info.per_class.filter((c) => c.support).map((c) => ({
        label: c.word, word: U.clean(c.word), support: c.support,
        hits: Math.round(c.recall * c.support), recall: c.recall,
      }));
      const hits = rows.reduce((a, r) => a + r.hits, 0), n = rows.reduce((a, r) => a + r.support, 0);
      return (recallCache = { rows, source: `per_class.json for ${info.run}`, top1: n ? hits / n : null });
    }
    // Release files carry no per-class file, but each was packed from a QAT run
    // whose held-out scoring is in the repository. Preview shows seed 0's.
    const key = S.mode === "preview" ? "isl_vit_tiny_lm_int4_s0.pt" : info.run;
    const data = DATA.recall[key];
    if (!data) return (recallCache = null);
    const rows = Object.entries(data.words).map(([label, [hits, support]]) => ({
      label, word: U.clean(label), hits, support, recall: hits / support,
    }));
    return (recallCache = { rows, source: data.source, top1: data.top1 });
  };

  /* ------------------------------------------------------------------ server -- */
  async function readJson(response) { try { return await response.json(); } catch { return {}; } }
  function networkError(error) {
    return error instanceof TypeError
      ? new Error("Can't reach the model server. Check that python -m islvit.serve is still running.")
      : error;
  }

  App.predict = async function (blob, filename = "clip.webm") {
    if (S.busy) throw new Error("Still reading the previous clip. Wait for it to finish.");
    S.busy = true;
    App.emit("busy", true);
    const started = performance.now();
    try {
      let res;
      if (S.mode === "preview") {
        res = await simulate();
      } else {
        const response = await fetch("/predict", {
          method: "POST", body: blob,
          headers: { "X-Filename": filename, "X-TTA": S.settings.views === 6 ? "1" : "0" },
        }).catch((e) => { throw networkError(e); });
        const body = await readJson(response);
        if (!response.ok) throw new Error(body.error || `The server couldn't read that clip (HTTP ${response.status}).`);
        res = body;
      }
      const clip = { id: newId(), at: new Date(), name: filename, elapsed: (performance.now() - started) / 1000, res, status: "open", chosen: null };
      S.clips.push(clip);
      if (!App.isAnswered(clip)) S.pending.push(clip.id);
      // Busy ends before the clip is announced, so screens render the result
      // rather than a reading state.
      S.busy = false;
      App.emit("busy", false);
      App.emit("clip:new", clip);
      updateBadges();
      return clip;
    } finally {
      if (S.busy) {
        S.busy = false;
        App.emit("busy", false);
      }
    }
  };

  function resolvePending(clip, status, chosen) {
    S.pending = S.pending.filter((id) => id !== clip.id);
    clip.status = status;
    clip.chosen = chosen ?? null;
    App.emit("clip:update", clip);
    updateBadges();
  }
  App.accept = (clip) => {
    const label = clip.res.candidates[0].word;
    resolvePending(clip, "accepted", label);
    App.addWord(label, clip.id);
  };
  App.correct = async (clip, label) => {
    if (S.mode === "live") {
      const response = await fetch("/correct", {
        method: "POST",
        body: JSON.stringify({
          chosen: label, predicted: clip.res.candidates[0].word, confidence: clip.res.confidence,
          detection_rate: clip.res.detection_rate, candidates: clip.res.candidates, run: S.info.run,
        }),
      }).catch((e) => { throw networkError(e); });
      if (!response.ok) throw new Error("The correction wasn't saved. Check the server console.");
    }
    S.corrections += 1;
    resolvePending(clip, "corrected", label);
    App.addWord(label, clip.id);
    return { saved: S.mode === "live" };
  };
  App.dismiss = (clip) => resolvePending(clip, "dismissed", null);

  App.addWord = (label, clipId = null) => {
    S.sentence.push({ id: newId(), label, at: new Date(), clipId });
    App.emit("sentence:change", S.sentence);
  };
  App.removeWord = (id) => { S.sentence = S.sentence.filter((w) => w.id !== id); App.emit("sentence:change", S.sentence); };
  App.moveWord = (from, to) => {
    if (from === to || from < 0 || to < 0 || from >= S.sentence.length || to >= S.sentence.length) return;
    const [word] = S.sentence.splice(from, 1);
    S.sentence.splice(to, 0, word);
    App.emit("sentence:change", S.sentence);
  };
  App.undoWord = () => { S.sentence.pop(); App.emit("sentence:change", S.sentence); };
  App.clearSentence = () => { S.sentence = []; App.emit("sentence:change", S.sentence); };

  App.enrolTake = async (blob, signer, label) => {
    // Header values must be plain ASCII; the server sanitises again on its side.
    const safeSigner = String(signer || "").replace(/[^A-Za-z0-9 _.-]/g, "").trim();
    if (!safeSigner) throw new Error("Enter a signer name using letters or numbers.");
    let result;
    if (S.mode === "live") {
      const response = await fetch("/enrol", {
        method: "POST", body: blob,
        headers: { "X-Filename": "take.webm", "X-Signer": safeSigner, "X-Word": label },
      }).catch((e) => { throw networkError(e); });
      result = await readJson(response);
      if (!response.ok) throw new Error(result.error || `The take wasn't saved (HTTP ${response.status}).`);
    } else {
      await delay(450);
      const take = (S.enrol.counts[label] || 0) + 1;
      result = { saved: `custom/${safeSigner}/${label}/take${take}.webm`, take, takes_for_word: take, preview: true };
    }
    S.enrol.counts[label] = result.takes_for_word;
    App.emit("enrol:change", S.enrol);
    updateBadges();
    return result;
  };

  /* ------------------------------------------------------------------ media -- */
  App.camera = {
    async start(video) {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser can't open a camera on this page. Use Chrome or Edge at http://127.0.0.1.");
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
        if (video) {
          video.srcObject = stream;
          video.muted = true;
          video.playsInline = true;
          await video.play().catch(() => {});
        }
        return stream;
      } catch (error) {
        if (error.name === "NotAllowedError") throw new Error("Camera access is blocked. Allow it from the address bar, then try again.");
        if (error.name === "NotFoundError") throw new Error("No camera was found. Connect one, or choose a video file instead.");
        throw new Error(`The camera didn't start: ${error.message}`);
      }
    },
    stop(stream) { stream?.getTracks?.().forEach((track) => track.stop()); },
  };

  App.record = async (stream, options = {}) => {
    if (!stream) throw new Error("Turn the camera on first.");
    if (typeof MediaRecorder === "undefined") throw new Error("This browser can't record video. Use Chrome or Edge.");
    const seconds = options.seconds ?? 3;
    for (let n = options.countdown ?? 3; n > 0; n--) { options.onCountdown?.(n); await delay(1000); }
    options.onCountdown?.(0);
    const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
    const recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
    const chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    const stopped = new Promise((resolve) => { recorder.onstop = resolve; });
    recorder.start(100);
    const started = performance.now();
    await new Promise((resolve) => {
      const tick = () => {
        const elapsed = (performance.now() - started) / 1000;
        options.onTick?.(Math.min(elapsed, seconds), seconds);
        if (elapsed >= seconds) resolve(); else requestAnimationFrame(tick);
      };
      tick();
    });
    recorder.stop();
    await stopped;
    return new Blob(chunks, { type: "video/webm" });
  };

  /* ------------------------------------------------------------------ preview -- */
  // Plausible responses in the server's exact shape, so every screen state can be
  // reviewed without the model. Clearly labelled as simulated in the interface.
  function provenance(direct, roi, interp) {
    const missing = Math.max(0, 1 - direct - roi - interp);
    return { direct, roi, interp, missing, genuine: direct + roi };
  }
  function candidates(top, rest) {
    const pool = DATA.deployed.filter((w) => w !== top[0] && !rest.some((r) => r[0] === w));
    const out = [[top[0], top[1]], ...rest];
    while (out.length < 5) out.push([pool.splice(Math.floor(Math.random() * pool.length), 1)[0], 0.01 + Math.random() * 0.02]);
    return out.sort((a, b) => b[1] - a[1]).map(([word, p]) => ({ word, p }));
  }
  const pick = (list) => list[Math.floor(Math.random() * list.length)];
  async function simulate() {
    await delay(1400 + Math.random() * 900);
    const scenario = previewNext;
    const recall = App.recall();
    const reliable = recall ? recall.rows.filter((r) => r.recall >= 0.9).map((r) => r.label) : DATA.deployed;
    let res;
    if (scenario === "unsure") {
      const group = pick(DATA.lookalikes.filter((g) => g.length >= 2));
      const p1 = 0.27 + Math.random() * 0.08, p2 = 0.18 + Math.random() * 0.07;
      const rest = [[group[1], p2]];
      if (group[2]) rest.push([group[2], 0.09 + Math.random() * 0.05]);
      res = { candidates: candidates([group[0], p1], rest),
        provenance: { left_hand: provenance(0.66, 0.12, 0.2), right_hand: provenance(0.6, 0.15, 0.22), face: provenance(0.97, 0, 0.03) } };
    } else if (scenario === "poor") {
      res = { candidates: candidates([pick(DATA.deployed), 0.22 + Math.random() * 0.06], [[pick(DATA.deployed), 0.12]]),
        provenance: { left_hand: provenance(0.1, 0.06, 0.62), right_hand: provenance(0.16, 0.06, 0.6), face: provenance(0.91, 0, 0.09) } };
    } else {
      const top = pick(reliable);
      res = { candidates: candidates([top, 0.8 + Math.random() * 0.14], [[pick(DATA.deployed.filter((w) => w !== top)), 0.04]]),
        provenance: { left_hand: provenance(0.81, 0.09, 0.1), right_hand: provenance(0.72, 0.13, 0.15), face: provenance(1, 0, 0) } };
    }
    res.frames = 32;
    res.views = S.settings.views;
    res.confidence = res.candidates[0].p;
    res.detection_rate = (res.provenance.left_hand.genuine + res.provenance.right_hand.genuine) / 2;
    return res;
  }

  /* ------------------------------------------------------------------ toasts -- */
  App.toast = (message, options = {}) => {
    const host = U.$("#toasts");
    const el = document.createElement("div");
    el.className = `toast tone-${options.tone || "info"}`;
    el.innerHTML = `<span class="toast-dot"></span><span class="toast-msg">${U.esc(message)}</span>`;
    if (options.action) {
      const button = document.createElement("button");
      button.className = "btn sm";
      button.textContent = options.action.label;
      button.addEventListener("click", () => { options.action.run(); close(); });
      el.append(button);
    }
    host.append(el);
    requestAnimationFrame(() => el.classList.add("in"));
    const close = () => { el.classList.remove("in"); setTimeout(() => el.remove(), 400); };
    setTimeout(close, options.action ? 6500 : 4200);
  };

  /* ------------------------------------------------------------------ routing -- */
  App.go = (name, opts) => {
    if (!SCREENS[name]) return;
    if (current && current !== name) {
      try { App.screens[current]?.onHide?.(); } catch (error) { console.error(error); }
    }
    current = name;
    for (const section of U.$$(".screen")) section.classList.toggle("on", section.dataset.screen === name);
    for (const button of U.$$("#nav [data-go]")) {
      if (button.dataset.go === name) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    }
    U.$("#screen-title").textContent = SCREENS[name].title;
    U.$("#screen-sub").textContent = SCREENS[name].sub;
    document.title = `${SCREENS[name].title} · ISL Reader`;
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    movePill();
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    try { App.screens[name]?.onShow?.(opts); } catch (error) { console.error(error); }
    choreograph(U.$(`.screen[data-screen="${name}"]`));
    App.emit("screen:show", name);
  };

  /* Arrival order for a screen that has just been shown (see "arrival" in base.css):
     outermost panels first, in document order, then the items of its lists. Runs after
     onShow so lists the screen re-rendered are the ones that animate. */
  const STAGGERED_LISTS = [".cap-alt-list", ".cap-samples-strip", ".cap-tiles", ".dg-rows", ".ns-tiles", ".ns-queue-list", ".sn-timeline",
    ".vb-stats", ".vb-grid", ".en-queue", ".en-steps", ".st-facts"];
  function choreograph(section) {
    if (!section || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const titles = U.$(".topbar-titles");
    const panels = U.$$(".glass, .note, .cap-controls, .en-controls, .dg-bar, .vb-controls", section)
      .filter((el) => el.offsetParent !== null && !el.parentElement.closest(".glass, .note"));
    const items = U.$$(STAGGERED_LISTS.map((list) => `${list} > *`).join(", "), section).filter((el) => el.offsetParent !== null);
    for (const el of [...panels, ...items, titles]) el?.classList.remove("enter", "enter-item", "swap");
    void section.offsetWidth;   // restart the animations even when the classes were already there
    titles?.classList.add("swap");
    panels.forEach((el, i) => { el.style.setProperty("--enter", `${Math.min(i, 9) * 55}ms`); el.classList.add("enter"); });
    // List items start once their panel is mostly in, and cascade 28 ms apart (capped at ~0.7 s).
    const lists = new Map();
    for (const el of items) {
      const parent = el.parentElement;
      const i = lists.get(parent) ?? 0;
      lists.set(parent, i + 1);
      el.style.setProperty("--enter", `${180 + Math.min(i, 18) * 28}ms`);
      el.classList.add("enter-item");
    }
  }

  function movePill() {
    const active = U.$("#nav [aria-current=page]");
    const pill = U.$("#nav-pill");
    if (!active || !pill) return;
    pill.style.setProperty("--x", `${active.offsetLeft}px`);
    pill.style.setProperty("--y", `${active.offsetTop}px`);
    pill.style.setProperty("--w", `${active.offsetWidth}px`);
    pill.style.setProperty("--h", `${active.offsetHeight}px`);
  }

  /* ------------------------------------------------------------------ chrome -- */
  function renderChrome() {
    U.$("#tau-text").textContent = S.settings.tau.toFixed(2);
    U.$("#views-text").textContent = String(S.settings.views);
    const pill = U.$("#mode-pill");
    pill.classList.toggle("live", S.mode === "live");
    pill.classList.toggle("preview", S.mode === "preview");
    U.$("#mode-text").textContent = S.mode === "live" ? `Live · ${(S.info?.device || "cpu").toUpperCase()}` : "Preview";
  }

  function renderRail() {
    const info = S.info;
    U.$("#rail-run").textContent = S.mode === "preview" ? "No model connected" : info.run;
    U.$("#rail-run").title = info.run;
    const words = info.classes.length;
    const deployed = App.deployed().length;
    U.$("#rail-spec").innerHTML = S.mode === "preview"
      ? "Simulated results. Start the server to read real clips."
      : `${words} words${deployed < words ? ` · ${deployed} deployed` : ""}<br>${info.n_frames} frames · ${info.img_size} px crops<br>${(+info.params_m).toFixed(2)} M parameters`;
  }

  function updateBadges() {
    const badge = U.$("#badge-notsure");
    badge.textContent = String(S.pending.length);
    badge.classList.toggle("on", S.pending.length > 0);
    const dot = U.$("#dot-diagnostics");
    const last = S.clips[S.clips.length - 1];
    dot.className = "nav-dot";
    if (last) {
      const worst = Math.min(last.res.provenance.left_hand.genuine, last.res.provenance.right_hand.genuine);
      dot.classList.add("on", worst < S.settings.floor ? "coral" : worst < 0.6 ? "amber" : "mint");
    }
  }

  /* ------------------------------------------------------------------ interaction --
     A specular highlight that follows the pointer over [data-glare] glass, and a
     slight perspective tilt on [data-tilt]. Both are skipped for coarse pointers
     and for people who ask for reduced motion. */
  function wireInteraction() {
    const fine = matchMedia("(pointer: fine)").matches;
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let tilted = null;
    document.addEventListener("pointermove", (event) => {
      // The light follows the pointer on whichever panel is under it, not only [data-glare] ones.
      const glare = event.target.closest?.("[data-glare], .screen .glass");
      if (glare) {
        const box = glare.getBoundingClientRect();
        glare.style.setProperty("--mx", `${event.clientX - box.left}px`);
        glare.style.setProperty("--my", `${event.clientY - box.top}px`);
      }
      if (!fine || calm) return;
      const tilt = event.target.closest?.("[data-tilt]");
      if (tilted && tilted !== tilt) { tilted.style.setProperty("--tilt-x", "0deg"); tilted.style.setProperty("--tilt-y", "0deg"); }
      tilted = tilt;
      if (tilt) {
        const box = tilt.getBoundingClientRect();
        const dx = (event.clientX - box.left) / box.width - 0.5;
        const dy = (event.clientY - box.top) / box.height - 0.5;
        tilt.style.setProperty("--tilt-x", `${(-dy * 6).toFixed(2)}deg`);
        tilt.style.setProperty("--tilt-y", `${(dx * 6).toFixed(2)}deg`);
      }
    }, { passive: true });
    document.addEventListener("pointerleave", () => {
      if (tilted) { tilted.style.setProperty("--tilt-x", "0deg"); tilted.style.setProperty("--tilt-y", "0deg"); tilted = null; }
    });

    document.addEventListener("keydown", (event) => {
      if (event.altKey && /^[1-7]$/.test(event.key)) {
        event.preventDefault();
        App.go(ORDER[+event.key - 1]);
        return;
      }
      if (event.key === "?" && !App.typing()) {
        event.preventDefault();
        const help = U.$("#help");
        help.open ? help.close() : help.showModal();
      }
    });
    window.addEventListener("resize", movePill);
  }

  function wireChrome() {
    for (const holder of U.$$("[data-icon]")) holder.outerHTML = U.icon(holder.dataset.icon);
    // Brand mark: the 21-tick ring the capture lens uses, in miniature.
    const ticks = U.$("#brand-ticks");
    if (ticks) {
      let lines = "";
      for (let i = 0; i < 21; i++) {
        const a = (135 + (270 * i) / 20) * Math.PI / 180;
        const r1 = 15.2, r2 = i % 5 === 0 ? 18.6 : 17.4;
        lines += `<line x1="${(20 + r1 * Math.cos(a)).toFixed(2)}" y1="${(20 + r1 * Math.sin(a)).toFixed(2)}" x2="${(20 + r2 * Math.cos(a)).toFixed(2)}" y2="${(20 + r2 * Math.sin(a)).toFixed(2)}" opacity="${i < 15 ? 1 : 0.3}"/>`;
      }
      ticks.innerHTML = lines;
    }
    U.$("#nav").addEventListener("click", (event) => {
      const button = event.target.closest("[data-go]");
      if (button) App.go(button.dataset.go);
    });
    U.$("#tau-chip").addEventListener("click", () => App.go("settings"));
    U.$("#views-chip").addEventListener("click", () => App.go("settings"));
    U.$("#help-btn").addEventListener("click", () => U.$("#help").showModal());
    U.$("#preview-seg").addEventListener("click", (event) => {
      const button = event.target.closest("[data-scenario]");
      if (!button) return;
      previewNext = button.dataset.scenario;
      for (const b of U.$$("#preview-seg button")) b.setAttribute("aria-pressed", String(b === button));
    });
    U.$("#preview-run").addEventListener("click", async () => {
      App.go("capture");
      try { await App.predict(new Blob(["preview"], { type: "video/webm" }), "sample.webm"); }
      catch (error) { App.toast(error.message, { tone: "coral" }); }
    });
  }

  /* ------------------------------------------------------------------ boot -- */
  async function loadInfo() {
    const forced = new URLSearchParams(location.search).has("preview");
    if (location.protocol !== "file:" && !forced) {
      try {
        const response = await fetch("/info", { cache: "no-store" });
        if (response.ok) { S.mode = "live"; return await response.json(); }
      } catch { /* fall through to preview */ }
    }
    S.mode = "preview";
    return { run: "preview", classes: DATA.deployed.slice(), n_frames: 16, img_size: 64, params_m: 3.89, device: "preview", per_class: null, tiers: [] };
  }

  App.boot = async function () {
    wireChrome();
    wireInteraction();
    S.info = await loadInfo();
    deployedSet = null;
    recallCache = undefined;
    U.$("#previewbar").classList.toggle("on", S.mode === "preview");
    renderChrome();
    renderRail();
    for (const name of ORDER) {
      const root = U.$(`#screen-${name}`);
      try { App.screens[name]?.init?.(root); } catch (error) {
        console.error(`screen ${name} failed to start`, error);
        root.insertAdjacentHTML("afterbegin", `<div class="note coral">This screen failed to load: ${U.esc(error.message)}</div>`);
      }
    }
    const start = location.hash.slice(1);
    App.go(SCREENS[start] ? start : "capture");
    updateBadges();
    requestAnimationFrame(movePill);
  };
})();
