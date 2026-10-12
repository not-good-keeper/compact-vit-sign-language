/* Capture: record or choose one clip, read it, and show the answer through the lens.

   The lens has four states. idle and reading are about the app; sure and unsure
   are about the model, decided by the same rule the rest of the app uses:
   confidence >= τ. The ring has 21 ticks -- one per tracked hand joint -- lit in
   proportion to confidence, with a notch where τ sits. */
(function () {
  const { $, $$, esc, clean, pct, icon, provbar } = App.U;
  const TICKS = 21;
  const ARC_START = 135, ARC_SPAN = 270;  // degrees, clockwise from +x
  let root, refs, stream = null, fileUrl = null, timer = null, busySince = 0;

  const ref = (name) => root.querySelector(`[data-ref="${name}"]`);
  const latest = () => App.S.clips[App.S.clips.length - 1] || null;

  function polar(r, degrees) {
    const a = degrees * Math.PI / 180;
    return [110 + r * Math.cos(a), 110 + r * Math.sin(a)];
  }

  function drawRing() {
    const [x0, y0] = polar(92, ARC_START), [x1, y1] = polar(92, ARC_START + ARC_SPAN);
    let svg = `<title>21 ticks, one per tracked hand joint</title>
      <path class="track" d="M ${x0.toFixed(2)} ${y0.toFixed(2)} A 92 92 0 1 1 ${x1.toFixed(2)} ${y1.toFixed(2)}"/>`;
    for (let i = 0; i < TICKS; i++) {
      const angle = ARC_START + (ARC_SPAN * i) / (TICKS - 1);
      const major = i % 5 === 0;
      const [ax, ay] = polar(major ? 85 : 87, angle), [bx, by] = polar(99, angle);
      svg += `<line class="t${major ? " major" : ""}" style="--i:${i}" x1="${ax.toFixed(2)}" y1="${ay.toFixed(2)}" x2="${bx.toFixed(2)}" y2="${by.toFixed(2)}"/>`;
    }
    svg += `<g class="notch" data-ref="notch"><path d="M 0 -4 L 7 0 L 0 4 Z"/></g>`;
    refs.ring.innerHTML = svg;
    placeNotch();
  }

  function placeNotch() {
    const notch = refs.ring.querySelector(".notch");
    if (!notch) return;
    const angle = ARC_START + ARC_SPAN * App.S.settings.tau;
    const [x, y] = polar(106, angle);
    notch.setAttribute("transform", `translate(${x.toFixed(2)} ${y.toFixed(2)}) rotate(${(angle + 180).toFixed(1)})`);
  }

  function lightTicks(confidence) {
    const lit = Math.round(Math.max(0, Math.min(1, confidence)) * TICKS);
    $$(".t", refs.ring).forEach((tick, i) => tick.classList.toggle("on", i < lit));
  }

  function countUp(el, to) {
    const from = +(el.dataset.v || 0);
    el.dataset.v = to;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { el.textContent = (to * 100).toFixed(0); return; }
    const start = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - start) / 750);
      const eased = 1 - Math.pow(1 - k, 3);
      el.textContent = ((from + (to - from) * eased) * 100).toFixed(0);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ---------------------------------------------------------------- lens */
  function renderLens(clip) {
    const lens = refs.lens;
    if (App.S.busy) {
      lens.dataset.state = "reading";
      refs.kicker.textContent = "Reading the clip";
      refs.word.textContent = "…";
      refs.word.classList.remove("long");
      refs.guess.textContent = "";
      refs.conf.innerHTML = "";
      lightTicks(0);
      refs.actions.innerHTML = "";
      return;
    }
    if (!clip) {
      lens.dataset.state = "idle";
      refs.kicker.textContent = "Waiting for a sign";
      refs.word.textContent = "—";
      refs.guess.textContent = "";
      refs.conf.innerHTML = "";
      lightTicks(0);
      refs.actions.innerHTML = "";
      refs.warn.innerHTML = "";
      return;
    }
    const top = clip.res.candidates[0];
    const sure = App.isAnswered(clip);
    lens.dataset.state = sure ? "sure" : "unsure";
    refs.kicker.textContent = sure ? "Recognised" : "Not sure";
    const word = clean(top.word);
    refs.word.textContent = word;
    refs.word.classList.toggle("long", word.length > 10);
    refs.guess.textContent = sure ? "" : `best guess, below τ ${App.S.settings.tau.toFixed(2)}`;
    refs.conf.innerHTML = `<b data-ref="confNum">0</b>% confident`;
    countUp(refs.conf.querySelector("b"), clip.res.confidence);
    lightTicks(clip.res.confidence);
    renderActions(clip, sure);
    renderWarning(clip);
  }

  function renderActions(clip, sure) {
    const status = clip.status;
    let html = "";
    if (status === "accepted" || status === "corrected") {
      html = `<span class="done">${icon("check")}Added “${esc(clean(clip.chosen))}” to the sentence</span>`;
    } else if (status === "dismissed") {
      html = `<span class="done">${icon("refresh")}Marked to sign again</span>`;
    } else if (sure) {
      html = `<button class="btn primary" type="button" data-act="accept">${icon("check")}Accept <kbd>A</kbd></button>
              <button class="btn" type="button" data-act="wrong">Not this word <kbd>W</kbd></button>`;
    } else {
      html = `<button class="btn primary" type="button" data-act="shortlist">Open shortlist ${icon("arrow-right")}</button>
              <button class="btn ghost" type="button" data-act="again">Sign it again</button>`;
    }
    refs.actions.innerHTML = html;
  }

  function renderWarning(clip) {
    const rate = clip.res.detection_rate;
    const floor = App.S.settings.floor;
    if (rate < floor) {
      refs.warn.innerHTML = `<div class="note coral"><b>Hands were found in only ${pct(rate)} of frames.</b>
        This answer reflects the background more than the sign. Move into better light and keep both hands in the frame.</div>`;
    } else if (rate < 0.6) {
      refs.warn.innerHTML = `<div class="note amber"><b>Hands were found in ${pct(rate)} of frames.</b> Treat this answer with caution.</div>`;
    } else {
      refs.warn.innerHTML = "";
    }
  }

  /* ---------------------------------------------------------------- cards */
  function renderAlternatives(clip) {
    if (!clip) {
      refs.alts.innerHTML = `<li class="cap-placeholder">The next most likely words appear here after a clip is read.</li>`;
      return;
    }
    refs.alts.innerHTML = clip.res.candidates.slice(1).map((c) => `
      <li class="cap-alt">
        <span class="cap-alt-word">${esc(clean(c.word))}${App.inDeployed(c.word) ? "" : ' <span class="tag">not in the 50</span>'}</span>
        <span class="cap-alt-p">${pct(c.p, 1)}</span>
        <span class="cap-alt-bar"><i style="width:${Math.max(1, c.p * 100).toFixed(1)}%"></i></span>
      </li>`).join("");
  }

  function renderHealth(clip) {
    const streams = [["left_hand", "Left hand", "hand"], ["right_hand", "Right hand", "hand"], ["face", "Face", "face"]];
    const floor = App.S.settings.floor;
    refs.tiles.innerHTML = streams.map(([key, name, glyph]) => {
      const p = clip?.res.provenance[key];
      if (!p) {
        return `<div class="cap-tile"><div class="cap-tile-head"><span>${name}</span>${icon(glyph)}</div>
          <div class="cap-tile-val muted">—</div>${provbar(null)}<span class="tag">No clip yet</span></div>`;
      }
      const verdict = p.genuine >= 0.75 ? ["Healthy", "mint"] : p.genuine >= floor ? ["Patchy", "amber"] : ["Lost", "coral"];
      return `<div class="cap-tile"><div class="cap-tile-head"><span>${name}</span>${icon(glyph)}</div>
        <div class="cap-tile-val">${(p.genuine * 100).toFixed(0)}<small>%</small></div>
        ${provbar(p)}<span class="tag ${verdict[1]}">${verdict[0]}</span></div>`;
    }).join("");
  }

  function renderChips() {
    const words = App.S.sentence;
    refs.chips.innerHTML = words.length
      ? words.map((w) => `<span class="cap-chip">${esc(clean(w.label))}<button type="button" data-remove="${w.id}" aria-label="Remove ${esc(clean(w.label))}">${icon("x")}</button></span>`).join("")
      : `<span class="cap-placeholder" style="margin:0">Accepted words appear here.</span>`;
  }

  function renderStatus(clip) {
    if (App.S.busy) return;
    refs.status.textContent = clip ? `Read in ${clip.elapsed.toFixed(1)} s · ${clip.res.views} view${clip.res.views === 1 ? "" : "s"} · ${clip.res.frames} frames` : "";
  }

  function render(focus = false) {
    const clip = latest();
    renderLens(clip);
    if (focus && clip) {
      refs.lens.classList.remove("focusing");
      void refs.lens.offsetWidth;
      refs.lens.classList.add("focusing");
    }
    renderAlternatives(clip);
    renderHealth(clip);
    renderChips();
    renderStatus(clip);
    syncControls();
  }

  /* ---------------------------------------------------------------- input */
  function syncControls() {
    refs.camBtn.innerHTML = stream ? `${icon("stop")}Turn off camera` : `${icon("camera")}Turn on camera`;
    refs.recBtn.disabled = !stream || App.S.busy;
  }

  function showVideo(on) { refs.stage.classList.toggle("has-video", on); }

  async function toggleCamera() {
    if (stream) {
      App.camera.stop(stream);
      stream = null;
      refs.video.srcObject = null;
      showVideo(!!fileUrl);
      if (fileUrl) { refs.video.src = fileUrl; refs.video.play().catch(() => {}); }
      syncControls();
      return;
    }
    try {
      if (fileUrl) { refs.video.removeAttribute("src"); }
      stream = await App.camera.start(refs.video);
      showVideo(true);
    } catch (error) {
      App.toast(error.message, { tone: "coral" });
    }
    syncControls();
  }

  function previewFile(file) {
    if (stream) { App.camera.stop(stream); stream = null; refs.video.srcObject = null; }
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    fileUrl = URL.createObjectURL(file);
    refs.video.src = fileUrl;
    refs.video.loop = true;
    refs.video.play().catch(() => {});
    showVideo(true);
    syncControls();
  }

  async function read(blob, name) {
    try {
      await App.predict(blob, name);
    } catch (error) {
      App.toast(error.message, { tone: "coral" });
      render();
    }
  }

  /* ------------------------------------------------------------ demo clips --
     Held-out test videos kept on the server (islvit/demo.py serves /samples), so a demo
     needs no files on the presenting computer. A clip goes through exactly the same path
     as a chosen file: it plays in the stage and the model reads it. Each card then shows
     what the model said against the sign the clip actually is. */
  let samples = [];
  const verdicts = new Map();   // index -> { state, text }

  async function loadSamples() {
    if (App.S.mode !== "live") return;
    try {
      const response = await fetch("/samples", { cache: "no-store" });
      if (!response.ok) return;
      const manifest = await response.json();
      samples = manifest.samples || [];
      if (!samples.length) return;
      refs.samplesNote.textContent = "Recordings from sessions the model never trained on. Pick one: it plays here and the model reads it.";
      renderSamples();
      refs.samplesCard.hidden = false;
    } catch { /* no demo clips on this server */ }
  }

  function renderSamples() {
    refs.samples.innerHTML = samples.map((s, i) => {
      const v = verdicts.get(i) || { state: "idle", text: "Read this clip" };
      return `<button class="cap-sample" type="button" role="listitem" data-sample="${i}" data-state="${v.state}"
          aria-label="Demo clip: the sign for ${esc(s.word)}">
        <span class="cap-sample-thumb"><img src="/samples/${encodeURIComponent(s.poster)}" alt="" loading="lazy">
          <span class="cap-sample-play">${App.U.icon("play")}</span><span class="cap-sample-bar"></span></span>
        <span class="cap-sample-meta"><span class="cap-sample-word">${esc(s.word)}</span>
          ${s.hard_case ? '<span class="tag amber">look-alike</span>' : `<span class="tag">${esc(s.category)}</span>`}</span>
        <span class="cap-sample-verdict">${esc(v.text)}</span>
      </button>`;
    }).join("");
    const done = [...verdicts.values()].filter((v) => v.state !== "reading");
    const right = done.filter((v) => v.state === "right").length;
    refs.samplesScore.textContent = done.length ? `${done.length} read · ${right} right` : `${samples.length} clips`;
  }

  async function playSample(i) {
    const sample = samples[i];
    if (!sample || App.S.busy) return;
    verdicts.set(i, { state: "reading", text: "Reading…" });
    renderSamples();
    try {
      const response = await fetch(`/samples/${encodeURIComponent(sample.file)}`);
      if (!response.ok) throw new Error(`Could not load the clip (${response.status})`);
      const file = new File([await response.blob()], sample.file, { type: "video/mp4" });
      previewFile(file);
      const before = App.S.clips.length;
      await read(file, sample.file);
      const clip = App.S.clips.length > before ? App.S.clips[App.S.clips.length - 1] : null;
      if (!clip) throw new Error("The model did not return an answer");
      const top = clip.res.candidates[0];
      const p = `${Math.round(top.p * 100)}%`;
      const verdict = !App.isAnswered(clip)
        ? { state: "unsure", text: `Not sure · best guess ${clean(top.word)} ${p}` }
        : top.word === sample.label
          ? { state: "right", text: `✓ ${clean(top.word)} · ${p}` }
          : { state: "wrong", text: `✗ read as ${clean(top.word)}` };
      verdicts.set(i, verdict);
    } catch (error) {
      verdicts.set(i, { state: "wrong", text: error.message });
    }
    renderSamples();
  }

  async function record() {
    if (!stream || App.S.busy) return;
    refs.recBtn.disabled = true;
    const fill = ref("recFill");
    try {
      const blob = await App.record(stream, {
        seconds: 3,
        countdown: 3,
        onCountdown(n) {
          refs.count.hidden = n === 0;
          if (n) { refs.countNum.textContent = n; refs.countNum.style.animation = "none"; void refs.countNum.offsetWidth; refs.countNum.style.animation = ""; }
          if (n === 0) { refs.rec.hidden = false; refs.stage.classList.add("recording"); }
        },
        onTick(t, total) {
          fill.style.strokeDashoffset = (94.25 * (1 - t / total)).toFixed(2);
          refs.recText.textContent = `REC ${t.toFixed(1)} s`;
        },
      });
      refs.rec.hidden = true;
      refs.stage.classList.remove("recording");
      await read(blob, "clip.webm");
    } catch (error) {
      App.toast(error.message, { tone: "coral" });
    } finally {
      refs.count.hidden = true;
      refs.rec.hidden = true;
      refs.stage.classList.remove("recording");
      syncControls();
    }
  }

  function onBusy(busy) {
    refs.scan.hidden = !busy;
    clearInterval(timer);
    if (busy) {
      busySince = performance.now();
      const tick = () => { refs.status.textContent = `Reading the clip · ${((performance.now() - busySince) / 1000).toFixed(1)} s`; };
      tick();
      timer = setInterval(tick, 100);
      renderLens(null);
    }
    syncControls();
  }

  function act(name) {
    const clip = latest();
    switch (name) {
      case "camera": toggleCamera(); break;
      case "choose": refs.file.click(); break;
      case "record": record(); break;
      case "accept":
        if (clip && clip.status === "open") {
          App.accept(clip);
          App.toast(`Added “${clean(clip.chosen)}” to the sentence`, { tone: "mint", action: { label: "Undo", run: () => App.undoWord() } });
        }
        break;
      case "wrong": case "shortlist": if (clip) App.go("notsure", { clipId: clip.id }); break;
      case "again": if (clip) { App.dismiss(clip); } break;
      case "undo": App.undoWord(); break;
      case "clear": App.clearSentence(); break;
      case "sentence": App.go("sentence"); break;
    }
  }

  App.screens.capture = {
    init(section) {
      root = section;
      refs = {};
      for (const el of $$("[data-ref]", root)) refs[el.dataset.ref] = el;
      drawRing();

      root.addEventListener("click", (event) => {
        const remove = event.target.closest("[data-remove]");
        if (remove) { App.removeWord(remove.dataset.remove); return; }
        const button = event.target.closest("[data-act]");
        if (button && !button.disabled) act(button.dataset.act);
      });
      refs.samples.addEventListener("click", (event) => {
        const card = event.target.closest("[data-sample]");
        if (card) playSample(+card.dataset.sample);
      });
      loadSamples();
      refs.file.addEventListener("change", () => {
        const file = refs.file.files[0];
        if (file) { previewFile(file); read(file, file.name); }
        refs.file.value = "";
      });
      const stage = refs.stage;
      ["dragenter", "dragover"].forEach((type) => stage.addEventListener(type, (e) => { e.preventDefault(); stage.classList.add("drag"); }));
      ["dragleave", "drop"].forEach((type) => stage.addEventListener(type, (e) => { e.preventDefault(); stage.classList.remove("drag"); }));
      stage.addEventListener("drop", (e) => {
        const file = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith("video/") || /\.(mp4|mov|webm|avi|mkv)$/i.test(f.name));
        if (!file) { App.toast("Drop a video file: .mp4, .mov or .webm.", { tone: "coral" }); return; }
        previewFile(file);
        read(file, file.name);
      });

      document.addEventListener("keydown", (event) => {
        if (App.active() !== "capture" || App.typing() || event.altKey || event.ctrlKey || event.metaKey) return;
        const key = event.key.toLowerCase();
        if (event.code === "Space") { event.preventDefault(); record(); }
        else if (key === "a") act("accept");
        else if (key === "w") { const clip = latest(); if (clip && clip.status === "open") act("wrong"); }
      });

      App.on("busy", onBusy);
      App.on("clip:new", () => render(true));
      App.on("clip:update", () => render());
      App.on("sentence:change", renderChips);
      App.on("settings:change", () => { placeNotch(); render(); });
      render();
    },
    onShow() { render(); },
    onHide() {},
  };
})();
