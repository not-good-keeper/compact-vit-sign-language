/* Settings: each control trades one measured quantity for another.

   The threshold chart plots the only gating curve on disk (runs/gate_mixed.json,
   an earlier pixel-only ensemble) and says so; the handle and the slider move
   together and set App.S.settings.tau. */
(function () {
  const { $$, esc, clean, pct } = App.U;
  // The chart is drawn in real pixels (W follows the card's width) so its labels stay
  // 10.5 px at any window size instead of scaling up with a fixed viewBox.
  let W = 640;
  const H = 300, L = 46, R = 18, T = 18, B = 34;
  const MAX_TAU = 0.9;
  let root, refs, dragging = false;

  const curve = () => App.DATA.gate?.curve || [];
  const x = (tau) => L + (tau / MAX_TAU) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);

  function at(tau, key) {
    const c = curve();
    if (!c.length) return null;
    if (tau <= c[0].threshold) return c[0][key];
    for (let i = 1; i < c.length; i++) {
      if (tau <= c[i].threshold) {
        const a = c[i - 1], b = c[i];
        const k = (tau - a.threshold) / (b.threshold - a.threshold);
        return a[key] + k * (b[key] - a[key]);
      }
    }
    return c[c.length - 1][key];
  }

  function path(key) {
    return curve().map((p, i) => `${i ? "L" : "M"}${x(p.threshold).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
  }

  function drawChart() {
    if (!curve().length) { refs.svg.hidden = true; refs.chartEmpty.hidden = false; return; }
    const width = Math.round(refs.svg.getBoundingClientRect().width);
    if (width > 0) W = Math.max(420, width);
    refs.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    refs.svg.setAttribute("tabindex", "0");
    let grid = "";
    for (const v of [0.25, 0.5, 0.75, 1]) {
      grid += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/>
               <text class="axis-label" x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
    }
    for (let t = 0; t <= MAX_TAU + 1e-9; t += 0.1) {
      grid += `<text class="axis-label" x="${x(t)}" y="${H - 12}" text-anchor="middle">${t.toFixed(1)}</text>`;
    }
    const c = curve();
    const area = `${path("yield")} L${x(c[c.length - 1].threshold)},${y(0)} L${x(c[0].threshold)},${y(0)} Z`;
    const points = c.map((p) => `<circle class="pt" cx="${x(p.threshold)}" cy="${y(p.accuracy)}" r="3" stroke="var(--mint)"/>
      <circle class="pt" cx="${x(p.threshold)}" cy="${y(p.coverage)}" r="3" stroke="var(--sky)"/>`).join("");
    refs.svg.innerHTML = `${grid}
      <text class="axis-label" x="${(L + W - R) / 2}" y="${H - 1}" text-anchor="middle">threshold τ</text>
      <path class="yld" d="${area}"/>
      <path class="cov" d="${path("coverage")}"/>
      <path class="acc" d="${path("accuracy")}"/>
      ${points}
      <g data-ref="handle">
        <line class="handle-glow" y1="${T}" y2="${H - B}"/>
        <line class="handle-line" y1="${T}" y2="${H - B}"/>
        <circle class="knob" data-k="acc" r="6.5" fill="var(--mint)"/>
        <circle class="knob" data-k="cov" r="6.5" fill="var(--sky)"/>
        <rect class="tag-bg" y="${T - 14}" width="62" height="20" rx="10"/>
        <text class="tag-text" y="${T}" text-anchor="middle"></text>
      </g>`;
    refs.points.innerHTML = c.map((p) => `<tr data-t="${p.threshold}"><td>${p.threshold.toFixed(2)}</td><td>${pct(p.coverage, 1)}</td>
      <td>${pct(p.accuracy, 1)}</td><td>${pct(p.yield, 1)}</td></tr>`).join("");
  }

  function renderTau() {
    const tau = App.S.settings.tau;
    refs.tauRange.value = tau;
    refs.tauOut.textContent = tau.toFixed(2);
    refs.roTau.textContent = tau.toFixed(2);
    const cov = at(tau, "coverage"), acc = at(tau, "accuracy"), yld = at(tau, "yield");
    refs.roCov.textContent = pct(cov);
    refs.roAcc.textContent = pct(acc);
    refs.roYld.textContent = pct(yld);
    const handle = refs.svg.querySelector('[data-ref="handle"]');
    if (handle && cov != null) {
      const hx = x(Math.min(tau, MAX_TAU));
      for (const line of handle.querySelectorAll("line")) { line.setAttribute("x1", hx); line.setAttribute("x2", hx); }
      handle.querySelector('[data-k="acc"]').setAttribute("cx", hx);
      handle.querySelector('[data-k="acc"]').setAttribute("cy", y(acc));
      handle.querySelector('[data-k="cov"]').setAttribute("cx", hx);
      handle.querySelector('[data-k="cov"]').setAttribute("cy", y(cov));
      const tagX = Math.max(L + 31, Math.min(W - R - 31, hx));
      handle.querySelector(".tag-bg").setAttribute("x", tagX - 31);
      const text = handle.querySelector(".tag-text");
      text.setAttribute("x", tagX);
      text.textContent = `τ ${tau.toFixed(2)}`;
    }
    for (const row of refs.points.querySelectorAll("tr")) row.classList.toggle("on", Math.abs(+row.dataset.t - tau) < 1e-6);
  }

  const snap = (v) => Math.round(Math.max(0, Math.min(MAX_TAU, v)) / 0.05) * 0.05;
  function setTau(v) {
    const tau = +snap(v).toFixed(2);
    if (tau !== App.S.settings.tau) App.settings.set({ tau });
  }
  function tauFromPointer(event) {
    const box = refs.svg.getBoundingClientRect();
    const vx = ((event.clientX - box.left) / box.width) * W;
    return ((vx - L) / (W - L - R)) * MAX_TAU;
  }

  function renderViews() {
    const views = App.S.settings.views;
    for (const b of refs.viewsSeg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(+b.dataset.views === views));
    // Cells in reading order: recorded p1 p2 p3, mirrored p1 p2 p3. One view = recorded, middle phase.
    $$("i", refs.vmap).forEach((cell, i) => cell.classList.toggle("on", views === 6 || i === 1));
    refs.viewsExplain.textContent = views === 6
      ? "Each clip is read six times: three timing offsets, each as recorded and mirrored, and the answers are averaged. This was worth about 2.5 points on earlier models and costs six passes per clip."
      : "Each clip is read once, at the middle timing offset. Faster, but expect slightly lower accuracy than with six views.";
  }

  function renderFloor() {
    const floor = App.S.settings.floor;
    refs.floorRange.value = Math.round(floor * 100);
    refs.floorOut.textContent = `${Math.round(floor * 100)}%`;
    refs.floorExplain.textContent = floor === 0
      ? "Warnings are off: every answer is shown without a detection warning."
      : `If hands are found in fewer than ${Math.round(floor * 100)}% of frames, the capture screen flags the answer as unreliable, because the model was mostly looking at background.`;
  }

  function renderVocab() {
    const info = App.S.info;
    const words = info.classes.length;
    const deployed = App.deployed();
    refs.nWords.textContent = String(words);
    refs.nWordsLabel.textContent = words === 1 ? "word in the loaded model" : "words in the loaded model";
    if (App.S.mode === "preview") {
      refs.vocabNote.className = "note amber";
      refs.vocabNote.innerHTML = "<b>Preview mode.</b> No model is loaded, so this shows the deployed list only.";
    } else if (words > deployed.length) {
      refs.vocabNote.className = "note amber";
      refs.vocabNote.innerHTML = `<b>This server answers from all ${words} words, not just the 50.</b> On the same held-out clips that answer is right about 72% of the time; the 85.9% figure assumes the 50-word list.`;
    } else {
      refs.vocabNote.className = "note mint";
      refs.vocabNote.innerHTML = "The loaded model's vocabulary is the deployed list.";
    }
    refs.deployedSummary.textContent = `Show the ${deployed.length} deployed words`;
    refs.deployed.innerHTML = deployed.map((w) => `<li>${esc(clean(w))}</li>`).join("");
    const runs = info.tiers || [];
    refs.runs.innerHTML = runs.length
      ? runs.map((name) => `<li><code>python -m islvit.serve --run runs/${esc(name)}</code>
          <button class="btn sm ghost" type="button" data-copy="${esc(name)}">Copy</button></li>`).join("")
      : `<li class="muted" style="border:0;background:none;padding:0">${App.S.mode === "preview" ? "Start the server to list the trained runs on disk." : "No other trained runs were found on disk."}</li>`;
  }

  function renderSpec() {
    const info = App.S.info;
    const rows = [
      ["Run", App.S.mode === "preview" ? "none (preview)" : info.run],
      ["Words", info.classes.length],
      ["Frames", info.n_frames],
      ["Crop size", `${info.img_size} px`],
      ["Parameters", `${(+info.params_m).toFixed(2)} M`],
      ["Device", String(info.device).toUpperCase()],
      ["Mode", App.S.mode === "live" ? "Live" : "Preview"],
    ];
    refs.spec.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  }

  function render() { renderTau(); renderViews(); renderFloor(); }

  App.screens.settings = {
    init(section) {
      root = section;
      refs = {};
      for (const el of $$("[data-ref]", root)) refs[el.dataset.ref] = el;
      if (App.DATA.gate?.source) refs.src.textContent = App.DATA.gate.source;
      drawChart();
      // Redraw at the new pixel width when the card resizes (window resize, rail collapse).
      new ResizeObserver(() => {
        const width = Math.round(refs.svg.getBoundingClientRect().width);
        if (width > 0 && Math.abs(Math.max(420, width) - W) > 4) { drawChart(); renderTau(); }
      }).observe(refs.svg);

      refs.tauRange.addEventListener("input", () => setTau(+refs.tauRange.value));
      refs.svg.addEventListener("pointerdown", (e) => { dragging = true; refs.svg.setPointerCapture(e.pointerId); setTau(tauFromPointer(e)); });
      refs.svg.addEventListener("pointermove", (e) => { if (dragging) setTau(tauFromPointer(e)); });
      refs.svg.addEventListener("pointerup", () => { dragging = false; });
      refs.svg.addEventListener("pointercancel", () => { dragging = false; });
      refs.svg.addEventListener("keydown", (e) => {
        const step = { ArrowLeft: -0.05, ArrowDown: -0.05, ArrowRight: 0.05, ArrowUp: 0.05 }[e.key];
        if (step) { e.preventDefault(); setTau(App.S.settings.tau + step); }
      });
      refs.viewsSeg.addEventListener("click", (e) => {
        const b = e.target.closest("[data-views]");
        if (b) App.settings.set({ views: +b.dataset.views });
      });
      refs.floorRange.addEventListener("input", () => App.settings.set({ floor: +refs.floorRange.value / 100 }));
      refs.runs.addEventListener("click", async (e) => {
        const b = e.target.closest("[data-copy]");
        if (!b) return;
        try { await navigator.clipboard.writeText(`python -m islvit.serve --run runs/${b.dataset.copy}`); App.toast("Copied", { tone: "mint" }); }
        catch { App.toast("The clipboard isn't available here. Select the command and copy it by hand.", { tone: "coral" }); }
      });
      App.on("settings:change", render);
      renderVocab();
      renderSpec();
      render();
    },
    // Draw once now and again next frame, when the screen has its final width.
    onShow() { drawChart(); render(); requestAnimationFrame(() => { drawChart(); renderTau(); }); },
    onHide() { dragging = false; },
  };
})();
