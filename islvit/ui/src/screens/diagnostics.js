/* Detection: where every crop box came from, over the last minute or the session.

   An interpolated box is a stale box -- it shows where a hand recently was -- so a
   stream at 80 % "covered" can still be mostly background. This screen keeps the
   four provenances apart for exactly that reason. */
(function () {
  const { $$, esc, clean, pct, time, icon, provbar } = App.U;
  const STREAMS = [["left_hand", "Left hand", "hand"], ["right_hand", "Right hand", "hand"], ["face", "Face", "face"]];
  // Drawn in real pixels: CW follows the card's width so labels do not scale with the window.
  let CW = 520;
  const CH = 220, CL = 36, CR = 12, CT = 12, CB = 26;
  let root, refs, windowName = "minute", ticker = null;

  function clips() {
    const all = App.S.clips;
    if (windowName === "session") return all;
    const since = Date.now() - 60_000;
    return all.filter((c) => c.at.getTime() >= since);
  }

  function mean(list, key, field) {
    return list.reduce((sum, c) => sum + c.res.provenance[key][field], 0) / list.length;
  }

  function renderVerdict(list) {
    const left = mean(list, "left_hand", "genuine"), right = mean(list, "right_hand", "genuine");
    const worst = Math.min(left, right);
    const floor = App.S.settings.floor;
    const [tone, title, body] = worst < floor
      ? ["coral", "Hands are hard to find", `One hand was found in only ${pct(worst)} of frames, below your warning floor of ${pct(floor)}. Answers in this window mostly reflect background. Move into better light and keep both hands in the frame.`]
      : worst < 0.6
        ? ["amber", "Detection is patchy", `One hand was found in ${pct(worst)} of frames. Expect weaker answers than on clean studio video.`]
        : ["mint", "Detection is healthy", `Both hands were found in at least ${pct(worst)} of frames, comparable to the studio recordings the model learned from (91% left, 85% right).`];
    refs.verdict.className = `dg-verdict ${tone}`;
    refs.verdict.innerHTML = `<span class="dot"></span><div><b>${title}</b>${body}</div>`;
  }

  function renderStreams(list) {
    refs.streams.innerHTML = STREAMS.map(([key, name, glyph]) => {
      const p = {};
      for (const f of ["direct", "roi", "interp", "missing", "genuine"]) p[f] = mean(list, key, f);
      return `<div class="dg-row">
        <span class="dg-row-name">${icon(glyph)}${name}</span>
        <span class="dg-row-val">${(p.genuine * 100).toFixed(0)}<small>%</small></span>
        ${provbar(p)}
        <span class="dg-row-parts"><span>direct ${pct(p.direct)}</span><span>ROI ${pct(p.roi)}</span><span>interpolated ${pct(p.interp)}</span><span>missing ${pct(p.missing)}</span></span>
      </div>`;
    }).join("");
  }

  function renderChart(list) {
    const width = Math.round(refs.chart.getBoundingClientRect().width);
    if (width > 0) CW = Math.max(320, width);
    refs.chart.setAttribute("viewBox", `0 0 ${CW} ${CH}`);
    const yv = (v) => CT + (1 - v) * (CH - CT - CB);
    const n = list.length;
    const xv = (i) => (n === 1 ? (CL + CW - CR) / 2 : CL + (i / (n - 1)) * (CW - CL - CR));
    let svg = "";
    for (const v of [0, 0.5, 1]) {
      svg += `<line class="grid" x1="${CL}" x2="${CW - CR}" y1="${yv(v)}" y2="${yv(v)}"/><text class="lab" x="${CL - 6}" y="${yv(v) + 3}" text-anchor="end">${v * 100}%</text>`;
    }
    const floor = App.S.settings.floor;
    svg += `<line class="floor" x1="${CL}" x2="${CW - CR}" y1="${yv(floor)}" y2="${yv(floor)}"/>
            <text class="floor-lab" x="${CW - CR}" y="${yv(floor) - 5}" text-anchor="end">floor ${pct(floor)}</text>`;
    const series = [["face", "rgba(236,239,246,.35)"], ["left_hand", "var(--lilac)"], ["right_hand", "var(--peri)"]];
    for (const [key, color] of series) {
      const d = list.map((c, i) => `${i ? "L" : "M"}${xv(i).toFixed(1)},${yv(c.res.provenance[key].genuine).toFixed(1)}`).join(" ");
      if (n > 1) svg += `<path class="line" d="${d}" stroke="${color}"/>`;
      svg += list.map((c, i) => `<circle class="dot" tabindex="0" data-i="${i}" data-k="${key}" cx="${xv(i).toFixed(1)}" cy="${yv(c.res.provenance[key].genuine).toFixed(1)}" r="${key === "face" ? 3 : 4.5}" fill="${color}"/>`).join("");
    }
    if (n) {
      svg += `<text class="lab" x="${CL}" y="${CH - 6}">${time(list[0].at)}</text>`;
      if (n > 1) svg += `<text class="lab" x="${CW - CR}" y="${CH - 6}" text-anchor="end">${time(list[n - 1].at)}</text>`;
    }
    refs.chart.innerHTML = svg;
  }

  function showTip(dot, list) {
    const clip = list[+dot.dataset.i];
    if (!clip) return;
    const name = STREAMS.find(([k]) => k === dot.dataset.k)[1];
    const answer = App.isAnswered(clip) ? clean(clip.res.candidates[0].word) : "not sure";
    refs.tip.innerHTML = `${esc(name)} ${pct(clip.res.provenance[dot.dataset.k].genuine)} · ${esc(answer)} <span class="num">${time(clip.at)}</span>`;
    const box = refs.chart.getBoundingClientRect(), wrap = refs.chart.parentElement.getBoundingClientRect();
    refs.tip.style.left = `${(+dot.getAttribute("cx") / CW) * box.width + (box.left - wrap.left)}px`;
    refs.tip.style.top = `${(+dot.getAttribute("cy") / CH) * box.height + (box.top - wrap.top)}px`;
    refs.tip.hidden = false;
  }

  function renderTable(list) {
    const floor = App.S.settings.floor;
    const cell = (v) => `<td class="${v < floor ? "low" : ""}">${pct(v)}</td>`;
    refs.rows.innerHTML = list.slice().reverse().map((c) => {
      const p = c.res.provenance;
      const result = App.isAnswered(c)
        ? `<td class="word">${esc(clean(c.res.candidates[0].word))}</td>`
        : `<td class="unsure">not sure</td>`;
      return `<tr><td>${time(c.at)}</td>${cell(p.left_hand.genuine)}${cell(p.right_hand.genuine)}${cell(p.face.genuine)}${result}</tr>`;
    }).join("");
  }

  function renderReference() {
    const ref = App.DATA.reference;
    const bar = (label, v, color) => `<div class="dg-ref-bar"><span>${label}</span><span class="track"><i style="width:${v * 100}%;background:${color}"></i></span><span class="num">${pct(v, 1)}</span></div>`;
    refs.reference.innerHTML = `
      <div class="dg-ref-group"><h3>INCLUDE, the studio the model learned from</h3>
        ${bar("Left hand", ref.include.left, "var(--lilac)")}${bar("Right hand", ref.include.right, "var(--peri)")}${bar("Face", ref.include.face, "rgba(236,239,246,.5)")}</div>
      <div class="dg-ref-group"><h3>CISLR, different signers and cameras</h3>
        ${bar("Left hand", ref.cislr.left, "var(--lilac)")}${bar("Right hand", ref.cislr.right, "var(--peri)")}</div>`;
  }

  function render() {
    const list = clips();
    const all = App.S.clips.length;
    refs.count.textContent = all ? `${list.length} of ${all} clip${all === 1 ? "" : "s"}` : "";
    refs.body.hidden = !list.length;
    refs.empty.hidden = !!list.length;
    if (!list.length) {
      refs.emptyTitle.textContent = all ? "Nothing in the last minute" : "No clips yet";
      refs.emptyBody.textContent = all
        ? "Read another clip, or switch to This session to see every clip so far."
        : "Read a clip on the capture screen. Each one reports how often the detector found each hand and the face.";
      return;
    }
    renderVerdict(list);
    renderStreams(list);
    renderChart(list);
    renderTable(list);
  }

  function startTicker() {
    clearInterval(ticker);
    if (windowName === "minute") ticker = setInterval(render, 5000);
  }

  App.screens.diagnostics = {
    init(section) {
      root = section;
      refs = {};
      for (const el of $$("[data-ref]", root)) refs[el.dataset.ref] = el;
      renderReference();
      new ResizeObserver(() => {
        const width = Math.round(refs.chart.getBoundingClientRect().width);
        if (width > 0 && Math.abs(Math.max(320, width) - CW) > 4 && App.active() === "diagnostics") render();
      }).observe(refs.chart);
      refs.windowSeg.addEventListener("click", (e) => {
        const b = e.target.closest("[data-window]");
        if (!b) return;
        windowName = b.dataset.window;
        for (const x of refs.windowSeg.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
        render();
        if (App.active() === "diagnostics") startTicker();
      });
      const tipFor = (e) => { const dot = e.target.closest(".dot"); if (dot) showTip(dot, clips()); };
      refs.chart.addEventListener("pointerover", tipFor);
      refs.chart.addEventListener("focusin", tipFor);
      refs.chart.addEventListener("pointerleave", () => { refs.tip.hidden = true; });
      refs.chart.addEventListener("focusout", () => { refs.tip.hidden = true; });
      root.addEventListener("click", (e) => { if (e.target.closest('[data-act="capture"]')) App.go("capture"); });
      App.on("clip:new", () => { if (App.active() === "diagnostics") render(); });
      App.on("settings:change", () => { if (App.active() === "diagnostics") render(); });
      render();
    },
    onShow() { render(); startTicker(); },
    onHide() { clearInterval(ticker); ticker = null; },
  };
})();
