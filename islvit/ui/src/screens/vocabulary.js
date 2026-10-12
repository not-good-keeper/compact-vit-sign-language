/* Vocabulary: how reliably each deployed word is read on sessions the model never saw.

   Recall comes from App.recall(): the shipped seed's per-clip hits on the 472
   held-out clips, about nine to eleven clips per word, so one clip moves a word
   by roughly ten points. Bands: reliable >= 90 %, patchy 50-89 %, weak < 50 %. */
(function () {
  const { $$, esc, clean, pct } = App.U;
  const DOT = 13, ROWS = 6;
  let root, refs, filter = "all", sort = "weak", openLabel = null, lastFocus = null;

  const band = (r) => (r == null ? "none" : r >= 0.9 ? "reliable" : r >= 0.5 ? "patchy" : "weak");

  function rows() {
    const recall = App.recall();
    if (recall) {
      const keep = new Set(App.deployed());
      const list = recall.rows.filter((r) => keep.has(r.label) || keep.size === 0);
      return list.length ? list : recall.rows;
    }
    return App.deployed().map((label) => ({ label, word: clean(label), hits: null, support: null, recall: null }));
  }

  function countUp(el, to, digits = 0, suffix = "") {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { el.textContent = to.toFixed(digits) + suffix; return; }
    const start = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - start) / 800);
      el.textContent = (to * (1 - Math.pow(1 - k, 3))).toFixed(digits) + suffix;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function renderStats() {
    const list = rows(), recall = App.recall();
    const counts = { reliable: 0, patchy: 0, weak: 0 };
    for (const r of list) if (r.recall != null) counts[band(r.recall)]++;
    const items = [
      ["", list.length, "words deployed", 0, ""],
      ["mint", counts.reliable, "reliable, 90% or more", 0, ""],
      ["amber", counts.patchy, "patchy, 50–89%", 0, ""],
      ["coral", counts.weak, "weak, below 50%", 0, ""],
      ["", recall ? recall.top1 * 100 : null, "overall, this seed", 1, "%"],
    ];
    refs.stats.innerHTML = items.map(([tone, , label]) => `<div class="vb-stat ${tone}"><b>—</b><span>${label}</span></div>`).join("");
    $$(".vb-stat b", refs.stats).forEach((el, i) => {
      const [, value, , digits, suffix] = items[i];
      if (value == null || (i > 0 && i < 4 && !recall)) { el.textContent = "—"; return; }
      countUp(el, value, digits, suffix);
    });
    refs.source.innerHTML = recall
      ? `Per-word recall on 472 held-out clips from recording sessions the model never saw. Source: <code>${esc(recall.source)}</code>. The three released seeds average 85.9%.`
      : "Per-word results exist only for the released model seeds. Load one of the release/ files to see recall here.";
    refs.spectrumCard.hidden = !recall;
  }

  function renderSpectrum() {
    const list = rows().filter((r) => r.recall != null);
    const groups = new Map();
    for (const r of list) {
      const key = r.recall.toFixed(3);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    let html = "", order = 0;
    for (const [key, members] of groups) {
      members.sort((a, b) => a.word.localeCompare(b.word));
      const cols = Math.ceil(members.length / ROWS);
      const half = (cols * DOT) / 2;
      members.forEach((r, i) => {
        const col = Math.floor(i / ROWS), row = i % ROWS;
        const dx = (col - (cols - 1) / 2) * DOT;
        html += `<button class="vb-dot ${band(r.recall)}" type="button" data-label="${esc(r.label)}"
          data-word="${esc(r.word)} · ${pct(r.recall)}" aria-label="${esc(r.word)}, ${pct(r.recall)} recall"
          style="left:calc(clamp(${half + 6}px, ${(+key * 100).toFixed(2)}%, calc(100% - ${half + 6}px)) + ${dx}px);bottom:${10 + row * DOT}px;--d:${order++ * 12}ms"></button>`;
      });
    }
    refs.dots.innerHTML = html;
    applyDim();
  }

  function passes(r) {
    const q = refs.search.value.trim().toLowerCase();
    if (q && !r.word.toLowerCase().includes(q)) return false;
    if (filter === "lookalike") return App.lookalikes(r.label).length > 0;
    if (filter !== "all") return band(r.recall) === filter;
    return true;
  }

  function applyDim() {
    const visible = new Set(rows().filter(passes).map((r) => r.label));
    for (const dot of $$(".vb-dot", refs.dots)) dot.classList.toggle("dim", !visible.has(dot.dataset.label));
  }

  function renderGrid() {
    let list = rows().filter(passes);
    list = list.sort(sort === "az" || list[0]?.recall == null
      ? (a, b) => a.word.localeCompare(b.word)
      : (a, b) => a.recall - b.recall || a.word.localeCompare(b.word));
    refs.grid.innerHTML = list.length
      ? list.map((r) => `<button class="vb-tile ${band(r.recall)}" type="button" data-label="${esc(r.label)}">
          <span class="vb-tile-top"><span class="vb-tile-word">${esc(r.word)}</span><span class="vb-tile-r">${r.recall == null ? "" : pct(r.recall)}</span></span>
          <span class="vb-tile-bar"><i style="width:${r.recall == null ? 0 : (r.recall * 100).toFixed(0)}%"></i></span>
          <span class="vb-tile-meta"><span>${r.support ? `${r.hits} of ${r.support} clips` : "no held-out data"}</span>
          ${App.lookalikes(r.label).length ? '<span class="tag amber">look-alike</span>' : ""}</span>
        </button>`).join("")
      : `<p class="vb-none">No words match. Clear the search or choose All.</p>`;
    applyDim();
  }

  function ring(r) {
    const c = 2 * Math.PI * 40;
    const color = { reliable: "var(--mint)", patchy: "var(--amber)", weak: "var(--coral)", none: "var(--ink-4)" }[band(r)];
    return `<svg viewBox="0 0 96 96" aria-hidden="true"><circle class="bg" cx="48" cy="48" r="40"/>
      <circle class="fg" cx="48" cy="48" r="40" stroke="${color}" stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${(c * (1 - (r || 0))).toFixed(1)}"/></svg>`;
  }

  function openDrawer(label) {
    const all = rows();
    const r = all.find((x) => x.label === label);
    if (!r) return;
    openLabel = label;
    const meaning = {
      reliable: "The model gets this word right on nearly every held-out clip.",
      patchy: "Right more often than not, but expect misses. More recorded takes would help.",
      weak: "Wrong on most held-out clips. This is where new recordings help most.",
      none: "No held-out results for this word with the loaded model.",
    }[band(r.recall)];
    const sibs = App.lookalikes(label).map((l) => all.find((x) => x.label === l)).filter(Boolean);
    const takes = App.S.enrol.counts[label] || 0;
    refs.drawerBody.innerHTML = `
      <p class="eyebrow">${band(r.recall) === "none" ? "Word" : band(r.recall)}</p>
      <h2 class="vb-d-word" id="vb-drawer-word">${esc(r.word)}</h2>
      <div class="vb-d-ring">${ring(r.recall)}<div><b>${r.recall == null ? "—" : pct(r.recall)}</b>
        <span>${r.support ? `${r.hits} of ${r.support} held-out clips` : "no held-out clips"}</span></div></div>
      <p class="vb-d-text">${meaning}</p>
      ${sibs.length ? `<div class="vb-d-section"><p class="eyebrow">Often confused with</p>
        <p class="vb-d-text" style="margin-top:6px">These signs differ mainly in wrist movement or the distance between the hands.</p>
        <div class="vb-d-sibs">${sibs.map((s) => `<button type="button" data-sib="${esc(s.label)}"><span>${esc(s.word)}</span><span class="num">${s.recall == null ? "—" : pct(s.recall)}</span></button>`).join("")}</div></div>` : ""}
      <div class="vb-d-section"><p class="eyebrow">Takes recorded this session</p>
        <div class="vb-d-takes"><b>${takes}</b><span class="muted">of 3 for one signer</span></div></div>
      <button class="btn primary vb-d-cta" type="button" data-act="record">Record takes for this word</button>`;
    if (!refs.drawer.classList.contains("open")) lastFocus = document.activeElement;
    refs.drawer.classList.add("open");
    refs.drawer.setAttribute("aria-hidden", "false");
    refs.drawer.focus();
  }

  function closeDrawer() {
    if (!refs.drawer.classList.contains("open")) return;
    refs.drawer.classList.remove("open");
    refs.drawer.setAttribute("aria-hidden", "true");
    openLabel = null;
    lastFocus?.focus?.();
  }

  function setSeg(seg, attr, value) {
    for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset[attr] === value));
  }

  App.screens.vocabulary = {
    init(section) {
      root = section;
      refs = {};
      for (const el of $$("[data-ref]", root)) refs[el.dataset.ref] = el;
      refs.filter.addEventListener("click", (e) => {
        const b = e.target.closest("[data-filter]");
        if (!b) return;
        filter = b.dataset.filter;
        setSeg(refs.filter, "filter", filter);
        renderGrid();
      });
      refs.sort.addEventListener("click", (e) => {
        const b = e.target.closest("[data-sort]");
        if (!b) return;
        sort = b.dataset.sort;
        setSeg(refs.sort, "sort", sort);
        renderGrid();
      });
      refs.search.addEventListener("input", renderGrid);
      root.addEventListener("click", (e) => {
        const target = e.target.closest("[data-label]") || e.target.closest("[data-sib]");
        if (target) { openDrawer(target.dataset.label || target.dataset.sib); return; }
        const act = e.target.closest("[data-act]")?.dataset.act;
        if (act === "close") closeDrawer();
        if (act === "record" && openLabel) { const label = openLabel; closeDrawer(); App.go("enrolment", { label }); }
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && App.active() === "vocabulary") closeDrawer();
      });
      App.on("enrol:change", () => { if (openLabel) openDrawer(openLabel); });
      renderStats();
      renderSpectrum();
      renderGrid();
    },
    onShow() { renderGrid(); },
    onHide() { closeDrawer(); },
  };
})();
