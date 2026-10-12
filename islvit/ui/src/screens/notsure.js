/* Not sure: resolve clips the model wouldn't commit to.

   A clip lands here either because its confidence fell below τ, or because the
   user rejected an answer on the capture screen. Picking a word saves a
   correction (a free labelled example) and adds the word to the sentence. */
(function () {
  const { $$, esc, clean, pct, time } = App.U;
  let root, refs, focusId = null;

  const byId = (id) => App.S.clips.find((c) => c.id === id) || null;

  function focusClip() {
    const focused = byId(focusId);
    if (focused && focused.status === "open") return focused;
    const pending = App.S.pending;
    focusId = pending.length ? pending[pending.length - 1] : null;
    return byId(focusId);
  }

  // Ghost words: the candidates, large and drifting behind the fog. They are
  // meant to be almost unreadable -- that is the state the model is in.
  const SPOTS = [[6, 14], [48, 8], [70, 52], [18, 62], [40, 72]];
  function renderGhosts(clip) {
    refs.ghosts.innerHTML = clip.res.candidates.map((c, i) => {
      const [x, y] = SPOTS[i % SPOTS.length];
      const opacity = (0.25 + c.p * 1.4).toFixed(2);
      return `<span class="ns-ghost" style="left:${x}%;top:${y}%;opacity:${Math.min(0.9, opacity)};--dx:${i % 2 ? -14 : 12}px;--dy:${i % 3 ? 8 : -6}px">${esc(clean(c.word))}</span>`;
    }).join("");
  }

  function render() {
    const clip = focusClip();
    refs.main.hidden = !clip;
    refs.empty.hidden = !!clip;
    renderQueue();
    refs.corrections.textContent = String(App.S.corrections);
    refs.corrNote.innerHTML = App.S.mode === "preview"
      ? "Preview mode: picks are not saved. With the server running they go to <code>runs/corrections.jsonl</code>."
      : "Each pick is saved to <code>runs/corrections.jsonl</code> with the clip's shortlist, so it can become a labelled training example.";
    if (!clip) return;

    const top = clip.res.candidates[0];
    const tau = App.S.settings.tau;
    refs.reason.innerHTML = App.isAnswered(clip)
      ? "You marked this answer as wrong. Pick the word you signed."
      : `Best guess <q>${esc(clean(top.word))}</q> at ${pct(top.p)}, below your threshold of ${pct(tau)}.`;
    refs.meta.textContent = `Clip read at ${time(clip.at)} · hands found in ${pct(clip.res.detection_rate)} of frames`;
    renderGhosts(clip);

    const labels = clip.res.candidates.map((c) => c.word);
    refs.tiles.innerHTML = clip.res.candidates.map((c, i) => {
      const twins = App.lookalikes(c.word).filter((w) => labels.includes(w));
      const tags = [
        App.inDeployed(c.word) ? "" : '<span class="tag">not in the 50</span>',
        twins.length ? `<span class="tag amber">looks like ${esc(twins.map(clean).join(", "))}</span>` : "",
      ].join("");
      return `<button class="ns-tile${i === 0 ? " first" : ""}" type="button" data-pick="${i}">
        <kbd>${i + 1}</kbd>
        <span class="ns-tile-word">${esc(clean(c.word))}</span>
        <span class="ns-tile-tags">${tags}</span>
        <span class="ns-tile-p">${pct(c.p, 1)}</span>
        <span class="ns-tile-bar"><i style="width:${Math.max(2, c.p * 100).toFixed(1)}%"></i></span>
      </button>`;
    }).join("");
  }

  function renderQueue() {
    const open = App.S.pending.map(byId).filter(Boolean);
    refs.queueCard.hidden = open.length < 2;
    refs.queue.innerHTML = open.slice().reverse().map((clip) => `
      <li><button type="button" data-focus="${clip.id}" aria-current="${clip.id === focusId}">
        <span class="num">${time(clip.at)}</span>
        <span>${esc(clean(clip.res.candidates[0].word))}?</span>
        <span class="num">${pct(clip.res.confidence)}</span>
      </button></li>`).join("");
  }

  async function pick(index) {
    const clip = focusClip();
    if (!clip) return;
    const candidate = clip.res.candidates[index];
    if (!candidate) return;
    const tile = refs.tiles.querySelector(`[data-pick="${index}"]`);
    tile?.classList.add("saving");
    try {
      const { saved } = await App.correct(clip, candidate.word);
      App.toast(saved ? `Saved: ${clean(candidate.word)}` : `Preview — not saved: ${clean(candidate.word)}`, { tone: "mint" });
      focusId = null;
      render();
    } catch (error) {
      tile?.classList.remove("saving");
      App.toast(error.message, { tone: "coral" });
    }
  }

  function signAgain() {
    const clip = focusClip();
    if (clip) App.dismiss(clip);
    focusId = null;
    App.go("capture");
  }

  App.screens.notsure = {
    init(section) {
      root = section;
      refs = {};
      for (const el of $$("[data-ref]", root)) refs[el.dataset.ref] = el;
      root.addEventListener("click", (event) => {
        const tile = event.target.closest("[data-pick]");
        if (tile) { pick(+tile.dataset.pick); return; }
        const focus = event.target.closest("[data-focus]");
        if (focus) { focusId = focus.dataset.focus; render(); return; }
        const button = event.target.closest("[data-act]");
        if (!button) return;
        if (button.dataset.act === "again") signAgain();
        if (button.dataset.act === "capture") App.go("capture");
      });
      document.addEventListener("keydown", (event) => {
        if (App.active() !== "notsure" || App.typing() || event.altKey || event.ctrlKey || event.metaKey) return;
        if (/^[1-5]$/.test(event.key)) { event.preventDefault(); pick(+event.key - 1); }
        else if (event.key.toLowerCase() === "r") signAgain();
      });
      App.on("clip:new", () => { if (App.active() === "notsure") render(); });
      App.on("clip:update", render);
      App.on("settings:change", render);
      render();
    },
    onShow(opts) {
      if (opts?.clipId) focusId = opts.clipId;
      render();
    },
    onHide() {},
  };
})();
