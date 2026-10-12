/* Sentence: arrange, speak, copy and export the accepted words.

   "Merge repeated words" changes only what is shown, spoken, copied and
   exported; the underlying sequence keeps every accepted clip. */
(function () {
  const { $$, esc, clean, pct, time, icon } = App.U;
  let root, refs, dragFrom = null, speaking = false, clearArmed = 0;

  const merged = () => refs.merge.checked;
  // Indices of words that survive merging (a word equal to the one before it is dropped).
  function kept() {
    const words = App.S.sentence;
    return words.map((w, i) => (!merged() || i === 0 || clean(w.label) !== clean(words[i - 1].label)));
  }
  const text = () => App.S.sentence.filter((_, i) => kept()[i]).map((w) => clean(w.label)).join(" ");

  function renderWords() {
    const words = App.S.sentence;
    const keep = kept();
    refs.count.textContent = words.length ? `${keep.filter(Boolean).length} word${keep.filter(Boolean).length === 1 ? "" : "s"}` : "";
    refs.words.innerHTML = words.length
      ? words.map((w, i) => `
        <li class="sn-word${keep[i] ? "" : " merged"}" draggable="true" tabindex="0" data-index="${i}" data-id="${w.id}"
            aria-label="${esc(clean(w.label))}, word ${i + 1} of ${words.length}">
          ${esc(clean(w.label))}
          <button class="sn-x" type="button" data-remove="${w.id}" aria-label="Remove ${esc(clean(w.label))}">${icon("x")}</button>
        </li>`).join("")
      : `<li class="sn-placeholder">Accepted words collect here, ready to speak or copy.</li>`;
    for (const button of [refs.speakBtn]) button.disabled = !words.length;
  }

  function outcome(clip) {
    if (clip.status === "accepted") return { dot: "mint", word: clean(clip.chosen), cls: "", tags: "" };
    if (clip.status === "corrected") return { dot: "sky", word: clean(clip.chosen), cls: "", tags: '<span class="tag sky">corrected</span>' };
    if (clip.status === "dismissed") return { dot: "", word: "signed again", cls: "dim", tags: "" };
    if (!App.isAnswered(clip)) return { dot: "amber", word: "not sure", cls: "amber", tags: `<a data-check="${clip.id}">Check it</a>` };
    return { dot: "", word: clean(clip.res.candidates[0].word), cls: "dim", tags: '<span class="tag">not accepted yet</span>' };
  }

  function renderTimeline() {
    const clips = App.S.clips.slice().reverse();
    refs.timeline.innerHTML = clips.length
      ? clips.map((clip) => {
        const o = outcome(clip);
        return `<li class="sn-event">
          <span class="sn-dot ${o.dot}"></span>
          <span class="sn-event-main"><span class="sn-event-word ${o.cls}">${esc(o.word)}</span>${o.tags}</span>
          <span class="sn-event-time">${time(clip.at)}</span>
          <span class="sn-event-meta">
            <span class="tag">${pct(clip.res.confidence)} sure</span>
            <span class="tag">hands ${pct(clip.res.detection_rate)}</span>
          </span>
        </li>`;
      }).join("")
      : `<li class="sn-empty-time">No clips yet. Each clip you read on the capture screen is listed here with what became of it.</li>`;
  }

  function render() { renderWords(); renderTimeline(); }

  /* ---------------------------------------------------------------- actions */
  function speak() {
    if (!("speechSynthesis" in window)) { App.toast("This browser can't speak text aloud.", { tone: "coral" }); return; }
    if (speaking) { speechSynthesis.cancel(); return; }
    const words = App.S.sentence.map((w, i) => ({ w, i })).filter(({ i }) => kept()[i]);
    if (!words.length) return;
    const spoken = words.map(({ w }) => clean(w.label).replace(" / ", " "));
    const utterance = new SpeechSynthesisUtterance(spoken.join(" "));
    utterance.lang = "en-IN";
    utterance.rate = 0.95;
    // Character offsets of each word, so boundary events can light the right tile.
    const starts = [];
    spoken.reduce((offset, word) => { starts.push(offset); return offset + word.length + 1; }, 0);
    const tiles = () => $$(".sn-word", refs.words);
    const light = (k) => tiles().forEach((t) => t.classList.toggle("speaking", +t.dataset.index === (k == null ? -1 : words[k].i)));
    let boundary = false, fallback = null;
    utterance.onboundary = (event) => {
      boundary = true;
      let k = 0;
      while (k + 1 < starts.length && starts[k + 1] <= event.charIndex) k++;
      light(k);
    };
    utterance.onstart = () => {
      speaking = true;
      refs.speakBtn.textContent = "Stop";
      light(0);
      // Some voices never fire boundary events; step through the words instead.
      let k = 0;
      fallback = setInterval(() => { if (!boundary && ++k < words.length) light(k); }, 520);
    };
    utterance.onend = utterance.onerror = () => {
      speaking = false;
      clearInterval(fallback);
      refs.speakBtn.textContent = "Speak";
      light(null);
    };
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
  }

  async function copy() {
    const value = text();
    if (!value) { App.toast("There are no words to copy yet.", { tone: "amber" }); return; }
    try { await navigator.clipboard.writeText(value); App.toast("Copied", { tone: "mint" }); }
    catch { App.toast("The clipboard isn't available here. Select the words and copy them by hand.", { tone: "coral" }); }
  }

  function exportJson() {
    const payload = {
      exported: new Date().toISOString(),
      model: App.S.info.run,
      mode: App.S.mode,
      settings: { ...App.S.settings },
      merge_repeated: merged(),
      sentence: App.S.sentence.filter((_, i) => kept()[i]).map((w) => ({ word: clean(w.label), label: w.label, at: w.at })),
      clips: App.S.clips.map((c) => ({
        at: c.at, outcome: c.status, word: c.chosen || c.res.candidates[0].word,
        confidence: c.res.confidence, detection_rate: c.res.detection_rate, answered: App.isAnswered(c),
      })),
    };
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
    link.download = `isl-session-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    App.toast("Exported the session as JSON", { tone: "mint" });
  }

  function clear() {
    if (!App.S.sentence.length) return;
    if (Date.now() - clearArmed < 3000) {
      clearArmed = 0;
      refs.clearBtn.textContent = "Clear";
      App.clearSentence();
      return;
    }
    clearArmed = Date.now();
    refs.clearBtn.textContent = "Click again to clear";
    setTimeout(() => { if (clearArmed && Date.now() - clearArmed >= 3000) { clearArmed = 0; refs.clearBtn.textContent = "Clear"; } }, 3100);
  }

  /* ---------------------------------------------------------------- reorder */
  function wireDrag() {
    refs.words.addEventListener("dragstart", (event) => {
      const tile = event.target.closest(".sn-word");
      if (!tile) return;
      dragFrom = +tile.dataset.index;
      tile.classList.add("dragging");
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", tile.dataset.id);
    });
    refs.words.addEventListener("dragover", (event) => {
      const tile = event.target.closest(".sn-word");
      if (!tile || dragFrom == null) return;
      event.preventDefault();
      $$(".sn-word.over", refs.words).forEach((t) => t !== tile && t.classList.remove("over"));
      tile.classList.add("over");
    });
    refs.words.addEventListener("drop", (event) => {
      const tile = event.target.closest(".sn-word");
      if (!tile || dragFrom == null) return;
      event.preventDefault();
      App.moveWord(dragFrom, +tile.dataset.index);
    });
    refs.words.addEventListener("dragend", () => {
      dragFrom = null;
      $$(".sn-word", refs.words).forEach((t) => t.classList.remove("dragging", "over"));
    });
    refs.words.addEventListener("keydown", (event) => {
      const tile = event.target.closest(".sn-word");
      if (!tile || !event.altKey || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const from = +tile.dataset.index, to = from + (event.key === "ArrowLeft" ? -1 : 1);
      App.moveWord(from, to);
      requestAnimationFrame(() => refs.words.querySelector(`[data-index="${Math.max(0, Math.min(App.S.sentence.length - 1, to))}"]`)?.focus());
    });
  }

  App.screens.sentence = {
    init(section) {
      root = section;
      refs = {};
      for (const el of $$("[data-ref]", root)) refs[el.dataset.ref] = el;
      wireDrag();
      root.addEventListener("click", (event) => {
        const remove = event.target.closest("[data-remove]");
        if (remove) { App.removeWord(remove.dataset.remove); return; }
        const check = event.target.closest("[data-check]");
        if (check) { App.go("notsure", { clipId: check.dataset.check }); return; }
        const button = event.target.closest("[data-act]");
        if (!button) return;
        ({ speak, copy, export: exportJson, clear })[button.dataset.act]?.();
      });
      refs.merge.addEventListener("change", renderWords);
      App.on("sentence:change", renderWords);
      App.on("clip:new", renderTimeline);
      App.on("clip:update", renderTimeline);
      App.on("settings:change", renderTimeline);
      render();
    },
    onShow() { render(); },
    onHide() { if (speaking) speechSynthesis.cancel(); },
  };
})();
