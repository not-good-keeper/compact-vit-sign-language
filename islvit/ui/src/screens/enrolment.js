/* Enrolment: record new signers, three takes per word.

   Takes go to custom/<signer>/<word>/takeN.webm through the existing /enrol
   endpoint, the layout islvit.data.ingest_custom reads. The queue puts the words
   the model is worst at -- and the look-alike groups behind most remaining errors
   -- first, because those are where new recordings are most likely to help. */
(function () {
  const { $$, esc, clean, pct, icon } = App.U;
  const TAKES = 3;
  let root, refs, stream = null, order = "need", current = null, recording = false;

  const takes = (label) => App.S.enrol.counts[label] || 0;
  const signer = () => refs.signer.value.trim();

  function queue() {
    const recall = App.recall();
    const byLabel = new Map((recall?.rows || []).map((r) => [r.label, r.recall]));
    const words = App.deployed().map((label) => ({ label, word: clean(label), recall: byLabel.get(label), look: App.lookalikes(label).length > 0 }));
    if (order === "az" || !recall) return words.sort((a, b) => a.word.localeCompare(b.word));
    // Needs data first: lowest recall first; look-alikes ahead of others at equal recall.
    return words.sort((a, b) => (a.recall - b.recall) || (b.look - a.look) || a.word.localeCompare(b.word));
  }

  function nextWord(after) {
    const list = queue();
    const start = after ? list.findIndex((w) => w.label === after) + 1 : 0;
    for (let i = 0; i < list.length; i++) {
      const w = list[(start + i) % list.length];
      if (takes(w.label) < TAKES) return w.label;
    }
    return null;
  }

  function renderQueue() {
    refs.queue.innerHTML = queue().map((w) => {
      const band = w.recall == null ? "" : w.recall >= 0.9 ? "" : w.recall >= 0.5 ? "patchy" : "weak";
      const n = takes(w.label);
      return `<li><button type="button" data-word="${esc(w.label)}" aria-current="${w.label === current}">
        <span class="en-q-word">${esc(w.word)}${w.look ? ' <span class="tag amber">look-alike</span>' : ""}</span>
        <span class="en-q-r ${band}">${w.recall == null ? "" : pct(w.recall)}</span>
        <span class="en-pips" aria-label="${n} of ${TAKES} takes">${Array.from({ length: TAKES }, (_, i) => `<i class="${i < n ? "on" : ""}"></i>`).join("")}</span>
      </button></li>`;
    }).join("");
  }

  function renderStage() {
    refs.stage.classList.toggle("live", !!stream);
    refs.word.textContent = current ? clean(current) : "—";
    refs.take.textContent = current ? `take ${Math.min(takes(current) + 1, TAKES)} of ${TAKES}` : "";
    refs.camBtn.innerHTML = stream ? `${icon("stop")}Turn off camera` : `${icon("camera")}Turn on camera`;
    refs.recBtn.disabled = !stream || !current || recording;
    refs.skipBtn.disabled = !current || recording;
  }

  function renderSteps() {
    const done1 = !!signer(), done2 = !!current;
    const states = [done1 ? "done" : "now", done1 ? (done2 ? "done" : "now") : "", done1 && done2 ? "now" : ""];
    $$("li", refs.steps).forEach((li, i) => { li.className = states[i]; });
  }

  function renderProgress() {
    const words = App.deployed().length;
    const total = words * TAKES;
    const done = App.deployed().reduce((sum, l) => sum + Math.min(TAKES, takes(l)), 0);
    refs.done.textContent = String(done);
    refs.doneOf.textContent = `of ${total} takes for this signer`;
    refs.ring.style.strokeDashoffset = (314.16 * (1 - done / Math.max(1, total))).toFixed(1);
    refs.saveNote.innerHTML = App.S.mode === "preview"
      ? "Preview mode: takes are not saved. With the server running they go to <code>custom/&lt;signer&gt;/&lt;word&gt;/</code>."
      : "Takes are saved to <code>custom/&lt;signer&gt;/&lt;word&gt;/takeN.webm</code> on the server.";
  }

  function render() { renderQueue(); renderStage(); renderSteps(); renderProgress(); }

  function select(label) {
    current = label;
    App.S.enrol.word = label;
    render();
  }

  function status(text, tone = "") {
    refs.status.textContent = text;
    refs.status.className = `en-status ${tone}`;
  }

  function checkSigner() {
    const ok = !!signer();
    refs.signer.classList.toggle("bad", !ok);
    refs.signerHelp.classList.toggle("err", !ok);
    refs.signerHelp.textContent = ok
      ? "Takes are saved per signer, so a test set can hold out people the model never saw."
      : "Enter a signer name before recording.";
    if (!ok) refs.signer.focus();
    return ok;
  }

  async function toggleCamera() {
    if (stream) {
      App.camera.stop(stream);
      stream = null;
      refs.video.srcObject = null;
      render();
      return;
    }
    try {
      stream = await App.camera.start(refs.video);
      if (!current) select(nextWord());
    } catch (error) {
      App.toast(error.message, { tone: "coral" });
    }
    render();
  }

  async function record() {
    if (!stream || !current || recording || App.typing()) return;
    if (!checkSigner()) return;
    recording = true;
    renderStage();
    const word = current;
    try {
      const blob = await App.record(stream, {
        seconds: 3, countdown: 3,
        onCountdown(n) {
          refs.count.hidden = n === 0;
          if (n) { refs.countNum.textContent = n; refs.countNum.style.animation = "none"; void refs.countNum.offsetWidth; refs.countNum.style.animation = ""; }
          if (n === 0) { refs.rec.hidden = false; refs.stage.classList.add("recording"); }
        },
        onTick(t, total) {
          refs.recFill.style.strokeDashoffset = (94.25 * (1 - t / total)).toFixed(2);
          refs.recText.textContent = `REC ${t.toFixed(1)} s`;
        },
      });
      refs.rec.hidden = true;
      refs.stage.classList.remove("recording");
      status("Saving…");
      const result = await App.enrolTake(blob, signer(), word);
      if (result.preview) {
        status(`Take ${result.take} of ${TAKES} for “${clean(word)}” (preview, not saved)`, "ok");
        App.toast("Preview — not saved", { tone: "amber" });
      } else {
        status(`Saved take ${result.take} of ${TAKES} for “${clean(word)}”`, "ok");
      }
      if (takes(word) >= TAKES) {
        const next = nextWord(word);
        select(next);
        if (!next) App.toast("Every word has three takes for this signer.", { tone: "mint" });
      }
    } catch (error) {
      status(error.message, "err");
      App.toast(error.message, { tone: "coral" });
    } finally {
      recording = false;
      refs.count.hidden = true;
      refs.rec.hidden = true;
      refs.stage.classList.remove("recording");
      render();
    }
  }

  App.screens.enrolment = {
    init(section) {
      root = section;
      refs = {};
      for (const el of $$("[data-ref]", root)) refs[el.dataset.ref] = el;
      refs.signer.value = App.S.enrol.signer || "";
      refs.signer.addEventListener("input", () => {
        App.S.enrol.signer = signer();
        try { localStorage.setItem("isl.signer", JSON.stringify(signer())); } catch { /* private window */ }
        refs.signer.classList.remove("bad");
        refs.signerHelp.classList.remove("err");
        renderSteps();
      });
      refs.order.addEventListener("click", (e) => {
        const b = e.target.closest("[data-order]");
        if (!b) return;
        order = b.dataset.order;
        for (const x of refs.order.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
        renderQueue();
      });
      refs.queue.addEventListener("click", (e) => {
        const b = e.target.closest("[data-word]");
        if (b && !recording) select(b.dataset.word);
      });
      root.addEventListener("click", (e) => {
        const act = e.target.closest("[data-act]")?.dataset.act;
        if (act === "camera") toggleCamera();
        if (act === "record") record();
        if (act === "skip" && current) select(nextWord(current));
      });
      document.addEventListener("keydown", (e) => {
        if (App.active() !== "enrolment" || App.typing() || e.altKey || e.ctrlKey || e.metaKey) return;
        if (e.code === "Space") { e.preventDefault(); record(); }
      });
      App.on("enrol:change", render);
      current = nextWord();
      render();
    },
    onShow(opts) {
      if (opts?.label) select(opts.label);
      else render();
    },
    onHide() {
      if (stream) { App.camera.stop(stream); stream = null; refs.video.srcObject = null; }
      render();
    },
  };
})();
