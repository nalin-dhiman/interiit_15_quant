import {
  DEFAULT_CONFIG,
  createSession,
  stepSession,
  snapshotSession,
} from "./engine/index.js";

const $ = (id) => document.getElementById(id);
const EMPTY_COMPARISON = $("comparison-results").innerHTML;
const POLICY_LABELS = {
  B4: "Basis-aware band",
  R2B: "Continuation reset",
  B3: "Inventory band",
};
const POLICY_HELP = {
  B4: "A conventional partial-target band that uses public basis and settlement information.",
  R2B: "A finite continuation comparison: retain exposure or choose a feasible, cost-paying partial target.",
  B3: "A conventional inventory band. Its hedging target does not forecast the public basis.",
};
const SCENARIOS = {
  reverting: {
    basisPersistence: 0.92,
    basisVolatility: 0.14,
    priceVolatility: 0.14,
  },
  persistent: {
    basisPersistence: 0.99,
    basisVolatility: 0.14,
    priceVolatility: 0.14,
  },
  fast: { basisPersistence: 0.4, basisVolatility: 0.18, priceVolatility: 0.14 },
  noisy: {
    basisPersistence: 0.92,
    basisVolatility: 0.38,
    priceVolatility: 0.24,
  },
};
const CONFIG_FIELDS = [
  "seed",
  "steps",
  "policy",
  "quoteEdge",
  "inventorySkew",
  "riskAversion",
  "hedgeFee",
  "hedgeSlippage",
  "resetFixedCost",
  "rfqsPerStep",
  "targetFraction",
];
let appliedConfig;
let session;
let snapshot;
let playing = false;
let playTimer;
let chartMode = "pnl";
let worker;
let activeComparisonId = 0;
let comparing = false;
let comparisonResult;
let comparisonFailure;
let sessionFailure;
let toastTimer;
let resizeTimer;

const number = (value, digits = 2) =>
  Number.isFinite(Number(value))
    ? Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "—";
const signed = (value, digits = 2) =>
  `${Number(value) > 0 ? "+" : ""}${number(value, digits)}`;
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const moneyClass = (value) => (Number(value) < -1e-9 ? "negative" : "");

function notify(message) {
  $("toast").textContent = message;
  $("toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("visible"), 4300);
}

function readConfig() {
  const config = { ...DEFAULT_CONFIG, ...SCENARIOS[$("scenario").value] };
  for (const field of CONFIG_FIELDS)
    config[field] =
      field === "policy" ? $(field).value : Number($(field).value);
  if (
    !Number.isInteger(config.seed) ||
    config.seed < 0 ||
    config.seed > 4294967295
  )
    throw new Error("Choose an integer seed from 0 to 4,294,967,295.");
  if (
    !Number.isInteger(config.steps) ||
    config.steps < 10 ||
    config.steps > 120
  )
    throw new Error("Choose 10 to 120 intervals.");
  return config;
}

function syncRangeLabels() {
  for (const key of [
    "quoteEdge",
    "inventorySkew",
    "riskAversion",
    "targetFraction",
  ]) {
    $(`${key}-value`).textContent = Number($(key).value).toFixed(
      key === "targetFraction" ? 2 : 3,
    );
  }
  $("policy-help").textContent = POLICY_HELP[$("policy").value];
}

function markPending() {
  syncRangeLabels();
  $("config-indicator").textContent = "Changes pending";
  $("config-indicator").classList.add("pending");
  $("apply-button").classList.add("pending");
}

function resetRun(config = appliedConfig, announce = false) {
  pauseRun();
  if (comparing) {
    worker?.postMessage({ id: activeComparisonId, type: "cancel" });
    activeComparisonId += 1;
    endComparison();
  }
  try {
    const nextSession = createSession({ ...config });
    if (
      appliedConfig &&
      JSON.stringify(config) !== JSON.stringify(appliedConfig)
    ) {
      comparisonResult = undefined;
      comparisonFailure = undefined;
      $("export-comparison").innerHTML =
        'Export comparison <span aria-hidden="true">↓</span>';
      $("comparison-results").innerHTML = EMPTY_COMPARISON;
      $("comparison-caption").textContent =
        "Configuration changed. Run a new comparison for these settings.";
      $("export-comparison").hidden = true;
    }
    session = nextSession;
    sessionFailure = undefined;
    appliedConfig = { ...config };
    snapshot = snapshotSession(session);
    $("config-indicator").textContent = "Applied";
    $("config-indicator").classList.remove("pending");
    $("apply-button").classList.remove("pending");
    render();
    if (announce) {
      document
        .querySelector(".controls-panel")
        .classList.remove("mobile-expanded");
      $("config-toggle").setAttribute("aria-expanded", "false");
      notify("Configuration applied. The session has restarted.");
    }
  } catch (error) {
    notify(error.message || "The configuration could not be applied.");
  }
}

function pauseRun() {
  playing = false;
  clearTimeout(playTimer);
  $("play-label").textContent = sessionFailure
    ? "Restart"
    : snapshot?.done
      ? "Run again"
      : snapshot?.step
        ? "Resume"
        : "Run session";
  $("play-icon").textContent = "▶";
  $("play-button").setAttribute(
    "aria-label",
    sessionFailure || snapshot?.done
      ? "Restart and run session"
      : "Run session",
  );
  if (snapshot) renderRunState();
}

function advance() {
  if (!session || snapshot?.done || sessionFailure) return;
  try {
    const result = stepSession(session);
    snapshot = result?.history ? result : snapshotSession(session);
    render();
    if (snapshot.done) pauseRun();
  } catch (error) {
    sessionFailure = error.failure || {
      message: error.message,
      config: { ...appliedConfig },
      step: session.step,
      market: { ...session.market },
    };
    pauseRun();
    notify(
      `Session stopped: ${error.message}. The failed path is available in Run JSON.`,
    );
    renderRunState();
  }
}

function playRun() {
  if (playing) {
    pauseRun();
    renderRunState();
    return;
  }
  if (snapshot?.done || sessionFailure) resetRun();
  playing = true;
  $("play-label").textContent = "Pause";
  $("play-icon").textContent = "Ⅱ";
  $("play-button").setAttribute("aria-label", "Pause session");
  const tick = () => {
    if (!playing) return;
    advance();
    if (playing && !snapshot?.done) playTimer = setTimeout(tick, 85);
  };
  renderRunState();
  tick();
}

function renderRunState() {
  $("run-state").textContent = sessionFailure
    ? "Stopped"
    : snapshot?.done
      ? "Settled"
      : playing
        ? "Running"
        : snapshot?.step
          ? "Paused"
          : "Ready";
  $("run-state").classList.toggle("running", playing);
  $("step-button").disabled = Boolean(
    snapshot?.done || playing || sessionFailure,
  );
}

function render() {
  if (!snapshot) return;
  renderRunState();
  const s = snapshot;
  const m = s.metrics || {};
  $("pnl-value").textContent = signed(s.pnl);
  $("pnl-value").classList.toggle("negative", s.pnl < -1e-9);
  $("pnl-caption").textContent = s.done
    ? "Valued at contractual settlement"
    : "Marked at current public prices";
  $("cash-value").textContent = number(s.cash);
  $("inventory-value").textContent =
    `${number(s.inventory.A, 0)} / ${number(s.inventory.B, 0)}`;
  $("exposure-caption").textContent =
    `Settlement exposure ${signed(s.inventory.A + 2 * s.inventory.B, 0)}`;
  $("fills-value").textContent = number(m.customerFills, 0);
  $("fees-caption").textContent =
    `Exchange costs ${number((m.fees || 0) + (m.slippage || 0))}`;
  $("session-progress-bar").style.width =
    `${100 * Math.min(1, s.step / s.totalSteps)}%`;
  $("session-progress-label").textContent = s.done
    ? `Settled · ${s.totalSteps} intervals`
    : `Interval ${s.step} / ${s.totalSteps}`;
  $("market-value").textContent =
    `${number(s.market.A)} / ${number(s.market.B)}`;
  $("chart-hint").hidden = Boolean(s.step);
  const orders = s.orders || [];
  const latest = orders[orders.length - 1];
  if (latest) {
    const customerBuy = String(latest.side).toUpperCase() === "BUY";
    $("rfq-description").textContent =
      `Customer ${customerBuy ? "buys" : "sells"} ${number(latest.quantity, 0)} ${latest.product} · dealer ${customerBuy ? "sells" : "buys"}`;
    $("quote-value").textContent =
      latest.quote === null || latest.quote === undefined
        ? "Withdrawn"
        : number(latest.quote);
    $("rfq-status").textContent = latest.filled
      ? "Filled"
      : latest.status === "pending" || latest.status === "OPEN"
        ? "Live quote"
        : "Not filled";
    $("rfq-status").classList.toggle("filled", Boolean(latest.filled));
  } else {
    $("rfq-description").textContent = "No customer order yet.";
    $("quote-value").textContent = "—";
    $("rfq-status").textContent = "Awaiting order";
    $("rfq-status").classList.remove("filled");
  }
  const d = s.decision;
  if (d && d.product) {
    const changed = Math.abs(Number(d.chosen) - Number(d.before)) > 1e-9;
    $("decision-tag").textContent = changed
      ? "Trade selected"
      : "Retain position";
    $("decision-tag").classList.toggle("trade", changed);
    $("decision-description").textContent = changed
      ? `${d.product}: ${number(d.before, 0)} → ${number(d.chosen, 0)} units after costs`
      : `${d.product}: retain ${number(d.before, 0)} units of exposure`;
    $("target-value").textContent = `${number(d.chosen, 0)} ${d.product}`;
    const candidate =
      (d.candidates || []).find((c) => c.selected) ||
      (d.candidates || []).find((c) => Math.abs(c.target - d.chosen) < 1e-9);
    $("advantage-value").textContent =
      candidate && Number.isFinite(candidate.advantage)
        ? `${signed(candidate.advantage, 3)} · model units`
        : "Conventional target";
  } else {
    $("decision-tag").textContent = s.done ? "Settled" : "No action";
    $("decision-tag").classList.remove("trade");
    $("decision-description").textContent = s.done
      ? "Residual inventory is valued at settlement."
      : "The dealer can retain its position.";
    $("target-value").textContent = "—";
    $("advantage-value").textContent = "—";
  }
  const ledgerError = Math.abs(Number(m.ledgerError || 0));
  $("ledger-status").classList.toggle("error", ledgerError > 1e-7);
  $("ledger-status").innerHTML =
    `<span class="status-dot"></span>${ledgerError > 1e-7 ? "Check required" : "Balanced"}`;
  renderLedger(s.trades || []);
  drawCharts();
  if (!playing)
    $("play-label").textContent = sessionFailure
      ? "Restart"
      : s.done
        ? "Run again"
        : s.step
          ? "Resume"
          : "Run session";
}

function renderLedger(trades) {
  $("ledger-count").textContent =
    `${number(trades.length, 0)} transaction${trades.length === 1 ? "" : "s"}`;
  if (!trades.length) {
    $("ledger-body").innerHTML =
      '<tr class="empty-row"><td colspan="7">Run or step the dealer to inspect its transactions.</td></tr>';
    return;
  }
  $("ledger-body").innerHTML = trades
    .slice(-12)
    .reverse()
    .map((t) => {
      const type = String(t.type).toLowerCase();
      const exchange =
        type.includes("hedge") ||
        type.includes("exchange") ||
        type.includes("reset");
      const label = exchange
        ? "Exchange"
        : type.includes("settle")
          ? "Settlement"
          : "Customer";
      const side = String(t.side).toUpperCase();
      const qty =
        side === "SELL" ? -Math.abs(t.quantity) : Math.abs(t.quantity);
      return `<tr><td>${escapeHTML(t.step)}</td><td><span class="transaction-type ${exchange ? "exchange" : ""}"></span>${label} · ${side === "SELL" ? "sell" : "buy"}</td><td>${escapeHTML(t.product)}</td><td>${signed(qty, 0)}</td><td>${number(t.price)}</td><td class="${moneyClass(t.cashChange)}">${signed(t.cashChange)}</td><td>${number(t.fee)}</td></tr>`;
    })
    .join("");
}

function niceRange(values, symmetric = false) {
  let lo = Math.min(...values),
    hi = Math.max(...values);
  if (symmetric) {
    const a = Math.max(Math.abs(lo), Math.abs(hi), 1);
    lo = -a;
    hi = a;
  } else {
    lo = Math.min(0, lo);
    hi = Math.max(0, hi);
  }
  const spread = hi - lo || 2;
  const stepBase = Math.pow(10, Math.floor(Math.log10(spread / 4)));
  const steps = [1, 2, 2.5, 5, 10];
  const step = steps.find((v) => v * stepBase >= spread / 4) * stepBase;
  lo = Math.floor((lo - spread * 0.04) / step) * step;
  hi = Math.ceil((hi + spread * 0.04) / step) * step;
  if (lo === hi) hi = lo + 1;
  return { lo, hi, step };
}

function axisNumber(value) {
  if (Math.abs(value) >= 10000) return `${number(value / 1000, 0)}k`;
  if (Math.abs(value) >= 1000) return `${number(value / 1000, 1)}k`;
  if (Math.abs(value) < 1 && value !== 0) return number(value, 2);
  return number(value, Math.abs(value) < 10 && value % 1 !== 0 ? 1 : 0);
}

function drawChart(svg, series, { height, inventory = false } = {}) {
  const width = Math.max(
    280,
    Math.round(svg.getBoundingClientRect().width || 800),
  );
  const chartHeight = height || (width < 450 ? 200 : 265);
  svg.setAttribute("viewBox", `0 0 ${width} ${chartHeight}`);
  const pad = {
    left: 41,
    right: 10,
    top: inventory ? 8 : 20,
    bottom: inventory ? 22 : 27,
  };
  const w = width - pad.left - pad.right,
    h = chartHeight - pad.top - pad.bottom;
  const history = snapshot.history || [];
  const values = series
    .flatMap((line) => line.values.map((point) => point.value))
    .filter(Number.isFinite);
  const range = niceRange(values.length ? values : [0, 1], inventory);
  const x = (step) => pad.left + (w * step) / Math.max(1, snapshot.totalSteps);
  const y = (value) =>
    pad.top + h - (h * (value - range.lo)) / (range.hi - range.lo);
  const textStyle =
    'font-family="SFMono-Regular,Consolas,Liberation Mono,monospace" font-size="9" fill="#858b80"';
  let content = "";
  const ticks = inventory
    ? [range.lo, 0, range.hi]
    : Array.from(
        {
          length: Math.min(
            7,
            Math.round((range.hi - range.lo) / range.step) + 1,
          ),
        },
        (_, i) => range.lo + i * range.step,
      );
  for (const tick of ticks) {
    if (tick > range.hi + 1e-8) continue;
    const yy = y(tick);
    content += `<line x1="${pad.left}" y1="${yy}" x2="${width - pad.right}" y2="${yy}" stroke="${Math.abs(tick) < 1e-8 ? "#c6cbc1" : "#e9ebe4"}" stroke-width="1" ${Math.abs(tick) < 1e-8 ? "" : 'stroke-dasharray="2 4"'}/><text x="${pad.left - 9}" y="${yy + 3}" text-anchor="end" ${textStyle}>${axisNumber(tick)}</text>`;
  }
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    Math.round(v * snapshot.totalSteps),
  );
  for (const tick of [...new Set(xTicks)])
    content += `<text x="${x(tick)}" y="${chartHeight - 6}" text-anchor="${tick === 0 ? "start" : tick === snapshot.totalSteps ? "end" : "middle"}" ${textStyle}>${tick}</text>`;
  for (const line of series) {
    const points = line.values.filter((point) => Number.isFinite(point.value));
    if (!points.length) continue;
    let path = points
      .map(
        (point, i) =>
          `${i ? "L" : "M"}${x(point.step).toFixed(2)},${y(point.value).toFixed(2)}`,
      )
      .join(" ");
    if (inventory)
      path = points
        .map((point, i) =>
          i
            ? `H${x(point.step).toFixed(2)}V${y(point.value).toFixed(2)}`
            : `M${x(point.step).toFixed(2)},${y(point.value).toFixed(2)}`,
        )
        .join(" ");
    if (series.length === 1 && !inventory && points.length > 1) {
      const area = `${path} L${x(points.at(-1).step)},${y(0)} L${x(points[0].step)},${y(0)} Z`;
      content += `<path d="${area}" fill="${line.color}" opacity="0.045"/>`;
    }
    content += `<path d="${path}" fill="none" stroke="${line.color}" stroke-width="${inventory ? 1.5 : 1.8}" stroke-linejoin="round" stroke-linecap="round"/>`;
    const end = points.at(-1);
    if (!inventory)
      content += `<circle cx="${x(end.step)}" cy="${y(end.value)}" r="3" fill="${line.color}" stroke="#fffefa" stroke-width="1.2"/>`;
  }
  if (snapshot.step && snapshot.step < snapshot.totalSteps)
    content += `<line x1="${x(snapshot.step)}" y1="${pad.top}" x2="${x(snapshot.step)}" y2="${chartHeight - pad.bottom}" stroke="#b4bcb0" stroke-dasharray="2 5" stroke-width=".7"/>`;
  svg.innerHTML = content;
  svg.setAttribute(
    "aria-label",
    `${inventory ? "Inventory" : chartMode === "pnl" ? "Marked PnL" : "Public basis"} across ${snapshot.step} completed intervals`,
  );
}

function drawCharts() {
  if (!snapshot) return;
  const history = snapshot.history || [];
  const series =
    chartMode === "pnl"
      ? [
          {
            color: "#476c85",
            values: history.map((h) => ({ step: h.step, value: h.pnl })),
          },
        ]
      : [
          {
            color: "#aa6250",
            values: history.map((h) => ({ step: h.step, value: h.basis })),
          },
        ];
  $("chart-legend").innerHTML =
    `<span><i class="legend-line ${chartMode === "market" ? "brick" : "blue"}"></i>${chartMode === "market" ? "Public basis 2A − B" : snapshot.done ? "PnL · final point at settlement" : "Marked PnL"}</span>`;
  drawChart($("main-chart"), series);
  drawChart(
    $("inventory-chart"),
    [
      {
        color: "#476c85",
        values: history.map((h) => ({ step: h.step, value: h.qA })),
      },
      {
        color: "#aa6250",
        values: history.map((h) => ({ step: h.step, value: h.qB })),
      },
    ],
    { inventory: true, height: 104 },
  );
}

function comparisonSeeds(count) {
  // Stable disjoint counterfactual path seeds, separated from the displayed session.
  return Array.from(
    { length: count },
    (_, i) => (Number(appliedConfig.seed) + Math.imul(i + 1, 2654435761)) >>> 0,
  );
}

function endComparison() {
  comparing = false;
  $("compare-button").innerHTML =
    'Compare policies <span aria-hidden="true">→</span>';
  $("batch-progress").hidden = true;
}

function startComparison() {
  if (comparing) {
    worker?.postMessage({ id: activeComparisonId, type: "cancel" });
    activeComparisonId += 1;
    endComparison();
    notify(
      "Comparison cancelled. Completed runs were not promoted as a full result.",
    );
    return;
  }
  if (!window.Worker) {
    notify(
      "Your browser does not support background workers. Use a current browser for comparisons.",
    );
    return;
  }
  try {
    if (!worker) {
      worker = new Worker(new URL("./engine/worker.js", import.meta.url), {
        type: "module",
      });
      worker.onmessage = ({ data }) => {
        if (data.id !== activeComparisonId) return;
        if (data.type === "progress") {
          $("batch-progress-bar").style.width =
            `${(100 * data.completed) / data.total}%`;
          $("batch-progress-label").textContent =
            `${number(data.completed, 0)} / ${number(data.total, 0)} policy sessions`;
        } else if (data.type === "result") {
          comparisonFailure = undefined;
          comparisonResult = data.result;
          $("export-comparison").innerHTML =
            'Export comparison <span aria-hidden="true">↓</span>';
          renderComparison(comparisonResult);
          endComparison();
          notify("Comparison complete. All requested paths are included.");
        } else if (data.type === "error") {
          endComparison();
          const message = data.error || data.message || "Unknown error";
          comparisonResult = undefined;
          comparisonFailure = {
            status: "failed",
            message,
            config: {
              ...(data.comparison?.config ||
                data.failure?.config ||
                appliedConfig),
            },
            failure: data.failure || null,
            comparison: data.comparison || null,
          };
          $("comparison-results").innerHTML =
            `<div class="comparison-empty"><p>Comparison stopped.<br>${escapeHTML(message)}<br><small>The failed case is retained for export.</small></p></div>`;
          $("comparison-caption").textContent =
            "No complete comparison was produced. No failing path has been excluded or replaced.";
          $("export-comparison").innerHTML =
            'Export failed run <span aria-hidden="true">↓</span>';
          $("export-comparison").hidden = false;
          notify(`Comparison stopped: ${message}`);
        } else if (data.type === "cancelled") endComparison();
      };
      worker.onerror = (error) => {
        endComparison();
        notify(`Comparison stopped: ${error.message || "worker error"}`);
      };
    }
    const count = Number($("batch-paths").value);
    comparing = true;
    activeComparisonId += 1;
    $("compare-button").innerHTML =
      'Cancel comparison <span aria-hidden="true">×</span>';
    $("batch-progress").hidden = false;
    $("batch-progress-bar").style.width = "0%";
    $("batch-progress-label").textContent = "Starting background computation…";
    worker.postMessage({
      id: activeComparisonId,
      type: "compare",
      config: { ...appliedConfig },
      options: { seeds: comparisonSeeds(count), policies: ["B4", "R2B", "B3"] },
    });
  } catch (error) {
    endComparison();
    notify(error.message);
  }
}

function renderComparison(result) {
  const summary = result.summary || [];
  const best = Math.max(...summary.map((s) => s.meanPnl));
  $("comparison-results").innerHTML =
    `<div class="table-scroll" style="border-top:0"><table class="comparison-table"><thead><tr><th scope="col">Policy</th><th scope="col">Mean PnL</th><th scope="col">95% interval</th><th scope="col">Mean fills</th><th scope="col">Worst path</th></tr></thead><tbody>${summary.map((s) => `<tr class="${s.meanPnl === best ? "best" : ""}"><td><span class="policy-name">${escapeHTML(POLICY_LABELS[s.policy] || s.policy)}</span><span class="policy-code">${escapeHTML(s.policy)} · ${number(s.n, 0)} paths</span></td><td class="${moneyClass(s.meanPnl)}">${signed(s.meanPnl)}</td><td>${number(s.ciLow)} to ${number(s.ciHigh)}</td><td>${number(s.meanFills, 1)}</td><td class="${moneyClass(s.worstPnl)}">${signed(s.worstPnl)}</td></tr>`).join("")}</tbody></table></div><p class="comparison-uncertainty">Descriptive 95% normal intervals across independent paths. The worst observed path is shown; this sample does not establish a tail guarantee.</p>`;
  const paired = (result.paired || []).find((p) => p.policy === "R2B");
  $("comparison-caption").textContent = paired
    ? `R2B − B4 paired mean: ${signed(paired.meanDifference)} · 95% interval [${number(paired.ciLow)}, ${number(paired.ciHigh)}]. Equal parameters; this is a diagnostic, not an equally tuned superiority study.`
    : result.interpretation || "Synthetic comparison complete.";
  $("export-comparison").hidden = false;
}

function download(content, name, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const csvCell = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
function asCSV(rows, keys) {
  return [
    keys.join(","),
    ...rows.map((row) => keys.map((key) => csvCell(row[key])).join(",")),
  ].join("\n");
}

$("config-toggle").addEventListener("click", () => {
  const expanded = document
    .querySelector(".controls-panel")
    .classList.toggle("mobile-expanded");
  $("config-toggle").setAttribute("aria-expanded", String(expanded));
});
$("config-form").addEventListener("input", markPending);
$("config-form").addEventListener("change", markPending);
$("config-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!$("config-form").reportValidity()) return;
  try {
    resetRun(readConfig(), true);
  } catch (error) {
    notify(error.message);
  }
});
$("play-button").addEventListener("click", playRun);
$("step-button").addEventListener("click", () => {
  pauseRun();
  advance();
});
$("reset-button").addEventListener("click", () => {
  resetRun();
  notify("Restarted the same applied configuration and seed.");
});
$("compare-button").addEventListener("click", startComparison);
for (const button of document.querySelectorAll("[data-chart]"))
  button.addEventListener("click", () => {
    chartMode = button.dataset.chart;
    for (const tab of document.querySelectorAll("[data-chart]")) {
      tab.classList.toggle("active", tab === button);
      tab.setAttribute("aria-selected", String(tab === button));
    }
    drawCharts();
  });
$("export-run").addEventListener("click", () => {
  download(
    asCSV(snapshot.trades || [], [
      "step",
      "type",
      "product",
      "side",
      "quantity",
      "price",
      "principal",
      "fee",
      "slippage",
      "cashChange",
      "inventoryAfter",
    ]),
    `inventory-lab-${appliedConfig.seed}-${appliedConfig.policy}-ledger.csv`,
    "text/csv;charset=utf-8",
  );
  notify("Full transaction ledger exported as CSV.");
});
$("export-config").addEventListener("click", () => {
  download(
    JSON.stringify(
      {
        exportedAt: new Date().toISOString(),
        environment: "synthetic",
        status: sessionFailure
          ? "failed"
          : snapshot.done
            ? "settled"
            : "in progress",
        failure: sessionFailure || null,
        scenario: Object.keys(SCENARIOS).find(
          (key) =>
            SCENARIOS[key].basisPersistence ===
              appliedConfig.basisPersistence &&
            SCENARIOS[key].basisVolatility === appliedConfig.basisVolatility,
        ),
        ...snapshot,
      },
      null,
      2,
    ),
    `inventory-lab-${appliedConfig.seed}-${appliedConfig.policy}-session.json`,
    "application/json",
  );
  notify("Configuration and complete session exported as JSON.");
});
$("export-comparison").addEventListener("click", () => {
  if (comparisonFailure) {
    download(
      JSON.stringify(comparisonFailure, null, 2),
      `inventory-lab-${comparisonFailure.config.seed}-failed-comparison.json`,
      "application/json",
    );
    return;
  }
  if (!comparisonResult) return;
  download(
    asCSV(comparisonResult.rows || [], [
      "seed",
      "policy",
      "pnl",
      "fees",
      "customerFills",
      "exchangeTurnover",
      "drawdown",
      "maxExposure",
      "ledgerError",
    ]),
    `inventory-lab-${comparisonResult.config?.seed ?? appliedConfig.seed}-comparison.csv`,
    "text/csv;charset=utf-8",
  );
});
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(drawCharts, 90);
});
document.addEventListener("keydown", (event) => {
  if (
    event.target.closest("input, select, textarea, button, summary, a") ||
    event.ctrlKey ||
    event.metaKey ||
    event.altKey
  )
    return;
  if (event.code === "Space") {
    event.preventDefault();
    playRun();
  } else if (event.code === "ArrowRight" && !playing) {
    event.preventDefault();
    advance();
  }
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && playing) {
    pauseRun();
    renderRunState();
  }
});

try {
  for (const key of CONFIG_FIELDS)
    if (DEFAULT_CONFIG[key] !== undefined) $(key).value = DEFAULT_CONFIG[key];
  syncRangeLabels();
  appliedConfig = readConfig();
  resetRun(appliedConfig);
} catch (error) {
  notify(`Unable to start the simulator: ${error.message}`);
}
