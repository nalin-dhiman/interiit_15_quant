import { eventRandom } from "./random.js";
import {
  applyQuoteUpdates,
  auditLedger,
  bookTrade,
  customerDelta,
  execution,
  observeEquity,
  reserveQuotes,
  roundToTick,
} from "./accounting.js";
import { chooseHedge, POLICIES, quotePrice, sizeSupport } from "./policies.js";

export { POLICIES };
export {
  auditLedger,
  customerDelta,
  reserveQuotes,
  applyQuoteUpdates,
} from "./accounting.js";

export const BATCH_LIMITS = Object.freeze({
  maxSeeds: 200,
  maxSteps: 120,
  maxRfqsPerStep: 6,
});

export const DEFAULT_CONFIG = Object.freeze({
  seed: 42,
  steps: 80,
  policy: "B4",
  rfqsPerStep: 4,
  quoteEdge: 0.1,
  inventorySkew: 0.006,
  riskAversion: 0.006,
  hedgeFee: 0.015,
  hedgeSlippage: 0.005,
  resetFixedCost: 0,
  basisPersistence: 0.92,
  basisVolatility: 0.14,
  priceVolatility: 0.14,
  initialBasis: 0,
  initialPrice: 100,
  inventoryLimit: 20,
  rfqSize: 3,
  forecastWeight: 0.35,
  targetFraction: 1,
  continuationDepth: 2,
  decisionMargin: 0.002,
});

const NUMERIC_BOUNDS = {
  steps: [1, 120, true],
  rfqsPerStep: [1, 6, true],
  quoteEdge: [0.01, 1],
  inventorySkew: [0, 0.1],
  riskAversion: [0, 0.1],
  hedgeFee: [0, 1],
  hedgeSlippage: [0, 1],
  resetFixedCost: [0, 5],
  basisPersistence: [0, 1.02],
  basisVolatility: [0, 2],
  priceVolatility: [0, 2],
  initialBasis: [-10, 10],
  initialPrice: [50, 1000],
  inventoryLimit: [2, 40, true],
  rfqSize: [1, 10, true],
  forecastWeight: [0, 1],
  targetFraction: [0.1, 1],
  continuationDepth: [1, 3, true],
  decisionMargin: [0, 5],
};

export function normalizeConfig(input = {}) {
  const config = { ...DEFAULT_CONFIG };
  for (const key of Object.keys(DEFAULT_CONFIG)) {
    if (input[key] !== undefined) config[key] = input[key];
  }
  if (!POLICIES.some(({ id }) => id === config.policy))
    throw new Error("Unknown policy. Choose B3, B4, or R2B.");
  if (
    !["string", "number"].includes(typeof config.seed) ||
    String(config.seed).length > 100
  )
    throw new Error("Seed must be a short string or finite number.");
  if (typeof config.seed === "number" && !Number.isFinite(config.seed))
    throw new Error("Seed must be finite.");
  for (const [key, [lower, upper, integer]] of Object.entries(NUMERIC_BOUNDS)) {
    const value = Number(config[key]);
    if (
      !Number.isFinite(value) ||
      value < lower ||
      value > upper ||
      (integer && !Number.isInteger(value))
    ) {
      throw new Error(
        `${key} must be ${integer ? "an integer " : ""}between ${lower} and ${upper}.`,
      );
    }
    config[key] = value;
  }
  return config;
}

const MODEL = Object.freeze({
  version: "1.0.0",
  drawdownSampling:
    "Observed price, trade and terminal-settlement events; no claim about unobserved continuous paths.",
  label: "Two-product synthetic RFQ laboratory",
  assumptions: [
    "A follows an exogenous arithmetic random walk; B = 2A minus an autoregressive basis.",
    "Final settlement values are A at its last price and B at twice A. Terminal holdings are valued, not forcibly liquidated.",
    "Customer limit offsets and opponent quote offsets are independent known uniforms. They are private to the matching engine.",
    "Quotes receive immediate acknowledgements and can be replaced until the batch deadline. All possible fill subsets reserve capacity.",
    "Exchange hedges execute and confirm between resolved RFQ batches. Finite depth, asynchronous transport and market impact are absent.",
    "R2B uses a finite continuation approximation with frozen within-batch prices and deterministic expected future basis; its comparison margin is heuristic.",
    "Risk preferences are separate from ledger PnL. These browser runs do not reproduce or validate a competition exchange.",
  ],
  responseAdaptation:
    "Disabled. A response model requires separate held-out calibration and evidence of incremental value.",
});

function markedEquity(session, final = session.done) {
  const prices = final
    ? { A: session.market.A, B: 2 * session.market.A }
    : session.market;
  return (
    session.cash +
    Object.keys(session.inventory).reduce(
      (total, product) =>
        total +
        session.inventory[product] *
          prices[product] *
          session.specs[product].multiplier,
      0,
    )
  );
}

function historyPoint(session) {
  const equity = observeEquity(session);
  const pnl = equity - session.initialEquity;
  session.metrics.maxExposure = Math.max(
    session.metrics.maxExposure,
    Math.abs(session.inventory.A) + Math.abs(session.inventory.B),
  );
  session.metrics.ledgerError = Math.max(
    session.metrics.ledgerError,
    auditLedger(session).error,
  );
  if (session.metrics.ledgerError > 1e-7)
    throw new Error(
      "Cash and inventory failed independent ledger reconciliation.",
    );
  return {
    step: session.step,
    A: session.market.A,
    B: session.market.B,
    basis: session.market.basis,
    cash: session.cash,
    equity,
    pnl,
    qA: session.inventory.A,
    qB: session.inventory.B,
    fees: session.metrics.fees,
    settled: session.done,
  };
}

export function createSession(input = {}) {
  const config = normalizeConfig(input);
  const initialInventory = { A: 0, B: 0 };
  const session = {
    config,
    step: 0,
    done: false,
    market: {
      A: config.initialPrice,
      B: 2 * config.initialPrice - config.initialBasis,
      basis: config.initialBasis,
    },
    specs: Object.fromEntries(
      ["A", "B"].map((product) => [
        product,
        {
          lower: -config.inventoryLimit,
          upper: config.inventoryLimit,
          tick: 0.01,
          lot: 1,
          multiplier: 1,
        },
      ]),
    ),
    inventory: { ...initialInventory },
    initialInventory,
    initialCash: 0,
    initialEquity: 0,
    cash: 0,
    history: [],
    trades: [],
    orders: [],
    activeOrders: [],
    rfqHistory: [],
    events: [],
    liveQuotes: {},
    decision: null,
    decisions: [],
    metrics: {
      customerFills: 0,
      customerNotional: 0,
      customerSpread: 0,
      customerRequests: 0,
      fees: 0,
      slippage: 0,
      exchangeTurnover: 0,
      exchangeQuantity: 0,
      exchangeTrades: 0,
      drawdown: 0,
      peakEquity: 0,
      maxExposure: 0,
      riskPenalty: 0,
      penalties: 0,
      ledgerError: 0,
      quoteUpdates: 0,
      capacityWithdrawals: 0,
    },
  };
  session.history.push(historyPoint(session));
  return session;
}

function moveMarket(session, tick) {
  const { config, step } = session;
  const shocks = eventRandom(config.seed, "prices", step, tick);
  const micro = 3;
  const persistence = config.basisPersistence ** (1 / micro);
  const varianceScale =
    Math.abs(config.basisPersistence - 1) < 1e-8
      ? 1 / Math.sqrt(micro)
      : Math.sqrt(
          Math.abs(
            (1 - persistence * persistence) /
              (1 - config.basisPersistence * config.basisPersistence),
          ),
        );
  session.market.A +=
    (config.priceVolatility * shocks.normal()) / Math.sqrt(micro);
  session.market.basis =
    persistence * session.market.basis +
    config.basisVolatility * varianceScale * shocks.normal();
  session.market.B = 2 * session.market.A - session.market.basis;
  if (session.market.A <= 0 || session.market.B <= 0) {
    const error = new Error(
      "The arithmetic price model produced a nonpositive price; the path is retained as a failure, not clipped.",
    );
    error.failure = {
      stage: "market",
      seed: config.seed,
      step,
      tick,
      config: { ...config },
      market: { ...session.market },
      partialSnapshot: snapshotSession(session),
    };
    throw error;
  }
  observeEquity(session);
}

function makeRequests(session) {
  const sizes = sizeSupport(session.config);
  const requests = [];
  for (let index = 0; index < session.config.rfqsPerStep; index += 1) {
    const customer = eventRandom(
      session.config.seed,
      "customer",
      session.step,
      index,
    );
    const product = customer.uniform() < 0.5 ? "A" : "B";
    const side = customer.uniform() < 0.5 ? "BUY" : "SELL";
    const quantity = sizes[Math.floor(customer.uniform() * sizes.length)];
    const reference = session.market[product];
    const offset = 0.04 + 0.26 * customer.uniform();
    const privateLimit = reference + (side === "BUY" ? 1 : -1) * offset;
    // Private attributes never enter quote or hedge functions.
    requests.push({
      public: {
        id: `${session.step}-${index + 1}`,
        product,
        side,
        quantity,
        reference,
        deadline: session.step,
      },
      privateLimit,
      tie: eventRandom(
        session.config.seed,
        "ties",
        session.step,
        index,
      ).uniform(),
    });
  }
  return requests;
}

function opponentQuote(session, request, tick, index) {
  const primitive = eventRandom(
    session.config.seed,
    "opponents",
    session.step,
    `${index}/${tick}`,
  );
  const offset = 0.03 + 0.25 * primitive.uniform();
  const direction = request.side === "BUY" ? 1 : -1;
  return roundToTick(
    session.market[request.product] + direction * offset,
    session.specs[request.product].tick,
    request.side,
  );
}

function updateQuotes(session, requests, tick) {
  const { updates } = reserveQuotes(
    requests.map(({ public: request }) => request),
    session.inventory,
    session.specs,
    (request) =>
      quotePrice(
        request,
        session.inventory,
        session.market,
        session.specs,
        session.config,
      ),
  );
  applyQuoteUpdates(session.liveQuotes, updates, tick);
  session.metrics.quoteUpdates += Object.values(updates).filter(
    (value) => value !== null,
  ).length;
  session.metrics.capacityWithdrawals += Object.values(updates).filter(
    (value) => value === null,
  ).length;
}

function hedgeBetweenBatches(session) {
  if (Object.keys(session.liveQuotes).length)
    throw new Error(
      "A hedge cannot allocate capacity reserved by live quotes.",
    );
  const decisions = [];
  for (const product of ["A", "B"]) {
    const decision = chooseHedge(
      session.inventory[product],
      product,
      session.market,
      session.specs[product],
      session.config,
      session.config.steps - session.step + 1,
    );
    decisions.push(decision);
    const trade = execution(
      session.inventory[product],
      decision.chosen,
      session.market[product],
      session.specs[product],
      session.config,
    );
    if (!trade)
      throw new Error(
        "The policy proposed an infeasible exchange transaction.",
      );
    if (trade.delta) {
      bookTrade(session, {
        type: "exchange",
        product,
        delta: trade.delta,
        price: trade.price,
        fee: trade.fee,
        slippage: trade.slippage,
      });
      session.events.push({
        step: session.step,
        type: "hedge",
        text: `${product}: exchange ${trade.delta > 0 ? "buy" : "sell"} ${Math.abs(trade.delta)}, inventory ${trade.before} → ${trade.target}.`,
      });
    }
  }
  session.decisions = decisions;
  session.decision =
    decisions.find(({ before, chosen }) => before !== chosen) || decisions[1];
}

export function stepSession(session) {
  if (session.done) return snapshotSession(session);
  session.step += 1;
  moveMarket(session, 0);
  hedgeBetweenBatches(session);
  const requests = makeRequests(session);
  session.activeOrders = requests.map(({ public: request }) => ({
    ...request,
  }));
  session.metrics.customerRequests += requests.length;
  for (let tick = 0; tick < 3; tick += 1) {
    if (tick > 0) moveMarket(session, tick);
    updateQuotes(session, requests, tick);
    session.metrics.riskPenalty +=
      (session.config.riskAversion *
        (session.inventory.A ** 2 + session.inventory.B ** 2)) /
      3;
  }
  const outcomes = [];
  for (const [
    index,
    { public: request, privateLimit, tie },
  ] of requests.entries()) {
    const live = session.liveQuotes[request.id];
    const quote = live?.price ?? null;
    const opponent = opponentQuote(session, request, 2, index);
    const direction = request.side === "BUY" ? 1 : -1;
    const eligible =
      quote !== null && direction * (quote - privateLimit) <= 1e-9;
    const better = quote !== null && direction * (quote - opponent) < -1e-9;
    const tied = quote !== null && Math.abs(quote - opponent) <= 1e-9;
    const filled = eligible && (better || (tied && tie < 0.5));
    if (filled) {
      bookTrade(session, {
        type: "customer",
        product: request.product,
        delta: customerDelta(request.side, request.quantity),
        price: quote,
        rfqId: request.id,
      });
    }
    outcomes.push({
      ...request,
      quote,
      opponentQuote: opponent,
      filled,
      status: quote === null ? "withdrawn" : filled ? "filled" : "not filled",
      acknowledgedAt: live?.acknowledgedAt ?? null,
      // A losing quote does not reveal the hidden customer limit.
    });
  }
  session.orders = outcomes;
  session.activeOrders = [];
  session.rfqHistory.push(...outcomes);
  session.liveQuotes = {};
  const fills = outcomes.filter(({ filled }) => filled).length;
  session.events.push({
    step: session.step,
    type: "batch",
    text: `${fills} of ${requests.length} RFQs filled at the matching deadline.`,
  });
  if (session.step >= session.config.steps) {
    session.done = true;
    session.events.push({
      step: session.step,
      type: "settlement",
      text: "Terminal holdings valued at the settlement graph; no forced exchange liquidation.",
    });
  }
  session.history.push(historyPoint(session));
  return snapshotSession(session);
}

export function snapshotSession(session) {
  const equity = markedEquity(session);
  const terminalResidual = session.done
    ? session.inventory.B * (2 * session.market.A - session.market.B)
    : 0;
  // A fresh JSON-safe object prevents rendering code from mutating the ledger.
  return {
    config: { ...session.config },
    step: session.step,
    totalSteps: session.config.steps,
    done: session.done,
    settled: session.done,
    market: { ...session.market },
    inventory: { ...session.inventory },
    cash: session.cash,
    equity,
    pnl: equity - session.initialEquity,
    riskAdjustedValue:
      equity - session.initialEquity - session.metrics.riskPenalty,
    terminalResidual,
    settlementPrices: session.done
      ? { A: session.market.A, B: 2 * session.market.A }
      : null,
    metrics: { ...session.metrics },
    history: session.history.map((row) => ({ ...row })),
    orders: session.orders.map((row) => ({ ...row })),
    activeOrders: session.activeOrders.map((row) => ({ ...row })),
    liveQuotes: Object.fromEntries(
      Object.entries(session.liveQuotes).map(([id, quote]) => [
        id,
        { ...quote },
      ]),
    ),
    rfqHistory: session.rfqHistory.map((row) => ({ ...row })),
    trades: session.trades.map((row) => ({ ...row })),
    decision: session.decision ? structuredDecision(session.decision) : null,
    decisions: session.decisions.map(structuredDecision),
    events: session.events.map((row) => ({ ...row })),
    model: { ...MODEL, assumptions: [...MODEL.assumptions] },
  };
}

function structuredDecision(decision) {
  return {
    ...decision,
    candidates: decision.candidates.map((row) => ({ ...row })),
  };
}

export function runSession(config = {}) {
  const session = createSession(config);
  while (!session.done) stepSession(session);
  return snapshotSession(session);
}

function sampleSummary(values) {
  const n = values.length;
  const mean = values.reduce((sum, value) => sum + value, 0) / n;
  const variance =
    n > 1
      ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (n - 1)
      : 0;
  const standardError = Math.sqrt(variance / n);
  return {
    n,
    mean,
    standardDeviation: Math.sqrt(variance),
    standardError,
    ciLow: n > 1 ? mean - 1.96 * standardError : null,
    ciHigh: n > 1 ? mean + 1.96 * standardError : null,
  };
}

export function createComparison(input = {}, options = {}) {
  const config = normalizeConfig(input);
  const count = options.seeds === undefined ? 12 : options.seeds;
  if (
    !Array.isArray(count) &&
    (!Number.isInteger(Number(count)) ||
      Number(count) < 1 ||
      Number(count) > BATCH_LIMITS.maxSeeds)
  )
    throw new Error(
      "Comparison seed count must be an integer between 1 and 200.",
    );
  const seeds = Array.isArray(count)
    ? [...count]
    : Array.from(
        { length: Number(count) },
        (_, index) => `${config.seed}/comparison/${index}`,
      );
  if (
    seeds.length < 1 ||
    seeds.length > BATCH_LIMITS.maxSeeds ||
    new Set(seeds.map(String)).size !== seeds.length
  )
    throw new Error("Comparison requires 1–200 distinct seeds.");
  const policies = options.policies
    ? [...options.policies]
    : ["B3", "B4", "R2B"];
  if (
    !policies.length ||
    policies.length > 3 ||
    new Set(policies).size !== policies.length ||
    policies.some((policy) => !POLICIES.some(({ id }) => id === policy))
  )
    throw new Error("Comparison policies must be distinct B3, B4, or R2B.");
  return {
    config,
    seeds,
    policies,
    completed: 0,
    total: seeds.length * policies.length,
    rows: [],
    done: false,
  };
}

export function advanceComparison(comparison, maxSteps = 1) {
  for (let index = 0; index < maxSteps && !comparison.done; index += 1) {
    const seed =
      comparison.seeds[
        Math.floor(comparison.completed / comparison.policies.length)
      ];
    const policy =
      comparison.policies[comparison.completed % comparison.policies.length];
    if (!comparison.currentSession)
      comparison.currentSession = createSession({
        ...comparison.config,
        seed,
        policy,
      });
    const snapshot = stepSession(comparison.currentSession);
    if (!snapshot.done) continue;
    comparison.rows.push({
      seed,
      policy,
      pnl: snapshot.pnl,
      fees: snapshot.metrics.fees,
      slippage: snapshot.metrics.slippage,
      customerFills: snapshot.metrics.customerFills,
      exchangeTurnover: snapshot.metrics.exchangeTurnover,
      drawdown: snapshot.metrics.drawdown,
      maxExposure: snapshot.metrics.maxExposure,
      ledgerError: snapshot.metrics.ledgerError,
      riskPenalty: snapshot.metrics.riskPenalty,
      terminalResidual: snapshot.terminalResidual,
    });
    comparison.completed += 1;
    comparison.currentSession = null;
    comparison.done = comparison.completed === comparison.total;
  }
  return {
    completed: comparison.completed,
    total: comparison.total,
    done: comparison.done,
    activeStep: comparison.currentSession?.step ?? 0,
    totalSteps: comparison.config.steps,
  };
}

export function stepComparison(comparison, count = 1) {
  const target = Math.min(comparison.total, comparison.completed + count);
  while (comparison.completed < target)
    advanceComparison(comparison, comparison.config.steps);
  return {
    completed: comparison.completed,
    total: comparison.total,
    done: comparison.done,
  };
}

export function summarizeComparison(comparison) {
  if (!comparison.done)
    throw new Error("The comparison must finish before it is summarized.");
  const summary = comparison.policies.map((policy) => {
    const rows = comparison.rows.filter((row) => row.policy === policy);
    const stats = sampleSummary(rows.map(({ pnl }) => pnl));
    const average = (key) =>
      rows.reduce((total, row) => total + row[key], 0) / rows.length;
    return {
      policy,
      n: stats.n,
      meanPnl: stats.mean,
      stdPnl: stats.standardDeviation,
      standardError: stats.standardError,
      ciLow: stats.ciLow,
      ciHigh: stats.ciHigh,
      meanFees: average("fees"),
      meanFills: average("customerFills"),
      meanTurnover: average("exchangeTurnover"),
      worstPnl: Math.min(...rows.map(({ pnl }) => pnl)),
      meanDrawdown: average("drawdown"),
    };
  });
  const baseline = comparison.policies.includes("B4")
    ? "B4"
    : comparison.policies[0];
  const paired = comparison.policies
    .filter((policy) => policy !== baseline)
    .map((policy) => {
      const differences = comparison.seeds.map((seed) => {
        const first = comparison.rows.find(
          (row) => row.seed === seed && row.policy === policy,
        );
        const second = comparison.rows.find(
          (row) => row.seed === seed && row.policy === baseline,
        );
        return first.pnl - second.pnl;
      });
      const stats = sampleSummary(differences);
      return {
        policy,
        baseline,
        n: stats.n,
        meanDifference: stats.mean,
        standardError: stats.standardError,
        ciLow: stats.ciLow,
        ciHigh: stats.ciHigh,
      };
    });
  return {
    config: { ...comparison.config },
    seeds: [...comparison.seeds],
    policies: [...comparison.policies],
    rows: comparison.rows.map((row) => ({ ...row })),
    summary,
    paired,
    interpretation:
      "Synthetic matched-seed comparison. Intervals are descriptive normal approximations across independent seeds; they are not market validation, a tail guarantee, or an adjustment for repeated experimentation. Drawdown is sampled at observed price, trade and settlement events.",
  };
}

export function comparePolicies(config = {}, options = {}) {
  const comparison = createComparison(config, options);
  while (!comparison.done) stepComparison(comparison);
  return summarizeComparison(comparison);
}
