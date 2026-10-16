import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_CONFIG,
  advanceComparison,
  auditLedger,
  comparePolicies,
  createComparison,
  createSession,
  normalizeConfig,
  runSession,
  snapshotSession,
  stepSession,
} from "../web/engine/index.js";
import {
  applyQuoteUpdates,
  bookTrade,
  customerDelta,
  exchangeFeasible,
  execution,
  observeEquity,
  reserveQuotes,
  roundToLot,
  roundToTick,
} from "../web/engine/accounting.js";
import {
  bandDecision,
  continuationDecision,
  futureFlowKernel,
  knownFillProbability,
  quotePrice,
} from "../web/engine/policies.js";
import { randomStream } from "../web/engine/random.js";

function close(actual, expected, tolerance = 1e-8) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected}`,
  );
}

test("customer signs, multiplier principal and cash fees reconcile by hand", () => {
  const session = createSession();
  session.specs.A.multiplier = 5;
  bookTrade(session, {
    type: "customer",
    product: "A",
    delta: customerDelta("BUY", 2),
    price: 101,
    fee: 0.2,
  });
  assert.equal(session.inventory.A, -2);
  close(session.cash, 1009.8);
  bookTrade(session, {
    type: "customer",
    product: "A",
    delta: customerDelta("SELL", 3),
    price: 99,
  });
  assert.equal(session.inventory.A, 1);
  close(session.cash, -475.2);
  bookTrade(session, {
    type: "exchange",
    product: "A",
    delta: -1,
    price: 100,
    fee: 0.3,
  });
  assert.equal(session.inventory.A, 0);
  close(session.cash, 24.5);
  close(session.metrics.fees, 0.5);
  assert.ok(auditLedger(session).reconciled);
});

test("customer execution is not clipped or rejected when it breaches a limit", () => {
  const session = createSession({ inventoryLimit: 2 });
  bookTrade(session, { type: "customer", product: "A", delta: 7, price: 100 });
  assert.equal(session.inventory.A, 7);
  assert.equal(session.cash, -700);
  assert.ok(auditLedger(session).reconciled);
});

test("exchange action may improve an existing breach but may not worsen it", () => {
  const spec = { lower: -2, upper: 2, lot: 1 };
  assert.equal(exchangeFeasible(5, 4, spec), true);
  assert.equal(exchangeFeasible(5, 0, spec), true);
  assert.equal(exchangeFeasible(5, -4, spec), true);
  assert.equal(exchangeFeasible(5, -5, spec), true);
  assert.equal(exchangeFeasible(5, -6, spec), false);
  assert.equal(exchangeFeasible(5, 6, spec), false);
  assert.equal(exchangeFeasible(0, 3, spec), false);
  assert.equal(exchangeFeasible(0, 0.5, spec), false);
});

test("all possible subsets of simultaneous quotes share capacity", () => {
  const requests = ["BUY", "SELL", "BUY", "SELL"].map((side, index) => ({
    id: String(index),
    product: "A",
    side,
    quantity: 3,
  }));
  const { updates } = reserveQuotes(
    requests,
    { A: 0 },
    { A: { lower: -4, upper: 4 } },
    () => 100,
  );
  assert.deepEqual(updates, { 0: 100, 1: 100, 2: null, 3: null });
  const admitted = requests.filter(({ id }) => updates[id] !== null);
  for (let mask = 0; mask < 2 ** admitted.length; mask += 1) {
    const q = admitted.reduce(
      (total, request, index) =>
        total +
        ((mask >> index) & 1
          ? customerDelta(request.side, request.quantity)
          : 0),
      0,
    );
    assert.ok(q >= -4 && q <= 4);
  }
});

test("omission preserves an acknowledged quote, replacement supersedes it, null withdraws", () => {
  const live = {};
  applyQuoteUpdates(live, { a: 100, b: 101 }, 0);
  applyQuoteUpdates(live, { a: 102 }, 1);
  assert.deepEqual(live, {
    a: { price: 102, acknowledgedAt: 1 },
    b: { price: 101, acknowledgedAt: 0 },
  });
  applyQuoteUpdates(live, { a: null }, 2);
  assert.equal(live.a, undefined);
  assert.equal(live.b.price, 101);
});

test("lot rounding and executable tick prices are included in action costs", () => {
  const session = createSession({ hedgeFee: 0.02, hedgeSlippage: 0.005 });
  const trade = execution(2, 0.3, 100.003, session.specs.A, session.config);
  assert.equal(trade.delta, -1);
  assert.equal(trade.target, 1);
  close(trade.price, 99.99);
  close(trade.cost, 0.033);
  assert.equal(roundToLot(-1.99), -1);
  assert.equal(roundToTick(10.001, 0.01, "BUY"), 10.01);
  assert.equal(roundToTick(10.009, 0.01, "SELL"), 10);
});

test("same seed is bit-reproducible and changing policy does not change primitive prices or RFQs", () => {
  const first = runSession({
    seed: "independent-tape",
    steps: 20,
    policy: "B4",
  });
  assert.deepEqual(
    first,
    runSession({ seed: "independent-tape", steps: 20, policy: "B4" }),
  );
  const second = runSession({
    seed: "independent-tape",
    steps: 20,
    policy: "R2B",
  });
  assert.deepEqual(
    first.history.map(({ A, B }) => [A, B]),
    second.history.map(({ A, B }) => [A, B]),
  );
  assert.deepEqual(
    first.orders.map(({ id, product, side, quantity, opponentQuote }) => [
      id,
      product,
      side,
      quantity,
      opponentQuote,
    ]),
    second.orders.map(({ id, product, side, quantity, opponentQuote }) => [
      id,
      product,
      side,
      quantity,
      opponentQuote,
    ]),
  );
  const changedCount = runSession({
    seed: "independent-tape",
    steps: 20,
    rfqsPerStep: 6,
  });
  assert.deepEqual(
    first.history.map(({ A, B }) => [A, B]),
    changedCount.history.map(({ A, B }) => [A, B]),
  );
});

test("price, customer and opponent streams are distinct deterministic primitives", () => {
  const price = randomStream("seed", "prices");
  const customer = randomStream("seed", "customers");
  assert.notEqual(price.uniform(), customer.uniform());
  const fresh = randomStream("seed", "prices");
  assert.equal(randomStream("seed", "prices").uniform(), fresh.uniform());
});

test("policy snapshots contain no private customer limits or future innovations", () => {
  const snapshot = runSession({ steps: 2 });
  const text = JSON.stringify(snapshot);
  assert.equal(text.includes("privateLimit"), false);
  assert.equal(text.includes("futureInnovations"), false);
  const session = createSession();
  const detached = snapshotSession(session);
  detached.inventory.A = 10000;
  detached.config.steps = 1;
  assert.equal(session.inventory.A, 0);
  assert.equal(session.config.steps, DEFAULT_CONFIG.steps);
});

test("invalid controls are rejected rather than silently clamped", () => {
  for (const input of [
    { steps: 121 },
    { steps: 3.5 },
    { rfqsPerStep: 7 },
    { hedgeFee: -1 },
    { policy: "R3" },
    { seed: Infinity },
    { targetFraction: 0 },
  ]) {
    assert.throws(() => normalizeConfig(input));
  }
  assert.throws(() => createComparison({}, { seeds: 201 }));
  assert.throws(() => createComparison({}, { seeds: ["same", "same"] }));
  assert.throws(() => createComparison({}, { seeds: 1.5 }));
});

test("terminal settlement retains inventory and adds the exact graph residual without a forced hedge", () => {
  const result = runSession({
    steps: 1,
    rfqsPerStep: 1,
    quoteEdge: 1,
    riskAversion: 0,
    initialBasis: 1,
    basisPersistence: 1,
    basisVolatility: 0,
    priceVolatility: 0,
  });
  assert.equal(result.inventory.B, 20);
  close(result.cash, -3980.5);
  close(result.terminalResidual, 20);
  close(result.pnl, 19.5);
  assert.equal(
    result.trades.filter(({ type }) => type === "exchange").length,
    1,
  );
  assert.equal(result.metrics.customerFills, 0);
  assert.equal(result.history.at(-1).pnl, result.pnl);
  assert.equal(result.rfqHistory.length, 1);
  assert.equal(result.riskAdjustedValue, result.pnl);
});

test("fees remain losses in a no-customer round trip", () => {
  const session = createSession({ hedgeFee: 0.02, hedgeSlippage: 0 });
  for (const delta of [3, -3]) {
    const trade = execution(
      session.inventory.A,
      session.inventory.A + delta,
      100,
      session.specs.A,
      session.config,
    );
    bookTrade(session, {
      type: "exchange",
      product: "A",
      delta: trade.delta,
      price: trade.price,
      fee: trade.fee,
    });
  }
  close(session.cash, -0.12);
  assert.equal(session.inventory.A, 0);
});

test("drawdown includes observed price moves and trade costs within a batch", () => {
  const session = createSession();
  bookTrade(session, { type: "exchange", product: "A", delta: 1, price: 100 });
  session.market.A = 110;
  observeEquity(session);
  assert.equal(session.metrics.peakEquity, 10);
  session.market.A = 103;
  observeEquity(session);
  assert.equal(session.metrics.drawdown, 7);
  bookTrade(session, {
    type: "exchange",
    product: "A",
    delta: -1,
    price: 103,
    fee: 0.5,
  });
  assert.equal(session.metrics.drawdown, 7.5);
});

test("conventional band agrees with an independently enumerated economic objective", () => {
  const session = createSession({
    riskAversion: 0.02,
    hedgeFee: 0.03,
    hedgeSlippage: 0,
    initialBasis: 1,
  });
  const decision = bandDecision(
    3,
    "B",
    session.market,
    session.specs.B,
    session.config,
    5,
  );
  const rows = Array.from({ length: 41 }, (_, index) => index - 20).map(
    (q) => ({ q, value: q - 0.1 * q * q - 0.03 * Math.abs(q - 3) }),
  );
  rows.sort((a, b) => b.value - a.value);
  assert.equal(decision.target, rows[0].q);
  close(decision.value, rows[0].value);
});

test("known fill law includes eligibility, rounded opponent quotes and tie allocation", () => {
  const reference = 100.003;
  const edge = 0.097;
  for (const side of ["BUY", "SELL"]) {
    const direction = side === "BUY" ? 1 : -1;
    const actualEdge = side === "BUY" ? edge : 0.103;
    const n = 50000;
    let winning = 0;
    for (let index = 0; index < n; index += 1) {
      const offset = 0.03 + (0.25 * (index + 0.5)) / n;
      const opponent = roundToTick(reference + direction * offset, 0.01, side);
      const opponentEdge = direction * (opponent - reference);
      winning +=
        actualEdge < opponentEdge - 1e-9
          ? 1
          : Math.abs(actualEdge - opponentEdge) <= 1e-9
            ? 0.5
            : 0;
    }
    const eligibility = (0.3 - actualEdge) / 0.26;
    close(
      knownFillProbability(actualEdge, 0.01, reference, side),
      (winning / n) * eligibility,
      2e-5,
    );
  }
  assert.equal(knownFillProbability(1), 0);
  close(knownFillProbability(-1), 1);
});

test("one-slot finite kernel agrees with independent RFQ enumeration", () => {
  const session = createSession({
    rfqsPerStep: 1,
    rfqSize: 1,
    inventorySkew: 0,
    inventoryLimit: 2,
  });
  const rows = futureFlowKernel(
    1,
    "A",
    session.market,
    session.specs.A,
    session.config,
  );
  const probability = knownFillProbability(0.1);
  close(
    rows.find(({ inventory }) => inventory === 0).probability,
    0.25 * probability,
  );
  close(
    rows.find(({ inventory }) => inventory === 2).probability,
    0.25 * probability,
  );
  close(
    rows.find(({ inventory }) => inventory === 1).probability,
    1 - 0.5 * probability,
  );
  close(
    rows.reduce((total, row) => total + row.rewardMass, 0),
    0.05 * probability,
  );
});

test("multi-RFQ finite kernel conserves probability and respects all-subset reservations", () => {
  const session = createSession({
    rfqsPerStep: 6,
    inventoryLimit: 3,
    rfqSize: 3,
  });
  for (const q of [-3, -2, 0, 2, 3]) {
    const rows = futureFlowKernel(
      q,
      "B",
      session.market,
      session.specs.B,
      session.config,
    );
    close(
      rows.reduce((sum, row) => sum + row.probability, 0),
      1,
    );
    assert.ok(rows.every(({ inventory }) => inventory >= -3 && inventory <= 3));
  }
});

test("one-step R2 values agree with direct independent enumeration including reward and cash costs", () => {
  const session = createSession({
    rfqsPerStep: 1,
    rfqSize: 1,
    inventoryLimit: 3,
    initialBasis: 0.2,
    riskAversion: 0.01,
    inventorySkew: 0.01,
  });
  const decision = continuationDecision(
    2,
    "B",
    session.market,
    session.specs.B,
    session.config,
    1,
  );
  for (const candidate of decision.candidates) {
    const t = candidate.target;
    const trade = execution(
      2,
      t,
      session.market.B,
      session.specs.B,
      session.config,
    );
    let expected = -trade.cost + 0.5 * (t * 0.2 - 0.01 * t * t);
    for (const side of ["BUY", "SELL"]) {
      const delta = side === "BUY" ? -1 : 1;
      const admitted = t + delta >= -3 && t + delta <= 3;
      const quote = quotePrice(
        { product: "B", side },
        { B: t },
        session.market,
        session.specs,
        session.config,
      );
      const edge = (side === "BUY" ? 1 : -1) * (quote - session.market.B);
      const fill = admitted
        ? knownFillProbability(edge, 0.01, session.market.B, side)
        : 0;
      const filledValue =
        -delta * (quote - session.market.B) + (t + delta) * 0.2 - 0.01 * t * t;
      const noFillValue = t * 0.2 - 0.01 * t * t;
      expected += 0.25 * (fill * filledValue + (1 - fill) * noFillValue);
    }
    close(candidate.value, expected);
  }
  const baseline = decision.candidates.find(
    ({ target }) => target === decision.baseline,
  ).value;
  const selected = decision.candidates.find(({ selected }) => selected).value;
  assert.ok(selected >= baseline - 1e-8);
  assert.equal(decision.errorCertificate, false);
});

test("risk preference is separately recorded and never subtracted from ledger PnL", () => {
  const result = runSession({ steps: 20, seed: "risk-separation" });
  assert.ok(result.metrics.riskPenalty > 0);
  close(result.riskAdjustedValue, result.pnl - result.metrics.riskPenalty);
  close(
    result.pnl,
    result.cash +
      result.inventory.A * result.settlementPrices.A +
      result.inventory.B * result.settlementPrices.B,
  );
});

test("continuation risk uses inventory held before deadline fills, matching the session clock", () => {
  const session = createSession({
    steps: 1,
    rfqsPerStep: 1,
    rfqSize: 1,
    inventorySkew: 0,
    riskAversion: 0.1,
    basisVolatility: 0,
    priceVolatility: 0,
  });
  const decision = continuationDecision(
    0,
    "A",
    session.market,
    session.specs.A,
    session.config,
    1,
  );
  const probability = knownFillProbability(0.1);
  const expectedSpread = 0.5 * 0.1 * probability;
  close(
    decision.candidates.find(({ target }) => target === 0).value,
    expectedSpread,
  );
  const result = stepSession(session);
  assert.equal(result.metrics.riskPenalty, 0);
});

test("a nonpositive arithmetic price failure retains its seed, config and complete partial ledger", () => {
  const config = {
    seed: 2,
    policy: "B3",
    steps: 120,
    initialPrice: 50,
    initialBasis: 10,
    basisPersistence: 1.02,
    basisVolatility: 2,
    priceVolatility: 0,
    quoteEdge: 1,
  };
  assert.throws(
    () => runSession(config),
    (error) => {
      assert.equal(error.failure.stage, "market");
      assert.equal(error.failure.seed, 2);
      assert.equal(error.failure.step, 92);
      assert.ok(error.failure.market.B < 0);
      assert.equal(error.failure.partialSnapshot.config.steps, 120);
      assert.ok(error.failure.partialSnapshot.history.length > 1);
      assert.equal(error.failure.partialSnapshot.done, false);
      return true;
    },
  );
});

test("comparison statistics use paired independent seeds rather than policy rows as extra samples", () => {
  const result = comparePolicies({ steps: 3 }, { seeds: ["a", "b", "c"] });
  assert.equal(result.rows.length, 9);
  assert.ok(result.summary.every(({ n }) => n === 3));
  assert.ok(result.paired.every(({ n }) => n === 3));
  const differences = ["a", "b", "c"].map(
    (seed) =>
      result.rows.find((row) => row.seed === seed && row.policy === "R2B").pnl -
      result.rows.find((row) => row.seed === seed && row.policy === "B4").pnl,
  );
  close(
    result.paired.find(({ policy }) => policy === "R2B").meanDifference,
    differences.reduce((a, b) => a + b, 0) / 3,
  );
  assert.ok(result.rows.every(({ ledgerError }) => ledgerError < 1e-7));
});

test("comparison can advance within a session, enabling responsive worker cancellation", () => {
  const comparison = createComparison(
    { steps: 5 },
    { seeds: 2, policies: ["B4"] },
  );
  const first = advanceComparison(comparison, 1);
  assert.equal(first.completed, 0);
  assert.equal(first.activeStep, 1);
  advanceComparison(comparison, 4);
  assert.equal(comparison.completed, 1);
  assert.equal(comparison.done, false);
});

test("every policy reconciles all trades over multiple independent paths", () => {
  for (const policy of ["B3", "B4", "R2B"]) {
    for (let seed = 0; seed < 8; seed += 1) {
      const result = runSession({ policy, seed: `audit/${seed}`, steps: 12 });
      assert.ok(result.metrics.ledgerError < 1e-7);
      assert.ok(
        result.trades.every(
          ({ inventoryAfter }) =>
            Math.abs(inventoryAfter) <= result.config.inventoryLimit,
        ),
      );
      assert.ok(Number.isFinite(result.pnl));
    }
  }
});

test("finished sessions are idempotent and null-edge paths do not invent profit", () => {
  const session = createSession({
    steps: 1,
    quoteEdge: 1,
    basisVolatility: 0,
    priceVolatility: 0,
  });
  const result = stepSession(session);
  assert.equal(result.pnl, 0);
  assert.deepEqual(stepSession(session), result);
});
