import { execution, roundToLot, roundToTick } from "./accounting.js";

export const POLICIES = Object.freeze([
  {
    id: "B3",
    label: "Inventory band",
    description: "Conventional inventory control without a basis forecast.",
  },
  {
    id: "B4",
    label: "Basis-aware band",
    description:
      "Conventional cost-aware target using public settlement information.",
  },
  {
    id: "R2B",
    label: "Continuation reset",
    description:
      "Partial exchange interventions evaluated against B4 continuation.",
  },
]);

export function productBasis(product, market) {
  return product === "B" ? 2 * market.A - market.B : 0;
}

export function quotePrice(
  request,
  inventory,
  market,
  specs,
  config,
  policy = config.policy,
) {
  const basis = policy === "B3" ? 0 : productBasis(request.product, market);
  const fair = market[request.product] + config.forecastWeight * basis;
  const direction = request.side === "BUY" ? 1 : -1;
  return roundToTick(
    fair +
      direction * config.quoteEdge -
      config.inventorySkew * inventory[request.product],
    specs[request.product].tick,
    request.side,
  );
}

export function targetObjective(target, basis, remaining, config) {
  return target * basis - config.riskAversion * remaining * target * target;
}

/** A conventional no-trade band: optimize a frozen-position proxy after real costs. */
export function bandDecision(
  q,
  product,
  market,
  spec,
  config,
  remaining,
  policy = config.policy,
) {
  const basis = policy === "B3" ? 0 : productBasis(product, market);
  const startValue = targetObjective(q, basis, remaining, config);
  let best = { target: q, value: startValue, cost: 0, quantity: 0 };
  for (let target = spec.lower; target <= spec.upper; target += spec.lot) {
    const trade = execution(q, target, market[product], spec, config);
    if (!trade) continue;
    const value =
      targetObjective(trade.target, basis, remaining, config) - trade.cost;
    if (value > best.value + 1e-10) {
      best = {
        target: trade.target,
        value,
        cost: trade.cost,
        quantity: trade.delta,
      };
    }
  }
  const partial =
    q + roundToLot((best.target - q) * config.targetFraction, spec.lot);
  const trade = execution(q, partial, market[product], spec, config);
  const value =
    targetObjective(trade.target, basis, remaining, config) - trade.cost;
  const selected = value > startValue + config.decisionMargin ? partial : q;
  return {
    target: selected,
    value: selected === q ? startValue : value,
    cost: selected === q ? 0 : trade.cost,
    quantity: selected - q,
    basis,
  };
}

/** Joint eligibility AND competition probability in the declared, known toy law. */
export function knownFillProbability(
  edge,
  tick = 0.01,
  reference = 0,
  side = "BUY",
) {
  const limitLow = 0.04;
  const limitHigh = 0.3;
  const opponentLow = 0.03;
  const opponentHigh = 0.28;
  const eligibility = Math.max(
    0,
    Math.min(
      1,
      (limitHigh - Math.max(edge, limitLow)) / (limitHigh - limitLow),
    ),
  );
  let winning = 0;
  const direction = side === "BUY" ? 1 : -1;
  const endpointA = reference + direction * opponentLow;
  const endpointB = reference + direction * opponentHigh;
  const first = Math.floor(Math.min(endpointA, endpointB) / tick) - 1;
  const last = Math.ceil(Math.max(endpointA, endpointB) / tick) + 1;
  for (let bin = first; bin <= last; bin += 1) {
    const lower =
      side === "BUY"
        ? (bin - 1) * tick - reference
        : reference - (bin + 1) * tick;
    const upper =
      side === "BUY" ? bin * tick - reference : reference - bin * tick;
    const mass =
      Math.max(
        0,
        Math.min(opponentHigh, upper) - Math.max(opponentLow, lower),
      ) /
      (opponentHigh - opponentLow);
    const rounded = direction * (bin * tick - reference);
    if (edge < rounded - 1e-9) winning += mass;
    else if (Math.abs(edge - rounded) <= 1e-9) winning += mass * 0.5;
  }
  return Math.max(0, Math.min(1, eligibility * winning));
}

export function sizeSupport(config) {
  return [...new Set([1, Math.min(2, config.rfqSize), config.rfqSize])];
}

/**
 * Finite future-batch model. Quotes are simultaneous, and possible fills reserve
 * capacity even when their realized outcome is no fill. Reward mass is retained
 * when paths with the same inventory merge. Public prices are held fixed within
 * this approximation; the browser session itself updates them within a batch.
 */
export function futureFlowKernel(
  q,
  product,
  market,
  spec,
  config,
  policy = "B4",
) {
  const sizes = sizeSupport(config);
  const quotes = {};
  const fills = {};
  for (const side of ["BUY", "SELL"]) {
    quotes[side] = quotePrice(
      { product, side },
      { [product]: q },
      market,
      { [product]: spec },
      config,
      policy,
    );
    const edge = (side === "BUY" ? 1 : -1) * (quotes[side] - market[product]);
    fills[side] = knownFillProbability(edge, spec.tick, market[product], side);
  }
  let states = new Map([
    ["0/0/0", { delta: 0, buy: 0, sell: 0, probability: 1, rewardMass: 0 }],
  ]);
  for (let slot = 0; slot < config.rfqsPerStep; slot += 1) {
    const next = new Map();
    function add(
      state,
      branchProbability,
      fillDelta = 0,
      spread = 0,
      buy = state.buy,
      sell = state.sell,
    ) {
      if (branchProbability <= 0) return;
      const delta = state.delta + fillDelta;
      const key = `${delta}/${buy}/${sell}`;
      const probability = state.probability * branchProbability;
      const existing = next.get(key) || {
        delta,
        buy,
        sell,
        probability: 0,
        rewardMass: 0,
      };
      existing.probability += probability;
      existing.rewardMass +=
        state.rewardMass * branchProbability + probability * spread;
      next.set(key, existing);
    }
    for (const state of states.values()) {
      add(state, 0.5); // RFQ belongs to the other product.
      for (const side of ["BUY", "SELL"]) {
        for (const quantity of sizes) {
          const probability = 0.25 / sizes.length;
          const admitted =
            side === "BUY"
              ? q - state.buy - quantity >= spec.lower
              : q + state.sell + quantity <= spec.upper;
          if (!admitted) {
            add(state, probability);
            continue;
          }
          const buy = state.buy + (side === "BUY" ? quantity : 0);
          const sell = state.sell + (side === "SELL" ? quantity : 0);
          const delta = side === "BUY" ? -quantity : quantity;
          const spread =
            -delta * (quotes[side] - market[product]) * spec.multiplier;
          add(state, probability * (1 - fills[side]), 0, 0, buy, sell);
          add(state, probability * fills[side], delta, spread, buy, sell);
        }
      }
    }
    states = next;
  }
  const totals = new Map();
  for (const state of states.values()) {
    const target = q + state.delta;
    const row = totals.get(target) || {
      inventory: target,
      probability: 0,
      rewardMass: 0,
    };
    row.probability += state.probability;
    row.rewardMass += state.rewardMass;
    totals.set(target, row);
  }
  return [...totals.values()];
}

export function continuationDecision(
  q,
  product,
  market,
  spec,
  config,
  remaining,
) {
  const baseline = bandDecision(
    q,
    product,
    market,
    spec,
    config,
    remaining,
    "B4",
  ).target;
  const memo = new Map();
  const kernelMemo = new Map();
  function atBasis(basis) {
    return { A: market.A, B: 2 * market.A - basis };
  }
  function valueOfAction(before, target, basis, time, depth) {
    const futureMarket = product === "B" ? atBasis(basis) : market;
    const trade = execution(
      before,
      target,
      futureMarket[product],
      spec,
      config,
    );
    if (!trade) return -Infinity;
    const key = `${trade.target}/${basis.toFixed(8)}`;
    let kernel = kernelMemo.get(key);
    if (!kernel) {
      kernel = futureFlowKernel(
        trade.target,
        product,
        futureMarket,
        spec,
        config,
      );
      kernelMemo.set(key, kernel);
    }
    // Customer fills occur at the deadline: target inventory is held throughout
    // this batch. Charging post-fill inventory here would move the risk clock.
    let value = -trade.cost - config.riskAversion * trade.target * trade.target;
    const nextBasis = product === "B" ? config.basisPersistence * basis : 0;
    for (const row of kernel) {
      const after = row.inventory;
      let continuation;
      if (time <= 1) continuation = after * basis;
      else {
        const priceMove = product === "B" ? after * (basis - nextBasis) : 0;
        continuation =
          priceMove + baselineValue(after, nextBasis, time - 1, depth - 1);
      }
      value += row.rewardMass + row.probability * continuation;
    }
    return value;
  }
  function baselineValue(before, basis, time, depth) {
    if (time <= 0) return before * basis;
    const key = `${before}/${basis.toFixed(8)}/${time}/${depth}`;
    if (memo.has(key)) return memo.get(key);
    const futureMarket = product === "B" ? atBasis(basis) : market;
    const band = bandDecision(
      before,
      product,
      futureMarket,
      spec,
      config,
      time,
      "B4",
    );
    const value =
      depth <= 0
        ? band.value
        : valueOfAction(before, band.target, basis, time, depth);
    memo.set(key, value);
    return value;
  }
  const basis = productBasis(product, market);
  const candidates = new Set([q, baseline, 0]);
  for (const fraction of [0.25, 0.5, 0.75]) {
    candidates.add(q + roundToLot(-q * fraction, spec.lot));
    candidates.add(q + roundToLot((baseline - q) * fraction, spec.lot));
  }
  const depth = Math.min(remaining, config.continuationDepth);
  const baselineValueNow = valueOfAction(q, baseline, basis, remaining, depth);
  const rows = [...candidates]
    .map((target) => {
      const trade = execution(q, target, market[product], spec, config);
      if (!trade) return null;
      const value = valueOfAction(q, trade.target, basis, remaining, depth);
      return {
        target: trade.target,
        quantity: trade.delta,
        cost: trade.cost,
        value,
        advantage: value - baselineValueNow,
        selected: false,
      };
    })
    .filter(Boolean);
  let bestTarget = baseline;
  let bestValue = baselineValueNow;
  for (const row of rows) {
    if (row.value > bestValue + 1e-10) {
      bestTarget = row.target;
      bestValue = row.value;
    }
  }
  const chosen =
    bestValue > baselineValueNow + config.decisionMargin
      ? bestTarget
      : baseline;
  for (const row of rows) row.selected = row.target === chosen;
  return {
    product,
    before: q,
    chosen,
    baseline,
    candidates: rows,
    reason:
      chosen === baseline
        ? "Retain the conventional action within the comparison margin."
        : "The modeled continuation gain exceeds the comparison margin.",
    approximate: true,
    errorCertificate: false,
  };
}

export function chooseHedge(q, product, market, spec, config, remaining) {
  if (config.policy === "R2B")
    return continuationDecision(q, product, market, spec, config, remaining);
  const band = bandDecision(q, product, market, spec, config, remaining);
  const noAction = targetObjective(q, band.basis, remaining, config);
  return {
    product,
    before: q,
    chosen: band.target,
    baseline: band.target,
    candidates: [...new Set([q, band.target])].map((target) => {
      const trade = execution(q, target, market[product], spec, config);
      const value =
        targetObjective(trade.target, band.basis, remaining, config) -
        trade.cost;
      return {
        target: trade.target,
        quantity: trade.delta,
        cost: trade.cost,
        value,
        advantage: value - noAction,
        selected: target === band.target,
      };
    }),
    reason:
      band.target === q
        ? "Exposure remains inside the conventional no-trade band."
        : "The conventional partial target covers its transaction costs.",
    approximate: true,
    errorCertificate: false,
  };
}
