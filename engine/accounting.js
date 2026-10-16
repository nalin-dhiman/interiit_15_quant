const EPSILON = 1e-9;

export function roundToTick(price, tick, side) {
  const units = price / tick;
  const rounded =
    side === "BUY" ? Math.ceil(units - EPSILON) : Math.floor(units + EPSILON);
  return Number((rounded * tick).toFixed(10));
}

export function roundToLot(quantity, lot = 1) {
  return (
    Math.sign(quantity) * Math.floor(Math.abs(quantity) / lot + EPSILON) * lot
  );
}

export function breach(quantity, spec) {
  return Math.max(spec.lower - quantity, 0, quantity - spec.upper);
}

/** Exchange actions may improve an existing breach but may not worsen it. */
export function exchangeFeasible(before, after, spec) {
  if (Math.abs(after - before) < EPSILON) return true;
  if (
    Math.abs(
      (after - before) / spec.lot - Math.round((after - before) / spec.lot),
    ) > EPSILON
  )
    return false;
  return breach(after, spec) <= breach(before, spec) + EPSILON;
}

export function execution(before, target, price, spec, config) {
  const delta = roundToLot(target - before, spec.lot);
  const after = before + delta;
  if (!exchangeFeasible(before, after, spec)) return null;
  if (Math.abs(delta) < EPSILON) {
    return {
      before,
      target: before,
      delta: 0,
      price,
      fee: 0,
      cost: 0,
      slippage: 0,
    };
  }
  const executionPrice = roundToTick(
    price + Math.sign(delta) * config.hedgeSlippage,
    spec.tick,
    delta > 0 ? "BUY" : "SELL",
  );
  const fee =
    Math.abs(delta) * config.hedgeFee * spec.multiplier + config.resetFixedCost;
  const slippage = delta * (executionPrice - price) * spec.multiplier;
  return {
    before,
    target: after,
    delta,
    price: executionPrice,
    fee,
    slippage,
    cost: fee + slippage,
  };
}

/** Customer BUY is a dealer sale. Customer fills are never clipped to a limit. */
export function customerDelta(side, quantity) {
  if (side !== "BUY" && side !== "SELL")
    throw new Error(`Unknown customer side: ${side}`);
  if (!Number.isFinite(quantity) || quantity <= 0)
    throw new Error("Trade quantity must be positive.");
  return side === "BUY" ? -quantity : quantity;
}

/** Drawdown is sampled at observed price, trade and settlement events. */
export function observeEquity(session) {
  const prices = session.done
    ? { A: session.market.A, B: 2 * session.market.A }
    : session.market;
  const equity =
    session.cash +
    Object.keys(session.inventory).reduce(
      (sum, product) =>
        sum +
        session.inventory[product] *
          prices[product] *
          session.specs[product].multiplier,
      0,
    );
  const pnl = equity - session.initialEquity;
  session.metrics.peakEquity = Math.max(session.metrics.peakEquity, pnl);
  session.metrics.drawdown = Math.max(
    session.metrics.drawdown,
    session.metrics.peakEquity - pnl,
  );
  return equity;
}

export function bookTrade(
  session,
  { type, product, delta, price, fee = 0, slippage = 0, rfqId = null },
) {
  const spec = session.specs[product];
  if (!spec || ![delta, price, fee].every(Number.isFinite))
    throw new Error("Invalid trade.");
  const principal = -delta * price * spec.multiplier;
  const cashChange = principal - fee;
  session.cash += cashChange;
  session.inventory[product] += delta;
  const trade = {
    id: session.trades.length + 1,
    step: session.step,
    type,
    product,
    side: delta > 0 ? "BUY" : "SELL",
    quantity: Math.abs(delta),
    delta,
    price,
    multiplier: spec.multiplier,
    principal,
    fee,
    slippage,
    cashChange,
    inventoryAfter: session.inventory[product],
    rfqId,
  };
  session.trades.push(trade);
  session.metrics.maxExposure = Math.max(
    session.metrics.maxExposure,
    Object.values(session.inventory).reduce(
      (sum, quantity) => sum + Math.abs(quantity),
      0,
    ),
  );
  session.metrics.fees += fee;
  session.metrics.slippage += slippage;
  if (type === "exchange") {
    session.metrics.exchangeTurnover += Math.abs(
      delta * price * spec.multiplier,
    );
    session.metrics.exchangeQuantity += Math.abs(delta);
    session.metrics.exchangeTrades += 1;
  } else if (type === "customer") {
    session.metrics.customerFills += 1;
    session.metrics.customerNotional += Math.abs(
      delta * price * spec.multiplier,
    );
    session.metrics.customerSpread +=
      -delta * (price - session.market[product]) * spec.multiplier;
  }
  observeEquity(session);
  return trade;
}

/** Independent reconstruction from transactions, rather than cached totals. */
export function auditLedger(session) {
  let cash = session.initialCash;
  const inventory = { ...session.initialInventory };
  let fees = 0;
  for (const trade of session.trades) {
    cash -=
      trade.delta * trade.price * session.specs[trade.product].multiplier +
      trade.fee;
    inventory[trade.product] += trade.delta;
    fees += trade.fee;
  }
  const error = Math.max(
    Math.abs(cash - session.cash),
    Math.abs(fees - session.metrics.fees),
    ...Object.keys(inventory).map((product) =>
      Math.abs(inventory[product] - session.inventory[product]),
    ),
  );
  return { cash, inventory, fees, error, reconciled: error <= 1e-7 };
}

/** A quote reserves its complete possible fill, including opposite-side RFQs. */
export function reserveQuotes(requests, inventory, specs, quoteFor) {
  const reserved = Object.fromEntries(
    Object.keys(specs).map((product) => [product, { buy: 0, sell: 0 }]),
  );
  const updates = {};
  for (const request of requests) {
    const q = inventory[request.product];
    const spec = specs[request.product];
    const used = reserved[request.product];
    const customerBuys = request.side === "BUY";
    const feasible = customerBuys
      ? q - used.buy - request.quantity >= spec.lower - EPSILON
      : q + used.sell + request.quantity <= spec.upper + EPSILON;
    if (!feasible) {
      updates[request.id] = null;
      continue;
    }
    updates[request.id] = quoteFor(request);
    if (customerBuys) used.buy += request.quantity;
    else used.sell += request.quantity;
  }
  return { updates, reserved };
}

/** Immediate acknowledgement model: omission preserves; explicit null withdraws. */
export function applyQuoteUpdates(live, updates, tick) {
  for (const [id, quote] of Object.entries(updates)) {
    if (quote === null) delete live[id];
    else {
      if (!Number.isFinite(quote))
        throw new Error("A quote must be finite or null.");
      live[id] = { price: quote, acknowledgedAt: tick };
    }
  }
  return live;
}
