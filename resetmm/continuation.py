"""Finite RFQ-aware continuation diagnostic, NOT an organiser exchange adapter.

State (decision time, inventory in lots, observed batch type) is sufficient only
under the translation-invariant, known, exogenous kernel documented in
``docs/model.md``. Customer transactions are never clipped.
Sparse recursion has no numerical inventory boundary. Risk-adjusted objective,
ledger PnL, and settlement residual are distinct quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from typing import Callable, Iterable
import numpy as np


@dataclass(frozen=True)
class RFQ:
    side: str
    quantity_lots: int = 1

    def __post_init__(self):
        if self.side not in ("BUY", "SELL") or not isinstance(self.quantity_lots, int) or self.quantity_lots < 1:
            raise ValueError("RFQ side and positive integer lot quantity required")

    @property
    def direction(self) -> int:
        return 1 if self.side == "BUY" else -1


@dataclass(frozen=True)
class PrimitiveAtom:
    probability: float
    limit_edges: tuple[float, ...]
    competitor_edges: tuple[tuple[float, ...], ...]
    tie_uniforms: tuple[float, ...]
    price_change: float = 0.0


@dataclass(frozen=True)
class Batch:
    rfqs: tuple[RFQ, ...]
    atoms: tuple[PrimitiveAtom, ...]


@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 6
    limit_lots: int = 3
    lot: float = 1.0
    multiplier: float = 1.0
    tick: float = .01
    reference_price: float = 100.0
    exchange_fee: float = .02       # currency / physical quantity
    fixed_cost: float = 0.0         # currency / nonzero exchange trade
    slippage_ticks: int = 0         # adverse execution ticks; no impact
    customer_fee: float = 0.0       # explicit synthetic fee / physical quantity
    breach_penalty: float = .1     # currency / physical excess / decision
    risk_weight: float = 0.0       # DESIGN penalty / physical inventory^2 / time
    dt: float = 1.0
    terminal_basis: float = 0.0     # settlement minus last reference / price unit
    quote_edges: tuple[float, ...] = (.02, .08, .16)


@dataclass(frozen=True)
class Action:
    target_lots: int | None
    quote_edges: tuple[float | None, ...]


@dataclass(frozen=True)
class Outcome:
    probability: float
    next_inventory: int
    ledger_pnl: float
    risk_penalty: float
    fees: float
    exchange_quantity: float
    customer_orders: int
    customer_quantity: float
    price_change: float
    fills: tuple[bool, ...]
    cash_delta_at_reference: float

    @property
    def objective(self) -> float:
        return self.ledger_pnl - self.risk_penalty


Policy = Callable[[int, int, int], Action]


class FiniteRFQModel:
    def __init__(self, config: ModelConfig, batches: tuple[Batch, ...],
                 batch_transition: Iterable[Iterable[float]] | None = None):
        self.config, self.batches = config, tuple(batches)
        c = config
        ints = (c.horizon, c.limit_lots, c.slippage_ticks)
        if any(not isinstance(v, int) for v in ints) or min(c.horizon, c.limit_lots) < 1 or c.slippage_ticks < 0:
            raise ValueError("Positive integer horizon/limit and nonnegative slippage required")
        values = (c.lot, c.multiplier, c.tick, c.reference_price, c.exchange_fee,
                  c.fixed_cost, c.customer_fee, c.breach_penalty, c.risk_weight, c.dt, c.terminal_basis)
        if not all(math.isfinite(v) for v in values) or min(c.lot, c.multiplier, c.tick, c.reference_price, c.dt) <= 0:
            raise ValueError("Finite model parameters and positive units required")
        if min(c.exchange_fee, c.fixed_cost, c.customer_fee, c.breach_penalty, c.risk_weight) < 0:
            raise ValueError("Fees, penalties, and risk preference must be nonnegative")
        if not c.quote_edges or not all(math.isfinite(v) for v in c.quote_edges):
            raise ValueError("Finite nonempty quote grid required")
        if not math.isclose(c.reference_price/c.tick, round(c.reference_price/c.tick), abs_tol=1e-8):
            raise ValueError("Reference must lie on the tick lattice")
        if not self.batches:
            raise ValueError("At least one observed batch required")
        max_shock = 0.0
        for batch in self.batches:
            if not batch.atoms:
                raise ValueError("Each batch needs finite primitive support")
            if not math.isclose(sum(a.probability for a in batch.atoms), 1.0, abs_tol=1e-12, rel_tol=0):
                raise ValueError("Atom probabilities must sum to one")
            for a in batch.atoms:
                n = len(batch.rfqs)
                if any(len(v) != n for v in (a.limit_edges, a.competitor_edges, a.tie_uniforms)):
                    raise ValueError("Atom fields must align with RFQs")
                vals = (a.probability, a.price_change) + a.limit_edges + a.tie_uniforms + tuple(e for es in a.competitor_edges for e in es)
                if not all(math.isfinite(v) for v in vals) or a.probability < 0 or not all(0 <= u < 1 for u in a.tie_uniforms):
                    raise ValueError("Invalid primitive atom")
                if not math.isclose(a.price_change / c.tick, round(a.price_change / c.tick), abs_tol=1e-8):
                    raise ValueError("Price innovations must preserve tick origin")
                max_shock = max(max_shock, abs(a.price_change))
        if c.reference_price - c.horizon * max_shock + min(0.0, min(c.quote_edges)) <= 0:
            raise ValueError("Finite primitive support permits nonpositive reference prices")
        self.minimum_reference = c.reference_price-c.horizon*max_shock
        n = len(self.batches)
        transition = np.asarray(batch_transition if batch_transition is not None else np.full((n, n), 1/n), float)
        if transition.shape != (n, n) or not np.isfinite(transition).all() or np.any(transition < 0) or not np.allclose(transition.sum(1), 1, atol=1e-12, rtol=0):
            raise ValueError("Invalid observed batch transition")
        self.batch_transition = transition
        self._outcome_cache: dict[tuple[int, int, Action, bool], tuple[Outcome, ...]] = {}

    def round_edge(self, edge: float) -> float:
        """Both sides' dealer margins round upward under conservative rounding."""
        if not math.isfinite(edge):
            raise ValueError("Finite quote margin required")
        return math.ceil(edge / self.config.tick - 1e-10) * self.config.tick

    def target(self, q: int, action: Action) -> int:
        if not isinstance(q, int) or (action.target_lots is not None and not isinstance(action.target_lots, int)):
            raise ValueError("Inventory/targets must be integer numbers of lots")
        return q if action.target_lots is None else action.target_lots

    def partial_targets(self, q: int) -> tuple[int, ...]:
        """No action and all lot-rounded same-sign exposure-reducing targets."""
        sign = 1 if q >= 0 else -1
        return (q,) + tuple(sign * z for z in range(abs(q)) if z <= self.config.limit_lots)

    def exchange_feasible(self, q: int, z: int) -> bool:
        # Same rejection convention as research Ledger; NOT SDK verification.
        L = self.config.limit_lots
        return min(-L, q) <= z <= max(L, q)

    def capacity_feasible(self, z: int, b: int, edges: tuple[float | None, ...]) -> bool:
        if len(edges) != len(self.batches[b].rfqs):
            return False
        if all(e is None for e in edges):
            return True  # withdrawal remains available in an existing breach
        lo = z - sum(r.quantity_lots for r, e in zip(self.batches[b].rfqs, edges) if e is not None and r.side == "BUY")
        hi = z + sum(r.quantity_lots for r, e in zip(self.batches[b].rfqs, edges) if e is not None and r.side == "SELL")
        return lo >= -self.config.limit_lots and hi <= self.config.limit_lots

    def feasible(self, q: int, b: int, action: Action, *, reserve: bool = True) -> bool:
        z = self.target(q, action)
        if len(action.quote_edges) != len(self.batches[b].rfqs) or not self.exchange_feasible(q, z):
            return False
        if not all(e is None or math.isfinite(e) for e in action.quote_edges):
            return False
        if any(e is not None and self.minimum_reference+r.direction*self.round_edge(e) <= 0
               for r,e in zip(self.batches[b].rfqs, action.quote_edges)):
            return False
        return not reserve or self.capacity_feasible(z, b, action.quote_edges)

    def hedge_cost(self, q: int, z: int) -> float:
        c = self.config
        amount = abs(z - q) * c.lot
        return 0.0 if amount == 0 else c.fixed_cost + amount * (c.exchange_fee + c.multiplier*c.slippage_ticks*c.tick)

    def terminal(self, q: int) -> float:
        # No terminal risk penalty and no forced liquidation.
        # A real breach charge is ledger cash, distinct from DESIGN risk.
        return (q * self.config.lot * self.config.multiplier * self.config.terminal_basis
                - self.terminal_breach_penalty(q))

    def terminal_breach_penalty(self, q: int) -> float:
        c = self.config
        return max(abs(q)-c.limit_lots, 0)*c.lot*c.breach_penalty

    def outcomes(self, q: int, b: int, action: Action, *, reserve: bool = True) -> tuple[Outcome, ...]:
        key = (q, b, action, reserve)
        if key in self._outcome_cache:
            return self._outcome_cache[key]
        if not self.feasible(q, b, action, reserve=reserve):
            raise ValueError("Infeasible exchange/quote action")
        c, batch = self.config, self.batches[b]
        z = self.target(q, action)
        edges = tuple(None if e is None else self.round_edge(e) for e in action.quote_edges)
        hedge_fee = 0.0 if z == q else c.fixed_cost + abs(z-q)*c.lot*c.exchange_fee
        hedge_cost = self.hedge_cost(q, z)
        cash0 = -(z-q)*c.lot*c.multiplier*c.reference_price - hedge_cost
        out = []
        for a in batch.atoms:
            next_q, spread, quantity, orders = z, 0.0, 0.0, 0
            cash_delta, fills = cash0, []
            for j, (r, edge) in enumerate(zip(batch.rfqs, edges)):
                limit = a.limit_edges[j]
                competitors = tuple(self.round_edge(e) for e in a.competitor_edges[j])
                eligible = tuple(e for e in competitors if e <= limit + 1e-12)
                best = min(eligible) if eligible else math.inf
                tied = sum(abs(e-edge) <= 1e-12 for e in eligible) if edge is not None else 0
                won = (edge is not None and edge <= limit + 1e-12 and edge <= best + 1e-12 and
                       (edge < best - 1e-12 or a.tie_uniforms[j] < 1/(1+tied)))
                fills.append(won)
                if won:
                    amount = r.quantity_lots*c.lot
                    delta = -r.direction*r.quantity_lots
                    next_q += delta  # never clamp, reject, or reflect customer fill
                    spread += edge*amount*c.multiplier
                    quantity += amount
                    orders += 1
                    price = c.reference_price + r.direction*edge
                    cash_delta -= delta*c.lot*c.multiplier*price + amount*c.customer_fee
            penalty = max(abs(next_q)-c.limit_lots, 0)*c.lot*c.breach_penalty
            customer_fee = quantity*c.customer_fee
            cash_delta -= penalty
            price_pnl = next_q*c.lot*c.multiplier*a.price_change
            ledger_pnl = spread - customer_fee - hedge_cost - penalty + price_pnl
            risk = c.risk_weight*(next_q*c.lot)**2*c.dt
            out.append(Outcome(a.probability, next_q, ledger_pnl, risk, hedge_fee+customer_fee,
                               abs(z-q)*c.lot, orders, quantity, a.price_change, tuple(fills), cash_delta))
        result = tuple(out)
        self._outcome_cache[key] = result
        return result

    def actions(self, q: int, b: int) -> tuple[Action, ...]:
        """Full finite joint-quote oracle within the partial-reset family."""
        actions = []
        edges = (None,) + tuple(dict.fromkeys(self.round_edge(e) for e in self.config.quote_edges))
        for z in self.partial_targets(q):
            for quotes in product(edges, repeat=len(self.batches[b].rfqs)):
                action = Action(None if z == q else z, tuple(quotes))
                if self.feasible(q, b, action):
                    actions.append(action)
        return tuple(actions)


@dataclass(frozen=True)
class BandPolicy:
    edge: float = .08
    skew: float = .01
    threshold_lots: int | None = 3
    target_fraction: float = .5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.edge, self.skew, self.target_fraction)) or self.skew < 0 or not 0 <= self.target_fraction < 1:
            raise ValueError("Invalid conventional band parameters")
        if self.threshold_lots is not None and (not isinstance(self.threshold_lots, int) or self.threshold_lots < 1):
            raise ValueError("Positive integer threshold or None required")

    def quotes_at(self, model: FiniteRFQModel, z: int, b: int) -> tuple[float | None, ...]:
        # Greedy reservation in observed RFQ order, all simultaneously live.
        c = model.config
        lo = hi = z
        quotes = []
        for r in model.batches[b].rfqs:
            nlo = lo - (r.quantity_lots if r.side == "BUY" else 0)
            nhi = hi + (r.quantity_lots if r.side == "SELL" else 0)
            if nlo >= -c.limit_lots and nhi <= c.limit_lots:
                edge = model.round_edge(self.edge-r.direction*self.skew*z)
                quotes.append(edge)
                lo, hi = nlo, nhi
            else:
                quotes.append(None)
        return tuple(quotes)

    def bind(self, model: FiniteRFQModel) -> Policy:
        def policy(t: int, q: int, b: int) -> Action:
            if self.threshold_lots is not None and abs(q) >= self.threshold_lots:
                mag = min(abs(q), math.floor(self.target_fraction*self.threshold_lots), model.config.limit_lots)
                z = mag if q >= 0 else -mag
            else:
                z = q
            return Action(None if z == q else z, self.quotes_at(model, z, b))
        return policy


class PolicyEvaluation:
    """Memoised baseline recursion; continuation includes all future RFQ flow."""
    def __init__(self, model: FiniteRFQModel, policy: Policy):
        self.model, self.policy = model, policy
        self.value = lru_cache(maxsize=None)(self._value)

    def _value(self, t: int, q: int, b: int) -> float:
        if t == self.model.config.horizon:
            return self.model.terminal(q)
        return self.action_value(t, q, b, self.policy(t, q, b))

    def action_value(self, t: int, q: int, b: int, action: Action) -> float:
        if not 0 <= t < self.model.config.horizon:
            raise ValueError("Action time outside horizon")
        return sum(o.probability*(o.objective + sum(float(p)*self.value(t+1, o.next_inventory, nb)
                    for nb, p in enumerate(self.model.batch_transition[b]) if p))
                   for o in self.model.outcomes(q, b, action))

    def initial_value(self, q0: int = 0, initial_batch: Iterable[float] | None = None) -> float:
        weights = initial_weights(self.model, initial_batch)
        return sum(float(p)*self.value(0, q0, b) for b, p in enumerate(weights) if p)


def initial_weights(model: FiniteRFQModel, weights: Iterable[float] | None) -> np.ndarray:
    n = len(model.batches)
    w = np.asarray(tuple(weights) if weights is not None else np.full(n, 1/n), float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or not math.isclose(float(w.sum()), 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Invalid initial observed-batch distribution")
    return w


def evaluate_policy(model: FiniteRFQModel, policy: Policy, q0: int = 0,
                    initial_batch: Iterable[float] | None = None) -> float:
    return PolicyEvaluation(model, policy).initial_value(q0, initial_batch)


class ImprovedPolicy:
    """R2: choose a partial target using exact Q^baseline, then replan next event.

    Current quote response is the same band quote map at each candidate target.
    This is one baseline-policy improvement, not the full joint-control oracle.
    """
    def __init__(self, model: FiniteRFQModel, baseline: BandPolicy):
        self.model, self.baseline = model, baseline
        self.baseline_policy = baseline.bind(model)
        self.evaluation = PolicyEvaluation(model, self.baseline_policy)
        self.__call_cached = lru_cache(maxsize=None)(self._choose)

    def candidates(self, t: int, q: int, b: int) -> tuple[Action, ...]:
        base = self.baseline_policy(t, q, b)
        actions = [base]
        for z in self.model.partial_targets(q):
            a = Action(None if z == q else z, self.baseline.quotes_at(self.model, z, b))
            if a not in actions and self.model.feasible(q, b, a):
                actions.append(a)
        return tuple(actions)

    def _choose(self, t: int, q: int, b: int) -> Action:
        actions = self.candidates(t, q, b)
        # Stable ties retain baseline; no arbitrary hedge at a zero action gap.
        best, value = actions[0], self.evaluation.action_value(t, q, b, actions[0])
        for a in actions[1:]:
            v = self.evaluation.action_value(t, q, b, a)
            if v > value + 1e-12:
                best, value = a, v
        return best

    def __call__(self, t: int, q: int, b: int) -> Action:
        return self.__call_cached(t, q, b)


def improve_policy(model: FiniteRFQModel, baseline: BandPolicy) -> ImprovedPolicy:
    return ImprovedPolicy(model, baseline)


class OptimalPolicy:
    """Exact joint-control DP only for the supplied finite partial-reset family."""
    def __init__(self, model: FiniteRFQModel):
        self.model = model
        self.selected: dict[tuple[int, int, int], Action] = {}
        self.value = lru_cache(maxsize=None)(self._value)

    def _value(self, t: int, q: int, b: int) -> float:
        if t == self.model.config.horizon:
            return self.model.terminal(q)
        best, best_value = None, -math.inf
        for a in self.model.actions(q, b):
            val = sum(o.probability*(o.objective + sum(float(p)*self.value(t+1, o.next_inventory, nb)
                      for nb, p in enumerate(self.model.batch_transition[b]) if p))
                      for o in self.model.outcomes(q, b, a))
            if val > best_value + 1e-12:
                best, best_value = a, val
        assert best is not None
        self.selected[t, q, b] = best
        return best_value

    def __call__(self, t: int, q: int, b: int) -> Action:
        self.value(t, q, b)
        return self.selected[t, q, b]


def solve_optimal(model: FiniteRFQModel) -> OptimalPolicy:
    return OptimalPolicy(model)


def tune_band(model: FiniteRFQModel, parameter_grid: Iterable[BandPolicy], *,
              q0: int = 0, initial_batch: Iterable[float] | None = None,
              improved: bool = False) -> tuple[BandPolicy, list[dict]]:
    """Known-kernel exploratory tuning; number of candidates is the budget.

    Calling once per method with the same grid gives equal candidate counts,
    not equal CPU costs. This is not data calibration or held-out evidence.
    """
    best, score, rows = None, -math.inf, []
    for band in parameter_grid:
        policy = improve_policy(model, band) if improved else band.bind(model)
        value = evaluate_policy(model, policy, q0, initial_batch)
        rows.append(dict(edge=band.edge, skew=band.skew, threshold_lots=band.threshold_lots,
                         target_fraction=band.target_fraction, objective=value))
        if value > score:
            best, score = band, value
    if best is None:
        raise ValueError("Nonempty parameter budget required")
    return best, rows


@dataclass(frozen=True)
class PrimitiveTape:
    seed: int
    batch_types: tuple[int, ...]
    atom_indices: tuple[int, ...]
    price_uniforms: tuple[float, ...]
    customer_uniforms: tuple[float, ...]
    batch_uniforms: tuple[float, ...]
    opponent_uniforms: tuple[float, ...]
    exploration_uniforms: tuple[float, ...]


def _sample(weights: Iterable[float], uniform: float) -> int:
    probabilities = np.asarray(tuple(weights), float)
    cumulative = np.cumsum(probabilities/probabilities.sum())
    return min(int(np.searchsorted(cumulative, uniform, side="right")), len(cumulative)-1)


def make_primitive_tape(model: FiniteRFQModel, seed: int, *,
                        initial_batch: Iterable[float] | None = None) -> PrimitiveTape:
    """Factor an arbitrary finite joint law with independent primitive streams.

    The ancestral factorisation is P(price|batch), P(customer limits|price,batch),
    P(opponent and ties|price,customer limits,batch). This preserves stated
    dependence; independent uniforms do not imply independent market outcomes.
    Batch transitions use the customer stream, separate fixed draw arrays.
    Exploration has a reserved independent stream; deterministic policies ignore
    it. All arrays and latent outcomes are generated before policy execution.
    """
    weights = initial_weights(model, initial_batch)
    H = model.config.horizon
    generators = tuple(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))
    rp, rc, ro, re = generators
    price_u = tuple(map(float, rp.random(H)))
    customer_u = tuple(map(float, rc.random(H)))
    batch_u = tuple(map(float, rc.random(H+1)))
    opponent_u = tuple(map(float, ro.random(H)))
    exploration_u = tuple(map(float, re.random(H)))
    b = _sample(weights, batch_u[0])
    batches, atom_indices = [], []
    for t in range(H):
        atoms = model.batches[b].atoms
        prices = tuple(dict.fromkeys(a.price_change for a in atoms if a.probability))
        pi = _sample((sum(a.probability for a in atoms if a.price_change == x) for x in prices), price_u[t])
        selected_price = prices[pi]
        customers = tuple(dict.fromkeys(a.limit_edges for a in atoms if a.probability and a.price_change == selected_price))
        ci = _sample((sum(a.probability for a in atoms if a.price_change == selected_price and a.limit_edges == x)
                      for x in customers), customer_u[t])
        selected_customer = customers[ci]
        choices = tuple(i for i,a in enumerate(atoms) if a.probability and a.price_change == selected_price and a.limit_edges == selected_customer)
        ai = choices[_sample((atoms[i].probability for i in choices), opponent_u[t])]
        batches.append(b)
        atom_indices.append(ai)
        b = _sample(model.batch_transition[b], batch_u[t+1])
    return PrimitiveTape(int(seed), tuple(batches), tuple(atom_indices), price_u,
                         customer_u, batch_u, opponent_u, exploration_u)


def simulate_policy(model: FiniteRFQModel, policy: Policy, seeds: Iterable[int], *,
                    q0: int = 0, initial_batch: Iterable[float] | None = None,
                    tapes: Iterable[PrimitiveTape] | None = None) -> list[dict]:
    """Replay matched exogenous tapes and independently reconcile cash.

    Policies receive only (t, confirmed q, observed b), never latent tape fields.
    Optional tapes must have been made for this exact model/kernel; callers are
    responsible for retaining the model/config hash with archived tapes.
    """
    weights = initial_weights(model, initial_batch)
    c, rows = model.config, []
    selected_tapes = tuple(tapes) if tapes is not None else tuple(make_primitive_tape(model, seed, initial_batch=weights) for seed in seeds)
    for tape in selected_tapes:
        if len(tape.batch_types) != c.horizon or len(tape.atom_indices) != c.horizon:
            raise ValueError("Primitive tape horizon mismatch")
        q, reference = q0, c.reference_price
        cash = -q0*c.lot*c.multiplier*reference  # zero initial marked wealth
        pnl = risk = fees = turnover = qty = exposure = customer_notional = exchange_notional = penalties = 0.0
        orders = 0
        peak = drawdown = 0.0
        for t in range(c.horizon):
            b = tape.batch_types[t]
            action = policy(t, q, b)
            outcomes = model.outcomes(q, b, action)
            ix = tape.atom_indices[t]
            o = outcomes[ix]
            z = model.target(q, action)
            execution = reference + (1 if z > q else -1)*c.slippage_ticks*c.tick
            exchange_notional += abs(z-q)*c.lot*c.multiplier*execution
            for r, edge, won in zip(model.batches[b].rfqs, action.quote_edges, o.fills):
                if won:
                    customer_notional += r.quantity_lots*c.lot*c.multiplier*(reference+r.direction*model.round_edge(edge))
            penalties += max(abs(o.next_inventory)-c.limit_lots, 0)*c.lot*c.breach_penalty
            # Translate the independent cash fixture from its fixed reference.
            cash += o.cash_delta_at_reference - (o.next_inventory-q)*c.lot*c.multiplier*(reference-c.reference_price)
            reference += o.price_change
            q = o.next_inventory
            pnl += o.ledger_pnl
            risk += o.risk_penalty
            fees += o.fees
            turnover += o.exchange_quantity
            qty += o.customer_quantity
            orders += o.customer_orders
            exposure += (q*c.lot)**2*c.dt
            peak = max(peak, pnl)
            drawdown = max(drawdown, peak-pnl)
        terminal_penalty = model.terminal_breach_penalty(q)
        cash -= terminal_penalty
        penalties += terminal_penalty
        pnl += model.terminal(q)
        reconciled = cash + q*c.lot*c.multiplier*(reference+c.terminal_basis)
        if not math.isclose(pnl, reconciled, abs_tol=1e-7, rel_tol=1e-9):
            raise AssertionError("Continuation fixture cash/marked-wealth mismatch")
        rows.append(dict(seed=tape.seed, pnl=pnl, objective=pnl-risk, risk_penalty=risk,
                         fees=fees, exchange_quantity=turnover, customer_quantity=qty,
                         customer_orders=orders, customer_notional=customer_notional,
                         exchange_notional=exchange_notional, penalties=penalties,
                         inventory_sq_time=exposure, max_drawdown=max(drawdown, peak-pnl),
                         terminal_inventory=q*c.lot, cash=cash, settlement_price=reference+c.terminal_basis))
    return rows


def make_synthetic_model(*, horizon: int = 8, limit_lots: int = 3,
                         regime: str = "balanced", risk_weight: float = .005,
                         exchange_fee: float = .025) -> FiniteRFQModel:
    """Transparent small kernels for exploratory diagnostics, no fitted data.

    Persistent same-side batches cause capacity pressure; offsetting batch types
    are included. Informed atoms couple a favourable fill to adverse price move.
    Within-batch fills share competition/eligibility and are dependent.
    """
    if regime not in ("balanced", "persistent", "informed", "expensive"):
        raise ValueError("Unknown synthetic continuation regime")
    c = ModelConfig(horizon=horizon, limit_lots=limit_lots, risk_weight=risk_weight,
                    exchange_fee=(.15 if regime == "expensive" else exchange_fee))
    batches = []
    for side in ("BUY", "SELL"):
        d = 1 if side == "BUY" else -1
        rfqs = (RFQ(side, 1), RFQ(side, 2))
        atoms = []
        for p, limit, competitor in ((.35, .18, .10), (.4, .10, .06), (.25, .04, .03)):
            for tie in (.1, .9):
                if regime == "informed":
                    # More eligible/winnable atoms have worse next dealer markout.
                    change = d*(.03 if competitor >= .06 else -.02)
                    atoms.append(PrimitiveAtom(p/2, (limit,)*2, ((competitor,),)*2, (tie,)*2, change))
                else:
                    for shock in (-.02, .02):
                        atoms.append(PrimitiveAtom(p/4, (limit,)*2, ((competitor,),)*2, (tie,)*2, shock))
        batches.append(Batch(rfqs, tuple(atoms)))
    persistence = .85 if regime == "persistent" else .5
    transition = ((persistence, 1-persistence), (1-persistence, persistence))
    return FiniteRFQModel(c, tuple(batches), transition)
