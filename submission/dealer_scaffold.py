"""ANONYMOUS SDK ADAPTATION SCAFFOLD -- NOT READY FOR SUBMISSION.

This file uses only Python's standard library. The quote/hedge signatures,
market['prices'], market['products'], portfolio['positions'], order_id,
product_id, side, tick_size and lot_size are shown in the supplied PS.
Other field names MUST be confirmed in PARTICIPANTS.md. Until that happens
verified=False deliberately withdraws every active quote and sends no hedges.
This scaffold is a conservative fallback, not the full research controller.
"""
import math

CONFIG = {
    "verified": False,
    "quantity_key": None,
    "long_limit_key": None,
    "short_limit_key": None,
    "edge": 0.10,
    "skew": 0.002,
    "trigger_fraction": 0.75,
    "target_fraction": 0.25,
}

class Dealer:
    def __init__(self):
        self.config = dict(CONFIG)

    def quote(self, orders, market, portfolio):
        output = {o["order_id"]: None for o in orders}
        cfg = self.config
        if not cfg["verified"]:
            return output
        required = (cfg["quantity_key"], cfg["long_limit_key"], cfg["short_limit_key"])
        if any(k is None for k in required):
            return output
        # With an actual SDK, these snapshots must include acknowledged and
        # outstanding orders. Confirm whether 'orders' is the complete active set.
        low = dict(portfolio["positions"])
        high = dict(low)
        for order in sorted(orders, key=lambda o: str(o["order_id"])):
            try:
                key, side = order["product_id"], order["side"]
                if side not in ("BUY", "SELL"):
                    continue
                spec = market["products"][key]
                quantity = float(order[cfg["quantity_key"]])
                price = float(market["prices"][key])
                tick = float(spec["tick_size"])
                lot = float(spec["lot_size"])
                long_limit = float(spec[cfg["long_limit_key"]])
                short_limit = float(spec[cfg["short_limit_key"]])
                q = float(portfolio["positions"].get(key, 0.0))
                values = (quantity, price, tick, lot, long_limit, short_limit, q)
                if not all(math.isfinite(v) for v in values) or min(quantity,price,tick,lot)<=0:
                    continue
                if min(long_limit,short_limit)<0:
                    continue
                if not math.isclose(quantity/lot,round(quantity/lot),rel_tol=0,abs_tol=1e-8):
                    continue
                delta = -quantity if side == "BUY" else quantity
                lo = low.get(key,q)+min(delta,0.0)
                hi = high.get(key,q)+max(delta,0.0)
                if lo < -short_limit or hi > long_limit:
                    continue
                raw = price - cfg["skew"]*q + (cfg["edge"] if side=="BUY" else -cfg["edge"])
                steps = math.ceil(raw/tick-1e-10) if side=="BUY" else math.floor(raw/tick+1e-10)
                quote = steps*tick
                if quote<=0 or not math.isfinite(quote):
                    continue
                # Replace rounding with the verified SDK helper before release.
                output[order["order_id"]] = quote
                low[key],high[key] = lo,hi
            except (KeyError, TypeError, ValueError, OverflowError):
                # An invalid RFQ is explicitly withdrawn rather than omitted.
                continue
        return output

    def hedge(self, market, portfolio):
        cfg = self.config
        if not cfg["verified"] or cfg["long_limit_key"] is None or cfg["short_limit_key"] is None:
            return []
        trades = []
        for key, raw_q in portfolio["positions"].items():
            try:
                q = float(raw_q)
                spec = market["products"][key]
                lot = float(spec["lot_size"])
                limit_key = cfg["long_limit_key"] if q>=0 else cfg["short_limit_key"]
                limit = float(spec[limit_key])
                if not all(math.isfinite(v) for v in (q,lot,limit)) or lot<=0 or limit<0:
                    continue
                if abs(q) < cfg["trigger_fraction"]*limit or q==0:
                    continue
                target = math.floor(cfg["target_fraction"]*limit/lot)*lot
                qty = math.floor(max(abs(q)-target,0.0)/lot+1e-10)*lot
                if qty>0:
                    trades.append((key,"SELL" if q>0 else "BUY",qty))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return trades
