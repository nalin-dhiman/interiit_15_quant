"""Independent ledger, exposure, reservation and inference audit of a sealed study.

No SDK or policy import is required. The already audited independent ledger
replay is reused; reservation-box arithmetic, cell counts, parameter selection,
pilot sizing and the bootstrap are reconstructed here from the raw records.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_research_study import independent_ledger, fixed_price_quote

DOMAIN = 'constrained-attribution-v1/20261017T051731Z'
BUDGETS = {'tight': {'A': 10., 'B': 5.}, 'wide': {'A': 20., 'B': 10.}}
CONTROLLERS = ('B4', 'R2B')
MODES = ('fixed', 'inventory')
FINAL_FAMILIES = ('size_mix', 'public_basis')
DEVELOPMENT = ('standard', 'wide')
PUBLIC_ORDER = ('order_id', 'customer_id', 'product_id', 'side', 'quantity', 'expires_interval')
TOLERANCE = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def close(actual, expected, name='quantity'):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise AssertionError(f'Nonfinite {name}')
    error = abs(float(actual) - float(expected))
    if error > TOLERANCE:
        raise AssertionError(f'{name}: {actual!r} != {expected!r}')
    return error


def seed_for(phase, scenario, path, stream):
    return int.from_bytes(hashlib.sha256(
        f'{DOMAIN}/{phase}/{scenario}/{path}/{stream}'.encode()).digest(), 'big')


def independent_box(positions, reservations=(), exchange_items=()):
    """Coordinate extrema over every subset; opposite actions never net."""
    box = {pid: [float(positions.get(pid, 0.)), float(positions.get(pid, 0.))]
           for pid in ('A', 'B')}
    for customer, items in ((True, reservations), (False, exchange_items)):
        for item in items:
            pid, side, quantity = ((item['product_id'], item['side'], item['quantity'])
                                   if customer else item)
            if pid not in box or side not in ('BUY', 'SELL') or not math.isfinite(quantity) or quantity <= 0:
                raise AssertionError('Invalid reservation or pending execution item')
            sign = (-1 if side == 'BUY' else 1) if customer else (1 if side == 'BUY' else -1)
            box[pid][0 if sign < 0 else 1] += sign * quantity
    return box


def audit_risk_trace(row):
    traces = row['risk_budget_trace']
    counters = row['risk_budget_counters']
    caps = BUDGETS[row['budget']]
    if not traces or counters['checks'] != len(traces) or counters['trace_dropped'] != 0:
        raise AssertionError('Missing or truncated risk-certificate history')
    if counters['actual_breaches'] or counters['possible_breaches']:
        raise AssertionError('Observed or possible private envelope breach')
    maximum_actual = maximum_possible = 0.
    pending_checks = reservation_checks = 0
    for record in traces:
        positions = record['positions']
        if set(positions) != {'A', 'B'}:
            raise AssertionError('Unsupported risk-trace inventory domain')
        if record.get('risk_is_variance_matched') is not False or record.get('cash_penalty_imposed') is not False:
            raise AssertionError('Private inventory preference confused with risk equality or cash penalty')
        if record['within_envelope'] is not True or record['all_subsets_within_envelope'] is not True:
            raise AssertionError('Failed private envelope certificate')
        if record['risk_weights'] != {'A': 1., 'B': 2.} or record['multipliers'] != {'A': 1., 'B': 1.}:
            raise AssertionError('Unexpected exposure weights or multipliers')
        reservations, items = record['reserved_orders'], record['pending_items']
        if len(reservations) != record['reserved_customer_orders'] or len(items) != record['pending_exchange_items']:
            raise AssertionError('Missing underlying reservation or pending item')
        if items:
            origin = record['pending_origin']
            if not isinstance(origin, dict) or set(origin) != {'A', 'B'} or reservations:
                raise AssertionError('Pending hedge lacks its original account or overlaps live customer capacity')
            box = independent_box(origin, exchange_items=items)
            pending_checks += 1
        else:
            box = independent_box(positions, reservations)
            reservation_checks += bool(reservations)
        if set(record['possible_box']) != set(box) or set(record['policy_bounds']) != set(caps):
            raise AssertionError('Incomplete possible-position box')
        for pid, cap in caps.items():
            lower, upper = record['policy_bounds'][pid]
            close(lower, -cap, 'private lower bound'); close(upper, cap, 'private upper bound')
            close(record['possible_box'][pid][0], box[pid][0], 'all-subset lower extremum')
            close(record['possible_box'][pid][1], box[pid][1], 'all-subset upper extremum')
            if box[pid][0] < -cap - TOLERANCE or box[pid][1] > cap + TOLERANCE:
                raise AssertionError('A possible fill/execution subset exceeds private capacity')
            if not box[pid][0] - TOLERANCE <= positions[pid] <= box[pid][1] + TOLERANCE:
                raise AssertionError('Delivered inventory not inside the certified execution box')
        gross = abs(positions['A']) + 2 * abs(positions['B'])
        possible = max(abs(v) for v in box['A']) + 2 * max(abs(v) for v in box['B'])
        limit = caps['A'] + 2 * caps['B']
        close(record['gross'], gross, 'actual gross exposure')
        close(record['possible_gross'], possible, 'all-subset gross exposure')
        close(record['gross_limit'], limit, 'common gross ceiling')
        maximum_actual = max(maximum_actual, gross)
        maximum_possible = max(maximum_possible, possible)
    close(counters['max_actual_gross'], maximum_actual, 'maximum traced actual exposure')
    close(counters['max_possible_gross'], maximum_possible, 'maximum traced subset exposure')
    return dict(checks=len(traces), pending_checks=pending_checks,
                reservation_checks=reservation_checks, max_actual_gross=maximum_actual,
                max_possible_gross=maximum_possible)


def audit_executed_envelope(row):
    """Reconstruct actual inventory after every full, unmodified trade."""
    caps = BUDGETS[row['budget']]
    q = {'A': 0., 'B': 0.}
    maxima = {'A': 0., 'B': 0.}
    max_gross = 0.
    for trade in row['own_trades']:
        pid = trade['product_id']
        if pid not in q or trade['side'] not in ('BUY', 'SELL'):
            raise AssertionError('Unsupported executed inventory update')
        q[pid] += trade['quantity'] * (1 if trade['side'] == 'BUY' else -1)
        close(trade['position_after'], q[pid], 'executed position')
        for name in q:
            maxima[name] = max(maxima[name], abs(q[name]))
            if abs(q[name]) > caps[name] + TOLERANCE:
                raise AssertionError('Actual customer/exchange execution exceeded the private envelope')
        max_gross = max(max_gross, abs(q['A']) + 2 * abs(q['B']))
    for index, pid in enumerate(('A', 'B')):
        close(row['max_abs_inventory'][index], maxima[pid], 'actual peak inventory')
    if row['account']['penalties_paid'] != 0:
        raise AssertionError('A private envelope must not create an official cash penalty')
    return dict(actual_max_gross=max_gross, actual_max_inventory=maxima)


def audit_observations(row, prices):
    observations = row['final_rfq_observations']
    keys = [(o['interval_id'], o['order_id']) for o in observations]
    if len(keys) != 36 or len(set(keys)) != 36 or row['rfqs'] != 36:
        raise AssertionError('Expected one final observation for each of 36 RFQs')
    if any(sum(o['interval_id'] == interval for o in observations) != 3 for interval in range(1, 13)):
        raise AssertionError('Incomplete RFQ interval schedule')
    public = [{key: o[key] for key in PUBLIC_ORDER} for o in observations]
    digest = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    if digest != row['public_customer_tape_sha256']:
        raise AssertionError('Public customer tape hash differs from raw RFQs')
    trades = {(t['interval_id'], t['order_id']): t for t in row['own_trades'] if t['counterparty'] == 'customer'}
    if len(trades) != sum(t['counterparty'] == 'customer' for t in row['own_trades']):
        raise AssertionError('Multiple customer executions for one RFQ')
    fixed = {}
    checked = 0
    for o in observations:
        pid, interval = o['product_id'], o['interval_id']
        matching_version = 10 * interval
        close(o['matching_reference'], prices[matching_version - 1][pid], 'matching reference')
        close(o['next_reference'], prices[matching_version][pid], 'next-reference label')
        direction = 1 if o['side'] == 'BUY' else -1
        if o['side'] not in ('BUY', 'SELL') or pid not in ('A', 'B'):
            raise AssertionError('Unsupported public RFQ')
        close(o['dealer_signed_one_tick_markout'], -direction * (
            o['next_reference'] - o['matching_reference']), 'post-fill label sign')
        live = o['own_quote']
        age = o['accepted_quote_age_seconds']
        if (live is None) != (age is None):
            raise AssertionError('Quote/age missingness mismatch')
        if live is not None:
            if age < 0:
                raise AssertionError('Negative quote age')
            close(age, o['matching_virtual_time'] - o['quote_installation_virtual_time'], 'quote age')
            close(o['edge'], direction * (live - o['matching_reference']), 'signed quote edge')
            if row['quote_mode'] == 'fixed':
                close(live, fixed_price_quote(o, prices, row['parameters']), 'fixed causal quote formula')
                checked += 1
        trade = trades.get((interval, o['order_id']))
        if bool(o['won']) != (trade is not None) or bool(o['won']) != (o['public_winner'] == 'dealer_1'):
            raise AssertionError('RFQ winner and actual fill disagree')
        if trade is not None:
            if trade['product_id'] != pid or trade['side'] != ('SELL' if direction == 1 else 'BUY'):
                raise AssertionError('Customer/dealer fill sign mismatch')
            close(trade['quantity'], o['quantity'], 'full customer fill quantity')
            close(trade['price'], live, 'acknowledged quote execution')
            close(o['public_winning_price'], live, 'public matching price')
        fixed[(interval, o['order_id'])] = live
    if set(trades) != {(o['interval_id'], o['order_id']) for o in observations if o['won']}:
        raise AssertionError('Customer-trade/observed-fill correspondence incomplete')
    return dict(fixed_formula_checks=checked, quotes=fixed)


def audit_record(row, prices):
    if row['controller'] not in CONTROLLERS or row['method'] != row['controller']:
        raise AssertionError('Unknown controller')
    if row['budget'] not in BUDGETS or row['quote_mode'] not in MODES:
        raise AssertionError('Unknown budget or quote mode')
    if row['crash'] or row['callback_errors'] or row['own_rejections'] or row['rejections']:
        raise AssertionError('Operational failure cannot be omitted from inference')
    if row['boundary_counters'].get('unsupported_histories', 0):
        raise AssertionError('Unsupported boundary history')
    if row['seed'] != seed_for(row['phase'], row['scenario'], row['path'], 'customers'):
        raise AssertionError('Incorrect or consumed primitive namespace')
    n = row['account']['price_version']
    if n != 124 or len(prices) != 124 or row['price_rows_consumed'] != n:
        raise AssertionError('Incomplete price/closing-window schedule')
    tape = [{'version': i + 1, 'prices': p} for i, p in enumerate(prices)]
    if hashlib.sha256(json.dumps(tape, sort_keys=True).encode()).hexdigest() != row['price_tape_sha256']:
        raise AssertionError('Archived price tape differs from actually observed prices')
    ledger = independent_ledger(row, prices)
    actual = audit_executed_envelope(row)
    risk = audit_risk_trace(row)
    observations = audit_observations(row, prices)
    return dict(ledger=ledger, actual=actual, risk=risk, observations=observations)


def expected_parameters(seal, row):
    params = json.loads(json.dumps(seal['selected'][row['budget'] + '/' + row['quote_mode']][row['package']]))
    if row['scenario'] == 7:
        params['settlement_coefficients'] = {}
    return params


def path_cells(rows, seal=None):
    """Validate all 32 cells; opponent repetitions are not extra paths."""
    if len(rows) != 32:
        raise AssertionError('Every path requires exactly 32 predeclared policy cells')
    cells = {}
    identities = {(r['scenario'], r['path'], r['phase']) for r in rows}
    if len(identities) != 1:
        raise AssertionError('Mixed primitive paths')
    for row in rows:
        key = (row['family'], row['budget'], row['quote_mode'], row['package'], row['controller'])
        if key in cells:
            raise AssertionError('Duplicate policy cell')
        cells[key] = row
        if seal is not None and row['parameters'] != expected_parameters(seal, row):
            raise AssertionError('Final package changed after its seal')
    expected = set(itertools.product(FINAL_FAMILIES, BUDGETS, MODES, CONTROLLERS, CONTROLLERS))
    if set(cells) != expected:
        raise AssertionError('Missing or undeclared final cells')
    if len({(r['price_tape_sha256'], r['public_customer_tape_sha256']) for r in rows}) != 1:
        raise AssertionError('Exogenous tapes differ between policy/budget/package/mode cells')
    fixed_common = fixed_one_side = 0
    for family, budget, mode, package in itertools.product(FINAL_FAMILIES, BUDGETS, MODES, CONTROLLERS):
        a = cells[family, budget, mode, package, 'B4']
        b = cells[family, budget, mode, package, 'R2B']
        if a['parameters'] != b['parameters']:
            raise AssertionError('Paired controllers use different parameter packages')
        if mode != 'fixed':
            continue
        qa = {(o['interval_id'], o['order_id']): o['own_quote'] for o in a['final_rfq_observations']}
        qb = {(o['interval_id'], o['order_id']): o['own_quote'] for o in b['final_rfq_observations']}
        if set(qa) != set(qb):
            raise AssertionError('Paired RFQ identities differ')
        for key in qa:
            if qa[key] is not None and qb[key] is not None:
                close(qa[key], qb[key], 'simultaneously live fixed-rule quote prices')
                fixed_common += 1
            elif (qa[key] is None) != (qb[key] is None):
                fixed_one_side += 1
    differences = [cells[f, 'wide', 'fixed', 'B4', 'R2B']['account']['pnl'] -
                   cells[f, 'wide', 'fixed', 'B4', 'B4']['account']['pnl'] for f in FINAL_FAMILIES]
    return dict(primary_difference=float(np.mean(differences)), fixed_common=fixed_common,
                fixed_one_side=fixed_one_side)


def recompute_primary(path_differences):
    groups = [np.asarray([value for (s, _), value in sorted(path_differences.items()) if s == scenario])
              for scenario in range(8)]
    if len({len(g) for g in groups}) != 1 or any(len(g) < 2 or not np.isfinite(g).all() for g in groups):
        raise AssertionError('Missing, unbalanced or invalid independent path strata')
    rng = np.random.default_rng(seed_for('analysis', 0, 0, 'bootstrap/B4/wide/fixed'))
    bootstrap = np.zeros(5000)
    for group in groups:
        samples = rng.integers(0, len(group), size=(5000, len(group)))
        bootstrap += np.mean(group[samples], axis=1) / 8.
    lower, upper = map(float, np.quantile(bootstrap, [.025, .975]))
    return dict(paired_mean_difference=float(np.mean([g.mean() for g in groups])),
                stratified_path_bootstrap_95=[lower, upper], independent_paths=sum(map(len, groups)),
                materiality=1., material_benefit_supported=lower > 1.,
                material_benefit_excluded=upper < 1.,
                practical_equivalence_supported=lower > -1. and upper < 1.)


def audit_source_archive(declaration, seal):
    declared = json.loads(declaration.read_text())
    if declared['namespace'] != DOMAIN or seal['namespace'] != DOMAIN:
        raise AssertionError('Unregistered experiment namespace')
    if declared['source_hashes'] != seal['source_hashes'] or sha(declaration) != seal['declaration_sha256']:
        raise AssertionError('Policy seal differs from prospective source declaration')
    if declared['final_prices_generated'] or declared['tuning_or_validation_executed'] or seal['final_generated']:
        raise AssertionError('Declaration did not precede selection/test generation')
    base = declaration.parent / 'executed_sources'
    for key, digest in seal['source_hashes'].items():
        target = base / key
        if not target.is_file() or sha(target) != digest:
            raise AssertionError('Archived declared source missing or modified: ' + key)
    return len(seal['source_hashes'])


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if not line.endswith('\n'):
                raise AssertionError('Incomplete raw JSON record')
            yield json.loads(line)


def audit_pilot(pilot, seal):
    with gzip.open(pilot / 'price_tapes.json.gz', 'rt') as stream:
        tapes = json.load(stream)
    scores, parameters, seen = defaultdict(list), {}, set()
    for row in read_rows(pilot / 'tuning.jsonl.gz'):
        key = (row['budget'], row['quote_mode'], row['controller'], row['candidate'])
        identity = key + (row['scenario'], row['path'])
        if (row['phase'] != 'tuning' or row['candidate'] not in range(24) or row['scenario'] not in range(4)
                or row['path'] not in range(3) or row['family'] != DEVELOPMENT[row['path'] % 2] or identity in seen):
            raise AssertionError('Missing, duplicate or out-of-protocol tuning case')
        audit_record(row, tapes[f'tuning/{row["scenario"]}/{row["path"]}'])
        seen.add(identity)
        scores[key].append(row['account']['pnl'])
        parameters.setdefault(key, row['parameters'])
        if parameters[key] != row['parameters']:
            raise AssertionError('A tuning candidate changed parameters')
    if len(seen) != 2304 or any(len(values) != 12 for values in scores.values()) or len(scores) != 192:
        raise AssertionError('Unequal or incomplete tuning budget')
    nominated = {}
    for budget, mode in itertools.product(BUDGETS, MODES):
        nominated[budget + '/' + mode] = {}
        for controller in CONTROLLERS:
            best = max(range(24), key=lambda candidate: np.mean(scores[budget, mode, controller, candidate]))
            nominated[budget + '/' + mode][controller] = best
    if nominated != seal['nominated_candidates']:
        raise AssertionError('Training nominations differ from independently reconstructed argmax')
    validation, validation_seen, paired = defaultdict(list), set(), defaultdict(dict)
    for row in read_rows(pilot / 'validation.jsonl.gz'):
        key = (row['budget'], row['quote_mode'], row['package'], row['controller'])
        identity = key + (row['scenario'], row['path'], row['family'])
        if (row['phase'] != 'validation' or row['scenario'] not in range(8) or row['path'] not in range(12)
                or row['family'] not in DEVELOPMENT or row['package'] not in CONTROLLERS or identity in validation_seen):
            raise AssertionError('Missing, duplicate or out-of-protocol validation case')
        index = nominated[row['budget'] + '/' + row['quote_mode']][row['package']]
        expected = json.loads(json.dumps(parameters[row['budget'], row['quote_mode'], row['controller'], index]))
        if row['scenario'] == 7:
            expected['settlement_coefficients'] = {}
        if row['candidate'] != index or row['parameters'] != expected:
            raise AssertionError('Validation differs from the nominated common package')
        audit_record(row, tapes[f'validation/{row["scenario"]}/{row["path"]}'])
        validation_seen.add(identity)
        validation[key].append(row['account']['pnl'])
        if (row['budget'], row['quote_mode']) == ('wide', 'fixed'):
            paired[row['package'], row['scenario'], row['path'], row['family']][row['controller']] = row['account']['pnl']
    if len(validation_seen) != 3072 or any(len(values) != 192 for values in validation.values()) or len(validation) != 16:
        raise AssertionError('Unequal or incomplete validation budget')
    selection = {}
    for budget, mode in itertools.product(BUDGETS, MODES):
        key = budget + '/' + mode
        selection[key] = {}
        for controller in CONTROLLERS:
            means = {label: float(np.mean(validation[budget, mode, label, controller])) for label in CONTROLLERS}
            best = max(CONTROLLERS, key=lambda label: (means[label], -nominated[key][label]))
            index = nominated[key][best]
            selection[key][controller] = best
            sealed = seal['selection'][key][controller]
            if sealed['chosen_nomination'] != best or sealed['candidate'] != index:
                raise AssertionError('Validation package argmax changed')
            for label, value in means.items():
                close(sealed['validation_means'][label], value, 'pilot validation mean')
            if seal['selected'][key][controller] != parameters[budget, mode, controller, index]:
                raise AssertionError('Selected common package differs from the raw pilot')
    primary_label = selection['wide/fixed']['B4']
    by_path = defaultdict(list)
    for (label, scenario, path, family), values in paired.items():
        if label == primary_label:
            if set(values) != set(CONTROLLERS):
                raise AssertionError('Incomplete paired pilot')
            by_path[scenario, path].append(values['R2B'] - values['B4'])
    if len(by_path) != 96 or any(len(values) != 2 for values in by_path.values()):
        raise AssertionError('Opponent repeats became extra pilot paths')
    groups = [np.asarray([np.mean(values) for (s, _), values in sorted(by_path.items()) if s == scenario]) for scenario in range(8)]
    variance = float(np.mean([np.var(g, ddof=1) for g in groups]))
    required = math.ceil((1.960 + .842) ** 2 * variance)
    n = min(4096, 8 * math.ceil(max(1024, required) / 8))
    close(seal['within_stratum_effective_variance'], variance, 'within-stratum planning variance')
    if (seal['required_independent_paths'] != required or seal['final_paths'] != n or seal['paths_per_scenario'] != n // 8
            or seal['precision_cap_binds'] != (required > n) or seal['tuning_sessions'] != 2304 or seal['validation_sessions'] != 3072):
        raise AssertionError('Prospective sample-size plan differs from raw pilot units')
    return dict(tuning_sessions=len(seen), validation_sessions=len(validation_seen), pilot_independent_paths=96,
                complete_pilot_ledger_replay=True, within_stratum_effective_variance=variance,
                required_independent_paths=required, final_paths=n, precision_cap_binds=required > n)


def audit_study(study, pilot, declaration, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    seal = json.loads((study / 'POLICY_SEAL.json').read_text())
    if seal != json.loads((pilot / 'POLICY_SEAL.json').read_text()):
        raise AssertionError('Final uses a different pilot seal')
    source_count = audit_source_archive(declaration, seal)
    pilot_result = audit_pilot(pilot, seal)
    with gzip.open(study / 'price_tapes.json.gz', 'rt') as stream:
        tapes = json.load(stream)
    expected_paths = {(s, p) for s in range(8) for p in range(seal['paths_per_scenario'])}
    if set(tapes) != {f'{s}/{p}' for s, p in expected_paths}:
        raise AssertionError('Missing, extra or excluded final price path')
    differences, seen_paths, maximum_errors = {}, set(), {}
    path_rows, current = [], None
    totals = dict(sessions=0, own_trades=0, rfqs=0, risk_checks=0, pending_subset_checks=0,
                  customer_subset_checks=0, fixed_formula_checks=0, fixed_common=0, fixed_one_side=0)
    prefix = list(read_rows(study / 'sequential_prefix.jsonl.gz'))
    if len(prefix) != 32:
        raise AssertionError('Sequential replay prefix must retain all 32 cells')
    from scripts.audit_research_study import economic_fingerprint
    prefix_hashes = [economic_fingerprint(row) for row in prefix]
    prefix_verified = False

    def finish_path(rows):
        nonlocal prefix_verified
        identity = rows[0]['scenario'], rows[0]['path']
        if identity in seen_paths or identity not in expected_paths:
            raise AssertionError('Duplicate, unexpected or excluded primitive path')
        seen_paths.add(identity)
        result = path_cells(rows, seal)
        differences[identity] = result['primary_difference']
        totals['fixed_common'] += result['fixed_common']
        totals['fixed_one_side'] += result['fixed_one_side']
        if identity == (0, 0):
            if [economic_fingerprint(row) for row in rows] != prefix_hashes:
                raise AssertionError('Sequential/parallel prefix changed economics or information')
            prefix_verified = True

    try:
        for row in read_rows(study / 'sessions.jsonl.gz'):
            identity = row['scenario'], row['path']
            if row['phase'] != 'final':
                raise AssertionError('Consumed pilot or smoke path entered final evidence')
            if current is not None and identity != current:
                finish_path(path_rows); path_rows = []
            current = identity
            prices = tapes[f'{identity[0]}/{identity[1]}']
            result = audit_record(row, prices)
            for key, error in result['ledger']['maximum_errors'].items():
                maximum_errors[key] = max(maximum_errors.get(key, 0.), error)
            totals['sessions'] += 1
            totals['own_trades'] += len(row['own_trades'])
            totals['rfqs'] += len(row['final_rfq_observations'])
            totals['risk_checks'] += result['risk']['checks']
            totals['pending_subset_checks'] += result['risk']['pending_checks']
            totals['customer_subset_checks'] += result['risk']['reservation_checks']
            totals['fixed_formula_checks'] += result['observations']['fixed_formula_checks']
            path_rows.append(row)
        if path_rows:
            finish_path(path_rows)
        if seen_paths != expected_paths or len(differences) != seal['final_paths'] or totals['sessions'] != 32 * seal['final_paths']:
            raise AssertionError('Incomplete or inflated final independent sample count')
        if not prefix_verified:
            raise AssertionError('Sequential/parallel prefix was not verified')
        primary = recompute_primary(differences)
        declared_primary = json.loads((study / 'primary_inference.json').read_text())
        for key in ('paired_mean_difference',):
            close(declared_primary[key], primary[key], 'independent primary mean')
        for a, b in zip(declared_primary['stratified_path_bootstrap_95'], primary['stratified_path_bootstrap_95']):
            close(a, b, 'independent primary bootstrap endpoint')
        for key in ('independent_paths', 'material_benefit_supported', 'material_benefit_excluded', 'practical_equivalence_supported'):
            if declared_primary[key] != primary[key]:
                raise AssertionError('Reported primary decision differs from independent raw inference')
        execution = json.loads((study / 'execution.json').read_text())
        if (execution['sessions'] != totals['sessions'] or execution['independent_paths'] != len(differences)
                or execution['sequential_parallel_prefix_sessions'] != 32 or not execution['prefix_identical']
                or not execution['source_unchanged'] or not execution['all_failures_retained']
                or execution['realised_risk_matched'] is not False or execution['common_ex_ante_envelope'] is not True):
            raise AssertionError('Execution receipt count or scope differs from audited evidence')
        if (study / 'failed_cases').exists() and any((study / 'failed_cases').rglob('*.json')):
            raise AssertionError('A failed case exists; successful-case-only inference is prohibited')
        report = dict(status='NUMERICALLY_CHECKED', complete=True, no_exclusions=True,
            independent_paths=len(differences), **totals, maximum_errors=maximum_errors,
            declared_source_files=source_count, primary=primary, pilot=pilot_result,
            sequential_parallel_prefix_verified=True, prefix_new_independent_samples=0,
            envelope_not_realised_variance_matching=True,
            limitations=['Bootstrap coverage is approximate under independent generated paths and fixed strata.',
                         'The existing verified independent ledger routine is reused, not an additional independent accounting derivation.',
                         'Trace arithmetic is independently reconstructed from archived reservations and pending items; a complete exchange action log is not reconstructed.',
                         'This audit establishes declared synthetic bookkeeping and inference, not market fidelity or novelty.'],
            elapsed_seconds=time.perf_counter() - started)
        dump(out / 'analysis.json', report)
        dump(out / 'environment.json', dict(python=sys.version, numpy=np.__version__, platform=platform.platform()))
        dump(out / 'config.json', dict(study=str(study), pilot=str(pilot), declaration=str(declaration), policy_changed=False))
        (out / 'REPORT.md').write_text(
            '# Independent constrained RFQ attribution audit\n\n'
            f"Audited all {totals['sessions']:,} final sessions on {len(differences):,} independent paths, "
            f"{totals['own_trades']:,} own trades and {totals['rfqs']:,} final RFQ records. "
            'No path was excluded or replaced. Complete pilot ledgers, nominations, common-package selection and within-stratum sample planning were also reconstructed.\n\n'
            'Customer signs, principal, fees, actual per-trade inventory, retained settlement, monetary decomposition and price-innovation exposure reconcile. '
            f"Independently recomputed {totals['risk_checks']:,} private-envelope checks from archived reserved quantities and pending execution subsets. "
            'Private caps impose no official money penalty. The common rectangle bounds admissible exposure; it does not equalize realised variance or downside.\n\n'
            f"Primary R2B-minus-B4 mean {primary['paired_mean_difference']:.6f}; approximate stratified path-bootstrap 95% interval {primary['stratified_path_bootstrap_95']}. "
            f"Material benefit supported: {primary['material_benefit_supported']}; material benefit excluded: {primary['material_benefit_excluded']}; "
            f"practical equivalence supported for this mean estimand: {primary['practical_equivalence_supported']}. "
            'Opponent repetitions remain inside each path; the 32-cell sequential/parallel prefix adds zero independent samples.\n\n'
            'Status: NUMERICALLY_CHECKED for this declared synthetic accounting, constraint and inference audit. '
            'The previous verified independent ledger routine is reused. Bootstrap coverage is an empirical assumption. '
            'Reservation-box arithmetic is independently checked against archived participant instrumentation, rather than a full exchange action history. '
            'No market-profitability, risk-dominance or novelty conclusion follows from reconciliation alone.\n')
        dump(out / 'SHA256SUMS.json', {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'})
        return report
    except Exception as error:
        dump(out / 'FAILED_CHECK.json', dict(error=repr(error), sessions_audited=totals['sessions'], paths_audited=len(seen_paths)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit_study(args.study.resolve(), args.pilot.resolve(), args.declaration.resolve(), args.out.resolve())
    print(json.dumps({key:report[key] for key in ('status','sessions','independent_paths','primary')}, indent=2))


if __name__ == '__main__':
    main()
