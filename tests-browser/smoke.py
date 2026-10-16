"""Real browser checks against the built static app, including its Pages prefix."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:5173/interiit_15_quant/')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/browser')
    parser.add_argument('--executable')
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    server = None
    if args.serve:
        server = subprocess.Popen(['node', 'tools/serve.mjs', '--production'], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                urllib.request.urlopen(args.url, timeout=1).close()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError('Static server did not start')
                time.sleep(.1)
        with sync_playwright() as p:
            options = dict(headless=True, args=['--no-sandbox'])
            downloads = ROOT / 'artifacts/browser-downloads'
            downloads.mkdir(parents=True, exist_ok=True)
            options['downloads_path'] = str(downloads)
            if args.executable:
                options['executable_path'] = args.executable
            browser = p.chromium.launch(**options)
            page = browser.new_page(viewport={'width': 1280, 'height': 900}, accept_downloads=True)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
            page.goto(args.url, wait_until='networkidle')
            assert page.locator('#run-state').inner_text() == 'Ready'
            assert page.locator('#pnl-value').inner_text() == '0.00'
            assert page.locator('#policy').input_value() == 'B4'
            page.locator('#step-button').click()
            assert page.locator('#session-progress-label').inner_text() == 'Interval 1 / 80'
            assert page.locator('#ledger-status').inner_text() == 'Balanced'
            page.locator('#reset-button').click()
            assert page.locator('#run-state').inner_text() == 'Ready'
            page.locator('#steps').fill('10')
            page.locator('#policy').select_option('R2B')
            page.locator('#apply-button').click()
            page.locator('#play-button').click()
            page.wait_for_function("document.querySelector('#run-state').textContent === 'Settled'", timeout=30000)
            with page.expect_download() as download:
                page.locator('#export-config').click()
            data = json.loads(Path(download.value.path()).read_text())
            assert data['done'] and data['step'] == 10 and data['config']['policy'] == 'R2B'
            assert len(data['rfqHistory']) == data['config']['steps'] * data['config']['rfqsPerStep']
            cash = sum((Decimal(str(t['cashChange'])) for t in data['trades']), Decimal('0'))
            assert abs(cash - Decimal(str(data['cash']))) < Decimal('0.0000001')
            q = {'A': Decimal('0'), 'B': Decimal('0')}
            for trade in data['trades']:
                sign = Decimal('1') if trade['side'] == 'BUY' else Decimal('-1')
                q[trade['product']] += sign * Decimal(str(trade['quantity']))
            assert all(q[k] == Decimal(str(data['inventory'][k])) for k in q)
            with page.expect_download() as download:
                page.locator('#export-run').click()
            assert 'cashChange' in Path(download.value.path()).read_text().splitlines()[0]
            page.locator('#batch-paths').select_option('32')
            page.locator('#compare-button').click()
            page.wait_for_function("!document.querySelector('#export-comparison').hidden", timeout=90000)
            assert page.locator('.policy-code').count() == 3
            assert all('32 paths' in value for value in page.locator('.policy-code').all_text_contents())
            assert 'R2B − B4 paired mean' in page.locator('#comparison-caption').inner_text()
            page.screenshot(path=str(args.out / 'desktop.png'), full_page=True)
            page.locator('#compare-button').click()
            page.locator('#reset-button').click()
            assert 'Compare policies' in page.locator('#compare-button').inner_text()
            assert page.locator('#run-state').inner_text() == 'Ready'
            mobile = browser.new_page(viewport={'width': 390, 'height': 844})
            mobile.goto(args.url, wait_until='networkidle')
            assert mobile.locator('#config-toggle').get_attribute('aria-expanded') == 'false'
            assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            mobile.locator('#config-toggle').click()
            assert mobile.locator('#config-toggle').get_attribute('aria-expanded') == 'true'
            mobile.locator('#config-toggle').click()
            mobile.locator('#step-button').click()
            mobile.screenshot(path=str(args.out / 'mobile.png'), full_page=True)
            assert not errors, errors
            browser.close()
        record = dict(status='PASS', desktop=[1280, 900], mobile=[390, 844],
            prefix_url=args.url, policies_checked=['B4', 'R2B'], comparison_paths=32,
            independent_exported_ledger_replay=True, full_rfq_history=True,
            cancellation=True, csv_json_downloads=True, console_errors=errors,
            mobile_overflow=False, screenshots=['desktop.png', 'mobile.png'])
        (args.out / 'validation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
    finally:
        if server:
            server.terminate()
            server.wait(timeout=5)


if __name__ == '__main__':
    main()
