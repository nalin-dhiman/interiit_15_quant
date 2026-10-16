"""Benchmark only the implemented foundation path, not the planned controller."""
import argparse,ast,json,platform,sys,time
from pathlib import Path
import numpy as np
from resetmm.accounting import Product
from resetmm.policies import InventoryQuote
from resetmm.risk import factor_risk

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="runs/benchmark.json")
    args=parser.parse_args()
    rng=np.random.default_rng(72);n=1000;k=8;R=100
    products={f'P{i}':Product(f'P{i}') for i in range(n)}
    positions={p:0.0 for p in products};prices={p:100.0 for p in products}
    orders=[dict(order_id=f'O{i:04d}',product_id=f'P{i%n}',side='BUY' if i%2 else 'SELL',quantity=1.) for i in range(R)]
    q=rng.normal(size=n);B=rng.normal(size=(n,k));d=rng.random(n);policy=InventoryQuote()
    elapsed=[]
    for _ in range(300):
        start=time.perf_counter_ns()
        policy.quote(orders,prices,positions,products,dt=.5);factor_risk(q,B,d)
        elapsed.append((time.perf_counter_ns()-start)/1e6)
    root=Path(__file__).resolve().parents[1]
    files=[]
    for folder in ['resetmm','experiments','tests','submission','scripts']:
        for path in (root/folder).glob('*.py'):
            ast.parse(path.read_text(),feature_version=(3,11));files.append(str(path.relative_to(root)))
    report=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
                products=n,factors=k,rfqs=R,repeats=len(elapsed),
                median_ms=float(np.median(elapsed)),p99_ms=float(np.quantile(elapsed,.99)),
                maximum_ms=float(max(elapsed)),python311_grammar_files=files,
                python311_runtime_tested=False,
                scope='InventoryQuote plus factor_risk only. Not IMC SDK or full planner certification.')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
