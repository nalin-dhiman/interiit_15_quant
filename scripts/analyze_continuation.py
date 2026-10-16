"""Produce auditable tables/figures from completed, unchanged diagnostic data."""
import argparse
import csv
import json
from pathlib import Path
import sys
import platform

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); a=p.parse_args()
    run=Path(a.run); out=run/'analysis'; out.mkdir(exist_ok=True)
    exact=json.loads((run/'exact_values.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    rows=[]
    for x in exact:
        method=x['methods']; label=x['model']
        s=next(r for r in summary if r['model']==label and r['method']=='R2_joint')
        rows.append(dict(model=label, B3_expected_pnl=method['B3']['raw_pnl'],
                         R2_expected_pnl=method['R2_joint']['raw_pnl'],
                         R2_fixed_map_expected_pnl=method['R2_fixed_quotes']['raw_pnl'],
                         B3_objective=method['B3']['objective'],
                         R2_objective=method['R2_joint']['objective'],
                         oracle_objective=x['oracle_objective'],
                         sampled_paired_difference=s['paired_difference'],
                         ci_low=s['descriptive_ci_low'],ci_high=s['descriptive_ci_high']))
    with (out/'economic_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Model & B3 $\E\Pi$ & R2 $\E\Pi$ & R2 fixed map & O objective\\',r'\midrule']
    for row in rows:
        label=row['model'].replace('_',r'\_')
        lines.append(f"{label} & {row['B3_expected_pnl']:.4f} & {row['R2_expected_pnl']:.4f} & {row['R2_fixed_map_expected_pnl']:.4f} & {row['oracle_objective']:.4f}\\\\")
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (out/'economic_table.tex').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs=list(range(len(rows)))
    fig,ax=plt.subplots(figsize=(10,5.2))
    means=[r['sampled_paired_difference'] for r in rows]
    ax.errorbar(xs,means,yerr=[[m-r['ci_low'] for m,r in zip(means,rows)],
                             [r['ci_high']-m for m,r in zip(means,rows)]],
                fmt='o',capsize=4,color='#126782',label='50 paired sessions; descriptive 95% bootstrap')
    ax.scatter(xs,[r['R2_expected_pnl']-r['B3_expected_pnl'] for r in rows],
               marker='D',s=35,color='#a04326',label='Exact expectation in the stated synthetic kernel')
    ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(xs,[r['model'].replace('_risk','\nrisk=') for r in rows],rotation=25,ha='right')
    ax.set_ylabel('R2 minus B3 raw PnL per synthetic session')
    ax.set_title('Exploratory finite-model diagnostic; no competition evidence')
    ax.legend(loc='upper right',fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'paired_pnl.png',dpi=180);fig.savefig(out/'paired_pnl.pdf');plt.close(fig)
    report=dict(python=sys.version,platform=platform.platform(),matplotlib=matplotlib.__version__,
                rows=len(rows),confidence_scope='descriptive, unadjusted, independent sessions within model',
                actual_market_claim=False)
    (out/'analysis_environment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
