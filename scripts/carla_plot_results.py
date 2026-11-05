"""Render actual final CARLA logs; does not connect to the simulator."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='outputs/carla');args=p.parse_args();out=Path(args.input)
    metrics=json.loads((out/'metrics.json').read_text())
    fig,axes=plt.subplots(3,1,figsize=(10,9))
    names=['calibrated_physics','learned_response'];colors=['#285f9e','#d66a25']
    for name,color in zip(names,colors):
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((out/('closed_%s_401.csv'%name)).open())]
        t=np.array([r['step']*.1 for r in rows])
        axes[0].plot(t,[r['speed'] for r in rows],label=name,color=color)
        axes[1].plot([r['x'] for r in rows],[r['y'] for r in rows],label=name,color=color)
        axes[2].plot(t,[r['cycle_ms'] for r in rows],label=name,color=color,alpha=.7)
    axes[0].plot(t,[r['target'] for r in rows],'k--',label='target')
    axes[0].set(xlabel='time (s)',ylabel='speed (m/s)',title='CARLA 0.9.15 / shared CEM / paired seed 401')
    axes[0].legend();axes[1].set(xlabel='right-handed x (m)',ylabel='right-handed y (m)',title='Observed trajectory')
    axes[2].set(xlabel='time (s)',ylabel='complete cycle (ms)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'closed_loop_comparison.png',dpi=150);plt.close(fig)
    summaries=[]
    for name in names:
        rows=[r for r in metrics['closed_loop'] if r['model']==name]
        summaries.append(dict(model=name,episodes=len(rows),collisions=sum(r['collision'] for r in rows),
            completions=sum(r['completion'] for r in rows),
            speed_rmse_mean_m_s=float(np.mean([r['speed_rmse_m_s'] for r in rows])),
            progress_mean_m=float(np.mean([r['progress_m'] for r in rows])),
            cycle_p95_range_ms=[min(r['cycle_p95_ms'] for r in rows),max(r['cycle_p95_ms'] for r in rows)]))
    (out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8');print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
