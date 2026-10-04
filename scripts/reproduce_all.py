#!/usr/bin/env python3
from pathlib import Path
import argparse, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

def run(cmd, env=None):
    print('+',' '.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)), cwd=ROOT, env=env, check=True)

def penv(sub):
    e=os.environ.copy(); e['PYTHONPATH']=str(ROOT/'experiments'/sub); return e

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','full'],default='smoke'); ap.add_argument('--outdir',default='results'); a=ap.parse_args()
    py=sys.executable
    if a.mode=='smoke':
        run([py,ROOT/'tests/test_hyperbolic.py'],penv('hyperbolic'))
        run([py,ROOT/'tests/test_jacobi.py'],penv('jacobi'))
        run([py,ROOT/'tests/test_spd.py'],penv('spd'))
        run([py,ROOT/'experiments/hyperbolic/hyperbolic_same_policy.py','--p','4','--rho','.04','--kappa','1','--max-iter','100','--lmo-grid','256','--outdir',Path(a.outdir)/'smoke_hyperbolic'],penv('hyperbolic'))
        print('SMOKE REPRODUCTION: PASS')
    else:
        out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
        run([py,ROOT/'experiments/hyperbolic/run_hyperbolic_grid.py','--outdir',out/'hyperbolic'],penv('hyperbolic'))
        run([py,ROOT/'experiments/jacobi/run_jacobi_grid.py','--outdir',out/'jacobi'],penv('jacobi'))
        run([py,ROOT/'experiments/spd/run_spd_grid.py','--outdir',out/'spd'],penv('spd'))
        print('FULL REPRODUCTION COMPLETE')
if __name__=='__main__': main()
