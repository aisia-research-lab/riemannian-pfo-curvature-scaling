#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def read_metric_table(path):
    df = pd.read_csv(path)
    if list(df.columns) == ['metric','value']:
        return dict(zip(df.metric, df.value))
    return df.iloc[0].to_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=None, help='Optional freshly reproduced result root.')
    a = ap.parse_args()
    base = Path(a.results) if a.results else ROOT/'reference_results'

    hyperbolic_path = base/'hyperbolic'/'hyperbolic_summary.csv' if a.results else base/'hyperbolic'/'hyperbolic_summary.csv'
    jacobi_path = base/'jacobi'/'jacobi_all_cases.csv'
    spd_path = base/'spd'/'spd_all_cases.csv'
    # Fresh scripts use hyperbolic_summary.csv/jacobi_all_cases.csv/spd_all_cases.csv under same subdirs.
    hyperbolic = pd.read_csv(hyperbolic_path)
    primary = hyperbolic[hyperbolic.case_class == 'primary']
    max_slope = primary.h_slope_error.abs().max()
    max_spread = primary.norm_h_spread_10_90.max()
    max_cap = primary.cap_fraction.max()

    jacobi = pd.read_csv(jacobi_path)
    max_smin = jacobi.err_smin_rel.max()
    max_dir = jacobi.err_dir_rel.max()
    min_slack = jacobi.certificate_slack.min()

    spd = pd.read_csv(spd_path)
    spd_anisotropic = spd[spd['case'].str.startswith('primary')]
    spd_isotropic = spd[spd['case'].str.startswith('isotropic')]
    max_spd_min = spd_anisotropic.relerr_min.max()
    max_spd_spec = spd_anisotropic.max_spectrum_relerr.max()
    max_iso = spd_isotropic.relerr_min.max()
    max_iso_dist = spd_isotropic.distance.max()

    print(f'Hyperbolic: {len(primary)}/72 primary cases')
    print(f'  max |fitted-predicted slope| = {max_slope:.12g}')
    print(f'  max normalized-tail spread  = {max_spread:.12g}')
    print(f'  max cap fraction             = {max_cap:.12g}')
    print(f'Jacobi: {len(jacobi)}/168 directional cases')
    print(f'  max smin relative error      = {max_smin:.12g}')
    print(f'  max directional rel. error   = {max_dir:.12g}')
    print(f'  min certificate slack        = {min_slack:.12g}')
    print(f'SPD: {len(spd_anisotropic)} primary + {len(spd_isotropic)} isotropic cases')
    print(f'  max smin relative error      = {max_spd_min:.12g}')
    print(f'  max full-spectrum rel. error = {max_spd_spec:.12g}')
    print(f'  max isotropic sigma error    = {max_iso:.12g}')
    print(f'  max isotropic distance       = {max_iso_dist:.12g}')

    assert len(primary) == 72
    assert max_slope < 3e-3
    assert max_spread < 1e-3
    assert max_cap == 0
    assert len(jacobi) == 168 and max_smin < 2e-10 and max_dir < 2e-10
    assert len(spd_anisotropic) == 56 and len(spd_isotropic) == 16
    assert max_spd_min < 2e-8 and max_spd_spec < 3e-8 and max_iso < 2e-8
    print('\nREFERENCE RESULT AUDIT: PASS')

if __name__ == '__main__':
    main()
