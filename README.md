# Reproducibility Artifact: Curvature, Scaling, and Complexity in Riemannian Projection-Free Optimization

This research artifact is accompanying the manuscript **“Curvature, Scaling, and Complexity in Riemannian Projection-Free Optimization.”**

This repository reproduces the three numerical studies reported in the manuscript:

1. **Hyperbolic same-policy rate.** The gap-feedback short-step rule is run on the negatively curved hard family for the full 72-case grid with `p in {4,5,6,8}`, `rho in {0.04,0.08}`, `kappa in {0.5,1,2}`, and three initial transverse states. The `p=3` runs are diagnostics and are not pooled with the theorem-validation grid.
2. **Jacobi distortion / Rauch certificate.** Intrinsic finite differences numerically evaluate the logarithm-map differential in constant curvature `-kappa^2` over 168 directional configurations.
3. **Affine-invariant SPD spectral certificate.** Finite differences evaluate the full differential spectrum of the affine-invariant logarithm map on SPD matrices for 56 anisotropic and 16 isotropic configurations.

No GPU is required. All experiments run on CPU with NumPy/SciPy/Pandas/Matplotlib.

## Repository layout

```text
experiments/
  hyperbolic/   # hard-family implementation and 72-case grid
  jacobi/       # space-form Jacobi / DLog finite-difference experiment
  spd/          # affine-invariant SPD DLog spectrum experiment
scripts/
  reproduce_all.py
  make_paper_figures.py
  verify_reference_results.py
reference_results/
  hyperbolic/   # frozen summary + representative trajectories
  jacobi/       # frozen 168-case results and audit summary
  spd/          # frozen 72-case results and audit summary
figures/        # figures used in the manuscript, for comparison
tests/          # geometry/numerics smoke tests
```

## Environment

Tested with Python 3.13. A clean environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1-minute verification

The following checks the frozen result tables against the numerical claims in the manuscript and regenerates the three manuscript-style figures from the archived data:

```bash
python scripts/verify_reference_results.py
python scripts/make_paper_figures.py
```

Expected headline values are approximately:

```text
Hyperbolic: 72/72 primary cases; max slope error 2.83e-3; max normalized-tail spread 9.66e-4; cap fraction 0.
Jacobi:     168 cases; max smin relative error 1.34e-11; max directional relative error 8.23e-11.
SPD:        56 primary + 16 isotropic; max smin relative error 1.04e-8; max full-spectrum relative error 1.93e-8.
```

Small floating-point differences across BLAS/LAPACK platforms are normal.

## Smoke tests

```bash
python scripts/reproduce_all.py --mode smoke
```

This executes the geometry/oracle self-tests for all three experiments and one short hyperbolic trajectory.

## Full reproduction from scratch

```bash
python scripts/reproduce_all.py --mode full --outdir results
```

The Jacobi and SPD grids are lightweight. The hyperbolic experiment is intentionally more expensive: the frozen grid contains 72 theorem-validation trajectories, each with 20,000 iterations and a 1,024-point LMO bracketing grid, plus three `p=3` diagnostics. Runtime therefore depends strongly on CPU speed.

Individual experiments can also be run directly:

```bash
PYTHONPATH=experiments/hyperbolic python experiments/hyperbolic/run_hyperbolic_grid.py --outdir results/hyperbolic
PYTHONPATH=experiments/jacobi     python experiments/jacobi/run_jacobi_grid.py     --outdir results/jacobi
PYTHONPATH=experiments/spd        python experiments/spd/run_spd_grid.py        --outdir results/spd
```

After a full run, compare the new summaries with the frozen reference values:

```bash
python scripts/verify_reference_results.py --results results
```

## Numerical protocol

### Hyperbolic hard family

The primary grid is fixed in `experiments/hyperbolic/run_hyperbolic_grid.py`. The reported slope is fitted on the preregistered final tail of each positive finite trajectory; there is no visual selection of the fitting interval. The theorem concerns `p>3`. The `p=3` experiment is retained only as a diagnostic for the marginal endpoint.

### Jacobi distortion

For each `(kappa, kappa*r)` pair, the code evaluates seven directions between radial and transverse modes. The numerical differential is obtained by centered intrinsic finite differences. The comparison target is the exact constant-curvature factor `kappa*r/sinh(kappa*r)` and its direction-dependent counterpart.

### SPD certificate

The experiment constructs general SPD pairs and computes a finite-difference matrix for `D Log`. Its numerical singular values are compared with the exact root-space spectrum. Isotropic pairs `Y=cX` separately test the prediction that relative spectral spread, rather than distance alone, controls distortion.

## Frozen reference data

The repository includes the compact data needed to independently verify every numerical statement in the manuscript. For the hyperbolic study, the complete 72-case summary and four representative 20,000-step trajectories are included; the full raw trajectory archive is omitted because it is substantially larger and is deterministically regenerated by the supplied code. The complete Jacobi and SPD result tables are included.

## Reference

If you use the code, numerical results, figures, or other materials from
this repository in academic work, please cite the following manuscript:

> B. T. Nguyen, *Curvature, Scaling, and Complexity in Riemannian
> Projection-Free Optimization*, manuscript.

```bibtex
@unpublished{nguyen_curvature_scaling_complexity,
  author = {Nguyen, Binh T.},
  title  = {Curvature, Scaling, and Complexity in Riemannian
            Projection-Free Optimization},
  note   = {Manuscript}
}


