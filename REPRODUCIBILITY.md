# Reproducibility checklist

| Manuscript numerical statement | Artifact source | Reproduction |
|---|---|---|
| 72 hyperbolic primary configurations | `reference_results/hyperbolic/hyperbolic_summary.csv` | `experiments/hyperbolic/run_hyperbolic_grid.py` |
| max slope discrepancy about `2.83e-3` | same | `scripts/verify_reference_results.py` |
| max normalized-tail spread about `9.66e-4` | same | `scripts/verify_reference_results.py` |
| short-step cap inactive | same | `scripts/verify_reference_results.py` |
| 168 Jacobi directional configurations | `reference_results/jacobi/jacobi_all_cases.csv` | `experiments/jacobi/run_jacobi_grid.py` |
| max DLog singular-value error about `1.34e-11` | same | `scripts/verify_reference_results.py` |
| max directional error about `8.23e-11` | same | `scripts/verify_reference_results.py` |
| 56 anisotropic + 16 isotropic SPD cases | `reference_results/spd/spd_all_cases.csv` | `experiments/spd/run_spd_grid.py` |
| max SPD smallest-singular-value error about `1.04e-8` | same | `scripts/verify_reference_results.py` |
| max full-spectrum error about `1.93e-8` | same | `scripts/verify_reference_results.py` |
| isotropic error about `1.51e-8`, distance up to `15.811` | same | `scripts/verify_reference_results.py` |

## Independence of numerical checks

The hyperbolic iteration uses exact hyperbolic geometry and a numerical LMO on the full superellipse boundary; the asymptotic recurrence from the proof is not inserted into the iteration. The Jacobi and SPD experiments compute logarithm-map differentials by finite differences and compare the resulting singular values with the analytical certificates only after the numerical differential has been formed.

## Determinism

The grids, seeds, fitting tail, finite-difference scales, and tolerances are fixed in source. No stochastic training, GPU kernels, or external datasets are used. Numerical roundoff may vary slightly with BLAS/LAPACK implementations.
