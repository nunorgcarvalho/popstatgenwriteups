#!/usr/bin/env python3
# Example sbatch (one array task per grid cell, 7 cells):
# sbatch --array=1-7 -J am_drift_grid -p short -t 2:00:00 --mem=8G -c 1 \
#   --wrap="/n/groups/price/nuno/.venv_py13/bin/python /n/groups/price/nuno/popstatgenwriteups/writeups/statistical_genetics/relative_covariance/simulation/drift_grid_cell.py --cell ${SLURM_ARRAY_TASK_ID}"
#
# Cost scales roughly as N x t, so the cells are deliberately unequal and the longest
# (N=4000, t=64) sets the wall clock. Split a cell's replicates across more tasks with
# --rep-start/--rep-count if it overruns.

"""Run ONE cell of the drift grid and write its per-replicate measurements to scratch.

Creates or updates:

- `/n/scratch/users/n/nur479/relative_covariance_sim/drift_grid/cell_<name>.npz`, one row per
  replicate: the measured `V_A`, the exact-theory `V_A`, the residual against it, the measured
  `rel_dhet`, and the parameters the prediction was evaluated at.

The question this grid settles: a residual of `-0.575% +- 0.262%` survives
`eq:VA-eq-correction` at `M = 50`, and the writeup's approximation chain is NOT the cause --
`eq:VA-eq-corrected` reproduces the exact fixed point of `eq:c-recursion` to 0.03% at that
`M`, and popstatgensim's direct measurement of `a_kk` moves the residual by 0.016 percentage
points. So the residual localizes to the simulator or a violated assumption, and finite-`N`
drift is the leading candidate.

## The pre-registered test, and its sign

Drift deflates heterozygosity: `E[2 p_t (1-p_t)] = 2 p_0 (1-p_0)(1-F)`. The prediction here is
built from the pinned `sum_k beta_k^2` at BASE frequencies, which do not drift, while measured
`V_A` tracks realized heterozygosity. So

    residual = V_A_measured / V_A_exact - 1 = (1 - F) - 1 = -F = rel_dhet

and the test is: **regress the residual on `rel_dhet`, expect slope +1 and intercept 0.**

`rel_dhet` is used directly and `F` is never formed. `F = -rel_dhet`, so any design stated in
`F` carries a negation, and a sign inversion there reads as "the predicted value" at a glance
and survives review -- which is exactly what happened to an earlier version of this spec.

Two properties of `rel_dhet` that matter:

- It is a **ratio of means** across variants, `mean(2 p_t (1-p_t)) / mean(2 p_0 (1-p_0)) - 1`,
  not a mean of per-variant ratios. The identity holds per variant with a common `F`, so the
  ratio of means is exactly `1 - F`, while a mean of ratios is noise-dominated by rare
  variants with small denominators. The two differ materially at small `p`.
- `mean_rel_dsd` must NOT be substituted for it. That field is an unsigned mean of `|dSD|` --
  drift magnitude, not signed heterozygosity loss -- and the two scale differently (`|dp|` like
  `sqrt(t/(2N))`, `E[2p(1-p)]` like `t/(2N)`). It is ~10x larger by `t = 24` and grows
  smoothly, so fitting it against a signed prediction would give a large, clean, WRONG slope.

## Why the theory baseline is the exact fixed point

`V_A_exact` iterates `eq:c-recursion` to its fixed point on the replicate's own `beta`
spectrum, so there is no approximation anywhere on the theory side and the residual cannot be
confounded with `eq:VA-eq`'s `1/M_e` error. The residual against `eq:VA-eq-corrected` is
recorded alongside for continuity with the earlier measurement, but the fit uses the exact
one.

`V_A^(0)` is the pinned `sum_k beta_k^2` and not the realized base `Var[g]`. That is what makes
`residual = rel_dhet` exact rather than attenuated: the pinned diagonal does not drift, so the
whole `(1 - F)` lands in the ratio. Feeding realized base `Var[g]` would put part of the drift
into the base measurement itself and attenuate the slope for a second reason, on top of
errors-in-variables.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from popstatgensim import BaseStandardization, GeneticEffect, Population
from popstatgensim.traits import effective_n_variants

OUT_DIR = Path("/n/scratch/users/n/nur479/relative_covariance_sim/drift_grid")

#: The grid. Six cells at M=50 spanning N and t, plus one M=200 cell.
#:
#: The signal lives at SMALL N and LONG t -- the neutral prediction is -t/(2N), so -3.2% at
#: (1000, 64) down to -0.4% at (4000, 32). The large-N cells are here to show the effect
#: VANISHING, not to detect it; reporting them as failures to detect would misread the design.
#:
#: The M=200 cell can falsify the hypothesis on its own: under drift the residual depends on
#: N and t but NOT on M, so small-M and mid-M must agree once N and t are controlled. The two
#: existing points sit at -0.73% (M=50) and +0.07% (M=200) at N=4000, t=32.
CELLS: tuple[dict, ...] = (
    {"name": "N1000_t32_M50",  "N": 1000, "t": 32, "M": 50},
    {"name": "N1000_t64_M50",  "N": 1000, "t": 64, "M": 50},
    {"name": "N2000_t32_M50",  "N": 2000, "t": 32, "M": 50},
    {"name": "N2000_t64_M50",  "N": 2000, "t": 64, "M": 50},
    {"name": "N4000_t32_M50",  "N": 4000, "t": 32, "M": 50},
    {"name": "N4000_t64_M50",  "N": 4000, "t": 64, "M": 50},
    {"name": "N4000_t32_M200", "N": 4000, "t": 32, "M": 200},
)

BURN_IN = 12          # measurement starts after this; V_A plateaus by ~generation 8
H2_0 = 0.5
AM_R = 0.5
N_REPS = 60           # raised from 40: the regressor carries its own per-replicate noise


def exact_fixed_point(beta: np.ndarray, V_E: float, rho_y: float,
                      iters: int = 20000) -> tuple[float, float]:
    """Iterate `eq:c-recursion` to its exact fixed point -- no approximation.

    The same recursion `figures/make_figures.py` uses in `check_equilibrium`.

    Returns `(V_A, sum_k beta_k^2 a_kk)`. The second value is the recursion's OWN `a_kk` at
    the fixed point, **exposed rather than recomputed**. That distinction matters: a second
    implementation of the same quantity would be a duplication whose agreement proves
    nothing, since both copies would be mine. The measured `a_kk` this is compared against
    comes from an entirely different route -- read off the haplotype array by
    `hardy_weinberg_departure` -- which is what makes the comparison informative.
    """
    b = np.asarray(beta, dtype=float)
    c = b / 2.0
    a_kk = np.zeros_like(b)
    for _ in range(iters):
        V_A = 2.0 * np.sum(b * c)
        V_P = V_A + V_E
        nxt = 0.5 * (1.0 + rho_y * V_A / V_P) * c + b * (0.25 - 0.5 * a_kk)
        a_kk = (rho_y / V_P) * c ** 2
        c = nxt
    return float(2.0 * np.sum(b * c)), float(np.sum(b ** 2 * a_kk))


def run_replicate(cell: dict, seed: int) -> dict:
    """One replicate: simulate, then measure the residual and `rel_dhet` at the same point."""
    np.random.seed(seed)
    M = cell["M"]
    pop = Population(N=cell["N"], M=M, seed=seed,
                     params={"keep_past_generations": 1})
    pop.assign_sex()
    pop.add_trait(name="y",
                  effects={"A": GeneticEffect(var_indep=H2_0, M=M, M_causal=M, name="A")},
                  var_Eps=1.0 - H2_0)
    std = BaseStandardization.from_population(pop, method="observed")

    beta = std.to_standardized_effects(
        np.asarray(pop.traits["y"].effects["A"].effects_per_allele, dtype=float))
    VA_0 = float(np.sum(beta ** 2))               # the pinned DIAGONAL (D-0018 / D-0023)

    pop.set_params(AM_r=AM_R, AM_trait="y", AM_type="phenotypic", s=0.0)
    pop.simulate_generations(cell["t"])

    # Measure over the post-burn-in tail of this replicate's own trajectory. Measuring at a
    # single generation would be noisier, and drift is cumulative so the tail is where it is
    # largest -- but rel_dhet is read at the SAME generation as V_A, so both sides of the
    # regression refer to the same state.
    g = np.asarray(pop.traits["y"].y_["A"], dtype=float)
    y = np.asarray(pop.traits["y"].y, dtype=float)
    V_A = float(g.var(ddof=1))
    V_E = float((y - g).var(ddof=1))

    from popstatgensim.estimation import measure_am_quantities
    m = measure_am_quantities(pop, "y", standardization=std, include_drift=True)
    rho_y, rho_g = float(m.rho_y), float(m.rho_g)
    M_e = float(effective_n_variants(pop, "y", std)["M_e"])

    VA_exact, sum_beta2_a_theory = exact_fixed_point(beta, V_E, rho_y)
    VA_corrected = (VA_0 / (1.0 - rho_g)) * (1.0 - rho_g / (2.0 * (1.0 - rho_g) * M_e))

    # -- the two channels ----------------------------------------------------------------
    # Channel A is excess DIAGONAL a_kk (inbreeding correlating an individual's two alleles
    # at one locus). Channel B is off-diagonal LD degradation. They separate because
    # eq:VA-eq-exact and eq:g-seg-var SHARE the numerator (V_A^(0) - 2 sum beta^2 a_kk),
    # with the segregation variance exactly half of it and carrying no rho_g at all -- so
    # Channel A deflates both in a fixed ratio while Channel B moves V_A only.
    from popstatgensim.estimation import predict_VA_eq_exact, predict_seg_var_exact
    from popstatgensim.traits import hardy_weinberg_departure

    hw = hardy_weinberg_departure(pop, "y", standardization=std)
    sum_beta2_a_meas = float(hw["sum_beta2_a"])
    # V_A and Var[eps_o] predicted from the MEASURED a_kk: if Channel A is the whole story,
    # substituting the measured diagonal should close the gap on both at once.
    VA_exact_meas_a = float(predict_VA_eq_exact(VA_0, sum_beta2_a_meas, rho_g))
    seg_exact_meas_a = float(predict_seg_var_exact(VA_0, sum_beta2_a_meas))
    seg_exact_theory_a = float(predict_seg_var_exact(VA_0, sum_beta2_a_theory))

    drift = m.drift or std.allele_freq_drift(pop)
    if "rel_dhet" not in drift:
        raise KeyError("allele_freq_drift() has no rel_dhet field; the signed quantity is "
                       "required and mean_rel_dsd must NOT be substituted for it")

    return {
        "seed": seed, "N": cell["N"], "t": cell["t"], "M": M,
        "V_A": V_A, "V_E": V_E, "VA_0": VA_0, "rho_y": rho_y, "rho_g": rho_g, "M_e": M_e,
        "VA_exact": VA_exact, "VA_corrected": VA_corrected,
        "residual_exact": V_A / VA_exact - 1.0,
        "residual_corrected": V_A / VA_corrected - 1.0,
        # -- the primary test: measured a_kk against the recursion's own -----------------
        "sum_beta2_a_meas": sum_beta2_a_meas,
        "sum_beta2_a_theory": sum_beta2_a_theory,
        "mean_a_kk": float(hw.get("mean_a_kk", np.nan)),
        # V_A residual once the MEASURED diagonal is substituted in: if Channel A accounts
        # for the gap this goes to zero, and what it does NOT close is Channel B's share.
        "VA_exact_meas_a": VA_exact_meas_a,
        "residual_exact_meas_a": V_A / VA_exact_meas_a - 1.0,
        # -- the confirmatory test: the shared numerator's other half --------------------
        "var_eps_o": float(m.var_eps_o),
        "seg_exact_meas_a": seg_exact_meas_a,
        "seg_exact_theory_a": seg_exact_theory_a,
        "residual_seg_meas_a": float(m.var_eps_o) / seg_exact_meas_a - 1.0,
        "residual_seg_theory_a": float(m.var_eps_o) / seg_exact_theory_a - 1.0,
        "rel_dhet": float(drift["rel_dhet"]),
        "mean_rel_dsd": float(drift.get("mean_rel_dsd", np.nan)),   # diagnostic ONLY
        "neutral_pred": -cell["t"] / (2.0 * cell["N"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=int, required=True,
                    help="1-based cell index (slurm array task id)")
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, c in enumerate(CELLS, start=1):
            print(f"  {k}  {c['name']:<18} N={c['N']:<5} t={c['t']:<3} M={c['M']:<4} "
                  f"neutral prediction {-100 * c['t'] / (2 * c['N']):+.2f}%")
        return

    if not 1 <= args.cell <= len(CELLS):
        raise SystemExit(f"--cell must be 1..{len(CELLS)}")
    cell = CELLS[args.cell - 1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"cell {args.cell}: {cell['name']} "
          f"(N={cell['N']} t={cell['t']} M={cell['M']}), {args.reps} replicates, "
          f"neutral prediction {100 * -cell['t'] / (2 * cell['N']):+.3f}%", flush=True)

    rows = []
    t0 = time.time()
    for k in range(args.rep_start, args.rep_start + args.reps):
        rows.append(run_replicate(cell, seed=args.seed0 + 1000 * args.cell + k))
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{args.rep_start + args.reps} reps, {el:.0f}s elapsed",
                  flush=True)

    keys = sorted(rows[0])
    arrays = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    suffix = "" if args.rep_start == 0 else f"_r{args.rep_start}"
    out = OUT_DIR / f"cell_{cell['name']}{suffix}.npz"
    np.savez(out, **arrays, cell=json.dumps(cell))

    res = arrays["residual_exact"]
    rd = arrays["rel_dhet"]
    se = res.std(ddof=1) / np.sqrt(res.size)
    am, at = arrays["sum_beta2_a_meas"], arrays["sum_beta2_a_theory"]
    print(f"  residual vs exact   {100 * res.mean():+.4f}% +- {100 * se:.4f}%")
    print(f"  a_kk measured/recursion  {(am / at).mean():.4f}  "
          f"(1.0 would exclude Channel A)")
    print(f"  residual with measured a_kk  "
          f"{100 * arrays['residual_exact_meas_a'].mean():+.4f}%")
    print(f"  measured rel_dhet   {100 * rd.mean():+.4f}% "
          f"(neutral {100 * rows[0]['neutral_pred']:+.4f}%)")
    print(f"  wrote {out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
