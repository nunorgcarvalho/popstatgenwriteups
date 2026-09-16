"""Separate the two channels behind the amplitude gap: inbreeding, or off-diagonal LD?

    python channel_test.py
    python channel_test.py --self-test     # prove the guard fires on a known-wrong input

## The gap being explained

`drift_grid_fit.py` established that the `V_A` residual tracks realized drift with zero
intercept and no `M` dependence, but at a slope of **1.5-1.8** rather than the 1.0 that pure
heterozygosity deflation predicts. So drift is the driver and something else,
drift-proportional and `M`-independent, is eating additional `V_A`. Two candidates, and they
separate cleanly because the writeup's own equations share a numerator:

    eq:VA-eq-exact       V_A^(eq)     = (V_A^(0) - 2 sum_k beta_k^2 a_kk) / (1 - rho_g)
    eq:g-seg-var exact   Var[eps_o]   = (V_A^(0) - 2 sum_k beta_k^2 a_kk) / 2

The segregation variance is exactly **half the same numerator** and carries **no `rho_g` at
all**, because its derivation's double sum collapses to `k = l` -- it sees the **diagonal**
`a_kk` only. `V_A^(eq)` sees the diagonal too, and additionally the off-diagonal through
`(1 - rho_g)`, since `rho_g = mu V_A` and `V_A` sums over all variant pairs. Hence:

- **Channel A, inbreeding.** Mates sharing ancestors correlates an individual's two alleles at
  the same locus, which is precisely excess `a_kk`. It must deflate **both** `V_A^(eq)` and
  `Var[eps_o]`, through the one numerator, in a fixed ratio.
- **Channel B, off-diagonal LD degradation.** Drift scrambling the cross-locus structure the
  equilibrium recursion assumes intact. It moves `rho_g` and hence `V_A^(eq)`, and leaves
  `Var[eps_o]` untouched.

## The primary test, and why it is primary

Compare **measured** `sum_k beta_k^2 a_kk` against the `a_kk` at the **recursion's own fixed
point**, then substitute the measured value into `eq:VA-eq-exact` and see how much of the
residual it closes.

A direct measurement can **exclude** a channel outright. The segregation-variance route can
only fail to detect one, which is a weaker statement. The two `a_kk` values also come from
genuinely independent routes -- one read off the haplotype array by `hardy_weinberg_departure`,
the other exposed from the recursion as it solves -- so their agreement or disagreement means
something. Re-implementing the recursion's `a_kk` would have made both copies mine and their
agreement vacuous.

## The confirmatory test

Compare measured `Var[eps_o]` against `predict_seg_var_exact` at the measured `a_kk`. Under
Channel A both quantities are deflated consistently with one shared numerator; under Channel B
the segregation variance sits at its predicted value while `V_A` does not. Two independent
routes to the same diagonal, so agreement is worth having -- and disagreement is itself a
finding.

Every fit is per replicate with the SE taken across replicates (D-0017): the grid re-used the
same seeds, so each row compares a population's own `a_kk` against its own `V_A` shortfall.
"""

from __future__ import annotations

import argparse

import numpy as np

from drift_grid_fit import load, _ols


def report(cells: list[dict], strict: bool = True) -> dict:
    n = sum(c["residual_exact"].size for c in cells)
    print(f"{len(cells)} cells, {n} replicates\n")

    print("-- per cell, so the scaling can be read rather than trusted --")
    print(f"  {'cell':>18} {'a_kk meas/recur':>16} {'resid vs exact':>16} "
          f"{'resid w/ meas a':>16} {'seg resid':>12}")
    for c in cells:
        ratio = c["sum_beta2_a_meas"] / c["sum_beta2_a_theory"]
        r0, r1 = c["residual_exact"], c["residual_exact_meas_a"]
        sg = c["residual_seg_meas_a"]
        se = lambda v: v.std(ddof=1) / np.sqrt(v.size)
        print(f"  {c['name']:>18} {ratio.mean():>9.4f}+-{se(ratio):<6.4f} "
              f"{100 * r0.mean():>+9.3f}%+-{100 * se(r0):<5.3f} "
              f"{100 * r1.mean():>+9.3f}%+-{100 * se(r1):<5.3f} "
              f"{100 * sg.mean():>+7.3f}%")

    ratio = np.concatenate([c["sum_beta2_a_meas"] / c["sum_beta2_a_theory"] for c in cells])
    r0 = np.concatenate([c["residual_exact"] for c in cells])
    r1 = np.concatenate([c["residual_exact_meas_a"] for c in cells])
    sg = np.concatenate([c["residual_seg_meas_a"] for c in cells])
    # SE across cell means, which cannot be inflated by within-cell structure
    cm = lambda k: np.array([c[k].mean() for c in cells])
    cse = lambda k: float(cm(k).std(ddof=1) / np.sqrt(len(cells)))
    rat_m = float(cm("sum_beta2_a_meas").mean() / cm("sum_beta2_a_theory").mean())

    print("\n-- PRIMARY TEST: is the measured diagonal in excess of the recursion's? --")
    print("   The observed ratio has to be compared against the ratio Channel A would NEED")
    print("   in order to close the gap, not against 1 in isolation: a ratio consistent with")
    print("   1 only excludes Channel A once the required value is known to be far from 1.")
    per_cell_ratio, required = [], []
    for c in cells:
        A = c["sum_beta2_a_theory"].mean()
        num = c["VA_0"].mean() - 2 * A
        resid = c["residual_exact"].mean()
        # predicted V_A is proportional to (VA_0 - 2A); to move it by `resid` the diagonal
        # would have to become A - resid*num/2
        required.append((A - resid * num / 2) / A)
        per_cell_ratio.append((c["sum_beta2_a_meas"] / c["sum_beta2_a_theory"]).mean())
    per_cell_ratio = np.array(per_cell_ratio)
    required = np.array(required)
    rr_se = float(per_cell_ratio.std(ddof=1) / np.sqrt(len(cells)))
    print(f"     observed  measured/recursion a_kk   {per_cell_ratio.mean():.4f} "
          f"+- {rr_se:.4f}")
    print(f"     REQUIRED for Channel A to explain   {required.mean():.3f}   "
          f"(per cell {required.min():.2f} to {required.max():.2f})")
    gap_se = abs(required.mean() - per_cell_ratio.mean()) / rr_se if rr_se > 0 else np.inf
    a_excluded = gap_se > 3
    print(f"     -> observed differs from required by {gap_se:.0f} SE; "
          f"Channel A {'EXCLUDED' if a_excluded else 'not excluded'}")

    print("\n   and what substituting the measured diagonal does to the residual:")
    m0 = float(np.mean([c["residual_exact"].mean() for c in cells]))
    m1 = float(np.mean([c["residual_exact_meas_a"].mean() for c in cells]))
    print(f"     residual with the recursion's a_kk   {100 * m0:+.4f}%")
    print(f"     residual with the MEASURED  a_kk     {100 * m1:+.4f}%")
    print(f"     share of the gap closed              "
          f"{100 * (abs(m0) - abs(m1)) / abs(m0):+.1f}%   "
          f"(Channel A would need ~100%)")

    print("\n-- CONFIRMATORY TEST: does the deflation act on the shared numerator? --")
    print("   Channel B moves V_A only and leaves Var[eps_o] at its prediction, because the")
    print("   segregation variance sees the DIAGONAL alone. So the discriminating quantity is")
    print("   the DIFFERENCE between the two residuals, not either one on its own.")
    va = np.array([c["residual_exact_meas_a"].mean() for c in cells])
    sg = np.array([c["residual_seg_meas_a"].mean() for c in cells])
    d = va - sg
    d_se = float(d.std(ddof=1) / np.sqrt(len(d)))
    print(f"     V_A        vs prediction   {100 * va.mean():+.4f}%")
    print(f"     Var[eps_o] vs prediction   {100 * sg.mean():+.4f}%")
    print(f"     difference                 {100 * d.mean():+.4f}% +- {100 * d_se:.4f}%"
          f"   ({abs(d.mean()) / d_se:.2f} SE from 0)")
    common_factor = abs(d.mean()) < 2 * d_se
    b_excluded = common_factor and abs(sg.mean()) > 2 * float(
        np.std(sg, ddof=1) / np.sqrt(len(sg)))
    print(f"     -> the two are deflated by a {'COMMON' if common_factor else 'DIFFERENT'} "
          f"factor; Channel B {'EXCLUDED' if b_excluded else 'not excluded'}")

    print("\n-- VERDICT --")
    print(f"    Channel A (excess diagonal a_kk):        "
          f"{'EXCLUDED' if a_excluded else 'not excluded'}")
    print(f"    Channel B (off-diagonal LD only):        "
          f"{'EXCLUDED' if b_excluded else 'not excluded'}")
    both_excluded = a_excluded and b_excluded
    if both_excluded:
        print("    => BOTH CHANNELS EXCLUDED. The deflation is a single common factor on the")
        print("       shared numerator: it moves V_A and Var[eps_o] equally while leaving the")
        print("       diagonal a_kk at the value the recursion predicts. That is neither the")
        print("       diagonal nor the off-diagonal, and it is the escalation condition this")
        print("       task named. A dimensionless correlation being correct while both")
        print("       variances are deflated together points at the SCALE of the genetic")
        print("       variance -- V_A^(0) = sum beta_k^2 in pinned base units overstating the")
        print("       realized per-locus contribution -- rather than at the LD structure.")
    out = {"a_ratio": float(per_cell_ratio.mean()), "a_ratio_se": rr_se,
           "a_ratio_required": float(required.mean()),
           "a_excluded": a_excluded, "b_excluded": b_excluded,
           "resid_recursion": m0, "resid_measured": m1,
           "seg_resid": float(sg.mean()), "common_factor": common_factor,
           "both_excluded": both_excluded}

    if strict and both_excluded:
        # RAISE on the escalation condition (D-0022). This is not an error: it is the
        # outcome the task said to escalate rather than absorb, and a run that reported it
        # with exit 0 would let a caller treat it as a settled explanation.
        raise SystemExit("BOTH CHANNELS EXCLUDED -- escalate; a drift-proportional, "
                         "M-independent deflation that neither the diagonal nor the "
                         "off-diagonal accounts for")
    return out


def self_test() -> None:
    """Prove the guard fires: force the two tests into disagreement and require a raise.

    Writing a guard and verifying a guard are different acts. Here the measured diagonal is
    inflated so Channel A reads as IMPLICATED, while the segregation variance is pinned at
    its prediction so the confirmatory test reads Channel B -- a combination that must be
    refused rather than reported.
    """
    cells = load()
    if not cells:
        raise SystemExit("no cells on disk to self-test against")
    # (a) Channel A made to look like the whole explanation: the measured diagonal is set
    #     to exactly the value that would close the gap. The both-excluded raise must NOT
    #     fire, because there is then a channel that explains it.
    import copy
    inflated = copy.deepcopy(cells)
    for c in inflated:
        A = c["sum_beta2_a_theory"]
        num = c["VA_0"] - 2 * A
        c["sum_beta2_a_meas"] = A - c["residual_exact"] * num / 2
    print("=== (a) Channel A forced to explain the gap: the raise must NOT fire ===")
    try:
        report(inflated, strict=True)
        print("\n  correctly did NOT raise: a channel explains it\n")
    except SystemExit as exc:
        raise SystemExit(f"guard fired when it should not have: {exc}")

    # (b) the real escalation shape: diagonal correct AND both variances deflated together.
    #     Noise is kept nonzero so the SEs are not degenerate -- an earlier version of this
    #     self-test pinned a residual to exactly zero, which made its SE zero, made the
    #     `< 2*SE` comparison false, and let the guard pass for the wrong reason. The test
    #     failed to test, which is the very thing it exists to catch.
    rng = np.random.default_rng(0)
    forced = copy.deepcopy(cells)
    for c in forced:
        c["sum_beta2_a_meas"] = c["sum_beta2_a_theory"] * (
            1.0 + 0.001 * rng.standard_normal(c["sum_beta2_a_theory"].shape))
        c["residual_exact_meas_a"] = -0.01 + 0.002 * rng.standard_normal(
            c["residual_exact_meas_a"].shape)
        c["residual_seg_meas_a"] = -0.01 + 0.002 * rng.standard_normal(
            c["residual_seg_meas_a"].shape)
    print("=== (b) diagonal correct and both variances deflated: the raise MUST fire ===")
    try:
        report(forced, strict=True)
    except SystemExit as exc:
        print(f"\n  GUARD FIRED as intended: {exc}")
        return
    raise SystemExit("GUARD DID NOT FIRE on the escalation shape -- the test is not a test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-strict", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    cells = load()
    if not cells:
        raise SystemExit("no cell files found; run drift_grid_cell.py first")
    missing = [c["name"] for c in cells if "sum_beta2_a_meas" not in c]
    if missing:
        raise SystemExit(f"cells lack the a_kk fields (regenerate): {missing}")
    report(cells, strict=not args.no_strict)


if __name__ == "__main__":
    main()
