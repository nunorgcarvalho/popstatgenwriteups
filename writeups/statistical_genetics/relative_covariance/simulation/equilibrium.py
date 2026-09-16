"""The equilibrium tier: section 2.2's predictions against simulation.

    python equilibrium.py                      # the smallM regime
    python equilibrium.py equilibrium-midM     # a different regime

Follows the revised verification plan of task-20260915-224203, which is organized around one
question per check: **does this check depend on CODE rather than on ALGEBRA?** The writeup's
algebra is already verified symbolically against pathMgr, and that leg is green, so a
simulation check earns its place only where it touches the mechanism, the finite-N reality,
or the simulator's implementation. Three consequences shape what is below.

**Some entries are measurements, not tests.** Under the measured-parameter rule (D-0009),
a prediction whose inputs already determine it carries no information. `Cov[g_m, g_p] =
rho_g V_A` is an identity once `rho_g` is measured as exactly that correlation, and so is
`Cov[y_m, y_p] = rho_y V_Y`. Both are reported, and both are **labelled** as measurements
rather than printed as passing checks (D-0011). The trap is not wasted effort; it is a green
tick that means nothing.

**`eq:hsq-eq` is demoted.** Once `V_A` and `V_E` are measured, `h^2 = V_A/(V_A+V_E)` is
definitional, and the equilibrium form is an algebraic rearrangement of `eq:VA-eq` that the
oracle already checks in all three printed forms. Its only simulation content is that `V_E`
actually stayed fixed, which is one assertion, not a section.

**`eq:VA-eq` is tested as an approximation, not to a tolerance.** `eq:VA-eq-correction`
gives the term it drops as a relative `rho_g/(2(1-rho_g)M_e)`, and states the sign: dropping
it **overstates** `V_A^(eq)`. So the prediction is a signed residual of known size, and
asserting `eq:VA-eq` "within Monte-Carlo error" would absorb exactly the interesting part
into the error bars.

## Reduction: predict per generation, then average

Every prediction is evaluated at each generation's own measured inputs, and only the
resulting residuals are averaged. Averaging the inputs first and predicting once is biased,
because the predictions are convex in `rho_g`: `E[V_A0/(1-rho_g)] > V_A0/(1-E[rho_g])`, so
pooling first understates the prediction and manufactures a positive residual. The bias is
small here (`rho_g` fluctuates by a couple of percent between generations) but it is the same
order as the effects being measured, and it is free to avoid.

For the same reason `h^2_0` is taken **covariance-free** where available: an empirically
measured `V_Y^(0)` includes the realized base covariance between `g` and `e`, whereas the
writeup's `V_Y^(0)` is `V_A^(0) + V_E` with no covariance term, because `GE-indep` says it is
zero. Feeding the empirical one in injects a noise term larger than the residual under test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import generate_populations as gen
import theory

#: Generations to discard before measuring. AM-equilibrium cites six to ten; the burn-in
#: actually used is reported and calibrated from the trajectory rather than assumed.
BURN_IN = 12


@dataclass
class Residual:
    """One equation's signed residual, averaged over generations and replicates."""

    label: str
    predicted: float
    measured: float
    rel_err: float          # (measured / predicted - 1), averaged per generation then pooled
    se: float               # SE of that mean across replicates
    kind: str               # 'test' or 'measurement' (D-0011)
    note: str = ""

    @property
    def n_se(self) -> float:
        return abs(self.rel_err) / self.se if self.se > 0 else float("nan")


def _post_burn_in(traj: list[dict], burn_in: int) -> list[dict]:
    rows = [r for r in traj if r["t"] > burn_in and r.get("rho_g") is not None]
    if not rows:
        raise ValueError(f"no post-burn-in rows past t={burn_in}")
    return rows


def _base_row(traj: list[dict]) -> dict:
    return next(r for r in traj if r["t"] == 0)


def per_replicate(bundle: dict, burn_in: int = BURN_IN) -> dict:
    """Evaluate every prediction per generation, then average within this replicate.

    Returns a dict of per-equation mean relative errors, plus the measured parameters and
    the diagnostics. Averaging happens over generations here and over replicates in
    `summarize`, so the prediction is never evaluated at a pooled input.
    """
    traj = bundle["trajectory"]
    if not traj:
        raise ValueError("bundle has no trajectory; regenerate with the current script")
    base, rows = _base_row(traj), _post_burn_in(traj, burn_in)

    # V_A^(0) is sum_k beta_k^2, NOT the realized base-generation Var[g]. The derivation of
    # eq:VA-eq-exact expands Var[g] over variant PAIRS and the V_A^(0) that falls out is the
    # diagonal; the realized Var[g] additionally carries base-generation LD and cross-pair
    # sampling noise, which is a zero-mean nuisance rather than part of the quantity. Using
    # the diagonal is both the right definition and substantially quieter -- it nearly halves
    # the SE of every residual below, because that nuisance cancels instead of propagating.
    VA_0 = base.get("sum_beta2") or base["V_A"]
    VA_0_realized = base["V_A"]
    # Covariance-free h2_0 where the simulator reports it; the writeup's h^2_0 has no
    # covariance term because GE-indep sets it to zero.
    h2_0 = base.get("h2_cov_free") or base["h2"]

    acc: dict[str, list[float]] = {k: [] for k in
                                   ("eq:rhog-eq", "eq:rhog-t", "eq:VA-eq", "eq:VA-eq-corrected",
                                    "eq:hsq-eq", "eq:g-seg-var",
                                    "cov[g_m,g_p]", "cov[y_m,y_p]")}
    meas: dict[str, list[float]] = {k: [] for k in
                                    ("V_A", "V_Y", "h2", "rho_y", "rho_g", "var_eps_o",
                                     "M_e", "V_E", "corr_g_e")}

    for r in rows:
        rho_y, rho_g = r["rho_y"], r["rho_g"]
        V_A, V_Y, h2 = r["V_A"], r["V_Y"], r["h2"]
        V_E = r.get("V_E") or (V_Y - V_A)
        M_e = r["M_e"]

        # -- eq:rhog-eq, split into its two claims (revised plan) --------------------------
        # (a) the MECHANISM claim, per generation: does the pairing induce rho_g = rho_y h^2?
        acc["eq:rhog-t"].append(rho_g / theory.rho_g_from_pair(rho_y, h2) - 1.0)
        # (b) the DYNAMICS claim: is the fixed point the one eq:rhog-eq solves for?
        acc["eq:rhog-eq"].append(rho_g / theory.rho_g_eq(rho_y, h2_0) - 1.0)

        # -- eq:VA-eq, and the same thing with the dropped drag term kept -----------------
        acc["eq:VA-eq"].append(V_A / theory.VA_eq(VA_0, rho_g) - 1.0)
        acc["eq:VA-eq-corrected"].append(
            V_A / theory.VA_eq_corrected(VA_0, rho_g, M_e) - 1.0)

        # -- demoted to an invariant: h^2 is definitional given V_A and V_E ----------------
        acc["eq:hsq-eq"].append(h2 / theory.h2_eq(h2_0, rho_g) - 1.0)

        # -- eq:g-seg-var: the only tier-1 quantity depending on the simulator's meiosis ---
        acc["eq:g-seg-var"].append(r["var_eps_o"] / theory.seg_var(V_A, rho_g) - 1.0)

        # -- the two identities (D-0011) ---------------------------------------------------
        # Deliberately NOT evaluated as residuals. Cov[g_m,g_p] = rho_g V_A and
        # Cov[y_m,y_p] = rho_y V_Y cannot be anything other than exact, because rho_g and
        # rho_y are measured as precisely those correlations. Computing "measured/predicted"
        # here would print 0.0000% and read as a passing check; the value of the entry is
        # the number itself, so that is what is recorded.
        acc["cov[g_m,g_p]"].append(rho_g * V_A)
        acc["cov[y_m,y_p]"].append(rho_y * V_Y)

        for k in meas:
            v = r.get(k)
            if v is not None:
                meas[k].append(float(v))

    out = {"rel_err": {k: float(np.mean(v)) for k, v in acc.items()},
           "measured": {k: float(np.mean(v)) for k, v in meas.items() if v},
           "VA_0": VA_0, "VA_0_realized": VA_0_realized, "h2_0": h2_0, "n_gen": len(rows),
           "regime": bundle["regime"], "trajectory": traj}
    return out


def summarize(reps: list[dict]) -> list[Residual]:
    """Pool per-replicate residuals into one row per equation, with an SE across replicates."""
    labels = [
        ("eq:g-seg-var",       "test",        "the simulator's meiosis; pathMgr ASSUMES this "
                                              "in the g-only model, deriving it only at the "
                                              "allele level"),
        ("eq:rhog-t",          "test",        "mechanism: does the rank-matching copula "
                                              "induce rho_g = rho_y h^2?"),
        ("eq:rhog-eq",         "test",        "dynamics: is the fixed point the right one?"),
        ("eq:VA-eq",           "test",        "approximation VA-inflation, uncorrected"),
        ("eq:VA-eq-corrected", "test",        "with the eq:VA-eq-correction drag term kept"),
        ("eq:hsq-eq",          "test",        "DEMOTED: definitional given V_A and V_E; the "
                                              "content is that V_E stayed fixed"),
        ("cov[g_m,g_p]",       "measurement", "IDENTITY: rho_g is measured as exactly this "
                                              "correlation (D-0011)"),
        ("cov[y_m,y_p]",       "measurement", "IDENTITY: rho_y is measured as exactly this "
                                              "correlation (D-0011)"),
    ]
    out = []
    for label, kind, note in labels:
        v = np.array([r["rel_err"][label] for r in reps], dtype=float)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        out.append(Residual(label=label, predicted=float("nan"), measured=float("nan"),
                            rel_err=float(v.mean()), se=float(se), kind=kind, note=note))
    return out


def burn_in_calibration(reps: list[dict]) -> list[tuple[int, float, float]]:
    """V_A by generation, averaged across replicates.

    Reframed per the revised plan: this does NOT verify the writeup's "six to ten
    generations", which follows from eq:c-closed and is already checked symbolically. What
    it adds is that the mechanism follows the recursion, and a practical burn-in calibration
    at the N and M actually in use -- which everything downstream depends on.
    """
    by_t: dict[int, list[float]] = {}
    for r in reps:
        for row in r["trajectory"]:
            by_t.setdefault(row["t"], []).append(row["V_A"])
    out = []
    for t in sorted(by_t):
        v = np.array(by_t[t], dtype=float)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        out.append((t, float(v.mean()), float(se)))
    return out


def report(reps: list[dict], burn_in: int = BURN_IN) -> None:
    reg = reps[0]["regime"]
    print(f"{reg['name']}: N={reg['N']} M={reg['M']} AM_r={reg['AM_r']} "
          f"h2_0(nominal)={reg['h2_0']}")
    print(f"{len(reps)} replicates, {reps[0]['n_gen']} post-burn-in generations each "
          f"(burn-in {burn_in})")

    m = {k: float(np.mean([r["measured"][k] for r in reps if k in r["measured"]]))
         for k in reps[0]["measured"]}
    VA_0 = float(np.mean([r["VA_0"] for r in reps]))
    h2_0 = float(np.mean([r["h2_0"] for r in reps]))
    VA_0_real = float(np.mean([r["VA_0_realized"] for r in reps]))
    print("\n-- measured (every prediction is evaluated at these, per generation) --")
    print(f"     base V_A^(0) {VA_0:.6f} = sum_k beta_k^2, the derivation's diagonal")
    print(f"     (realized base Var[g] is {VA_0_real:.6f}, "
          f"{100 * (VA_0_real / VA_0 - 1):+.4f}% -- base LD, a zero-mean nuisance)")
    print(f"     base h^2_0   {h2_0:.6f} (covariance-free)")
    for k in ("V_A", "V_Y", "V_E", "h2", "rho_y", "rho_g", "var_eps_o", "M_e"):
        if k in m:
            print(f"  {k:>11} {m[k]:.6f}")
    if "corr_g_e" in m:
        print(f"  {'corr_g_e':>11} {m['corr_g_e']:+.6f}   (GE-indep says 0; this is the "
              "realized sampling value)")
    inflation = m["V_A"] / VA_0
    print(f"\n  V_A inflated {VA_0:.4f} -> {m['V_A']:.4f}  ({inflation:.3f}x), so the "
          "equilibrium tier is genuinely exercised")

    # The size of the thing eq:VA-eq drops, at the M actually used.
    corr = theory.VA_eq_relative_error(m["rho_g"], m["M_e"])
    print(f"  eq:VA-eq's dropped term at M_e={m['M_e']:.1f}: {-100 * corr:+.3f}% "
          f"(the SIGNED prediction; simulation should sit BELOW eq:VA-eq by this much)")

    print("\n-- residuals: measured vs predicted, per generation then pooled --")
    print(f"  {'equation':>20}  {'rel err':>9}  {'SE':>8}  {'n SE':>5}  kind")
    for r in summarize(reps):
        if r.kind == "measurement":
            # A value, not a residual -- see the note in `per_replicate`.
            print(f"  {r.label:>20}  {r.rel_err:+8.6f}   {'':>7}  {'':>5}  "
                  f"measurement (identity, not a test)")
            continue
        flag = "  <-- mismatch" if r.n_se > 2 else ""
        print(f"  {r.label:>20}  {100 * r.rel_err:+8.4f}%  {100 * r.se:7.4f}%  "
              f"{r.n_se:5.2f}  {r.kind}{flag}")
    print("\n  notes:")
    for r in summarize(reps):
        print(f"    {r.label:>20}  {r.note}")

    print("\n-- burn-in calibration: V_A by generation (NOT a test of '6 to 10') --")
    cal = burn_in_calibration(reps)
    target = cal[-1][1]
    reached = next((t for t, v, _ in cal if v >= 0.99 * target), None)
    for t, v, se in cal:
        if t <= 14 or t % 4 == 0:
            bar = "#" * int(40 * v / max(c[1] for c in cal))
            print(f"    t={t:2d}  {v:.5f} +- {se:.5f}  {bar}")
    print(f"  V_A reaches 99% of its plateau by generation {reached}; "
          f"burn-in of {burn_in} used")


def main() -> None:
    regime = sys.argv[1] if len(sys.argv) > 1 else "equilibrium-smallM"
    avail = gen.available(regime)
    if not avail:
        raise SystemExit(f"no saved replicates for {regime!r}; run "
                         f"`python generate_populations.py {regime} --reps 20` first")
    reps = [per_replicate(gen.load(regime, r)) for r in avail]
    report(reps)


if __name__ == "__main__":
    main()
