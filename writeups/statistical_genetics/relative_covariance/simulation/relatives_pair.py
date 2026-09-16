"""`tab:g-only-cov` at equilibrium: the mate-pair and parent--offspring columns.

    python relatives_pair.py [regime]

The eight entries of `tab:g-only-cov`, measured at equilibrium. Two of them are identities
rather than tests, and the module labels them as such rather than printing eight
equally-meaningful checks (D-0011):

| entry | value | status |
|---|---|---|
| `Cov[g_m, g_p]` | `rho_g V_A` | **measurement** -- `rho_g` is measured as exactly this correlation |
| `Cov[y_m, y_p]` | `rho_y V_Y` | **measurement** -- likewise for `rho_y` |
| `Cov[g_m, y_p]` = `Cov[y_m, g_p]` | `rho_y V_A` | test of `eq:leg-rule` |
| `Cov[g_m, g_o]` | `V_A (1+rho_g)/2` | test |
| `Cov[g_m, y_o]` | `V_A (1+rho_g)/2` | test |
| `Cov[y_m, g_o]` | `V_A (1+rho_y)/2` | test |
| `Cov[y_m, y_o]` | `V_A (1+rho_y)/2` | test |

The parent--offspring column is the best-powered thing in the equilibrium tier, because
every offspring is a data point rather than every pair.

## The asymmetry, which is the point of the table

`Cov[g_m, y_o]` and `Cov[y_m, g_o]` are **not equal**, and the table's caption says why: a
parent's environment predicts the genetic value the *other* parent transmits, but not the
offspring's own environment. Since `rho_g = rho_y h^2 <= rho_y`, the prediction is

    Cov[y_m, g_o] = V_A (1+rho_y)/2   >   V_A (1+rho_g)/2 = Cov[g_m, y_o]

so the parent-phenotype-against-offspring-genotype direction is the **larger** of the two.
That ordering is a directed claim the simulation can get wrong in a way no tolerance check
on either entry alone would catch, so it is tested as a signed difference with its own
standard error, not just as two independent residuals.

Everything is aggregated over pairs. No per-individual value is displayed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import generate_populations as gen
import theory
from pair_structure import _mate_index


@dataclass
class PairCovariances:
    """One replicate's measured and predicted entries of tab:g-only-cov."""

    n_mate_pairs: int
    n_po_pairs: int
    rho_y: float
    rho_g: float
    V_A: float
    V_Y: float
    observed: dict[str, float]
    predicted: dict[str, float]
    asym_obs: float          # Cov[y_m, g_o] - Cov[g_m, y_o], observed
    asym_pred: float         # V_A (rho_y - rho_g) / 2, predicted
    asym_se: float


#: (label, parent level, other level, kind). For the mate pair the two members are
#: exchangeable; for parent--offspring the FIRST level is the parent's.
MATE_ENTRIES = (("Cov[g_m,g_p]", "g", "g", "measurement"),
                ("Cov[y_m,y_p]", "y", "y", "measurement"),
                ("Cov[g_m,y_p]", "g", "y", "test"))
PO_ENTRIES = (("Cov[g_m,g_o]", "g", "g", "test"),
              ("Cov[g_m,y_o]", "g", "y", "test"),
              ("Cov[y_m,g_o]", "y", "g", "test"),
              ("Cov[y_m,y_o]", "y", "y", "test"))


def _parent_index(pop) -> tuple[np.ndarray, np.ndarray]:
    """Offspring indices and one parent index each, for the current generation.

    `relations['parents']` is an `(N, 2)` array of indices into the PARENT generation. Only
    one parent per offspring is taken, so that each pair contributes once and the estimate
    is not inflated by using both parents of the same child as if independent.
    """
    par = np.asarray(pop.relations["parents"], dtype=np.int64).reshape(pop.N, -1)
    ok = np.flatnonzero((par >= 0).all(axis=1))
    if ok.size == 0:
        raise ValueError("no parent links recorded on this generation")
    return ok, par[ok, 0]


def analyze(bundle: dict, trait: str = "y", component: str = "A") -> PairCovariances:
    pop, std = bundle["population"], bundle["standardization"]
    parent = pop.past[1]

    lev_child = {"g": np.asarray(pop.traits[trait].y_[component], dtype=float),
                 "y": np.asarray(pop.traits[trait].y, dtype=float)}
    lev_par = {"g": np.asarray(parent.traits[trait].y_[component], dtype=float),
               "y": np.asarray(parent.traits[trait].y, dtype=float)}

    # Measured parameters, from the generation whose mating is recorded on the parent.
    i, j = _mate_index(parent)
    rho_y = float(np.corrcoef(lev_par["y"][i], lev_par["y"][j])[0, 1])
    rho_g = float(np.corrcoef(lev_par["g"][i], lev_par["g"][j])[0, 1])
    # V_A and V_Y describe the OFFSPRING generation, which is where the covariances with
    # the offspring are read; at equilibrium the two generations agree.
    V_A = float(lev_child["g"].var(ddof=1))
    V_Y = float(lev_child["y"].var(ddof=1))

    obs: dict[str, float] = {}
    pred: dict[str, float] = {}

    for label, la, lb, _kind in MATE_ENTRIES:
        # Symmetrized over the two labellings, since mates are exchangeable.
        obs[label] = float(0.5 * (np.cov(lev_par[la][i], lev_par[lb][j])[0, 1]
                                  + np.cov(lev_par[la][j], lev_par[lb][i])[0, 1]))
        pred[label] = theory.pair_cov("mates", la, lb, V_A=float(lev_par["g"].var(ddof=1)),
                                      V_Y=float(lev_par["y"].var(ddof=1)),
                                      rho_g=rho_g, rho_y=rho_y)

    kids, par_of = _parent_index(pop)
    for label, la, lb, _kind in PO_ENTRIES:
        obs[label] = float(np.cov(lev_par[la][par_of], lev_child[lb][kids])[0, 1])
        pred[label] = theory.pair_cov("parent-offspring", la, lb,
                                      V_A=V_A, V_Y=V_Y, rho_g=rho_g, rho_y=rho_y)

    # The asymmetry, as a signed difference with its own SE -- a directed claim, so it is
    # tested directly rather than inferred from two separate residuals.
    d = (lev_par["y"][par_of] * lev_child["g"][kids]
         - lev_par["g"][par_of] * lev_child["y"][kids])
    # Both terms are products of mean-zero-ish variables; subtracting the empirical means
    # keeps this a difference of covariances rather than of raw cross-moments.
    d = d - (lev_par["y"][par_of].mean() * lev_child["g"][kids].mean()
             - lev_par["g"][par_of].mean() * lev_child["y"][kids].mean())
    asym_obs = float(obs["Cov[y_m,g_o]"] - obs["Cov[g_m,y_o]"])
    asym_pred = float(0.5 * V_A * (rho_y - rho_g))
    asym_se = float(d.std(ddof=1) / np.sqrt(d.size))

    return PairCovariances(
        n_mate_pairs=int(i.size), n_po_pairs=int(kids.size),
        rho_y=rho_y, rho_g=rho_g, V_A=V_A, V_Y=V_Y,
        observed=obs, predicted=pred,
        asym_obs=asym_obs, asym_pred=asym_pred, asym_se=asym_se,
    )


def report(results: list[PairCovariances]) -> None:
    def ms(vals):
        v = np.array(vals, dtype=float)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        return float(v.mean()), float(se)

    r0 = results[0]
    print(f"{len(results)} replicate(s); {r0.n_mate_pairs} mate pairs, "
          f"{r0.n_po_pairs} parent-offspring pairs per replicate")
    ry, _ = ms([r.rho_y for r in results])
    rg, _ = ms([r.rho_g for r in results])
    print(f"measured rho_y {ry:.6f}, rho_g {rg:.6f}")

    print(f"\n  {'entry':>14}  {'observed':>10}  {'predicted':>10}  {'rel err':>9}  "
          f"{'n SE':>5}  kind")
    for label, _la, _lb, kind in MATE_ENTRIES + PO_ENTRIES:
        o, o_se = ms([r.observed[label] for r in results])
        p, _ = ms([r.predicted[label] for r in results])
        if kind == "measurement":
            print(f"  {label:>14}  {o:10.6f}  {p:10.6f}  {'—':>9}  {'—':>5}  "
                  f"measurement (identity)")
            continue
        rel = o / p - 1.0
        n_se = abs(o - p) / o_se if o_se > 0 else float("nan")
        flag = "  <-- mismatch" if n_se > 2 else ""
        print(f"  {label:>14}  {o:10.6f}  {p:10.6f}  {100 * rel:+8.3f}%  {n_se:5.2f}  "
              f"test{flag}")

    print("\n-- the asymmetry: Cov[y_m,g_o] - Cov[g_m,y_o], predicted V_A(rho_y-rho_g)/2 --")
    ao, ao_se = ms([r.asym_obs for r in results])
    ap, _ = ms([r.asym_pred for r in results])
    print(f"    observed  {ao:+.6f} +- {ao_se:.6f}")
    print(f"    predicted {ap:+.6f}")
    print(f"    ratio {ao / ap:.4f};  observed is {abs(ao) / ao_se:.1f} SE from zero, "
          f"and {abs(ao - ap) / ao_se:.2f} SE from its prediction")
    print("    The SIGN is the directed claim: a parent's environment predicts the genetic")
    print("    value the OTHER parent transmits, but not the offspring's own environment,")
    print("    so the parent-phenotype-to-offspring-genotype direction must be the larger.")


def main() -> None:
    regime = sys.argv[1] if len(sys.argv) > 1 else "equilibrium-smallM"
    avail = gen.available(regime)
    if not avail:
        raise SystemExit(f"no saved replicates for {regime!r}")
    print(f"{regime}: tab:g-only-cov at equilibrium")
    report([analyze(gen.load(regime, r)) for r in avail])


if __name__ == "__main__":
    main()
