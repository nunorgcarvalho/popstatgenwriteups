"""The simulation leg of `tab:relationship-classes`: Monte-Carlo per relationship class.

    python classes_sim.py [regime]

The third leg. `three_way.py` established that the closed-form formula and the pathMgr
oracle are **identically equal** as expressions for all eleven rows, so the
formula-vs-oracle question is settled and any disagreement found here localizes to the
simulator or a violated assumption rather than to the writeup's algebra.

## How pairs are obtained, and why not by classification

Pairs come from `enumerate_class`, which generates each class **constructively** from local
pedigree structure in time proportional to the number of instances. Classifying all `N^2`
pairs is gone by design (D-0007): at `N = 4000` over four generations it is ~10^9 pairs, and
the classes that matter most are the rarest.

## Measurement decisions that change the answer

**Centering is per generation.** A class pair spans generations (parent--offspring,
grandparent--grandchild, first cousins), and each member is centered on its OWN generation's
mean rather than on a pooled mean. At equilibrium the generation means agree, so this makes
little difference to the estimate -- but it makes the estimator correct during any residual
drift instead of absorbing a mean shift into the covariance.

**Predictions are evaluated at measured `rho_y`, `rho_g`, `V_A`** (D-0009), and `V_A` is
measured on the generation the pairs are drawn from, in pinned base units.

**`self` is a measurement at every level, on all three legs, and never a test** -- `Var[g]` is
`V_A` and `Var[y]` is `V_Y` once those are measured, so there is nothing to check. It is
excluded here rather than reported as a twelfth confirmation.

**Two classes are checked first, deliberately.** popstatgensim hit two bugs in exactly these
paths, and both produced wrong numbers rather than errors: candidate generation that walked
only upward made *parents of a married couple* structurally unreachable, and a missing
containment rule scored HALF sibs as FULL sibs. The second is the dangerous one, because half
sibs coming out equal to full sibs would masquerade as one of the two *coincidences* the
table's caption points at. So the report asserts `full > half > step` explicitly, as a
contrast the mechanism predicts rather than an inequality noise can flip (D-0015).

**Per-round `rho_y`.** D-0014 requires mate-chain predictions to be evaluated at the product
of measured per-round `rho_y` rather than a power of a pooled one. These regimes use
`q2 >= 0.6`, where the measured round-2 attenuation is ~0 (`r2/r1 ~ 1.000`), so the two
targets coincide and the comparison does not depend on the choice. Both are reported anyway,
because the point of keeping both is that observed-vs-product isolates the mechanism while
product-vs-square isolates per-round calibration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import generate_populations as gen
import theory

#: popstatgensim's class spec -> the writeup's own row name in tab:relationship-classes.
#: `self` is deliberately absent: it is a measurement at every level, never a test.
SPEC_TO_WRITEUP = {
    "parent_offspring":       "parent-offspring",
    "grandparent":            "grandparent-grandchild",
    "full_sibs":              "full siblings",
    "avuncular":              "avuncular",
    "first_cousins":          "first cousins",
    "half_sibs":              "half siblings",
    "step_sibs":              "step siblings",
    "parent_parent":          "parent-parent",
    "married_couple_parents": "parents of a married couple",
}

#: The sibling ladder, which differs only in the mating distance N_mu = 1, 2, 3 within one
#: mating chain and must therefore come out monotonically weaker (see the module docstring).
SIB_LADDER = ("full siblings", "half siblings", "step siblings")


@dataclass
class ClassResult:
    name: str
    n_pairs: int
    cov_g_obs: float
    cov_g_se: float
    cov_y_obs: float
    cov_y_se: float
    cov_g_pred: float
    cov_y_pred: float


def _flatten(pop, trait: str = "y", component: str = "A"):
    """Trait vectors in `PedigreeGraph`'s flattened index space.

    `from_population` flattens `pop.past` with the OLDEST retained generation first, so the
    same ordering is reproduced here. Getting this backwards would silently pair the right
    individuals' indices with the wrong individuals' values -- a wrong number, not an error.
    """
    gens = list(reversed(pop.past))                      # oldest first, matching the graph
    g = np.concatenate([np.asarray(p.traits[trait].y_[component], float) for p in gens])
    y = np.concatenate([np.asarray(p.traits[trait].y, float) for p in gens])
    gen_id = np.concatenate([np.full(p.N, k) for k, p in enumerate(gens)])
    return g, y, gen_id


def _centered(v: np.ndarray, gen_id: np.ndarray) -> np.ndarray:
    """Center each individual on its own generation's mean (see the module docstring)."""
    out = v.astype(float).copy()
    for k in np.unique(gen_id):
        m = gen_id == k
        out[m] -= out[m].mean()
    return out


def _pair_cov(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean cross-product over pairs, with a NAIVE across-pairs SE.

    The inputs are already centered per generation, so the mean product is the covariance
    estimate. The SE returned here is **optimistic and must not be used as the headline**:
    it treats pairs as independent, and they are heavily dependent -- 6400 parent--offspring
    pairs come from at most a few thousand individuals, and every grandparent appears in
    many pairs. The honest uncertainty comes from INDEPENDENT REPLICATE POPULATIONS, which
    is what `report` uses whenever more than one replicate is available; this value is only
    a fallback for a single-replicate smoke run, and is labelled as such in the output.
    """
    prod = a * b
    return float(prod.mean()), float(prod.std(ddof=1) / np.sqrt(prod.size))


def analyze(bundle: dict, limit: int = 20000, trait: str = "y") -> list[ClassResult]:
    from popstatgensim.pedigree import PedigreeGraph, enumerate_class

    pop, std = bundle["population"], bundle["standardization"]
    graph = PedigreeGraph.from_population(pop)
    g, y, gen_id = _flatten(pop, trait)
    if g.size != graph.N:
        raise ValueError(f"flattened {g.size} individuals but the graph has {graph.N}")
    gc, yc = _centered(g, gen_id), _centered(y, gen_id)

    # Measured parameters, POOLED over the same generations the pairs are drawn from.
    # Measuring them on one generation while enumerate_class draws pairs from every retained
    # generation would compare a prediction built from one sample against a covariance
    # measured on another -- harmless in expectation at equilibrium, but it injects variance
    # and would bias anything measured before equilibrium. All retained generations here are
    # post-burn-in, so pooling is both consistent and quieter.
    from pair_structure import _mate_index
    rys, rgs, vas = [], [], []
    for p_gen in pop.past:
        vas.append(float(np.asarray(p_gen.traits[trait].y_["A"], float).var(ddof=1)))
        try:
            i, j = _mate_index(p_gen)
        except (ValueError, KeyError):
            continue                       # the youngest generation has not mated yet
        yp = np.asarray(p_gen.traits[trait].y, float)
        gp = np.asarray(p_gen.traits[trait].y_["A"], float)
        rys.append(float(np.corrcoef(yp[i], yp[j])[0, 1]))
        rgs.append(float(np.corrcoef(gp[i], gp[j])[0, 1]))
    rho_y, rho_g, V_A = float(np.mean(rys)), float(np.mean(rgs)), float(np.mean(vas))

    out = []
    for spec, name in SPEC_TO_WRITEUP.items():
        pairs = np.array(list(enumerate_class(graph, spec, limit=limit)), dtype=np.int64)
        if pairs.size == 0:
            continue
        A, B = pairs[:, 0], pairs[:, 1]
        cg, cg_se = _pair_cov(gc[A], gc[B])
        cy, cy_se = _pair_cov(yc[A], yc[B])
        out.append(ClassResult(
            name=name, n_pairs=len(pairs),
            cov_g_obs=cg, cov_g_se=cg_se, cov_y_obs=cy, cov_y_se=cy_se,
            cov_g_pred=theory.class_cov(name, ("g", "g"),
                                        V_A=V_A, rho_g=rho_g, rho_y=rho_y),
            cov_y_pred=theory.class_cov(name, ("y", "y"),
                                        V_A=V_A, rho_g=rho_g, rho_y=rho_y),
        ))
    return out, dict(rho_y=rho_y, rho_g=rho_g, V_A=V_A, graph_N=graph.N)


def report(per_rep: list[tuple[list[ClassResult], dict]]) -> None:
    params = {k: float(np.mean([p[1][k] for p in per_rep]))
              for k in ("rho_y", "rho_g", "V_A")}
    print(f"{len(per_rep)} replicate(s); measured rho_y {params['rho_y']:.6f}, "
          f"rho_g {params['rho_g']:.6f}, V_A {params['V_A']:.6f}")
    print("predictions are evaluated at those measured values, never the nominal AM_r")
    if len(per_rep) > 1:
        print(f"SEs are ACROSS the {len(per_rep)} independent replicate populations\n")
    else:
        print("WARNING: one replicate only, so the SEs below are the naive across-pairs\n"
              "ones, which are OPTIMISTIC -- pairs share individuals and are not\n"
              "independent. Treat this as a smoke run, not as a result.\n")

    names = [r.name for r in per_rep[0][0]]
    print(f"  {'class':>28} {'pairs':>7}  {'level':>5}  {'observed':>10}  {'predicted':>10}"
          f"  {'rel err':>9}  {'n SE':>5}")
    summary = {}
    for name in names:
        rs = [r for reps, _ in per_rep for r in reps if r.name == name]
        n_pairs = int(np.mean([r.n_pairs for r in rs]))
        for level in ("g", "y"):
            obs = np.array([getattr(r, f"cov_{level}_obs") for r in rs])
            pred = np.array([getattr(r, f"cov_{level}_pred") for r in rs])
            o, p = obs.mean(), pred.mean()
            # SE across replicates where there are several, else the within-replicate SE
            se = (obs.std(ddof=1) / np.sqrt(len(obs)) if len(obs) > 1
                  else float(np.mean([getattr(r, f"cov_{level}_se") for r in rs])))
            n_se = abs(o - p) / se if se > 0 else float("nan")
            flag = "  <-- mismatch" if n_se > 2 else ""
            print(f"  {name:>28} {n_pairs:>7}  {level:>5}  {o:10.6f}  {p:10.6f}  "
                  f"{100 * (o / p - 1):+8.3f}%  {n_se:5.2f}{flag}")
            if level == "g":
                summary[name] = (o, se)
        print()

    print("-- the sibling ladder: same path, one further mating crossed each time --")
    print("   full/half/step differ ONLY in N_mu = 1/2/3, so they must be monotonically")
    print("   weaker. Half sibs coming out EQUAL to full sibs is the signature of a missing")
    print("   containment rule, and would masquerade as one of the table's coincidences.")
    vals = [(n, *summary[n]) for n in SIB_LADDER if n in summary]
    for n, v, se in vals:
        print(f"     {n:>16}  {v:.6f} +- {se:.6f}")
    if len(vals) < 2:
        print("   only one rung present -- the ladder needs non-monogamy (half sibs require "
              "a\n   mating chain of N_mu = 2), so there is nothing to order here. NOT a "
              "passing test.")
        print("\n-- the two coincidences the table's caption points at --")
        print("   these arrive from DIFFERENT rows of eq:chain-weight, which is why they are")
        print("   evidence the general machinery is right rather than tuned")
        for a, b in (("full siblings", "parent-offspring"),
                     ("avuncular", "grandparent-grandchild")):
            if a in summary and b in summary:
                (va, sa), (vb, sb) = summary[a], summary[b]
                print(f"     {a} vs {b}: {va:.6f} vs {vb:.6f}  "
                      f"({abs(va - vb) / np.hypot(sa, sb):.2f} SE apart)")
        return
    ok = all(vals[k][1] > vals[k + 1][1] for k in range(len(vals) - 1))
    gaps = [f"{vals[k][0]} - {vals[k+1][0]} = {vals[k][1] - vals[k+1][1]:+.6f} "
            f"({abs(vals[k][1] - vals[k+1][1]) / np.hypot(vals[k][2], vals[k+1][2]):.1f} SE)"
            for k in range(len(vals) - 1)]
    print(f"   monotone: {'YES' if ok else 'NO -- investigate the containment rule'}")
    for gp in gaps:
        print(f"     {gp}")
    if not ok:
        # The ladder is a hard claim about the mechanism, not a soft diagnostic: full, half
        # and step siblings are the same path with one further mating crossed, so a
        # non-monotone result means the containment rule is off and half sibs are being
        # scored as full sibs. Failing loudly matters because that specific failure LOOKS
        # like one of the writeup's own coincidences.
        raise SystemExit("sibling ladder is not monotone -- see the containment rule")

    print("\n-- the two coincidences the table's caption points at --")
    print("   these arrive from DIFFERENT rows of eq:chain-weight, which is why they are")
    print("   evidence the general machinery is right rather than tuned")
    for a, b in (("full siblings", "parent-offspring"),
                 ("avuncular", "grandparent-grandchild")):
        if a in summary and b in summary:
            (va, sa), (vb, sb) = summary[a], summary[b]
            print(f"     {a} vs {b}: {va:.6f} vs {vb:.6f}  "
                  f"({abs(va - vb) / np.hypot(sa, sb):.2f} SE apart)")


def main() -> None:
    regime = sys.argv[1] if len(sys.argv) > 1 else "classes-multimate"
    avail = gen.available(regime)
    if not avail:
        raise SystemExit(f"no saved replicates for {regime!r}; run "
                         f"`python generate_populations.py {regime} --reps 8` first")
    print(f"{regime}: tab:relationship-classes, simulation leg\n")
    report([analyze(gen.load(regime, r)) for r in avail])


if __name__ == "__main__":
    main()
