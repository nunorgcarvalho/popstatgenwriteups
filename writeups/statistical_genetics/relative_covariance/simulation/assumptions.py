"""The finite-N assumption diagnostics: Constant-freq and No-inbreeding.

    python assumptions.py [regime]

These are not tests of the writeup's algebra. They are checks on whether the writeup's
**assumptions** hold well enough at the `N` and generation counts actually simulated, which
is something no symbolic check can address -- pathMgr's path model simply assumes them. The
settled treatment (D-0009) is to **document once and not correct for**: if the assumption is
reasonable at the scale in use, say so with numbers and move on.

Two assumptions are at risk once a population is run forward for tens of generations at
finite `N`:

**`Constant-freq`** -- the writeup standardizes by a fixed `sqrt(P p (1-p))` from the base
population and holds allele frequencies still. Real frequencies drift. The diagnostic reports
how far they moved and, more to the point, how much the genotype standard deviations moved,
since that is the part that reaches `V_A`.

**`No-inbreeding`** -- assumed outright, but after ten generations of mating in a closed
population of size `N`, mate pairs necessarily share ancestors. Nothing symbolic can say
whether that is negligible at the `N` in use, and it is the assumption most likely to bite
quietly, because its effect on `V_A` has the same sign as a real signal.

## Measuring mate relatedness without confounding it with assortment

The obvious diagnostic -- average genomic relatedness between mates -- does not work
naively, because under assortative mating mates are genetically correlated *by assortment*,
with no shared ancestry at all. That is the whole point of the cross-mate structure checked
in `pair_structure.py`, and it has to be removed before any excess can be read as ancestry.

It is worth being explicit about a tempting argument that **does not work**, because it was
the first thing tried here. Assortment's contribution to a cross-mate genotype covariance is
`mu beta_k beta_l` (`eq:leg-rule`), which carries the sign of the effect sizes, so one might
hope that uniform weights over loci make it average away while sign-independent ancestry
sharing survives. That is wrong. The standard relatedness estimator `x_m . x_p / M` contains
only the **diagonal** terms `k = l`, where the assortment contribution is `mu beta_k^2` --
always positive, never cancelling. The sign argument applies to the off-diagonal terms,
which that estimator never touches. Uncorrected, the estimator is therefore dominated by
assortment, and at `M = 50` the assortment term alone accounts for most of what it reports.

What is done instead: the assortment contribution to the uniform-weighted estimator is
exactly `mu * sum_k beta_k^2 / M`, computable from measured quantities, so it is subtracted
analytically. The diagnostic reports the raw mate-pair estimate, a random non-mate baseline,
that predicted assortment term, and only then the **residual excess** as the ancestry-driven
part. The random-pair baseline additionally absorbs any drift away from base frequencies
shared by everyone, which would otherwise read as relatedness for every pair alike.

Everything reported is a mean over pairs and over loci. No per-individual or per-pair value
is displayed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import generate_populations as gen
from pair_structure import _mate_index


@dataclass
class Diagnostics:
    """One replicate's assumption diagnostics."""

    t: int
    N: int
    M: int
    generations: int
    # -- Constant-freq --
    mean_abs_dp: float
    max_abs_dp: float
    mean_rel_dsd: float          # relative change in genotype SD; the part reaching V_A
    n_lost: int
    # -- No-inbreeding --
    rel_mates: float             # uniform-weighted relatedness among mate pairs (RAW)
    rel_random: float            # same statistic among random non-mate pairs
    rel_assort_pred: float       # mu * sum(beta^2) / M, assortment's share of the raw value
    rel_excess: float            # raw - random - assortment = the ancestry-driven part
    rel_excess_se: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _uniform_relatedness(X: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Per-pair relatedness with uniform weights over loci: the usual `x_i . x_j / M`.

    `X` is standardized in pinned base units. This is the RAW statistic -- under assortative
    mating it contains an assortment term as well as any ancestry sharing, and the caller
    must subtract the former (see the module docstring for why uniform weighting does NOT
    remove it on its own).
    """
    return np.einsum("nk,nk->n", X[i], X[j]) / X.shape[1]


def analyze(bundle: dict, trait: str = "y", rng_seed: int = 0) -> Diagnostics:
    pop, std = bundle["population"], bundle["standardization"]
    drift = std.allele_freq_drift(pop)

    # Mate pairs are recorded on the parent of the generation they produced.
    parent = pop.past[1] if pop.t > 0 else pop
    i, j = _mate_index(parent)
    X = std.standardize(np.asarray(parent.G, dtype=float))
    # Center on the generation so that drift shared by everyone is not read as relatedness.
    X = X - X.mean(0, keepdims=True)

    rel_m = _uniform_relatedness(X, i, j)

    # Assortment's own contribution to the diagonal terms this estimator sums:
    # Cov[x_mk, x_pk] = mu beta_k^2 by eq:leg-rule, so the estimator picks up
    # mu * sum(beta^2) / M whether or not the pair shares any ancestor.
    tr = parent.traits[trait]
    beta = std.to_standardized_effects(
        np.asarray(tr.effects["A"].effects_per_allele, dtype=float))
    y = np.asarray(tr.y, dtype=float)
    rho_y = float(np.corrcoef(y[i], y[j])[0, 1])
    mu = rho_y / float(y.var(ddof=1))
    assort_pred = float(mu * np.sum(beta ** 2) / std.M)

    # Random non-mate pairs, matched in number, as the baseline.
    rng = np.random.default_rng(rng_seed)
    a = rng.integers(0, parent.N, size=i.size * 2)
    b = rng.integers(0, parent.N, size=i.size * 2)
    keep = a != b
    a, b = a[keep][:i.size], b[keep][:i.size]
    rel_r = _uniform_relatedness(X, a, b)

    excess = float(rel_m.mean() - rel_r.mean() - assort_pred)
    excess_se = float(np.sqrt(rel_m.var(ddof=1) / rel_m.size + rel_r.var(ddof=1) / rel_r.size))

    return Diagnostics(
        t=int(pop.t), N=int(parent.N), M=int(std.M),
        generations=int(bundle["regime"]["generations"]),
        mean_abs_dp=float(drift.get("mean_abs_dp", np.nan)),
        max_abs_dp=float(drift.get("max_abs_dp", np.nan)),
        mean_rel_dsd=float(drift.get("mean_rel_dsd", np.nan)),
        n_lost=int(drift.get("n_lost", 0)),
        rel_mates=float(rel_m.mean()), rel_random=float(rel_r.mean()),
        rel_assort_pred=assort_pred, rel_excess=excess, rel_excess_se=excess_se,
    )


def report(results: list[Diagnostics]) -> None:
    def ms(key):
        v = np.array([getattr(r, key) for r in results], dtype=float)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        return float(v.mean()), float(se)

    r0 = results[0]
    print(f"N={r0.N} M={r0.M}, {r0.generations} generations simulated, "
          f"{len(results)} replicate(s)")
    print("\n-- Constant-freq: how far did allele frequencies move? --")
    for key, label in (("mean_abs_dp", "mean |dp|"), ("max_abs_dp", "max |dp|"),
                       ("mean_rel_dsd", "mean rel change in genotype SD"),
                       ("n_lost", "variants lost or fixed")):
        m, se = ms(key)
        print(f"    {label:<32} {m:.6f} +- {se:.6f}")
    print("    The SD change is the part that reaches V_A; the pinned standardization is")
    print("    what keeps V_A^(0) and V_A^(eq) in the same units despite it.")

    print("\n-- No-inbreeding: relatedness among mates, above a random-pair baseline --")
    m, m_se = ms("rel_mates")
    r, r_se = ms("rel_random")
    a, _ = ms("rel_assort_pred")
    x, x_se = ms("rel_excess")
    within, _ = ms("rel_excess_se")
    se = x_se if len(results) > 1 else within
    print(f"    mate pairs, RAW                        {m:+.6f} +- {m_se:.6f}")
    print(f"    random non-mate pairs (baseline)       {r:+.6f} +- {r_se:.6f}")
    print(f"    assortment's share, mu sum(beta^2)/M   {a:+.6f}   "
          f"({100 * a / m:.0f}% of the raw value)")
    print(f"    residual excess = ancestry-driven      {x:+.6f} +- {se:.6f}   "
          f"({abs(x) / se:.2f} SE)")
    print("    Uniform weights do NOT remove assortment: the estimator sums only the")
    print("    diagonal terms, where its contribution is mu beta_k^2 and always positive.")
    print("    Per D-0009 this documents the assumption, it does not correct for it.")


def main() -> None:
    regime = sys.argv[1] if len(sys.argv) > 1 else "equilibrium-smallM"
    avail = gen.available(regime)
    if not avail:
        raise SystemExit(f"no saved replicates for {regime!r}")
    print(f"{regime}:")
    report([analyze(gen.load(regime, r), rng_seed=r) for r in avail])


if __name__ == "__main__":
    main()
