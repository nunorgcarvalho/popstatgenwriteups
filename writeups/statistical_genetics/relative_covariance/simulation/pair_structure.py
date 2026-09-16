"""The base-generation cross-mate covariance structure: tab:pair-2v-between.

    python pair_structure.py                 # run the check on the saved pair-structure regime

This is the **first** check of the verification, and the one no symbolic check can reach.
`eq:leg-rule` says that for any variable `a` of one mate and `b` of the other,

    Cov[a_m, b_p] = Cov[a_m, y_m] * mu * Cov[y_p, b_p],       mu = rho_y / V_Y

so the entire cross-mate block is the **outer product of one vector with itself**, scaled by
`mu`. Across the base generation's loci that means

    Cov[x_mk, x_pl] = beta_k beta_l rho_y / V_Y       for EVERY pair (k, l), k != l included
    Cov[e_m, g_p]   = V_A V_E rho_y / V_Y
    Cov[x_mk, x_ml] = 0                               WITHIN an individual, base generation only

a rank-1 structural claim about the whole `M x M` cross-mate matrix, and the part of the
writeup most exposed to how mating is actually implemented. The simulator pairs mates by
sorting on rank with noise -- a rank-matching copula, not a bivariate normal. Both can hit a
target `rho_y` exactly while inducing different per-locus structure, so **reproducing `rho_y`
is necessary but not sufficient**. Checking `rho_g` instead would not help: under the
measured-parameter rule `Cov[g_m, g_p] = rho_g V_A` is an identity, not a test (D-0011).

The base generation is the right place for it. Its loci are still in linkage equilibrium, so
the predicted within-individual covariance is exactly zero and any cross-mate covariance is
attributable to the pairing alone. It also needs no burn-in, which makes it the cheapest
check in the task.

## Why the obvious statistics do not work, and what is used instead

A single entry of the observed cross-mate matrix has sampling SE about `1/sqrt(n_pairs)`,
while the entry it is being compared against is only about `mu V_A / M`. At any feasible `N`
the entrywise comparison is therefore noise-dominated -- at `M=120`, `n_pairs=2000` the noise
is ten times the signal -- so an entrywise `R^2`, or a relative Frobenius residual, comes out
near its no-information value even when the structure is exactly right. Reporting those as if
they were near 0 and 1 would be a false negative dressed as a measurement.

What does work is the **matched filter**: project the observed matrix onto the predicted
rank-1 direction and ask whether the coefficient is 1. It combines all `M^2` entries
coherently, so its precision depends only on the number of mate pairs and **not on `M`** --
`M=30` and `M=480` give the same precision at the same `N`. The estimator is computed as a
per-mate-pair contribution so its standard error comes from scatter across pairs rather than
from an assumed noise model.

Everything reported is aggregate over loci and over pairs -- a projection coefficient, a
regression slope, matrix norms. No per-individual value or identifier is computed or
displayed, per the standing rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import generate_populations as gen


@dataclass
class PairStructure:
    """Aggregate summary of one replicate's base-generation cross-mate structure."""

    n_pairs: int
    N: int
    M: int
    # -- measured parameters; every prediction below is evaluated at these --
    rho_y: float
    rho_g: float
    h2: float
    V_A: float
    V_Y: float
    V_E: float
    mu: float
    V_Y_empirical: float          # Var[y] as measured; differs from V_A+V_E by 2Cov[g,e]
    corr_g_e: float               # GE-indep says 0; reported, never fed into a prediction
    # -- the rank-1 claim: matched-filter coefficient, which should be 1 --
    filt: float
    filt_se: float
    # -- secondary, and interpretable only against their noise floors --
    rel_frob_resid: float
    rel_frob_null: float          # what rel_frob_resid would be if the structure were exact
    rank1_share: float
    # -- the named entries that are genuine tests --
    cov_e_g_obs: float
    cov_e_g_pred: float
    cov_g_y_obs: float
    cov_g_y_pred: float
    # -- the within-individual control, which must sit at its sampling floor --
    within_offdiag_rms: float
    within_offdiag_floor: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _mate_index(pop) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays for the two sides of each mate pair.

    Handles both shapes `relations['spouses']` has taken: a `MateGraph`, whose documented
    `edges` is a `(U, 2)` array of `(maternal_idx, paternal_idx)` and which can represent an
    individual's second mate, and the legacy length-`N` spouse array, which structurally
    could not. Reading `edges` rather than calling `to_spouse_array()` keeps every pair,
    including the ones a multi-mate individual sits on -- the back-compatible array keeps
    only each individual's first mate, which would silently drop edges in any non-monogamous
    regime.
    """
    # Preference order matters, and the first choice is the only lossless one.
    # `relations['mate_edges']` is a (U, 2) int32 array of (maternal, paternal) pairs and
    # keeps EVERY edge; `relations['spouses']` is retained but is lossy under non-monogamy,
    # holding only each individual's first mate because a length-N array structurally
    # cannot hold a second (D-0013). Reading the lossy one silently drops later-round mate
    # pairs -- which is a wrong number rather than an error, and is exactly the bug that
    # biased popstatgensim's own measure_mate_correlations toward round 1.
    edges = pop.relations.get("mate_edges")
    if edges is None:
        # A MateGraph exposes the same thing as `.edges`; the legacy array is the last resort.
        edges = getattr(pop.relations.get("spouses"), "edges", None)
    if edges is not None:
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        if edges.size == 0:
            raise ValueError("no mate pairs recorded on this generation")
        return edges[:, 0], edges[:, 1]
    spouse = np.asarray(pop.relations["spouses"], dtype=np.int32)
    i = np.flatnonzero((spouse >= 0) & (np.arange(pop.N) < spouse))
    if i.size == 0:
        raise ValueError("no mate pairs recorded on this generation")
    return i, spouse[i]


def analyze(bundle: dict, trait: str = "y", component: str = "A") -> PairStructure:
    """Run the rank-1 check on one saved replicate.

    `bundle` is what `generate_populations.load` returns. The measurement happens on the
    BASE generation, which is `pop.past[1]` after one generation has been simulated: the
    mating that produced generation `t` is recorded on the parent, not on `t` itself.
    """
    pop, std = bundle["population"], bundle["standardization"]
    base = pop.past[1] if pop.t > 0 else pop
    if base.t != std.t_base:
        raise ValueError(f"standardization pinned at t={std.t_base} but measuring t={base.t}")

    i, j = _mate_index(base)
    tr = base.traits[trait]
    y, g = np.asarray(tr.y, dtype=float), np.asarray(tr.y_[component], dtype=float)
    e = y - g                                     # the environment, as the model defines it

    # Genotypes in PINNED base units, and effects to match (D-0009 / Constant-freq).
    X = std.standardize(np.asarray(base.G, dtype=float))
    beta = std.to_standardized_effects(
        np.asarray(tr.effects[component].effects_per_allele, dtype=float))

    # Measured parameters. Nothing below uses the nominal AM_r: the pairing is a
    # rank-matching copula, so realized rho_y differs from it by a few percent.
    rho_y = float(np.corrcoef(y[i], y[j])[0, 1])
    rho_g = float(np.corrcoef(g[i], g[j])[0, 1])
    V_A, V_E = float(g.var(ddof=1)), float(e.var(ddof=1))
    # V_Y is taken as V_A + V_E, NOT as the empirical Var[y]. The writeup's V_Y has no
    # covariance term because GE-indep sets Cov[g, e] to zero, whereas an empirical Var[y]
    # absorbs the realized sampling covariance -- and feeding that back in would put a
    # g-e covariance term inside the very prediction whose structure is under test. The
    # realized correlation is reported separately as a diagnostic instead.
    V_Y = V_A + V_E
    V_Y_empirical = float(y.var(ddof=1))
    corr_g_e = float(np.corrcoef(g, e)[0, 1])
    mu = rho_y / V_Y

    # -- the rank-1 claim -------------------------------------------------------------------
    # Center on the base generation as a whole, not on the mated subset, so the two sides of
    # the pairing are expressed in common units.
    Xc = X - X.mean(0, keepdims=True)
    A, B = Xc[i], Xc[j]
    n = A.shape[0]

    P = mu * np.outer(beta, beta)                 # eq:leg-rule's prediction
    np.fill_diagonal(P, 0.0)                      # the diagonal is a different claim
    denom = float(np.sum(P * P))

    # Per-mate-pair contribution to the projection. Mates are exchangeable -- which member
    # is called m is arbitrary -- and since P is symmetric, a_m' P a_p already averages the
    # two labellings. Aggregating per pair is what gives an honest SE across pairs.
    contrib = np.einsum("nk,kl,nl->n", A, P, B)
    filt = float(contrib.mean() / denom)
    filt_se = float(contrib.std(ddof=1) / np.sqrt(n) / denom)

    # Secondary norms, reported only against their noise floors. A cross-mate entry has
    # sampling variance ~1/n_pairs, so E||obs - truth||_F^2 ~ M^2/n_pairs regardless of
    # whether the structure is right -- which is why these are not the headline.
    obs = A.T @ B / (n - 1)
    obs = 0.5 * (obs + obs.T)
    off = ~np.eye(len(beta), dtype=bool)
    noise_sq = len(beta) ** 2 / n
    rel_frob = float(np.linalg.norm(obs[off] - P[off]) / np.linalg.norm(obs[off]))
    rel_frob_null = float(np.sqrt(noise_sq / (denom + noise_sq)))
    sv = np.linalg.svd(obs, compute_uv=False)
    rank1_share = float(sv[0] ** 2 / np.sum(sv ** 2))

    # -- the two named entries the text highlights ------------------------------------------
    # Cov[e_m, g_p]: one mate's ENVIRONMENT against the other's genetic value. Nonzero, and
    # it is the break of Env-indep and No-IGE that the section flags at its start.
    cov_e_g_obs = float(0.5 * (np.cov(e[i], g[j])[0, 1] + np.cov(e[j], g[i])[0, 1]))
    cov_e_g_pred = V_A * V_E * mu
    # Cov[g_m, y_p] is a genuine test, unlike Cov[g_m,g_p] and Cov[y_m,y_p], which are
    # identities once rho_g and rho_y are measured as exactly those correlations (D-0011).
    cov_g_y_obs = float(0.5 * (np.cov(g[i], y[j])[0, 1] + np.cov(g[j], y[i])[0, 1]))
    cov_g_y_pred = rho_y * V_A

    # -- the within-individual control ------------------------------------------------------
    # In the BASE generation the writeup claims Cov[x_mk, x_ml] is exactly zero for k != l:
    # the disequilibrium exists only BETWEEN mates. The comparison is against the sampling
    # floor 1/sqrt(N), not against zero.
    within = Xc.T @ Xc / (base.N - 1)
    within_rms = float(np.sqrt(np.mean(within[off] ** 2)))

    return PairStructure(
        n_pairs=int(n), N=int(base.N), M=int(len(beta)),
        rho_y=rho_y, rho_g=rho_g, h2=V_A / V_Y, V_A=V_A, V_Y=V_Y, V_E=V_E, mu=mu,
        V_Y_empirical=V_Y_empirical, corr_g_e=corr_g_e,
        filt=filt, filt_se=filt_se,
        rel_frob_resid=rel_frob, rel_frob_null=rel_frob_null, rank1_share=rank1_share,
        cov_e_g_obs=cov_e_g_obs, cov_e_g_pred=cov_e_g_pred,
        cov_g_y_obs=cov_g_y_obs, cov_g_y_pred=cov_g_y_pred,
        within_offdiag_rms=within_rms, within_offdiag_floor=float(1.0 / np.sqrt(base.N)),
    )


def report(results: list[PairStructure]) -> None:
    """Print the aggregate summary across replicates."""
    def ms(key):
        v = np.array([getattr(r, key) for r in results], dtype=float)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        return float(v.mean()), float(se)

    r0 = results[0]
    print(f"base generation: N={r0.N}, M={r0.M}, {r0.n_pairs} mate pairs, "
          f"{len(results)} replicate(s)")

    print("\n-- measured parameters (what every prediction is evaluated at) --")
    for key in ("rho_y", "rho_g", "h2", "V_A", "V_Y", "V_E"):
        m, se = ms(key)
        print(f"  {key:>6}  {m:.6f} +- {se:.6f}")
    vye, _ = ms("V_Y_empirical")
    vy, _ = ms("V_Y")
    cge, _ = ms("corr_g_e")
    print(f"  V_Y is V_A + V_E by GE-indep, NOT the empirical Var[y] = {vye:.6f} "
          f"({100 * (vye / vy - 1):+.3f}%)")
    print(f"  realized Corr[g, e] = {cge:+.6f}  (assumed 0; diagnostic only, never fed in)")
    rg, rg_se = ms("rho_g")
    ry, _ = ms("rho_y")
    h2, _ = ms("h2")
    print(f"\n  rho_g^(0) = rho_y * h2_0 is a MECHANISM claim, not an identity:")
    print(f"    rho_y * h2_0   {ry * h2:.6f}")
    print(f"    measured rho_g {rg:.6f} +- {rg_se:.6f}   "
          f"({100 * (rg / (ry * h2) - 1):+.3f}%)")

    print("\n-- eq:leg-rule's rank-1 claim: matched-filter coefficient (predicted 1) --")
    f, f_se_across = ms("filt")
    within_se, _ = ms("filt_se")
    se = f_se_across if len(results) > 1 else within_se
    print(f"    coefficient    {f:.4f} +- {se:.4f}      "
          f"({'across replicates' if len(results) > 1 else 'within replicate'})")
    print(f"    deviation from 1: {f - 1:+.4f}  =  {abs(f - 1) / se:.2f} SE")

    print("\n-- secondary norms, against their noise floors (NOT against 0 and 1) --")
    rf, _ = ms("rel_frob_resid")
    rfn, _ = ms("rel_frob_null")
    r1, _ = ms("rank1_share")
    print(f"    relative Frobenius residual  {rf:.4f}   "
          f"(floor if structure exact: {rfn:.4f})")
    print(f"    top-singular-value share     {r1:.4f}   "
          f"(entrywise noise dominates at this N; see module docstring)")

    print("\n-- named entries: TESTS, not identities --")
    for obs_k, pred_k, label in (("cov_g_y_obs", "cov_g_y_pred", "Cov[g_m, y_p]"),
                                 ("cov_e_g_obs", "cov_e_g_pred", "Cov[e_m, g_p]")):
        o, o_se = ms(obs_k)
        p, _ = ms(pred_k)
        dev = 100 * (o / p - 1)
        n_se = abs(o - p) / o_se if o_se == o_se and o_se > 0 else float("nan")
        print(f"    {label:>14}  observed {o:+.6f} +- {o_se:.6f}   predicted {p:+.6f}   "
              f"({dev:+.2f}%, {n_se:.2f} SE)")

    print("\n-- control: base-generation within-individual Cov[x_mk, x_ml] must be ~0 --")
    w, _ = ms("within_offdiag_rms")
    fl, _ = ms("within_offdiag_floor")
    print(f"    RMS over k != l  {w:.6f}   sampling floor 1/sqrt(N) = {fl:.6f}   "
          f"ratio {w / fl:.2f}")


def main() -> None:
    reps = gen.available("pair-structure")
    if not reps:
        raise SystemExit("no saved replicates; run "
                         "`python generate_populations.py pair-structure --reps 5` first")
    report([analyze(gen.load("pair-structure", r)) for r in reps])


if __name__ == "__main__":
    main()
