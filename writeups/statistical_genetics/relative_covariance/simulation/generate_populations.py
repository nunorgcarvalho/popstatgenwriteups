"""Generate stage: build populations per parameter regime and save them to scratch.

    python generate_populations.py --list
    python generate_populations.py pair-structure
    python generate_populations.py equilibrium-smallM --reps 10

This is a SCRIPT and not a notebook on purpose. It is the step that can take an hour, it
has no output worth reading inline, and a verify notebook that had to simulate before it
could plot would be a notebook nobody runs. Verification loads what this writes and never
simulates -- see README.md.

Each invocation runs one regime, for one or more independent replicates, and writes one
pickle per replicate under SCRATCH_ROOT. The pickle holds the Population AND the
BaseStandardization pinned at generation 0: the base allele frequencies are unrecoverable
once the population has been simulated forward, so a population saved without its
standardization cannot be measured in the writeup's units afterwards.

Regimes are deliberately small by default. Nothing here scales up or submits a slurm job
without being asked to.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from popstatgensim import Population, GeneticEffect, BaseStandardization
from popstatgensim.estimation import measure_am_quantities
# NB: these two live in `traits`, not `estimation`, despite being measurements.
from popstatgensim.traits import effective_n_variants, var_A_pinned

#: Heavy artifacts live on scratch (prunable, not backed up), never in the repo.
SCRATCH_ROOT = Path("/n/scratch/users/n/nur479/relative_covariance_sim")


@dataclass(frozen=True)
class Regime:
    """One parameter regime: what to simulate and how far.

    `generations` is the number of generations to simulate past the base. For the
    pair-structure regime it is 1 -- just far enough that the base generation has been
    paired, since the mating that produced generation t is recorded on the PARENT.
    """

    name: str
    purpose: str
    N: int
    M: int
    h2_0: float
    AM_r: float
    generations: int
    keep_past: int = 1
    notes: str = ""
    #: Survival array of mate counts (D-0005): `q[k]` is the fraction of individuals with AT
    #: LEAST k+1 mates, so `q=[1.0]` is monogamy and `q=[1.0, 0.6]` gives 60% of individuals
    #: a second mate. Non-monogamy is what makes half, step and step-step siblings exist at
    #: all -- they need a mating chain of N_mu = 2, 3, 4, which monogamy cannot produce.
    mate_q: tuple[float, ...] = (1.0,)


#: The regimes this task needs. Small M is not a shortcut in the M_e regimes -- it is a
#: requirement. For iid normal effects M_e is about M/3, so the eq:VA-eq correction is
#: ~rho_g/(2(1-rho_g)M_e); at M=1000 that is about -0.06% against a Monte-Carlo SE near
#: 0.9%, i.e. ~14x too small to detect, and a null result there would mean nothing. At
#: M=50 the correction reaches ~1%, which is measurable.
REGIMES: dict[str, Regime] = {
    "pair-structure": Regime(
        name="pair-structure",
        purpose="base-generation cross-mate covariance structure (tab:pair-2v-between)",
        N=4000, M=120, h2_0=0.5, AM_r=0.6, generations=1,
        notes="Needs NO burn-in: the claim is about the base generation, whose genotypes "
              "are still in linkage equilibrium. Cheapest check in the task, so it runs "
              "first. One generation is simulated only so that the base generation carries "
              "its spouse pairing.",
    ),
    "equilibrium-smallM": Regime(
        name="equilibrium-smallM",
        purpose="equilibrium tier with a DETECTABLE eq:VA-eq correction",
        N=4000, M=50, h2_0=0.5, AM_r=0.5, generations=32,
        notes="M=50 gives M_e ~ 17 and a correction near 1%, which clears the "
              "Monte-Carlo SE. This is the regime for the signed-residual test.",
    ),
    "classes-monogamous": Regime(
        name="classes-monogamous",
        purpose="the classes reachable under monogamy: parent-offspring, grandparent, "
                "full sibs, avuncular, first cousins, parent-parent, married-couple-parents",
        N=4000, M=200, h2_0=0.5, AM_r=0.5, generations=16, keep_past=4,
        notes="Monogamy (q=[1.0]) reaches every class whose mating chain has N_mu <= 1. "
              "Needs 4 generations retained so first cousins and married-couple-parents "
              "have the depth to exist.",
    ),
    "classes-multimate": Regime(
        name="classes-multimate",
        purpose="the sibling ladder beyond full sibs: half sibs (N_mu=2) and step sibs (3)",
        N=4000, M=200, h2_0=0.5, AM_r=0.5, generations=16, keep_past=4,
        mate_q=(1.0, 0.8),
        notes="q2 = 0.8 puts the round-2 attenuation at ~0 (r2/r1 ~ 1.000 for q2 >= 0.6), "
              "so the per-round and pooled targets of D-0014 coincide and the class "
              "comparison does not depend on which is used. Step-step sibs need three "
              "consecutive multi-mate individuals and stay out of reach at feasible N.",
    ),
    "equilibrium-midM": Regime(
        name="equilibrium-midM",
        purpose="the same tier at larger M, to show the correction shrinking like 1/M_e",
        N=4000, M=200, h2_0=0.5, AM_r=0.5, generations=32,
        notes="Second point for the 1/M_e scaling. The prediction is not just that the "
              "residual is small but that it shrinks in a specific way.",
    ),
}


def build(regime: Regime, seed: int,
          trajectory: bool = True) -> tuple[Population, BaseStandardization, list[dict]]:
    """Construct a base population, pin its standardization, then simulate forward.

    The standardization is pinned BEFORE any generation is simulated, which is what makes
    V_A^(0) and V_A^(eq) commensurable (the writeup's Constant-freq assumption).

    With `trajectory`, the measured quantities are recorded **one generation at a time as
    the population evolves**. That has to happen here rather than in a verify notebook: the
    default `keep_past_generations=1` means a saved population remembers only its immediate
    parent, so a trajectory cannot be reconstructed afterwards without re-simulating -- and
    verification never simulates. The table is a few floats per generation, so it is free to
    carry, and having per-generation values is what allows a prediction to be evaluated at
    each generation's own measured inputs and only then averaged. Averaging the inputs first
    and predicting once is a biased reduction, because the predictions are nonlinear in
    `rho_g` (Jensen): `E[V_A0/(1-rho_g)] > V_A0/(1-E[rho_g])`.
    """
    np.random.seed(seed)
    # keep_past_generations must be set at CONSTRUCTION: the relationship classes need
    # pedigree depth. First cousins span three generations and married-couple-parents span
    # two plus a mating, so with the default of 1 retained generation they are structurally
    # unreachable rather than merely rare -- PedigreeGraph can only see what is retained.
    pop = Population(N=regime.N, M=regime.M, seed=seed,
                     params={"keep_past_generations": regime.keep_past})
    pop.assign_sex()
    pop.add_trait(
        name="y",
        effects={"A": GeneticEffect(var_indep=regime.h2_0, M=regime.M,
                                    M_causal=regime.M, name="A")},
        var_Eps=1.0 - regime.h2_0,
    )
    std = BaseStandardization.from_population(pop, method="observed")

    # AM_type='phenotypic' and s=0 are settled (D-0009): 'genetic' is genetic homogamy,
    # which Primary-AM explicitly excludes, and selection is not part of the model.
    params = dict(AM_r=regime.AM_r, AM_trait="y", AM_type="phenotypic", s=0.0)
    if tuple(regime.mate_q) != (1.0,):
        # q2 >= 0.6 is chosen deliberately where possible: popstatgensim measured the
        # round-2 rho_y attenuation at r2/r1 = 0.958 at q2 = 0.3 rising to ~1.000 at
        # q2 >= 0.6, so at these values D-0014's per-round-vs-pooled distinction is moot
        # and the comparison does not depend on which target is used.
        params["mate_q"] = list(regime.mate_q)
    pop.set_params(**params)

    rows: list[dict] = []
    if not trajectory:
        pop.simulate_generations(regime.generations)
        return pop, std, rows

    # Generation 0 carries no mating yet, so only its variance components are recorded.
    rows.append(_measure(pop, std, mates=False))
    for _ in range(regime.generations):
        pop.simulate_generations(1)
        rows.append(_measure(pop, std, mates=True))
    return pop, std, rows


def _measure(pop: Population, std: BaseStandardization, mates: bool) -> dict:
    """One row of the trajectory: the measured quantities at the current generation.

    `measure_am_quantities` labels what it returns carefully, and the labels matter:
    `V_A`/`V_Y`/`h2` describe generation `t`, while `rho_y`/`rho_g` describe the **mating
    that produced it** and are therefore measured among generation `t-1`, where the mate
    pairing is recorded. At equilibrium the two coincide; during burn-in they do not.
    """
    row: dict = {"t": int(pop.t)}
    if mates:
        m = measure_am_quantities(pop, "y", standardization=std, include_drift=False)
        row.update(m.as_dict())
    else:
        # No mating has happened yet, so ask only for the variance components.
        row.update({
            "V_A": float(var_A_pinned(pop, "y", std)),
            "V_Y": float(np.asarray(pop.traits["y"].y, dtype=float).var(ddof=1)),
        })
        row["h2"] = row["V_A"] / row["V_Y"]
    row.update(effective_n_variants(pop, "y", std))
    return row


def _drop_placeholder_kinship(pop: Population) -> int:
    """Strip the dense `N x N` `pop.K` from a population and its retained generations.

    `Population.__init__` sets `self.K = np.diag(np.ones(N))` and its own comment marks it
    "not functional yet" -- it is a placeholder identity matrix carrying no information. At
    `N = 4000` that is 128 MB **per generation**, and it dominated the saved bundle by 99%
    (260 MB, of which 256 MB was two copies of an identity). Since it scales as `N^2` it
    would be 800 MB per generation at `N = 10000`.

    Dropping it loses nothing: it is exactly `np.eye(N)` by construction, reconstructible in
    one line if anything ever needs it. This MUTATES the population, which is safe here
    because the generate stage saves and then discards it.
    """
    freed = 0
    for gen_pop in {id(g): g for g in [pop, *getattr(pop, "past", [])]}.values():
        K = getattr(gen_pop, "K", None)
        if K is not None:
            freed += getattr(K, "nbytes", 0)
            gen_pop.K = None
    return freed


def save(regime: Regime, rep: int, pop: Population, std: BaseStandardization,
         trajectory: list[dict] | None = None) -> Path:
    out_dir = SCRATCH_ROOT / regime.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rep{rep:03d}.pkl"
    freed = _drop_placeholder_kinship(pop)
    with open(path, "wb") as fh:
        # The standardization travels WITH the population: it cannot be rebuilt later,
        # because simulating forward destroys the base allele frequencies it is pinned to.
        pickle.dump({"population": pop, "standardization": std,
                     "regime": asdict(regime), "rep": rep,
                     "trajectory": trajectory or [],
                     "dropped_placeholder_kinship_bytes": freed}, fh)
    return path


def load(regime_name: str, rep: int) -> dict:
    """Load one saved replicate. This is what the verify notebooks call."""
    path = SCRATCH_ROOT / regime_name / f"rep{rep:03d}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- run `python generate_populations.py {regime_name}` first"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def available(regime_name: str) -> list[int]:
    """Which replicates of a regime are on disk."""
    out_dir = SCRATCH_ROOT / regime_name
    if not out_dir.is_dir():
        return []
    return sorted(int(p.stem[3:]) for p in out_dir.glob("rep*.pkl"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("regime", nargs="?", help="which regime to generate")
    ap.add_argument("--reps", type=int, default=1, help="independent replicates (default 1)")
    ap.add_argument("--seed0", type=int, default=100, help="first seed; rep r uses seed0 + r")
    ap.add_argument("--list", action="store_true", help="list the regimes and exit")
    args = ap.parse_args()

    if args.list or not args.regime:
        print(f"scratch root: {SCRATCH_ROOT}\n")
        for r in REGIMES.values():
            have = available(r.name)
            print(f"{r.name}")
            print(f"    {r.purpose}")
            print(f"    N={r.N} M={r.M} h2_0={r.h2_0} AM_r={r.AM_r} "
                  f"generations={r.generations}")
            print(f"    on disk: {len(have)} replicate(s)")
            if r.notes:
                print(f"    note: {r.notes}")
            print()
        return

    if args.regime not in REGIMES:
        raise SystemExit(f"unknown regime {args.regime!r}; known: {sorted(REGIMES)}")
    regime = REGIMES[args.regime]

    print(f"{regime.name}: N={regime.N} M={regime.M} AM_r={regime.AM_r} "
          f"generations={regime.generations}, {args.reps} replicate(s)")
    for rep in range(args.reps):
        t0 = time.time()
        pop, std, traj = build(regime, seed=args.seed0 + rep)
        path = save(regime, rep, pop, std, traj)
        size_mb = path.stat().st_size / 1e6
        print(f"  rep {rep:3d}  t={pop.t}  {time.time() - t0:6.1f}s  "
              f"{size_mb:7.1f} MB  {len(traj):3d} traj rows  -> {path}")


if __name__ == "__main__":
    main()
