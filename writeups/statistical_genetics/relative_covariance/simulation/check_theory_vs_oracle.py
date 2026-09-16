"""Cross-check `theory.py` against the pathMgr oracle already built in `figures/make_figures.py`.

    python simulation/check_theory_vs_oracle.py

This is the formula-vs-oracle leg of the verification, and it is cheap because the oracle
exists: `figures/make_figures.py` already builds a pedigree carrying every standard
relationship class and asserts the writeup's formulas against pathMgr's RAM engine. What it
does *not* do is check `simulation/theory.py`, which is a separate, numeric transcription of
the same equations written for the notebooks to call.

So the two are compared here. A disagreement means one of the two transcriptions is wrong --
not that the writeup is wrong, which is what `make_figures.py` itself already tests. Keeping
this file separate from `theory.py` is deliberate: `theory.py` must stay importable with no
pathMgr or sympy dependency, so a verify notebook can call it without the symbolic stack.

This reaches into `make_figures.py`'s internals (`_pedigree`, `PATH_CLS_CASES`), which is the
concrete argument for packaging that script's oracle behind a stable API rather than leaving
it as assertions inside a figure generator -- see the note in `README.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

import theory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))
import make_figures as mf                                                    # noqa: E402

#: A numeric point to evaluate the oracle's symbolic answers at. V_A and rho_y are the
#: EQUILIBRIUM values, matching what `_pedigree` builds; V_E is held fixed as the writeup does.
#: Deliberately away from round numbers, and with V_A + V_E != 1, so that a stray factor of
#: V_Y or h^2 cannot hide inside a coincidence.
POINT = {"V_A": 0.63, "V_E": 0.47, "rho_y": 0.34}

#: The writeup's own class names, mapped onto the names `PATH_CLS_CASES` uses. The two classes
#: of tab:relationship-classes with no oracle row are recorded as None: "self" (the degenerate
#: path, trivially V_A) and "step-step sibs" (the N_mu = 4 class, which is also the one class
#: with no simulation leg -- so it has no independent check at all today).
CLASS_TO_ORACLE = {
    "self":                        None,
    "parent-parent":               "mates",
    "parent-offspring":            "parent-offspring",
    "grandparent-grandchild":      "grandparent",
    "full siblings":               "full sibs",
    "avuncular":                   "uncle-nephew",
    "first cousins":               "first cousins",
    "half siblings":               "half sibs",
    "step siblings":               "step sibs",
    "step-step siblings":          None,
    "parents of a married couple": "co-parents-in-law",
}


def _numeric(expr, model) -> float:
    """Evaluate one of the engine's symbolic covariances at `POINT`."""
    return float(sp.N(expr.subs({model.sym(k): v for k, v in POINT.items()})))


def main() -> int:
    V_A, V_E, rho_y = POINT["V_A"], POINT["V_E"], POINT["rho_y"]
    rho_g = theory.rho_g_from_pair(rho_y, V_A / (V_A + V_E))     # the model's own rho_g
    kw = dict(V_A=V_A, rho_g=rho_g, rho_y=rho_y)
    tol = 1e-10
    checked = 0

    # -- every class in the oracle's pedigree, at all four level combinations ----------------
    model, e = mf._pedigree(mf.PATH_CLS_FOUNDERS, mf.PATH_CLS_KIDS, mf.PATH_CLS_MATINGS,
                            "path classes")
    for name, (X, Y), lineals, matings, kinds in mf.PATH_CLS_CASES:
        for levels in (("g", "g"), ("y", "y"), ("y", "g"), ("g", "y")):
            want = _numeric(e.cov(f"{levels[0]}_{X}", f"{levels[1]}_{Y}"), model)
            got = theory.path_cov(lineals, matings, kinds, levels, **kw)
            assert abs(got - want) < tol, f"{name} at {levels}: theory {got} vs oracle {want}"
            checked += 1

    # -- the writeup's named classes, so the RELATIONSHIP_CLASSES table itself is checked ----
    # This is the stronger check of the two: above, theory.py is fed the oracle's own path
    # description, so only the formula is tested. Here it is fed only a class NAME and has to
    # produce the right path description from its own table.
    by_name = {c[0]: c for c in mf.PATH_CLS_CASES}
    for cls, oracle_name in CLASS_TO_ORACLE.items():
        if oracle_name is None:
            continue
        _, (X, Y), _, _, _ = by_name[oracle_name]
        for levels in (("g", "g"), ("y", "y"), ("y", "g"), ("g", "y")):
            want = _numeric(e.cov(f"{levels[0]}_{X}", f"{levels[1]}_{Y}"), model)
            got = theory.class_cov(cls, levels, **kw)
            assert abs(got - want) < tol, f"{cls} at {levels}: theory {got} vs oracle {want}"
            checked += 1

    # "self" has no oracle row, but the pedigree carries each individual's own variances, and
    # this row is the one place Cov[y, y] is not Cov[g, g] times corrections -- so check all
    # three readings against the oracle rather than just the genetic one.
    V_Y_pt = V_A + V_E
    for levels, want in ((("g", "g"), _numeric(e.var("g_m"), model)),
                         (("y", "y"), _numeric(e.var("y_m"), model)),
                         (("g", "y"), _numeric(e.cov("g_m", "y_m"), model))):
        got = theory.class_cov("self", levels, V_Y=V_Y_pt, **kw)
        assert abs(got - want) < tol, f"self at {levels}: theory {got} vs oracle {want}"
        checked += 1

    # -- the worked example of fig:path-example ----------------------------------------------
    ex_model, ex = mf._pedigree(mf.PATH_EX_FOUNDERS, mf.PATH_EX_KIDS, mf.PATH_EX_MATINGS,
                                "path example")
    lineals, matings = (2, 3, 1), ((2, 0), (3, 2))
    for levels, cases in ((("g", "g"), ("old", "young")), (("y", "y"), ("old", "young"))):
        want = _numeric(ex.cov(f"{levels[0]}_A", f"{levels[1]}_B"), ex_model)
        got = theory.path_cov(lineals, matings, cases, levels, **kw)
        assert abs(got - want) < tol, f"path example at {levels}: {got} vs {want}"
        checked += 1

    # -- the pair tier: tab:g-only-cov against the oracle's own mated pair and offspring -----
    V_Y = V_A + V_E
    pair_kw = dict(V_A=V_A, V_Y=V_Y, rho_g=rho_g, rho_y=rho_y)
    for levels in (("g", "g"), ("y", "y"), ("y", "g"), ("g", "y")):
        want = _numeric(e.cov(f"{levels[0]}_m", f"{levels[1]}_p"), model)
        got = theory.pair_cov("mates", *levels, **pair_kw)
        assert abs(got - want) < tol, f"tab:g-only-cov mates at {levels}: {got} vs {want}"
        want = _numeric(e.cov(f"{levels[0]}_m", f"{levels[1]}_a"), model)
        got = theory.pair_cov("parent-offspring", *levels, **pair_kw)
        assert abs(got - want) < tol, f"tab:g-only-cov PO at {levels}: {got} vs {want}"
        checked += 2

    # -- eq:g-seg-var, as the oracle's own disturbance on an offspring's genetic value -------
    # Var[g_a] - Var[(g_m + g_p)/2] is what the disturbance has to make up, and it must equal
    # eq:g-seg-var at the measured parameters.
    var_mean_parent = 0.25 * (_numeric(e.var("g_m"), model) + _numeric(e.var("g_p"), model)
                              + 2 * _numeric(e.cov("g_m", "g_p"), model))
    assert abs(theory.seg_var(V_A, rho_g)
               - (_numeric(e.var("g_a"), model) - var_mean_parent)) < tol, "eq:g-seg-var"
    checked += 1

    print(f"theory.py vs pathMgr oracle: {checked} numeric comparisons agree")
    print(f"  classes with no oracle row: "
          f"{[c for c, o in CLASS_TO_ORACLE.items() if o is None]}")
    return checked


if __name__ == "__main__":
    main()
