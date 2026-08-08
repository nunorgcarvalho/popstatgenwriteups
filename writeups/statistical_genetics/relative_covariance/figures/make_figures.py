"""Generate the path diagrams for `relative_covariance.tex`, and check the tables against them.

    python figures/make_figures.py          # regenerate the .tikz files, run the checks
    python figures/make_figures.py --check  # checks only, touch nothing

Each figure is emitted as a bare `tikzpicture` for `\\input`, using pathMgr
(https://github.com/nunixnunix04/pathMgr; local checkout at ~/alkes_nuno/pathMgr).

The covariance tables in the writeup are typeset by hand, because their *structure* is the point
being made and a generated table cannot carry it. So every entry in them is asserted here against
what pathMgr's RAM engine actually returns: if a hand-typed entry drifts, this script fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp

import pathmgr as pm
from pathmgr.render import DiagramStyle, Layout, to_tikz

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------
# Figure 1: one mated pair at generation 0, trait built from two causal variants.
# Nothing about the assortment is specified except the single co-path on the phenotypes; every
# covariance in the table below is *derived* from it.
# ----------------------------------------------------------------------------------------------

MATED_PAIR_2V = """
    latent: g_m, e_m, g_p, e_p
    positive: V_E
    real: beta_1, beta_2
    label: x_m1 = $x_{m,1}$
    label: x_m2 = $x_{m,2}$
    label: g_m = $g_m$
    label: e_m = $e_m$
    label: y_m = $y_m$
    label: x_p1 = $x_{p,1}$
    label: x_p2 = $x_{p,2}$
    label: g_p = $g_p$
    label: e_p = $e_p$
    label: y_p = $y_p$
    g_m ~ beta_1*x_m1 + beta_2*x_m2
    g_p ~ beta_1*x_p1 + beta_2*x_p2
    y_m ~ g_m + e_m
    y_p ~ g_p + e_p
    x_m1 ~~ 1*x_m1
    x_m2 ~~ 1*x_m2
    x_p1 ~~ 1*x_p1
    x_p2 ~~ 1*x_p2
    e_m ~~ V_E*e_m
    e_p ~~ V_E*e_p
    y_m -- [rho_y]*y_p
"""

MATED_PAIR_2V_LAYOUT = Layout(
    {
        "x_m1": (0.0, 3.2), "x_m2": (2.0, 3.2),
        "e_m": (-1.3, 1.6), "g_m": (1.0, 1.6),
        "y_m": (1.0, 0.0),
        "x_p1": (7.0, 3.2), "x_p2": (9.0, 3.2),
        "g_p": (8.0, 1.6), "e_p": (10.3, 1.6),
        "y_p": (8.0, 0.0),
    }
)


def mated_pair_2v() -> pm.Model:
    return pm.from_text(MATED_PAIR_2V, name="mated pair, two causal variants")


# ----------------------------------------------------------------------------------------------
# checking the hand-typed tables
# ----------------------------------------------------------------------------------------------

def _tidy(expr, b1, b2, V_E, V_A, V_P):
    """Rewrite in terms of V_A and V_P, which is how the writeup states these."""
    e = sp.expand(expr)
    # b2**2 -> V_A - b1**2 turns every sum of squares into V_A, including inside b2**4
    e = sp.expand(e.subs(b2**2, V_A - b1**2))
    e = sp.expand(e.subs(V_E, V_P - V_A))
    return sp.simplify(sp.factor(e))


def check_mated_pair_2v() -> int:
    """Assert the writeup's two tables entry by entry. Returns the number of entries checked."""
    model = mated_pair_2v()
    issues = [i for i in model.validate() if i.severity == "error"]
    assert not issues, issues
    engine = pm.RAMEngine(model)

    b1, b2, V_E, rho_y = (model.sym(s) for s in ("beta_1", "beta_2", "V_E", "rho_y"))
    V_A, V_P = sp.symbols("V_A V_P", positive=True)

    def cov(a, b):
        return _tidy(engine.cov(a, b), b1, b2, V_E, V_A, V_P)

    # -- Table 1: within one individual (stated for the maternal member; the paternal member is
    #    identical by construction, and that is asserted rather than assumed) -------------------
    within = {
        ("x_m1", "x_m1"): sp.Integer(1),
        ("x_m1", "x_m2"): sp.Integer(0),
        ("x_m1", "g_m"): b1,
        ("x_m1", "e_m"): sp.Integer(0),
        ("x_m1", "y_m"): b1,
        ("x_m2", "x_m2"): sp.Integer(1),
        ("x_m2", "g_m"): b2,
        ("x_m2", "e_m"): sp.Integer(0),
        ("x_m2", "y_m"): b2,
        ("g_m", "g_m"): V_A,
        ("g_m", "e_m"): sp.Integer(0),
        ("g_m", "y_m"): V_A,
        ("e_m", "e_m"): V_P - V_A,   # V_E
        ("e_m", "y_m"): V_P - V_A,   # V_E
        ("y_m", "y_m"): V_P,
    }
    checked = 0
    for (a, b), expected in within.items():
        got = cov(a, b)
        assert sp.simplify(got - expected) == 0, f"Cov[{a},{b}]: got {got}, table says {expected}"
        # the same entry with m and p swapped must agree
        mirror = cov(a.replace("_m", "_p"), b.replace("_m", "_p"))
        assert sp.simplify(mirror - expected) == 0, f"m/p asymmetry at Cov[{a},{b}]: {mirror}"
        checked += 2

    # -- Table 2: between the two mates. The claim is stronger than a list of entries: every
    #    cross-mate covariance is the OUTER PRODUCT of each variable's covariance with its own
    #    phenotype, scaled by the co-path coefficient rho_y / V_P. Assert that form directly,
    #    which covers all 25 entries at once and is the statement the writeup makes. -----------
    # Checked on the engine's raw output rather than the tidied form, so the rewriting in
    # `_tidy` cannot be what makes the two sides agree.
    mu_raw = rho_y / (b1**2 + b2**2 + V_E)
    for a in ("x_m1", "x_m2", "g_m", "e_m", "y_m"):
        for b in ("x_p1", "x_p2", "g_p", "e_p", "y_p"):
            got = engine.cov(a, b)
            leg_a = engine.cov(a, "y_m")
            leg_b = engine.cov("y_p", b)
            assert sp.simplify(got - leg_a * mu_raw * leg_b) == 0, (
                f"leg rule fails at Cov[{a},{b}]: {got} != {leg_a} * {mu_raw} * {leg_b}"
            )
            checked += 1

    # -- the four cross-mate entries the writeup writes out in full ---------------------------
    named = {
        ("x_m1", "x_p1"): b1**2 * rho_y / V_P,
        ("x_m1", "x_p2"): b1 * b2 * rho_y / V_P,
        ("g_m", "g_p"): V_A**2 * rho_y / V_P,
        ("e_m", "g_p"): V_A * (V_P - V_A) * rho_y / V_P,
        ("y_m", "y_p"): rho_y * V_P,
    }
    for (a, b), expected in named.items():
        got = cov(a, b)
        assert sp.simplify(got - expected) == 0, f"Cov[{a},{b}]: got {got}, table says {expected}"
        checked += 1

    # -- rho_g: the induced genetic correlation is rho_y h^2, not rho_y ------------------------
    rho_g = sp.simplify(cov("g_m", "g_p") / sp.sqrt(cov("g_m", "g_m") * cov("g_p", "g_p")))
    assert sp.simplify(rho_g - rho_y * V_A / V_P) == 0, rho_g
    checked += 1

    # -- and the trap: a bidirected edge in place of the co-path reaches none of the causes ----
    bidirected = pm.from_text(
        MATED_PAIR_2V.replace("y_m -- [rho_y]*y_p", "y_m ~~ rho_y*V_P_sym*y_p").replace(
            "positive: V_E", "positive: V_E, V_P_sym"
        ),
        name="mated pair, bidirected instead of co-path",
    )
    other = pm.RAMEngine(bidirected)
    assert other.cov("x_m1", "x_p1") == 0, other.cov("x_m1", "x_p1")
    assert other.cov("g_m", "g_p") == 0, other.cov("g_m", "g_p")
    checked += 2

    return checked


def mated_pair_general(n_variants: int) -> pm.Model:
    """The same model with `n_variants` causal variants, for checking the general formulas."""
    betas = [f"beta_{k}" for k in range(1, n_variants + 1)]
    lines = [
        "latent: g_m, e_m, g_p, e_p",
        "positive: V_E",
        "real: " + ", ".join(betas),
    ]
    for who in ("m", "p"):
        terms = " + ".join(f"{b}*x_{who}{k}" for k, b in enumerate(betas, start=1))
        lines.append(f"g_{who} ~ {terms}")
        lines.append(f"y_{who} ~ g_{who} + e_{who}")
        lines.append(f"e_{who} ~~ V_E*e_{who}")
        for k in range(1, n_variants + 1):
            lines.append(f"x_{who}{k} ~~ 1*x_{who}{k}")
    lines.append("y_m -- [rho_y]*y_p")
    return pm.from_text("\n".join(lines), name=f"mated pair, {n_variants} causal variants")


def check_general_formulas(n_variants: int = 3) -> int:
    """Assert the three results the writeup states for a trait with M_c causal variants.

    The section derives them on a two-variant diagram and then claims them in general, so the
    general claim is what has to be checked -- at M_c > 2, where a cross-variant term that only
    happened to work for two variants would show up.
    """
    model = mated_pair_general(n_variants)
    assert not [i for i in model.validate() if i.severity == "error"]
    engine = pm.RAMEngine(model)
    betas = [model.sym(f"beta_{k}") for k in range(1, n_variants + 1)]
    V_E, rho_y = model.sym("V_E"), model.sym("rho_y")
    V_A = sp.Add(*[b**2 for b in betas])          # sum_k beta_k^2
    V_P = V_A + V_E
    checked = 0

    # 1. Cov[x_mk, x_pl] = beta_k beta_l rho_y / V_P, for EVERY pair including k == l.
    for k in range(1, n_variants + 1):
        for lidx in range(1, n_variants + 1):
            got = engine.cov(f"x_m{k}", f"x_p{lidx}")
            want = betas[k - 1] * betas[lidx - 1] * rho_y / V_P
            assert sp.simplify(got - want) == 0, f"Cov[x_m{k},x_p{lidx}]: {got}"
            checked += 1
    # ... and within an individual it is still exactly zero, which is what makes the cross-mate
    # term the whole of the new disequilibrium.
    for k in range(1, n_variants + 1):
        for lidx in range(k + 1, n_variants + 1):
            assert engine.cov(f"x_m{k}", f"x_m{lidx}") == 0
            checked += 1

    # 2. Cov[e_m, g_p] = V_A V_E rho_y / V_P
    assert sp.simplify(engine.cov("e_m", "g_p") - V_A * V_E * rho_y / V_P) == 0
    # 3. Cov[g_m, g_p] = V_A^2 rho_y / V_P, hence rho_g = rho_y h^2
    assert sp.simplify(engine.cov("g_m", "g_p") - V_A**2 * rho_y / V_P) == 0
    rho_g = engine.cov("g_m", "g_p") / engine.var("g_m")
    assert sp.simplify(rho_g - rho_y * V_A / V_P) == 0, rho_g
    checked += 3

    # and V_A really is the sum of squared effects, rather than that being an assumption
    assert sp.simplify(engine.var("g_m") - V_A) == 0
    assert sp.simplify(engine.var("y_m") - V_P) == 0
    checked += 2
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run the checks, write nothing")
    args = parser.parse_args()

    n = check_mated_pair_2v()
    print(f"mated_pair_2v: {n} covariance entries agree with pathMgr")
    for m_c in (3, 5):
        print(f"general M_c={m_c}: {check_general_formulas(m_c)} results agree with pathMgr")

    if args.check:
        return 0

    model = mated_pair_2v()
    b1, b2, V_E = (model.sym(s) for s in ("beta_1", "beta_2", "V_E"))
    # pathMgr renders the phenotypic variance as the sum it is derived from; two pages of prose
    # around this figure call it \VPo. `latex_names` is how a figure is told the document's own
    # name for a quantity -- it reaches edge labels and captions alike, so the figure stays
    # internally consistent as well as consistent with the text.
    names = {b1**2 + b2**2 + V_E: r"\VPo"}

    # ONE figure, doing both jobs: the full model *and* a traced chain on it.
    #  - show_variances=True keeps the exogenous variances that are not on the chain. They are
    #    faded rather than dropped, so the figure still states the whole model -- a reader
    #    checking any other covariance has everything they need.
    #  - show_unit_coefficients=True is against the usual convention, but this figure's job is to
    #    let a reader multiply along the chain, so every factor has to be visible. The caption
    #    follows the same setting, so the product printed underneath matches the edges drawn.
    # Cov[x_m1, x_p2] is the chain worth tracing: two genotypes in different people, at different
    # variants, with no common ancestor, whose covariance is nonetheless nonzero.
    decomposition = pm.WrightTracer(model).trace("x_m1", "x_p2")
    assert len(decomposition) == 1, f"expected a single chain, got {len(decomposition)}"
    style = DiagramStyle(show_variances=True, show_unit_coefficients=True, latex_names=names)
    target = HERE / "mated_pair_2v.tikz"
    target.write_text(
        to_tikz(
            model,
            layout=MATED_PAIR_2V_LAYOUT,
            style=style,
            highlight=decomposition.chains[0],
            caption_name=r"\operatorname{Cov}\left[x_{m,1}, x_{p,2}\right]",
        )
    )
    print(f"wrote {target.relative_to(HERE.parent)}")

    stale = HERE / "mated_pair_2v_traced.tikz"
    if stale.exists():
        stale.unlink()
        print(f"removed {stale.relative_to(HERE.parent)} (merged into mated_pair_2v)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
