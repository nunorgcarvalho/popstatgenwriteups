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

# The exogenous nodes are the HAPLOID alleles, not the diploid genotypes. Var[z] = 1/2 is fixed
# by the standardization and by allele frequencies not moving, so it is the same in every
# generation; Var[x] = 1 + alpha is not, and would have to be re-derived each time. Putting the
# fixed quantity at the bottom of the model is what keeps the diagram reusable downstream.
_ALLELES = [(who, origin, k) for who in "mp" for origin in ("mat", "pat") for k in (1, 2)]


def _z(who: str, origin: str, k: int) -> str:
    return f"z_{origin}_{who}{k}"


MATED_PAIR_2V = "\n".join(
    [
        "latent: g_m, e_m, g_p, e_p",
        "latent: " + ", ".join(_z(*a) for a in _ALLELES),
        "positive: V_E",
        "real: beta_1, beta_2",
        *[
            f"label: {_z(who, o, k)} = $z^{{({o[0]})}}_{{{who},{k}}}$"
            for who, o, k in _ALLELES
        ],
        *[f"label: x_{who}{k} = $x_{{{who},{k}}}$" for who in "mp" for k in (1, 2)],
        *[f"label: {v}_{who} = ${v}_{who}$" for who in "mp" for v in "gey"],
        # a genotype is the sum of the two alleles behind it -- unit coefficients, no free
        # parameter, and no variance of its own
        *[
            f"x_{who}{k} ~ {_z(who, 'mat', k)} + {_z(who, 'pat', k)}"
            for who in "mp"
            for k in (1, 2)
        ],
        "g_m ~ beta_1*x_m1 + beta_2*x_m2",
        "g_p ~ beta_1*x_p1 + beta_2*x_p2",
        "y_m ~ g_m + e_m",
        "y_p ~ g_p + e_p",
        *[f"{_z(*a)} ~~ 1/2*{_z(*a)}" for a in _ALLELES],
        "e_m ~~ V_E*e_m",
        "e_p ~~ V_E*e_p",
        "y_m -- [rho_y]*y_p",
    ]
)

# Two variants side by side per individual, alleles on the top row. Kept under ~14.5cm wide so it
# fits the a4/1in textwidth without scaling.
def _person_layout(who: str, dx: float) -> dict[str, tuple[float, float]]:
    return {
        _z(who, "mat", 1): (dx + 0.0, 4.6), _z(who, "pat", 1): (dx + 1.6, 4.6),
        f"x_{who}1": (dx + 0.8, 3.0),
        _z(who, "mat", 2): (dx + 3.4, 4.6), _z(who, "pat", 2): (dx + 5.0, 4.6),
        f"x_{who}2": (dx + 4.2, 3.0),
        f"e_{who}": (dx + 0.0, 1.5), f"g_{who}": (dx + 2.5, 1.5),
        f"y_{who}": (dx + 2.5, 0.0),
    }


MATED_PAIR_2V_LAYOUT = Layout({**_person_layout("m", 0.0), **_person_layout("p", 8.0)})


def mated_pair_2v() -> pm.Model:
    return pm.from_text(MATED_PAIR_2V, name="mated pair, two causal variants, allele level")


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


def check_allele_level() -> int:
    """Assert the allele-level facts the text states for the base population.

    These are the ones that justify putting the alleles at the bottom of the model rather than
    the genotypes: the allele variance is fixed at 1/2, alpha is zero here, and each of the four
    cross-mate allele pairs carries exactly a quarter of the genotype covariance.
    """
    model = mated_pair_2v()
    engine = pm.RAMEngine(model)
    b1, b2, V_E, rho_y = (model.sym(s) for s in ("beta_1", "beta_2", "V_E", "rho_y"))
    V_P = b1**2 + b2**2 + V_E
    betas = {1: b1, 2: b2}
    checked = 0

    for who in "mp":
        for origin in ("mat", "pat"):
            for k in (1, 2):
                assert engine.var(_z(who, origin, k)) == sp.Rational(1, 2)
                checked += 1
        # alpha = 0 in the base population, at every pairing of alleles within an individual
        for a, b in (
            (_z(who, "mat", 1), _z(who, "pat", 1)),   # same variant, the alpha of Yengo et al.
            (_z(who, "mat", 1), _z(who, "mat", 2)),   # same gamete, different variants
            (_z(who, "mat", 1), _z(who, "pat", 2)),   # different gamete, different variants
        ):
            assert engine.cov(a, b) == 0, f"Cov[{a},{b}] = {engine.cov(a, b)}, expected 0"
            checked += 1
        # ... so the genotype variance is 1 + alpha = 1, and only here
        for k in (1, 2):
            assert sp.simplify(engine.var(f"x_{who}{k}") - 1) == 0
            checked += 1

    # Across the mates, each of the four allele pairs carries a quarter, and they sum back to the
    # genotype-level covariance. This is the statement the figure's traced chain makes.
    for k in (1, 2):
        for lidx in (1, 2):
            total = 0
            for u in ("mat", "pat"):
                for v in ("mat", "pat"):
                    got = engine.cov(_z("m", u, k), _z("p", v, lidx))
                    want = betas[k] * betas[lidx] * rho_y / (4 * V_P)
                    assert sp.simplify(got - want) == 0, f"Cov[z_{u}_m{k}, z_{v}_p{lidx}]: {got}"
                    total += got
                    checked += 1
            assert sp.simplify(total - engine.cov(f"x_m{k}", f"x_p{lidx}")) == 0
            checked += 1
        # each allele contributes half of its variant's effect to the phenotype
        assert sp.simplify(engine.cov(_z("m", "mat", k), "y_m") - betas[k] / 2) == 0
        checked += 1
    return checked


def check_segregation_variance() -> int:
    """Assert that (1 - alpha)/4 is forced, not chosen.

    A transmitting parent whose own two alleles are ALREADY correlated -- so the check bites at
    the general alpha, not only at the alpha = 0 of the base population, which is the case a
    generation-1 model would silently pass.
    """
    model = pm.Model("transmission from a parent with alpha != 0")
    alpha = model.declare("alpha", real=True)
    for v in ("z_mat_par", "z_pat_par", "s_o", "z_o"):
        model.add_var(v, latent=True)
    half = sp.Rational(1, 2)
    model.add_variance("z_mat_par", half)
    model.add_variance("z_pat_par", half)
    model.add_cov("z_mat_par", "z_pat_par", alpha / 2)      # eq:alpha-def
    model.add_path("z_mat_par", "z_o", half)                # eq:transmit-mean
    model.add_path("z_pat_par", "z_o", half)
    model.add_variance("s_o", (1 - alpha) / 4)              # eq:seg-var-alpha
    model.add_path("s_o", "z_o", 1)
    engine = pm.RAMEngine(model)

    # the whole point: the offspring's allele still has variance 1/2, for every alpha
    assert sp.simplify(engine.var("z_o") - half) == 0, engine.var("z_o")
    # the parent's genotype variance is 1 + alpha (eq:varx-alpha)
    parent_x = engine.var("z_mat_par") + engine.var("z_pat_par") + 2 * engine.cov(
        "z_mat_par", "z_pat_par"
    )
    assert sp.simplify(parent_x - (1 + alpha)) == 0, parent_x
    # ... of which the transmitted half carries a quarter, leaving exactly the segregation term
    assert sp.simplify(parent_x / 4 + (1 - alpha) / 4 - half) == 0
    # and the residual is uncorrelated with the parental alleles, so it is a legitimate
    # exogenous node rather than a fudge factor
    for parental in ("z_mat_par", "z_pat_par"):
        assert sp.simplify(engine.cov("s_o", parental)) == 0
    return 5


# ----------------------------------------------------------------------------------------------
# Figure 2: the same pair plus two offspring. Transmission is the ONLY thing added -- 1/2 from
# each of the transmitting parent's two alleles, plus the segregation residual.
# ----------------------------------------------------------------------------------------------

#: the parent each origin inherits from
_FROM = {"mat": "m", "pat": "p"}
_KIDS = ("o1", "o2")


def _offspring_lines(who: str) -> list[str]:
    """Transmission into one offspring, for both parental origins and both variants."""
    out = []
    for origin, parent in _FROM.items():
        for k in (1, 2):
            child = _z(who, origin, k)
            # eq:transmit-mean -- the linear projection, 1/2 from EACH of the parent's alleles
            out.append(
                f"{child} ~ 1/2*{_z(parent, 'mat', k)} + 1/2*{_z(parent, 'pat', k)}"
            )
            # eq:seg-var-alpha at alpha^(0) = 0, drawn as this allele's own disturbance variance
            out.append(f"{child} ~~ 1/4*{child}")
    for k in (1, 2):
        out.append(f"x_{who}{k} ~ {_z(who, 'mat', k)} + {_z(who, 'pat', k)}")
    out += [
        f"g_{who} ~ beta_1*x_{who}1 + beta_2*x_{who}2",
        f"y_{who} ~ g_{who} + e_{who}",
        f"e_{who} ~~ V_E*e_{who}",
    ]
    return out


PAIR_OFFSPRING_2V = "\n".join(
    [MATED_PAIR_2V]
    + [
        "latent: " + ", ".join(
            [_z(who, o, k) for who in _KIDS for o in ("mat", "pat") for k in (1, 2)]
            + [f"g_{who}" for who in _KIDS]
            + [f"e_{who}" for who in _KIDS]
        )
    ]
    + [
        f"label: {_z(who, o, k)} = $z^{{({o[0]})}}_{{{who},{k}}}$"
        for who in _KIDS for o in ("mat", "pat") for k in (1, 2)
    ]
    + [f"label: x_{who}{k} = $x_{{{who},{k}}}$" for who in _KIDS for k in (1, 2)]
    + [f"label: {v}_{who} = ${v}_{{{who}}}$" for who in _KIDS for v in "gey"]
    + [line for who in _KIDS for line in _offspring_lines(who)]
)


# Layout. The transmission edges are the ones that must stay short and uncrossed, since there are
# sixteen of them; everything else can afford to travel. So:
#   - each parent is drawn flowing UPWARD (alleles at the bottom of its block, phenotype at the
#     top), putting the two allele rows adjacent across the generation boundary;
#   - a child's maternally inherited alleles sit directly beneath the mother's, and its paternally
#     inherited ones beneath the father's, rather than the child occupying one contiguous block.
# The cost is that a child's z -> x edges then span the figure, but there are eight of those
# against sixteen transmission edges, and they fan cleanly into the centre.
_PARENT_Z_Y = 8.4
#: a full 3.4cm below the parents' alleles: the self-loops carrying the segregation variance are
#: drawn above their node, and need clear air between the two generations
_CHILD_Z_Y = 5.0
#: x positions of each parent's four alleles, as (variant 1 pair, variant 2 pair)
_PARENT_Z_X = {"m": (0.0, 2.0, 4.6, 6.6), "p": (14.6, 16.6, 19.2, 21.2)}
#: a child's inherited allele sits between the two parental alleles that feed it, offset so the
#: two siblings do not overlap
_CHILD_OFFSET = {"o1": -0.85, "o2": 0.85}


def _parent_layout_up(who: str) -> dict[str, tuple[float, float]]:
    a, b, c, d = _PARENT_Z_X[who]
    mid1, mid2 = (a + b) / 2, (c + d) / 2
    centre = (mid1 + mid2) / 2
    return {
        _z(who, "mat", 1): (a, _PARENT_Z_Y), _z(who, "pat", 1): (b, _PARENT_Z_Y),
        _z(who, "mat", 2): (c, _PARENT_Z_Y), _z(who, "pat", 2): (d, _PARENT_Z_Y),
        f"x_{who}1": (mid1, 10.0), f"x_{who}2": (mid2, 10.0),
        f"g_{who}": (centre, 11.6), f"e_{who}": (centre + 3.6, 11.6),
        f"y_{who}": (centre, 13.2),
    }


def _offspring_layout(who: str, xs: tuple[float, float], ys: tuple[float, float]):
    """`xs` places this child's genotype nodes, `ys` its two lower rows."""
    out = {}
    for origin, parent in _FROM.items():
        a, b, c, d = _PARENT_Z_X[parent]
        for k, (lo, hi) in ((1, (a, b)), (2, (c, d))):
            out[_z(who, origin, k)] = ((lo + hi) / 2 + _CHILD_OFFSET[who], _CHILD_Z_Y)
    out[f"x_{who}1"], out[f"x_{who}2"] = (xs[0], ys[0]), (xs[1], ys[0])
    centre = (xs[0] + xs[1]) / 2
    out[f"g_{who}"] = (centre, ys[1])
    out[f"e_{who}"] = (centre - 3.4, ys[1])
    out[f"y_{who}"] = (centre, ys[1] - 1.4)
    return out


PAIR_OFFSPRING_2V_LAYOUT = Layout(
    {
        **_parent_layout_up("m"), **_parent_layout_up("p"),
        **_offspring_layout("o1", (5.2, 8.2), (2.6, 1.1)),
        **_offspring_layout("o2", (13.0, 16.0), (2.6, 1.1)),
    }
)


def pair_offspring_2v() -> pm.Model:
    return pm.from_text(PAIR_OFFSPRING_2V, name="mated pair with two offspring, allele level")


def check_offspring() -> int:
    """Assert what the offspring figure is there to show, at generation 1."""
    model = pair_offspring_2v()
    issues = [i for i in model.validate() if i.severity == "error"]
    assert not issues, issues
    engine = pm.RAMEngine(model)
    b1, b2, V_E, rho_y = (model.sym(s) for s in ("beta_1", "beta_2", "V_E", "rho_y"))
    betas = {1: b1, 2: b2}
    V_A, V_P = b1**2 + b2**2, b1**2 + b2**2 + V_E
    rho_g = rho_y * V_A / V_P
    checked = 0

    for who in _KIDS:
        for k in (1, 2):
            # alleles still have variance 1/2 -- the segregation term is exactly what preserves it
            for origin in ("mat", "pat"):
                assert engine.var(_z(who, origin, k)) == sp.Rational(1, 2)
                checked += 1
            # alpha^(1) is now NONZERO: the offspring departs from Hardy-Weinberg in one
            # generation, though neither parent did
            alpha1 = 2 * engine.cov(_z(who, "mat", k), _z(who, "pat", k))
            assert sp.simplify(alpha1 - betas[k] ** 2 * rho_y / (2 * V_P)) == 0, alpha1
            assert sp.simplify(engine.var(f"x_{who}{k}") - (1 + alpha1)) == 0
            checked += 2
        # gametic phase disequilibrium WITHIN the offspring, which was zero in both parents
        assert sp.simplify(
            engine.cov(f"x_{who}1", f"x_{who}2") - b1 * b2 * rho_y / (2 * V_P)
        ) == 0
        # the standard one-generation recursion
        assert sp.simplify(engine.var(f"g_{who}") - V_A * (1 + rho_g / 2)) == 0
        checked += 2

    # eq:alpha-gen1 as the writeup derives it: FOUR chains, each worth beta_k^2 mu / 16. Checked
    # against the tracer's own enumeration, so the chain count in the prose is verified too.
    mu = rho_y / V_P
    for k in (1, 2):
        d = pm.WrightTracer(model).trace(_z("o1", "mat", k), _z("o1", "pat", k))
        assert len(d) == 4, f"prose says four chains, tracer found {len(d)}"
        for chain in d:
            assert sp.simplify(chain.contribution - betas[k] ** 2 * mu / 16) == 0
        # eq:alpha-gen1-rhog -- the same thing as a share of V_A times half rho_g
        alpha1 = 2 * d.total
        assert sp.simplify(alpha1 - (betas[k] ** 2 / V_A) * (rho_g / 2)) == 0
        checked += 6

    # parent-offspring and full-sib covariances, from the same model
    assert sp.simplify(engine.cov("g_m", "g_o1") - V_A * (1 + rho_g) / 2) == 0
    assert sp.simplify(engine.cov("y_m", "g_o1") - V_A * (1 + rho_y) / 2) == 0
    # the two children are NOT identical: independent coins are what separates them
    sibs = engine.cov("g_o1", "g_o2")
    assert sp.simplify(sibs - V_A * (1 + rho_g) / 2) == 0, sibs
    assert sp.simplify(engine.var("g_o1") - sibs) != 0, "sibs must differ from a single child"
    checked += 4
    return checked


def pair_offspring_general(n_variants: int) -> pm.Model:
    """The pair-plus-two-offspring model at arbitrary M, for checking Table 3."""
    kk = range(1, n_variants + 1)
    lines = [
        "latent: "
        + ", ".join(
            [f"{v}_{w}" for w in ("m", "p", "o1", "o2") for v in "ge"]
            + [
                _z(w, o, k)
                for w in ("m", "p", "o1", "o2")
                for o in ("mat", "pat")
                for k in kk
            ]
        ),
        "positive: V_E",
        "real: " + ", ".join(f"beta_{k}" for k in kk),
    ]
    for who in ("m", "p"):                                   # founders: alleles are exogenous
        lines += [f"{_z(who, o, k)} ~~ 1/2*{_z(who, o, k)}" for k in kk for o in ("mat", "pat")]
    for who in ("o1", "o2"):                                 # children: transmission
        for k in kk:
            for origin, parent in _FROM.items():
                lines.append(
                    f"{_z(who, origin, k)} ~ 1/2*{_z(parent, 'mat', k)} + 1/2*{_z(parent, 'pat', k)}"
                )
                lines.append(f"{_z(who, origin, k)} ~~ 1/4*{_z(who, origin, k)}")
    for who in ("m", "p", "o1", "o2"):
        lines += [f"x_{who}{k} ~ {_z(who, 'mat', k)} + {_z(who, 'pat', k)}" for k in kk]
        lines.append(f"g_{who} ~ " + " + ".join(f"beta_{k}*x_{who}{k}" for k in kk))
        lines.append(f"y_{who} ~ g_{who} + e_{who}")
        lines.append(f"e_{who} ~~ V_E*e_{who}")
    lines.append("y_m -- [rho_y]*y_p")
    return pm.from_text("\n".join(lines), name=f"pair + 2 offspring, M={n_variants}")


def check_relative_table(n_variants: int = 3) -> int:
    """Assert Table 3 -- the six covariances, for full sibs and for parent-offspring.

    Run at M > 2 because the table is stated for general M, and a cross-variant term that
    happened to work for two variants would not show up at two.
    """
    model = pair_offspring_general(n_variants)
    engine = pm.RAMEngine(model)
    kk = range(1, n_variants + 1)
    betas = {k: model.sym(f"beta_{k}") for k in kk}
    V_E, rho_y = model.sym("V_E"), model.sym("rho_y")
    V_A = sp.Add(*[b**2 for b in betas.values()])
    V_P = V_A + V_E
    rho_g = rho_y * V_A / V_P
    checked = 0

    for k in kk:
        alpha1 = betas[k] ** 2 * rho_y / (2 * V_P)
        # the genotype rows are the SAME for both relationships -- that is the table's claim
        for a, b in (("x_o11", "x_o21"), ("x_m1", "x_o11")):
            a, b = a[:-1] + str(k), b[:-1] + str(k)
            assert sp.simplify(engine.cov(a, b) - (sp.Rational(1, 2) + alpha1)) == 0
            checked += 1
        for lidx in kk:
            if lidx == k:
                continue
            want = betas[k] * betas[lidx] * rho_y / (2 * V_P)
            assert sp.simplify(engine.cov(f"x_o1{k}", f"x_o2{lidx}") - want) == 0
            assert sp.simplify(engine.cov(f"x_m{k}", f"x_o1{lidx}") - want) == 0
            checked += 2

    # the "self" column: variances, and the cross-variant entry that matches the other columns
    for k in kk:
        alpha1 = betas[k] ** 2 * rho_y / (2 * V_P)
        assert sp.simplify(engine.var(f"x_o1{k}") - (1 + alpha1)) == 0
        checked += 1
        for lidx in kk:
            if lidx == k:
                continue
            # gametic phase disequilibrium WITHIN one individual is the same expression as
            # between relatives -- the claim the prose makes about the cross-variant row
            want = betas[k] * betas[lidx] * rho_y / (2 * V_P)
            assert sp.simplify(engine.cov(f"x_o1{k}", f"x_o1{lidx}") - want) == 0
            checked += 1
    assert sp.simplify(engine.var("g_o1") - V_A * (1 + rho_g / 2)) == 0
    assert sp.simplify(engine.var("y_o1") - (V_A * (1 + rho_g / 2) + V_E)) == 0
    # and the expanded forms the table prints, with rho_g written out as rho_y V_A / V_P
    assert sp.simplify(engine.var("g_o1") - V_A * (1 + rho_y * V_A / (2 * V_P))) == 0
    assert sp.simplify(engine.cov("g_o1", "g_o2") - V_A / 2 * (1 + rho_y * V_A / V_P)) == 0
    checked += 4

    # genetic values agree; phenotypes do NOT -- rho_g for sibs, rho_y for parent-offspring
    assert sp.simplify(engine.cov("g_o1", "g_o2") - V_A * (1 + rho_g) / 2) == 0
    assert sp.simplify(engine.cov("g_m", "g_o1") - V_A * (1 + rho_g) / 2) == 0
    assert sp.simplify(engine.cov("y_o1", "y_o2") - V_A * (1 + rho_g) / 2) == 0
    assert sp.simplify(engine.cov("y_m", "y_o1") - V_A * (1 + rho_y) / 2) == 0
    checked += 4

    # the two explanations the prose gives for the table, checked rather than asserted:
    # (1) every genotype-level excess is half the parents' cross-mate genotype covariance
    assert sp.simplify(
        engine.cov("x_o11", "x_o21") - sp.Rational(1, 2) - engine.cov("x_m1", "x_p1") / 2
    ) == 0
    # (2) the parent-offspring phenotype excess is exactly Cov[e_m, g_o]
    assert sp.simplify(
        engine.cov("y_m", "y_o1") - engine.cov("g_m", "g_o1") - engine.cov("e_m", "g_o1")
    ) == 0
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
    print(f"allele level:  {check_allele_level()} results agree with pathMgr")
    print(f"segregation:   {check_segregation_variance()} results agree with pathMgr")
    print(f"offspring:     {check_offspring()} results agree with pathMgr")
    for m_c in (3, 4):
        print(f"table 3, M={m_c}:  {check_relative_table(m_c)} results agree with pathMgr")
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
    # The chain worth tracing is now allele to allele: two single alleles, in different people, at
    # different variants, with no common ancestor, whose covariance is nonetheless nonzero. It
    # carries the 1/4 that makes the four allele pairs sum to the genotype-level covariance.
    decomposition = pm.WrightTracer(model).trace(_z("m", "mat", 1), _z("p", "pat", 2))
    assert len(decomposition) == 1, f"expected a single chain, got {len(decomposition)}"
    style = DiagramStyle(show_variances=True, show_unit_coefficients=True, latex_names=names)
    target = HERE / "mated_pair_2v.tikz"
    target.write_text(
        to_tikz(
            model,
            layout=MATED_PAIR_2V_LAYOUT,
            style=style,
            highlight=decomposition.chains[0],
            caption_name=r"\operatorname{Cov}\left[z^{(m)}_{m,1}, z^{(p)}_{p,2}\right]",
        )
    )
    print(f"wrote {target.relative_to(HERE.parent)}")

    # Figure 2: the pair plus two offspring. No highlight -- this figure's job is to state the
    # model, and the previous one has already shown the tracing rules in use.
    offspring = HERE / "pair_offspring_2v.tikz"
    offspring.write_text(
        to_tikz(
            pair_offspring_2v(),
            layout=PAIR_OFFSPRING_2V_LAYOUT,
            style=DiagramStyle(
                show_variances=True, show_unit_coefficients=False, latex_names=names
            ),
        )
    )
    print(f"wrote {offspring.relative_to(HERE.parent)}")

    stale = HERE / "mated_pair_2v_traced.tikz"
    if stale.exists():
        stale.unlink()
        print(f"removed {stale.relative_to(HERE.parent)} (merged into mated_pair_2v)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
