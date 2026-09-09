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
import collections
import math
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
# Figure 3: the REDUCED diagram. The base population is gone; what it left behind is carried by
# bidirected edges among each generation-1 individual's own alleles, per eq:reduction. Two such
# individuals mate and produce one generation-2 offspring.
# ----------------------------------------------------------------------------------------------

def reduced_pair_offspring() -> pm.Model:
    lines = [
        "latent: g_m, e_m, g_p, e_p, g_o, e_o",
        "latent: " + ", ".join(
            _z(w, o, k) for w in ("m", "p", "o") for o in ("mat", "pat") for k in (1, 2)
        ),
        "positive: V_E",
        "real: beta_1, beta_2",
        *[
            f"label: {_z(w, o, k)} = $z^{{({o[0]})}}_{{{w},{k}}}$"
            for w in ("m", "p", "o") for o in ("mat", "pat") for k in (1, 2)
        ],
        *[f"label: x_{w}{k} = $x_{{{w},{k}}}$" for w in ("m", "p", "o") for k in (1, 2)],
        *[f"label: {v}_{w} = ${v}_{{{w}}}$" for w in ("m", "p", "o") for v in "gey"],
    ]
    for who in ("m", "p"):
        # generation-1 individuals are now FOUNDERS: allele variance still 1/2 ...
        lines += [f"{_z(who, o, k)} ~~ 1/2*{_z(who, o, k)}" for o in ("mat", "pat") for k in (1, 2)]
        # ... but no longer mutually uncorrelated. eq:reduction, consolidated form: opposite
        # gametes carry beta_k beta_l rho_y / (4 V_P(0)) for EVERY pair of variants, k = l
        # included; same-gamete pairs carry nothing, so no edge is drawn for them.
        for k in (1, 2):
            for lidx in (1, 2):
                lines.append(
                    f"{_z(who, 'mat', k)} ~~ "
                    f"beta_{k}*beta_{lidx}*rho_y/(4*(beta_1**2+beta_2**2+V_E))*{_z(who, 'pat', lidx)}"
                )
    for k in (1, 2):
        for origin, parent in _FROM.items():
            child = _z("o", origin, k)
            lines.append(f"{child} ~ 1/2*{_z(parent, 'mat', k)} + 1/2*{_z(parent, 'pat', k)}")
            # eq:seg-var-a with the transmitting parent's a^(1)_kk, NOT 1/4
            lines.append(
                f"{child} ~~ (1 - beta_{k}**2*rho_y/(2*(beta_1**2+beta_2**2+V_E)))/4*{child}"
            )
    for who in ("m", "p", "o"):
        lines += [f"x_{who}{k} ~ {_z(who, 'mat', k)} + {_z(who, 'pat', k)}" for k in (1, 2)]
        lines.append(f"g_{who} ~ beta_1*x_{who}1 + beta_2*x_{who}2")
        lines.append(f"y_{who} ~ g_{who} + e_{who}")
        lines.append(f"e_{who} ~~ V_E*e_{who}")
    # declared by the CORRELATION, so mu^(1) = rho_y / V_P^(1) is derived from this model's own
    # variances rather than hand-computed -- V_P^(1) is an output here, not an input
    lines.append("y_m -- [rho_y]*y_p")
    return pm.from_text("\n".join(lines), name="reduced: two generation-1 mates, one offspring")


#: Landscape, like Figure 2. Four bidirected edges run among each parent's four alleles, and
#: they carry labels; squeezed into a text-column width that row becomes unreadable, which a
#: node count does not predict and only the render shows. Width is spent on the allele rows and
#: recovered by keeping the vertical rows tight, so the caption still fits the rotated page.
def _reduced_layout():
    out = {}
    for who, dx in (("m", 0.0), ("p", 13.0)):
        a, b, c, d = dx + 0.0, dx + 2.2, dx + 5.0, dx + 7.2
        out[_z(who, "mat", 1)] = (a, 6.2); out[_z(who, "pat", 1)] = (b, 6.2)
        out[_z(who, "mat", 2)] = (c, 6.2); out[_z(who, "pat", 2)] = (d, 6.2)
        out[f"x_{who}1"] = ((a + b) / 2, 7.7); out[f"x_{who}2"] = ((c + d) / 2, 7.7)
        out[f"g_{who}"] = (dx + 3.6, 9.1); out[f"e_{who}"] = (dx + 8.2, 9.1)
        out[f"y_{who}"] = (dx + 3.6, 10.5)
    # the child's maternal alleles under the mother, paternal under the father, so the eight
    # transmission edges stay short; the cost is four long z -> x edges into the centre
    out[_z("o", "mat", 1)] = (1.1, 4.0); out[_z("o", "mat", 2)] = (6.1, 4.0)
    out[_z("o", "pat", 1)] = (14.1, 4.0); out[_z("o", "pat", 2)] = (19.1, 4.0)
    out["x_o1"] = (7.6, 2.6); out["x_o2"] = (12.6, 2.6)
    out["g_o"] = (10.1, 1.2); out["e_o"] = (5.6, 1.2)
    out["y_o"] = (10.1, -0.2)
    return Layout(out)


REDUCED_LAYOUT = _reduced_layout()


def reduced_general(n_variants: int) -> pm.Model:
    """The reduced diagram at arbitrary M, for checking the generation-2 allele table."""
    kk = range(1, n_variants + 1)
    V_P0 = "(" + "+".join(f"beta_{k}**2" for k in kk) + "+V_E)"
    lines = [
        "latent: " + ", ".join(
            [f"{v}_{w}" for w in ("m", "p", "o") for v in "ge"]
            + [_z(w, o, k) for w in ("m", "p", "o") for o in ("mat", "pat") for k in kk]
        ),
        "positive: V_E",
        "real: " + ", ".join(f"beta_{k}" for k in kk),
    ]
    for who in ("m", "p"):
        lines += [f"{_z(who, o, k)} ~~ 1/2*{_z(who, o, k)}" for o in ("mat", "pat") for k in kk]
        # eq:reduction: opposite gametes only, every pair of variants
        lines += [
            f"{_z(who, 'mat', k)} ~~ beta_{k}*beta_{lidx}*rho_y/(4*{V_P0})*{_z(who, 'pat', lidx)}"
            for k in kk for lidx in kk
        ]
    for k in kk:
        for origin, parent in _FROM.items():
            child = _z("o", origin, k)
            lines.append(f"{child} ~ 1/2*{_z(parent, 'mat', k)} + 1/2*{_z(parent, 'pat', k)}")
            lines.append(f"{child} ~~ (1 - beta_{k}**2*rho_y/(2*{V_P0}))/4*{child}")
    for who in ("m", "p", "o"):
        lines += [f"x_{who}{k} ~ {_z(who, 'mat', k)} + {_z(who, 'pat', k)}" for k in kk]
        lines.append(f"g_{who} ~ " + " + ".join(f"beta_{k}*x_{who}{k}" for k in kk))
        lines.append(f"y_{who} ~ g_{who} + e_{who}")
        lines.append(f"e_{who} ~~ V_E*e_{who}")
    lines.append("y_m -- [rho_y]*y_p")
    return pm.from_text("\n".join(lines), name=f"reduced, M={n_variants}")


def check_gen2_alleles(n_variants: int = 3) -> int:
    """Assert the generation-2 allele table -- the four allele-pair cases inside one individual.

    Run at M > 2 because the table is stated for general M; the same-gamete entry in particular
    is a sum over the OTHER variants in disguise and a two-variant model barely exercises it.
    """
    model = reduced_general(n_variants)
    assert not [i for i in model.validate() if i.severity == "error"]
    engine = pm.RAMEngine(model)
    kk = range(1, n_variants + 1)
    betas = {k: model.sym(f"beta_{k}") for k in kk}
    V_E, rho_y = model.sym("V_E"), model.sym("rho_y")
    V_A0 = sp.Add(*[b**2 for b in betas.values()])
    V_P0 = V_A0 + V_E
    V_A1 = V_A0 * (1 + rho_y * V_A0 / (2 * V_P0))
    V_P1 = V_A1 + V_E
    ratio = V_A1 / V_A0
    checked = 0

    for k in kk:
        # same variant, same gamete: still exactly 1/2, which is [Constant-freq]
        assert sp.simplify(engine.var(_z("o", "mat", k)) - sp.Rational(1, 2)) == 0
        checked += 1
        for lidx in kk:
            # opposite gametes: ONE expression for every pair of variants, k = l included
            want = betas[k] * betas[lidx] * ratio**2 * rho_y / (4 * V_P1)
            got = engine.cov(_z("o", "mat", k), _z("o", "pat", lidx))
            assert sp.simplify(got - want) == 0, f"u!=v, k={k}, l={lidx}: {sp.simplify(got)}"
            checked += 1
            if lidx == k:
                continue
            # same gamete, different variants: ZERO at generation 1, nonzero now. It is a
            # quarter of the parent's own within-individual disequilibrium, so it lags a
            # generation -- note it carries V_P(0), not V_P(1).
            want_same = betas[k] * betas[lidx] * rho_y / (8 * V_P0)
            for origin in ("mat", "pat"):
                got = engine.cov(_z("o", origin, k), _z("o", origin, lidx))
                assert sp.simplify(got - want_same) == 0, f"u=v, k={k}, l={lidx}"
                checked += 1
    return checked


def check_as_recursion(n_variants: int = 3) -> int:
    """Assert the (a, s) recursion the section is now built on.

    a_kl = Cov[z^(u)_k, z^(v)_l], u != v      s_kl = Cov[z^(u)_k, z^(u)_l], k != l
    with the convention s_kk = 1/2 (the allele variance, held there by the segregation term).

        c_k       = sum_l beta_l (s_kl + a_kl)
        a'_kl     = mu c_k c_l
        s'_kl     = (s_kl + a_kl) / 2
        V_A       = 2 sum_k beta_k c_k,   V_P = V_A + V_E,   mu = rho_y / V_P

    Implemented from the equations as written in the document, then compared against pathMgr's
    generation-1 and generation-2 models. The point is that the recursion is checked against the
    diagrams, not merely against itself.
    """
    kk = list(range(1, n_variants + 1))
    betas = {k: 0.55 - 0.08 * (k - 1) for k in kk}      # deliberately unequal
    V_E, rho_y = 0.83, 0.32
    V_A0 = sum(b * b for b in betas.values())
    assert abs(V_A0 + V_E - 1.0) > 0.1, "keep V_P(0) away from 1 so a factor of V_P cannot hide"

    a = {(k, l): 0.0 for k in kk for l in kk}
    s = {(k, l): (0.5 if k == l else 0.0) for k in kk for l in kk}   # eq:skk

    def advance(a, s):
        c = {k: sum(betas[l] * (s[(k, l)] + a[(k, l)]) for l in kk) for k in kk}   # eq:c-def
        V_A = 2 * sum(betas[k] * c[k] for k in kk)                                 # eq:VA-from-as
        mu = rho_y / (V_A + V_E)
        a2 = {(k, l): mu * c[k] * c[l] for k in kk for l in kk}                    # eq:a-recursion
        s2 = {(k, l): (0.5 if k == l else 0.5 * (s[(k, l)] + a[(k, l)]))           # eq:s-recursion
              for k in kk for l in kk}
        return a2, s2, c, V_A

    checked = 0
    # t = 0 must reproduce the base population
    _, _, c0, V_A_at_0 = advance(a, s)
    assert abs(V_A_at_0 - V_A0) < 1e-12, V_A_at_0
    for k in kk:
        assert abs(c0[k] - betas[k] / 2) < 1e-12, "c^(0)_k should be beta_k / 2"
        checked += 1
    checked += 1

    a1, s1, _, _ = advance(a, s)
    a2, s2, _, _ = advance(a1, s1)

    # closed forms the document prints for generations 1 and 2
    V_P0 = V_A0 + V_E
    V_A1 = V_A0 * (1 + rho_y * V_A0 / (2 * V_P0))
    V_P1 = V_A1 + V_E
    for k in kk:
        for lidx in kk:
            assert abs(a1[(k, lidx)] - betas[k] * betas[lidx] * rho_y / (4 * V_P0)) < 1e-12
            want2 = betas[k] * betas[lidx] * rho_y / (4 * V_P1) * (V_A1 / V_A0) ** 2
            assert abs(a2[(k, lidx)] - want2) < 1e-12, f"a^(2)_{k}{lidx}"
            checked += 2
            if lidx != k:
                assert abs(s1[(k, lidx)]) < 1e-15, "s^(1) must be exactly zero"
                assert abs(s2[(k, lidx)] - betas[k] * betas[lidx] * rho_y / (8 * V_P0)) < 1e-12
                checked += 2

    # ... and against the actual path diagrams, which is the check that matters
    subs = {sp.Symbol(f"beta_{k}", real=True): betas[k] for k in kk}
    subs[sp.Symbol("V_E", positive=True)] = V_E
    subs[sp.Symbol("rho_y", real=True)] = rho_y

    def numeric(expr):
        expr = sp.sympify(expr)
        return float(sp.N(expr.subs({v: subs[v] for v in expr.free_symbols if v in subs})))

    e1 = pm.RAMEngine(pair_offspring_general(n_variants))       # Figure 2's model
    e2 = pm.RAMEngine(reduced_general(n_variants))              # Figure 3's model
    for k in kk:
        for lidx in kk:
            assert abs(numeric(e1.cov(_z("o1", "mat", k), _z("o1", "pat", lidx)))
                       - a1[(k, lidx)]) < 1e-10, f"a^(1) disagrees with Figure 2 at {k},{lidx}"
            assert abs(numeric(e2.cov(_z("o", "mat", k), _z("o", "pat", lidx)))
                       - a2[(k, lidx)]) < 1e-10, f"a^(2) disagrees with Figure 3 at {k},{lidx}"
            checked += 2
            if lidx != k:
                assert abs(numeric(e1.cov(_z("o1", "mat", k), _z("o1", "mat", lidx)))
                           - s1[(k, lidx)]) < 1e-10
                assert abs(numeric(e2.cov(_z("o", "mat", k), _z("o", "mat", lidx)))
                           - s2[(k, lidx)]) < 1e-10
                checked += 2

    # the two claims the section closes on
    prev = a1
    for _ in range(4000):
        a_next, s_next, _, _ = advance(a1, s1)
        a1, s1 = a_next, s_next
    k, lidx = kk[0], kk[-1]
    assert abs(s1[(k, lidx)] / a1[(k, lidx)] - 1.0) < 1e-9, "s and a must converge to each other"
    checked += 1
    return checked


def check_c_recursion(n_variants: int = 4) -> int:
    """Assert the collapsed c recursion, eq:c-recursion, and the two halves it is built from.

        c'_k = (1 + rho_y V_A / V_P) c_k / 2  +  beta_k (1/4 - a_kk / 2)

    SYMBOLIC in an ARBITRARY state (a_kl, s_kl free symbols), not in a state reached by iterating
    from t = 0. That is the claim the document makes -- "no assumption has been made about the
    effect sizes, and it holds for any state whatever" -- and iterating forward from the base
    population could not test it, since every reachable state is already rank-one-ish in beta.
    """
    kk = range(1, n_variants + 1)
    beta = {k: sp.Symbol(f"beta_{k}", real=True) for k in kk}
    rho_y, V_E = sp.symbols("rho_y V_E", positive=True)
    half, quarter = sp.Rational(1, 2), sp.Rational(1, 4)

    # a free symmetric state; s_kk is not a symbol -- it is pinned at 1/2 by eq:skk
    sym = {}
    for k in kk:
        for lidx in kk:
            key = (min(k, lidx), max(k, lidx))
            sym.setdefault(("a",) + key, sp.Symbol(f"a_{key[0]}{key[1]}", real=True))
            if k != lidx:
                sym.setdefault(("s",) + key, sp.Symbol(f"s_{key[0]}{key[1]}", real=True))
    A = lambda k, l: sym[("a", min(k, l), max(k, l))]
    S = lambda k, l: half if k == l else sym[("s", min(k, l), max(k, l))]

    c = {k: sum(beta[l] * (S(k, l) + A(k, l)) for l in kk) for k in kk}     # eq:c-def
    V_A = 2 * sum(beta[k] * c[k] for k in kk)                              # eq:VA-from-as
    mu = rho_y / (V_A + V_E)
    # the document deliberately does NOT name this rho_g in the equilibrium section -- it keeps
    # everything in V_A, V_P and rho_y -- so the checks compare against the expanded ratio.
    inflate = rho_y * V_A / (V_A + V_E)

    a_next = {(k, l): mu * c[k] * c[l] for k in kk for l in kk}            # eq:a-recursion
    s_next = lambda k, l: half if k == l else half * (S(k, l) + A(k, l))   # eq:s-recursion, eq:skk

    checked = 0
    # the two identities eq:c-part-a leans on, checked before the step that uses them
    assert sp.simplify(sum(beta[k] * c[k] for k in kk) - V_A / 2) == 0
    assert sp.simplify(mu * V_A - inflate) == 0
    checked += 2

    for k in kk:
        part_a = sum(beta[l] * a_next[(k, l)] for l in kk)                 # eq:c-part-a
        assert sp.simplify(part_a - half * inflate * c[k]) == 0, f"(i) at k={k}"
        part_s = sum(beta[l] * s_next(k, l) for l in kk)                   # eq:c-part-s
        want_s = half * c[k] + quarter * beta[k] - half * beta[k] * A(k, k)
        assert sp.simplify(part_s - want_s) == 0, f"(ii) at k={k}"
        # eq:c-recursion itself, assembled from eq:c-def one generation on
        c_next = sum(beta[l] * (s_next(k, l) + a_next[(k, l)]) for l in kk)
        want = half * (1 + inflate) * c[k] + quarter * beta[k] - half * beta[k] * A(k, k)
        assert sp.simplify(sp.together(c_next - want)) == 0, f"eq:c-recursion at k={k}"
        checked += 3

    # the readings the prose gives of the result
    for k in kk:
        # the bracket in eq:c-recursion IS eq:seg-var-a, not a new quantity
        seg_var = quarter - half * A(k, k)
        assert sp.simplify(quarter * beta[k] - half * beta[k] * A(k, k)
                           - beta[k] * seg_var) == 0
        checked += 1
    # random mating: rho_y = 0 leaves c_k = beta_k / 2 a fixed point
    assert sp.simplify((half * beta[1] / 2 + quarter * beta[1]) - half * beta[1]) == 0
    # a null variant stays null forever, so causal vs non-causal needs no separate treatment.
    # Checked by iterating the recursion with beta_k = 0 rather than by inspecting it: the
    # claim is about every generation, and c enters a_kl multiplicatively.
    null = {k: (0.0 if k == 1 else 0.4 + 0.1 * k) for k in kk}
    c_num = {k: null[k] / 2 for k in kk}
    for _ in range(50):
        V_A_n = 2 * sum(null[k] * c_num[k] for k in kk)
        rg = 0.4 * V_A_n / (V_A_n + 0.7)
        mu_n = 0.4 / (V_A_n + 0.7)
        c_num = {k: 0.5 * (1 + rg) * c_num[k] + null[k] * (0.25 - 0.5 * mu_n * c_num[k] ** 2)
                 for k in kk}
        assert c_num[1] == 0.0, "a null variant must stay null at every generation"
    assert all(c_num[k] > 0 for k in kk if k != 1), "the others must not be null"
    checked += 2
    return checked


def check_state_readouts(n_variants: int = 4) -> int:
    """Assert the restructured section's central claim: c is the state, the rest are readouts.

    The section is now organized around the claim that (c, a_kk) advances on its own and that
    V_A, V_P, a_kl and s_kl are computed FROM it rather than tracked alongside it. That is a
    claim about the whole trajectory, not about any one equation, so it is checked by running
    the two systems side by side for many generations and comparing every readout.

    Also pins eq:c-def and eq:c-gen1 against the path diagrams, since c is now introduced as a
    covariance in its own right rather than as an abbreviation for a sum.
    """
    kk = list(range(1, n_variants + 1))
    betas = {k: 0.62 - 0.09 * (k - 1) for k in kk}          # deliberately unequal
    V_E, rho_y = 0.71, 0.38
    V_A0 = sum(b * b for b in betas.values())
    V_P0 = V_A0 + V_E
    checked = 0

    # -- eq:c-def and eq:c-gen1 against Figure 2's model ---------------------------------------
    subs = {sp.Symbol(f"beta_{k}", real=True): betas[k] for k in kk}
    subs[sp.Symbol("V_E", positive=True)] = V_E
    subs[sp.Symbol("rho_y", real=True)] = rho_y

    def num(expr):
        expr = sp.sympify(expr)
        return float(sp.N(expr.subs({v: subs[v] for v in expr.free_symbols if v in subs})))

    e1 = pm.RAMEngine(pair_offspring_general(n_variants))
    for k in kk:
        # the base-population parents: c^(0)_k = beta_k / 2  (eq:as-gen0)
        for origin in ("mat", "pat"):
            assert abs(num(e1.cov(_z("m", origin, k), "y_m")) - betas[k] / 2) < 1e-12
            checked += 1
        # their generation-1 offspring: eq:c-gen1, and it must not depend on which gamete
        want = betas[k] / 2 * (1 + rho_y * V_A0 / (2 * V_P0))
        for origin in ("mat", "pat"):
            got = num(e1.cov(_z("o1", origin, k), "y_o1"))
            assert abs(got - want) < 1e-12, f"c^(1)_{k} via {origin}: {got} != {want}"
            checked += 1
        # eq:c-def is an expansion of that same covariance over every variant, not just k
        expand = sum(betas[l] * num(e1.cov(_z("o1", "mat", k), f"x_o1{l}")) for l in kk)
        assert abs(expand - want) < 1e-12, f"eq:c-def expansion at k={k}"
        checked += 1
    # eq:VA-from-as at t = 1, against the diagram's own genetic variance
    V_A1 = num(e1.var("g_o1"))
    assert abs(V_A1 - 2 * sum(betas[k] * betas[k] / 2 * (1 + rho_y * V_A0 / (2 * V_P0))
                              for k in kk)) < 1e-12
    assert abs(V_A1 - V_A0 * (1 + rho_y * V_A0 / (2 * V_P0))) < 1e-12
    checked += 2

    # -- the closure claim, over a long trajectory ---------------------------------------------
    # (A) the full system the section derives: two M x M matrices.
    a = {(k, l): 0.0 for k in kk for l in kk}
    s = {(k, l): (0.5 if k == l else 0.0) for k in kk for l in kk}

    def step_as(a, s):
        c = {k: sum(betas[l] * (s[(k, l)] + a[(k, l)]) for l in kk) for k in kk}   # eq:c-def
        V_A = 2 * sum(betas[k] * c[k] for k in kk)                                 # eq:VA-from-as
        mu = rho_y / (V_A + V_E)
        return ({(k, l): mu * c[k] * c[l] for k in kk for l in kk},                # eq:a-recursion
                {(k, l): (0.5 if k == l else 0.5 * (s[(k, l)] + a[(k, l)]))        # eq:s-recursion
                 for k in kk for l in kk}, c, V_A)

    # (B) the state the section claims is sufficient: c and the diagonal of a. No s at all.
    c_b = {k: betas[k] / 2 for k in kk}
    a_kk = {k: 0.0 for k in kk}

    def step_c(c, a_kk):
        V_A = 2 * sum(betas[k] * c[k] for k in kk)
        V_P = V_A + V_E
        nxt = {k: 0.5 * (1 + rho_y * V_A / V_P) * c[k]                             # eq:c-recursion
               + betas[k] * (0.25 - 0.5 * a_kk[k]) for k in kk}
        return nxt, {k: (rho_y / V_P) * c[k] ** 2 for k in kk}, V_A, V_P

    a_history, c_trace = [], []
    for _ in range(150):
        a_history.append(a)
        c_trace.append(dict(c_b))
        a_n, s_n, c_a, V_A_a = step_as(a, s)
        c_b_n, a_kk_n, V_A_b, V_P_b = step_c(c_b, a_kk)
        # c and V_A agree at every generation
        assert max(abs(c_a[k] - c_b[k]) for k in kk) < 1e-12, "c diverged"
        assert abs(V_A_a - V_A_b) < 1e-12, "V_A diverged"
        # every off-diagonal a_kl is a readout of c, not part of the state
        mu_b = rho_y / V_P_b
        assert max(abs(a_n[(k, l)] - mu_b * c_b[k] * c_b[l])
                   for k in kk for l in kk) < 1e-12, "a_kl is not a readout of c"
        a, s, c_b, a_kk = a_n, s_n, c_b_n, a_kk_n
    checked += 3

    # eq:c-closed: the same trajectory with V_A, V_P and a_kk all substituted away, so that only
    # past c values and (beta, rho_y, V_E) appear. Run as a SEPARATE second-order iteration rather
    # than by rearranging the first, since the document's claim is that this form stands alone.
    def V_A_of(cc):
        return 2 * sum(betas[j] * cc[j] for j in kk)

    c_prev, c_cur = None, {k: betas[k] / 2 for k in kk}
    for t in range(150):
        assert max(abs(c_cur[k] - c_trace[t][k]) for k in kk) < 1e-12, f"eq:c-closed at t={t}"
        drag = ({k: 0.0 for k in kk} if c_prev is None else
                {k: rho_y * betas[k] * c_prev[k] ** 2 / (2 * (V_A_of(c_prev) + V_E)) for k in kk})
        c_prev, c_cur = c_cur, {
            k: 0.5 * (1 + 2 * rho_y * sum(betas[j] * c_cur[j] for j in kk)
                      / (V_A_of(c_cur) + V_E)) * c_cur[k] + 0.25 * betas[k] - drag[k]
            for k in kk}
    checked += 1

    # s as the discounted history of a, the form printed in the readout list. OFF-DIAGONAL only:
    # s_kk is pinned at 1/2 by eq:skk and is not governed by eq:s-recursion, so the unrolling
    # does not apply there -- which is the same k = l exception eq:c-part-s has to correct for.
    for k, lidx in ((1, n_variants), (2, 3)):
        unrolled = sum(0.5 ** (j + 1) * a_history[len(a_history) - 1 - j][(k, lidx)]
                       for j in range(len(a_history)))
        assert abs(s[(k, lidx)] - unrolled) < 1e-12, f"s unrolling at {k},{lidx}"
        checked += 1
    return checked


def check_equilibrium() -> int:
    """Assert the equilibrium section: the exact per-variant quadratic, and what the
    [Uniform-inflation] approximation costs.

    Numeric, and with MIXED-SIGN effects: eq:c-eq claims beta_k enters the denominator only as
    beta_k^2, so c_k inherits its sign from the numerator alone. An all-positive spectrum could
    not test that.
    """
    b = {1: 0.62, 2: -0.44, 3: 0.31, 4: -0.18, 5: 0.09}     # deliberately unequal, mixed signs
    kk = list(b)
    V_E, rho_y = 0.71, 0.38
    V_A0 = sum(v * v for v in b.values())
    V_P0 = V_A0 + V_E
    h0 = V_A0 / V_P0
    checked = 0

    # -- the exact fixed point, reached by iterating eq:c-recursion --------------------------
    c = {k: b[k] / 2 for k in kk}
    a_kk = {k: 0.0 for k in kk}
    for _ in range(6000):
        V_A = 2 * sum(b[j] * c[j] for j in kk)
        V_P = V_A + V_E
        nxt = {k: 0.5 * (1 + rho_y * V_A / V_P) * c[k] + b[k] * (0.25 - 0.5 * a_kk[k])
               for k in kk}
        a_kk = {k: (rho_y / V_P) * c[k] ** 2 for k in kk}
        c = nxt
    V_A = 2 * sum(b[j] * c[j] for j in kk)
    V_P = V_A + V_E
    mu, rho_g = rho_y / V_P, rho_y * V_A / V_P
    u = 1 - rho_g

    # eq:c-eq-quadratic -- the fixed point must satisfy it, at every variant
    for k in kk:
        resid = 2 * mu * b[k] * c[k] ** 2 + 2 * u * c[k] - b[k]
        assert abs(resid) < 1e-12, f"eq:c-eq-quadratic at k={k}: {resid}"
        checked += 1
    # eq:c-eq -- the rationalized positive root, and the sign claim
    for k in kk:
        root = b[k] / (u + math.sqrt(u * u + 2 * mu * b[k] ** 2))
        assert abs(root - c[k]) < 1e-12, f"eq:c-eq at k={k}"
        assert (root > 0) == (b[k] > 0), "c_k must inherit the sign of beta_k"
        checked += 2
    # the rho_y = 0 limit stated under eq:c-eq
    for k in kk:
        assert abs(b[k] / (1 + math.sqrt(1.0)) - b[k] / 2) < 1e-15
        checked += 1
    # Summing eq:c-eq over variants must reproduce V_A. The document no longer prints this
    # implicit form -- the prose says only that the M radicals cannot be gathered -- but the
    # identity is what that claim is about, so it stays checked.
    lhs = 2 * sum(b[k] ** 2 / (u + math.sqrt(u * u + 2 * mu * b[k] ** 2)) for k in kk)
    assert abs(lhs - V_A) < 1e-12, "sum of eq:c-eq over variants"
    checked += 1

    # -- the approximation, and the closed forms it buys -------------------------------------
    # eq:rhog-eq, then eq:VA-eq. Checked against the CLOSED recursion (drop the drag term),
    # which is what the approximation actually corresponds to -- not against the exact fixed
    # point, which it only approximates.
    disc = 1 - 4 * rho_y * h0 * (1 - h0)
    assert disc >= 0, "eq:rhog-eq discriminant must be non-negative"
    rho_g_ap = (1 - math.sqrt(disc)) / (2 * (1 - h0))
    assert abs((1 - h0) * rho_g_ap**2 - rho_g_ap + rho_y * h0) < 1e-14   # eq:rhog-quadratic
    V_A_ap = V_A0 / (1 - rho_g_ap)                                       # eq:VA-eq
    # eq:hsq-eq, in BOTH boxed forms plus the intermediate. These are algebraic rewrites of
    # V_A/(V_A+V_E), so a slip in either would leave the V_A checks green -- each is compared
    # against the ratio directly, not against the other.
    h_direct = V_A_ap / (V_A_ap + V_E)
    box1 = V_A0 / (V_A0 + (1 - rho_g_ap) * V_E)
    mid = h0 / (h0 + (1 - rho_g_ap) * (1 - h0))
    box2 = h0 / (1 - rho_g_ap * (1 - h0))
    for label, val in (("eq:hsq-eq box 1", box1), ("eq:hsq-eq middle", mid),
                       ("eq:hsq-eq box 2", box2)):
        assert abs(val - h_direct) < 1e-14, label
        checked += 1
    # the shortened rho_g route: substituting box 2 into rho_g = rho_y h^2 must give the same
    # quadratic the long route gave, so eq:rhog-eq is unchanged by the shortcut
    assert abs(rho_g_ap - rho_y * box2) < 1e-14, "rho_g = rho_y * eq:hsq-eq box 2"
    assert abs(rho_g_ap - rho_g_ap**2 * (1 - h0) - rho_y * h0) < 1e-14, "eq:rhog-quadratic line 2"
    checked += 2
    V_ap = V_A0
    for _ in range(6000):
        rg = rho_y * V_ap / (V_ap + V_E)
        V_ap = 0.5 * (1 + rg) * V_ap + 0.5 * V_A0
    assert abs(V_ap - V_A_ap) < 1e-9, "eq:VA-eq vs the drag-free recursion"
    # self-consistency: rho_g computed from V_A_ap must reproduce rho_g_ap
    assert abs(rho_y * V_A_ap / (V_A_ap + V_E) - rho_g_ap) < 1e-12
    checked += 4
    # the h0 = 1 degenerate case the text calls out
    assert abs((1 - 1.0) * rho_y**2 - rho_y + rho_y * 1.0) < 1e-15
    # ... and that the discriminant is non-negative across the whole admissible square
    for i in range(51):
        for j in range(51):
            ry, hh = i / 50, j / 50
            assert 1 - 4 * ry * hh * (1 - hh) >= -1e-15
    checked += 2

    # -- eq:VA-eq-exact, step by step -------------------------------------------------------
    # Built from the FULL a and s matrices rather than from the closed forms, so the split into
    # "one rule for every pair" plus a diagonal correction is checked as bookkeeping, not assumed.
    a_s, s_s = {}, {}
    aa = {(k, l): 0.0 for k in kk for l in kk}
    ss = {(k, l): (0.5 if k == l else 0.0) for k in kk for l in kk}
    for _ in range(6000):
        cc = {k: sum(b[l] * (ss[(k, l)] + aa[(k, l)]) for l in kk) for k in kk}
        VA = 2 * sum(b[k] * cc[k] for k in kk)
        m = rho_y / (VA + V_E)
        aa = {(k, l): m * cc[k] * cc[l] for k in kk for l in kk}
        ss = {(k, l): (0.5 if k == l else 0.5 * (ss[(k, l)] + aa[(k, l)])) for k in kk for l in kk}
    cc = {k: sum(b[l] * (ss[(k, l)] + aa[(k, l)]) for l in kk) for k in kk}
    VA = 2 * sum(b[k] * cc[k] for k in kk)
    m, rgx = rho_y / (VA + V_E), rho_y * VA / (VA + V_E)
    cov = lambda k, l: 2 * (ss[(k, l)] + aa[(k, l)])          # eq:x-from-as, from the state

    # the ingredients the derivation names
    assert max(abs(ss[(k, l)] - aa[(k, l)]) for k in kk for l in kk if k != l) < 1e-15
    assert max(abs(cov(k, l) - 4 * aa[(k, l)]) for k in kk for l in kk if k != l) < 1e-15
    assert max(abs(cov(k, k) - (1 + 2 * aa[(k, k)])) for k in kk) < 1e-15
    checked += 3
    # the split, term by term
    total = sum(b[k] * b[l] * cov(k, l) for k in kk for l in kk)
    rule = sum(b[k] * b[l] * 4 * aa[(k, l)] for k in kk for l in kk)
    diag = sum(b[k] ** 2 * ((1 + 2 * aa[(k, k)]) - 4 * aa[(k, k)]) for k in kk)
    assert abs(total - VA) < 1e-12 and abs(rule + diag - total) < 1e-12
    # the square, and its collapse to rho_g V_A
    assert abs(rule - 4 * m * sum(b[k] * cc[k] for k in kk) ** 2) < 1e-12
    assert abs(rule - rgx * VA) < 1e-12
    S_exact = sum(b[k] ** 2 * aa[(k, k)] for k in kk)
    assert abs(diag - (V_A0 - 2 * S_exact)) < 1e-12
    # eq:VA-eq-exact solved
    assert abs(VA - (V_A0 - 2 * S_exact) / (1 - rgx)) < 1e-12
    checked += 5

    # eq:VA-eq-drag and eq:VA-eq-correction: the approximated drag, and that keeping it is a
    # strict improvement on eq:VA-eq -- the point of printing the correction at all.
    drag = rgx / (2 * V_A0 * (1 - rgx)) * sum(v**4 for v in b.values())
    plain = V_A0 / (1 - rgx)
    corrected = plain * (1 - rgx / (2 * (1 - rgx)) * sum(v**4 for v in b.values()) / V_A0**2)
    assert abs(corrected - (V_A0 - drag) / (1 - rgx)) < 1e-12, "eq:VA-eq-correction factoring"
    assert abs(corrected - VA) < abs(plain - VA), "the correction must reduce the error"
    checked += 2

    # eq:Me-def and eq:Me-bound: M_e as defined, its relation to O'Connor's 3M/kappa, and the
    # bound by which [Large-Me] delivers the small per-variant share it is also used for.
    # [Small-effects] no longer exists as a separate box -- the two were merged, and the bound is
    # what licenses using either form, so it is checked rather than trusted.
    M_e = V_A0**2 / sum(v**4 for v in b.values())
    n = len(kk)
    E2, E4 = V_A0 / n, sum(v**4 for v in b.values()) / n
    kappa = E4 / E2**2
    assert abs(M_e - n / kappa) < 1e-12, "M_e == M/kappa"
    assert abs(3 * n / kappa - 3 * M_e) < 1e-12, "O'Connor's 3M/kappa is 3x ours"
    max_share = max(v * v for v in b.values()) / V_A0
    assert max_share <= 1 / math.sqrt(M_e) + 1e-12, "eq:Me-bound"
    assert M_e >= 1 / max_share - 1e-9, "the reverse direction claimed in [Large-Me]"
    # equal effects must give M_e = M exactly, as the text claims
    eq_b = {j: math.sqrt(1.0 / 7) for j in range(1, 8)}
    eq_Me = sum(v * v for v in eq_b.values())**2 / sum(v**4 for v in eq_b.values())
    assert abs(eq_Me - 7) < 1e-12, "M_e = M at equal effects"
    checked += 5

    # [VA-inflation]'s justification: the error written through M_e, and its SIGN
    assert abs(rho_g / (2 * (1 - rho_g)) / M_e
               - rho_g / (2 * (1 - rho_g)) * sum(v**4 for v in b.values()) / V_A0**2) < 1e-14
    assert V_A0 / (1 - rho_g) > V_A, "dropping the term must OVERSTATE V_A, as the box claims"
    checked += 2

    # -- the justification's error term -------------------------------------------------------
    # relative error in V_A of order rho_g / (2(1-rho_g) M_e), with M_e the effective count.
    M_e = V_A0**2 / sum(v**4 for v in b.values())
    predicted = rho_g / (2 * (1 - rho_g) * M_e)
    actual = (V_A_ap - V_A) / V_A
    assert actual > 0, "the approximation should overestimate V_A"
    assert 0.5 < actual / predicted < 2.0, f"error term off by {actual / predicted:.2f}x"
    checked += 2
    return checked


def check_g_transmit(n_variants: int = 2) -> int:
    """Assert eq:g-transmit and eq:g-seg-var -- the drop from alleles to genetic values.

    The claim is that g_o = g_m/2 + g_p/2 + eps_o is an EXACT linear model, so the checks take
    the expectation over all 2^M Bernoulli configurations per parent explicitly, with the allele
    second moments (a_kl, s_kl, and the cross-mate covariances) left as free symbols. Iterating
    the recursion to equilibrium would not test this: the independence claims must hold for an
    arbitrary state, and a state reached from t = 0 is a special one.

    M = 2 is enough: every claim is about the independence of coins, and two variants already
    give a same-variant pair, a cross-variant pair, a cross-parent pair and a cross-sib pair.
    The symbolic cost grows like 4^M, so a larger M buys nothing and costs minutes.
    """
    import itertools

    ks = list(range(1, n_variants + 1))
    half = sp.Rational(1, 2)
    zm = {k: sp.Symbol(f"zm{k}") for k in ks}      # mother, maternal gamete
    zp = {k: sp.Symbol(f"zp{k}") for k in ks}      # mother, paternal gamete
    wm = {k: sp.Symbol(f"wm{k}") for k in ks}      # father, maternal gamete
    wp = {k: sp.Symbol(f"wp{k}") for k in ks}
    beta = {k: sp.Symbol(f"beta_{k}", real=True) for k in ks}
    A = {(min(k, l), max(k, l)): sp.Symbol(f"a_{min(k,l)}{max(k,l)}") for k in ks for l in ks}
    S = {(min(k, l), max(k, l)): sp.Symbol(f"s_{min(k,l)}{max(k,l)}")
         for k in ks for l in ks if k != l}
    X = {(k, l): sp.Symbol(f"x_{k}{l}") for k in ks for l in ks}   # cross-mate

    def moment(p1, u1, k, p2, u2, l):
        if p1 != p2:                                   # different individuals
            return X[(k, l)]
        if k == l:
            return half if u1 == u2 else A[(k, k)]
        return S[(min(k, l), max(k, l))] if u1 == u2 else A[(min(k, l), max(k, l))]

    sub = {}
    for k in ks:
        for l in ks:
            for n1, d1, pp1, uu1 in (("zm", zm, 0, "m"), ("zp", zp, 0, "p"),
                                     ("wm", wm, 1, "m"), ("wp", wp, 1, "p")):
                for n2, d2, pp2, uu2 in (("zm", zm, 0, "m"), ("zp", zp, 0, "p"),
                                         ("wm", wm, 1, "m"), ("wp", wp, 1, "p")):
                    sub[d1[k] * d2[l]] = moment(pp1, uu1, k, pp2, uu2, l)
    E = lambda e: sp.expand(e).subs(sub)

    cfgs = list(itertools.product([0, 1], repeat=n_variants))
    epsM = lambda c: {k: (sp.Rational(c[i]) - half) * (zm[k] - zp[k]) for i, k in enumerate(ks)}
    epsP = lambda c: {k: (sp.Rational(c[i]) - half) * (wm[k] - wp[k]) for i, k in enumerate(ks)}
    eps_o = lambda c1, c2: sum(beta[k] * (epsM(c1)[k] + epsP(c2)[k]) for k in ks)
    g_m = sum(beta[k] * (zm[k] + zp[k]) for k in ks)
    g_p = sum(beta[k] * (wm[k] + wp[k]) for k in ks)
    avg2 = lambda f: sum(f(c1, c2) for c1 in cfgs for c2 in cfgs) / len(cfgs) ** 2
    checked = 0

    # the three bulleted properties, in the order the document lists them
    for parent in (g_m, g_p):
        assert sp.expand(avg2(lambda a, b: E(eps_o(a, b) * parent))) == 0
        checked += 1
    for k in ks:                                       # different variants, same parent
        for l in ks:
            if k == l:
                continue
            v = sum(E(epsM(c)[k] * epsM(c)[l]) for c in cfgs) / len(cfgs)
            assert sp.expand(v) == 0, f"cross-variant at {k},{l}"
            checked += 1
    # the two parents, and two sibs (independent coin draws each)
    assert sp.expand(avg2(lambda a, b: E(sum(beta[k]*epsM(a)[k] for k in ks)
                                           * sum(beta[k]*epsP(b)[k] for k in ks)))) == 0
    # two sibs: their coin draws are independent, so E[eps_1 eps_2] = E[eps_1] E[eps_2] and it
    # suffices that each has mean zero. Checked that way rather than by a 2^(4M) double sum.
    mean_eps = avg2(lambda a, b: sp.expand(eps_o(a, b)))
    assert sp.expand(mean_eps) == 0, "each sib's residual must have mean zero"
    checked += 2

    # eq:g-seg-var, and that ONE parent's share is half of it -- the factor that is easy to drop
    var_eps = avg2(lambda a, b: E(eps_o(a, b) ** 2))
    want = sum(beta[k] ** 2 * (half - A[(k, k)]) for k in ks)
    assert sp.expand(var_eps - want) == 0, "eq:g-seg-var"
    one = sum(E(sum(beta[k] * epsM(c)[k] for k in ks) ** 2) for c in cfgs) / len(cfgs)
    assert sp.expand(2 * one - want) == 0, "each parent contributes exactly half"
    checked += 2

    # eq:g-transmit itself: g_o built from eq:transmit must equal the boxed decomposition
    for c1 in cfgs[:2]:
        for c2 in cfgs[:2]:
            z_om = {k: sp.Rational(c1[i]) * zm[k] + (1 - sp.Rational(c1[i])) * zp[k]
                    for i, k in enumerate(ks)}
            z_op = {k: sp.Rational(c2[i]) * wm[k] + (1 - sp.Rational(c2[i])) * wp[k]
                    for i, k in enumerate(ks)}
            g_o = sum(beta[k] * (z_om[k] + z_op[k]) for k in ks)
            assert sp.expand(g_o - (g_m / 2 + g_p / 2 + eps_o(c1, c2))) == 0
            checked += 1
    return checked


def check_reduced_figure() -> int:
    """The reduced diagram must reproduce what the full three-generation pedigree gives."""
    model = reduced_pair_offspring()
    assert not [i for i in model.validate() if i.severity == "error"]
    engine = pm.RAMEngine(model)
    b1, b2, V_E, rho_y = (model.sym(s) for s in ("beta_1", "beta_2", "V_E", "rho_y"))
    betas = {1: b1, 2: b2}
    V_A0 = b1**2 + b2**2
    V_P0 = V_A0 + V_E
    checked = 0
    # generation-1 individuals: alleles still variance 1/2, genotypes now inflated by alpha^(1)
    for who in ("m", "p"):
        for k in (1, 2):
            assert engine.var(_z(who, "mat", k)) == sp.Rational(1, 2)
            alpha1 = betas[k] ** 2 * rho_y / (2 * V_P0)
            assert sp.simplify(engine.var(f"x_{who}{k}") - (1 + alpha1)) == 0
            checked += 2
    # V_A(1) must come out right, which is what the cross-variant edges buy
    V_A1 = V_A0 * (1 + rho_y * V_A0 / (2 * V_P0))
    assert sp.simplify(engine.var("g_m") - V_A1) == 0, engine.var("g_m")
    # [Uniform-inflation] at t = 1, exactly
    for k in (1, 2):
        assert sp.simplify(engine.cov(f"x_m{k}", "y_m") - betas[k] * V_A1 / V_A0) == 0
        checked += 1
    # the generation-2 offspring: alleles STILL variance 1/2 (the (1-alpha)/4 residual), and
    # alpha^(2) matching eq:alpha-recursion-VA
    V_P1 = V_A1 + V_E
    for k in (1, 2):
        assert sp.simplify(engine.var(_z("o", "mat", k)) - sp.Rational(1, 2)) == 0
        want = rho_y * betas[k] ** 2 * (V_A1 / V_A0) ** 2 / (2 * V_P1)
        got = 2 * engine.cov(_z("o", "mat", k), _z("o", "pat", k))
        assert sp.simplify(got - want) == 0, f"alpha^(2)_{k}: {sp.simplify(got)}"
        checked += 2

    # eq:VA-from-as and eq:hsq-t at t = 1, in BOTH forms the document prints: the ratio of
    # variance components, and the same thing written out in betas and c's.
    c1 = {k: engine.cov(_z("m", "mat", k), "y_m") for k in (1, 2)}
    from_c = 2 * sum(betas[k] * c1[k] for k in (1, 2))
    assert sp.simplify(from_c - V_A1) == 0, "eq:VA-from-as boxed form at t = 1"
    assert sp.simplify(V_A1 / V_P1 - from_c / (from_c + V_E)) == 0, "eq:hsq-t in betas and c's"
    checked += 2

    # eq:rhog-t at t = 1, derived rather than assumed: the document builds it from the co-path
    # rule plus Cov[g_i, y_i] = V_A, at general t. This checks it at the first generation where
    # V_A and V_P have actually moved, so it is not vacuous, and in every form printed.
    # both mates have genetic variance V_A^(t), which is the step that turns the correlation's
    # sqrt(Var Var) denominator into a plain V_A^(t) -- asserted rather than left to sympy, which
    # will not reduce sqrt(V_A^2) without a positivity assumption on the symbols.
    assert sp.simplify(engine.var("g_m") - V_A1) == 0
    assert sp.simplify(engine.var("g_p") - V_A1) == 0
    rho_g1 = engine.cov("g_m", "g_p") / V_A1
    assert sp.simplify(rho_g1 - (rho_y / V_P1) * V_A1) == 0, "rho_g = mu V_A"
    assert sp.simplify(rho_g1 - rho_y * V_A1 / V_P1) == 0, "rho_g = rho_y V_A / V_P"
    assert sp.simplify(rho_g1 - rho_y * (V_A1 / V_P1)) == 0, "rho_g = rho_y h^2"
    # the intermediate the derivation leans on: a genetic value against its own phenotype is V_A
    for who in ("m", "p"):
        assert sp.simplify(engine.cov(f"g_{who}", f"y_{who}") - V_A1) == 0
    checked += 7
    return checked


# ----------------------------------------------------------------------------------------------
# Figure 4: the genetic-value-level diagram. Alleles and genotypes are gone; what is left is
# eq:g-transmit -- two 1/2 edges into each offspring's genetic value and a disturbance carrying
# eq:g-seg-var. Everything is at equilibrium, so V_A and rho_g are the equilibrium values and the
# diagram must reproduce Var[g_o] = V_A on its own.
# ----------------------------------------------------------------------------------------------

#: the equilibrium disturbance on an offspring's genetic value, (1/2) V_A (1 - rho_g), written in
#: the model's own free parameters so that pathMgr derives rho_g rather than being handed it.
_G_SEG = "(V_A*(1 - rho_y*V_A/(V_A+V_E))/2)"


def g_only_pair_offspring() -> pm.Model:
    """Genetic values only. The environment is not drawn as its own node: since e enters only
    through y and is uncorrelated with everything else, it is equivalent -- and smaller -- to
    carry V_E as exogenous variance ON the phenotype. Var[y] = V_A + V_E either way."""
    kids = ("o1", "o2")
    lines = [
        "latent: " + ", ".join(f"g_{w}" for w in ("m", "p") + kids),
        "positive: V_A, V_E",
        "real: rho_y",
        *[f"label: {v}_{w} = ${v}_{w}$" for w in ("m", "p") for v in "gy"],
        *[f"label: {v}_{k} = ${v}_{{{k[0]}_{k[1]}}}$" for k in kids for v in "gy"],
        # founders: the equilibrium genetic variance is exogenous to the diagram
        "g_m ~~ V_A*g_m",
        "g_p ~~ V_A*g_p",
        # eq:g-transmit, twice: each offspring gets 1/2 from each parent plus its own disturbance
        *[f"g_{k} ~ 1/2*g_m + 1/2*g_p" for k in kids],
        *[f"g_{k} ~~ {_G_SEG}*g_{k}" for k in kids],
        # the environment, folded into the phenotype as its own disturbance
        *[f"y_{w} ~ g_{w}" for w in ("m", "p") + kids],
        *[f"y_{w} ~~ V_E*y_{w}" for w in ("m", "p") + kids],
        "y_m -- [rho_y]*y_p",
    ]
    return pm.from_text("\n".join(lines), name="genetic values only, at equilibrium")


# Each phenotype sits directly below its own genetic value, so the four g -> y edges are vertical
# and a reader can read each person as a column. This is the arrangement the figure wants, and it
# was briefly not available: pathMgr used to draw every self-loop ABOVE its node, which put each
# parent's V_A label on the phenotype above it, so the phenotypes had to be offset sideways to get
# out of the way. Loops now choose a free side per node, and the offset is no longer needed.
# Measured on the way back: this layout has no collisions and no ambiguous labels, and it places
# the two offspring disturbances SYMMETRICALLY (loops mirrored at 180/0, each with a leader),
# which the sideways version did not -- there g_o1 sat inline and g_o2 was pushed out.
G_ONLY_LAYOUT = Layout({
    "y_m": (2.0, 5.2), "y_p": (8.0, 5.2),
    "g_m": (2.0, 3.4), "g_p": (8.0, 3.4),
    # the two disturbance labels are wide; this separation is what lets both step out sideways
    "g_o1": (3.3, 1.6), "g_o2": (6.7, 1.6),
    "y_o1": (3.3, 0.0), "y_o2": (6.7, 0.0),
})


def check_g_only_figure() -> int:
    """Assert the genetic-value diagram reproduces the allele-level results at equilibrium.

    The disturbance is written in the model's own parameters, so Var[g_o] = V_A is a real test:
    if eq:g-seg-var had the wrong factor the diagram would not be stationary, and that is exactly
    the error the author's first draft made (one parent's share instead of both).
    """
    model = g_only_pair_offspring()
    assert not [i for i in model.validate() if i.severity == "error"], model.validate()
    e = pm.RAMEngine(model)
    V_A, V_E, rho_y = (model.sym(s) for s in ("V_A", "V_E", "rho_y"))
    V_P = V_A + V_E
    rho_g = rho_y * V_A / V_P
    checked = 0

    # eq:rhog-t, at the genetic-value level
    assert sp.simplify(e.cov("g_m", "g_p") - rho_g * V_A) == 0
    # stationarity: an offspring's genetic variance must come back to V_A
    for k in ("o1", "o2"):
        assert sp.simplify(e.var(f"g_{k}") - V_A) == 0, f"Var[g_{k}]"
        assert sp.simplify(e.var(f"y_{k}") - V_P) == 0
        checked += 2
    # the two first-degree covariances, and that they coincide at the GENETIC level
    for a, b in (("g_o1", "g_o2"), ("g_m", "g_o1"), ("g_p", "g_o1")):
        assert sp.simplify(e.cov(a, b) - V_A * (1 + rho_g) / 2) == 0, f"Cov[{a},{b}]"
        checked += 1
    # ... but NOT at the phenotypic level: parent-offspring carries rho_y, siblings rho_g.
    # This is the asymmetry tab:relatives-gen1 records, and it must survive the reduction.
    assert sp.simplify(e.cov("y_o1", "y_o2") - V_A * (1 + rho_g) / 2) == 0
    assert sp.simplify(e.cov("y_m", "y_o1") - V_A * (1 + rho_y) / 2) == 0
    assert sp.simplify(e.cov("y_m", "y_o1") - e.cov("y_o1", "y_o2")) != 0
    checked += 4

    # Fisher's law, as Eftedal et al. state it: correlations fall by (1+rho_g)/2 per degree.
    # Checked by tracing one extra generation rather than by asserting the closed form.
    h2 = V_A / V_P
    assert sp.simplify(e.cov("y_o1", "y_o2") / V_P - h2 * (1 + rho_g) / 2) == 0
    assert sp.simplify(e.cov("g_m", "g_m") - V_A) == 0
    checked += 2

    # the disturbance string really is eq:g-seg-var. sympify must be given the MODEL's symbols,
    # not fresh ones -- pathMgr declares V_A and V_E positive, and a bare sympify would build
    # unconstrained duplicates that never cancel.
    assert sp.simplify(sp.sympify(_G_SEG, locals={"V_A": V_A, "V_E": V_E, "rho_y": rho_y})
                       - V_A * (1 - rho_g) / 2) == 0
    checked += 1
    return checked


# ----------------------------------------------------------------------------------------------
# Section 2.3: expected covariance between relatives, by (modified) degree.
#
# Every model here is at equilibrium and genetic-value level, and every co-path is declared by its
# RAW coefficient mu rather than by a correlation. That is not a style choice: as soon as one
# individual has two mates -- which is every half-relative and every in-law -- the correlation form
# cannot be resolved, because the second co-path changes the variance the first one was normalised
# against. pathMgr raises CoPathVarianceError rather than returning a wrong number.
# ----------------------------------------------------------------------------------------------

_MU = "(rho_y/(V_A+V_E))"
_SEG_G = "(V_A*(1 - rho_y*V_A/(V_A+V_E))/2)"      # eq:g-seg-var, in the model's own parameters


def _pedigree(founders, kids, matings, name):
    """Build a genetic-value-level pedigree at equilibrium and return (model, engine).

    founders: names whose genetic variance is exogenous (V_A).
    kids:     (child, parent, parent) triples, each carrying eq:g-transmit and eq:g-seg-var.
    matings:  (a, b) pairs joined by a co-path of raw coefficient mu.
    """
    people = list(founders) + [c for c, _, _ in kids]
    lines = [
        "positive: V_A, V_E",
        "real: rho_y",
        "latent: " + ", ".join(f"g_{p}" for p in people),
        *[f"g_{f} ~~ V_A*g_{f}" for f in founders],
    ]
    for c, a, b in kids:
        lines += [f"g_{c} ~ 1/2*g_{a} + 1/2*g_{b}", f"g_{c} ~~ {_SEG_G}*g_{c}"]
    for n in people:
        lines += [f"y_{n} ~ g_{n}", f"y_{n} ~~ V_E*y_{n}"]
    lines += [f"y_{a} -- {_MU}*y_{b}" for a, b in matings]
    model = pm.from_text("\n".join(lines), name=name)
    bad = [i for i in model.validate() if i.severity == "error"]
    assert not bad, (name, bad)
    return model, pm.RAMEngine(model)


#: A step-sibling pedigree, which is the smallest one exhibiting every mating count the section
#: uses. Four parents in a row, joined by three matings, and one child per mating. Reading left to
#: right, p1--p2 are the parents of o1, p2--p3 of o3, and p3--p4 of o2.
#:
#: Giving the MIDDLE mating a child is what makes the figure show the classes it is cited for
#: rather than only the mating counts. o3 shares p2 with o1 and p3 with o2, so (o1,o3) and (o2,o3)
#: are half-sibling pairs -- and the n=2 pairs (p1,p3) and (p2,p4) are then exactly the "non-shared
#: parents of a half-sibling pair" the text uses as its n=2 example, visible in the same picture
#: instead of asserted about an absent one. It also makes o1 and o2 step-siblings in the sense
#: Eftedal et al. operationalise (pairs not related by blood with a half-sibling in common), which
#: the two-child version did not.
STEP_SIB_PARENTS = ("p1", "p2", "p3", "p4")
STEP_SIB_MATINGS = (("p1", "p2"), ("p2", "p3"), ("p3", "p4"))
STEP_SIB_KIDS = (("o1", "p1", "p2"), ("o3", "p2", "p3"), ("o2", "p3", "p4"))


def step_sib_pedigree():
    return _pedigree(STEP_SIB_PARENTS, STEP_SIB_KIDS, STEP_SIB_MATINGS, "step-sibling pedigree")


# ----------------------------------------------------------------------------------------------
# Section 2.3.1: the mean-parent device, and the decomposition of a same-generation covariance
# through the two mean-parents.
#
# Everything here is a linear combination of a pedigree's nodes rather than a node, so the checks
# go through _cov_lc rather than engine.cov. Covariance is bilinear and the engine returns exact
# symbolic entries, so this is not an approximation -- it is the same arithmetic the text does.
# ----------------------------------------------------------------------------------------------

def _cov_lc(e, lc1, lc2):
    """Cov of two linear combinations, each a dict {node name: coefficient}."""
    return sp.expand(sum(c1 * c2 * e.cov(n1, n2)
                         for n1, c1 in lc1.items() for n2, c2 in lc2.items()))


def _mp(kind, a, b):
    """eq:mean-parent: the mean-parent of the couple (a, b), at level `kind`."""
    return {f"{kind}_{a}": sp.Rational(1, 2), f"{kind}_{b}": sp.Rational(1, 2)}


def _nmu(matings, a, b):
    """N_mu: fewest matings connecting a and b, or sympy.oo when no chain of matings does."""
    adj = collections.defaultdict(set)
    for x, y in matings:
        adj[x].add(y)
        adj[y].add(x)
    dist, queue = {a: 0}, collections.deque([a])
    while queue:
        u = queue.popleft()
        for v in adj[u] - dist.keys():
            dist[v] = dist[u] + 1
            queue.append(v)
    return dist.get(b, sp.oo)


def _disjoint_couples(k):
    """Two disjoint couples (a1,a2) and (b1,b2) joined by a chain of k matings.

    a2 and b1 are the near parents, so N_mu(a2,b1) = k and eq:nmu-bar-tree predicts the other
    three. A and B are one child of each couple.
    """
    between = [f"c{i}" for i in range(1, k)]                 # k-1 people strictly inside
    run = ["a2"] + between + ["b1"]
    matings = tuple([("a1", "a2"), ("b1", "b2")]
                    + [(run[i], run[i + 1]) for i in range(len(run) - 1)])
    founders = tuple(["a1", "a2", "b1", "b2"] + between)
    kids = (("A", "a1", "a2"), ("B", "b1", "b2"))
    return _pedigree(founders, kids, matings, f"disjoint couples k={k}"), matings


# ----------------------------------------------------------------------------------------------
# Section 2.3.2: the set-theoretic pedigree notation, and what
# [Single-recent-mating-chain] actually excludes.
#
# These are claims about the PEDIGREE rather than about the model, so they are checked on the
# pedigree directly and then tied back to pathMgr: the assumption earns its place only if the
# closed form is exact exactly when it holds. Function names below mirror the document's symbols.
# ----------------------------------------------------------------------------------------------

def _generations(founders, kids):
    """t_X for every individual. `founders` may be a tuple (all at generation 0) or a dict.

    A married-in founder has to be placed alongside their mate rather than at the root, because
    [Discrete-generations] puts the two parents of a child in the same generation -- which this
    asserts rather than assumes.
    """
    g = dict(founders) if isinstance(founders, dict) else {f: 0 for f in founders}
    pending = list(kids)
    while pending:
        rest = []
        for c, a, b in pending:
            if a in g and b in g:
                assert g[a] == g[b], f"parents of {c} span two generations"
                g[c] = g[a] + 1
            else:
                rest.append((c, a, b))
        assert len(rest) < len(pending), "pedigree has a cycle in descent"
        pending = rest
    return g


def _anc(parents, gen, X, t):
    """A^(t)(X): climb one generation at a time, exactly as the text's recursion does."""
    cur = {X}
    for _ in range(gen[X] - t):
        cur = {q for y in cur for q in parents.get(y, ())}
    return cur


def _mu(matings, X):
    """mu(X): one step of mating, INCLUDING X itself."""
    return {X}.union(*[{a, b} for a, b in matings if X in (a, b)]) if any(
        X in (a, b) for a, b in matings) else {X}


def _mu_pow(matings, X, k):
    """mu^k(X): k steps of mating."""
    cur = {X}
    for _ in range(k):
        cur = set().union(*[_mu(matings, y) for y in cur])
    return cur


def _Mu(matings, X):
    """M(X): arbitrarily many steps -- iterate mu to a fixed point."""
    cur, prev = {X}, None
    while cur != prev:
        prev = cur
        cur = set().union(*[_mu(matings, y) for y in cur])
    return cur


def _Mubar(parents, matings, gen, X, Y, t):
    """Mbar^(t)(X,Y): everyone of generation t on a mating chain holding an ancestor of each."""
    aX, aY = _anc(parents, gen, X, t), _anc(parents, gen, Y, t)
    return {Z for Z in gen if gen[Z] == t
            and _Mu(matings, Z) & aX and _Mu(matings, Z) & aY}


def _single_recent_mating_chain(parents, matings, gen, A, B):
    """Return (t_AB, holds) for [Single-recent-mating-chain], or (None, False) if unconnected."""
    for t in range(min(gen[A], gen[B]), -1, -1):
        M = _Mubar(parents, matings, gen, A, B, t)
        if not M:
            continue
        two = (len(_anc(parents, gen, A, t) & M) == 2
               and len(_anc(parents, gen, B, t) & M) == 2)
        # the document's tree test: half the summed mating degrees equals |Mhat| - 1
        edges = sum(len(_mu(matings, Z)) - 1 for Z in M) / 2
        return t, (two and edges == len(M) - 1)
    return None, False


def check_pedigree_sets() -> int:
    """Assert the Section 2.3.2 definitions, and that the assumption is the right dividing line.

    Three claims are worth stating as assertions rather than as prose:

      - N_mu(X,Y) = min{k : Y in mu^k(X)}, which is what ties the mu / M notation to the mating
        distance the closed forms are written in.
      - Mbar^(t)(X,Y) is closed under mating. The tree test counts each member's matings as
        |mu(Z)| - 1, which is only the degree WITHIN the chain if the chain contains every mate
        of every member; without closure that count would silently include outside matings.
      - the payoff: across the battery, eq:mp-expand is exact **exactly** when the assumption
        holds and t_AB is the parental generation. That is what makes it necessary and
        sufficient rather than merely sufficient.
    """
    eq = lambda a, b: sp.simplify(sp.together(sp.expand(a - b))) == 0
    checked = 0

    hs_m = (("Q1", "P"), ("P", "Q2"))
    ss_m = (("p1", "p2"), ("p2", "p3"), ("p3", "p4"))
    dj_m = (("p1", "p2"), ("p2", "x"), ("x", "p3"), ("p3", "p4"))
    fc_m = (("G1", "G2"), ("b1", "w1"), ("b2", "w2"))
    dfc_m = (("G1", "G2"), ("H1", "H2"), ("b1", "s1"), ("b2", "s2"))
    cyc_m = (("A1", "B1"), ("B1", "C1"), ("C1", "D1"), ("D1", "A1"))
    rem_m = (("a1", "a2"), ("b1", "b2"), ("a2", "b1"), ("a1", "b2"))
    # (name, founders, kids, matings, assumption holds, generations up to t_AB)
    battery = (
        ("full sibs", ("m", "p"), (("A", "m", "p"), ("B", "m", "p")), (("m", "p"),), True, 1),
        ("half sibs", ("P", "Q1", "Q2"), (("A", "P", "Q1"), ("B", "P", "Q2")), hs_m, True, 1),
        ("step sibs", ("p1", "p2", "p3", "p4"),
         (("A", "p1", "p2"), ("B", "p3", "p4")), ss_m, True, 1),
        ("disjoint k=2", ("p1", "p2", "x", "p3", "p4"),
         (("A", "p1", "p2"), ("B", "p3", "p4")), dj_m, True, 1),
        ("first cousins", {"G1": 0, "G2": 0, "w1": 1, "w2": 1},
         (("b1", "G1", "G2"), ("b2", "G1", "G2"), ("A", "b1", "w1"), ("B", "b2", "w2")),
         fc_m, True, 2),
        ("double first cousins", ("G1", "G2", "H1", "H2"),
         (("b1", "G1", "G2"), ("b2", "G1", "G2"), ("s1", "H1", "H2"), ("s2", "H1", "H2"),
          ("A", "b1", "s1"), ("B", "b2", "s2")), dfc_m, False, 2),
        ("mating 4-cycle", ("A1", "B1", "C1", "D1"),
         (("A", "A1", "B1"), ("B", "C1", "D1"), ("x1", "B1", "C1"), ("x2", "D1", "A1")),
         cyc_m, False, 1),
        ("cross-remarriage", ("a1", "a2", "b1", "b2"),
         (("A", "a1", "a2"), ("B", "b1", "b2")), rem_m, False, 1),
    )

    for name, founders, kids, matings, want_holds, want_up in battery:
        parents = {c: (a, b) for c, a, b in kids}
        gen = _generations(founders, kids)
        t_ab, holds = _single_recent_mating_chain(parents, matings, gen, "A", "B")
        assert t_ab is not None, f"{name}: t_AB undefined"
        assert holds == want_holds, f"{name}: assumption {holds}, expected {want_holds}"
        assert gen["A"] - t_ab == want_up, f"{name}: t_AB is {gen['A']-t_ab} generations up"
        checked += 3

        Mhat = _Mubar(parents, matings, gen, "A", "B", t_ab)
        # closure under mating, which the tree test depends on
        assert all(_mu(matings, Z) <= Mhat for Z in Mhat), f"{name}: Mhat not closed under mu"
        # X in mu(X) in mu^k(X) in M(X), and mu^0 is the singleton
        for Z in Mhat:
            assert _mu_pow(matings, Z, 0) == {Z}, f"{name}: mu^0"
            assert _mu(matings, Z) <= _Mu(matings, Z), f"{name}: mu subset M"
            assert _mu_pow(matings, Z, len(gen)) == _Mu(matings, Z), f"{name}: mu^k reaches M"
        checked += 4
        # N_mu is the fewest steps of mu, for every same-generation pair
        for x in gen:
            for y in gen:
                if gen[x] != gen[y]:
                    continue
                steps = next((j for j in range(len(gen) + 1) if y in _mu_pow(matings, x, j)), sp.oo)
                assert _nmu(matings, x, y) == steps, f"{name}: N_mu({x},{y})"
        checked += 1

        # ---- the payoff: eq:mp-expand exact <=> assumption holds AND t_AB = t_A - 1
        model, e = _pedigree(tuple(founders), kids, matings, name)
        V_A, V_E, rho_y = (model.sym(v) for v in ("V_A", "V_E", "rho_y"))
        rho_g = rho_y * V_A / (V_A + V_E)
        mA, pA = parents["A"]
        mB, pB = parents["B"]
        total = 0
        for x in (mA, pA):
            for y in (mB, pB):
                k = _nmu(matings, x, y)          # shortest chain, per [Shortest-chain]
                total += V_A if k == 0 else (0 if k is sp.oo else rho_g * rho_y ** (k - 1) * V_A)
        exact = eq(e.cov("y_A", "y_B"), total / 4)
        assert exact == (holds and gen["A"] - t_ab == 1), \
            f"{name}: eq:mp-expand exact={exact}, assumption={holds}, up={gen['A']-t_ab}"
        checked += 1
    return checked


def check_nmu_bar() -> int:
    """Assert eq:nmu-bar through tab:nmu-classes, and the two boxes that bound them.

    Three claims here are combinatorial rather than model claims, and are asserted on the mating
    graph directly: that two disjoint couples joined by one chain always give the four distances
    (k, k+1, k+1, k+2); that the mean is therefore exactly k+1 (it steps up by ONE per additional
    mating -- not, as one might guess from 1/2, 1, 2, by doubling); and that N_mu-bar >= 2 is the
    same condition as the two couples being disjoint, which is what lets eq:mp-split-pheno state
    its side condition in N_mu-bar.

    The rest are model claims checked against pathMgr, including the two that bound the framework:

      - [Shortest-chain]: chains contribute ADDITIVELY, so its error term is exact arithmetic
        rather than an estimate. Asserted on three multi-chain pedigrees, including the worst
        case of two equal-length chains where the truth is exactly double what is kept.
      - the scope limit: when the two couples are joined by DESCENT rather than by matings
        (first cousins), every N_mu is infinite and eq:mp-expand returns zero against a nonzero
        truth. A second application of eq:mp-reduce recovers it exactly, which is the claim the
        closing paragraph makes.
    """
    eq = lambda a, b: sp.simplify(sp.together(sp.expand(a - b))) == 0
    checked = 0

    cases = [("the same couple", ("m", "p"), ("m", "p"),
              _pedigree(("m", "p"), (("A", "m", "p"), ("B", "m", "p")), (("m", "p"),), "same"),
              (("m", "p"),)),
             ("share one parent", ("P", "Q1"), ("P", "Q2"),
              _pedigree(("P", "Q1", "Q2"), (("A", "P", "Q1"), ("B", "P", "Q2")),
                        (("Q1", "P"), ("P", "Q2")), "shared"),
              (("Q1", "P"), ("P", "Q2")))]
    for k in range(1, 6):
        ped, matings = _disjoint_couples(k)
        cases.append((f"disjoint k={k}", ("a1", "a2"), ("b1", "b2"), ped, matings))

    for name, cA, cB, (model, e), matings in cases:
        V_A, V_E, rho_y = (model.sym(s) for s in ("V_A", "V_E", "rho_y"))
        V_P = V_A + V_E
        h2, rho_g = V_A / V_P, rho_y * V_A / V_P
        ds = sorted(_nmu(matings, x, y) for x in cA for y in cB)
        nbar = sp.Rational(sum(ds), 4)
        got = e.cov("y_A", "y_B")
        shared = len(set(cA) & set(cB))

        # eq:nmu-bar-tree, and that N_mu-bar >= 2 iff the couples are disjoint
        if shared == 0:
            k = ds[0]
            assert ds == [k, k + 1, k + 1, k + 2], f"{name}: distances {ds}"
            assert nbar == k + 1, f"{name}: eq:nmu-bar-tree"
            assert nbar >= 2, f"{name}: disjoint must give nbar >= 2"
            checked += 3
        else:
            assert nbar < 2, f"{name}: sharing a parent must give nbar < 2"
            assert nbar == (sp.Rational(1, 2) if shared == 2 else 1), f"{name}: nbar"
            checked += 2

        # tab:nmu-classes, one row per case
        row = {2: V_A * (1 + rho_g) / 2,
               1: V_A * (1 + 2 * rho_g + rho_g * rho_y) / 4,
               0: V_A * rho_g * rho_y ** (nbar - 2) * (1 + rho_y) ** 2 / 4}[shared]
        assert eq(got, row), f"{name}: tab:nmu-classes row"
        checked += 1
        # eq:nmu-law is that row for every disjoint case, stated in nbar rather than in k
        if shared == 0:
            assert eq(got, V_A * rho_g * rho_y ** (nbar - 2) * (1 + rho_y) ** 2 / 4), \
                f"{name}: eq:nmu-law"
            checked += 1
        # the (1-h^2) per shared parent claim: continuing eq:mating-chain down to N_mu = 0 would
        # pay h^2*V_A where the truth pays V_A, and that gap is the whole difference between the
        # three rows.
        assert eq(got, V_A * (h2 * sum(rho_y ** d for d in ds) + shared * (1 - h2)) / 4), \
            f"{name}: shared-parent correction"
        checked += 1
        # eq:mp-split-pheno's side condition, now stated as nbar >= 2
        mpA = {f"y_{x}": sp.Rational(1, 2) for x in cA}
        mpB = {f"y_{x}": sp.Rational(1, 2) for x in cB}
        pheno = h2 * _cov_lc(e, mpA, mpB) * h2
        assert eq(got, pheno) == (nbar >= 2), f"{name}: eq:mp-split-pheno iff nbar >= 2"
        checked += 1

    # ---- [Shortest-chain]: chains add, so the approximation's error term is exact.
    two_chain = (
        # (name, founders, matings, the chain lengths joining u and v)
        ("two length-2 chains", ("u", "v", "x", "z"),
         (("u", "x"), ("x", "v"), ("u", "z"), ("z", "v")), (2, 2)),
        ("lengths 2 and 3", ("u", "v", "x", "z1", "z2"),
         (("u", "x"), ("x", "v"), ("u", "z1"), ("z1", "z2"), ("z2", "v")), (2, 3)),
        ("lengths 1 and 3", ("u", "v", "w1", "w2"),
         (("u", "v"), ("u", "w1"), ("w1", "w2"), ("w2", "v")), (1, 3)),
    )
    for name, founders, matings, ks in two_chain:
        model, e = _pedigree(founders, (), matings, name)
        V_A, V_E, rho_y = (model.sym(s) for s in ("V_A", "V_E", "rho_y"))
        rho_g = rho_y * V_A / (V_A + V_E)
        got = e.cov("g_u", "g_v")
        assert eq(got, V_A * sum(rho_g * rho_y ** (j - 1) for j in ks)), f"{name}: chains add"
        # the fraction [Shortest-chain] drops
        k1, k2 = min(ks), max(ks)
        kept = V_A * rho_g * rho_y ** (k1 - 1)
        assert eq((got - kept) / got, rho_y ** (k2 - k1) / (1 + rho_y ** (k2 - k1))), \
            f"{name}: dropped fraction"
        checked += 2
    # ---- the scope limit: couples joined by DESCENT, not by matings (first cousins).
    # b1,b2 are siblings; b1 marries w1 and b2 marries w2; A and B are their children.
    matings = (("G1", "G2"), ("b1", "w1"), ("b2", "w2"))
    model, e = _pedigree(("G1", "G2", "w1", "w2"),
                         (("b1", "G1", "G2"), ("b2", "G1", "G2"),
                          ("A", "b1", "w1"), ("B", "b2", "w2")), matings, "first cousins")
    got = e.cov("y_A", "y_B")
    assert all(_nmu(matings, x, y) is sp.oo for x in ("b1", "w1") for y in ("b2", "w2")), \
        "cousins: every cross pair must be unreachable by matings"
    assert got != 0, "cousins: the truth is not zero"
    checked += 2
    # eq:mp-reduce applied once, then again at each parent, recovers it exactly
    assert eq(got, _cov_lc(e, _mp("g", "b1", "w1"), _mp("g", "b2", "w2"))), "cousins: level 1"
    recursed = (_cov_lc(e, _mp("g", "G1", "G2"), _mp("g", "b2", "w2"))
                + _cov_lc(e, {"g_w1": 1}, _mp("g", "b2", "w2"))) / 2
    assert eq(got, recursed), "cousins: level 2"
    checked += 2
    return checked


def check_mean_parent() -> int:
    """Assert eq:mean-parent through eq:mp-expand, plus the two claims that bound them.

    The two bounding claims are the ones worth stating as assertions rather than as prose, because
    each is a place where a plausible-looking statement is false:

      - eq:mp-reduce holds only where BOTH e_A and eps_A drop out. Against a proband's own MATE it
        fails, since assortment matched them on the whole phenotype; that failure is asserted, not
        just described.
      - eq:mp-split-pheno, the form with mean-parent PHENOTYPES and two factors of h^2, is exact
        when the two couples are disjoint and false when they share a parent. Both directions are
        asserted on real pedigrees, so the side condition in the text is pinned rather than hoped.
    """
    eq = lambda a, b: sp.simplify(sp.together(sp.expand(a - b))) == 0
    checked = 0

    # ---- eq:mean-parent-var, eq:g-partition and tab:mean-parent-cov, on the figure they are
    # read off (fig:g-only) rather than on a rebuilt copy of it.
    model = g_only_pair_offspring()
    e = pm.RAMEngine(model)
    V_A, V_E, rho_y = (model.sym(s) for s in ("V_A", "V_E", "rho_y"))
    V_P = V_A + V_E
    h2, rho_g = V_A / V_P, rho_y * V_A / V_P
    G, Y = _mp("g", "m", "p"), _mp("y", "m", "p")
    # e_mp = y_mp - g_mp, which is what eq:mean-parent asserts by writing y_mp = g_mp + e_mp
    E = {**{k: sp.Rational(1, 2) for k in ("y_m", "y_p")},
         **{k: sp.Rational(-1, 2) for k in ("g_m", "g_p")}}

    assert eq(_cov_lc(e, G, G), V_A * (1 + rho_g) / 2), "Var[g_mp]"
    assert eq(_cov_lc(e, Y, Y), V_P * (1 + rho_y) / 2), "Var[y_mp]"
    assert eq(_cov_lc(e, G, Y), V_A * (1 + rho_y) / 2), "Cov[g_mp,y_mp]"
    assert eq(_cov_lc(e, E, G), V_A * (rho_y - rho_g) / 2), "Cov[e_mp,g_mp]"
    checked += 4
    # tab:mean-parent-cov, all four covariance cells. The two right-hand columns are equal
    # because e_o is exogenous; promoting the mean-parent is what changes rho_g into rho_y.
    for z in ("g_o1", "y_o1"):
        assert eq(_cov_lc(e, G, {z: 1}), V_A * (1 + rho_g) / 2), f"Cov[g_mp,{z}]"
        assert eq(_cov_lc(e, Y, {z: 1}), V_A * (1 + rho_y) / 2), f"Cov[y_mp,{z}]"
        checked += 2
    # eq:g-transmit-mp, in the only two forms that can be checked without naming eps_o:
    # the mean-parent's genetic variance IS its covariance with the child, and adding the
    # residual returns V_A (eq:g-partition).
    assert eq(_cov_lc(e, G, {"g_o1": 1}), _cov_lc(e, G, G)), "eq:g-transmit-mp"
    assert eq(_cov_lc(e, G, G) + V_A * (1 - rho_g) / 2, V_A), "eq:g-partition"
    # the mid-parent regression is h^2 exactly, with no residual rho_y
    assert eq(_cov_lc(e, Y, {"y_o1": 1}) / _cov_lc(e, Y, Y), h2), "midparent regression"
    checked += 3
    # eq:mp-reduce FAILS against a proband's own mate, which is half of its side condition.
    # o1 has no mate in fig:g-only, so this is asserted where a mate exists: m against p.
    assert not eq(e.cov("y_m", "y_p"), _cov_lc(e, _mp("g", "m", "p"), {"y_p": 1})), \
        "eq:mp-reduce must fail against a mate"
    checked += 1

    # ---- eq:mp-reduce, eq:mp-split, eq:mating-chain, eq:mp-expand and eq:mp-split-pheno,
    # across the sibling classes of tab:nmu-classes and two longer mating chains.
    half_matings = (("Q1", "P"), ("P", "Q2"))
    chain_matings = tuple((f"p{i}", f"p{i+1}") for i in range(1, 6))
    cases = (
        ("full sibs (tab:nmu-classes r1)", V_A * (2 + 2 * rho_g) / 4,
         _pedigree(("m", "p"), (("o1", "m", "p"), ("o2", "m", "p")), (("m", "p"),), "full sibs"),
         (("m", "p"),), ("o1", "m", "p"), ("o2", "m", "p")),
        ("half sibs (tab:nmu-classes r2)", V_A * (1 + 2 * rho_g + rho_g * rho_y) / 4,
         _pedigree(("P", "Q1", "Q2"), (("A", "P", "Q1"), ("B", "P", "Q2")), half_matings,
                   "half sibs"),
         half_matings, ("A", "P", "Q1"), ("B", "P", "Q2")),
        ("half sibs on fig:step-sib", V_A * (1 + 2 * rho_g + rho_g * rho_y) / 4,
         step_sib_pedigree(), STEP_SIB_MATINGS, ("o1", "p1", "p2"), ("o3", "p2", "p3")),
        ("step sibs (tab:nmu-classes r3)", V_A * rho_g * (1 + rho_y) ** 2 / 4,
         step_sib_pedigree(), STEP_SIB_MATINGS, ("o1", "p1", "p2"), ("o2", "p3", "p4")),
        # a five-mating chain, so eq:mating-chain is exercised past the counts in the figures
        ("N_mu = 3 chain", V_A * rho_g * rho_y ** 2 * (1 + rho_y) ** 2 / 4,
         _pedigree(tuple(f"p{i}" for i in range(1, 7)),
                   (("A", "p1", "p2"), ("B", "p5", "p6")), chain_matings, "long chain"),
         chain_matings, ("A", "p1", "p2"), ("B", "p5", "p6")),
    )

    for name, want, (model, e), matings, (A, mA, pA), (B, mB, pB) in cases:
        GA, GB = _mp("g", mA, pA), _mp("g", mB, pB)
        YA, YB = _mp("y", mA, pA), _mp("y", mB, pB)
        got = e.cov(f"y_{A}", f"y_{B}")
        # eq:mp-reduce, at every admissible Z the pedigree offers on B's side
        for z in (f"y_{B}", f"g_{B}", f"y_{mB}", f"g_{mB}", f"y_{pB}", f"g_{pB}"):
            assert eq(e.cov(f"y_{A}", z), _cov_lc(e, GA, {z: 1})), f"{name}: reduce y_A vs {z}"
            assert eq(e.cov(f"g_{A}", z), _cov_lc(e, GA, {z: 1})), f"{name}: reduce g_A vs {z}"
            checked += 2
        # eq:mp-split
        assert eq(got, _cov_lc(e, GA, GB)), f"{name}: eq:mp-split"
        assert eq(got, e.cov(f"g_{A}", f"g_{B}")), f"{name}: g and y ends agree"
        checked += 2
        # eq:mating-chain, one cross pair at a time, and then eq:mp-expand over the four
        total = 0
        for x in (mA, pA):
            for y in (mB, pB):
                k = _nmu(matings, x, y)
                f = 1 if k == 0 else (0 if k is sp.oo else rho_g * rho_y ** (k - 1))
                assert eq(e.cov(f"g_{x}", f"g_{y}"), f * V_A), \
                    f"{name}: eq:mating-chain ({x},{y}) at N_mu = {k}"
                total += f
                checked += 1
        assert eq(got, V_A * total / 4), f"{name}: eq:mp-expand"
        assert eq(got, want), f"{name}: closed form"
        checked += 2
        # eq:mp-split-pheno: exact for disjoint couples, and FALSE when they share a parent.
        pheno = h2 * _cov_lc(e, YA, YB) * h2
        if {mA, pA} & {mB, pB}:
            assert not eq(got, pheno), f"{name}: eq:mp-split-pheno must fail (shared parent)"
        else:
            assert eq(got, pheno), f"{name}: eq:mp-split-pheno"
        checked += 1
    return checked


#: Every person is a column: phenotype above the genetic value for the parents, below it for the
#: children, matching Figure 4. The three co-paths are the three horizontal segments along the top.
#:
#: The two spacings are both load-bearing. `_DX` has to be at least 3.8 or each child's
#: segregation label, which is wide, touches the two parent nodes above it (4 label-node
#: collisions at 3.4). It also has to be no MORE than that: the picture is 13.26cm wide at 3.8
#: and this is an upright figure inside a `floatcard`, so on A4 with 1in margins there is about
#: 15cm to play with once the box is drawn. That is the lesson from Figure 2, where a
#: collision-free placement pushed labels outside the node span and made the page overfull -- the
#: collision metric cannot see a bounding box, so the width is chosen here rather than discovered
#: in the LaTeX log. Dropping the children to 1.8 rather than Figure 4's 1.6 is what buys the
#: clearance at the narrower spacing.
_DX = 3.8
STEP_SIB_LAYOUT = Layout({
    **{f"y_{w}": (_DX * i, 5.2) for i, w in enumerate(STEP_SIB_PARENTS)},
    **{f"g_{w}": (_DX * i, 3.4) for i, w in enumerate(STEP_SIB_PARENTS)},
    "g_o3": (1.5 * _DX, 1.8),
    "y_o3": (1.5 * _DX, 0.0),
    "g_o1": (_DX / 2, 1.8), "g_o2": (2.5 * _DX, 1.8),
    "y_o1": (_DX / 2, 0.0), "y_o2": (2.5 * _DX, 0.0),
})


def check_step_sib_figure() -> int:
    """Assert every claim the step-sibling figure's caption makes about it.

    The caption lists which pairs sit at each mating count, so the caption is a set of assertions
    and this is the test of it: a relabelled or re-wired pedigree would break here rather than
    quietly ship a figure whose caption is wrong.
    """
    model, e = step_sib_pedigree()
    V_A, V_E, rho_y = (model.sym(s) for s in ("V_A", "V_E", "rho_y"))
    V_P = V_A + V_E
    h2, rho_g = V_A / V_P, rho_y * V_A / V_P
    checked = 0

    # the pedigree is at equilibrium: every genetic value has variance V_A, every phenotype V_P.
    # The children are the real test -- their variance is 1/2+1/2 plus eq:g-seg-var, and it only
    # comes out at V_A because the parents were mates rather than independent.
    for w in STEP_SIB_PARENTS + ("o1", "o2", "o3"):
        assert sp.simplify(e.var(f"g_{w}") - V_A) == 0, f"Var[g_{w}]"
        assert sp.simplify(e.var(f"y_{w}") - V_P) == 0, f"Var[y_{w}]"
        checked += 2

    # n = 1: the three mated pairs themselves
    for a, b in STEP_SIB_MATINGS:
        assert sp.simplify(e.cov(f"g_{a}", f"g_{b}") - rho_g * V_A) == 0, f"n=1 {a},{b}"
        checked += 1
    # n = 2: two people who share a mate. p1 and p3 share p2; p2 and p4 share p3.
    for a, b in (("p1", "p3"), ("p2", "p4")):
        assert sp.simplify(e.cov(f"g_{a}", f"g_{b}") - rho_g * rho_y * V_A) == 0, f"n=2 {a},{b}"
        checked += 1
    # n = 3: the two outermost parents, connected by the whole chain of three matings
    assert sp.simplify(e.cov("g_p1", "g_p4") - rho_g * rho_y**2 * V_A) == 0, "n=3 p1,p4"
    checked += 1

    # the step-siblings themselves: an in-law pair at N_m = 2, per the in-law row of
    # tab:degree-classes
    inlaw = h2 * rho_g * ((1 + rho_y) / 2) ** 2
    assert sp.simplify(e.cov("y_o1", "y_o2") / V_P - inlaw) == 0, "step-sibs"
    checked += 1
    # and they are NOT what the plain degree law would give at d = 2, which is the point of the
    # class existing at all
    assert sp.simplify(e.cov("y_o1", "y_o2") / V_P - h2 * ((1 + rho_g) / 2) ** 2) != 0
    checked += 1

    # The middle child makes two HALF-SIBLING pairs: o3 shares p2 with o1 and p3 with o2. These
    # are what the n=2 pairs above are for -- (p1,p3) are the non-shared parents of (o1,o3), and
    # (p2,p4) of (o2,o3) -- so asserting both here ties the mating count to the class it prices.
    half = h2 * ((1 + rho_g) / 2) ** 2 * (1 + 2 * rho_g + rho_y * rho_g) / (1 + rho_g) ** 2
    for a, b, unshared in (("o1", "o3", ("p1", "p3")), ("o3", "o2", ("p2", "p4"))):
        assert sp.simplify(e.cov(f"y_{a}", f"y_{b}") / V_P - half) == 0, f"half-sibs {a},{b}"
        # What the half-relative multiplier is worth, exactly. The plain degree law does not treat
        # the unshared parents as unrelated -- it implicitly gives them rho_g^2 V_A, the value they
        # would have if separated by two GENETIC legs. The excess is therefore the difference
        # between their true n=2 covariance and that, carried down one meiosis each (the 1/4).
        # This is the half turn factor of eq:turn-K stated as an equation: replace rho_g*rho_y by
        # rho_g^2 and the excess is zero, i.e. the half factor collapses onto the full one.
        plain = h2 * ((1 + rho_g) / 2) ** 2
        excess = sp.simplify(e.cov(f"y_{a}", f"y_{b}") / V_P - plain)
        true_n2 = e.cov(f"g_{unshared[0]}", f"g_{unshared[1]}")
        assert sp.simplify(excess - (true_n2 - rho_g**2 * V_A) / (4 * V_P)) == 0, \
            f"half-sib excess is the n=2 chain's departure from rho_g^2, {a},{b}"
        checked += 2
    # o1 and o2 are still step-siblings, not half-siblings: no shared parent, so their value is
    # the in-law form above and not `half`. Pinned so a re-wiring cannot make them siblings.
    assert sp.simplify(e.cov("y_o1", "y_o2") / V_P - half) != 0, "o1,o2 must not be half-sibs"
    checked += 1

    # A step-PARENT pair: p3 is mated to o1's parent p2 but is not o1's parent. This is the
    # N_m = 1 case that the in-law row excludes, and it is worth pinning because it confirms
    # WHY the exclusion is there. One side of the connecting mating has a step and the other has
    # none, so exactly ONE factor of (1+rho_y)/2 appears -- not the two of the in-law row. That is
    # the p_i of eq:segment-full, and the count of
    # those factors is the number of sides carrying at least one step, which is what makes
    # dtilde = 2 the smallest case with the symmetric squared form.
    for kid, step_parent in (("o1", "p3"), ("o2", "p2")):
        expected = rho_g * V_A * (1 + rho_y) / 2
        assert sp.simplify(e.cov(f"g_{kid}", f"g_{step_parent}") - expected) == 0, \
            f"step-parent {kid},{step_parent}"
        # and NOT the all-rho_g form, which is the error the in-law row guards against
        assert sp.simplify(e.cov(f"g_{kid}", f"g_{step_parent}") - rho_g * V_A * (1 + rho_g) / 2) != 0
        checked += 2

    return checked


def check_one_step_rules() -> int:
    """Assert the two relations behind eq:chain-parent, and that their condition is necessary.

    Both are boxed with a precondition -- no chain from i to P may cross any of P's matings -- and
    a precondition nobody tests is a precondition nobody believes. So this checks a case where it
    holds and a case where it does not, and pins the wrong answer that follows from ignoring it.
    """
    checked = 0

    # P and Pm are mates with two children A and B; B mates Bm and they have a child j.
    # i = A is a SIBLING of B, so no chain from A to B crosses a mating of B: condition holds.
    model, e = _pedigree(
        ("P", "Pm", "Bm"),
        (("A", "P", "Pm"), ("B", "P", "Pm"), ("j", "B", "Bm")),
        (("P", "Pm"), ("B", "Bm")),
        "sibling pair, one with a child",
    )
    V_A, V_E, rho_y = (model.sym(s) for s in ("V_A", "V_E", "rho_y"))
    V_P = V_A + V_E
    rho_g = rho_y * V_A / V_P

    # the condition itself, stated as the thing it actually means: Cov[g_A, e_B] = 0
    assert sp.simplify(e.cov("g_A", "y_B") - e.cov("g_A", "g_B")) == 0, "Cov[g_A, e_B] != 0"
    checked += 1
    base = e.cov("g_A", "g_B")
    # crossing B's mating scales A's covariance by rho_g
    assert sp.simplify(e.cov("g_A", "g_Bm") - rho_g * base) == 0, "crossing a mating costs rho_g"
    checked += 1
    # descending to B's child scales it by (1+rho_g)/2 -- this is eq:chain-parent
    assert sp.simplify(e.cov("g_A", "g_j") - (1 + rho_g) / 2 * base) == 0, "eq:chain-parent"
    checked += 1

    # NOW THE CONDITION VIOLATED, which is the case the text works through. o1 is a child of p2
    # by p1, so o1 reaches p3 both through p2's phenotype AND through p1, who was matched to that
    # phenotype. Cov[g_o1, e_p2] is therefore nonzero and the rho_g scaling does not apply.
    m2, e2 = step_sib_pedigree()
    V_A2, V_E2, rho_y2 = (m2.sym(s) for s in ("V_A", "V_E", "rho_y"))
    V_P2 = V_A2 + V_E2
    rho_g2 = rho_y2 * V_A2 / V_P2
    assert sp.simplify(e2.cov("g_o1", "y_p2") - e2.cov("g_o1", "g_p2")) != 0, \
        "the condition should FAIL for a child of P"
    checked += 1
    # the true value, and the value the rule would have given -- the two the document prints
    truth = rho_g2 * (1 + rho_y2) * V_A2 / 2
    naive = rho_g2 * (1 + rho_g2) * V_A2 / 2
    assert sp.simplify(e2.cov("g_o1", "g_p3") - truth) == 0, "step-child true value"
    assert sp.simplify(e2.cov("g_o1", "g_p2") * rho_g2 - naive) == 0, "step-child naive value"
    assert sp.simplify(truth - naive) != 0, "the two must differ"
    checked += 3

    return checked


def check_path_framework() -> int:
    """Assert the boxed results of the rewritten Section 2.3: segments, matings, assembly.

    The section is built on three claims -- a segment's value factorises as
    ((1+rho_g)/2)^N_m * K, a path factorises as mu^n times its segments, and the g-versus-y
    distinction lives only at the outer ends -- and each is asserted here against pathMgr on
    pedigrees built to exhibit it.
    """
    checked = 0

    def parts(model):
        V_A, V_E, ry = (model.sym(s) for s in ("V_A", "V_E", "rho_y"))
        V_P = V_A + V_E
        return V_A, V_E, ry, V_P, ry / V_P, ry * V_A / V_P, V_A / V_P   # .., mu, rho_g, h2

    def lineage(n):
        f, k, mt = ["c0"], [], []
        cur = "c0"
        for s in range(1, n + 1):
            m2 = f"m{s}"
            f.append(m2); mt.append((cur, m2)); k.append((f"a{s}", cur, m2)); cur = f"a{s}"
        return _pedigree(tuple(f), tuple(k), tuple(mt), f"lineage{n}"), "c0", f"a{n}"

    def coll(u, w, full):
        if full:
            f, mt = ["S", "Sm"], [("S", "Sm")]
            k = [("a1", "S", "Sm"), ("b1", "S", "Sm")]
        else:
            f, mt = ["S", "Ma", "Mb"], [("S", "Ma"), ("S", "Mb")]
            k = [("a1", "S", "Ma"), ("b1", "S", "Mb")]
        for side, steps in (("a", u), ("b", w)):
            cur = f"{side}1"
            for s in range(2, steps + 1):
                m2 = f"{side}m{s}"
                f.append(m2); mt.append((cur, m2)); k.append((f"{side}{s}", cur, m2)); cur = f"{side}{s}"
        return _pedigree(tuple(f), tuple(k), tuple(mt), "coll"), f"a{u}", f"b{w}"

    # -- eq:mated-pair: the three ways to take a mated pair's covariance -------------------
    model, e = _pedigree(("m", "p"), (), (("m", "p"),), "mated pair")
    V_A, V_E, ry, V_P, mu, rg, h2 = parts(model)
    assert sp.simplify(e.cov("g_m", "g_p") - rg * V_A) == 0
    assert sp.simplify(e.cov("g_m", "y_p") - ry * V_A) == 0
    assert sp.simplify(e.cov("y_m", "y_p") - ry * V_P) == 0
    assert sp.simplify(e.cov("y_m", "y_p") / V_P - ry) == 0          # eq:mate-corr
    checked += 4

    # -- eq:segment and eq:turn-K, for all three kinds of turn ----------------------------
    for n in (1, 2, 3, 4):
        (model, e), i, j = lineage(n)
        V_A, V_E, ry, V_P, mu, rg, h2 = parts(model)
        assert sp.simplify(e.cov(f"g_{i}", f"g_{j}") - V_A * ((1 + rg) / 2) ** n) == 0, f"lineal {n}"
        checked += 1
    for full in (True, False):
        K_name = "full" if full else "half"
        for u, w in ((1, 1), (1, 2), (2, 2), (2, 3)):
            (model, e), i, j = coll(u, w, full)
            V_A, V_E, ry, V_P, mu, rg, h2 = parts(model)
            K = 2 / (1 + rg) if full else (1 + 2 * rg + ry * rg) / (1 + rg) ** 2
            want = V_A * ((1 + rg) / 2) ** (u + w) * K
            assert sp.simplify(e.cov(f"g_{i}", f"g_{j}") - want) == 0, f"{K_name} {u},{w}"
            checked += 1

    # -- eq:assemble + eq:segment-full ----------------------------------------------------
    # p_i counts a segment's mating-facing ends at which the individual is a PARENT on it.
    SHAPES = (
        ("step sibs: lineal-M-lineal, both ends parents",
         (("p1", "p2", "p3", "p4"), (("o1", "p1", "p2"), ("o2", "p3", "p4")),
          (("p1", "p2"), ("p2", "p3"), ("p3", "p4"))), "o1", "o2",
         1, ((1, "lineal", 1), (1, "lineal", 1))),
        ("full-M-lineal: one end a child, one a parent",
         (("C1", "C2", "Y", "Z"), (("A", "C1", "C2"), ("X", "C1", "C2"), ("B", "Y", "Z")),
          (("C1", "C2"), ("X", "Y"), ("Y", "Z"))), "A", "B",
         1, ((2, "full", 0), (1, "lineal", 1))),
        ("full-M-full: both ends children",
         (("C1", "C2", "D1", "D2"),
          (("A", "C1", "C2"), ("X", "C1", "C2"), ("Y", "D1", "D2"), ("B", "D1", "D2")),
          (("C1", "C2"), ("D1", "D2"), ("X", "Y"))), "A", "B",
         1, ((2, "full", 0), (2, "full", 0))),
        ("half-M-lineal",
         (("S", "Ma", "Mb", "Z", "W"),
          (("A", "S", "Ma"), ("X", "S", "Mb"), ("B", "Z", "W")),
          (("S", "Ma"), ("S", "Mb"), ("X", "Z"), ("Z", "W"))), "A", "B",
         1, ((2, "half", 0), (1, "lineal", 1))),
        ("two matings, middle segment with one promoted end",
         (("q1", "q2", "q4", "q4m", "q5", "q5m"),
          (("a", "q1", "q2"), ("q3", "q4", "q4m"), ("b", "q5", "q5m")),
          (("q1", "q2"), ("q2", "q3"), ("q4", "q4m"), ("q4", "q5"), ("q5", "q5m"))), "a", "b",
         2, ((1, "lineal", 1), (1, "lineal", 1), (1, "lineal", 1))),
    )
    for label, spec, x, y, n_mat, segs in SHAPES:
        model, e = _pedigree(*spec, label)
        V_A, V_E, ry, V_P, mu, rg, h2 = parts(model)
        Ks = {"lineal": 1, "full": 2 / (1 + rg), "half": (1 + 2 * rg + ry * rg) / (1 + rg) ** 2}
        pred = mu**n_mat
        for Nm, kind, pi in segs:
            pred *= V_A * Ks[kind] * ((1 + ry) / 2) ** pi * ((1 + rg) / 2) ** (Nm - pi)
        assert sp.simplify(e.cov(f"g_{x}", f"g_{y}") - pred) == 0, label
        checked += 1

    # -- eq:passthrough: an interior one-person segment contributes V_P, not V_A ----------
    # i and j both mated to s, and to nobody else on the path
    model, e = _pedigree(("i", "s", "j"), (), (("i", "s"), ("s", "j")), "share a mate")
    V_A, V_E, ry, V_P, mu, rg, h2 = parts(model)
    assert sp.simplify(e.cov("g_i", "g_j") - mu**2 * V_A * V_P * V_A) == 0
    assert sp.simplify(e.cov("g_i", "g_j") - rg * ry * V_A) == 0        # the n=2 mating rule
    assert sp.simplify(e.cov("g_i", "g_j") - mu**2 * V_A * V_A * V_A) != 0, "V_P, not V_A"
    checked += 3

    # -- Section 2.3.7: g vs y differs only where the end's own mating is on the path -----
    same = (("full sibs", (("S", "Sm"), (("a", "S", "Sm"), ("b", "S", "Sm")), (("S", "Sm"),)), "a", "b"),
            ("step sibs", (("p1", "p2", "p3", "p4"),
                           (("o1", "p1", "p2"), ("o2", "p3", "p4")),
                           (("p1", "p2"), ("p2", "p3"), ("p3", "p4"))), "o1", "o2"))
    for label, spec, x, y in same:
        model, e = _pedigree(*spec, label)
        assert sp.simplify(e.cov(f"g_{x}", f"g_{y}") - e.cov(f"y_{x}", f"y_{y}")) == 0, label
        checked += 1
    # lineal: the ancestor's own mating IS on the path, so exactly one factor is promoted
    for n in (1, 2, 3):
        (model, e), i, j = lineage(n)
        V_A, V_E, ry, V_P, mu, rg, h2 = parts(model)
        assert sp.simplify(e.cov(f"g_{i}", f"g_{j}") - e.cov(f"y_{i}", f"y_{j}")) != 0
        want = h2 * ((1 + rg) / 2) ** (n - 1) * ((1 + ry) / 2)          # eq:lineal-pheno
        assert sp.simplify(e.cov(f"y_{i}", f"y_{j}") / V_P - want) == 0, f"lineal pheno {n}"
        checked += 2

    return checked


def check_degree_classes() -> int:
    """Assert every closed form Section 2.3 states, class by class, against traced pedigrees.

    Each class is built as its own pedigree rather than carved out of one big one, so a mistake in
    one cannot mask a mistake in another, and so the mating structure of each class is explicit.
    """
    checked = 0

    def syms(m):
        V_A, V_E, ry = (m.sym(s) for s in ("V_A", "V_E", "rho_y"))
        V_P = V_A + V_E
        return V_A, V_E, ry, V_P, V_A / V_P, ry * V_A / V_P, ry / V_P   # .., h2, rho_g, mu

    # -- the mating rule: crossing k matings costs rho_g rho_y^(k-1) V_A ---------------------
    chain = [(f"X{i}", f"X{i+1}") for i in range(1, 5)]
    m, e = _pedigree([f"X{i}" for i in range(1, 6)], [], chain, "mating chain")
    V_A, V_E, ry, V_P, h2, rg, mu = syms(m)
    for k in (1, 2, 3, 4):
        got = e.cov("g_X1", f"g_X{1+k}")
        assert sp.simplify(got - rg * ry**(k-1) * V_A) == 0, f"mating rule at k={k}"
        checked += 1
    # the two leg values, which are what generate rho_g vs rho_y
    assert sp.simplify(e.cov("g_X1", "g_X2") - mu * V_A * V_A) == 0      # g leg, g leg
    assert sp.simplify(e.cov("y_X1", "g_X2") - mu * V_P * V_A) == 0      # y leg, g leg
    assert sp.simplify(ry / rg - V_P / V_A) == 0                          # rho_y = rho_g / h^2
    checked += 3

    # -- collateral full relatives: h2 * rate^d, and phenotypic == genetic -------------------
    # A-B mated; sibs S1,S2; S1 marries C -> K1; S2 marries E -> K2; K1 marries H -> GC
    m, e = _pedigree(
        ["A", "B", "C", "E", "H"],
        [("S1", "A", "B"), ("S2", "A", "B"), ("K1", "S1", "C"), ("K2", "S2", "E"),
         ("GC", "K1", "H")],
        [("A", "B"), ("S1", "C"), ("S2", "E"), ("K1", "H")],
        "collateral and direct",
    )
    V_A, V_E, ry, V_P, h2, rg, mu = syms(m)
    rate = (1 + rg) / 2
    for a, b, d in (("S1", "S2", 1), ("S2", "K1", 2), ("K1", "K2", 3)):
        assert sp.simplify(e.cov(f"g_{a}", f"g_{b}") - V_A * rate**d) == 0, f"collateral g {a},{b}"
        # no environmental cross term for a collateral pair, so the phenotypic correlation is
        # the genetic covariance over V_P -- this is the claim that makes the degree law clean
        assert sp.simplify(e.cov(f"y_{a}", f"y_{b}") - e.cov(f"g_{a}", f"g_{b}")) == 0
        assert sp.simplify(e.cov(f"y_{a}", f"y_{b}") / V_P - h2 * rate**d) == 0
        checked += 3

    # -- direct relatives: exactly ONE factor becomes (1+rho_y)/2 ----------------------------
    for a, b, d in (("S1", "K1", 1), ("S1", "GC", 2)):
        # the GENETIC covariance still follows the plain degree law ...
        assert sp.simplify(e.cov(f"g_{a}", f"g_{b}") - V_A * rate**d) == 0, f"direct g {a},{b}"
        # ... but the phenotypic one does not, and the excess is exactly one swapped factor
        want = h2 * rate**(d - 1) * (1 + ry) / 2
        assert sp.simplify(e.cov(f"y_{a}", f"y_{b}") / V_P - want) == 0, f"direct y {a},{b}"
        assert sp.simplify(e.cov(f"y_{a}", f"y_{b}") / V_P - h2 * rate**d) != 0
        checked += 3

    # -- half siblings: Nagylaki, and NOT the plain degree-2 law -----------------------------
    # P2 has children with P1 and with P3; the two children are half-sibs through P2
    m, e = _pedigree(["P1", "P2", "P3"], [("H1", "P1", "P2"), ("H2", "P3", "P2")],
                     [("P1", "P2"), ("P2", "P3")], "half sibs")
    V_A, V_E, ry, V_P, h2, rg, mu = syms(m)
    rate = (1 + rg) / 2
    # the two unshared parents are co-mates of P2, and THAT is the whole of the multiplier
    assert sp.simplify(e.cov("g_P1", "g_P3") - ry * rg * V_A) == 0, "co-mates = rho_y rho_g V_A"
    assert sp.simplify(e.cov("g_H1", "g_H2") - V_A * (1 + 2*rg + ry*rg) / 4) == 0, "Nagylaki"
    # Nagylaki's form as a multiplier on the plain degree-2 law
    assert sp.simplify(e.cov("g_H1", "g_H2") / V_P
                       - h2 * rate**2 * (1 + 2*rg + ry*rg) / (1 + rg)**2) == 0
    # half-sibs are collateral, so phenotypic == genetic
    assert sp.simplify(e.cov("y_H1", "y_H2") - e.cov("g_H1", "g_H2")) == 0
    # and it is strictly ABOVE the unmultiplied law, since rho_y > rho_g
    assert sp.simplify(sp.factor(e.cov("g_H1", "g_H2") - V_A * rate**2)) != 0
    checked += 5

    # -- step siblings (in-law, modified degree 2) -------------------------------------------
    # M-F currently mated; M had C1 with A, F had C2 with B
    m, e = _pedigree(["M", "F", "A", "B"], [("C1", "M", "A"), ("C2", "F", "B")],
                     [("A", "M"), ("M", "F"), ("F", "B")], "step sibs")
    V_A, V_E, ry, V_P, h2, rg, mu = syms(m)
    # ours: h2 * rho_g * ((1+rho_y)/2)^2 -- the rate factor carries rho_y, not rho_g
    ours = h2 * rg * ((1 + ry) / 2)**2
    assert sp.simplify(e.cov("y_C1", "y_C2") / V_P - ours) == 0, "step sibs"
    # Eftedal et al. give h2 * rho_g * ((1+rho_g)/2)^2; the two differ, and ours is larger
    theirs = h2 * rg * ((1 + rg) / 2)**2
    assert sp.simplify(ours - theirs) != 0, "the two in-law forms must differ"
    num = {V_A: sp.Rational(1), V_E: sp.Rational(1), ry: sp.Rational(1, 2)}
    assert float(ours.subs(num)) > float(theirs.subs(num))
    checked += 3

    # -- in-laws at general modified degree --------------------------------------------------
    # An "in-law ladder": C1 is one step from M, and C2/D2/E2 are one/two/three steps from F.
    # The law must hold at every dtilde, not only at the step-sibling value of 2 -- checked
    # because the all-rho_y form DOES hold at dtilde = 2 and fails at 3, so a single point
    # would have licensed the wrong general formula.
    m, e = _pedigree(["M", "F", "A", "B", "G", "J"],
                     [("C1", "M", "A"), ("C2", "F", "B"), ("D2", "C2", "G"), ("E2", "D2", "J")],
                     [("A", "M"), ("M", "F"), ("F", "B"), ("C2", "G"), ("D2", "J")],
                     "in-law ladder")
    V_A, V_E, ry, V_P, h2, rg, mu = syms(m)
    law = lambda dt: h2 * rg * ((1 + ry) / 2)**2 * ((1 + rg) / 2)**(dt - 2)
    for b, dt in (("C2", 2), ("D2", 3), ("E2", 4)):
        assert sp.simplify(e.cov("y_C1", f"y_{b}") / V_P - law(dt)) == 0, f"in-law dtilde={dt}"
        checked += 1
    # the all-rho_y form is right at dtilde = 2 and WRONG beyond it -- the trap this guards
    assert sp.simplify(e.cov("y_C1", "y_C2") / V_P - h2 * rg * ((1 + ry) / 2)**2) == 0
    assert sp.simplify(e.cov("y_C1", "y_D2") / V_P - h2 * rg * ((1 + ry) / 2)**3) != 0
    checked += 2

    # dtilde is SUFFICIENT: two pedigrees of the same modified degree but different shape
    # (1 step + 3 steps versus 2 steps + 2 steps) must agree, which is what licenses indexing
    # in-laws by a single number at all.
    m2, e2 = _pedigree(["M", "F", "A", "B", "K", "G"],
                       [("C1", "M", "A"), ("D1", "C1", "K"), ("C2", "F", "B"), ("D2", "C2", "G")],
                       [("A", "M"), ("M", "F"), ("F", "B"), ("C1", "K"), ("C2", "G")],
                       "in-law, symmetric 2+2")
    V_A2, V_E2, ry2, V_P2, h22, rg2, mu2 = syms(m2)
    law2 = h22 * rg2 * ((1 + ry2) / 2)**2 * ((1 + rg2) / 2)**2
    assert sp.simplify(e2.cov("y_D1", "y_D2") / V_P2 - law2) == 0, "in-law 2+2 shape"
    checked += 1
    return checked


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
    the genotypes: the allele variance is fixed at 1/2, a^(0) and s^(0) are zero here, and each of
    the four cross-mate allele pairs carries exactly a quarter of the genotype covariance.
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
                # eq:skk -- s^(t)_kk = 1/2, the convention every later formula relies on
                assert engine.var(_z(who, origin, k)) == sp.Rational(1, 2)
                checked += 1
        # eq:as-gen0, exhaustively rather than on a representative pair: EVERY a^(0)_kl is zero
        # (both orderings of the two gametes), and every off-diagonal s^(0)_kl is zero.
        for k in (1, 2):
            for lidx in (1, 2):
                for u, v in (("mat", "pat"), ("pat", "mat")):
                    got = engine.cov(_z(who, u, k), _z(who, v, lidx))       # a^(0)_kl
                    assert got == 0, f"a^(0)_{k}{lidx} = {got}, expected 0"
                    checked += 1
                if lidx == k:
                    continue
                for origin in ("mat", "pat"):
                    got = engine.cov(_z(who, origin, k), _z(who, origin, lidx))   # s^(0)_kl
                    assert got == 0, f"s^(0)_{k}{lidx} = {got}, expected 0"
                    checked += 1
        # ... so eq:varx-a gives Var[x] = 1 + 2 a^(0)_kk = 1, and only here
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
    """Assert that 1/4 - a_kk/2 is forced, not chosen.

    A transmitting parent whose own two alleles are ALREADY correlated -- so the check bites at
    a general a^(t)_kk, not only at the a = 0 of the base population, which is the case a
    generation-1 model would silently pass.
    """
    model = pm.Model("transmission from a parent with a_kk != 0")
    a_kk = model.declare("a_kk", real=True)
    for v in ("z_mat_par", "z_pat_par", "eps_o", "z_o"):
        model.add_var(v, latent=True)
    half = sp.Rational(1, 2)
    model.add_variance("z_mat_par", half)                       # eq:skk
    model.add_variance("z_pat_par", half)
    model.add_cov("z_mat_par", "z_pat_par", a_kk)               # eq:as-def
    model.add_path("z_mat_par", "z_o", half)                    # eq:transmit-mean
    model.add_path("z_pat_par", "z_o", half)
    model.add_variance("eps_o", sp.Rational(1, 4) - a_kk / 2)   # eq:seg-var-a
    model.add_path("eps_o", "z_o", 1)
    engine = pm.RAMEngine(model)

    # the whole point: the offspring's allele still has variance 1/2, for every a_kk
    assert sp.simplify(engine.var("z_o") - half) == 0, engine.var("z_o")
    # the parent's genotype variance is 1 + 2 a_kk (eq:varx-a)
    parent_x = engine.var("z_mat_par") + engine.var("z_pat_par") + 2 * engine.cov(
        "z_mat_par", "z_pat_par"
    )
    assert sp.simplify(parent_x - (1 + 2 * a_kk)) == 0, parent_x
    # eq:seg-var-forcing, term by term as the document prints it: the transmitted half carries a
    # quarter of the parent's genotype variance, and the segregation term is exactly the shortfall
    assert sp.simplify(parent_x / 4 + (sp.Rational(1, 4) - a_kk / 2) - half) == 0
    # and the residual is uncorrelated with the parental alleles, so it is a legitimate
    # exogenous node rather than a fudge factor
    for parental in ("z_mat_par", "z_pat_par"):
        assert sp.simplify(engine.cov("eps_o", parental)) == 0
    checked = 5

    # eq:seg-var-deriv, derived from the Bernoulli draw itself rather than read off the diagram.
    # The diagram cannot check this: it already ASSUMES the linear projection and the residual
    # variance the document is claiming. Averaging explicitly over B in {0, 1} tests the
    # factorization -- E[(B - 1/2)^2] Var[z_mat - z_pat] -- instead of assuming it.
    zm, zp = sp.symbols("z_mat z_pat", real=True)          # second moments supplied below
    moments = {zm**2: half, zp**2: half, zm * zp: a_kk}
    def expectation(expr):
        """E[.] of a polynomial in z_mat, z_pat, using the second moments above (means are 0)."""
        return sp.expand(expr).subs(moments)

    eps = {b: (sp.Rational(b) - half) * (zm - zp) for b in (0, 1)}
    # B is Bernoulli(1/2) and independent of the alleles, so E[f] averages the two branches
    e_eps = sum(half * expectation(e) for e in eps.values())
    e_eps2 = sum(half * expectation(e**2) for e in eps.values())
    assert sp.simplify(e_eps) == 0, e_eps                                     # mean zero
    assert sp.simplify(e_eps2 - (sp.Rational(1, 4) - a_kk / 2)) == 0, e_eps2  # eq:seg-var-a
    # ... and uncorrelated with each parental allele, which is what licenses the disturbance
    for allele in (zm, zp):
        cross = sum(half * expectation(e * allele) for e in eps.values())
        assert sp.simplify(cross) == 0, cross
    # the transmitted allele itself, built from eq:transmit rather than eq:transmit-mean
    z_o = {b: sp.Rational(b) * zm + (1 - sp.Rational(b)) * zp for b in (0, 1)}
    var_zo = sum(half * expectation(z**2) for z in z_o.values())
    assert sp.simplify(var_zo - half) == 0, var_zo
    checked += 5
    return checked


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
            # eq:seg-var-a at a^(0)_kk = 0, drawn as this allele's own disturbance variance
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

    # eq:a-gen1 as the writeup derives it: FOUR chains, each worth beta_k^2 mu / 16. Checked
    # against the tracer's own enumeration, so the chain count in the prose is verified too.
    mu = rho_y / V_P
    for k in (1, 2):
        d = pm.WrightTracer(model).trace(_z("o1", "mat", k), _z("o1", "pat", k))
        assert len(d) == 4, f"prose says four chains, tracer found {len(d)}"
        for chain in d:
            assert sp.simplify(chain.contribution - betas[k] ** 2 * mu / 16) == 0
        # the same quantity read as a share of V_A times half rho_g. The document no longer
        # states this form -- rho_g is deferred with alpha to the equilibrium step -- but it is
        # the relation the equilibrium section will need, so it is worth holding the model to.
        assert sp.simplify(2 * d.total - (betas[k] ** 2 / V_A) * (rho_g / 2)) == 0
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
    """Assert the first-degree relative table (tab:relatives-gen1) -- six covariances.

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

    # eq:reduction, in its CONSOLIDATED form: for opposite gametes one expression covers every
    # pair of variants, the same-variant case included -- that is the claim the two-case version
    # of the equation makes, and it is only true if the k = l diagonal agrees with the rest.
    for k in kk:
        for lidx in kk:
            got = engine.cov(_z("o1", "mat", k), _z("o1", "pat", lidx))
            want = betas[k] * betas[lidx] * rho_y / (4 * V_P)
            assert sp.simplify(got - want) == 0, f"opposite gametes, k={k}, l={lidx}: {got}"
            checked += 1
            if lidx != k:      # same gamete, different variants: still zero at generation 1
                for origin in ("mat", "pat"):
                    assert engine.cov(_z("o1", origin, k), _z("o1", origin, lidx)) == 0
                    checked += 1

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

    # tab:relatives-gen1 as it is now printed: every added term through rho_g instead of
    # rho_y / V_P. These are algebraic rewrites of the assertions above, which is exactly where a
    # slip would go unnoticed -- the rho_y forms would still pass -- so each is checked against the
    # diagram independently rather than against its own rho_y version.
    for k in kk:
        share = betas[k] ** 2 / V_A                       # variant k's share of V_A
        assert sp.simplify(engine.var(f"x_o1{k}") - (1 + share * rho_g / 2)) == 0
        assert sp.simplify(engine.cov(f"x_o1{k}", f"x_o2{k}")
                           - (sp.Rational(1, 2) + share * rho_g / 2)) == 0
        checked += 2
        for lidx in kk:
            if lidx == k:
                continue
            want = betas[k] * betas[lidx] * rho_g / (2 * V_A)
            assert sp.simplify(engine.cov(f"x_o1{k}", f"x_o2{lidx}") - want) == 0
            checked += 1
    # eq:a-gen1-rhog, the allele-level statement the same rewrite gives
    for k in kk:
        for lidx in kk:
            want = betas[k] * betas[lidx] * rho_g / (4 * V_A)
            got = engine.cov(_z("o1", "mat", k), _z("o1", "pat", lidx))
            assert sp.simplify(got - want) == 0, f"eq:a-gen1-rhog at {k},{lidx}"
            checked += 1
    # eq:c-gen1's rho_g form, and the genetic-value and phenotype rows
    for k in kk:
        assert sp.simplify(engine.cov(_z("o1", "mat", k), "y_o1")
                           - betas[k] / 2 * (1 + rho_g / 2)) == 0, "eq:c-gen1 in rho_g"
        checked += 1
    assert sp.simplify(engine.var("g_o1") - V_A * (1 + rho_g / 2)) == 0
    assert sp.simplify(engine.var("y_o1") - (V_A * (1 + rho_g / 2) + V_E)) == 0
    assert sp.simplify(engine.cov("g_o1", "g_o2") - V_A / 2 * (1 + rho_g)) == 0
    assert sp.simplify(engine.cov("y_o1", "y_o2") - V_A / 2 * (1 + rho_g)) == 0
    assert sp.simplify(engine.cov("g_m", "g_o1") - V_A / 2 * (1 + rho_g)) == 0
    # the one cell that carries rho_y rather than rho_g -- the point the caption now makes
    assert sp.simplify(engine.cov("y_m", "y_o1") - V_A / 2 * (1 + rho_y)) == 0
    assert sp.simplify(engine.cov("y_m", "y_o1") - engine.cov("y_o1", "y_o2")) != 0
    checked += 7

    # The parent-offspring phenotype display, underbrace by underbrace as the document prints it.
    # The point of writing it expanded rather than through rho_g is that V_P cancels, and it only
    # cancels because the two terms carry V_A and V_E over the SAME V_P -- so both are checked
    # separately, not just their sum.
    assert sp.simplify(engine.cov("g_m", "g_o1") - (V_A / 2 + rho_y * V_A**2 / (2 * V_P))) == 0
    assert sp.simplify(engine.cov("e_m", "g_o1") - rho_y * V_A * V_E / (2 * V_P)) == 0
    assert sp.simplify(
        engine.cov("y_m", "y_o1") - (V_A / 2 + rho_y * V_A * (V_A + V_E) / (2 * V_P))
    ) == 0
    checked += 3
    return checked


def check_dynamics() -> int:
    """Assert the dynamics section by building THREE generations explicitly.

    Numeric rather than symbolic: the generation-1 co-path coefficient depends on V_P(1), which
    is an output of the generation-0 half of the model, so the model has to be built in two
    passes. Parameters deliberately do not sum to 1 (V_P(0) = 1.39).

    The trap this exists to catch: a generation-2 child's segregation variance is
    (1 - alpha^(1)_k)/4, NOT 1/4, because its parents' alleles are already correlated. With 1/4
    the generation-2 alleles stop having variance 1/2 and [Constant-freq] is violated, silently.
    """
    b1, b2, V_E, rho_y = 0.6, 0.4, 0.87, 0.30
    betas = {1: b1, 2: b2}
    kk = (1, 2)
    V_A0 = b1**2 + b2**2
    V_P0 = V_A0 + V_E

    def person(model, who, exogenous):
        for k in kk:
            for o in ("mat", "pat"):
                model.add_var(_z(who, o, k), latent=True)
                if exogenous:
                    model.add_variance(_z(who, o, k), 0.5)
            model.add_var(f"x_{who}{k}")
            for o in ("mat", "pat"):
                model.add_path(_z(who, o, k), f"x_{who}{k}", 1)
        for v in ("g", "e"):
            model.add_var(f"{v}_{who}", latent=True)
        model.add_var(f"y_{who}")
        for k in kk:
            model.add_path(f"x_{who}{k}", f"g_{who}", betas[k])
        model.add_path(f"g_{who}", f"y_{who}", 1)
        model.add_path(f"e_{who}", f"y_{who}", 1)
        model.add_variance(f"e_{who}", V_E)

    def transmit(model, child, mother, father, alpha=None):
        for k in kk:
            a = 0.0 if alpha is None else alpha[k]
            for o, parent in (("mat", mother), ("pat", father)):
                for po in ("mat", "pat"):
                    model.add_path(_z(parent, po, k), _z(child, o, k), 0.5)
                model.add_variance(_z(child, o, k), (1.0 - a) / 4)   # eq:seg-var-a

    def build(mu1=None, alpha1=None):
        model = pm.Model("three generations")
        for who in ("A", "B", "C", "D"):
            person(model, who, True)
        for who in ("E", "F"):
            person(model, who, False)
        transmit(model, "E", "A", "B")
        transmit(model, "F", "C", "D")
        model.add_copath("y_A", "y_B", rho_y / V_P0, process="c0a")
        model.add_copath("y_C", "y_D", rho_y / V_P0, process="c0b")
        if mu1 is not None:
            person(model, "G", False)
            transmit(model, "G", "E", "F", alpha=alpha1)
            model.add_copath("y_E", "y_F", mu1, process="c1")
        return model

    close = lambda a, b: abs(a - b) <= 1e-9 * max(1.0, abs(b))
    checked = 0

    e1 = pm.RAMEngine(build())
    V_A1, V_P1 = float(e1.var("g_E")), float(e1.var("y_E"))
    rho_g0 = rho_y * V_A0 / V_P0
    rho_g1 = rho_y * V_A1 / V_P1
    alpha1 = {k: 2 * float(e1.cov(_z("E", "mat", k), _z("E", "pat", k))) for k in kk}
    assert close(V_A1, V_A0 * (1 + rho_g0 / 2))
    assert e1.cov("g_E", "g_F") == 0, "the two generation-1 mates must start out unrelated"
    checked += 2
    # eq:effective-effect / [Uniform-inflation], exact at t = 1
    for k in kk:
        assert close(float(e1.cov(f"x_E{k}", "y_E")), betas[k] * V_A1 / V_A0)
        checked += 1

    e2 = pm.RAMEngine(build(mu1=rho_y / V_P1, alpha1=alpha1))
    # [Constant-freq] must survive into generation 2 -- this is what (1-alpha)/4 buys
    for k in kk:
        assert close(float(e2.var(_z("G", "mat", k))), 0.5), "generation-2 allele variance moved"
        checked += 1
    # eq:alpha-recursion-VA at t = 1, i.e. alpha^(2)
    for k in kk:
        want = rho_y * betas[k] ** 2 * (V_A1 / V_A0) ** 2 / (2 * V_P1)
        assert close(2 * float(e2.cov(_z("G", "mat", k), _z("G", "pat", k))), want)
        checked += 1
    # eq:VA-recursion, INCLUDING the segregation deficit the standard form drops
    V_A2 = float(e2.var("g_G"))
    V_K1 = sum(betas[k] ** 2 * (1 - alpha1[k]) for k in kk) / 2
    assert close(V_A2, V_K1 + V_A1 * (1 + rho_g1) / 2)
    naive = V_A0 / 2 + V_A1 * (1 + rho_g1) / 2
    assert close(naive - V_A2, sum(betas[k] ** 2 * alpha1[k] for k in kk) / 2)
    checked += 2

    # eq:reduction -- deleting the base population is only faithful WITH the cross-variant terms
    def reduced(cross_variant):
        model = pm.Model("reduced")
        for who in ("E", "F"):
            person(model, who, True)
            for k in kk:
                model.add_cov(_z(who, "mat", k), _z(who, "pat", k), alpha1[k] / 2)
            if cross_variant:
                for u, v in (("mat", "pat"), ("pat", "mat")):
                    model.add_cov(
                        _z(who, u, 1), _z(who, v, 2), b1 * b2 * rho_y / (4 * V_P0)
                    )
        person(model, "G", False)
        transmit(model, "G", "E", "F", alpha=alpha1)
        model.add_copath("y_E", "y_F", rho_y / V_P1, process="c1")
        return pm.RAMEngine(model)

    full = {k: 2 * float(e2.cov(_z("G", "mat", k), _z("G", "pat", k))) for k in kk}
    with_cv = reduced(True)
    without = reduced(False)
    for k in kk:
        assert close(2 * float(with_cv.cov(_z("G", "mat", k), _z("G", "pat", k))), full[k])
        # and the same-variant-only reduction must NOT reproduce it -- if it ever does, the
        # warning in the text is wrong and should come out
        bad = 2 * float(without.cov(_z("G", "mat", k), _z("G", "pat", k)))
        assert abs(bad - full[k]) / full[k] > 0.01, "same-variant-only reduction is not wrong?"
        checked += 2

    # eq:VA-eq and eq:alpha-eq, by iterating the map to its fixed point
    V_A = V_A0
    for _ in range(2000):
        V_A = V_A0 / 2 + V_A * (1 + rho_y * V_A / (V_A + V_E)) / 2
    rho_g_eq = rho_y * V_A / (V_A + V_E)
    assert close(V_A, V_A0 / (1 - rho_g_eq))
    for k in kk:
        direct = rho_y * betas[k] ** 2 * (V_A / V_A0) ** 2 / (2 * (V_A + V_E))
        closed = (betas[k] ** 2 / V_A0) * rho_g_eq / (2 * (1 - rho_g_eq))
        assert close(direct, closed)
        checked += 1
    checked += 1
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
        print(f"relatives M={m_c}: {check_relative_table(m_c)} results agree with pathMgr")
    print(f"dynamics:      {check_dynamics()} results agree with pathMgr")
    print(f"a/s recursion: {check_as_recursion(3)} results agree with pathMgr")
    print(f"c recursion:   {check_c_recursion(4)} results agree with pathMgr")
    print(f"state&readouts: {check_state_readouts(4)} results agree with pathMgr")
    print(f"equilibrium:   {check_equilibrium()} results agree with pathMgr")
    print(f"g transmit:    {check_g_transmit(2)} results agree with pathMgr")
    print(f"g-only figure: {check_g_only_figure()} results agree with pathMgr")
    print(f"mean-parent:   {check_mean_parent()} results agree with pathMgr")
    print(f"N_mu-bar:      {check_nmu_bar()} results agree with pathMgr")
    print(f"pedigree sets: {check_pedigree_sets()} results agree with pathMgr")
    print(f"path framework: {check_path_framework()} results agree with pathMgr")
    print(f"one-step rules: {check_one_step_rules()} results agree with pathMgr")
    print(f"degree classes: {check_degree_classes()} results agree with pathMgr")
    print(f"step-sib fig:  {check_step_sib_figure()} results agree with pathMgr")
    print(f"reduced fig:   {check_reduced_figure()} results agree with pathMgr")
    # M = 3 only: the reduced model carries M^2 bidirected edges per parent, so the symbolic
    # cost climbs steeply and M = 4 does not exercise anything M = 3 misses here.
    print(f"gen-2 alleles: {check_gen2_alleles(3)} results agree with pathMgr")
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
    # The eight offspring self-loops carry the segregation variance, and placement leans each one
    # away from its sibling -- o1's up-left, o2's up-right -- except the two on the OUTERMOST
    # alleles, which have open air beside them and so go flat sideways. That is collision-free but
    # it puts their labels ~1.15cm outside the node span, and this figure is a sidewaysfigure whose
    # width is already at the page limit: the overhang alone made it 14.7pt overfull. Sending just
    # those two straight down keeps the figure inside the text block at no cost -- still zero
    # collisions and zero ambiguous labels -- where forcing ALL eight down (or up) for uniformity
    # costs 8 label-edge collisions, because the inner six have nowhere vertical to go.
    outer_loops = {
        _z("o1", "mat", 1): (270.0, 6.0),
        _z("o2", "pat", 2): (270.0, 6.0),
    }
    offspring = HERE / "pair_offspring_2v.tikz"
    offspring.write_text(
        to_tikz(
            pair_offspring_2v(),
            layout=PAIR_OFFSPRING_2V_LAYOUT,
            style=DiagramStyle(
                show_variances=True, show_unit_coefficients=False, latex_names=names,
                loop_overrides=outer_loops,
            ),
        )
    )
    print(f"wrote {offspring.relative_to(HERE.parent)}")

    # Figure 3: the reduced diagram. The raw coefficients here are long expressions in beta and
    # V_P, and the document has already given each of them a name, so every one is relabelled to
    # the name the surrounding text uses. `label_overrides` is keyed on the edge, so this cannot
    # silently relabel the wrong thing.
    # The eight within-parent bidirected edges are left UNLABELLED, and stay that way now that
    # placement has improved enough to make labelling them an option. Re-measured: labelling all
    # eight with the document's own symbol a^(1)_kl is collision-free, but four of them come out
    # ATTRIBUTABLE TO THE WRONG EDGE (ambiguous 2 -> 6, the worst by 1.59 cm). That is structural,
    # not a placement failure -- four bidirected edges among four collinear nodes have overlapping
    # midpoints, and widening the allele row does not fix it (tried 13/15/17/19 cm spans: 5, 9, 6,
    # 7 ambiguous). Labelling only the four same-variant edges IS clean, but it would imply the
    # k != l pairs are different or zero when a^(1)_kl is one formula for every k and l -- a
    # partial labelling here states something false, where no labelling states nothing.
    # So: the pattern of arrows is what this figure contributes; the value is eq:a-gen1.
    suppressed = frozenset(
        (_z(who, "mat", k), _z(who, "pat", lidx))
        for who in ("m", "p") for k in (1, 2) for lidx in (1, 2)
    )
    overrides = {}
    for k in (1, 2):
        for origin in ("mat", "pat"):
            node = _z("o", origin, k)
            # eq:seg-var-a in the document's own symbols. alpha is no longer defined at this
            # point in the text -- it is deferred to the equilibrium step -- so a label written
            # in alpha would name a symbol the reader has not met.
            overrides[(node, node)] = rf"\tfrac14 - \tfrac12 a^{{(1)}}_{{{k}{k}}}"
    reduced = HERE / "reduced_pair_offspring.tikz"
    reduced.write_text(
        to_tikz(
            reduced_pair_offspring(),
            layout=REDUCED_LAYOUT,
            style=DiagramStyle(
                show_variances=True,
                show_unit_coefficients=False,
                latex_names=names,
                label_overrides=overrides,
                suppressed_labels=suppressed,
            ),
        )
    )
    print(f"wrote {reduced.relative_to(HERE.parent)}")

    # Figure 4: genetic values only. The disturbance's raw expression is long and the document
    # has already named it, so it is relabelled to the name the surrounding text uses; every
    # other label here is a single symbol and needs no help.
    g_only = g_only_pair_offspring()
    V_A_s, V_E_s, rho_y_s = (g_only.sym(s) for s in ("V_A", "V_E", "rho_y"))
    seg = V_A_s * (1 - rho_y_s * V_A_s / (V_A_s + V_E_s)) / 2
    target = HERE / "g_only_pair_offspring.tikz"
    target.write_text(
        to_tikz(
            g_only,
            layout=G_ONLY_LAYOUT,
            style=DiagramStyle(
                show_variances=True,
                show_unit_coefficients=False,
                latex_names={seg: r"\tfrac12 V_A\big(1-\rho_g\big)"},
                # The four transmission edges converge on two nodes, so their midpoint labels
                # pile up in the middle of the figure -- and all four read 1/2, so they carry no
                # information the caption cannot. Same call as Figure 3's bidirected edges: the
                # pattern of arrows is what the diagram contributes, the value is in eq:g-transmit.
                suppressed_labels=frozenset((f"g_{w}", f"g_{k}")
                                            for w in ("m", "p") for k in ("o1", "o2")),
            ),
        )
    )
    print(f"wrote {target.relative_to(HERE.parent)}")

    # Figure 5: the step-sibling pedigree that the mating-count definition is read off. Same
    # conventions as Figure 4, so the two can be compared directly; the disturbance is relabelled
    # to the document's name for it for the same reason.
    ss_model, _ = step_sib_pedigree()
    V_A_s, V_E_s, rho_y_s = (ss_model.sym(s) for s in ("V_A", "V_E", "rho_y"))
    ss_seg = V_A_s * (1 - rho_y_s * V_A_s / (V_A_s + V_E_s)) / 2
    target = HERE / "step_sib_pedigree.tikz"
    target.write_text(
        to_tikz(
            ss_model,
            layout=STEP_SIB_LAYOUT,
            style=DiagramStyle(
                show_variances=True,
                show_unit_coefficients=False,
                latex_names={
                    ss_seg: r"\tfrac12 V_A\big(1-\rho_g\big)",
                    # This pedigree has to declare its co-paths by the raw mu, because p2 and p3
                    # each have two mates and the correlation form cannot be resolved then. Left
                    # alone that prints rho_y/(V_A+V_E) on all three lines, which is both ugly
                    # and against the document's convention of labelling a co-path with the
                    # correlation it induces (see the text below eq:copath-mu). At equilibrium
                    # that correlation is rho_y, so this restores the label Figure 4 carries.
                    rho_y_s / (V_A_s + V_E_s): r"[\rho_y]",
                },
                # pathMgr subscripts a trailing digit on its own (g_o1 -> g_{o_1}) for a variable
                # it infers, but not for one named in an explicit `latent:` line, which is how
                # _pedigree declares these. Set them by hand so this figure matches Figure 4 and
                # the caption's own $p_1$, $o_1$ notation.
                node_label_overrides={
                    f"{v}_{w}": rf"{v}_{{{w[0]}_{w[1]}}}"
                    for v in ("g", "y") for w in STEP_SIB_PARENTS + ("o1", "o2", "o3")
                },
                # all four transmission edges read 1/2 and pile up between the two generations;
                # same call as Figures 3 and 4, and the value is in eq:g-transmit
                suppressed_labels=frozenset((f"g_{parent}", f"g_{child}")
                                            for child, a, b in STEP_SIB_KIDS
                                            for parent in (a, b)),
            ),
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
