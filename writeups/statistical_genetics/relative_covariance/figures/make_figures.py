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

    # -- the justification's error term -------------------------------------------------------
    # relative error in V_A of order rho_g / (2(1-rho_g) M_e), with M_e the effective count.
    M_e = V_A0**2 / sum(v**4 for v in b.values())
    predicted = rho_g / (2 * (1 - rho_g) * M_e)
    actual = (V_A_ap - V_A) / V_A
    assert actual > 0, "the approximation should overestimate V_A"
    assert 0.5 < actual / predicted < 2.0, f"error term off by {actual / predicted:.2f}x"
    checked += 2
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

    # Figure 3: the reduced diagram. The raw coefficients here are long expressions in beta and
    # V_P, and the document has already given each of them a name, so every one is relabelled to
    # the name the surrounding text uses. `label_overrides` is keyed on the edge, so this cannot
    # silently relabel the wrong thing.
    # The four within-parent bidirected edges are left UNLABELLED. Their values are one equation
    # above the figure in the document, and eight labels among four closely spaced allele nodes
    # collide with the node text -- tried it, the render is unreadable. The pattern of arrows is
    # what this figure contributes; the numbers are eq:reduction.
    overrides = {
        (_z(who, "mat", k), _z(who, "pat", lidx)): ""
        for who in ("m", "p") for k in (1, 2) for lidx in (1, 2)
    }
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
            ),
        )
    )
    print(f"wrote {reduced.relative_to(HERE.parent)}")

    stale = HERE / "mated_pair_2v_traced.tikz"
    if stale.exists():
        stale.unlink()
        print(f"removed {stale.relative_to(HERE.parent)} (merged into mated_pair_2v)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
