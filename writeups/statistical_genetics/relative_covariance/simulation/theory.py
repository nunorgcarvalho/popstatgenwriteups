"""The writeup's closed-form predictions, as numeric functions of MEASURED parameters.

    python simulation/theory.py        # run the self-consistency checks

This module is the "closed-form formula" leg of the verification. It restates what
`relative_covariance.tex` derives and nothing else: no simulation, no path tracing, no
symbolic algebra. Every function here is a direct transcription of a boxed equation or a
table entry, named for the label it comes from, so a reader with the PDF open can check the
transcription line by line.

Two rules govern the interface, both settled decisions (thesisMgr D-0009):

1.  **Predictions are evaluated at measured parameters, never nominal inputs.** Nothing here
    takes a simulation parameter like `AM_r`. The callers pass `rho_y`, `rho_g`, `V_A`, `V_Y`
    as *measured* in the population being checked. The pairing mechanism is a rank-matching
    copula rather than a bivariate normal, so realized `rho_y` differs from the nominal
    input, and realized `rho_g` from `rho_y * h^2`; feeding nominal values in would confound
    "is the equilibrium algebra right" with "did the mechanism hit its target".
2.  **`V_A` is measured against a pinned base-generation standardization** (the writeup's
    `Constant-freq`), not per-generation observed standardization. That is the caller's
    responsibility; this module cannot check it and only documents which quantity it means.

The equilibrium functions are the one exception to rule 1, by necessity: `rho_g_eq` exists
precisely to predict `rho_g` from the base-generation `h^2`, so it takes the nominal `rho_y`
and a measured *base* `h^2`. Everything downstream of it takes measured values.

Where an equation is an approximation, the approximation is named and its correction is
given its own function rather than being folded in silently -- see `VA_eq_relative_error`,
which is what makes `eq:VA-eq` testable as an approximation instead of merely asserted
within a tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# A proband's position on the path it terminates, which fixes which correction it takes in
# eq:proband-correction: the young end of a lineal chain (no correction), the old end, or a
# node on a terminal mating chain. These are the three cases the writeup splits.
PROBAND_CASES = ("young", "old", "mating")


# ----------------------------------------------------------------------------------------------
# Equilibrium, section 2.2. These predict the population's own parameters, and are the only
# functions here that take a nominal rho_y -- everything downstream takes measured values.
# ----------------------------------------------------------------------------------------------

def rho_g_eq(rho_y: float, h2_0: float) -> float:
    """eq:rhog-eq -- mate genetic correlation at equilibrium, from base-generation h^2.

    The minus root is the admissible one and the discriminant is never negative for
    h2_0 in [0, 1] and rho_y in [0, 1].
    """
    disc = 1.0 - 4.0 * rho_y * h2_0 * (1.0 - h2_0)
    if disc < 0.0:
        raise ValueError(f"negative discriminant in eq:rhog-eq: rho_y={rho_y}, h2_0={h2_0}")
    if h2_0 == 1.0:
        # the (1 - h2_0) denominator vanishes; the quadratic degenerates to rho_g = rho_y
        return rho_y
    return (1.0 - math.sqrt(disc)) / (2.0 * (1.0 - h2_0))


def VA_eq(VA_0: float, rho_g: float) -> float:
    """eq:VA-eq -- the additive variance inflated by one factor. Approximation VA-inflation."""
    return VA_0 / (1.0 - rho_g)


def h2_eq(h2_0: float, rho_g: float) -> float:
    """eq:hsq-eq, second boxed form -- equilibrium heritability from the base-generation one."""
    return h2_0 / (1.0 - rho_g * (1.0 - h2_0))


def seg_var(V_A: float, rho_g: float) -> float:
    """eq:g-seg-var -- Var[g_o^perp], the offspring genetic value's variance about the
    parental mean.

    Takes the MEASURED equilibrium `V_A`. Under approximation VA-inflation this equals
    half the base-generation additive variance, i.e. assortment does not change it; passing
    a measured `V_A` and a measured `rho_g` keeps that a prediction rather than an identity.
    """
    return 0.5 * V_A * (1.0 - rho_g)


def M_e(betas: Sequence[float]) -> float:
    """eq:Me-def -- the effective number of variants, (sum beta^2)^2 / sum beta^4.

    Normalized so that M_e = M when every variant has the same effect size.
    """
    s2 = sum(b * b for b in betas)
    s4 = sum(b ** 4 for b in betas)
    if s4 == 0.0:
        raise ValueError("all effect sizes are zero; M_e is undefined")
    return s2 * s2 / s4


def VA_eq_relative_error(rho_g: float, m_e: float) -> float:
    """The relative error that approximation VA-inflation incurs, from eq:VA-eq-correction.

    `eq:VA-eq` drops a term worth `rho_g / (2 (1 - rho_g) M_e)` of `V_A^(eq)`. The writeup
    states the sign: dropping it OVERSTATES `V_A^(eq)`, because a variant's own departure
    from Hardy-Weinberg drags its contribution down. So the prediction to test is

        V_A^(measured)  ~=  VA_eq(...) * (1 - VA_eq_relative_error(...))

    and the signed residual should shrink like 1/M_e. This is a sharper test than asserting
    `eq:VA-eq` within a Monte-Carlo tolerance: it predicts the direction and the size of the
    discrepancy, which a tolerance check would absorb.
    """
    return rho_g / (2.0 * (1.0 - rho_g) * m_e)


def VA_eq_corrected(VA_0: float, rho_g: float, m_e: float) -> float:
    """eq:VA-eq-correction -- `eq:VA-eq` with the dropped drag term kept."""
    return VA_eq(VA_0, rho_g) * (1.0 - VA_eq_relative_error(rho_g, m_e))


# ----------------------------------------------------------------------------------------------
# The pair tier: tab:g-only-cov, at equilibrium. Keyed by (pair, level_A, level_B) where a
# level is "g" or "y". The asymmetry across the diagonal of the parent--offspring column is
# the point of the table, so the two mixed readings are kept distinct.
# ----------------------------------------------------------------------------------------------

def pair_cov(pair: str, level_A: str, level_B: str, *,
             V_A: float, V_Y: float, rho_g: float, rho_y: float) -> float:
    """A single entry of tab:g-only-cov, at MEASURED parameters.

    `pair` is "mates" for a mated pair (m, p) or "parent-offspring" for (m, o).
    `level_A`/`level_B` are "g" or "y", naming which value of each member is taken. For the
    mate pair the two members are exchangeable, so the order does not matter; for
    parent--offspring, `level_A` is the PARENT's level and `level_B` the OFFSPRING's, and the
    two mixed readings differ -- that is the asymmetry the text points out:

        Cov[g_m, y_o] = V_A (1 + rho_g) / 2      the parent's genotype against the child's phenotype
        Cov[y_m, g_o] = V_A (1 + rho_y) / 2      the parent's phenotype against the child's genotype

    and since rho_g = rho_y h^2 <= rho_y the second is the LARGER. A parent's environment
    predicts the genetic value the other parent transmits, but not the offspring's own
    environment.
    """
    if level_A not in ("g", "y") or level_B not in ("g", "y"):
        raise ValueError(f"levels must be 'g' or 'y', got {level_A!r} and {level_B!r}")
    if pair == "mates":
        if level_A == "g" and level_B == "g":
            return rho_g * V_A
        if level_A == "y" and level_B == "y":
            return rho_y * V_Y
        return rho_y * V_A                       # one of each, either way round
    if pair == "parent-offspring":
        # the parent's own level is what picks the coefficient; the offspring's does not
        rho = rho_g if level_A == "g" else rho_y
        return 0.5 * V_A * (1.0 + rho)
    raise ValueError(f"unknown pair {pair!r}; expected 'mates' or 'parent-offspring'")


def rho_g_from_pair(rho_y: float, h2: float) -> float:
    """The mate genetic correlation as the phenotypic one attenuated by heritability.

    Stated for the base generation under tab:pair-2v-between as `rho_g^(0) = rho_y h^2_0`,
    and holding at equilibrium as eq:rhog-t. Feeding this a MEASURED `rho_y` and `h^2` and
    comparing against a measured `rho_g` tests the pairing mechanism itself: the mechanism
    matches on ranks, so there is no guarantee it induces exactly this.
    """
    return rho_y * h2


# ----------------------------------------------------------------------------------------------
# The relationship-class tier, section 2.3. eq:chain-weight is the single source of truth for
# a mating chain's weight (thesisMgr D-0008); the N_S case split of eq:chain-weight-cases is
# presentational and is deliberately not implemented -- `chain_weight_cases` below exists only
# to check the binomial form against it, and is not used to compute anything.
# ----------------------------------------------------------------------------------------------

def _c(n: int, rho_g: float, rho_y: float) -> float:
    """The mating-distance factor of eq:mating-dist: 1 at zero distance, else rho_g rho_y^(n-1)."""
    return 1.0 if n == 0 else rho_g * rho_y ** (n - 1)


def chain_weight(N_mu: int, N_o: int, rho_g: float, rho_y: float) -> float:
    """eq:chain-weight -- the weight of one mating chain, as the binomial sum.

    `N_mu` is the chain's mating distance and `N_o` the total out-degree of its nodes (the
    quantity earlier drafts called N_U). The binomial form reproduces all three rows of
    eq:chain-weight-cases on its own, including the boundary, which is why the N_S
    parameterization is never computed.
    """
    return sum(math.comb(N_o, s) * _c(abs(N_mu - s), rho_g, rho_y)
               for s in range(N_o + 1)) / 2 ** N_o


def chain_weight_cases(N_mu: int, N_o: int, rho_g: float, rho_y: float) -> float:
    """eq:chain-weight-cases -- the presentational case split, for CHECKING only.

    Not used to compute any prediction; `chain_weight` is. Kept so the self-test can assert
    the two agree everywhere the case form is defined, which is what licenses ignoring N_S.
    """
    N_S = max(0, N_o + 1 - N_mu)
    if N_S == 2:
        return lam_g(rho_g)
    if N_S == 1:
        return (1.0 + 2.0 * rho_g + rho_g * rho_y) / 4.0
    return rho_g * rho_y ** (N_mu - 1 - N_o) * lam_y(rho_y) ** N_o


def lam_g(rho_g: float) -> float:
    """The genetic transmission coefficient (1 + rho_g) / 2, carried by one lineal step."""
    return (1.0 + rho_g) / 2.0


def lam_y(rho_y: float) -> float:
    """The phenotypic counterpart (1 + rho_y) / 2, carried by a mating chain's out-degree."""
    return (1.0 + rho_y) / 2.0


def proband_correction(case: str, rho_g: float, rho_y: float) -> float:
    """eq:proband-correction -- the factor a proband contributes by being read at the
    phenotypic rather than the genetic level.

    The three cases are where the proband sits on the path it terminates:
      "young"   the young end of a lineal chain -- no correction
      "old"     the old end -- (1 + rho_y) / (1 + rho_g)
      "mating"  a node on a terminal mating chain -- rho_y / rho_g, which is 1 / h^2
    """
    if case == "young":
        return 1.0
    if case == "old":
        return (1.0 + rho_y) / (1.0 + rho_g)
    if case == "mating":
        if rho_g == 0.0:
            raise ValueError("the mating-chain correction rho_y/rho_g is undefined at rho_g = 0")
        return rho_y / rho_g
    raise ValueError(f"unknown proband case {case!r}; expected one of {PROBAND_CASES}")


def path_cov_g(lineals: Sequence[int], matings: Sequence[tuple[int, int]], *,
               V_A: float, rho_g: float, rho_y: float) -> float:
    """eq:path-product -- the genetic covariance along one path between two probands.

    `lineals` gives N_lin per lineal chain, `matings` gives (N_mu, N_o) per mating chain.
    The lineal steps contribute lam_g to a power reduced by the mating chains' out-degrees,
    since each out-edge of a mating chain absorbs one lineal step.
    """
    out = V_A * lam_g(rho_g) ** (sum(lineals) - sum(N_o for _, N_o in matings))
    for N_mu, N_o in matings:
        out *= chain_weight(N_mu, N_o, rho_g, rho_y)
    return out


def path_cov(lineals: Sequence[int], matings: Sequence[tuple[int, int]],
             cases: tuple[str, str], levels: tuple[str, str] = ("y", "y"), *,
             V_A: float, rho_g: float, rho_y: float) -> float:
    """eq:path-product-pheno -- the same path read at whichever level each proband is taken.

    `cases` is the proband case at each end (see `proband_correction`), `levels` is "g" or
    "y" per end. A proband read at "g" takes no correction, so ("g", "g") returns
    `path_cov_g` and the mixed readings take exactly one correction -- the reason the table
    in the writeup carries four rows per class rather than one.
    """
    out = path_cov_g(lineals, matings, V_A=V_A, rho_g=rho_g, rho_y=rho_y)
    for case, level in zip(cases, levels):
        if level == "y":
            out *= proband_correction(case, rho_g, rho_y)
        elif level != "g":
            raise ValueError(f"levels must be 'g' or 'y', got {level!r}")
    return out


#: The relationship classes of tab:relationship-classes, as path descriptions: the lineal
#: chain lengths, the (N_mu, N_o) of each mating chain, and the proband case at each end.
#: These are the writeup's own eleven rows, and the keys are that table's OWN names --
#: which are also the names pathMgr's RelativeOracle
#: exposes in RELATIONSHIP_CLASSES — one shared vocabulary across the three legs, so a
#: three-way table can be joined on the class name without a translation layer. The
#: pedigree that realizes each one lives in `figures/make_figures.py` (PATH_CLS_CASES), which
#: asserts these same descriptions against pathMgr -- so a disagreement between this dict and
#: that one is a transcription error in one of the two places.
#:
#: "self" is DEGENERATE: there is no path, so no proband correction applies, and the
#: phenotypic reading is V_Y rather than V_A -- the one row whose Cov[y, y] is not
#: Cov[g, g] scaled by corrections. `class_cov` special-cases it and its `cases` is None so
#: that a correction cannot be applied to it by accident. "step-step siblings" is the N_mu = 4
#: class that no simulation is expected to reach.
RELATIONSHIP_CLASSES: dict[str, dict] = {
    #  name                           lineal chains   mating chains    proband cases
    "self":                        {"lineals": (),     "matings": (),        "cases": None},
    "parent-parent":               {"lineals": (),     "matings": ((1, 0),), "cases": ("mating", "mating")},
    "parent-offspring":            {"lineals": (1,),   "matings": (),        "cases": ("old", "young")},
    "grandparent-grandchild":      {"lineals": (2,),   "matings": (),        "cases": ("old", "young")},
    "full siblings":               {"lineals": (1, 1), "matings": ((1, 2),), "cases": ("young", "young")},
    "avuncular":                   {"lineals": (1, 2), "matings": ((1, 2),), "cases": ("young", "young")},
    "first cousins":               {"lineals": (2, 2), "matings": ((1, 2),), "cases": ("young", "young")},
    "half siblings":               {"lineals": (1, 1), "matings": ((2, 2),), "cases": ("young", "young")},
    "step siblings":               {"lineals": (1, 1), "matings": ((3, 2),), "cases": ("young", "young")},
    "step-step siblings":          {"lineals": (1, 1), "matings": ((4, 2),), "cases": ("young", "young")},
    "parents of a married couple": {"lineals": (1, 1), "matings": ((1, 0),), "cases": ("old", "old")},
}


def class_cov(name: str, levels: tuple[str, str] = ("y", "y"), *,
              V_A: float, rho_g: float, rho_y: float, V_Y: float | None = None) -> float:
    """The predicted covariance for a named row of tab:relationship-classes.

    `V_Y` is needed only for the degenerate "self" row read phenotypically, where the answer
    is `V_Y` rather than anything the path product can produce.
    """
    if name not in RELATIONSHIP_CLASSES:
        raise KeyError(f"unknown class {name!r}; known: {sorted(RELATIONSHIP_CLASSES)}")
    spec = RELATIONSHIP_CLASSES[name]
    if spec["cases"] is None:                       # "self": no path, so no correction
        if levels == ("y", "y"):
            if V_Y is None:
                raise ValueError("the 'self' row at ('y', 'y') is V_Y; pass V_Y")
            return V_Y                              # Var[y] = V_A + V_E
        return V_A                                  # Var[g], and Cov[g, y] = Cov[g, g + e]
    return path_cov(spec["lineals"], spec["matings"], spec["cases"], levels,
                    V_A=V_A, rho_g=rho_g, rho_y=rho_y)


# ----------------------------------------------------------------------------------------------
# Self-test. These are internal consistency checks of the transcription -- the identities and
# coincidences the writeup itself asserts, plus agreement between the two forms of the chain
# weight. They do NOT verify the writeup (that is what the notebooks and the pathMgr oracle in
# figures/make_figures.py do); they catch a typo here.
# ----------------------------------------------------------------------------------------------

def _self_test() -> int:
    checked = 0
    close = lambda a, b, tol=1e-12: abs(a - b) < tol

    # -- a concrete equilibrium, carried through the whole chain of formulas ------------------
    rho_y, h2_0, VA_0 = 0.3, 0.5, 0.4
    V_E = VA_0 / h2_0 - VA_0
    rho_g = rho_g_eq(rho_y, h2_0)
    V_A = VA_eq(VA_0, rho_g)
    h2 = h2_eq(h2_0, rho_g)
    V_Y = V_A + V_E

    # eq:rhog-quadratic -- the root must satisfy the quadratic it came from
    assert close((1 - h2_0) * rho_g ** 2 - rho_g + rho_y * h2_0, 0.0)
    # eq:rhog-eq must be consistent with rho_g = rho_y h^2 at the EQUILIBRIUM h^2
    assert close(rho_g, rho_y * h2)
    assert close(rho_g, rho_g_from_pair(rho_y, h2))
    # h^2 read two ways: eq:hsq-eq, and V_A / V_Y with V_E held fixed (the writeup holds V_E)
    assert close(h2, V_A / V_Y)
    # eq:g-seg-var equals half the BASE additive variance under VA-inflation
    assert close(seg_var(V_A, rho_g), 0.5 * VA_0)
    # assortment must inflate, and h^2 must rise with it
    assert V_A > VA_0 and h2 > h2_0
    # switching assortment off must return every quantity to the base population
    assert close(rho_g_eq(0.0, h2_0), 0.0) and close(VA_eq(VA_0, 0.0), VA_0)
    assert close(h2_eq(h2_0, 0.0), h2_0)
    checked += 9

    # -- M_e and the signed VA-inflation error ------------------------------------------------
    assert close(M_e([0.1] * 20), 20.0)                       # equal effects => M_e = M
    assert M_e([1.0] + [0.05] * 19) < 2.0                     # one variant dominating
    # the correction must be a strict shrink, and must vanish as M_e grows
    assert VA_eq_corrected(VA_0, rho_g, 50.0) < VA_eq(VA_0, rho_g)
    assert VA_eq_relative_error(rho_g, 1e9) < 1e-9
    assert VA_eq_relative_error(rho_g, 10.0) > VA_eq_relative_error(rho_g, 100.0)
    checked += 5

    # -- tab:g-only-cov, including the asymmetry that is the table's point --------------------
    assert close(pair_cov("mates", "g", "g", V_A=V_A, V_Y=V_Y, rho_g=rho_g, rho_y=rho_y),
                 rho_g * V_A)
    assert close(pair_cov("mates", "y", "y", V_A=V_A, V_Y=V_Y, rho_g=rho_g, rho_y=rho_y),
                 rho_y * V_Y)
    # the two mixed mate readings agree with each other, and with rho_y V_A
    for a, b in (("g", "y"), ("y", "g")):
        assert close(pair_cov("mates", a, b, V_A=V_A, V_Y=V_Y, rho_g=rho_g, rho_y=rho_y),
                     rho_y * V_A)
    g_to_y = pair_cov("parent-offspring", "g", "y", V_A=V_A, V_Y=V_Y, rho_g=rho_g, rho_y=rho_y)
    y_to_g = pair_cov("parent-offspring", "y", "g", V_A=V_A, V_Y=V_Y, rho_g=rho_g, rho_y=rho_y)
    assert close(g_to_y, 0.5 * V_A * (1 + rho_g)) and close(y_to_g, 0.5 * V_A * (1 + rho_y))
    assert y_to_g > g_to_y, "the asymmetry is signed: rho_g <= rho_y, so y_m--g_o is larger"
    # eq:g-partition -- the parental mean and the residual partition V_A exactly
    assert close(pair_cov("parent-offspring", "g", "g",
                          V_A=V_A, V_Y=V_Y, rho_g=rho_g, rho_y=rho_y)
                 + seg_var(V_A, rho_g), V_A)
    checked += 8

    # -- eq:chain-weight against eq:chain-weight-cases, and D-0008's three worked rows --------
    for N_o in (0, 1, 2):
        for N_mu in range(1, 8):
            if max(0, N_o + 1 - N_mu) == 1 and N_o == 1:
                continue        # N_mu = N_o = 1 is a meiosis, absorbed into a lineal chain
            assert close(chain_weight(N_mu, N_o, rho_g, rho_y),
                         chain_weight_cases(N_mu, N_o, rho_g, rho_y)), f"w({N_mu},{N_o})"
            checked += 1
    assert close(chain_weight(1, 2, rho_g, rho_y), lam_g(rho_g)), "full siblings"
    assert close(chain_weight(2, 2, rho_g, rho_y),
                 (1 + 2 * rho_g + rho_g * rho_y) / 4), "half siblings"
    assert close(chain_weight(3, 2, rho_g, rho_y),
                 rho_g * lam_y(rho_y) ** 2), "step siblings"
    assert close(chain_weight(1, 0, rho_g, rho_y), rho_g), "a bare mating edge carries rho_g"
    checked += 4

    # -- eq:proband-correction ----------------------------------------------------------------
    assert close(proband_correction("mating", rho_g, rho_y), 1.0 / h2), "C_mating = 1/h^2"
    assert close(proband_correction("young", rho_g, rho_y), 1.0)
    assert proband_correction("old", rho_g, rho_y) > 1.0, "rho_y >= rho_g, so the old end lifts"
    checked += 3

    # -- the classes, and the two coincidences the text points at -----------------------------
    cov = lambda name, lv=("g", "g"): class_cov(name, lv, V_A=V_A, rho_g=rho_g, rho_y=rho_y,
                                                V_Y=V_Y)
    assert close(cov("self"), V_A), "the degenerate path is just the variance"
    # the self row is the one place Cov[y, y] is NOT Cov[g, g] times corrections
    assert close(cov("self", ("y", "y")), V_Y), "Var[y_X] is V_Y, not V_A"
    assert close(cov("self", ("g", "y")), V_A), "Cov[g_X, y_X] = Cov[g, g + e] = V_A"
    assert close(cov("parent-parent"), rho_g * V_A), "must agree with tab:g-only-cov"
    assert close(cov("parent-offspring"), 0.5 * V_A * (1 + rho_g)), "ditto"
    # the coincidences: they arrive from different rows of the case split
    assert close(cov("full siblings"), cov("parent-offspring")), "full sibs == parent-offspring"
    assert close(cov("avuncular"), cov("grandparent-grandchild")), "avuncular == grandparent"
    # the phenotypic readings must reproduce tab:g-only-cov's own entries. Two independent
    # routes to the mate-pair row: the path formula reaches it through two mating-chain
    # corrections, the table states it directly -- and rho_g V_A (rho_y/rho_g)^2 = rho_y V_Y
    # only because rho_g = rho_y V_A / V_Y, so this is a real check of the correction, not a
    # restatement.
    assert close(cov("parent-parent", ("y", "y")), rho_y * V_Y), "y_m--y_p"
    assert close(cov("parent-parent", ("g", "y")), rho_y * V_A), "g_m--y_p"
    assert close(cov("parent-offspring", ("y", "g")), 0.5 * V_A * (1 + rho_y)), \
        "the parent at the old end, read phenotypically, is tab:g-only-cov's y_m--g_o"
    # covariance must decay along a lineage, and every class must be positive at rho_y > 0
    assert cov("self") > cov("parent-offspring") > cov("grandparent-grandchild")
    assert all(cov(name) > 0 for name in RELATIONSHIP_CLASSES)
    # the rarer the class, the weaker -- within the sib series, which differ only in N_mu
    assert (cov("full siblings") > cov("half siblings") > cov("step siblings")
            > cov("step-step siblings")), "the sib series must be ordered by mating distance"
    checked += 12

    # -- the worked example of fig:path-example: 2 mating chains, 3 lineal chains -------------
    # the writeup evaluates it to V_A rho_g^2 rho_y lam_g^4 lam_y^2
    lineals, matings = (2, 3, 1), ((2, 0), (3, 2))
    assert sum(lineals) - sum(N_o for _, N_o in matings) == 4, "the text's exponent"
    want = (V_A * rho_g ** 2 * rho_y * lam_g(rho_g) ** 4 * lam_y(rho_y) ** 2)
    assert close(path_cov_g(lineals, matings, V_A=V_A, rho_g=rho_g, rho_y=rho_y), want), \
        "fig:path-example's final line"
    # A is the old end of its lineal chain and B the young end, so exactly one correction
    assert close(path_cov(lineals, matings, ("old", "young"),
                          V_A=V_A, rho_g=rho_g, rho_y=rho_y),
                 proband_correction("old", rho_g, rho_y) * want)
    checked += 3

    # -- rho_y = 0 must collapse everything to the randomly mating case ----------------------
    rm = dict(V_A=VA_0, rho_g=0.0, rho_y=0.0)
    assert close(class_cov("parent-offspring", ("g", "g"), **rm), 0.5 * VA_0), "PO -> 1/2 V_A"
    assert close(class_cov("full siblings", ("g", "g"), **rm), 0.5 * VA_0), "full sibs -> 1/2 V_A"
    assert close(class_cov("grandparent-grandchild", ("g", "g"), **rm), 0.25 * VA_0), "-> 1/4 V_A"
    assert close(class_cov("first cousins", ("g", "g"), **rm), 0.125 * VA_0), "-> 1/8 V_A"
    assert close(class_cov("half siblings", ("g", "g"), **rm), 0.25 * VA_0), "-> 1/4 V_A"
    assert close(class_cov("step siblings", ("g", "g"), **rm), 0.0), "unrelated without assortment"
    assert close(class_cov("parent-parent", ("g", "g"), **rm), 0.0), "ditto"
    checked += 7

    return checked


if __name__ == "__main__":
    n = _self_test()
    print(f"theory.py: {n} internal consistency checks pass")
