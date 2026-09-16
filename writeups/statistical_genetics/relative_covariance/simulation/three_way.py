"""The three legs of `tab:relationship-classes`, compared as ALGEBRA rather than at points.

    python three_way.py

`tab:relationship-classes` can be checked three ways, and each disagreement localizes a
different failure:

| legs that disagree | what it means |
|---|---|
| closed-form formula vs pathMgr oracle | an algebra error in the writeup |
| pathMgr oracle vs simulation | a simulator bug, or a violated assumption |
| all three agree | done |

This module does the first comparison, and it does it **symbolically**. Both the formula leg
and the oracle leg are polymorphic -- popstatgensim's `predicted_cov` is written to accept
anything arithmetic, and pathMgr's `RelativeOracle` returns sympy expressions -- so the two
can be shown **identically equal as functions of `V_A`, `rho_g`, `rho_y`** rather than merely
agreeing at sampled parameter values. That is a strictly stronger claim, and it also removes
the choice of sampling point as a source of false agreement: two expressions can coincide at
a point and differ as functions.

Three independent transcriptions are compared, not two:

- `theory.py` here -- my own hand transcription of the writeup
- `popstatgensim.pedigree.predicted_cov` -- their transcription, driven by a `PathSignature`
- `pathmgr.genetics.RelativeOracle` -- the engine, which derives rather than transcribes

The first two are transcriptions and can share a misreading; the third derives the answer
from a path model, so an identity across all three is what rules that out.

## The `N_mu = 1, N_o = 1` exclusion, which must be applied consistently

`eq:chain-weight-cases` is printed as a general reparameterization but silently excludes the
`N_mu = 1, N_o = 1` cell: a single mating edge with one out-edge is a meiosis, already
carried by a lineal chain. It matters far beyond bookkeeping. For a parent `A` and offspring
`B`, the route `A -- Q -> B` through the parent's mating edge is **formally valid** under the
printed rules, and scores the same `Cov[g] = V_A(1+rho_g)/2` as the direct route -- but it
puts `A` on a terminal mating chain, so `A` takes `C_A = 1/h^2` instead of
`(1+rho_y)/(1+rho_g)`, inflating the phenotypic covariance by about 1.48 and **winning** the
dominant-path comparison.

The trap for this module is not that the route gives a wrong number. It is that if the
exclusion is applied on one leg and not the other, parent--offspring and
grandparent--grandchild disagree between formula and oracle by roughly 50%, and that
disagreement has exactly the signature the three-way table is built to read as "algebra error
in the writeup". So the exclusion's consistency across legs is checked FIRST, before any
class-by-class comparison, and which leg applies it where is reported rather than assumed.

Naming and conventions reconciled here, since all three legs use different vocabularies:
proband cases are `young`/`old`/`mating` in `theory.py` and `sink`/`source`/`terminal_mating`
in the chain-digraph language of `PathSignature`, and the A/B ordering differs with them --
`('sink', 'source')` for parent--offspring means A is the OFFSPRING.
"""

from __future__ import annotations

import sympy as sp

import theory

#: Chain-digraph position -> the writeup's proband case. A `source` has no incoming edge, so
#: it is the OLDEST individual on its lineal chain and takes the old-end correction; a `sink`
#: is the youngest and takes none.
CASE_FROM_DIGRAPH = {"sink": "young", "source": "old", "terminal_mating": "mating"}


def _is_zero(expr) -> bool:
    """Is this difference identically zero, ignoring float-vs-rational contamination?

    `theory.py` is written for NUMERIC use, so its coefficients are Python floats (`0.5`,
    `1.0`), and popstatgensim's predictor is the same. Sympy will not reduce
    `1.0*rho_g - rho_g` to exactly `0`, so a naive `simplify(a - b) == 0` reports a
    disagreement of the form `V_A*(rho_g - rho_g)` for every class -- a false negative that,
    reported as a headline count, would read as a catastrophic algebra failure. Rationalizing
    first is what makes the identity test actually test the identity.
    """
    return sp.simplify(sp.nsimplify(sp.expand(expr), rational=True)) == 0


def _symbols(oracle=None):
    """The basis all three legs are compared in.

    **Must come from the oracle's own model when one is in play.** A freshly built
    `sp.Symbol('rho_g')` does not carry the model's assumptions and therefore does not
    compare equal to the model's symbol of the same name, so a difference between the two
    legs comes out as `V_A*(rho_g - rho_g)` -- an expression that is visibly zero and that
    sympy will not reduce, because the two `rho_g` are different objects. The pathMgr worker
    flagged exactly this when publishing the API, and it is why `oracle.at()` exists rather
    than `.subs()`. Getting it wrong presents as twenty algebra disagreements across every
    class, which reads as a catastrophic failure of the writeup rather than as a symbol
    mismatch -- the same wrong-number-not-an-error pattern as every other defect in this
    endeavor.
    """
    if oracle is not None:
        s = oracle.symbols
        return s["V_A"], s["rho_g"], s["rho_y"]
    return sp.symbols("V_A rho_g rho_y", positive=True)


def signature_for(name: str):
    """Build popstatgensim's `PathSignature` for one writeup class from `theory.py`'s table.

    Deliberately built from MY table rather than looked up in their `KNOWN_SIGNATURES`: if
    both legs read the signature from the same source, the comparison tests only the
    arithmetic downstream of it and not the path description itself, which is the part most
    likely to be mis-transcribed.
    """
    from popstatgensim.pedigree import PathSignature

    spec = theory.RELATIONSHIP_CLASSES[name]
    if spec["cases"] is None:                      # the degenerate self row
        cases = ("sink", "sink")
    else:
        inverse = {v: k for k, v in CASE_FROM_DIGRAPH.items()}
        cases = tuple(inverse[c] for c in spec["cases"])
    return PathSignature(
        sum_N_lin=sum(spec["lineals"]),
        sum_N_o=sum(N_o for _, N_o in spec["matings"]),
        mating_chains=tuple(spec["matings"]),
        proband_A_case=cases[0],
        proband_B_case=cases[1],
    )


def compare_symbolic() -> tuple[int, list[str]]:
    """Assert the formula leg and the oracle leg are identically equal, per class per level.

    Returns the number of identities established and a list of disagreements.
    """
    import pathmgr.genetics as pg
    from popstatgensim.pedigree import predicted_cov

    oracle_by_name = {rc.name: rc for rc in pg.WRITEUP_TABLE_CLASSES}
    checked, bad = 0, []

    print(f"  {'class':>30} {'level':>6}  {'identity':>8}  expression")
    for name in theory.RELATIONSHIP_CLASSES:
        rc = oracle_by_name.get(name)
        if rc is None:
            bad.append(f"{name}: no oracle row")
            continue
        o = rc.oracle()
        # The oracle's OWN symbols, per class -- see `_symbols`.
        V_A, rho_g, rho_y = _symbols(o)
        sig = signature_for(name)

        for level in ("g", "y"):
            phenotypic = level == "y"
            # -- leg B: the engine, which DERIVES rather than transcribes ------------------
            want = o.cov_y(rc.a, rc.b) if phenotypic else o.cov_g(rc.a, rc.b)
            want = sp.simplify(want)

            # -- leg A': their transcription, driven by the signature ----------------------
            # `self` read phenotypically is V_Y, which no path product can produce, so the
            # path formula does not apply to it (their oracle refuses it for the same reason).
            if sig.mating_chains == () and sig.sum_N_lin == 0 and phenotypic:
                print(f"  {name:>30} {level:>6}  {'n/a':>8}  "
                      f"V_Y — degenerate row, no path product applies")
                continue
            got_sim = predicted_cov(sig, V_A=V_A, rho_y=rho_y, rho_g=rho_g,
                                    phenotypic=phenotypic)

            # -- leg A: my transcription ---------------------------------------------------
            got_mine = theory.class_cov(name, (level, level), V_A=V_A,
                                        rho_g=rho_g, rho_y=rho_y)

            ok_mine = _is_zero(got_mine - want)
            ok_theirs = _is_zero(got_sim - want)
            ok = ok_mine and ok_theirs
            checked += 2 if ok else 0
            if not ok:
                which = []
                if not ok_mine:
                    which.append(f"theory.py differs by "
                                 f"{sp.simplify(sp.nsimplify(got_mine - want, rational=True))}")
                if not ok_theirs:
                    which.append(f"predicted_cov differs by "
                                 f"{sp.simplify(sp.nsimplify(got_sim - want, rational=True))}")
                bad.append(f"{name} [{level}]: " + "; ".join(which))
            print(f"  {name:>30} {level:>6}  {'YES' if ok else 'NO':>8}  "
                  f"{sp.factor(want)}")
    return checked, bad


def check_exclusion_consistency() -> list[str]:
    """Is the `N_mu = 1, N_o = 1` cell excluded on BOTH legs, or only one?

    Tests it directly rather than by inspection: build the signature for the route
    `A -- Q -> B` between a parent and their offspring, and ask each leg what it scores. If
    one leg scores it higher than the direct parent--offspring route while the other refuses
    or excludes it, the two legs will disagree on parent--offspring by ~50% for a reason that
    has nothing to do with the writeup's algebra.
    """
    from popstatgensim.pedigree import PathSignature, chain_weight, predicted_cov

    V_A, rho_g, rho_y = _symbols()
    notes = []
    # This check compares two TRANSCRIPTIONS against each other rather than against the
    # oracle, so plain symbols are fine here -- both sides are built from the same objects.

    direct = theory.class_cov("parent-offspring", ("y", "y"),
                              V_A=V_A, rho_g=rho_g, rho_y=rho_y)
    # The offending route: one lineal step, plus a mating chain of (N_mu=1, N_o=1), with the
    # parent sitting on that terminal mating chain instead of at a lineal chain's old end.
    via_mate = PathSignature(sum_N_lin=1, sum_N_o=1, mating_chains=((1, 1),),
                             proband_A_case="sink", proband_B_case="terminal_mating")
    try:
        scored = predicted_cov(via_mate, V_A=V_A, rho_y=rho_y, rho_g=rho_g, phenotypic=True)
        ratio = sp.simplify(scored / direct)
        notes.append(f"their predicted_cov SCORES the (1,1) route: {sp.factor(scored)}")
        notes.append(f"  ratio to the direct parent-offspring route: {sp.factor(ratio)}")
        num = float(ratio.subs({rho_y: sp.Rational(1, 2), rho_g: sp.Rational(1, 4)}))
        notes.append(f"  at rho_y=0.5, rho_g=0.25 that is {num:.4f}x "
                     f"({'WINS the dominant-path comparison' if num > 1 else 'loses'})")
    except Exception as exc:
        notes.append(f"their predicted_cov REFUSES the (1,1) route: "
                     f"{type(exc).__name__}: {exc}")

    # And the weight function itself at the excluded cell, on both transcriptions.
    try:
        w_theirs = chain_weight(1, 1, rho_y=rho_y, rho_g=rho_g)
        notes.append(f"their chain_weight(N_mu=1, N_o=1) = {sp.factor(sp.simplify(w_theirs))}")
    except Exception as exc:
        notes.append(f"their chain_weight(1, 1) REFUSES: {type(exc).__name__}: {exc}")
    w_mine = theory.chain_weight(1, 1, rho_g, rho_y)
    notes.append(f"my   chain_weight(N_mu=1, N_o=1) = {sp.factor(sp.simplify(w_mine))}")
    notes.append("Both transcriptions COMPUTE the cell -- eq:chain-weight's binomial form is "
                 "defined there. The exclusion is a claim about which PATHS are admissible, "
                 "not about the weight function, so it has to live in path enumeration.")
    return notes


def main() -> None:
    print("=== 1. is the N_mu=1, N_o=1 cell handled consistently across legs? ===")
    for line in check_exclusion_consistency():
        print("   ", line)

    print("\n=== 2. algebraic identity, per class per level ===")
    print("    formula legs (theory.py and predicted_cov) vs the pathMgr ORACLE,")
    print("    as expressions in (V_A, rho_g, rho_y) -- not at sampled points\n")
    checked, bad = compare_symbolic()
    print(f"\n  {checked} algebraic identities established")
    if bad:
        print(f"  {len(bad)} DISAGREEMENT(S):")
        for b in bad:
            print(f"    {b}")
        # EXIT NON-ZERO. Printing a disagreement and exiting 0 is the D-0021 failure mode:
        # a caller that gates on this module's status would sail past a real mismatch, and a
        # human skimming the tail of the output would see the identity table and miss the
        # footer. A check that cannot fail is not a check.
        raise SystemExit(f"{len(bad)} disagreement(s) between the formula and oracle legs")
    print("  no disagreements: all three transcriptions are identically equal")


if __name__ == "__main__":
    main()
