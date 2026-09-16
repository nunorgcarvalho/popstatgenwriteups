# Simulation verification of `relative_covariance`

Verification of the assortative-mating results in section 2 of `relative_covariance.tex`
against simulation. The reader is assumed to have the writeup's PDF open in parallel: nothing
here re-derives or re-explains a formula, it names the equation being checked, restates that
one formula, and shows the result.

## The three legs, and which one is actually missing

Each result in section 2 can be checked three ways, and each disagreement localizes a
different failure:

| legs that disagree | what it means |
|---|---|
| closed-form formula vs pathMgr oracle | an algebra error in the writeup |
| pathMgr oracle vs simulation | a simulator bug, or a violated assumption |
| all three agree | done |

Two of the three legs already existed before this folder, which is worth knowing before
adding to it:

- **The pathMgr oracle leg is built and green.** `../figures/make_figures.py` drives pathMgr's
  RAM engine over a pedigree carrying every standard relationship class and asserts the
  writeup's hand-typed tables and formulas against it -- 24 check groups, ~1500 agreeing
  results. Run `python ../figures/make_figures.py --check`.
- **The closed-form leg is `theory.py` here**, a numeric transcription of the same equations
  for the notebooks to call, cross-checked against that oracle by
  `check_theory_vs_oracle.py` (106 numeric comparisons).

For `tab:relationship-classes` the two algebraic legs turn out to be **one statement, proved**:
`three_way.py` shows the closed form and the oracle are *identically equal as expressions* in
`(V_A, rho_g, rho_y)` for all eleven rows at both levels. An identity is a proof, not
evidence, so the three-way table collapses to a two-way one and the notebook says so rather
than presenting the same statement twice as corroboration. What makes it meaningful is which
implementations it spans: `theory.py` and popstatgensim's `predicted_cov` are independent
**transcriptions** (which could share a misreading of the prose), while pathMgr's oracle
**derives** the answer from a path model, which is what rules that out.

So the formula-vs-oracle diagnosis above is already answered for every class the oracle
covers: there is no algebra error. What none of it establishes is anything about the
**simulator**. Every existing check compares the writeup's algebra to a path model built from
the same assumptions -- self-consistency, not a test against data. Nothing tests whether
allele-level Monte-Carlo at finite `N`, with real genotypes and a rank-matching-copula pairing
mechanism rather than a bivariate normal, reproduces the equations. **That is the open leg and
the purpose of this folder.**

`step-step siblings` is the one row with **no simulation leg, by design**: an `N_mu = 4`
mating chain needs three consecutive multi-mate individuals, so the class scales as `q_2^3`
and cannot be sampled at feasible `N`. The designed-pedigree mode that would fix it is
deliberately out of scope. pathMgr now covers that row (and `self`) on the oracle leg, so it
rests on an identity between a derivation and two readings of it, with no independent check
against data -- which the notebook states plainly rather than stretching a handful of pairs
into a claimed confirmation.

`self` is a **measurement at every level on all three legs, never a test**: `Var[g]` is `V_A`
and `Var[y]` is `V_Y` once those are measured. pathMgr's oracle refuses its phenotypic
reading, which is the same judgement expressed as code.

## Layout: generate, then verify

Data generation and verification are separate stages, because the heavy runs must not execute
inline -- a notebook that has to simulate ten generations before it can plot anything is a
notebook nobody can run.

```
simulation/
  theory.py                  the writeup's closed-form predictions, as numeric functions
  check_theory_vs_oracle.py  theory.py vs the pathMgr oracle in ../figures/make_figures.py

  generate_populations.py    GENERATE: builds Populations per regime, saves them to scratch
  pair_structure.py          base-generation cross-mate structure (tab:pair-2v-between)
  equilibrium.py             the equilibrium tier (eq:rhog-eq, eq:VA-eq, eq:g-seg-var, ...)
  relatives_pair.py          tab:g-only-cov at equilibrium, and the parent--offspring asymmetry
  assumptions.py             Constant-freq and No-inbreeding diagnostics
  three_way.py               formula == oracle as an ALGEBRAIC IDENTITY, all 11 classes
  classes_sim.py             the simulation leg of tab:relationship-classes

  build_notebook.py               generates verify_equilibrium.ipynb
  build_classes_notebook.py       generates verify_relationship_classes.ipynb
  verify_equilibrium.ipynb        VERIFY: section 2.2, loads saved artifacts, never simulates
  verify_relationship_classes.ipynb   VERIFY: section 2.3, same
```

Each analysis module runs standalone (`python equilibrium.py`, `python assumptions.py
equilibrium-midM`, ...) and prints the same numbers the notebook shows, so a result can be
reproduced or re-checked without opening a notebook at all.

The notebook is **generated** rather than hand-edited, so that its narrative and the code it
runs cannot drift apart, and so regenerating it is a clean diff rather than a churn of
execution counts. Edit `build_notebook.py`, not the `.ipynb`.

```sh
python generate_populations.py --list                        # what regimes exist
python generate_populations.py pair-structure --reps 50       # generate
python pair_structure.py                                     # check one tier
python build_notebook.py --run                               # build and execute the notebook
```

- **generate** is a *script*, not a notebook: it is the thing that might take an hour, and it
  has no output worth reading inline. One regime per invocation, saving to scratch.
- **verify** is where the notebooks live. A verify notebook loads saved objects and never
  simulates. It may be beefy -- it doubles as a worked example of the simulator -- and should
  use plots wherever a plot makes a comparison clearer than a number.

Heavy artifacts go under `/n/scratch/users/n/nur479/` (prunable, not backed up), never into
the repo; `.gitignore` here is a backstop. Each notebook records the scratch path it read.

## Rules that bind this folder

Settled decisions, not preferences (thesisMgr `DECISIONS.md` D-0007 through D-0009):

- **Predictions are always evaluated at MEASURED `rho_y`, `rho_g`, `V_A`, `V_Y`**, never the
  nominal simulation inputs. The pairing mechanism matches on ranks, so realized `rho_y`
  differs from the nominal `AM_r` and realized `rho_g` from `rho_y h^2`; that deviation is a
  quantity worth reporting, not a bug to hide. `theory.py`'s signatures are built to make
  passing a nominal value awkward.
- **`V_A` is measured against a pinned base-generation standardization** (the writeup's
  `Constant-freq`), not per-generation observed standardization.
- `AM_type='phenotypic'` throughout (`'genetic'` is genetic homogamy, excluded by
  `Primary-AM`), and no selection (`s = 0`).
- **Drift is assumed absent.** Show the diagnostic once to document that the assumption holds
  at the `N` and generation counts used; do not correct for it.
- **Several populations tuned per relationship class**, not one population for everything --
  rare classes cannot be sampled from a single realistic pedigree.
- **Quantify agreement.** Predicted vs observed with a standard error or CI, not a visual
  impression.
- **A mismatch is a finding about the writeup, reported with numbers.** Do not tune parameters
  or reclassify pairs until it goes away.
- **Never display individual-level data.** Aggregate summaries only -- distributions,
  per-generation summaries, class means -- never per-individual values or identifiers.

## `theory.py`

The closed-form leg. Pure numeric Python: no pathMgr, no sympy, no popstatgensim, so a verify
notebook can import it without the symbolic stack. Each function is named for the equation it
transcribes.

```sh
python theory.py                    # 71 internal consistency checks
python check_theory_vs_oracle.py    # 104 comparisons against the pathMgr oracle
```

The equilibrium tier (`rho_g_eq`, `VA_eq`, `h2_eq`, `seg_var`), the pair tier (`pair_cov`, for
`tab:g-only-cov`), and the relationship-class tier (`chain_weight`, `proband_correction`,
`path_cov`, `class_cov` over `RELATIONSHIP_CLASSES`) are all there.

One function deserves attention when reading results: **`VA_eq_relative_error`**. `eq:VA-eq`
is an *approximation*, and `eq:VA-eq-correction` gives the term it drops as a relative error
of `rho_g / (2 (1 - rho_g) M_e)`, with the sign stated -- dropping it **overstates**
`V_A^(eq)`. So the simulation should sit predictably *below* `eq:VA-eq` by a computable
amount that shrinks like `1/M_e`. Checking that, rather than asserting `eq:VA-eq` within a
Monte-Carlo tolerance, tests the approximation instead of absorbing it into the error bars,
and only needs a couple of values of `M`.

`chain_weight` implements `eq:chain-weight` as the binomial sum and is the single source of
truth for a mating chain's weight; the `N_S` case split of `eq:chain-weight-cases` is
presentational and is never used to compute anything (D-0008). `chain_weight_cases` exists
only so the self-test can assert the two agree.

### Where the importable oracle lives

`check_theory_vs_oracle.py` currently reaches into `make_figures.py`'s internals (`_pedigree`,
`PATH_CLS_CASES`), because that script keeps its oracle as assertions inside a figure
generator rather than behind an API, and the three-way table needs to *call* an oracle per
class and get an expression back.

That is being fixed **in pathMgr**, not here: the importable oracle is packaged there, and
`make_figures.py` stays a read-only reference. So nothing in this repo changes, and the
notebooks will import the oracle from pathMgr rather than from the figure generator. When
that lands, this file's imports move and the private-name reach-through goes away.

The two rows that had no oracle coverage -- `self` and step-step siblings -- were part of
that same pathMgr work, and both are now covered: `self` by refusing its phenotypic reading
(the correct answer), and step-step siblings on two independent hand-built pedigrees.
