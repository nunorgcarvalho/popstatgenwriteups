# relative_covariance — restructure plan (working note, 2026-10-07)

Temporary note to self. Condensed from a voice brainstorm, then revised over several rounds of
decisions and a section-by-section read-through of the current document. It captures the
target structure and the ideas behind it so the rewrite can be implemented in pieces.
Everything outside **Open** and **To verify** is settled. Nothing here is in the `.tex` yet.

## The core idea

Build the document **from the roots up**. A new first section sets up a framework that
everything downstream refers to strictly, so later sections can be short. The framework has
three parts:
1. an explicit notation for **what information is known** about a pair, and how the kinds of
   information relate to each other;
2. the two **fundamental processes**, meiosis and mating, written as explicit assumptions;
3. definitions of the **measures of relatedness** built on those processes.

The current "ladder / tower" of Levels 0–3 is replaced by **exploring combinations of
information**: what can you say given only the path, given the path plus transmission, given
the path plus genotypes, and so on.

## Notation

**Convention.**
- A **curly (calligraphic) letter** is the *full* information of a kind.
- A **bold letter** is the concrete object we work with, typically a matrix.
- **Functions are plain $f$.** There are no special letter styles for functions and no curly
  function symbols.
- Change as little of the existing practical notation as possible:
  - $g_i$ stays the genetic value;
  - $x_{ik}$ stays the standardized genotype;
  - $z$ stays the standardized allele;
  - $y = g + e$ is unchanged.

| kind | full (curly) | concrete (bold) | notes |
|---|---|---|---|
| Genetic data | $\mathcal{G}$: all true genetic data of the individuals **in the study** (sample / population under study) | $\mathbf{G}$: the raw genotype matrix (allele counts 0/1/2). Entries stay **capitalized**, $G_{ik}$, so that lowercase $g_i$ remains the genetic value. $\mathbf{X}$: the standardized genotype matrix, entries $x_{ik}$ | $\mathcal{G}$ does **not** include ancestors outside the study. Their contribution arrives through the pedigree. |
| Pedigree / path | $\mathcal{P}$: the **pedigree**, i.e. the full history of descent and mating **back to a fixed depth** (some number of generations): everyone involved in any mating, all their offspring, all ancestors and descendants within that window. Its oldest members are the **founders**. | $\mathbf{P}$: the **path**, an ordered list of mating and lineal **sequences** between two people | Every current $\mathcal{P}$ (all of which mean *path*) is renamed $\mathbf{P}$. The path carries only the familial relationship, with no variant-specific information. |
| Transmission | $\mathcal{T}$: **backward-looking**, in the spirit of coalescent theory. For every allele, at every variant, it traces whom that allele was inherited from, back through the past: its ancestral lineage. | $\mathbf{T}$: open | Variant-specific information about where each allele came from. Phasing and parent-of-origin assignment are the first step back along a lineage. True IBD is two lineages meeting. IBD-inference output constrains where lineages meet. All of these are constraints on $\mathcal{T}$. |
| Environment | $\mathcal{E}$: all environmental information, including random noise | $e$ | |
| Parameters | — | $\beta_k$, $\rho_y$, $V_E$, ... | **Not information about the pair.** Parameters are a separate category from data-information. Level 0 and Level 1 of the current Section 1 differ only in whether $\beta$ is a known parameter or a random effect. |

**Renames.**
- **$c_k \to b_k$** everywhere: $b_k := \mathrm{Cov}[z^{(u)}_{ik}, y_i]$.
  - Reason: $b_k$ is a **marginal** covariance, the kind of quantity a naive marginal (OLS)
    regression estimates, in contrast to the joint effect $\beta_k$.
  - It is also really a genotype-level quantity: $\mathrm{Cov}[x_{ik}, g_i] = 2b_k$, so no
    $\mathcal{T}$ is needed despite its allele-level definition.
  - Nuance worth a clause: the OLS slope of $y$ on $x_k$ is
    $\mathrm{Cov}[x_k,y]/\mathrm{Var}[x_k] = 2b_k/(1+2a_{kk})$, so $b_k$ is the marginal
    covariance, and equals half the OLS slope only up to the $a_{kk}$ factor.
  - Side benefit: this frees $c$. The current text also uses $c(N_\mathcal{M})$ for the
    correlation function inside the mating-sequence weight (`eq:chain-weight`).
  - The existing `\bb` macro ("tested effect size vector") already fits this reading.
- **Mating chain → mating sequence; lineal chain → lineal sequence.**
  - The symbols $\mathcal{M}$, $\mathcal{L}$, $N_\mathcal{M}$, $N_\mathcal{L}$ are unchanged.
  - This resolves the naming collision with Wright's tracing rules, which keep the word
    **chain** (a "valid chain" through a path diagram).
  - It follows that the "chain digraph" becomes the **sequence digraph**, and every
    "chain" in the pedigree material becomes "sequence".

Other notation rules:
- **Estimand, not estimate.** Everything is derived assuming perfect knowledge of whatever we
  claim to know. Hats ($\hat{\cdot}$) are reserved for estimates, if they are ever added.
- **Genotypes are the core level.**
  - Without phasing, an allele carries nothing beyond its genotype (the "average allele" is
    $x/2$).
  - Speaking of a *maternal* or *paternal* allele $z^{(m)}, z^{(p)}$ therefore **requires
    transmission information** ($\mathcal{T}$), as in real data, where phasing is always
    inferred.

## How the kinds of information relate (goes in §1.1)

They are not independent axes. They partly determine one another, and §1.1 should say how
with a small **information-flow diagram**.
- **$\mathcal{T}$ and $\mathcal{P}$ overlap; neither contains the other.**
  - $\mathcal{T}$ is backward-looking. Tracing an allele's lineage reveals the ancestors it
    passed through, i.e. the part of the pedigree that matters for that allele.
  - $\mathcal{T}$ does not give the rest of the pedigree. In particular it does not give
    matings, nor relatives who carry none of the traced alleles.
  - The pedigree does not give $\mathcal{T}$: it says who could have passed an allele on,
    not who did.
  - In the coalescent sense, $\mathcal{T}$ is meaningful even with no explicit pedigree.
  - Meiosis (§1.3) is the forward process; $\mathcal{T}$ is the backward record of its
    outcomes.
- **Mating information lives only in $\mathcal{P}$.**
  - A backward lineage never records a mating.
  - Under random mating, mating information carries nothing genetic.
  - Under assortative mating, mates (and relatives along a mating sequence) are correlated
    without sharing any lineage. Mating information then carries **all** of assortment's
    cross-variant covariance.
  - This is why mating stands beside meiosis as its own fundamental process.
  - It is also the reason behind the key 2.4 conclusion: the path cannot be dropped once
    $A_{ij}$ or $\pi_{ij}$ is known. $A_{ij}$ is a function of $\mathcal{G}$, $\pi_{ij}$ of
    $\mathcal{T}$, and neither sees the mating information.
- **$\mathcal{T}$ and $\mathcal{G}$ are closely connected but not equivalent.**
  - Transmission together with the genetic data of ancestors largely generates the genetic
    data of descendants.
  - Mutation (not modeled, but conceptually possible) can change $\mathcal{G}$ in a way
    $\mathcal{T}$ does not record.
  - $\mathcal{G}$ covers only the individuals in the study. The pedigree and transmission add
    information about everyone else.
- **Information-approximation assumptions are a recurring pattern.**
  - Each analysis implicitly assumes that some information can be obtained from, or is
    adequately summarized by, another.
  - The first instance is **path sufficiency** (§1.4).

## Phenotype model as constraints on unknown functions

- In general the genetic value is some unknown function $f$ of the available information.
- The phenotype first splits into a genetic part plus an environmental part, $y = g + e$,
  with $e$ drawn from $\mathcal{E}$.
- **Additivity** constrains the genetic function to be linear in the genotypes through
  $\beta$ only, $g_i = \sum_k \beta_k x_{ik}$. In particular it depends on neither
  transmission, the pedigree, nor the environment.
- Each assumption is therefore a precise statement about what an unknown function may depend
  on. Later extensions (GxE, parent-of-origin, indirect effects) are relaxations of these
  constraints.
- **Separate causal constraints from independence statements.**
  - GE-indep, No-IGE and Env-indep each combine two kinds of claim:
    - a **domain** constraint: $e_i$ depends only on $\mathcal{E}_i$, the individual's own
      environment;
    - an **independence** statement about the inputs, e.g. $\mathcal{E}_i$ independent of
      $\mathcal{G}$ and of $\mathcal{E}_j$.
  - **Assortative mating changes no function's domain.** It correlates the inputs, because
    mating selects on $y$.
  - This makes precise the current text's "breaks the weakest forms of No-IGE and
    Env-indep". It is also the source of the proband corrections $C_X$ and of the asymmetry
    between parent–offspring covariances read at $g$ versus $y$.

## Target structure

### Section 1 (NEW), working title "Measuring relatedness"

A **running example** carries through the whole section so that each definition lands on
something concrete. It uses one pedigree containing **both a full-sibling pair and a
step-sibling pair**, extending the current step-sibling figure.

#### 1.1 Information and notation
- $\mathcal{G}/\mathbf{G}/\mathbf{X}$, $\mathcal{P}/\mathbf{P}$, $\mathcal{T}$, $\mathcal{E}$;
  parameters as a separate category.
- The curly-vs-bold convention and functions as $f$.
- The information-flow diagram, including where mating information sits.

#### 1.2 Population baseline
- Reference (do not rebuild) the simple random-mating model in which alleles are drawn
  independently from one pool (Hardy–Weinberg / random union of gametes).
- Define an **allele frequency** only as what an allele drawn at random from the population
  tends to be.
- Standardize $\mathbf{G}$ into $\mathbf{X}$ (and alleles into $z$) under HWE with the true
  frequency.
- **Define $a$ and $s$ here, as descriptors of departure from the baseline.** They are zero
  under HWE and linkage equilibrium.
  - $a_{kk}$ is **excess homozygosity** (departure from HWE), with
    $\mathrm{Var}[x_{ik}] = 1 + 2a_{kk}$.
  - For $k \neq l$, $a_{kl}$ is **non-gametic** and $s_{kl}$ **gametic** disequilibrium.
  - **Observability:** separating $s$ from $a$ requires $\mathcal{T}$. Genotypes alone show
    only $2(s+a)$.
- Assortative mating is later studied as a **deviation** from this baseline.

#### 1.3 The two fundamental processes
Each is written as explicit assumptions.

**Meiosis.**
- Random 50/50 segregation of the parent's maternal vs paternal chromosome, with
  recombination switching between them. Formalized with the existing transmission indicators.
- The **segregation residual** is defined here, on this explicit foundation. Its variance
  $\tfrac14 - \tfrac12 a_{kk}$ also belongs here, since it holds for any population.
- **Gametic disequilibrium under meiosis**, written with a recombination fraction $r_{kl}$:
  $$s^{(t+1)}_{kl} = (1-r_{kl})\,s^{(t)}_{kl} + r_{kl}\,a^{(t)}_{kl}.$$
  Unlinked is the special case $r_{kl} = \tfrac12$, i.e. the current recursion. See
  **To verify** for what this does to the linkage `\unsure`.
- **Constant-freq mostly follows from fair segregation.**
  - The 50/50 assumption means no meiotic drive, so $\mathrm{E}[z_o] = \tfrac12\mathrm{E}[x_m]$
    preserves the expected frequency.
  - What remains to assume is "no selection, no drift".

**Mating.**
- A separate fundamental act: relatedness can arise from mating without any offspring.
- State how mating is encoded.
- Move here the current remark that "mates" means any pair assorting on the phenotype with
  strength $\rho_y$, including non-reproducing and same-sex pairs.

**Processes ↔ path-diagram edges.** The two processes are exactly the two special edge types
of path diagrams:
- meiosis is the $\tfrac12$ edges plus the segregation self-loop;
- mating is the co-path.

§1.3 defines the processes as structural equations, and §1.4's path diagrams then just draw
them.

#### 1.4 Pedigrees, paths and path diagrams
- **Path diagrams** and Wright's tracing rules (current 2.2.1).
  - Define the co-path **generically**, as matching on some variable. §3 makes it specific to
    $y$ under Primary-AM.
- **The graph material** (current 2.3.2–2.3.5): mating graph, meiosis graph, mating
  sequences, lineal sequences, tracing paths, the sequence digraph.
  - The genetic-value-level path diagram (`fig:g-only`) is the pedigree graph with
    coefficients attached. Say so; it ties the two halves of this subsection together.
- **Path sufficiency**, one assumption replacing Outbred-mating-chain and Dominant-path.
  - Model-free statement: $\mathrm{E}[\,\cdot \mid \mathcal{P}] \approx \mathrm{E}[\,\cdot \mid \mathbf{P}]$.
    The information the whole pedigree carries about the data-generating process for $y$ is
    approximated by a single path.
  - It has two halves, one per fundamental process:
    - **meiosis/coalescence half** (the current Outbred-mating-chain): all coalescences
      within the window happen at ancestors on the path's lineal sequences;
    - **mating half** (the current Dominant-path): other mating routes are negligible.
  - The covariance-based justification (path factors below 1) comes later, in §3.
- **No-inbreeding becomes exact** under the pedigree window. It replaces the current "a couple
  generations back".

#### 1.5 Measures of relatedness
- **Allelic sharing** $A_{ij}$, a function of genotypes.
- **IBD at variant $k$**, defined as a function of $\mathcal{T}$: two alleles are IBD when
  their traced lineages **coalesce within the pedigree's time frame**, i.e. meet at a common
  ancestral allele no older than the founders.
- **Realized relatedness** $\pi_{ij}$, a summary of $\mathcal{T}$.
- **No-mutation** is restricted to the same window: no mutations occur within the pedigree's
  time frame, so IBD ⇒ IBS.
- **No-inbreeding:** an individual's two alleles do not coalesce within the window.
- **Founders:**
  - Alleles whose lineages do not coalesce within the window reach distinct founder alleles.
  - Founders are draws from **the population**, whose distribution depends on the mating
    regime.
    - Under random mating they are §1.2 baseline draws (HWE, linkage equilibrium).
    - Under assortative mating they are draws from the equilibrium population ($a, s \neq 0$).
  - Under random mating, Non-IBD-indep therefore splits into:
    - (a) founder alleles are independent baseline draws;
    - (b) matings inside the window are independent of genotypes.
  - §3 relaxes only (b).

### Section 2 (current Section 1, rewritten): additive genetics under random mating
- Formalized with the new framework; the phenotype model is stated as constraints on $f$.
- **The cross-product object $C_{ij}(\mathcal{I})$:**
  - $\mathcal{I}$ is a choice of information from $\{\mathbf{P}, \mathcal{T}, \mathcal{G}\}$
    for the pair.
  - "Cross-product vs covariance" becomes: it is a covariance unless the information fixes $g$,
    which takes $\mathcal{G}$ together with a known $\beta$.
- **Level 0 → "$\mathcal{G}$ with known $\beta$".** "Needs no relatedness information" becomes
  a one-line consequence of Additivity: once $f$ depends only on $x$, genotypes screen off
  $\mathcal{P}$ and $\mathcal{T}$.
- **Level 1 → "$\mathcal{G}$ with random $\beta$".** Genotype-independent-effects becomes "the
  information does not inform $\beta$".
- **Level 2 → "$\mathcal{T}$, summarized by $\pi_{ij}$".** Same derivation, with IBD now
  defined as coalescence within the window.
- **Level 3 → "$\mathbf{P}$ only".** **The Genetic-degree assumption is replaced by a
  derivation.**
  - The §1.3 meiosis process (each lineage follows a given parent with probability
    $\tfrac12$) gives $\mathrm{E}[\pi_{ij}\mid\mathbf{P}]$ from the path. For example, full
    siblings have two source ancestors at $\tfrac14$ each.
  - This fills in the current "$2^{-d}$, not fully derived".
- **Fix the tower/screening conflation in the current Summary.**
  - The tower property $\mathrm{E}[\mathrm{E}[\cdot\mid\mathcal{I}_2]\mid\mathcal{I}_1] = \mathrm{E}[\cdot\mid\mathcal{I}_1]$
    holds whenever $\mathcal{I}_1 \subseteq \mathcal{I}_2$, i.e. it needs only nesting.
  - **Screening** is the separate property that the finer information alone suffices, e.g.
    $\mathrm{E}[y_iy_j\mid\pi_{ij},\mathbf{P}] = \mathrm{E}[y_iy_j\mid\pi_{ij}]$.
  - Under random mating, screening holds. Under assortative mating the tower property still
    holds but screening fails, because mating information (only in $\mathcal{P}$) becomes
    genetically informative.
  - Section 2 should set up exactly this distinction, so §3 can point at it.
- **The Level 0–3 summary table becomes a grid.**
  - One axis is the **target**: $y$, $g$, $x$, $z$.
  - The other is the **information**: $\mathbf{P}$, $+\mathcal{T}$, $+\mathcal{G}$.
  - The allele ($z$) row is empty wherever $\mathcal{T}$ is unknown. Those structural blanks
    visualize "genotypes are the core level".

### Section 3 (current Section 2): assortative mating, much more concise

**Mating model.**
- Primary-AM: the co-path of §1.4 specialized to $y$.
- Its effect in framework terms: matings in $\mathcal{P}$ become dependent on $\mathcal{G}$
  and $\mathcal{E}$. That is the relaxation of Non-IBD-indep (b).
- In the same terms, $\mathcal{G}$ now leaks $\beta$, which breaks Genotype-independent-effects.

**Dynamics and equilibrium.**
- Go straight in, since the processes and path diagrams already exist; show quickly what this
  looks like on a real family.
- The generation-1 subsection and its relatives table shrink to an illustration (generation 1
  is just the $t = 0 \to 1$ step of the recursion).
- The equilibrium results stay as they are, in $b_k$ notation.

**Covariance between relatives at the level of genetic values and phenotypes.**
- Jump straight to the formulas, since sequences and paths are already defined.
- The genetic-value transmission equation $g_o = \tfrac12 g_m + \tfrac12 g_p + g^\perp_o$
  becomes one line: meiosis plus Additivity.
- The covariance-based justification of path sufficiency goes here.

**Current 2.4 (genotypes/alleles).**
- Cut the build-up, trust the tools, and give the final formulas plus the interpretive points.
- Organize by information set, each across target levels per the grid:
  1. **Path only.**
     - The genotype covariance between relatives given the path (`eq:x-rel`), its
       genotype-to-genetic-value companion (`eq:x-g-rel`), and the examples table.
     - "Neither $A_{ij}$ nor $\pi_{ij}$ sees $\Delta_{ij}$" becomes nearly definitional
       ($\Delta_{ij}$ comes from mating information). The mates example is the cleanest
       illustration.
  2. **Path + transmission.**
     - The conditional-on-meioses results. The full-sibling derivation already conditions on
       part of $\mathcal{T}$.
     - **Condition on the full $\mathcal{T}$, not only IBD.** IBD is a coarsening of
       $\mathcal{T}$.
       - Trace each allele back to its **source ancestor**.
       - Two alleles' covariance is then either the IBD term (same source allele) or set by
         the mating distance between the two source ancestors, via `eq:z-rel`.
       - This is the general correction term the half-cousin / great-grandparent `\unsure`
         says is missing. The IBD-only formula is the special case where source identity
         does not matter.
  3. **Path + genotypes.**
     - The $A_{ij}$ `\todo`, framed as the $\beta$-leak.
     - Transmission comes before genotypes because meioses are independent of genotypes, so
       conditioning on them stays a second-moment calculation. Conditioning on genotypes is
       the fourth-moment problem.

### Later / maybe
- Finer sub-categories of transmission information (phasing, parent-of-origin, IBD segments,
  ...).
- GxE and other extensions as relaxations of the function constraints.
- Estimates with hat notation.
- Generalizing §1 beyond "measuring relatedness".

## Where current material moves (first pass)

| current | destination |
|---|---|
| 1.1 phenotype model, assumptions | §2, re-cast as constraints on $f$ (domain vs independence) |
| 1.2 the object $C_{ij}(\mathcal{I})$ | §2, with $\mathcal{I}$ drawn from the information categories |
| 1.4–1.5 Levels 1–2: $A_{ij}$, IBD, $\pi_{ij}$, No-mutation, Non-IBD-indep, No-inbreeding | definitions → §1.5; random-mating results stay in §2 |
| 1.6 Level 3, Genetic-degree | §2, with $\mathrm{E}[\pi\mid\mathbf{P}]$ derived via §1.3 (assumption dropped) |
| 1.7 ladder summary | §2, as the target × information grid, with tower vs screening separated |
| 2.2.1 Path analysis | §1.4 (co-path generic) |
| 2.2.2 "Building the model from haploid alleles", Constant-freq, standardization | §1.2 (Constant-freq mostly → §1.3 fair segregation) |
| 2.2.3 $a$, $s$ definitions | §1.2, as HWE / disequilibrium descriptors |
| 2.2.3 transmission / segregation deviation (`eq:transmit`, `eq:seg-def`) and its variance | §1.3 meiosis |
| 2.2.3 $c_k$ definition | §3, renamed $b_k$ |
| 2.2.3 generation-1 results, `tab:relatives-gen1` | §3, shortened to an illustration |
| Unlinked, `eq:s-recursion` | §1.3 meiosis, written with $r_{kl}$ |
| 2.3.1 genetic-value transmission | §3, one line from §1.3 + Additivity |
| 2.3.2–2.3.5 framework, mating/lineal chains, path tracing | §1.4, as mating/lineal **sequences** and the sequence digraph |
| "mates" includes non-reproducing / same-sex pairs (2.3.2) | §1.3 mating |
| Outbred-mating-chain, Dominant-path | merged into path sufficiency: statement → §1.4; covariance justification → §3 |
| step-sibling figure | §1 running example, extended with a full-sibling pair |
| 2.4.x | §3, restructured by information set |

## To verify

1. **Linkage leaves the equilibrium unchanged.**
   - The claim: with the $r_{kl}$ recursion above, the cross-gamete term
     $a^{(t+1)}_{kl} = \mu\,b_kb_l$ is unaffected by linkage (it passes only through the
     co-path). So the fixed point is $s_{kl} = a_{kl}$ for **any** $r_{kl} > 0$. Linkage
     changes the speed of convergence, not the endpoint.
   - Consequences if it holds:
     - The current linkage `\unsure` is wrong to say linkage makes $s_{kl} \neq a_{kl}$ at
       equilibrium.
     - The full-sibling conditional-on-meioses derivation uses only $s_{kl} = a_{kl}$, so those
       results extend to a linked genome at equilibrium. That is exactly where $\pi_{ij}$ varies
       enough to be worth conditioning on.
     - The cost moves to AM-equilibrium: tightly linked pairs need about $1/r$ generations,
       far more than "six to ten".
   - Check numerically (small $M$ with set $r_{kl}$) before relying on it.
2. **The source-ancestor formula conditional on $\mathcal{T}$.** Check against the half-cousin
   and great-grandparent cases in `make_figures.py`.

## Open

1. **Bold $\mathbf{T}$**: what concrete object represents transmission. Candidates: the array
   of transmission indicators $B$ (forward), or each allele's source ancestor (backward).
2. **Telling several functions apart**: plain $f$ for the genetic value, but how do we write
   the phenotype and environment functions (subscripts such as $f_g$, $f_e$, or only $f$ when
   there is one)?
3. **Generative model for $\mathbf{G}$, or only definitions?** Leaning: define only what is
   needed (allele frequency for random draws, HWE standardization).
4. **`\bA` clash**: it is both the GRM (Section 1) and the directed-edge matrix of path
   analysis (2.2.1). Rename one during the restructure.
5. **Section 1 title**: "Measuring relatedness" is the working title.
