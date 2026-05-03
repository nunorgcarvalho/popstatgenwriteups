# Population & Statistical Genetics Writeups

This repository contains LaTeX writeups with derivations and explanations for selected population/statistical genetics papers, concepts, and more general statistics topics.

## Writeups

### Statistical Genetics

- [Heritability](writeups/statistical_genetics/heritability/heritability.pdf) ([source](writeups/statistical_genetics/heritability/heritability.tex))
- [Mixed Linear Models](writeups/statistical_genetics/mixed_linear_models/mixed_linear_models.pdf) ([source](writeups/statistical_genetics/mixed_linear_models/mixed_linear_models.tex))

### Statistics

- [Bayesian Statistics](writeups/statistics/bayesian_statistics/bayesian_statistics.pdf) ([source](writeups/statistics/bayesian_statistics/bayesian_statistics.tex))
- [Linear Regression](writeups/statistics/linear_regression/linear_regression.pdf) ([source](writeups/statistics/linear_regression/linear_regression.tex))
- [Maximum Likelihood](writeups/statistics/maximum_likelihood/maximum_likelihood.pdf) ([source](writeups/statistics/maximum_likelihood/maximum_likelihood.tex))
- [Random Effects](writeups/statistics/random_effects/random_effects.pdf) ([source](writeups/statistics/random_effects/random_effects.tex))
- [Random Variables](writeups/statistics/random_variables/random_variables.pdf) ([source](writeups/statistics/random_variables/random_variables.tex))

## Building

To build every PDF from the repository root:

```sh
make all
```

This runs the full LaTeX/BibTeX sequence needed for in-text citations and bibliography sections:

```sh
pdflatex file.tex
bibtex file
pdflatex file.tex
pdflatex file.tex
```

The first `pdflatex` pass records citation keys, `bibtex` reads those keys and `bib/references.bib`, and the final two `pdflatex` passes resolve the bibliography and citation labels.

If `make all` prints `Nothing to be done for 'all'.`, it means `make` thinks the PDFs are already up to date. To force a rebuild anyway:

```sh
make -B all
```

To rebuild a single writeup manually, run the same sequence from that writeup's directory. For example:

```sh
cd writeups/statistics/maximum_likelihood
pdflatex maximum_likelihood.tex
bibtex maximum_likelihood
pdflatex maximum_likelihood.tex
pdflatex maximum_likelihood.tex
```

Note that `bibtex` uses the filename without `.tex`.

## Citations

Writeups use numeric in-text citations through `natbib`.
Use `\citep{key}` for normal citations:

```tex
These derivations follow Yang et al. \citep{yang2010natgen}.
```

Bibliographies use `unsrtnat`, so references are numbered in order of first citation.

BibTeX keys should use this pattern:

```text
[author][year][journal-or-source]
```

Examples:

```text
yang2010natgen
lynch1998book
gundersen2019web
```

When multiple web references would otherwise have the same key, add a short topic suffix:

```text
wikipedia2026webBinomial
wikipedia2026webMVN
```

To remove LaTeX auxiliary files while keeping PDFs:

```sh
make clean
```

Shared LaTeX configuration lives in `tex/`, and shared BibTeX references live in `bib/references.bib`.
