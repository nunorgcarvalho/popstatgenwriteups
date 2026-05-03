TEXFILES := $(shell find writeups -name '*.tex' | sort)
PDFS := $(TEXFILES:.tex=.pdf)

.PHONY: all clean

all: $(PDFS)

%.pdf: %.tex bib/references.bib tex/config.sty tex/macros.sty
	cd $(dir $<) && pdflatex -interaction=nonstopmode $(notdir $<)
	cd $(dir $<) && bibtex $(basename $(notdir $<))
	cd $(dir $<) && pdflatex -interaction=nonstopmode $(notdir $<)
	cd $(dir $<) && pdflatex -interaction=nonstopmode $(notdir $<)

clean:
	find writeups -type f \( -name '*.aux' -o -name '*.bbl' -o -name '*.bcf' -o -name '*.blg' -o -name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.log' -o -name '*.out' -o -name '*.run.xml' -o -name '*.synctex.gz' \) -delete
