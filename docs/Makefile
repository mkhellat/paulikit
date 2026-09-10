# Minimal Sphinx Makefile (standard sphinx-quickstart layout).
#
# Usage: from this directory (docs/), with paulikit installed in
# editable mode with its docs extra (`pip install -e ..[docs]`):
#
#   make html   - build HTML docs into _build/html
#   make clean  - remove the _build directory

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
