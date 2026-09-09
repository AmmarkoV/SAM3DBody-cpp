#!/usr/bin/env bash
# scripts/doxygen.sh — Generate Doxygen HTML + PDF documentation for SAM3DBody-cpp
# Output: doc/html/index.html  and  doc/latex/refman.pdf

set -euo pipefail

if ! command -v doxygen &>/dev/null; then
    echo "Error: doxygen not found. Install it with:"
    echo "  sudo apt install doxygen        # Debian/Ubuntu"
    echo "  sudo dnf install doxygen        # Fedora/RHEL"
    echo "  brew install doxygen            # macOS"
    exit 1
fi

HAVE_PDFLATEX=true
if ! command -v pdflatex &>/dev/null; then
    HAVE_PDFLATEX=false
    echo "Warning: pdflatex not found — HTML only. Install it with:"
    echo "  sudo apt install texlive-latex-recommended texlive-fonts-recommended  # Debian/Ubuntu"
    echo "  sudo dnf install texlive-scheme-basic                                 # Fedora/RHEL"
    echo "  brew install --cask mactex                                             # macOS"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Generating Doxygen docs → $REPO_ROOT/doc/"

cd "$REPO_ROOT"

doxygen - <<DOXYEOF
PROJECT_NAME           = "SAM3DBody-cpp"
PROJECT_BRIEF          = "Standalone C++ inference engine for SAM-3D-Body"
OUTPUT_DIRECTORY       = doc
HTML_OUTPUT            = html

# Source tree
INPUT                  = src
FILE_PATTERNS          = *.c *.cpp *.h *.hpp
RECURSIVE              = YES
EXCLUDE_PATTERNS       = */build/* */venv/*

# Extraction
EXTRACT_ALL            = YES
EXTRACT_STATIC         = YES
EXTRACT_PRIVATE        = YES

# Diagrams (skipped if dot is absent)
HAVE_DOT               = NO

# Output formats
GENERATE_LATEX         = $( $HAVE_PDFLATEX && echo YES || echo NO )
LATEX_OUTPUT           = latex
QUIET                  = YES
WARN_IF_UNDOCUMENTED   = NO
DOXYEOF

echo "HTML docs: $REPO_ROOT/doc/html/index.html"

if $HAVE_PDFLATEX; then
    echo "Building PDF…"
    make -C "$REPO_ROOT/doc/latex" pdf 2>&1 | tail -5
    PDF="$REPO_ROOT/doc/latex/refman.pdf"
    if [ -f "$PDF" ]; then
        cp "$PDF" "$REPO_ROOT/doc/SAM3DBody-cpp.pdf"
        echo "PDF docs:  $REPO_ROOT/doc/SAM3DBody-cpp.pdf"
    else
        echo "Warning: PDF build finished but refman.pdf not found."
    fi
fi
