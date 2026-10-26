#!/usr/bin/env bash
# Build the one-page CV. Requires XeLaTeX (TinyTeX installed at ~/Library/TinyTeX).
set -euo pipefail

cd "$(dirname "$0")"
export PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"

# Two passes so hyperref resolves its bookmark/label files.
xelatex -interaction=nonstopmode -halt-on-error Jiaxuan_Zou_CV.tex
xelatex -interaction=nonstopmode -halt-on-error Jiaxuan_Zou_CV.tex

rm -f Jiaxuan_Zou_CV.aux Jiaxuan_Zou_CV.log Jiaxuan_Zou_CV.out

# assets/ holds the copy the site serves at /assets/Jiaxuan_Zou_CV.pdf
cp Jiaxuan_Zou_CV.pdf ../assets/Jiaxuan_Zou_CV.pdf
echo "Built cv/Jiaxuan_Zou_CV.pdf and published to assets/Jiaxuan_Zou_CV.pdf"
