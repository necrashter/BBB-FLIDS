#!/bin/bash

# Compiles the report to pdf

pdflatex -shell-escape main.tex
bibtex main.aux
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
