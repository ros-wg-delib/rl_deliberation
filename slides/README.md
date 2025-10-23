# Slides

The accompanying slides are built with [Pandoc](https://pandoc.org/) and `pdflatex`.

You must first install these tools:

```bash
sudo apt install pandoc texlive-latex-base texlive-latex-extra
```

Then, to build the slides:

```bash
pandoc -t beamer main.md -o slides.pdf --listings --slide-level=2
```

Refer to https://pandoc.org/MANUAL.html#variables-for-beamer-slides for useful information.
