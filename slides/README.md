# Slides

The accompanying slides are built with [Pandoc](https://pandoc.org/) and `pdflatex`.

Refer to <https://pandoc.org/MANUAL.html#variables-for-beamer-slides> for useful information.

## Running pandoc natively

You must first install these tools:

```bash
sudo apt install pandoc texlive-latex-base texlive-latex-extra
```

Then, to build the slides:

```bash
pandoc -t beamer main.md -o slides.pdf --listings --slide-level=2
```

## Running pandoc with Docker (recommended)

Or use the docker image that is also used in CI:
(Run this command from the `slides/` directory)

```bash
docker run \
    --rm \
    --volume "$(pwd):/data" \
    --user $(id -u):$(id -g) \
    pandoc/latex:3.7 -t beamer main.md -o slides.pdf --listings --slide-level=2
```
