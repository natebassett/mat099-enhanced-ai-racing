# MAT099 LaTeX Manuscript

`main.tex` is the manuscript root. The executive summary and Chapters 1--12 are converted from the revised dissertation draft. The manuscript contains four native LaTeX tables and seven figures, so captions, labels and lists remain synchronised as the dissertation grows.

Compile with Biber:

```text
pdflatex main
biber main
pdflatex main
pdflatex main
```

Overleaf can compile the same project by setting `main.tex` as the main document and selecting Biber as the bibliography tool. Figure 6.1 is stored in `figures/architecture-overview.png`, so the `dissertation` directory can be uploaded as a self-contained LaTeX project.

Before submission, replace all bracketed front-matter placeholders with the University's required wording and your own details. The acknowledgements are intentionally left to the author.
