# The paper

`not_all_feasibility_repairs_are_equal_newinml2026.pdf` is the NewInML 2026 camera-ready
(8-page body, then acknowledgments, references, Appendices A-H and the NeurIPS checklist).

- Appendix H, "Use of AI assistance", is on page 38 of the PDF; its source is
  [`appendix_ai.tex`](appendix_ai.tex).
- The NeurIPS checklist starts on page 39; its source is [`checklist.tex`](checklist.tex).

To rebuild: `pdflatex camera_ready.tex` three times (the NeurIPS 2026 style file is included).
`camera_ready.tex` inputs `preamble.tex`, `body.tex` (abstract and Sections 1-8), `refs.tex`,
`appendix1.tex` (A-D), `appendix2.tex` (E-G), `appendix_ai.tex` (H) and `checklist.tex`.
