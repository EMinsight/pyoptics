# Notes for Claude

## Refer to notebooks by number and name

Always name a notebook by its number **and** its name, never by the number
alone. This holds everywhere: notebook markdown and code comments, docs,
README files, docstrings and comments in `pyoptics/` and `test/`, commit
messages, PR descriptions and chat.

- In prose: "notebook 450 (rays and waves) showed ...", "notebooks 722
  (dispersion in fibres) and 723 (pulse broadening)".
- Inside parentheses: "(notebook 111, ABCD matrices)".
- As a link: `notebook [300 (Abbe image formation)](300_abbe_image_formation.ipynb)`,
  or `[450 rays and waves](450_rays_and_waves.ipynb)` in lists.
- A range names both ends: "notebooks 310 (phase contrast and DIC) to 312
  (Fourier ptychography)".

The short names live in `notebooks/names.json`; use them as written there
(capitalize at the start of a sentence). A new notebook needs an entry in
that file. `test/test_notebook_references.py` fails on any bare
"notebook NN", "nbNN", `[NN](NN_....ipynb)` or name-less `{doc}` link, and
on a notebook without a name.

## Numbering of the notebooks

Notebook numbers have three digits and follow the reading order: the
hundreds digit is the part, the tens digit the chapter, the last digit the
place in the chapter (`notebooks/toc.json` lists them all). A new notebook
takes the next free number at the end of its chapter, or opens a new
chapter on the next free multiple of ten; existing notebooks are never
renumbered. Add it to `notebooks/toc.json` and `notebooks/names.json`, then
run `python scripts/make_notebook_index.py` to update `docs/examples.md` and
`notebooks/README.md`; `test/test_notebook_toc.py` checks all of this.

A branch started before the renumbering converts its own references after
merging master with `python scripts/renumber_notebooks.py --no-move` (and
moves its new notebooks by hand to their slot).
