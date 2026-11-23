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
