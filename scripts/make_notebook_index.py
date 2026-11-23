#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Write the notebook indexes from ``notebooks/toc.json``.

``notebooks/toc.json`` lists the parts and chapters and the notebooks in
each, in reading order. This script writes from it

- ``docs/examples.md``: one section per part and chapter, with the toctrees
  of the documentation,
- the parts between the ``<!-- toc:... -->`` markers in ``notebooks/README.md``:
  the overview by part and the table of all notebooks (whose Modules and
  Contents columns are kept from the existing table).

Run it after adding a notebook to ``toc.json`` and ``names.json``::

    python scripts/make_notebook_index.py          # write
    python scripts/make_notebook_index.py --check  # fail if out of date

``test/test_notebook_toc.py`` runs the check.
"""

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT/"notebooks"


def load():
    toc = json.loads((NB/"toc.json").read_text(encoding="utf-8"))
    names = json.loads((NB/"names.json").read_text(encoding="utf-8"))
    return toc, names


def _num(stem):
    return stem.split("_", 1)[0]


def _range(chapter):
    nums = [_num(s) for s in chapter["notebooks"]]
    return nums[0] if len(nums) == 1 else f"{nums[0]}–{nums[-1]}"


def _part_range(part):
    p = part["number"]
    return f"{p}00–{p}99"


def examples_md(toc, names):
    out = ["# Examples", "",
           "The example notebooks illustrate the modules with examples whose physics",
           "is easy to follow, and compare the numerics with analytic results wherever",
           "an analytic solution exists. They are shown here with their stored outputs;",
           "to run them yourself, see [Running the notebooks](#running-the-notebooks).",
           "",
           "The notebooks are numbered in reading order: the hundreds digit is the",
           "part, the tens digit the chapter, the last digit the place in the chapter.",
           "Reading a chapter from its first notebook to its last is a good default; for",
           "routes across the parts by background and interest, see {doc}`learning_paths`.",
           ""]
    for part in toc["parts"]:
        out += [f"## {part['number']}xx {part['title']}", "", part["intro"], ""]
        for chapter in part["chapters"]:
            out += [f"### {chapter['title']} ({_range(chapter)})", "", chapter["intro"], ""]
            for stem in chapter["notebooks"]:
                out.append(f"- {{doc}}`{_num(stem)} {names[_num(stem)][0].upper() + names[_num(stem)][1:]} <notebooks/{stem}>`")
            out.append("")
        out += ["```{toctree}", ":hidden:", f":caption: {part['number']}xx {part['title']}", ""]
        out += [f"notebooks/{stem}" for chapter in part["chapters"] for stem in chapter["notebooks"]]
        out += ["```", ""]
    out += ["## Running the notebooks", "",
            "```{include} ../notebooks/README.md",
            ":start-after: <!-- docs:running-start -->",
            ":end-before: <!-- docs:running-end -->",
            "```", ""]
    return "\n".join(out)


def _old_rows(readme):
    """Modules and contents of each notebook from the existing table."""
    rows = {}
    for line in readme.splitlines():
        m = re.match(r"\|\s*\[[^\]]*\]\((\d{3}_[\w]+)\.ipynb\)\s*\|(.*)\|(.*)\|\s*$", line)
        if m:
            rows[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
    return rows


def overview_md(toc, names):
    out = []
    for part in toc["parts"]:
        out.append(f"- **{part['number']}xx {part['title']}** ({_part_range(part)})")
        for chapter in part["chapters"]:
            links = ", ".join(f"[{_num(s)} {names[_num(s)]}]({s}.ipynb)" for s in chapter["notebooks"])
            out.append(f"  - {chapter['title']}: {links}")
    return "\n".join(out)


def table_md(toc, names, rows):
    out = []
    for part in toc["parts"]:
        out += [f"### {part['number']}xx {part['title']}", "",
                "| Notebook | Modules | Contents |", "|---|---|---|"]
        for chapter in part["chapters"]:
            for stem in chapter["notebooks"]:
                modules, contents = rows.get(stem, ("", ""))
                out.append(f"| [{_num(stem)} {names[_num(stem)]}]({stem}.ipynb) | {modules} | {contents} |")
        out.append("")
    return "\n".join(out).rstrip("\n")


def _replace(text, key, body):
    pattern = re.compile(r"(<!-- toc:" + key + r"-start -->\n).*?(<!-- toc:" + key + r"-end -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"notebooks/README.md has no <!-- toc:{key}-start/end --> markers")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def readme_md(toc, names, readme):
    rows = _old_rows(readme)
    readme = _replace(readme, "overview", overview_md(toc, names))
    return _replace(readme, "table", table_md(toc, names, rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the files are out of date")
    args = parser.parse_args()
    toc, names = load()
    targets = {ROOT/"docs"/"examples.md": examples_md(toc, names)}
    readme_path = NB/"README.md"
    targets[readme_path] = readme_md(toc, names, readme_path.read_text(encoding="utf-8"))
    stale = [p for p, text in targets.items() if p.read_text(encoding="utf-8") != text]
    if args.check:
        for p in stale:
            print(f"out of date: {p.relative_to(ROOT)} (run scripts/make_notebook_index.py)")
        return 1 if stale else 0
    for p in stale:
        p.write_text(targets[p], encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
