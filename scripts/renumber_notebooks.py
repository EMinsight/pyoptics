#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Renumber the example notebooks and rewrite every reference to them.

The map from old to new file names (without ``.ipynb``), with the short
name of each notebook, is read from a JSON file, by default
``scripts/notebook_renumbering.json``. The script

1. renames the notebooks with ``git mv`` (``--no-move`` skips this, for a
   branch whose files were already moved),
2. re-keys ``notebooks/names.json``,
3. rewrites file names, ``{doc}`` paths and link texts, and references such as
   "notebook 47 (rays and waves)" or "notebooks 117 (dispersion in fibres)
   and 118 (pulse broadening)" in the Markdown files, the docs, the notebooks
   (all cells) and the comments and strings of the Python sources.

A reference in prose is only rewritten when the notebook's short name stands
next to its number, which the project rule (see ``CLAUDE.md``) guarantees.
Numbers without a name are left alone and listed, as are ranges ("notebooks
52 (...) to 55 (...)"), which may not be ranges any more after renumbering.

Run it from the repository root. It can be run again on a branch that was
started before the renumbering and has since merged the renumbered master:
a number is only rewritten when the old notebook's name stands next to it,
so references that already use the new numbers stay as they are.

    python scripts/renumber_notebooks.py [--map MAP] [--no-move] [--dry-run]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT/"notebooks"
TEXT_SUFFIXES = {".md", ".py", ".pyx", ".ipynb", ".json", ".yml", ".yaml", ".toml", ".txt", ".cfg", ".rst"}
SKIP = {"scripts/notebook_renumbering.json", "scripts/renumber_notebooks.py",
        "test/test_notebook_references.py"}  # the last holds old numbers on purpose

NUM = r"\d{2,3}"
ITEM = NUM + r"(?![\d.,]\d)(?:\s*\([^()]*\))?"
SEP = r"(?:\s*,\s*|\s+(?:and|or|to)\s+|\s*[–-]\s*|\s*,\s*(?:and|or)\s+)"
REF = re.compile(r"(?<![\w\[/])(?:[Nn]otebooks?|[Nn][Bb]s?)\s*(?P<list>" + ITEM
                 + r"(?:" + SEP + ITEM + r")*)(?!\w)")
ITEM_NUM = re.compile(r"(?<![\w(])" + NUM + r"(?![\d.,]\d)")
RANGE = re.compile(r"\)?\s+to\s+\d")


class Renumberer:
    def __init__(self, stem_map, names):
        self.stem_map = stem_map
        self.num_map = {old.split("_", 1)[0]: new.split("_", 1)[0] for old, new in stem_map.items()}
        self.names = names                   # old number -> short name
        alts = "|".join(re.escape(s) for s in sorted(stem_map, key=len, reverse=True))
        self.stem_re = re.compile(r"(?<![\w])(" + alts + r")(?![\w])")
        self.unnamed = []
        self.ranges = []

    def _named(self, num, before, after):
        name = self.names[num].lower()
        if name in before[-(len(name) + 12):].lower():
            return True
        return re.match(r"\s*(?:,\s*|\(\s*|,?\s*\*)?" + re.escape(name), after.lower()) is not None

    def _prose(self, text, where):
        out, pos = [], 0
        for m in REF.finditer(text):
            refs = m.group("list")
            start = m.start("list")
            if len(ITEM_NUM.findall(refs)) > 1 and RANGE.search(refs):
                self.ranges.append((where, m.group(0)))
            for item in ITEM_NUM.finditer(refs):
                num = item.group(0)
                a, b = start + item.start(), start + item.end()
                if num not in self.num_map:
                    continue
                if self._named(num, text[:m.start()], text[b:]):
                    out.append(text[pos:a])
                    out.append(self.num_map[num])
                    pos = b
                else:
                    self.unnamed.append((where, text[max(0, a - 40):b + 40]))
        out.append(text[pos:])
        return "".join(out)

    def _link_numbers(self, text):
        # {doc}`47 Rays and waves <notebooks/450_rays_and_waves>`
        text = re.sub(r"(\{doc\}`)(\d{2,3})(?=\s[^<`]*<notebooks/(\d{3})_)", lambda m: m.group(1) + m.group(3), text)
        # [47 rays and waves](450_rays_and_waves.ipynb), [52 (Abbe ...)](notebooks/310_...ipynb)
        text = re.sub(r"\[(\d{2,3})(?=[^\]\[]*\]\((?:[^)\s]*/)?(\d{3})_[^)\s]*\.ipynb)",
                      lambda m: "[" + m.group(2), text)
        return text

    def _pairs(self, text):
        """An old number directly followed by its own name, as in lists that
        name the notebook once ("notebooks 156, EUV multilayers ..., to 158,
        EUV stochastics and flare") or in tables ("| **52** | Abbe ...")."""
        for num, name in sorted(self.names.items(), key=lambda kv: -len(kv[1])):
            pattern = (r"(?<![\w.\-])" + num + r"(?=(?:\*\*)?(?:\s*\(\s*|\s*,\s*|\s+|\s*\|\s*)"
                       + re.escape(name).replace(r"\ ", r"\s+") + r"(?![\w\-]))")
            text = re.sub(pattern, self.num_map[num], text, flags=re.IGNORECASE)
        return text

    def rewrite(self, text, where):
        text = self._prose(text, where)
        text = self._pairs(text)
        text = self.stem_re.sub(lambda m: self.stem_map[m.group(1)], text)
        return self._link_numbers(text)


def _notebook(path, ren, dry):
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    changed = False

    def fix(lines, where):
        nonlocal changed
        if isinstance(lines, str):
            new = ren.rewrite(lines, where)
            changed |= new != lines
            return new
        text = "".join(lines)
        new = ren.rewrite(text, where)
        if new == text:
            return lines
        changed = True
        return new.splitlines(keepends=True)

    for i, cell in enumerate(nb["cells"]):
        where = f"{path.name} cell {i}"
        cell["source"] = fix(cell["source"], where)
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = fix(out["text"], where + " output")
            for mime in ("text/plain", "text/markdown", "text/html"):
                if mime in out.get("data", {}):
                    out["data"][mime] = fix(out["data"][mime], where + " output")
    if changed and not dry:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--map", default=str(ROOT/"scripts"/"notebook_renumbering.json"))
    parser.add_argument("--no-move", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = json.loads(Path(args.map).read_text(encoding="utf-8"))
    stem_map = {old: e["new"] for old, e in entries.items()}
    old_names = {old.split("_", 1)[0]: e["name"] for old, e in entries.items()}
    ren = Renumberer(stem_map, old_names)
    names_path = NB_DIR/"names.json"
    names = json.loads(names_path.read_text(encoding="utf-8"))

    if not args.no_move:
        for old, new in stem_map.items():
            src = NB_DIR/(old + ".ipynb")
            if src.exists():
                print(f"git mv {src.name} {new}.ipynb")
                if not args.dry_run:
                    subprocess.run(["git", "mv", str(src), str(NB_DIR/(new + ".ipynb"))], cwd=ROOT, check=True)

    if set(old_names) <= set(names) and all(names[k] == v for k, v in old_names.items()):
        rekeyed = {ren.num_map.get(k, k): v for k, v in names.items()}
        rekeyed = dict(sorted(rekeyed.items()))
        if not args.dry_run:
            names_path.write_text(json.dumps(rekeyed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print("notebooks/names.json does not hold the old numbers, left as it is")

    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.split("\0")
    n_changed = 0
    for rel in files:
        path = ROOT/rel
        if rel in SKIP or path.suffix not in TEXT_SUFFIXES or rel == "notebooks/names.json":
            continue
        if not rel or not path.is_file():
            continue
        if path.suffix == ".ipynb":
            n_changed += _notebook(path, ren, args.dry_run)
            continue
        text = path.read_text(encoding="utf-8")
        new = ren.rewrite(text, rel)
        if new != text:
            n_changed += 1
            if not args.dry_run:
                path.write_text(new, encoding="utf-8")
    print(f"{n_changed} files rewritten")
    for where, snippet in ren.unnamed:
        print(f"UNNAMED  {where}: {snippet!r}")
    for where, ref in ren.ranges:
        print(f"RANGE    {where}: {ref!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
