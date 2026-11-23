#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the numbering of the notebooks against ``notebooks/toc.json``.

The hundreds digit of a notebook's number is its part, the tens digit its
chapter; every chapter starts on a multiple of ten, and the numbers grow
in reading order. The indexes in ``docs/examples.md`` and
``notebooks/README.md`` are generated from ``toc.json`` by
``scripts/make_notebook_index.py`` and must be up to date.
"""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
NB = ROOT/"notebooks"
TOC = json.loads((NB/"toc.json").read_text(encoding="utf-8"))
NAMES = json.loads((NB/"names.json").read_text(encoding="utf-8"))
STEMS = [s for part in TOC["parts"] for chapter in part["chapters"] for s in chapter["notebooks"]]


def test_toc_lists_every_notebook_once():
    files = sorted(p.stem for p in NB.glob("*.ipynb"))
    assert len(STEMS) == len(set(STEMS)), "a notebook is listed twice in notebooks/toc.json"
    assert sorted(STEMS) == files, ("notebooks/toc.json and the notebooks differ: missing "
                                    f"{sorted(set(files) - set(STEMS))}, unknown {sorted(set(STEMS) - set(files))}")


def test_numbers_follow_parts_and_chapters():
    previous = -1
    for part in TOC["parts"]:
        for chapter in part["chapters"]:
            first = int(chapter["notebooks"][0].split("_", 1)[0])
            assert first % 10 == 0, f"chapter {chapter['title']!r} must start on a multiple of ten"
            for stem in chapter["notebooks"]:
                num = stem.split("_", 1)[0]
                assert len(num) == 3 and num[0] == part["number"], f"{stem} is not in part {part['number']}"
                assert int(num) > previous, f"{stem} is out of order in notebooks/toc.json"
                previous = int(num)


def test_names_match_numbers():
    assert sorted(NAMES) == sorted(s.split("_", 1)[0] for s in STEMS)


def test_indexes_up_to_date():
    result = subprocess.run([sys.executable, str(ROOT/"scripts"/"make_notebook_index.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
