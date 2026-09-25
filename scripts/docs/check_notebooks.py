#!/usr/bin/env python
"""Check that the tutorials are executed, error-free, and match the documentation.

Run in CI before ``mkdocs build``. Fails (exit code 1) if any notebook in
``notebooks/``:

- has a code cell that was never executed, or a saved error output
- lacks an "Open in Colab" badge pointing at its own path on ``main``
- lacks exactly one cell tagged ``hero`` with an image output
- has a tagged figure whose copy under ``docs/assets/`` is missing or differs
  (re-run ``scripts/docs/export_figures.py``)
- is larger than ``MAX_MB``
- uses MkDocs admonition syntax (``!!!``), which Jupyter and Colab do not render

It also checks that every tutorial is listed in ``mkdocs.yml``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_figures import NOTEBOOKS, REPO, tagged_figures  # noqa: E402

COLAB = "https://colab.research.google.com/github/bullocke/nice-sar/blob/main/notebooks/{}"
MAX_MB = 4.0

logger = logging.getLogger("check_notebooks")


def check(nb_path: Path) -> list[str]:
    """Problems found in one notebook."""
    problems = []
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]

    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code" or not "".join(cell["source"]).strip():
            continue
        if cell.get("execution_count") is None and "colab-setup" not in cell.get(
            "metadata", {}
        ).get("tags", []):
            problems.append(f"cell {i} was not executed")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(f"cell {i} has an error output: {out.get('ename')}")

    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and "\n!!! " in "\n" + "".join(cell["source"]):
            problems.append(f"cell {i} uses '!!!' admonition syntax; use a blockquote")

    first_md = next((c for c in cells if c["cell_type"] == "markdown"), None)
    if first_md is None or COLAB.format(nb_path.name) not in "".join(first_md["source"]):
        problems.append("missing or wrong Colab badge in the first markdown cell")

    heroes = [c for c in cells if "hero" in c.get("metadata", {}).get("tags", [])]
    if len(heroes) != 1:
        problems.append(f"expected one cell tagged 'hero', found {len(heroes)}")

    figures = tagged_figures(nb_path)
    if heroes and not any(f.tag == "hero" for f in figures):
        problems.append("the hero cell has no image output")
    for fig in figures:
        if not fig.target.exists():
            problems.append(f"{fig.target.relative_to(REPO)} is missing (run export_figures.py)")
        elif fig.target.read_bytes() != fig.data:
            problems.append(f"{fig.target.relative_to(REPO)} is out of date (run export_figures.py)")

    size_mb = nb_path.stat().st_size / 1e6
    if size_mb > MAX_MB:
        problems.append(f"{size_mb:.1f} MB exceeds {MAX_MB} MB")
    return problems


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mkdocs = (REPO / "mkdocs.yml").read_text()
    failed = False
    for nb in sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb")):
        problems = check(nb)
        if f"tutorials/{nb.name}" not in mkdocs:
            problems.append("not listed in mkdocs.yml nav")
        for p in problems:
            logger.error("%s: %s", nb.name, p)
        if not problems:
            logger.info("%s: ok", nb.name)
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
