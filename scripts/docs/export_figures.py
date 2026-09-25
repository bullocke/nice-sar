#!/usr/bin/env python
"""Copy tagged figures from the executed tutorials into the documentation.

Every figure shown on the documentation site comes from a notebook output, so the
site always matches what the tutorials produce:

- the cell tagged ``hero`` in ``notebooks/NN_name.ipynb`` becomes
  ``docs/assets/gallery/NN_name.<ext>`` (gallery and landing page)
- a cell tagged ``figure:<name>`` becomes ``docs/assets/figures/<name>.<ext>``
  (science pages)

Run after executing the notebooks (``scripts/docs/run_notebooks.sh``):

    python scripts/docs/export_figures.py

``scripts/docs/check_notebooks.py`` fails CI when an exported file no longer
matches its notebook output.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "notebooks"
GALLERY = REPO / "docs" / "assets" / "gallery"
FIGURES = REPO / "docs" / "assets" / "figures"
MIME = {"image/png": ".png", "image/jpeg": ".jpg"}

logger = logging.getLogger("export_figures")


@dataclass
class TaggedFigure:
    """One figure output from a tagged notebook cell."""

    notebook: Path
    tag: str
    data: bytes
    ext: str

    @property
    def target(self) -> Path:
        if self.tag == "hero":
            return GALLERY / f"{self.notebook.stem}{self.ext}"
        return FIGURES / f"{self.tag.split(':', 1)[1]}{self.ext}"


def tagged_figures(notebook: Path) -> list[TaggedFigure]:
    """The last image output of each cell tagged ``hero`` or ``figure:<name>``."""
    nb = json.loads(notebook.read_text())
    found = []
    for cell in nb["cells"]:
        tags = [
            t
            for t in cell.get("metadata", {}).get("tags", [])
            if t == "hero" or t.startswith("figure:")
        ]
        if not tags:
            continue
        images = [
            (mime, out["data"][mime])
            for out in cell.get("outputs", [])
            for mime in MIME
            if mime in out.get("data", {})
        ]
        if not images:
            continue
        mime, payload = images[-1]
        if isinstance(payload, list):
            payload = "".join(payload)
        for tag in tags:
            found.append(TaggedFigure(notebook, tag, base64.b64decode(payload), MIME[mime]))
    return found


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    GALLERY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for nb in sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb")):
        for fig in tagged_figures(nb):
            for stale in fig.target.parent.glob(fig.target.stem + ".*"):
                stale.unlink()
            fig.target.write_bytes(fig.data)
            logger.info("%s [%s] -> %s", nb.name, fig.tag, fig.target.relative_to(REPO))


if __name__ == "__main__":
    main()
