"""Génère le rapport scientifique PDF à partir de la source versionnée.

Usage, depuis la racine du dépôt :

    python scripts/generate_rapport.py

Entrée :  docs/rapport/rapport.md
Sorties : docs/Rapport_MiniProject.pdf
          docs/Rapport_Projet.pdf (copie d'alias)

La source utilise un sous-ensemble volontairement simple de Markdown :
titres ``#``/``##``/``###``, paragraphes, listes ``- ``, tables ``| ... |``
et images ``![légende](chemin)``. Seule la bibliothèque matplotlib est
nécessaire (déjà présente dans l'environnement du projet).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import textwrap
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
# TrueType embeddé : le texte du PDF reste sélectionnable et reproductible.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

SOURCE = PROJECT_ROOT / "docs" / "rapport" / "rapport.md"
OUTPUT = PROJECT_ROOT / "docs" / "Rapport_MiniProject.pdf"
ALIAS = PROJECT_ROOT / "docs" / "Rapport_Projet.pdf"

# Format A4, marges en pouces.
PAGE_W, PAGE_H = 8.27, 11.69
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 0.72, 0.72, 0.78, 0.70
CONTENT_H = PAGE_H - MARGIN_T - MARGIN_B

BODY_FS = 9.2
BODY_WIDTH = 104
BODY_LEAD = BODY_FS * 1.46 / 72.0
TABLE_FS = 7.1
TABLE_LEAD = TABLE_FS * 1.45 / 72.0
FIGURE_MAX_H = 0.39 * PAGE_H

INK = "#1b2a3a"
BLUE = "#1f4e79"
LIGHT_BLUE = "#2c7fb8"
GRAY = "#5a6673"


# --------------------------------------------------------------------------- #
# Analyse de la source
# --------------------------------------------------------------------------- #
def parse_source(path: Path) -> list[tuple]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple] = []
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue

        if raw.startswith("# "):
            title = raw[2:].strip()
            meta: list[str] = []
            i += 1
            while i < len(lines):
                candidate = lines[i].strip()
                if not candidate or candidate.startswith(("#", "|", "!", "- ")):
                    break
                meta.append(candidate)
                i += 1
            blocks.append(("cover", title, meta))
            continue

        if raw.startswith("## "):
            blocks.append(("h2", raw[3:].strip()))
            i += 1
            continue

        if raw.startswith("### "):
            blocks.append(("h3", raw[4:].strip()))
            i += 1
            continue

        if raw.startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue

        match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if match:
            blocks.append(("figure", match.group(1), match.group(2)))
            i += 1
            continue

        if stripped.startswith("- "):
            items: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(("bullets", items))
            continue

        paragraph: list[str] = []
        while i < len(lines):
            candidate = lines[i].strip()
            if not candidate or candidate.startswith(("#", "|", "!", "- ")):
                break
            paragraph.append(candidate)
            i += 1
        blocks.append(("para", " ".join(paragraph)))

    return blocks


# --------------------------------------------------------------------------- #
# Mise en page
# --------------------------------------------------------------------------- #
class ReportPDF:
    def __init__(self, pdf: PdfPages, source: Path) -> None:
        self.pdf = pdf
        self.source = source.resolve()
        self.page_no = 0
        self.used = 0.0
        self.fig = None
        self.ax = None

    def new_page(self) -> None:
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page_no += 1
        self.used = 0.0
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.ax.text(
            0.5, 0.035,
            f"Mini-projet de Data Mining - Rapport scientifique - page {self.page_no}",
            ha="center", va="center", fontsize=7.5, color=GRAY,
        )

    def close(self) -> None:
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None

    def need(self, height: float) -> None:
        if self.used + height > CONTENT_H:
            self.new_page()

    @staticmethod
    def xy(y_top: float) -> float:
        """Distance depuis le haut de la page -> coordonnée axe (0 en bas)."""
        return 1.0 - y_top / PAGE_H

    def line(self, text: str, fs: float = BODY_FS, *, x: float | None = None,
             bold: bool = False, italic: bool = False, color: str = INK,
             family: str = "DejaVu Sans", leading: float | None = None,
             ha: str = "left", gap: float = 0.0) -> None:
        height = leading if leading is not None else fs * 1.46 / 72.0
        self.need(height)
        self.ax.text(
            (MARGIN_L if x is None else x) / PAGE_W,
            self.xy(MARGIN_T + self.used),
            text,
            ha=ha, va="top", fontsize=fs, color=color, family=family,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal",
        )
        self.used += height + gap

    def rule(self, color: str = LIGHT_BLUE) -> None:
        self.need(0.06)
        y = self.xy(MARGIN_T + self.used + 0.03)
        self.ax.plot([MARGIN_L / PAGE_W, 1 - MARGIN_R / PAGE_W], [y, y],
                     color=color, linewidth=0.8, transform=self.ax.transAxes)
        self.used += 0.10

    def cover(self, title: str, meta: list[str]) -> None:
        self.ax.text(0.5, 0.79, title, ha="center", va="center", fontsize=21,
                     fontweight="bold", color=BLUE, wrap=True)
        y = 0.70
        for item in meta:
            self.ax.text(0.5, y, item, ha="center", va="center", fontsize=11,
                         color=INK)
            y -= 0.055
        self.ax.plot([0.22, 0.78], [0.43, 0.43], color=LIGHT_BLUE, linewidth=1.2)
        self.ax.text(0.5, 0.38, "Source : docs/rapport/rapport.md",
                     ha="center", fontsize=9, color=GRAY)
        self.new_page()

    def heading2(self, text: str) -> None:
        self.need(0.52)
        self.line(text, fs=14.0, bold=True, color=BLUE, leading=0.28, gap=0.02)
        self.rule()

    def heading3(self, text: str) -> None:
        self.need(0.40)
        self.line(text, fs=11.4, bold=True, color=LIGHT_BLUE, leading=0.24, gap=0.06)

    def paragraph(self, text: str) -> None:
        for part in textwrap.wrap(text, width=BODY_WIDTH, break_long_words=False,
                                  break_on_hyphens=False):
            self.line(part, leading=BODY_LEAD)
        self.used += 0.09

    def bullets(self, items: list[str]) -> None:
        for item in items:
            wrapped = textwrap.wrap(item, width=BODY_WIDTH - 2,
                                    break_long_words=False, break_on_hyphens=False)
            for rank, part in enumerate(wrapped):
                prefix = "- " if rank == 0 else "  "
                self.line(prefix + part, leading=BODY_LEAD)
        self.used += 0.09

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        ncols = max(len(row) for row in rows)
        grid = [row + [""] * (ncols - len(row)) for row in rows]
        widths = [min(42, max(len(row[col]) for row in grid)) for col in range(ncols)]
        total = sum(widths) + 3 * (ncols - 1)
        if total > BODY_WIDTH:
            factor = BODY_WIDTH / total
            widths = [max(6, int(w * factor)) for w in widths]

        def render(cells: list[str]) -> str:
            return "  ".join(cell[:widths[c]].ljust(widths[c])
                             for c, cell in enumerate(cells[:ncols]))

        self.need(TABLE_LEAD * len(grid) + 0.12)
        for index, cells in enumerate(grid):
            self.line(render(cells), fs=TABLE_FS, family="DejaVu Sans Mono",
                      bold=(index == 0), color=BLUE if index == 0 else INK,
                      leading=TABLE_LEAD)
        self.used += 0.12

    def figure(self, caption: str, relative: str) -> None:
        # Les liens sont relatifs à la source Markdown (affichage GitHub), avec
        # repli sur la racine du dépôt pour les anciens chemins.
        path = (self.source.parent / relative).resolve()
        if not path.exists():
            path = (PROJECT_ROOT / relative).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Figure introuvable : {path}")
        with Image.open(path) as source:
            image = source.copy()
        image.thumbnail((1600, 1600))

        width_frac = 0.86
        height_frac = width_frac * (image.height / image.width) * (PAGE_W / PAGE_H)
        if height_frac * PAGE_H > FIGURE_MAX_H:
            height_frac = FIGURE_MAX_H / PAGE_H
            width_frac = height_frac * (image.width / image.height) * (PAGE_H / PAGE_W)

        height_in = height_frac * PAGE_H
        self.need(height_in + 0.34)
        x0 = MARGIN_L / PAGE_W
        y0 = self.xy(MARGIN_T + self.used + height_in)
        # Un axe dédié par figure : le backend PDF ne rend qu'une seule image
        # lors de plusieurs appels à imshow sur le même Axes.
        image_ax = self.fig.add_axes([x0, y0, width_frac, height_frac])
        image_ax.imshow(image, aspect="auto", interpolation="bilinear")
        image_ax.set_axis_off()
        self.used += height_in + 0.05

        legend = f"{caption}" if caption else "Figure"
        for part in textwrap.wrap(legend, width=BODY_WIDTH,
                                  break_long_words=False, break_on_hyphens=False):
            self.line(part, fs=8.1, italic=True, color=GRAY, leading=0.15)
        self.used += 0.11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="source Markdown du rapport")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="chemin du PDF principal")
    args = parser.parse_args()

    blocks = parse_source(args.source)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:
        report = ReportPDF(pdf, args.source)
        report.new_page()
        for block in blocks:
            kind = block[0]
            if kind == "cover":
                report.cover(block[1], block[2])
            elif kind == "h2":
                report.heading2(block[1])
            elif kind == "h3":
                report.heading3(block[1])
            elif kind == "para":
                report.paragraph(block[1])
            elif kind == "bullets":
                report.bullets(block[1])
            elif kind == "table":
                report.table(block[1])
            elif kind == "figure":
                report.figure(block[1], block[2])
            else:
                raise ValueError(f"Bloc inconnu : {kind}")
        report.close()

    shutil.copyfile(output, ALIAS)
    print(f"  ✔ {output.relative_to(PROJECT_ROOT)}")
    print(f"  ✔ {ALIAS.relative_to(PROJECT_ROOT)}")
    print(f"  {len(blocks)} blocs rendus")


if __name__ == "__main__":
    main()
