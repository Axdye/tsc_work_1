# Paired-CF-GRPO overview figure

- **overview.drawio** — editable diagram source; open in draw.io / diagrams.net.
- **overview.pdf** — vector manuscript export, embedded TrueType font.
- **overview.svg** — vector export with editable text.
- **overview.png** — 4000 × 1920 high-resolution preview.
- **caption.md** — English/Chinese captions and scientific scope.
- **defect-log.md** — four render checks, reviewed automated warnings, self-score 43/50.
- **edit-preview.html** — optional diagrams.net editing bridge; needs external diagrams.net access. Offline users should open overview.drawio in a local draw.io desktop installation.

The figure focuses on paired first-step counterfactuals and current-policy multi-step continuation. It shows actual policy visitation separately from training-only branches. No critic, judge, span credit or future fast/slow architecture is implied.

## Regenerate exports

The draw.io XML is the authority for export geometry/text:

```bash
python figures/paired_cf_grpo/build_figure.py           # regenerate XML from the current generator
python figures/paired_cf_grpo/build_figure.py --render  # export existing XML to SVG/PDF/PNG
```

If editing the XML directly, run only the second command to preserve edits. The limited renderer supports the current rounded rectangles, ellipses, rich sub/sup labels, and explicit polyline geometry. Other draw.io shapes should be exported through draw.io itself.

Matplotlib is the only extra rendering dependency. Arial is used for the shown exports; if unavailable on another OS, install an appropriate font or adjust the font setting before reproducing exact text metrics.
