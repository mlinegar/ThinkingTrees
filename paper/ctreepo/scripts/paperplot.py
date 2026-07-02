"""Shared matplotlib defaults for minimal-paper figures.

Every paper figure generator should call `rcparams()` once at startup so
that fonts, sizes, and colors render consistently across panels. Two size
presets (`FIGSIZE_ONE_COL`, `FIGSIZE_FULL`) match the main-text column
budgets. The `ANCHOR_COLORS` dict fixes a topic->color mapping so the
reader can build a consistent mental map.
"""

from __future__ import annotations

import matplotlib as mpl


FIGSIZE_ONE_COL = (3.5, 2.2)
FIGSIZE_ONE_COL_TALL = (3.5, 2.9)
FIGSIZE_FULL = (7.0, 3.0)
FIGSIZE_FULL_TALL = (7.0, 4.2)


ANCHOR_COLORS = {
    "markov": "#1f77b4",
    "hll": "#d95f02",
    "manifesto": "#1b7837",
    "neural_operator": "#762a83",
    "theory": "#555555",
    "baseline": "#999999",
}


def rcparams() -> None:
    """Apply the shared rc parameters in-place on the active matplotlib."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "serif",
            "font.size": 10.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save(fig, output_stem, *, formats=("pdf", "png")) -> list:
    """Save `fig` under `<output_stem>.<fmt>` for each format, return paths."""
    from pathlib import Path

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path)
        written.append(path)
    return written


def heatmap_with_reference(
    ax,
    matrix,
    *,
    row_labels,
    col_labels,
    cmap,
    vmin=None,
    vmax=None,
    vcenter=None,
    reference_value=None,
    reference_label=None,
    bold_reference=True,
    cell_format="{:.3f}",
    inherited_mask=None,
    colorbar_label=None,
    title=None,
    xlabel=None,
    ylabel=None,
):
    """Draw a cell grid heatmap with a bolded reference tick on its colorbar.

    Parameters
    ----------
    ax
        Target matplotlib axis.
    matrix
        2D numpy array of finite values; NaN cells are drawn as hatched grey.
    row_labels, col_labels
        Tick labels for each axis; one per row/column of ``matrix``.
    cmap
        Matplotlib colormap name or object. Sequential for external-quality
        (ramps toward the topic anchor color); ``RdBu_r`` for signed gaps.
    vmin, vmax
        Colormap limits. Pin these across paired subfigures so cell darkness
        is comparable across lanes.
    reference_value, reference_label
        Horizontal reference value on the colorbar; if present within
        [vmin, vmax], drawn as a bolded tick. ``reference_label`` is stored
        on the colorbar for the caller to reference in the caption.
    bold_reference
        If True, bold the reference tick label. Set False to draw a thinner
        guide (e.g. for the gap=0 parity line).
    cell_format
        ``str.format`` pattern for cell text. Pass an empty string to
        suppress numeric overlays.
    inherited_mask
        Boolean 2D mask same shape as ``matrix``. ``True`` entries are drawn
        with a small ``=`` glyph rather than the numeric value, indicating
        inherited (unchanged) evaluations reused across ladder stages.
    colorbar_label, title, xlabel, ylabel
        Passthrough labels.
    """
    import math as _math

    import matplotlib as _mpl
    import matplotlib.pyplot as _plt
    import numpy as _np

    data = _np.asarray(matrix, dtype=float)
    masked = _np.ma.masked_invalid(data)
    cmap_obj = _mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad(color="#e8e8e8", alpha=1.0)

    if vcenter is not None and vmin is not None and vmax is not None:
        norm = _mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect="auto")
    else:
        im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(_np.arange(data.shape[1]))
    ax.set_xticklabels([str(value) for value in col_labels], rotation=25, ha="right")
    ax.set_yticks(_np.arange(data.shape[0]))
    ax.set_yticklabels([str(value) for value in row_labels])

    if cell_format:
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                value = data[y, x]
                if not _math.isfinite(float(value)):
                    ax.text(
                        x, y, "—", ha="center", va="center",
                        fontsize=7, color="#777777",
                    )
                    continue
                if inherited_mask is not None and inherited_mask[y, x]:
                    ax.text(
                        x, y, "=", ha="center", va="center",
                        fontsize=9, color="#333333",
                    )
                    continue
                rgba = cmap_obj(im.norm(float(value)))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "white" if luminance < 0.55 else "#111111"
                ax.text(
                    x, y, cell_format.format(float(value)),
                    ha="center", va="center",
                    fontsize=7, color=text_color,
                )

    cbar = _plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=8)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(8)

    if (
        reference_value is not None
        and vmin is not None
        and vmax is not None
        and vmin <= reference_value <= vmax
    ):
        cbar.ax.axhline(
            reference_value,
            color="#111111" if bold_reference else "#333333",
            linewidth=1.4 if bold_reference else 1.0,
            linestyle="-" if bold_reference else ":",
        )
        if bold_reference:
            existing_ticks = list(cbar.get_ticks())
            if not any(abs(float(t) - reference_value) < 1e-6 for t in existing_ticks):
                new_ticks = sorted(existing_ticks + [reference_value])
                cbar.set_ticks(new_ticks)
                idx = new_ticks.index(reference_value)
                cbar.ax.get_yticklabels()[idx].set_fontweight("bold")

    return im, cbar
