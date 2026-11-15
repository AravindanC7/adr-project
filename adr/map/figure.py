"""Dependency-free SVG heatmap for failure-map log-likelihood ratios."""

from __future__ import annotations

from html import escape
from pathlib import Path

from adr.contracts import FailureMap, MapCell


def _colour(value: float, limit: float = 4.0) -> str:
    """Diverging blue-white-red colour for human-to-AI evidence."""

    scaled = max(-limit, min(limit, value)) / limit
    if scaled >= 0:
        channel = round(255 * (1 - scaled))
        return f"rgb(255,{channel},{channel})"
    channel = round(255 * (1 + scaled))
    return f"rgb({channel},{channel},255)"


def _row_label(cell: MapCell) -> str:
    setting = "/".join(
        value for value in (cell.attack_mode, cell.attacker_level) if value is not None
    )
    suffix = setting or "pooled"
    return f"{cell.mechanism} · {cell.item_type} · {suffix}"


def write_heatmap_svg(
    failure_map: FailureMap,
    path: str | Path,
    *,
    include_specific: bool = False,
) -> Path:
    """Write an SVG where red supports AI and blue supports human use."""

    cells = [
        cell
        for cell in failure_map.cells
        if include_specific or (cell.attack_mode is None and cell.attacker_level is None)
    ]
    cells.sort(
        key=lambda cell: (
            cell.mechanism,
            cell.item_type,
            cell.attack_mode or "",
            cell.attacker_level or "",
        )
    )
    if not cells:
        raise ValueError("failure map has no cells for the requested heatmap")

    label_width = 300
    cell_width = 205
    row_height = 64
    top = 105
    max_outcomes = max(len(cell.p_ai) for cell in cells)
    width = label_width + max_outcomes * cell_width + 35
    height = top + len(cells) * row_height + 70
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<style>text{font-family:Arial,sans-serif;fill:#18181b}'
            '.title{font-size:22px;font-weight:700}.label{font-size:14px;font-weight:600}'
            '.cell{font-size:12px}.legend{font-size:12px}</style>'
        ),
        f'<text x="18" y="32" class="title">Failure-mode map: {escape(failure_map.version)}</text>',
        (
            '<text x="18" y="56" class="legend">Log evidence: blue = human, '
            'white = neutral, red = AI (held-out rows excluded)</text>'
        ),
    ]
    legend_x = 18
    for index, (colour, label) in enumerate(
        (("rgb(0,0,255)", "≤ −4"), ("white", "0"), ("rgb(255,0,0)", "≥ +4"))
    ):
        x = legend_x + index * 80
        parts.append(
            f'<rect x="{x}" y="69" width="22" height="16" fill="{colour}" stroke="#71717a"/>'
        )
        parts.append(f'<text x="{x + 27}" y="82" class="legend">{label}</text>')

    for row, cell in enumerate(cells):
        y = top + row * row_height
        parts.append(
            f'<text x="18" y="{y + 30}" class="label">{escape(_row_label(cell))}</text>'
        )
        for column, outcome in enumerate(cell.p_ai):
            x = label_width + column * cell_width
            llr = cell.llr(outcome)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 8}" height="50" rx="4" '
                f'fill="{_colour(llr)}" stroke="#a1a1aa"/>'
            )
            parts.append(
                f'<text x="{x + 8}" y="{y + 20}" class="cell">{escape(outcome)}</text>'
            )
            parts.append(
                f'<text x="{x + 8}" y="{y + 39}" class="cell">LLR {llr:+.2f} · '
                f'nAI {sum(cell.counts_ai.values())} · nH {sum(cell.counts_human.values())}</text>'
            )
    parts.append("</svg>")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return destination
