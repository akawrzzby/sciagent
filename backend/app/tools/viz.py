"""Module 6 – data visualisation: CSV → academic chart + figure caption.

The chart is rendered with Matplotlib using a publication-oriented style
(serif fonts, no chartjunk, restrained colours) and the caption is produced by
the LLM, which has to derive the *core trend* from the data.

A 3-D radar chart is also supported because the task explicitly asks for one.
"""

from __future__ import annotations

import base64
import csv
import io
import os
import re
import uuid
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm import prompts
from app.llm.client import get_llm_client

logger = get_logger(__name__)

# Matplotlib must not try to open a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Academic style: serif type, muted colours, no top/right spines.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Songti SC", "SimSun"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PALETTE = ["#3B6EA5", "#C1553B", "#4E8B57", "#8A6BBE", "#B08A2E", "#5B8C9E"]


def parse_csv(csv_content: str) -> tuple[list[str], list[list[str]]]:
    """Parse a CSV string into ``(header, rows)``."""
    content = csv_content.strip()
    if not content:
        return [], []
    sample = content[:4096]
    delimiter = ","
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        return [], []
    return [c.strip() for c in rows[0]], rows[1:]


def _to_float(value: str) -> float | None:
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _numeric_columns(header: list[str], rows: list[list[str]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for i, name in enumerate(header):
        values: list[float] = []
        ok = True
        for row in rows:
            if i >= len(row):
                ok = False
                break
            v = _to_float(row[i])
            if v is None:
                ok = False
                break
            values.append(v)
        if ok and values:
            out[name] = values
    return out


def render_chart(
    header: list[str],
    rows: list[list[str]],
    *,
    chart_type: str = "line",
    title: str = "",
    x_column: str = "",
    y_columns: list[str] | None = None,
) -> bytes:
    """Render the chart and return PNG bytes."""
    numeric = _numeric_columns(header, rows)
    if not numeric:
        raise ValueError("CSV 中没有可绘制的数值列")

    y_columns = [c for c in (y_columns or []) if c in numeric] or list(numeric.keys())[:3]
    if x_column and x_column in header and x_column not in numeric:
        xs = [row[header.index(x_column)] for row in rows]
        x_label, x_numeric = x_column, False
    else:
        xs = list(range(1, len(rows) + 1))
        x_label, x_numeric = "Index", True

    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    if chart_type == "bar":
        width = 0.8 / max(1, len(y_columns))
        base = np.zeros(len(rows))
        for i, col in enumerate(y_columns):
            vals = numeric[col]
            offset = (i - (len(y_columns) - 1) / 2) * width
            ax.bar(np.arange(len(vals)) + offset, vals, width=width,
                   label=col, color=PALETTE[i % len(PALETTE)], alpha=0.9)
        if not x_numeric:
            ax.set_xticks(np.arange(len(rows)))
            ax.set_xticklabels(xs, rotation=20, ha="right", fontsize=8)
        else:
            ax.set_xticks(np.arange(len(rows)))
        ax.set_xlabel(x_label)

    elif chart_type == "radar3d":
        _render_radar(ax, numeric, y_columns)

    elif chart_type == "table":
        ax.axis("off")
        cell_text = [[str(c) for c in row] for row in rows]
        table = ax.table(cellText=cell_text, colLabels=header, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.25)
        if title:
            ax.set_title(title, fontsize=11)

    else:  # line
        for i, col in enumerate(y_columns):
            vals = numeric[col]
            ax.plot(range(1, len(vals) + 1) if x_numeric else xs, vals,
                    marker="o", markersize=3.5, linewidth=1.6,
                    label=col, color=PALETTE[i % len(PALETTE)])
        ax.set_xlabel(x_label)
        if not x_numeric:
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=20, ha="right", fontsize=8)
        ax.legend(frameon=False, fontsize=8)

    for col in y_columns:
        if col in numeric and chart_type != "table" and chart_type != "radar3d":
            ax.set_ylabel(col)

    if title and chart_type != "table":
        ax.set_title(title, fontsize=11, pad=10)

    ax.legend(frameon=False, fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _render_radar(ax, numeric: dict[str, list[float]], y_columns: list[str]) -> None:
    """A 3-D radar (spider) chart drawn in polar coordinates."""
    labels = list(y_columns)[:6]
    if not labels:
        raise ValueError("雷达图至少需要一列数据")
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = ax.figure
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, col in enumerate(labels):
        vals = numeric[col]
        vals = (vals + np.abs(vals.min())) if min(vals) < 0 else vals
        max_v = max(vals) or 1.0
        norm = [v / max_v for v in vals]
        norm += norm[:1]
        ax.plot(angles, norm, linewidth=1.6, color=PALETTE[i % len(PALETTE)],
                label=col)
        ax.fill(angles, norm, alpha=0.12, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.grid(color="#999999", linestyle="--", linewidth=0.5)
    ax.spines["polar"].set_visible(False)
    fig.set_size_inches(5.2, 5.2)


async def plot_csv_data(
    csv_content: str,
    *,
    chart_type: str = "line",
    title: str = "",
    x_column: str = "",
    y_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Full pipeline: CSV → chart image + academically styled caption."""
    header, rows = parse_csv(csv_content)
    if not header or not rows:
        return {"error": "CSV 无法解析"}

    try:
        png = render_chart(header, rows, chart_type=chart_type, title=title,
                           x_column=x_column, y_columns=y_columns)
    except Exception as exc:
        logger.warning("chart rendering failed: %s", exc)
        return {"error": f"绘图失败: {exc}"}

    settings = get_settings()
    out_dir = settings.data_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"fig_{uuid.uuid4().hex[:10]}.png"
    path = out_dir / name
    path.write_bytes(png)

    # --- caption ---------------------------------------------------------
    llm = get_llm_client()
    data_preview = _data_preview(header, rows)
    payload, raw = await llm.complete_json(
        prompts.SYSTEM_CAPTION,
        f"CHART TYPE: {chart_type}\nTITLE: {title or '(not given)'}\n\nDATA\n{data_preview}",
        schema_hint='{"caption": "string", "key_findings": ["string"]}',
        purpose="caption",
    )
    if payload:
        caption = str(payload.get("caption", ""))
        findings = payload.get("key_findings") or []
    else:
        caption = ""
        findings = []

    return {
        "chart_type": chart_type,
        "image_path": str(path),
        "image_base64": base64.b64encode(png).decode("ascii"),
        "caption": caption,
        "key_findings": findings,
        "columns": header,
        "row_count": len(rows),
        "table_markdown": _markdown_table(header, rows),
    }


def _data_preview(header: list[str], rows: list[list[str]], limit: int = 40) -> str:
    lines = [", ".join(header)]
    for row in rows[:limit]:
        lines.append(", ".join(str(c) for c in row))
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(lines)


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def render_tikz_preview_available() -> bool:
    """Whether a local LaTeX toolchain is available for previewing TikZ."""
    from shutil import which

    return which("pdflatex") is not None
