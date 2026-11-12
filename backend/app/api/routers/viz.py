"""Module 6 – CSV to academic chart and figure caption."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.tools.viz import parse_csv, plot_csv_data

router = APIRouter(prefix="/viz", tags=["visualisation"])


class PlotRequest(BaseModel):
    csv_content: str = Field(..., min_length=3, max_length=2_000_000)
    chart_type: str = "line"
    title: str = ""
    x_column: str = ""
    y_columns: list[str] = Field(default_factory=list)


@router.post("/plot")
async def plot(req: PlotRequest) -> dict:
    """Render a chart from CSV data and generate the figure caption."""
    result = await plot_csv_data(
        req.csv_content, chart_type=req.chart_type, title=req.title,
        x_column=req.x_column, y_columns=req.y_columns,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/inspect")
async def inspect(req: PlotRequest) -> dict:
    """Parse the CSV and report its structure (columns, numeric columns)."""
    header, rows = parse_csv(req.csv_content)
    numeric = []
    for i, name in enumerate(header):
        values = []
        ok = True
        for row in rows:
            try:
                values.append(float(str(row[i]).replace(",", "")))
            except (IndexError, ValueError):
                ok = False
                break
        if ok and values:
            numeric.append({"name": name, "min": min(values), "max": max(values),
                            "count": len(values)})
    return {"columns": header, "row_count": len(rows), "numeric_columns": numeric}
