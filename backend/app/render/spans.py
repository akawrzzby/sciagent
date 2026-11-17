"""Rebuild the merged cells of a Markdown table.

A PDF table that is exported to GFM Markdown loses its merge geometry: a header
that spans three columns is written ``| Label |  |  |`` and a label that spans
five rows is written once, followed by four empty cells underneath.  The reading
view has to put that geometry back, because ``colspan`` / ``rowspan`` are what
make the HTML table look like the printed one.

:func:`parse_markdown` is a pure text function -- no database, no framework, no
network.  It splits the rows of a GFM table into cells (escape- and code-aware),
folds every *empty* cell into a non-empty neighbour, and returns
:class:`~app.render.model.Cell` rows together with the number of header rows.

The merging rules, and the order they are tried in
--------------------------------------------------
Only empty cells are merged, and only into a neighbour that carries text: two
cells that happen to hold the same text are never merged, because that would
guess.  For every empty cell, in this order:

1. **left** -- if the cell covering the position to its left is non-empty, the
   empty cell joins that cell and the cell's ``colspan`` grows by one.
2. **above** -- otherwise (the left cell is empty too, or there is no left cell
   because we are in the first column), if the *block* directly above is
   non-empty, the empty cell joins that block and the block's ``rowspan`` grows
   by one.
3. **neither** -- the cell stays a cell of its own, with empty text.

Rule 1 is applied to every row before rule 2 runs, so a horizontal merge always
wins: an empty cell that has a filled cell to its left *and* a filled cell above
it is read as ``colspan``, not ``rowspan``, and the first row of a table can
never grow a ``rowspan``.  Rule 2 extends a block only when *every* column the
block spans is an empty cell on the new row; a merge never swallows a cell that
holds text.

Shape of the result
-------------------
Rows hold the visible cells only: a cell that was absorbed is not repeated in
the row (``model.Cell``: "a merged cell appears once and the cells it covers are
omitted").  What *is* uniform is the logical grid.  The input rows are padded
with empty cells first, so each row of the table is covered column by column
exactly once by the returned cells, and a renderer can simply walk the rows,
placing the next cell in the next column that is not already covered by a
``rowspan`` from an earlier row.
"""

from __future__ import annotations

import re

from .model import Cell

__all__ = ["parse_markdown"]

#: A fenced code block (``` or ~~~).  A table inside it is a code sample.
_FENCE = re.compile(r"^(```+|~~~+)")

#: One cell of a GFM delimiter row: ``---``, ``:---``, ``---:``, ``:---:``.
_DELIMITER = re.compile(r":?-+:?")

#: A cell whose whole content is a number: ``41``, ``-3``, ``41.0``, ``82%``.
_NUMBER = re.compile(r"[<>≤≥≈±]?\s*[+-]?\d[\d,]*(?:\.\d+)?\s*[%‰]?")


def parse_markdown(markdown: str) -> tuple[list[list[Cell]], int]:
    """Parse the first table of ``markdown`` into cells with their spans.

    Args:
        markdown: A document, or a single table, in GitHub flavoured Markdown.

    Returns:
        ``(rows, header_rows)``.  ``rows`` holds the visible cells of each row,
        left to right; cells absorbed by a merge are not repeated, so the cells
        of a row tile it exactly (see the module docstring).  ``header_rows`` is
        1 when the first row is a header, and 0 when the first row is empty or
        is obviously numeric -- a numeric row is data that was written without a
        delimiter row above it.
    """
    raw_rows = _read_table(markdown or "")
    if not raw_rows:
        return [], 0
    rows = [cells for cells in raw_rows if not _is_delimiter(cells)]
    if not rows:
        return [], 0
    width = max(len(cells) for cells in rows)
    if width == 0:
        return [], 0
    grid = [cells + [""] * (width - len(cells)) for cells in rows]
    header_rows = _count_header_rows(raw_rows, rows)
    merged = _merge(grid)
    for row in merged[:header_rows]:
        for cell in row:
            cell.header = True
    return merged, header_rows


# --------------------------------------------------------------------- reading
def _read_table(markdown: str) -> list[list[str]]:
    """Return the raw cell texts of the first table in ``markdown``.

    A table is a run of consecutive lines that either start with ``|`` or hold
    a pipe that is neither escaped nor inside inline code.  The run counts as a
    table when it contains a GFM delimiter row or when every one of its lines
    starts with ``|``; otherwise it is prose that happens to contain a pipe and
    is skipped.  Fenced code blocks are opaque.
    """
    runs: list[list[tuple[str, list[str]]]] = []
    run: list[tuple[str, list[str]]] = []
    fence = ""
    for line in markdown.splitlines():
        stripped = line.strip()
        fence_match = _FENCE.match(stripped)
        if fence_match:
            token = fence_match.group(1)[:3]
            if not fence:
                fence = token
            elif token == fence:
                fence = ""
            if run:
                runs.append(run)
            run = []
            continue
        if fence:
            continue
        cells = _split_row(stripped) if stripped else []
        if not stripped or not (stripped.startswith("|") or len(cells) > 1):
            if run:
                runs.append(run)
            run = []
            continue
        run.append((stripped, cells))
    if run:
        runs.append(run)
    for candidate in runs:
        if any(_is_delimiter(cells) for _line, cells in candidate) or all(
            line.startswith("|") for line, _cells in candidate
        ):
            return [cells for _line, cells in candidate]
    return []


def _split_row(line: str) -> list[str]:
    """Split one table line into its cells.

    ``\\|`` is a literal pipe inside a cell (it is unescaped in the returned
    text, which is what the reader shows), and so is any pipe inside an inline
    code span.  The optional leading and trailing pipes of the GFM syntax are
    not cells of their own.
    """
    text = line.strip()
    cells: list[str] = []
    buf: list[str] = []
    ticks = 0
    index = 1 if text.startswith("|") else 0
    total = len(text)
    ended_with_pipe = False
    while index < total:
        char = text[index]
        if char == "\\" and index + 1 < total:
            if text[index + 1] == "|" and not ticks:
                buf.append("|")  # an escaped pipe is a character, not a border
                ended_with_pipe = False
                index += 2
                continue
            if not ticks:
                # Keep the escape sequence: ``\*`` must survive into the renderer.
                buf.append(text[index:index + 2])
                ended_with_pipe = False
                index += 2
                continue
            buf.append(char)
            ended_with_pipe = False
            index += 1
            continue
        if char == "`":
            close = index
            while close < total and text[close] == "`":
                close += 1
            run = close - index
            if not ticks:
                ticks = run
            elif run == ticks:
                ticks = 0
            buf.append(text[index:close])
            ended_with_pipe = False
            index = close
            continue
        if char == "|" and not ticks:
            cells.append("".join(buf).strip())
            buf = []
            ended_with_pipe = True
            index += 1
            continue
        buf.append(char)
        ended_with_pipe = False
        index += 1
    tail = "".join(buf).strip()
    if tail or not ended_with_pipe:
        cells.append(tail)
    return cells


def _is_delimiter(cells: list[str]) -> bool:
    """True for the ``| --- | :---: |`` row that splits header and body."""
    filled = [cell for cell in cells if cell]
    return bool(filled) and all(_DELIMITER.fullmatch(cell) for cell in filled)


def _count_header_rows(raw_rows: list[list[str]], rows: list[list[str]]) -> int:
    """Return 1 when the first row is a header, else 0.

    A delimiter row normally sits under the header, so a table that starts with
    one has no header at all.  A first row that is empty, or that holds nothing
    but numbers, is data that lost its delimiter row and must not be promoted to
    a header.
    """
    if _is_delimiter(raw_rows[0]):
        return 0
    first = rows[0]
    filled = [cell for cell in first if cell]
    if not filled or all(_NUMBER.fullmatch(cell) for cell in filled):
        return 0
    return 1


# --------------------------------------------------------------------- merging
def _merge(grid: list[list[str]]) -> list[list[Cell]]:
    """Build the cells of a rectangular text grid and apply the merge rules."""
    height = len(grid)
    width = len(grid[0])
    #: the cell that covers each position, ``None`` only while it is being built
    cover: list[list[Cell | None]] = [[None] * width for _ in range(height)]
    #: whether a cell *starts* at a position, i.e. the position is not absorbed
    start: list[list[bool]] = [[False] * width for _ in range(height)]
    #: top left corner of every cell, keyed by ``id()`` because ``Cell`` is mutable
    origin: dict[int, tuple[int, int]] = {}

    # (1) horizontal merges: one row at a time, so they win over vertical ones.
    for r in range(height):
        for c in range(width):
            text = grid[r][c]
            left = cover[r][c - 1] if c else None
            if not text and left is not None and left.text:
                left.colspan += 1
                cover[r][c] = left
                continue
            cell = Cell(text=text)
            cover[r][c] = cell
            start[r][c] = True
            origin[id(cell)] = (r, c)

    # (2) vertical merges: top to bottom, so a block can grow over many rows.
    for r in range(1, height):
        for c in range(width):
            if not start[r][c]:
                continue
            cell = cover[r][c]
            if cell is None or cell.text:
                continue
            if c and grid[r][c - 1]:
                continue  # a filled left cell: rule 1 should have claimed it
            above = cover[r - 1][c]
            if above is None or not above.text:
                continue
            left_col = origin[id(above)][1]
            span = above.colspan
            if c != left_col or left_col + span > width:
                continue
            if left_col and grid[r][left_col - 1]:
                continue
            if not all(start[r][col] and not grid[r][col]
                       for col in range(left_col, left_col + span)):
                continue  # the block would cover a cell that holds text
            for col in range(left_col, left_col + span):
                cover[r][col] = above
                start[r][col] = False
            above.rowspan += 1

    rows: list[list[Cell]] = []
    for r in range(height):
        row: list[Cell] = []
        for c in range(width):
            cell = cover[r][c]
            if start[r][c] and cell is not None:
                row.append(cell)
        rows.append(row)
    return rows
