"""Table geometry for the reading view: ``spans.parse_markdown``.

The cases below are the shapes a PDF table loses first when it is serialised to
Markdown: a header that spans columns, a label that spans rows, a pipe inside a
cell, and rows that do not line up.
"""

from __future__ import annotations

from app.render.model import Cell
from app.render.spans import parse_markdown


# ------------------------------------------------------------------ helpers
def _texts(row: list[Cell]) -> list[str]:
    return [cell.text for cell in row]


def _grid_texts(rows: list[list[Cell]]) -> list[list[str]]:
    return [_texts(row) for row in rows]


def _render(rows: list[list[Cell]], width: int) -> list[list[str]]:
    """Place the cells the way an HTML renderer must: next free column.

    A cell is written into the first column of its row that no ``rowspan`` from
    an earlier row covers, and it claims ``rowspan * colspan`` positions.  The
    grid therefore checks the whole geometry, not just one span: an overlap
    fails the assertion, and so does a hole.
    """
    grid: list[list[str | None]] = [[None] * width for _ in rows]
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while c < width and grid[r][c] is not None:
                c += 1
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    assert grid[r + dr][c + dc] is None, "overlapping cells"
                    grid[r + dr][c + dc] = cell.text
            c += cell.colspan
    assert all(cell is not None for line in grid for cell in line), f"hole in {grid}"
    return [[cell or "" for cell in line] for line in grid]


def _all_spans(rows: list[list[Cell]]) -> list[tuple[int, int]]:
    return [(cell.rowspan, cell.colspan) for row in rows for cell in row]


# ---------------------------------------------------------------- plain grid
def test_a_plain_grid_keeps_every_cell() -> None:
    markdown = (
        "| Model | Accuracy | F1 |\n"
        "| --- | --- | --- |\n"
        "| BERT | 80.5 | 79.1 |\n"
        "| GPT | 82.0 | 80.3 |\n"
    )
    rows, header_rows = parse_markdown(markdown)
    assert header_rows == 1
    assert _grid_texts(rows) == [
        ["Model", "Accuracy", "F1"],
        ["BERT", "80.5", "79.1"],
        ["GPT", "82.0", "80.3"],
    ]
    assert _all_spans(rows) == [(1, 1)] * 9
    assert _render(rows, 3) == _grid_texts(rows)


def test_the_delimiter_row_is_not_a_data_row() -> None:
    rows, _ = parse_markdown("| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    assert len(rows) == 2


def test_only_header_cells_are_marked_as_header() -> None:
    rows, _ = parse_markdown("| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    assert [cell.header for cell in rows[0]] == [True, True]
    assert [cell.header for cell in rows[1]] == [False, False]


# -------------------------------------------------------- colspan from blanks
def test_an_empty_header_cell_extends_the_header_to_the_left() -> None:
    markdown = "| A | | B |\n| --- | --- | --- |\n| 1 | 2 | 3 |\n"
    rows, _ = parse_markdown(markdown)
    assert _texts(rows[0]) == ["A", "B"]
    assert (rows[0][0].rowspan, rows[0][0].colspan) == (1, 2)
    assert (rows[0][1].rowspan, rows[0][1].colspan) == (1, 1)
    assert _render(rows, 3) == [["A", "A", "B"], ["1", "2", "3"]]


def test_two_empty_cells_in_a_row_give_a_colspan_of_three() -> None:
    rows, _ = parse_markdown("| A | | |\n| --- | --- | --- |\n| 1 | 2 | 3 |\n")
    assert rows[0][0].colspan == 3
    assert len(rows[0]) == 1


# --------------------------------------------------------- rowspan from blanks
def test_an_empty_first_column_cell_extends_the_cell_above() -> None:
    markdown = "| Model | Acc |\n| --- | --- |\n| BERT | 80 |\n| | 81 |\n"
    rows, header_rows = parse_markdown(markdown)
    assert header_rows == 1
    assert rows[1][0].text == "BERT"
    assert (rows[1][0].rowspan, rows[1][0].colspan) == (2, 1)
    assert _texts(rows[2]) == ["81"]
    assert _render(rows, 2) == [["Model", "Acc"], ["BERT", "80"], ["BERT", "81"]]


def test_a_block_can_grow_over_three_rows() -> None:
    markdown = "| Model | Acc |\n| --- | --- |\n| BERT | 80 |\n| | 81 |\n| | 82 |\n"
    rows, _ = parse_markdown(markdown)
    assert rows[1][0].rowspan == 3
    assert _render(rows, 2) == [
        ["Model", "Acc"], ["BERT", "80"], ["BERT", "81"], ["BERT", "82"],
    ]


def test_a_wide_block_absorbs_the_empty_cells_below_it() -> None:
    """The block above spans two columns, so the whole width is absorbed once."""
    markdown = "| A | | B |\n| --- | --- | --- |\n| | | 1 |\n"
    rows, _ = parse_markdown(markdown)
    assert (rows[0][0].rowspan, rows[0][0].colspan) == (2, 2)
    assert len(rows[1]) == 1
    assert _render(rows, 3) == [["A", "A", "B"], ["A", "A", "1"]]


def test_a_row_absorbed_entirely_by_the_block_above_has_no_cells() -> None:
    markdown = "| A | B |\n| --- | --- |\n| | |\n| 1 | 2 |\n"
    rows, _ = parse_markdown(markdown)
    assert rows[1] == []
    assert (rows[0][0].rowspan, rows[0][1].rowspan) == (2, 2)
    assert _render(rows, 2) == [["A", "B"], ["A", "B"], ["1", "2"]]


def test_a_header_cell_that_spans_rows_stays_a_header() -> None:
    rows, _ = parse_markdown("| A | B |\n| --- | --- |\n| | 1 |\n")
    assert rows[0][0].rowspan == 2
    assert rows[0][0].header is True


def test_a_merge_never_covers_a_cell_that_holds_text() -> None:
    """The left cell wins over the upper one, and nothing is swallowed."""
    markdown = "| H | X |\n| --- | --- |\n| 1 | |\n| | y |\n"
    rows, _ = parse_markdown(markdown)
    assert rows[1][0].text == "1"
    assert (rows[1][0].rowspan, rows[1][0].colspan) == (1, 2)  # left, not above
    assert _texts(rows[2]) == ["", "y"]
    assert _render(rows, 2) == [["H", "X"], ["1", "1"], ["", "y"]]


# ------------------------------------------------------------ pipes in a cell
def test_an_escaped_pipe_does_not_split_a_cell() -> None:
    markdown = "| a \\| b | c |\n| --- | --- |\n| 1 | 2 |\n"
    rows, _ = parse_markdown(markdown)
    assert _texts(rows[0]) == ["a | b", "c"]
    assert _render(rows, 2) == [["a | b", "c"], ["1", "2"]]


def test_a_pipe_inside_inline_code_does_not_split_a_cell() -> None:
    markdown = "| `a | b` | c |\n| --- | --- |\n| 1 | 2 |\n"
    rows, _ = parse_markdown(markdown)
    assert _texts(rows[0]) == ["`a | b`", "c"]
    assert _render(rows, 2) == [["`a | b`", "c"], ["1", "2"]]


def test_a_pipe_inside_inline_code_with_an_escaped_pipe_next_to_it() -> None:
    markdown = "| `x | y` | a \\| b |\n| --- | --- |\n| 1 | 2 |\n"
    rows, _ = parse_markdown(markdown)
    assert _texts(rows[0]) == ["`x | y`", "a | b"]


# --------------------------------------------------------------- degenerate
def test_empty_input_has_no_table() -> None:
    assert parse_markdown("") == ([], 0)
    assert parse_markdown("   \n\n") == ([], 0)
    assert parse_markdown("Just prose, no table at all.") == ([], 0)


def test_a_table_with_only_a_header() -> None:
    rows, header_rows = parse_markdown("| A | B |\n| --- | --- |\n")
    assert header_rows == 1
    assert _grid_texts(rows) == [["A", "B"]]
    assert all(cell.header for cell in rows[0])


def test_a_table_that_is_only_a_delimiter_row_is_empty() -> None:
    assert parse_markdown("| --- | --- |\n") == ([], 0)


def test_ragged_rows_are_padded_to_the_same_columns() -> None:
    markdown = "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 |\n| 3 | 4 | 5 |\n"
    rows, _ = parse_markdown(markdown)
    assert _texts(rows[1]) == ["1", "2"]
    assert rows[1][1].colspan == 2  # the missing third cell joins the left one
    assert _render(rows, 3) == [["A", "B", "C"], ["1", "2", "2"], ["3", "4", "5"]]


def test_a_body_row_wider_than_the_header_extends_the_grid() -> None:
    rows, _ = parse_markdown("| A | B |\n| --- | --- |\n| 1 | 2 | 3 |\n")
    assert rows[0][1].colspan == 2
    assert _render(rows, 3) == [["A", "B", "B"], ["1", "2", "3"]]


# --------------------------------------------------------------- header rows
def test_a_numeric_first_row_is_not_a_header() -> None:
    rows, header_rows = parse_markdown("| 1 | 2 | 3 |\n| --- | --- | --- |\n| 4 | 5 | 6 |\n")
    assert header_rows == 0
    assert not any(cell.header for row in rows for cell in row)


def test_an_all_empty_first_row_is_not_a_header() -> None:
    rows, header_rows = parse_markdown("| | |\n| --- | --- |\n| 1 | 2 |\n")
    assert header_rows == 0
    assert not any(cell.header for row in rows for cell in row)


def test_a_table_that_starts_with_the_delimiter_row_has_no_header() -> None:
    rows, header_rows = parse_markdown("| --- | --- |\n| 1 | 2 |\n")
    assert header_rows == 0
    assert _grid_texts(rows) == [["1", "2"]]


# --------------------------------------------------------------- surroundings
def test_prose_around_the_table_is_ignored() -> None:
    markdown = (
        "Table 2 shows the results.\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "As shown above, the numbers improve.\n"
    )
    rows, _ = parse_markdown(markdown)
    assert _grid_texts(rows) == [["A", "B"], ["1", "2"]]


def test_a_table_inside_a_code_fence_is_ignored() -> None:
    markdown = (
        "```\n| not | a table |\n| --- | --- |\n| x | y |\n```\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    )
    rows, _ = parse_markdown(markdown)
    assert _grid_texts(rows) == [["A", "B"], ["1", "2"]]


def test_only_the_first_table_is_parsed() -> None:
    markdown = (
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "prose\n\n"
        "| C | D |\n| --- | --- |\n| 3 | 4 |\n"
    )
    rows, _ = parse_markdown(markdown)
    assert _grid_texts(rows) == [["A", "B"], ["1", "2"]]


def test_alignment_and_padding_are_accepted() -> None:
    markdown = (
        "|   Left   |  Center  | Right |\n"
        "|:---------|:--------:|------:|\n"
        "|   a      |    b     |   c   |\n"
    )
    rows, header_rows = parse_markdown(markdown)
    assert header_rows == 1
    assert _grid_texts(rows) == [["Left", "Center", "Right"], ["a", "b", "c"]]


def test_a_pipe_table_without_a_delimiter_row_still_parses() -> None:
    rows, header_rows = parse_markdown("| A | B |\n| 1 | 2 |\n")
    assert header_rows == 1
    assert _grid_texts(rows) == [["A", "B"], ["1", "2"]]


def test_parse_markdown_is_pure() -> None:
    """The same text always yields the same cells, with no shared state."""
    markdown = "| A | | B |\n| --- | --- | --- |\n| 1 | 2 | 3 |\n"
    first, first_headers = parse_markdown(markdown)
    second, second_headers = parse_markdown(markdown)
    assert first_headers == second_headers
    assert _grid_texts(first) == _grid_texts(second)
    assert _all_spans(first) == _all_spans(second)
    first[0][0].text = "changed"
    assert second[0][0].text == "A"
