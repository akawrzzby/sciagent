"""Convert the technical report from Markdown to LaTeX.

The Markdown is our own dialect: ATX headings carrying explicit section numbers,
GFM tables, fenced code, block quotes, figures and inline emphasis.  The LaTeX
keeps those numbers as literal text (the body cross-references them, e.g.
"§6.3.4"), so headings are emitted unnumbered and registered in the TOC by hand.

Usage:
    python scripts/md2tex.py [src.md] [dst.tex]
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "technical-report.md"
DST = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "technical-report.tex"

PREAMBLE = r"""\documentclass[UTF8,fontset=windows,zihao=5,a4paper]{ctexart}

\usepackage{geometry}
\geometry{top=1.9cm,bottom=1.8cm,left=1.8cm,right=1.8cm}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{fancyvrb}
\usepackage{fvextra}
\usepackage[hidelinks]{hyperref}

\definecolor{ink}{HTML}{1D2836}
\definecolor{accent}{HTML}{B74C06}
\definecolor{rule}{HTML}{CFD9E4}

\ctexset{
  section/format = \Large\bfseries\color{ink},
  subsection/format = \large\bfseries\color{ink},
  subsubsection/format = \normalsize\bfseries\color{ink},
}

\setlist[itemize]{leftmargin=1.4em,itemsep=2pt,topsep=3pt,parsep=0pt}
\setlist[enumerate]{leftmargin=1.6em,itemsep=2pt,topsep=3pt,parsep=0pt}

\captionsetup{font=small,labelfont=bf,labelsep=quad,skip=4pt}
\setlength{\parskip}{2.5pt}
\linespread{1.02}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.06}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\scriptsize}
\AtBeginDocument{\fontsize{9}{11.4}\selectfont}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\small\color{rule}\thepage}

\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}
\newtcolorbox{aside}{
  breakable, enhanced, colback=orange!4!white, colframe=orange!45!black,
  boxrule=0pt, leftrule=2pt, arc=1pt, left=6pt, right=6pt, top=4pt, bottom=4pt,
  fontupper=\small
}

\begin{document}
"""

POSTAMBLE = "\n\\end{document}\n"


def escape_code(text: str) -> str:
    r"""Escape a code span for \texttt{}.

    ``^`` and ``_`` are math characters even in text mode, so they need the
    same treatment as ``&`` and friends.
    """
    pairs = [
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("_", r"\_"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("$", r"\$"),
    ]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def escape(text: str) -> str:
    """Escape LaTeX specials in plain text (not for code spans)."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def inline(text: str) -> str:
    """Render inline Markdown to LaTeX, protecting code spans first."""
    spans: list[str] = []

    def stash(fragment: str) -> str:
        spans.append(fragment)
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`",
                  lambda m: stash(r"\texttt{" + escape_code(m.group(1)) + "}"),
                  text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: stash(""), text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: stash(r"\href{" + m.group(2) + "}{" + escape(m.group(1)) + "}"), text)
    text = escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"\\emph{\1}", text)
    for index, fragment in enumerate(spans):
        text = text.replace(f"\x00{index}\x00", fragment)
    return text


def is_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(re.fullmatch(r"\|[\s:|-]+\|", stripped)) and "-" in stripped


def split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells, buf, escaped = [], [], False
    for ch in stripped:
        if escaped:
            buf.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "|":
            cells.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf))
    return [c.strip() for c in cells]


def table_tex(header: list[str], rows: list[list[str]]) -> str:
    columns = len(header)
    if columns == 1:
        spec = r"@{}p{0.96\textwidth}@{}"
    elif columns == 2:
        spec = r"@{}p{0.22\textwidth}p{0.74\textwidth}@{}"
    elif columns == 3:
        spec = r"@{}p{0.17\textwidth}p{0.36\textwidth}p{0.43\textwidth}@{}"
    elif columns == 4:
        spec = r"@{}p{0.10\textwidth}p{0.34\textwidth}p{0.30\textwidth}p{0.18\textwidth}@{}"
    elif columns == 5:
        spec = (r"@{}p{0.07\textwidth}p{0.26\textwidth}p{0.15\textwidth}"
                r"p{0.17\textwidth}p{0.27\textwidth}@{}")
    else:
        width = 0.94 / columns
        spec = "@{}" + "".join(f"p{{{width:.3f}\\textwidth}}" for _ in range(columns)) + "@{}"

    out = [r"\begin{longtable}{" + spec + "}", r"\toprule"]
    out.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in header) + r" \\")
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append(r"\toprule")
    out.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in header) + r" \\")
    out.append(r"\midrule")
    out.append(r"\endhead")
    out.append(r"\bottomrule")
    out.append(r"\endfoot")
    for row in rows:
        cells = row + [""] * (columns - len(row))
        out.append(" & ".join(inline(c) for c in cells[:columns]) + r" \\")
    out.append(r"\end{longtable}")
    out.append("")
    return "\n".join(out)


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    in_cover = False
    cover_rows: list[tuple[str, str]] = []
    cover_lines: list[str] = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped == "<!-- cover -->":
            in_cover = True
            i += 1
            continue
        if stripped == "<!-- /cover -->":
            in_cover = False
            i += 1
            continue

        if in_cover:
            if stripped.startswith("|") and not is_separator(stripped):
                cells = split_row(stripped)
                if len(cells) == 2 and cells[1]:
                    cover_rows.append((cells[0], cells[1]))
                elif len(cells) == 1 and cells[0]:
                    cover_lines.append(cells[0])
                elif cells and cells[0] == "" and len(cells) > 1:
                    cover_lines.append(cells[1])
            i += 1
            continue

        # fenced code
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append(r"\begin{Verbatim}[fontsize=\scriptsize,breaklines=true,"
                       r"breakanywhere=true,frame=single,framerule=0.2pt,"
                       r"rulecolor=\color{rule}]")
            out.extend(buf)
            out.append(r"\end{Verbatim}")
            out.append("")
            continue

        # figure
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m:
            caption, path = m.group(1), m.group(2)
            out.append(r"\begin{figure}[htbp]")
            out.append(r"\centering")
            out.append(r"\includegraphics[width=0.97\textwidth]{" +
                       path.replace("\\", "/") + "}")
            out.append(r"\caption*{" + inline(caption) + "}")
            out.append(r"\end{figure}")
            out.append("")
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if title == "目录":
                # the LaTeX build generates its own table of contents
                i += 1
                while i < n and not re.match(r"^#{1,6}\s+", lines[i].strip()):
                    i += 1
                continue
            title_tex = inline(title)
            if level == 1:
                out.append(r"\section*{" + title_tex + "}")
                out.append(r"\addcontentsline{toc}{section}{" + title_tex + "}")
                out.append(r"\vspace{-2pt}\hrule\vspace{6pt}")
            elif level == 2:
                out.append(r"\subsection*{" + title_tex + "}")
                out.append(r"\addcontentsline{toc}{subsection}{" + title_tex + "}")
            else:
                out.append(r"\subsubsection*{" + title_tex + "}")
            i += 1
            continue

        # table
        if stripped.startswith("|") and i + 1 < n and is_separator(lines[i + 1]):
            header = split_row(stripped)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            out.append(table_tex(header, rows))
            continue

        # block quote
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            inner = " ".join(b for b in buf if b)
            out.append(r"\begin{aside}" + inline(inner) + r"\end{aside}")
            out.append("")
            continue

        # horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            i += 1
            continue

        # lists
        if re.match(r"^(\s*)([-*+]|\d+\.)\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.\s", line))
            env = "enumerate" if ordered else "itemize"
            out.append(r"\begin{" + env + "}")
            while i < n and re.match(r"^(\s*)([-*+]|\d+\.)\s+", lines[i]):
                item = re.sub(r"^\s*([-*+]|\d+\.)\s+", "", lines[i])
                i += 1
                while i < n and lines[i].strip() and not re.match(
                        r"^(\s*)([-*+]|\d+\.)\s+", lines[i]) and not lines[i].strip().startswith(
                        ("#", "|", "```", ">")):
                    item += " " + lines[i].strip()
                    i += 1
                out.append(r"\item " + inline(item))
            out.append(r"\end{" + env + "}")
            out.append("")
            continue

        if not stripped:
            i += 1
            continue

        # paragraph
        buf = []
        while i < n:
            cur = lines[i]
            if (not cur.strip() or cur.strip().startswith(("#", "|", "```", ">"))
                    or re.match(r"^(\s*)([-*+]|\d+\.)\s+", cur)
                    or re.match(r"^!\[", cur.strip())):
                break
            buf.append(cur.strip())
            i += 1
        out.append(inline(" ".join(buf)))
        out.append("")

    # ---- title page
    title = "SciAgent"
    subtitle = "意图驱动的全流程科研助手"
    cover = [
        r"\thispagestyle{empty}",
        r"\begin{center}",
        r"\vspace*{2.2cm}",
        r"{\small\color{accent} 全国高校 AI 大模型创新应用挑战赛}\\[1.2cm]",
        r"{\Huge\bfseries " + title + r"}\\[0.5cm]",
        r"{\LARGE " + subtitle + r"}\\[1.0cm]",
        r"{\large 技 术 文 档}\\[2.2cm]",
        r"\begin{tabular}{@{}r@{\hspace{1.2em}}l@{}}",
    ]
    for key, value in cover_rows:
        value_tex = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
        cover.append(r"{\color{gray}" + inline(key) + r"} & " + inline(value_tex) + r" \\[3pt]")
    cover.extend([
        r"\end{tabular}",
        r"\end{center}",
        r"\vfill",
        r"\clearpage",
        r"\tableofcontents",
        r"\clearpage",
    ])

    return PREAMBLE + "\n".join(cover) + "\n" + "\n".join(out) + POSTAMBLE


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    tex = convert(md)
    DST.write_text(tex, encoding="utf-8")
    print(f"wrote {DST} ({len(tex)} chars)")


if __name__ == "__main__":
    main()
