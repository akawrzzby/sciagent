#!/usr/bin/env python3
"""Convert a Markdown document into a print-ready PDF.

The competition requires the technical document as a PDF of at most 30 pages.
This script renders the Markdown to a styled HTML page and prints it with a
headless Chromium/Edge, which is available on every Windows machine.

It has **no third-party dependencies** - the Markdown subset used by the
technical report (headings, paragraphs, lists, tables, code blocks, block
quotes, inline formatting and links) is handled by the small converter below.

Usage::

    python scripts/md2pdf.py docs/technical-report.md docs/technical-report.pdf

Optional::

    --browser <path>   path to chrome.exe / msedge.exe
    --title <text>     document title shown in the header
    --footer <text>    text shown in the page footer
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- CSS
CSS = """
@page {
  size: A4;
  margin: 12mm 11mm 13mm 11mm;
}
:root {
  --ink: #1d2836;
  --ink-soft: #405670;
  --ink-faint: #7890a8;
  --accent: #b74c06;
  --rule: #cfd9e4;
  --soft: #f5f7fa;
}
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Source Serif 4", "Noto Serif SC", Georgia, "Songti SC", serif;
  color: var(--ink);
  font-size: 8.9pt;
  line-height: 1.37;
  margin: 0;
}
h1, h2, h3, h4, h5 {
  font-family: "Inter", "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif;
  color: var(--ink);
  line-height: 1.3;
  page-break-after: avoid;
  break-after: avoid;
}
h1 {
  font-size: 15pt;
  margin: 0 0 7pt 0;
  padding-bottom: 4pt;
  border-bottom: 2px solid var(--ink);
  page-break-before: always;
  break-before: page;
}
.cover h1 { page-break-before: avoid; break-before: auto; border: none; }
h2 { font-size: 11.8pt; margin: 11pt 0 4pt 0; }
h3 { font-size: 10.4pt; margin: 9pt 0 3pt 0; color: var(--ink-soft); }
h4 { font-size: 9.8pt; margin: 8pt 0 2pt 0; color: var(--ink-soft); }
p { margin: 0 0 3.8pt 0; text-align: justify; }
ul, ol { margin: 0 0 4.5pt 0; padding-left: 15pt; }
li { margin: 1pt 0; }
li > ul, li > ol { margin: 1pt 0; }
strong { color: var(--ink); }
em { color: var(--ink-soft); }
code {
  font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
  font-size: 8.0pt;
  background: var(--soft);
  border: 1px solid #e8edf3;
  border-radius: 3px;
  padding: 0.4pt 2.2pt;
  word-break: break-all;
}
pre {
  font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
  font-size: 7.5pt;
  line-height: 1.38;
  background: #131b26;
  color: #f8f6f0;
  border-radius: 5px;
  padding: 6pt 8pt;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  page-break-inside: avoid;
  break-inside: avoid;
}
pre code { background: none; border: none; color: inherit; padding: 0; font-size: inherit; }
blockquote {
  margin: 6pt 0;
  padding: 4pt 8pt;
  border-left: 3px solid var(--accent);
  background: #fff8ed;
  color: var(--ink-soft);
  page-break-inside: avoid;
  break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
hr { border: none; border-top: 1px solid var(--rule); margin: 9pt 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 5pt 0 7pt 0;
  font-family: "Inter", "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif;
  font-size: 7.5pt;
  page-break-inside: avoid;
  break-inside: avoid;
}
th {
  background: #eef2f7;
  font-weight: 600;
  text-align: left;
  padding: 2.6pt 4pt;
  border: 1px solid var(--rule);
}
td { padding: 2.5pt 4pt; border: 1px solid var(--rule); vertical-align: top; }
tr:nth-child(even) td { background: #fafbfd; }
a { color: var(--accent); text-decoration: none; }
img { max-width: 100%; }
figure {
  margin: 5pt 0 6pt 0;
  text-align: center;
  page-break-inside: avoid;
  break-inside: avoid;
}
figure img { width: 94%; }
figcaption {
  margin-top: 2pt;
  font-family: "Inter", "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif;
  font-size: 7.2pt;
  color: var(--ink-faint);
}
.cover {
  min-height: 235mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
  page-break-after: always;
  break-after: page;
}
.cover .kicker { font-family: "Inter", sans-serif; font-size: 10pt; letter-spacing: .18em;
  color: var(--ink-faint); text-transform: uppercase; }
.cover h1 { font-size: 26pt; border: none; margin: 8pt 0 4pt 0; page-break-before: avoid; }
.cover h2 { font-size: 14pt; color: var(--accent); font-weight: 500; margin: 0 0 26pt 0; }
.cover table { width: 78%; margin: 0 auto; font-size: 9pt; }
.cover table td { border: none; border-bottom: 1px solid var(--rule); text-align: left;
  padding: 4pt 6pt; background: none !important; }
.cover table td:first-child { width: 30%; color: var(--ink-faint); }
.toc { page-break-after: always; break-after: page; }
.toc ol { list-style: none; padding-left: 0; }
.toc ol li { margin: 2.5pt 0; font-size: 9.4pt; }
.toc ol li ol { padding-left: 16pt; }
.toc ol li ol li { color: var(--ink-soft); }
"""

INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")


def inline(text: str) -> str:
    """Render inline Markdown to HTML.

    The text is escaped **first** and the markup is applied afterwards, so
    that the tags we generate are not escaped into visible angle brackets.
    """
    text = html.escape(text)
    placeholders: list[str] = []

    def stash(fragment: str) -> str:
        placeholders.append(fragment)
        return f"\x00{len(placeholders) - 1}\x00"

    def on_code(match: re.Match[str]) -> str:
        return stash(f"<code>{match.group(1)}</code>")

    def on_link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        return stash(f'<a href="{target}">{label}</a>')

    text = INLINE_CODE.sub(on_code, text)
    text = LINK.sub(on_link, text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    text = ITALIC.sub(r"<em>\1</em>", text)
    for i, fragment in enumerate(placeholders):
        text = text.replace(f"\x00{i}\x00", fragment)
    return text


def render_table(rows: list[list[str]], align: list[str]) -> str:
    """Render a GFM table.

    A table whose header row is empty (used on the cover page) is rendered
    without the ``thead`` so that no empty band is printed.
    """
    if not rows:
        return ""
    out = ["<table>"]
    header = rows[0]
    has_header = any(cell.strip() for cell in header)
    if has_header:
        out.append("<thead><tr>")
        for cell in header:
            out.append(f"<th>{inline(cell.strip())}</th>")
        out.append("</tr></thead>")
    out.append("<tbody>")
    body_rows = rows[1:] if has_header else rows
    for row in body_rows:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{inline(cell.strip())}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    # Split on unescaped pipes.
    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in line:
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
    return cells


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return bool(re.fullmatch(r"\|[\s:|-]+\|", stripped))


def markdown_to_html(md: str) -> str:
    """Convert the Markdown subset used by the report to HTML."""
    # An optional cover page: everything between <!-- cover --> and
    # <!-- /cover --> is rendered inside a centred, full-page block.
    cover = ""
    if "<!-- cover -->" in md and "<!-- /cover -->" in md:
        head, rest = md.split("<!-- cover -->", 1)
        cover_md, md = rest.split("<!-- /cover -->", 1)
        cover = '<div class="cover">' + markdown_to_html(cover_md) + "</div>"

    lines = md.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # ---- fenced code ------------------------------------------------
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            buf: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(buf))}</code></pre>")
            continue

        # ---- heading ------------------------------------------------------
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            out.append(f"<h{level}>{inline(text)}</h{level}>")
            i += 1
            continue

        # ---- figure -------------------------------------------------------
        m = IMAGE.match(stripped)
        if m:
            alt, src = m.group(1), m.group(2)
            out.append(
                f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"/>'
                f"<figcaption>{inline(alt)}</figcaption></figure>"
            )
            i += 1
            continue

        # ---- table --------------------------------------------------------
        if stripped.startswith("|") and i + 1 < n and is_table_separator(lines[i + 1]):
            header = split_row(stripped)
            i += 2
            body: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                body.append(split_row(lines[i]))
                i += 1
            out.append(render_table([header] + body, []))
            continue

        # ---- block quote ---------------------------------------------------
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            inner = markdown_to_html("\n".join(buf))
            out.append(f"<blockquote>{inner}</blockquote>")
            continue

        # ---- horizontal rule ------------------------------------------------
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            out.append("<hr/>")
            i += 1
            continue

        # ---- list -----------------------------------------------------------
        if re.match(r"^(\s*)([-*+]|\d+\.)\s+", line):
            out.append(_render_list(lines, i)[0])
            i = _render_list(lines, i)[1]
            continue

        # ---- paragraph -------------------------------------------------------
        if not stripped:
            i += 1
            continue
        buf = []
        while i < n:
            cur = lines[i]
            if (not cur.strip() or cur.strip().startswith(("#", "|", "```", ">"))
                    or re.match(r"^(\s*)([-*+]|\d+\.)\s+", cur)):
                break
            buf.append(cur.strip())
            i += 1
        out.append(f"<p>{inline(' '.join(buf))}</p>")

    return cover + "\n".join(out)


def _render_list(lines: list[str], start: int) -> tuple[str, int]:
    """Render a (possibly nested) list starting at ``start``."""
    i = start
    n = len(lines)
    base_indent = len(lines[i]) - len(lines[i].lstrip())
    ordered = bool(re.match(r"^\s*\d+\.\s", lines[i]))
    items: list[str] = []
    tag = "ol" if ordered else "ul"

    while i < n:
        line = lines[i]
        if not line.strip():
            # A blank line ends the list unless the next line continues it.
            if i + 1 < n and re.match(rf"^\s{{0,{base_indent + 3}}}[-*+]\s|^\s{{0,{base_indent + 3}}}\d+\.\s",
                                      lines[i + 1]):
                i += 1
                continue
            break
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if not m:
            break
        indent = len(m.group(1))
        if indent < base_indent:
            break
        if indent > base_indent:
            sub, i = _render_list(lines, i)
            if items:
                items[-1] = items[-1][: items[-1].rfind("</li>")] + sub + "</li>"
            else:
                items.append(f"<li>{sub}</li>")
            continue
        text = m.group(3).strip()
        i += 1
        # Continuation lines belong to the current item.
        while i < n and lines[i].strip() and not re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]) \
                and not lines[i].strip().startswith(("#", "|", "```")):
            text += " " + lines[i].strip()
            i += 1
        items.append(f"<li>{inline(text)}</li>")

    return f"<{tag}>" + "".join(items) + f"</{tag}>", i


def build_document(body_html: str, title: str, footer: str) -> str:
    """Wrap the rendered body in a full HTML document."""
    footer_html = ""
    if footer:
        footer_html = (
            '<div style="position:fixed;left:0;right:0;bottom:-11mm;text-align:center;'
            'font-family:Inter,sans-serif;font-size:7.4pt;color:#7890a8;">'
            f"{html.escape(footer)}</div>"
        )
    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
        '<meta charset="utf-8"/>\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        f"{footer_html}\n{body_html}\n</body>\n</html>\n"
    )


def find_browser(explicit: str = "") -> str:
    """Locate a Chromium based browser for headless printing."""
    if explicit:
        return explicit
    env = os.environ.get("CHROME_PATH") or os.environ.get("BROWSER")
    if env and Path(env).exists():
        return env
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("No Chromium browser found; pass --browser <path>")


def html_to_pdf(html_path: Path, pdf_path: Path, browser: str) -> None:
    """Print an HTML file to PDF with a headless browser."""
    profile = tempfile.mkdtemp(prefix="md2pdf-")
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if not pdf_path.exists():
            raise SystemExit(
                "PDF was not produced.\n"
                + result.stdout.decode("utf-8", "ignore")[:2000]
                + result.stderr.decode("utf-8", "ignore")[:2000]
            )
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def count_pages(pdf_path: Path) -> int:
    """Count the pages of a PDF without any third-party dependency.

    The page objects of a Chromium produced PDF are written as ``/Type /Page``
    followed by the page dictionary; the page tree uses ``/Type /Pages``.
    """
    data = pdf_path.read_bytes()
    matches = re.findall(rb"/Type\s*/Page(?![s])", data)
    if matches:
        return len(matches)
    # Fall back to the page-tree /Count entry.
    counts = [int(m) for m in re.findall(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data)]
    return max(counts) if counts else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="the Markdown file")
    ap.add_argument("target", nargs="?", help="the PDF file (default: <source>.pdf)")
    ap.add_argument("--browser", default="", help="path to chrome/edge")
    ap.add_argument("--title", default="", help="document title")
    ap.add_argument("--footer", default="", help="running footer text")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="fail if the PDF has more pages than this (0 = no limit)")
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    dst = (Path(args.target) if args.target else src.with_suffix(".pdf")).resolve()
    md = src.read_text(encoding="utf-8")

    body = markdown_to_html(md)
    title = args.title or src.stem
    html_text = build_document(body, title, args.footer)

    tmp_html = dst.with_suffix(".html")
    tmp_html.write_text(html_text, encoding="utf-8")
    try:
        html_to_pdf(tmp_html, dst, find_browser(args.browser))
    finally:
        if not args.keep_html:
            tmp_html.unlink(missing_ok=True)

    size = dst.stat().st_size
    pages = count_pages(dst)
    print(f"PDF written: {dst}  ({size / 1024:.0f} KB, {pages} pages)")
    if args.max_pages and pages > args.max_pages:
        print(f"ERROR: {pages} pages exceeds the limit of {args.max_pages}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
