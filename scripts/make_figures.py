"""Generate the report figures as SVG, then rasterise them with headless Edge.

Why SVG + a browser instead of matplotlib: the figures follow the visual
language of academic overview diagrams (tinted zones, small icons, little agent
characters, elbow arrows).  All of that is trivial in SVG and painful in a
plotting library.  Text stays real text in the SVG, and the PNG is rendered at
3x so it stays sharp in the PDF.

Usage:
    python scripts/make_figures.py              # every figure
    python scripts/make_figures.py arch agent   # a subset
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

# palette
INK = "#1d2836"
SOFT = "#405670"
FAINT = "#7d8fa5"
LINE = "#c8d4e2"
WHITE = "#ffffff"
ACCENT = "#b74c06"

ZONE_BLUE = "#eaf1f9"
ZONE_ORANGE = "#fdf1e7"
ZONE_GREEN = "#ecf5ee"
ZONE_PURPLE = "#f1edf9"
ZONE_GREY = "#f4f6f9"

# Coloured header bars, in the manner of an overview figure: a solid band on top
# of a very light body.
TONES = {
    "blue":    {"head": "#3f6fb5", "body": "#f2f6fc", "edge": "#b9c9e2"},
    "purple":  {"head": "#6d4fa3", "body": "#f6f3fb", "edge": "#cabfe4"},
    "green":   {"head": "#3f8f6a", "body": "#f1f8f4", "edge": "#b6d8c6"},
    "amber":   {"head": "#c07a2c", "body": "#fdf6ef", "edge": "#e8c9a8"},
    "grey":    {"head": "#5b6b80", "body": "#f6f8fa", "edge": "#ccd6e0"},
}

FONT = "'Microsoft YaHei','Segoe UI',sans-serif"


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ------------------------------------------------------------------ primitives
def rect(x, y, w, h, *, fill=WHITE, stroke=INK, sw=1.4, r=10, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} '
        f'opacity="{opacity}"/>'
    )


def text(x, y, s, *, size=18, fill=INK, weight="normal", anchor="middle", family=FONT,
         opacity=1.0):
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" '
        f'opacity="{opacity}" dominant-baseline="middle">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, *, stroke=SOFT, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d} stroke-linecap="round"/>')


def arrow(x1, y1, x2, y2, *, stroke=SOFT, sw=1.8, dash=None, head=9):
    """Straight arrow with a triangular head."""
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    left = (hx - head * 0.55 * math.cos(ang - 1.05), hy - head * 0.55 * math.sin(ang - 1.05))
    right = (hx - head * 0.55 * math.cos(ang + 1.05), hy - head * 0.55 * math.sin(ang + 1.05))
    parts = [line(x1, y1, hx, hy, stroke=stroke, sw=sw, dash=dash)]
    parts.append(
        f'<polygon points="{x2},{y2} {left[0]:.1f},{left[1]:.1f} {right[0]:.1f},{right[1]:.1f}" '
        f'fill="{stroke}"/>'
    )
    return "".join(parts)


def elbow(x1, y1, x2, y2, *, stroke=SOFT, sw=1.8, mid=None, dash=None):
    """Right-angle connector: horizontal, vertical, horizontal."""
    mx = mid if mid is not None else (x1 + x2) / 2
    return (
        line(x1, y1, mx, y1, stroke=stroke, sw=sw, dash=dash)
        + line(mx, y1, mx, y2, stroke=stroke, sw=sw, dash=dash)
        + arrow(mx, y2, x2, y2, stroke=stroke, sw=sw, dash=dash)
    )


def zone(x, y, w, h, title, *, tone="grey", r=12, bar=40):
    """Panel with a solid coloured header bar and a very light body."""
    colours = TONES.get(tone, TONES["grey"])
    head, body, edge = colours["head"], colours["body"], colours["edge"]
    out = [rect(x, y, w, h, fill=body, stroke=edge, sw=1.3, r=r)]
    # header bar: rounded on the top two corners only
    out.append(
        f'<path d="M {x + r} {y} H {x + w - r} A {r} {r} 0 0 1 {x + w} {y + r} '
        f'V {y + bar} H {x} V {y + r} A {r} {r} 0 0 1 {x + r} {y} Z" '
        f'fill="{head}"/>'
    )
    out.append(text(x + 18, y + bar / 2 + 1, title, size=18, weight="bold",
                    fill="#ffffff", anchor="start"))
    return "".join(out)


# ------------------------------------------------------------------ iconography
def agent(x, y, s=1.0, *, tone="#dfe9f6", stroke=INK, label=None, size=13):
    """A small agent character: head, antenna, eyes, body."""
    out = []
    w, h = 34 * s, 30 * s
    # body
    out.append(rect(x - w / 2 + 4 * s, y, w - 8 * s, h * 0.62, fill=tone, stroke=stroke,
                    sw=1.4 * s, r=7 * s))
    # head
    hy = y - h * 0.72
    out.append(rect(x - w / 2, hy, w, h * 0.62, fill=WHITE, stroke=stroke, sw=1.4 * s, r=8 * s))
    # eyes
    out.append(f'<circle cx="{x - 6 * s}" cy="{hy + h * 0.30}" r="{2.4 * s}" fill="{stroke}"/>')
    out.append(f'<circle cx="{x + 6 * s}" cy="{hy + h * 0.30}" r="{2.4 * s}" fill="{stroke}"/>')
    # antenna
    out.append(line(x, hy - 2 * s, x, hy - 9 * s, stroke=stroke, sw=1.3 * s))
    out.append(f'<circle cx="{x}" cy="{hy - 11 * s}" r="{2.6 * s}" fill="{ACCENT}"/>')
    if label:
        out.append(text(x, y + h * 0.62 + 14 * s, label, size=size, fill=SOFT))
    return "".join(out)


def doc(x, y, s=1.0, *, fill=WHITE, stroke=INK, lines=3):
    w, h = 22 * s, 28 * s
    out = [rect(x - w / 2, y - h / 2, w, h, fill=fill, stroke=stroke, sw=1.2 * s, r=3 * s)]
    for i in range(lines):
        ly = y - h / 2 + 8 * s + i * 6 * s
        out.append(line(x - w / 2 + 5 * s, ly, x + w / 2 - 5 * s, ly, stroke=FAINT, sw=1.1 * s))
    return "".join(out)


def database(x, y, s=1.0, *, fill="#dfe9f6", stroke=INK):
    w, h, ry = 30 * s, 34 * s, 5 * s
    out = [
        f'<rect x="{x - w/2}" y="{y - h/2}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{1.3*s}" rx="{ry}"/>',
        f'<ellipse cx="{x}" cy="{y - h/2}" rx="{w/2}" ry="{ry}" fill="{WHITE}" '
        f'stroke="{stroke}" stroke-width="{1.3*s}"/>',
    ]
    for i in (0.06, 0.42):
        yy = y - h / 2 + h * i + ry
        out.append(f'<path d="M {x - w/2} {yy} A {w/2} {ry} 0 0 0 {x + w/2} {yy}" fill="none" '
                   f'stroke="{FAINT}" stroke-width="{1.1*s}"/>')
    return "".join(out)


def shield(x, y, s=1.0, *, fill="#f6e6dc", stroke=ACCENT):
    w, h = 24 * s, 28 * s
    return (
        f'<path d="M {x} {y - h/2} L {x + w/2} {y - h/2 + 5*s} L {x + w/2} {y + h*0.1} '
        f'Q {x + w/2} {y + h/2} {x} {y + h/2} Q {x - w/2} {y + h/2} {x - w/2} {y + h*0.1} '
        f'L {x - w/2} {y - h/2 + 5*s} Z" fill="{fill}" stroke="{stroke}" stroke-width="{1.3*s}"/>'
        + f'<path d="M {x-5*s} {y} L {x-1*s} {y+5*s} L {x+6*s} {y-5*s}" fill="none" '
          f'stroke="{stroke}" stroke-width="{1.8*s}" stroke-linecap="round"/>'
    )


def search(x, y, s=1.0, *, fill=WHITE, stroke=INK):
    return (
        f'<circle cx="{x - 2*s}" cy="{y - 2*s}" r="{9*s}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{1.4*s}"/>'
        + line(x + 5 * s, y + 5 * s, x + 12 * s, y + 12 * s, stroke=stroke, sw=2.2 * s)
    )


def graph_icon(x, y, s=1.0, *, stroke=INK, fill="#dfe9f6"):
    pts = [(x, y - 12 * s), (x - 12 * s, y + 8 * s), (x + 12 * s, y + 8 * s), (x + 16 * s, y - 6 * s)]
    out = []
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            out.append(line(pts[a][0], pts[a][1], pts[b][0], pts[b][1], stroke=FAINT, sw=1.1 * s))
    for px, py in pts:
        out.append(f'<circle cx="{px}" cy="{py}" r="{4.2*s}" fill="{fill}" stroke="{stroke}" '
                   f'stroke-width="{1.3*s}"/>')
    return "".join(out)


def chip(x, y, s=1.0, *, fill="#fdf1e7", stroke=INK, label=""):
    w, h = 30 * s, 30 * s
    out = [rect(x - w / 2, y - h / 2, w, h, fill=fill, stroke=stroke, sw=1.3 * s, r=5 * s)]
    for i in range(3):
        off = -8 * s + i * 8 * s
        out.append(line(x - w / 2 - 5 * s, y + off, x - w / 2, y + off, stroke=stroke, sw=1.2 * s))
        out.append(line(x + w / 2, y + off, x + w / 2 + 5 * s, y + off, stroke=stroke, sw=1.2 * s))
    if label:
        out.append(text(x, y + h / 2 + 12 * s, label, size=13, fill=SOFT))
    return "".join(out)


def users(x, y, s=1.0, *, stroke=INK, fill="#e8eef7"):
    out = []
    for dx, dy in ((0, 0), (-11 * s, 5 * s), (11 * s, 5 * s)):
        out.append(f'<circle cx="{x+dx}" cy="{y+dy-6*s}" r="{5*s}" fill="{fill}" '
                   f'stroke="{stroke}" stroke-width="{1.2*s}"/>')
        out.append(f'<path d="M {x+dx-7*s} {y+dy+8*s} Q {x+dx} {y+dy-4*s} {x+dx+7*s} {y+dy+8*s}" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="{1.2*s}"/>')
    return "".join(out)


# ----------------------------------------------------------------------- canvas
class Figure:
    def __init__(self, width: int, height: int, name: str):
        self.w, self.h, self.name = width, height, name
        self.parts: list[str] = []

    def add(self, *chunks: str) -> "Figure":
        self.parts.extend(chunks)
        return self

    def svg(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">'
            + f'<rect width="{self.w}" height="{self.h}" fill="{WHITE}"/>'
            + "".join(self.parts)
            + "</svg>"
        )

    def write(self) -> pathlib.Path:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{self.name}.svg"
        path.write_text(self.svg(), encoding="utf-8")
        return path


def render(svg_path: pathlib.Path, *, scale: int = 3) -> pathlib.Path:
    """Rasterise with headless Edge/Chrome at `scale` device pixels per unit."""
    edge = next((p for p in EDGE_CANDIDATES if pathlib.Path(p).exists()), None)
    if edge is None:
        raise SystemExit("no Edge/Chrome found")
    png = svg_path.with_suffix(".png")
    profile = tempfile.mkdtemp(prefix="fig-")
    width = int(svg_path.read_text(encoding="utf-8").split('width="')[1].split('"')[0])
    height = int(svg_path.read_text(encoding="utf-8").split('height="')[1].split('"')[0])
    subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--no-first-run", f"--user-data-dir={profile}",
         f"--force-device-scale-factor={scale}",
         f"--window-size={width},{height}",
         f"--screenshot={png}", svg_path.as_uri()],
        capture_output=True, timeout=180,
    )
    return png


# ----------------------------------------------------------------------- fig 1
def fig_arch() -> None:
    f = Figure(1560, 935, "fig-arch")

    # ---- zone A: presentation
    f.add(zone(40, 60, 1480, 150, "展现层 · 六个工作台", tone="blue"))
    benches = ["文献库", "阅读器", "问答", "图谱", "检索", "写作"]
    for index, name in enumerate(benches):
        x = 110 + index * 235
        f.add(rect(x, 112, 190, 70, fill=WHITE, stroke=LINE, sw=1.2, r=12))
        f.add(text(x + 95, 122 + 26, name, size=22, weight="bold", fill=SOFT))
        f.add(text(x + 95, 122 + 52, "React 组件", size=14, fill=FAINT))

    # ---- zone B: orchestration
    f.add(zone(40, 275, 1480, 215, "编排层 · Agent 两阶段", tone="amber"))
    stages = [
        ("Router", "意图路由", agent),
        ("Planner", "任务 DAG", agent),
        ("Executor", "工具循环", agent),
        ("Reflection", "护栏与引用", shield),
    ]
    for index, (name, sub, glyph) in enumerate(stages):
        x = 210 + index * 330
        f.add(rect(x - 95, 330, 190, 118, fill=WHITE, stroke=LINE, sw=1.2, r=14))
        if glyph is agent:
            f.add(agent(x, 386, 1.05, tone="#fde7d5"))
        else:
            f.add(shield(x, 380, 1.15))
        f.add(text(x, 420, name, size=20, weight="bold", fill=INK))
        f.add(text(x, 442, sub, size=15, fill=FAINT))
        if index < len(stages) - 1:
            f.add(arrow(x + 96, 389, x + 330 - 96, 389))

    f.add(text(760, 472, "阶段 A 取证：产出带 chunk_id / 页码 / 字符区间的证据列表　｜　阶段 B 生成：只依据这些证据",
               size=15, fill=SOFT))

    # ---- zone C: capabilities
    f.add(zone(40, 545, 1480, 150, "能力层", tone="green"))
    caps = [
        (doc, "版式解析", "栏聚类 · 标题树 · 图表公式"),
        (search, "混合检索", "BM25 + 向量 + 重排"),
        (graph_icon, "引文图谱", "PageRank · 基石 · 矛盾"),
        (chip, "外部工具", "arXiv · GitHub · 绘图"),
    ]
    for index, (glyph, name, sub) in enumerate(caps):
        x = 230 + index * 360
        f.add(rect(x - 150, 590, 300, 80, fill=WHITE, stroke=LINE, sw=1.2, r=12))
        f.add(glyph(x - 108, 630, 1.05))
        f.add(text(x - 58, 616, name, size=20, weight="bold", fill=SOFT, anchor="start"))
        f.add(text(x - 58, 646, sub, size=14.5, fill=FAINT, anchor="start"))

    # ---- zone D: storage + models side by side
    f.add(zone(40, 745, 720, 130, "存储层", tone="grey"))
    f.add(database(120, 810, 1.2))
    f.add(text(160, 800, "SQLite：papers · sections · chunks", size=16, fill=SOFT, anchor="start"))
    f.add(text(160, 824, "向量索引 .npy，memmap 按需加载", size=14.5, fill=FAINT, anchor="start"))

    f.add(zone(800, 745, 720, 130, "模型层", tone="purple"))
    f.add(chip(880, 810, 1.15, label=""))
    f.add(text(915, 796, "vLLM → Qwen3.8-27B", size=16, fill=SOFT, anchor="start"))
    f.add(text(915, 820, "bge-m3 稠密 · bge-reranker 精排", size=14.5, fill=FAINT, anchor="start"))

    # ---- flow arrows between zones
    f.add(arrow(780, 212, 780, 272, stroke=FAINT, sw=2))
    f.add(arrow(780, 492, 780, 542, stroke=FAINT, sw=2))

    # ---- left edge entry
    f.add(users(1440, 40, 0.85))
    f.add(text(1470, 40, "浏览器 / 评审", size=14, fill=FAINT, anchor="start"))

    png = render(f.write())
    print("wrote", png.name)


# ----------------------------------------------------------------------- fig 2
def fig_agent() -> None:
    """Two-stage orchestration: evidence first, then generation on evidence only."""
    f = Figure(1560, 500, "fig-agent")

    # stage A
    f.add(zone(40, 70, 800, 400, "阶段 A · 取证", tone="amber"))

    f.add(rect(86, 150, 150, 76, fill=WHITE, stroke=LINE, sw=1.2, r=12))
    f.add(users(161, 186, 0.8))
    f.add(text(161, 214, "用户问题", size=15, fill=SOFT))

    f.add(arrow(240, 188, 292, 188))
    f.add(rect(296, 140, 190, 96, fill=WHITE, stroke=LINE, sw=1.2, r=12))
    f.add(agent(391, 190, 0.95, tone="#fde7d5"))
    f.add(text(391, 220, "Router / Planner", size=15, weight="bold", fill=INK))

    f.add(arrow(489, 188, 541, 188))
    f.add(rect(545, 140, 190, 96, fill=WHITE, stroke=LINE, sw=1.2, r=12))
    f.add(agent(640, 190, 0.95, tone="#fde7d5"))
    f.add(text(640, 220, "Executor · 工具循环", size=15, weight="bold", fill=INK))

    # tool row
    f.add(arrow(640, 240, 640, 296))
    tools = [(search, "search_library"), (doc, "read_passages"), (graph_icon, "build_graph"),
             (chip, "arxiv · github")]
    for index, (glyph, label) in enumerate(tools):
        x = 200 + index * 165
        f.add(rect(x - 62, 300, 124, 96, fill=WHITE, stroke=LINE, sw=1.1, r=10))
        f.add(glyph(x, 336, 0.95))
        f.add(text(x, 378, label, size=12.5, fill=FAINT))

    f.add(arrow(640, 396, 640, 420))
    f.add(rect(500, 424, 280, 36, fill=WHITE, stroke=ACCENT, sw=1.4, r=8))
    f.add(text(640, 443, "证据列表：页码 · 字符区间 · 得分", size=13, fill=ACCENT))

    # stage B
    f.add(zone(880, 70, 640, 400, "阶段 B · 带证据生成", tone="blue"))

    f.add(arrow(782, 442, 906, 292))

    f.add(rect(912, 150, 250, 100, fill=WHITE, stroke=LINE, sw=1.2, r=12))
    f.add(agent(1037, 204, 0.95, tone="#dfe9f6"))
    f.add(text(1037, 232, "生成（流式）", size=15, weight="bold", fill=INK))

    f.add(arrow(1164, 200, 1214, 200))
    f.add(rect(1218, 150, 250, 100, fill=WHITE, stroke=LINE, sw=1.2, r=12))
    f.add(shield(1343, 194, 1.1))
    f.add(text(1343, 232, "Reflection 护栏", size=15, weight="bold", fill=INK))

    f.add(arrow(1343, 252, 1343, 320))
    f.add(rect(1180, 324, 300, 58, fill=WHITE, stroke=INK, sw=1.4, r=10))
    f.add(text(1330, 353, "带引用的答案", size=16, weight="bold", fill=INK))
    f.add(text(1060, 300, "超范围引用 → 删除", size=12.5, fill=FAINT))
    f.add(arrow(1178, 353, 1100, 353, stroke=FAINT, sw=1.2))
    f.add(text(1040, 378, "→ SSE 流式回前端", size=12.5, fill=FAINT))

    png = render(f.write())
    print("wrote", png.name)


# ----------------------------------------------------------------------- fig 3
def fig_seq() -> None:
    """Sequence of one cross-paper question."""
    f = Figure(1420, 800, "fig-seq")

    lanes = [
        ("用户", 120),
        ("前端", 470),
        ("Agent 编排", 820),
        ("检索层", 1180),
    ]
    for name, x in lanes:
        f.add(rect(x - 82, 34, 164, 48, fill=ZONE_BLUE if x < 800 else ZONE_ORANGE,
                   stroke=LINE, sw=1.2, r=10))
        f.add(text(x, 58, name, size=16, weight="bold", fill=SOFT))
        f.add(line(x, 84, x, 760, stroke=LINE, sw=1.2, dash="5 6"))

    messages = [
        (120, 470, 120, "POST /api/rag/ask", INK),
        (470, 820, 165, "Router：意图 = 跨文献对比", SOFT),
        (470, 820, 210, "Planer：DAG = [检索 × 2, 综合]", SOFT, "above"),
        (820, 470, 255, "SSE：route / plan", SOFT, "above"),
        (820, 1180, 305, "tool_call：search_papers", SOFT),
        (1180, 820, 350, "passages 列表（带字符区间）", SOFT, "above"),
        (820, 470, 395, "SSE：tool_call / tool_result", SOFT, "above"),
        (820, 1180, 445, "generate（只读 evidence）", ACCENT),
        (820, 470, 490, "SSE：thinking → delta", SOFT, "above"),
        (820, 470, 545, "SSE：guardrail（删掉无据引用）", SOFT, "above"),
        (820, 470, 600, "SSE：final（answer + citations）", INK, "above"),
    ]
    for message in messages:
        x1, x2, y, label = message[0], message[1], message[2], message[3]
        color = message[4] if len(message) > 4 else SOFT
        above = len(message) > 5 and message[5] == "above"
        f.add(arrow(x1, y, x2, y, stroke=color, sw=1.6))
        f.add(text((x1 + x2) / 2, y - 16 if above else y + 16, label, size=14, fill=color))

    f.add(text(710, 700, "自反馈：未被证据覆盖的断言 → 重新检索、再生成（最多 2 轮）",
               size=15, fill=FAINT))
    f.add(rect(430, 730, 560, 46, fill=WHITE, stroke=LINE, sw=1.2, r=10, dash="6 5"))
    f.add(text(710, 753, "两轮后仍无依据 → 诚实拒答，而不是编一个答案", size=14, fill=SOFT))

    png = render(f.write())
    print("wrote", png.name)


# ----------------------------------------------------------------------- fig 4
def fig_deploy() -> None:
    """Request path of the single-process deployment."""
    f = Figure(1260, 620, "fig-deploy")

    f.add(rect(430, 40, 400, 66, fill=ZONE_BLUE, stroke=LINE, sw=1.2, r=12))
    f.add(users(470, 72, 0.7))
    f.add(text(640, 73, "浏览器 / 评审", size=17, weight="bold", fill=SOFT))

    f.add(arrow(630, 108, 630, 158))
    f.add(rect(430, 160, 400, 70, fill=WHITE, stroke=LINE, sw=1.2, r=12))
    f.add(doc(470, 195, 0.85))
    f.add(text(645, 196, "Cloudflare 隧道（HTTPS）", size=16, weight="bold", fill=SOFT))

    f.add(arrow(630, 232, 630, 282))
    f.add(rect(330, 284, 600, 196, fill=ZONE_ORANGE, stroke=LINE, sw=1.2, r=14))
    f.add(text(630, 316, "FastAPI :9000（单进程同源）", size=18, weight="bold", fill=INK))
    f.add(rect(370, 344, 240, 52, fill=WHITE, stroke=LINE, sw=1.1, r=10))
    f.add(text(490, 370, "/api/* → 路由", size=14.5, fill=SOFT))
    f.add(rect(650, 344, 240, 52, fill=WHITE, stroke=LINE, sw=1.1, r=10))
    f.add(text(770, 370, "/* → SPA dist", size=14.5, fill=SOFT))
    f.add(rect(370, 408, 520, 50, fill=WHITE, stroke=LINE, sw=1.1, r=10))
    f.add(text(630, 433, "静态资源与接口同一 origin，无跨域、无第二个 URL", size=13.5, fill=FAINT))

    f.add(arrow(630, 480, 630, 530))
    f.add(rect(430, 532, 400, 66, fill=ZONE_PURPLE, stroke=LINE, sw=1.2, r=12))
    f.add(chip(470, 565, 0.95))
    f.add(text(645, 566, "vLLM · Qwen3.8-27B", size=16, weight="bold", fill=SOFT))

    # side notes
    f.add(rect(60, 284, 230, 196, fill=ZONE_GREY, stroke=LINE, sw=1.2, r=12, dash="6 5"))
    f.add(text(175, 312, "同一台主机", size=15, weight="bold", fill=SOFT))
    f.add(text(175, 348, "解析 / 索引 / 检索", size=13.5, fill=FAINT))
    f.add(text(175, 374, "全部本地", size=13.5, fill=FAINT))
    f.add(text(175, 412, "GPU 只服务推理", size=13.5, fill=FAINT))
    f.add(text(175, 438, "CPU 64 线程做向量", size=13.5, fill=FAINT))

    f.add(arrow(290, 382, 330, 382, stroke=FAINT, sw=1.4))

    f.add(rect(970, 284, 230, 196, fill=ZONE_GRAY if False else ZONE_GREY,
                stroke=LINE, sw=1.2, r=12, dash="6 5"))
    f.add(text(1085, 312, "数据与索引", size=15, weight="bold", fill=SOFT))
    f.add(database(1085, 360, 1.0))
    f.add(text(1085, 420, "SQLite + .npy", size=13.5, fill=FAINT))
    f.add(text(1085, 446, "按篇落盘，可备份", size=13.5, fill=FAINT))

    f.add(arrow(930, 382, 970, 382, stroke=FAINT, sw=1.4))

    png = render(f.write())
    print("wrote", png.name)


# ----------------------------------------------------------------------- fig 5
def fig_retrieval() -> None:
    """Hybrid retrieval: two recall channels, fusion, rerank."""
    f = Figure(1460, 430, "fig-retrieval")

    f.add(rect(40, 210, 190, 90, fill=ZONE_BLUE, stroke=LINE, sw=1.2, r=12))
    f.add(text(135, 244, "查询", size=19, weight="bold", fill=INK))
    f.add(text(135, 274, "中文或英文", size=13, fill=FAINT))

    # two channels
    f.add(arrow(232, 240, 288, 180))
    f.add(arrow(232, 270, 288, 330))

    f.add(rect(292, 130, 300, 100, fill=WHITE, stroke=LINE, sw=1.2, r=12))
    f.add(search(340, 180, 0.9))
    f.add(text(380, 166, "BM25 稀疏", size=17, weight="bold", fill=SOFT, anchor="start"))
    f.add(text(380, 194, "精确术语 · 缩写 · 型号", size=13, fill=FAINT, anchor="start"))

    f.add(rect(292, 290, 300, 100, fill=WHITE, stroke=LINE, sw=1.2, r=12))
    f.add(chip(340, 340, 0.85))
    f.add(text(380, 326, "bge-m3 稠密", size=17, weight="bold", fill=SOFT, anchor="start"))
    f.add(text(380, 354, "语义近似 · 跨语言", size=13, fill=FAINT, anchor="start"))

    f.add(arrow(594, 180, 660, 232))
    f.add(arrow(594, 340, 660, 288))

    f.add(rect(664, 214, 220, 92, fill=ZONE_ORANGE, stroke=LINE, sw=1.2, r=12))
    f.add(text(774, 246, "RRF 融合", size=17, weight="bold", fill=INK))
    f.add(text(774, 274, "只用排名，不调权重", size=12.5, fill=FAINT))

    f.add(arrow(886, 260, 946, 260))

    f.add(rect(950, 208, 250, 104, fill=ZONE_GREEN, stroke=LINE, sw=1.2, r=12))
    f.add(text(1075, 240, "cross-encoder 精排", size=16, weight="bold", fill=INK))
    f.add(text(1075, 268, "bge-reranker-v2-m3 · Top-24", size=12.5, fill=FAINT))
    f.add(text(1075, 292, "元数据前缀参与打分", size=12.5, fill=FAINT))

    f.add(arrow(1202, 260, 1258, 260))

    f.add(rect(1262, 190, 170, 140, fill=WHITE, stroke=ACCENT, sw=1.5, r=12))
    f.add(text(1347, 226, "Top-K 证据", size=17, weight="bold", fill=ACCENT))
    f.add(text(1347, 258, "chunk_id", size=12.5, fill=FAINT))
    f.add(text(1347, 282, "页码 · 字符区间", size=12.5, fill=FAINT))
    f.add(text(1347, 306, "可点回原文", size=12.5, fill=FAINT))

    f.add(text(730, 480, "两路分数不可比：BM25 无界、余弦有界，所以融合只用排名",
               size=14, fill=FAINT))

    png = render(f.write())
    print("wrote", png.name)


FIGURES = {
    "arch": fig_arch,
    "agent": fig_agent,
    "seq": fig_seq,
    "deploy": fig_deploy,
    "retrieval": fig_retrieval,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=[])
    args = ap.parse_args()
    for name in (args.names or list(FIGURES)):
        FIGURES[name]()


if __name__ == "__main__":
    main()
