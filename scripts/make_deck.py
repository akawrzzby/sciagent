"""Build the defence deck from the XJTU template.

The template is kept as the base so the master, the fonts and the cover design
come from it; the content slides are rebuilt in the same visual language (red
top bar, bold section title, white body) and every feature slide carries an
empty 16:9 frame where the recorded clip goes.
"""

from __future__ import annotations

import copy
import pathlib

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

TEMPLATE = pathlib.Path(r"E:\西交\研究生\面试ppt\党员工作站 - 副本.pptx")
RENDERED = pathlib.Path(r"E:\西交\研究生\比赛\AI大模型创新应用挑战赛\sciagent\docs\_tpl_p1.png")
OUT = pathlib.Path(r"E:\西交\研究生\面试ppt\SciAgent-答辩.pptx")
FIGURES = pathlib.Path(r"E:\西交\研究生\比赛\AI大模型创新应用挑战赛\sciagent\docs\figures")

# the red of the cover overlay, sampled rather than guessed
image = Image.open(RENDERED).convert("RGB")
RED = RGBColor(*image.getpixel((30, 30)))
INK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x5B, 0x66, 0x70)
LINE = RGBColor(0xD8, 0xDC, 0xE0)

prs = Presentation(str(TEMPLATE))
SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height

# ---- keep the cover, drop everything else ---------------------------------
cover = prs.slides[0]
for index in range(len(prs.slides) - 1, 0, -1):
    slide_id = prs.slides._sldIdLst[index]
    prs.part.drop_rel(slide_id.rId)
    del prs.slides._sldIdLst[index]

# cover texts
for shape in cover.shapes:
    if not shape.has_text_frame:
        continue
    text = shape.text_frame.text
    if "党员工作站" in text:
        para = shape.text_frame.paragraphs[0]
        for run in para.runs[1:]:
            run._r.getparent().remove(run._r)
        if para.runs:
            para.runs[0].text = "SciAgent：意图驱动的科研助手"
            # the original title was shorter, so the size comes down to keep
            # the new one on a single line, and it is aligned with the rule
            para.runs[0].font.size = Pt(40)
            para.alignment = PP_ALIGN.LEFT
    elif "汇报人" in text or "时间" in text:
        para = shape.text_frame.paragraphs[0]
        for run in para.runs[1:]:
            run._r.getparent().remove(run._r)
        if para.runs:
            para.runs[0].text = "参赛项目：科研小助手 Agent        2026 年 11 月"


def blank_slide():
    layout = prs.slide_layouts[6]
    return prs.slides.add_slide(layout)


def textbox(slide, x, y, w, h, text, *, size=18, bold=False, color=INK,
            align=PP_ALIGN.LEFT, font="微软雅黑", spacing=1.25):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.TOP
    lines = text.split("\n")
    for index, line in enumerate(lines):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return box


def rect(slide, x, y, w, h, color, *, shape=MSO_SHAPE.RECTANGLE, line=None,
         dash=False):
    box = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    box.shadow.inherit = False
    if color is None:
        box.fill.background()
    else:
        box.fill.solid()
        box.fill.fore_color.rgb = color
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line
        box.line.width = Pt(1.25)
        if dash:
            from pptx.enum.dml import MSO_LINE_DASH_STYLE

            box.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return box


def page_title(slide, title, subtitle="", number=None):
    """The header every content slide shares."""
    rect(slide, 0, 0, 20.0, 0.16, RED)
    textbox(slide, 0.85, 0.62, 14.4, 0.9, title, size=34, bold=True, color=INK)
    rect(slide, 0.9, 1.62, 1.6, 0.09, RED)
    if subtitle:
        textbox(slide, 0.9, 1.86, 17.0, 0.5, subtitle, size=15, color=MUTED)
    if number is not None:
        textbox(slide, 18.2, 10.35, 1.2, 0.4, str(number), size=12,
                color=MUTED, align=PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, items, *, size=17, gap=0.62, bullet_color=RED):
    for index, item in enumerate(items):
        top = y + index * gap
        rect(slide, x, top + 0.12, 0.14, 0.14, bullet_color, shape=MSO_SHAPE.OVAL)
        textbox(slide, x + 0.34, top, w - 0.34, gap, item, size=size)


def video_frame(slide, x, y, w, h, label):
    """An empty 16:9 area where the recorded clip is dropped in."""
    frame = rect(slide, x, y, w, h, RGBColor(0xF7, 0xF8, 0xFA),
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=RED, dash=True)
    frame.adjustments[0] = 0.04
    play = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + w / 2 - 0.45),
                                  Inches(y + h / 2 - 0.62), Inches(0.9), Inches(0.9))
    play.fill.solid()
    play.fill.fore_color.rgb = RED
    play.line.fill.background()
    play.shadow.inherit = False
    textbox(slide, x + w / 2 - 1.1, y + h / 2 - 0.6, 2.2, 0.9, "▶", size=26,
            bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
    textbox(slide, x, y + h / 2 + 0.38, w, 0.5, label, size=13, color=MUTED,
            align=PP_ALIGN.CENTER)
    textbox(slide, x, y + h - 0.62, w, 0.4, "插入 → 视频 → 此设备", size=11,
            color=MUTED, align=PP_ALIGN.CENTER)


def section(label, number):
    slide = blank_slide()
    rect(slide, 0, 0, 20.0, 11.25, RGBColor(0xFA, 0xF3, 0xF4))
    rect(slide, 0, 3.9, 0.42, 3.4, RED)
    textbox(slide, 1.3, 4.15, 8.0, 1.0, f"{number:02d}", size=54, bold=True, color=RED)
    textbox(slide, 1.3, 5.25, 14.0, 1.4, label, size=40, bold=True, color=INK)
    return slide


# --------------------------------------------------------------------- 目录
slide = blank_slide()
page_title(slide, "目录", "CONTENTS", 2)
columns = [
    ("一、项目背景与定位", ["科研阅读的五类困难", "系统定位与设计原则", "赛题要求映射"]),
    ("二、系统设计", ["总体架构与分层", "AI 技术方案：选型", "集成方式与优化策略"]),
    ("三、功能演示", ["六个工作台逐一演示", "特色：证据链与引用卡片"]),
    ("四、验证与总结", ["实测数据", "创新点", "部署与开源"]),
]
for index, (head, items) in enumerate(columns):
    left = 0.95 + (index % 2) * 9.6
    top = 2.55 + (index // 2) * 4.1
    textbox(slide, left, top, 8.6, 0.6, head, size=22, bold=True, color=RED)
    for sub in range(len(items)):
        textbox(slide, left + 0.1, top + 0.78 + sub * 0.62, 8.4, 0.55,
                f"{sub + 1}. {items[sub]}", size=16, color=INK)

# --------------------------------------------------------------- 背景与痛点
slide = blank_slide()
page_title(slide, "科研文献的机器化利用受制于五类问题", "这些困难都来自同一个缺口：结论无法指回原文", 3)
bullets(slide, 0.95, 2.6, 8.9, [
    "PDF 并非结构化文本：双栏顺序、跨栏图表、公式与脚注，直接抽取即错序",
    "生成内容不可信：杜撰参考文献、混淆作者、把一篇的结论安到另一篇",
    "阅读工具割裂：阅读器不懂语义，翻译丢失指代，管理软件不能提问",
    "方向判断缺依据：读完二十篇，说不清哪些空白尚未被填补",
    "写作从零起笔：框架、架构图、数据表，参考文献最易出错",
], size=16, gap=1.15)
rect(slide, 10.3, 2.55, 8.9, 6.7, RGBColor(0xFA, 0xF3, 0xF4),
     shape=MSO_SHAPE.ROUNDED_RECTANGLE)
textbox(slide, 10.7, 2.85, 8.1, 0.6, "有据可查的量级", size=18, bold=True, color=RED)
textbox(slide, 10.7, 3.55, 8.1, 5.4,
        "GPT-4o 在带引用生成任务中的引用幻觉率\n"
        "        78% – 90%\n\n"
        "PubMed Central 上被判定伪造的引用比例\n"
        "        2023 年 1/2828 → 2025 年 1/458\n\n"
        "仅约 26.5% 的 AI 生成引用完全正确\n"
        "        （OpenScholar, Nature 650:857–863）",
        size=15, color=INK, spacing=1.5)

# ------------------------------------------------------------------ 系统定位
slide = blank_slide()
page_title(slide, "SciAgent 的定位：结论必须能指回原文", "三条设计原则，对应三项可度量目标", 4)
principles = [
    ("证据优先", "每一项结论都指回原文的\n字符区间；指不回去就不说"),
    ("结构优先", "标题、表格、公式、参考文献\n分别建模，不做无结构拼接"),
    ("可降级", "稠密通道失效退化为 BM25；\n模型超时给出明确错误"),
]
for index, (head, body) in enumerate(principles):
    left = 0.95 + index * 6.35
    rect(slide, left, 2.75, 5.85, 3.1, RGBColor(0xFA, 0xF3, 0xF4),
         shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    textbox(slide, left + 0.4, 3.0, 5.0, 0.6, head, size=22, bold=True, color=RED)
    textbox(slide, left + 0.4, 3.75, 5.1, 1.9, body, size=15, color=INK, spacing=1.45)
textbox(slide, 0.95, 6.35, 8.6, 0.6, "对应的三个可度量目标", size=18, bold=True, color=INK)
bullets(slide, 0.95, 6.95, 18.2, [
    "解析准确：标准论文的标题层级、作者、表格、参考文献字段全部正确；扫描件产出可检索正文",
    "回答可信：全部回答携带可点击的溯源锚点，含文献、页码与字符区间；证据不足时明确拒答",
    "闭环完整：从文献入库到综述、框架与图表生成，全流程在系统内完成",
], size=15, gap=0.72)

# ------------------------------------------------------------------ 架构图
slide = blank_slide()
page_title(slide, "系统架构：五层，每层只依赖下一层", "分层的目的只有一个：每一层都能单独测", 5)
picture = FIGURES / "fig-arch.png"
if picture.exists():
    slide.shapes.add_picture(str(picture), Inches(1.0), Inches(2.35), width=Inches(11.6))
bullets(slide, 13.1, 2.6, 6.2, [
    "展现层经 /api 与 SSE 事件流访问编排层",
    "编排层调用能力层的纯函数",
    "能力层读写存储，再向下使用模型",
    "解析器不依赖 HTTP，检索器不依赖模型在线",
    "稠密通道不可用时退化为 BM25",
], size=14, gap=0.95)

# ------------------------------------------------------------------ AI 选型
slide = blank_slide()
page_title(slide, "AI 技术方案：选型的理由", "每一项选型都对应一条实测依据", 6)
rows = [
    ("大模型", "Qwen3.8-27B 自托管", "27B 在成本与能力之间取平衡；结构化解析、精确检索、图论计算下沉到确定性代码"),
    ("解析", "PyMuPDF-Layout", "DocLayNet F1 0.8356，对 Docling 的 0.8102；参数 1.8M 对 20M，且无需 GPU"),
    ("检索", "BM25 + bge-m3 + RRF", "BM25 精确命中术语与缩写，稠密通道覆盖语义；RRF 只用排名，无需归一化与调参"),
    ("重排", "bge-reranker-v2-m3", "交叉编码器可捕捉查询与段落的细粒度交互，把正确段落由第 2 位提到第 1 位"),
    ("扫描件", "多模态 VLM", "只有它能同时处理双栏顺序与数学公式，tesseract 两者都不行"),
]
top = 2.45
for name, choice, why in rows:
    rect(slide, 0.95, top, 1.9, 0.62, RED, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    textbox(slide, 0.98, top + 0.1, 1.85, 0.45, name, size=15, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
    textbox(slide, 3.1, top + 0.04, 4.2, 0.55, choice, size=15, bold=True, color=INK)
    textbox(slide, 7.4, top + 0.02, 11.7, 0.62, why, size=13, color=MUTED)
    top += 1.28

# ------------------------------------------------------------ 集成与优化
slide = blank_slide()
page_title(slide, "集成方式与优化策略", "两阶段取证后生成；三处实测优化", 7)
rect(slide, 0.95, 2.5, 9.0, 2.5, RGBColor(0xFA, 0xF3, 0xF4), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
textbox(slide, 1.25, 2.72, 8.4, 0.5, "集成：取证与生成分离", size=19, bold=True, color=RED)
textbox(slide, 1.25, 3.3, 8.4, 1.6,
        "阶段 A 只取证，产出带字符区间的证据集合；\n"
        "阶段 B 只依据证据生成，系统提示中限定引用编号范围；\n"
        "两阶段之间以证据为唯一接口，界面可逐条审计。",
        size=14, color=INK, spacing=1.45)
rect(slide, 10.35, 2.5, 8.85, 2.5, RGBColor(0xF5, 0xF7, 0xFA), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
textbox(slide, 10.65, 2.72, 8.2, 0.5, "优化一：编码吞吐", size=19, bold=True, color=INK)
textbox(slide, 10.65, 3.3, 8.2, 1.6,
        "192 线程下吞吐反而下降；实测 64 线程时\n"
        "由每秒 7.2 篇升至 99.7 篇，约 13.8 倍。",
        size=14, color=INK, spacing=1.45)
bullets(slide, 0.95, 5.45, 18.2, [
    "优化二：向量索引按论文文件化，查询期只映射参与检索的几篇，索引内存与库规模解耦",
    "优化三：答案阶段关闭推理通道，避免推理写满 token 预算后正文为空；跨论文题耗时由 214 秒降至 0.8 秒",
    "护栏六层：证据绑定、引用真实性、提示注入隔离、结构化输出、无据拒答、空答案即失败",
], size=15, gap=0.95)

def record_hint(slide, x, y, w, hint):
    """What the clip on this page has to show — the recording script."""
    rect(slide, x, y, w, 1.55, RGBColor(0xF5, 0xF7, 0xFA),
         shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    textbox(slide, x + 0.3, y + 0.18, 1.6, 0.4, "录像要点", size=12, bold=True, color=RED)
    textbox(slide, x + 0.3, y + 0.6, w - 0.6, 0.85, hint, size=12, color=MUTED,
            spacing=1.35)


# --------------------------------------------------------- 功能页模板
FEATURES = [
    ("功能一：文献库", "上传即解析，版本自动归并",
     ["拖入 PDF，后台并行解析；断点续传与失败重试",
      "同篇论文的 v1 / v7 判为同一文献，给出保留建议与版本差异",
      "541 页书籍 149.4 秒解析完成，产出 1254 个章节节点与 131 万字符",
      "扫描件自动识别并逐页转写，识别结果参与后续检索"],
     "文献库：上传、解析进度、版本归并",
     "拖入一篇 arXiv 论文 → 看解析进度到 100% → 打开版本归并面板，"
     "展示同一篇的 v1 与 v7 判为重复、给出的保留建议"),
    ("功能二：阅读器", "重排、中英对照、引用可回溯",
     ["按解析结构重排：表格是真表格，公式用 KaTeX 排版",
      "中英对照逐句对齐，悬停中文即显示英文原句",
      "点击引用弹卡片，需要时再用按钮跳到参考文献条目",
      "标题树定位、图表裁剪图与原图切换"],
     "阅读器：重排视图、中英对照、引用卡片",
     "打开 BERT → 展示重排后的表格与公式 → 切到中英对照，悬停一句中文 → "
     "点正文的 [11] 弹卡片 → 按「跳到参考文献」看高亮"),
    ("功能三：文献问答", "先取证，再作答",
     ["意图路由 → 任务规划 → 工具调用 → 证据绑定 → 流式作答",
      "每条结论带可点击锚点：文献、页码、字符区间",
      "证据不足时明确拒答，不给猜测性结论",
      "思考过程面板逐条可见，自反馈补检索单独成条"],
     "问答：提问、思考过程、带引用的回答、拒答",
     "问一次跨论文题 → 展开思考过程，看意图、计划与工具调用 → "
     "答案里的引用点开可回原文 → 再问一个库外问题，展示拒答"),
    ("功能四：图谱综述", "看引用关系，找奠基工作",
     ["在引用图上算 PageRank、介数、k-core 与 HITS，联合判定基石",
      "共引与共参考两种相似关系，可切换边视图",
      "矛盾检测：同一指标取值不同判为数值冲突",
      "一键生成领域综述与后续研究机会"],
     "图谱：三视图切换、基石与衍生双视图、矛盾检测、综述生成",
     "切三种边视图 → 点一个基石节点看指标 → 打开矛盾检测，"
     "展示 Attention 两个版本的 BLEU 冲突 → 生成综述"),
    ("功能五：检索推荐", "追踪 arXiv，标注可复现性",
     ["关键词检索与分类追踪，结果一键入库并解析",
      "基于文献库的推荐，优先有公开代码的工作",
      "可复现性检查：仓库是否存在、是否有可运行入口",
      "用户反馈参与排序"],
     "检索：arXiv 检索、一键入库、可复现性标注",
     "搜一个关键词 → 结果列表 → 点一键入库并解析 → "
     "打开可复现性检查，展示仓库地址与是否可运行"),
    ("功能六：写作工作台", "框架、架构图与数据图表",
     ["由文献库生成论文框架，参考文献必须是库内真实存在的",
      "生成 TikZ、Mermaid、Graphviz 架构图",
      "CSV 数据生成学术图表，同时给出图注与关键结论",
      "幻觉引用进入已拦截列表，在界面单独展示"],
     "写作：框架生成、架构图、CSV 转图表、拦截的幻觉引用",
     "生成一次论文框架 → 生成一张 Mermaid 架构图 → "
     "上传一份 CSV 生成图表与图注 → 指出已拦截的幻觉引用"),
]

number = 8
for title, subtitle, points, video_label, hint in FEATURES:
    slide = blank_slide()
    page_title(slide, title, subtitle, number)
    bullets(slide, 0.95, 2.5, 7.3, points, size=14, gap=1.0)
    record_hint(slide, 0.95, 7.35, 7.3, hint)
    video_frame(slide, 8.6, 2.45, 10.5, 5.9, video_label)
    number += 1

# ------------------------------------------------------------------ 特色页
slide = blank_slide()
page_title(slide, "特色：证据链与引用卡片", "这两处是系统可信度的落点", number)
bullets(slide, 0.95, 2.55, 7.3, [
    "取证与生成分离：先生成证据集合，再只依据证据写正文",
    "引用真实性校验：参考文献与库内文献做标题模糊匹配与编号精确匹配",
    "空答案即失败：生成结果为空即判失败，触发自反馈并翻倍预算",
    "引用卡片：点标记看条目，按钮才跳转，页面不因误点而跳动",
], size=14, gap=1.15)
video_frame(slide, 8.6, 2.45, 10.5, 5.9, "特色：证据面板、拦截列表、引用卡片")

# ------------------------------------------------------------------ 实测数据
slide = blank_slide()
page_title(slide, "实测数据", "15 篇文献的冻结快照，三轮运行取均值", number + 1)
metrics = [
    ("忠实度", "0.8034", "三轮 0.7898 – 0.8168，标准差 0.0110"),
    ("引用可溯源率", "1.0000", "104 条支持的断言全部指向字符区间"),
    ("上下文精确率", "0.2791", "偏低的来源与改进方向见文档 §6.3"),
    ("库外对照题", "3/3", "全部诚实拒答，0 条编造"),
    ("检索命中率", "1.0000", "元数据感知检索后 top-5 由 0.9286 提升"),
    ("结构化解析", "149.4 s", "541 页书籍，1254 章节，131 万字符"),
]
for index, (label, value, note) in enumerate(metrics):
    left = 0.95 + (index % 3) * 6.35
    top = 2.55 + (index // 3) * 3.35
    rect(slide, left, top, 5.9, 2.95, RGBColor(0xFA, 0xF3, 0xF4),
         shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    textbox(slide, left + 0.4, top + 0.28, 5.0, 0.5, label, size=15, color=MUTED)
    textbox(slide, left + 0.4, top + 0.82, 5.1, 0.95, value, size=40, bold=True, color=RED)
    textbox(slide, left + 0.4, top + 1.95, 5.2, 0.8, note, size=12, color=INK)
textbox(slide, 0.95, 9.3, 18.2, 0.5,
        "测试：容器内 227 通过 / 9 跳过；有真实 PDF 与权重的主机上 236 通过，耗时 52 秒",
        size=13, color=MUTED)

# ------------------------------------------------------------------ 创新点
slide = blank_slide()
page_title(slide, "创新点", "技术创新与应用创新，各取三条", number + 2)
textbox(slide, 0.95, 2.5, 9.0, 0.5, "技术创新", size=20, bold=True, color=RED)
bullets(slide, 0.95, 3.1, 9.1, [
    "取证与生成分离的两阶段编排，以证据为契约",
    "按断言级的引用对齐修复，引文与指标核算同源",
    "图论联合判定的基石识别，规则化矛盾检测",
], size=14, gap=1.05)
textbox(slide, 10.35, 2.5, 9.0, 0.5, "应用创新", size=20, bold=True, color=RED)
bullets(slide, 10.35, 3.1, 8.9, [
    "重排阅读：表格可选中、公式可排版、中英逐句对照",
    "引用卡片与溯源锚点，读者可当场核对每句结论",
    "从文献库直接产出框架、架构图与数据图表",
], size=14, gap=1.05)
rect(slide, 0.95, 7.1, 18.25, 1.9, RGBColor(0xF5, 0xF7, 0xFA), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
textbox(slide, 1.3, 7.35, 17.6, 1.5,
        "明确不做：多智能体角色扮演层。Agent Laboratory 报告模型评审系统性高估自身产出"
        "（自评 6.1/10，人类 3.8/10），角色越多越难发现问题；本项目宁可少报，不报错。",
        size=14, color=INK, spacing=1.4)

# ------------------------------------------------------------------ 部署
slide = blank_slide()
page_title(slide, "部署、开源与复现", "一条命令起全栈；全部数字可复跑", number + 3)
bullets(slide, 0.95, 2.5, 9.2, [
    "单进程同源托管：一个 HTTPS 入口，接口与静态资源同源",
    "容器编排文件与镜像定义齐备，具备容器权限的机器可三服务复现",
    "模型接口可自定义：界面内添加任意 OpenAI 兼容端点并切换",
    "评测脚本、解析脚本与图表脚本随仓库提供",
], size=15, gap=1.15)
rect(slide, 10.3, 2.45, 8.9, 3.5, RGBColor(0xFA, 0xF3, 0xF4), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
textbox(slide, 10.65, 2.7, 8.2, 0.5, "在线演示", size=17, bold=True, color=RED)
textbox(slide, 10.65, 3.25, 8.2, 2.4,
        "https://finished-editing-shore-mrna.\ntrycloudflare.com\n\n"
        "仓库：github.com/akawrzzby/sciagent",
        size=14, color=INK, spacing=1.45)
rect(slide, 10.3, 6.15, 8.9, 2.9, RGBColor(0xF5, 0xF7, 0xFA), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
textbox(slide, 10.65, 6.4, 8.2, 0.5, "技术文档", size=17, bold=True, color=INK)
textbox(slide, 10.65, 6.95, 8.2, 1.9,
        "27 页，按赛题要求分七部分，另列创新点；\n"
        "每一项指标都标注出处，附可复跑脚本。",
        size=14, color=INK, spacing=1.45)

# ------------------------------------------------------------------ 结束页
slide = blank_slide()
rect(slide, 0, 0, 20.0, 11.25, RED)
textbox(slide, 0, 4.3, 20.0, 1.4, "谢谢各位老师", size=48, bold=True,
        color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
textbox(slide, 0, 5.9, 20.0, 0.7, "SciAgent：意图驱动的全流程科研助手", size=20,
        color=RGBColor(0xFF, 0xE8, 0xEA), align=PP_ALIGN.CENTER)

prs.save(str(OUT))
print("已生成:", OUT)
print("页数:", len(prs.slides))
