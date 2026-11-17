"""Build the defence deck by editing the XJTU template itself.

The template owns the look: the red navigation ribbon with its four sections,
the active-pill, the logo, the 32pt title and the 28pt body.  This script keeps
those elements — it copies the ribbon from the template onto every page, moves
the pill to the section the page belongs to, and replaces only the text.
"""

from __future__ import annotations

import copy
import pathlib
import tempfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

TEMPLATE = pathlib.Path(r"E:\西交\研究生\面试ppt\党员工作站 - 副本.pptx")
OUT = pathlib.Path(r"E:\西交\研究生\面试ppt\SciAgent-答辩.pptx")
ARCH = pathlib.Path(
    r"E:\西交\研究生\比赛\AI大模型创新应用挑战赛\sciagent\docs\figures\fig-arch.png")

RED = RGBColor(0xC0, 0x1B, 0x2E)
INK = RGBColor(0x33, 0x33, 0x33)
MUTED = RGBColor(0x66, 0x66, 0x66)

#: the four sections of the ribbon, and where its pill sits when active
SECTIONS = ["项目背景", "系统设计", "功能演示", "验证总结"]
SECTION_EN = ["Background", "Design", "Features", "Validation"]
PILL_X = [0.42, 4.24, 7.68, 10.99]

prs = Presentation(str(TEMPLATE))
slides = list(prs.slides)

# ---- take what we need from the template, then empty the deck -------------
cover = slides[0]
ribbon_source = None
logo_blob = None
for shape in slides[1].shapes:
    if shape.name.startswith("组合 38"):
        ribbon_source = shape._element
    if shape.shape_type == 13 and logo_blob is None:
        # the university logo sits next to the ribbon, not inside it
        logo_blob = shape.image.blob

assert ribbon_source is not None, "模板里没有找到导航条"
ribbon_xml = copy.deepcopy(ribbon_source)
# the picture inside the ribbon cannot travel with the XML: its relationship
# belongs to the source slide, so it is dropped here and re-added per slide
for blip in ribbon_xml.iter():
    if blip.tag.endswith("}pic"):
        blip.getparent().remove(blip)

for index in range(len(prs.slides) - 1, 0, -1):
    slide_id = prs.slides._sldIdLst[index]
    prs.part.drop_rel(slide_id.rId)
    del prs.slides._sldIdLst[index]

logo_path = pathlib.Path(tempfile.gettempdir()) / "xjtu-logo.png"
logo_path.write_bytes(logo_blob)


def _walk(shapes):
    """Every shape of a group, however deep."""
    for shape in shapes:
        yield shape
        if shape.shape_type == 6:
            yield from _walk(shape.shapes)


def _set_two_lines(shape, first: str, second: str) -> None:
    """Replace a two-paragraph label without losing its formatting."""
    frame = shape.text_frame
    paragraphs = frame.paragraphs
    if paragraphs and paragraphs[0].runs:
        paragraphs[0].runs[0].text = first
        for run in paragraphs[0].runs[1:]:
            run.text = ""
    if len(paragraphs) > 1 and paragraphs[1].runs:
        paragraphs[1].runs[0].text = second
        for run in paragraphs[1].runs[1:]:
            run.text = ""


def _set_one_line(shape, text: str) -> None:
    frame = shape.text_frame
    if frame.paragraphs and frame.paragraphs[0].runs:
        frame.paragraphs[0].runs[0].text = text
        for run in frame.paragraphs[0].runs[1:]:
            run.text = ""


def add_ribbon(slide, section: int) -> None:
    """Put the template's ribbon on the page, with this section's pill."""
    element = copy.deepcopy(ribbon_xml)
    slide.shapes._spTree.append(element)
    group = slide.shapes[-1]

    tabs = []
    for shape in _walk(group.shapes):
        if not shape.has_text_frame:
            continue
        text = shape.text_frame.text.strip()
        paragraphs = [p for p in shape.text_frame.paragraphs if p.text.strip()]
        # the active pill is the only label with a Chinese line and an English
        # one; the tabs are single lines
        if shape.name.startswith("任意多边形") or len(paragraphs) >= 2:
            _set_two_lines(shape, SECTIONS[section], SECTION_EN[section])
            shape.left = Inches(PILL_X[section])
            for para in shape.text_frame.paragraphs:
                para.alignment = PP_ALIGN.CENTER
                for run in para.runs:
                    run.font.size = Pt(24)
            continue
        # the three inactive tabs, named after the sections they lead to
        if text and len(text) <= 5 and shape.left > Inches(4):
            tabs.append(shape)

    tabs.sort(key=lambda shape: shape.left)
    # the four slots of the ribbon; the pill takes one, the tabs take the rest
    slots = [1.15, 5.00, 8.44, 11.75]
    free = [index for index in range(4) if index != section]
    for shape, slot in zip(tabs[:3], free):
        shape.left = Inches(slots[slot])
        _set_one_line(shape, SECTIONS[slot])
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(27)

    slide.shapes.add_picture(str(logo_path), Inches(15.41), Inches(0.58),
                             Inches(3.50), Inches(0.94))


def textbox(slide, x, y, w, h, text, *, size=28, bold=False, color=INK,
            align=PP_ALIGN.LEFT, spacing=1.35):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.TOP
    for index, line in enumerate(text.split("\n")):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "微软雅黑"
    return box


def page(title, section, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_ribbon(slide, section)
    textbox(slide, 0.60, 1.68, 15.0, 0.75, title, size=32, bold=True)
    return slide


def bars(slide, x, y, w, items, *, size=22, gap=1.05):
    for index, item in enumerate(items):
        top = y + index * gap
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(top + 0.13),
                                     Inches(0.16), Inches(0.16))
        dot.fill.solid()
        dot.fill.fore_color.rgb = RED
        dot.line.fill.background()
        dot.shadow.inherit = False
        textbox(slide, x + 0.42, top, w - 0.42, gap, item, size=size)


def video_frame(slide, x, y, w, h, label):
    frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    frame.adjustments[0] = 0.03
    frame.fill.solid()
    frame.fill.fore_color.rgb = RGBColor(0xFA, 0xF7, 0xF7)
    frame.line.color.rgb = RED
    frame.line.width = Pt(1.5)
    from pptx.enum.dml import MSO_LINE_DASH_STYLE

    frame.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    frame.shadow.inherit = False
    play = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + w / 2 - 0.5),
                                  Inches(y + h / 2 - 0.85), Inches(1.0), Inches(1.0))
    play.fill.solid()
    play.fill.fore_color.rgb = RED
    play.line.fill.background()
    play.shadow.inherit = False
    textbox(slide, x + w / 2 - 0.5, y + h / 2 - 0.85, 1.0, 1.0, "▶", size=28, bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
    textbox(slide, x, y + h / 2 + 0.3, w, 0.5, label, size=16, color=MUTED,
            align=PP_ALIGN.CENTER)
    textbox(slide, x, y + h - 0.62, w, 0.4, "插入 → 视频 → 此设备", size=12,
            color=MUTED, align=PP_ALIGN.CENTER)


# ------------------------------------------------------------------ 封面
for shape in cover.shapes:
    if shape.has_text_frame and "党员工作站" in shape.text_frame.text:
        para = shape.text_frame.paragraphs[0]
        for run in para.runs[1:]:
            run._r.getparent().remove(run._r)
        para.runs[0].text = "SciAgent 科研小助手"
        para.runs[0].font.size = Pt(40)
    elif shape.has_text_frame and ("汇报人" in shape.text_frame.text
                                  or "时间" in shape.text_frame.text):
        para = shape.text_frame.paragraphs[0]
        for run in para.runs[1:]:
            run._r.getparent().remove(run._r)
        para.runs[0].text = "全国高校 AI 大模型创新应用挑战赛    2026 年 11 月"

# --------------------------------------------------------------- 项目背景
slide = page("科研文献的五类困难", 0, 2)
bars(slide, 0.75, 2.65, 18.3, [
    "PDF 并非结构化文本：双栏顺序、跨栏图表、公式与脚注，直接抽取即错序",
    "生成内容不可信：杜撰参考文献、混淆作者、把一篇的结论安到另一篇",
    "阅读工具割裂：阅读器不懂语义，翻译丢失指代，管理软件不能提问",
    "方向判断缺依据：读完二十篇，仍说不清哪些空白尚未被填补",
    "写作从零起笔：框架、架构图、数据表，参考文献最易出错",
], size=24, gap=1.35)

slide = page("有据可查的量级", 0, 3)
textbox(slide, 0.75, 2.7, 18.3, 6.8,
        "GPT-4o 在带引用生成任务中的引用幻觉率　78% – 90%\n\n"
        "PubMed Central 上被判定伪造的引用比例\n"
        "　　2023 年 1/2828 → 2025 年 1/458 → 2026 年前七周 1/277\n\n"
        "仅约 26.5% 的 AI 生成引用完全正确\n\n"
        "来源：OpenScholar，Nature 650:857–863",
        size=26, spacing=1.5)

# --------------------------------------------------------------- 系统设计
slide = page("系统定位：结论必须指回原文", 1, 4)
bars(slide, 0.75, 2.65, 18.3, [
    "证据优先：每一项结论都指回原文的字符区间，指不回去就不说",
    "结构优先：标题、表格、公式、参考文献分别建模，不做无结构拼接",
    "可降级：稠密通道失效退化为 BM25；模型超时给出明确错误",
    "可度量：解析准确、回答可信、闭环完整，三项都有实测",
], size=24, gap=1.4)

slide = page("系统架构：五层，每层只依赖下一层", 1, 5)
if ARCH.exists():
    slide.shapes.add_picture(str(ARCH), Inches(0.75), Inches(2.6), width=Inches(11.4))
textbox(slide, 12.5, 2.75, 6.9, 7.0,
        "展现层经 /api 与 SSE 访问编排层\n\n"
        "编排层调用能力层的纯函数\n\n"
        "能力层读写存储，再向下使用模型\n\n"
        "解析器不依赖 HTTP，检索器不依赖模型在线",
        size=18, spacing=1.5)

slide = page("AI 技术方案：选型的理由", 1, 6)
textbox(slide, 0.75, 2.6, 18.3, 7.4,
        "大模型　Qwen3.8-27B 自托管\n"
        "　　27B 在成本与能力之间取平衡；解析、检索、图论计算下沉到确定性代码\n\n"
        "解析　PyMuPDF-Layout\n"
        "　　DocLayNet F1 0.8356，对 Docling 0.8102；参数 1.8M 对 20M，且无需 GPU\n\n"
        "检索　BM25 + bge-m3 + RRF\n"
        "　　BM25 精确命中术语，稠密通道覆盖语义；RRF 只用排名，无需归一化\n\n"
        "重排　bge-reranker-v2-m3\n"
        "　　交叉编码器捕捉细粒度交互，把正确段落由第 2 位提到第 1 位",
        size=23, spacing=1.45)

slide = page("集成方式与优化策略", 1, 7)
textbox(slide, 0.75, 2.6, 18.3, 7.4,
        "集成：取证与生成分离\n"
        "　　阶段 A 只取证，产出带字符区间的证据集合；阶段 B 只依据证据生成\n"
        "　　两阶段以证据为唯一接口，界面可逐条审计\n\n"
        "优化一：编码吞吐\n"
        "　　192 线程反而下降；64 线程时由每秒 7.2 篇升至 99.7 篇，约 13.8 倍\n\n"
        "优化二：答案阶段关闭推理通道\n"
        "　　跨论文题由 214 秒无答案，降到 0.8 秒且正文完整\n\n"
        "优化三：向量索引按论文文件化，查询期只映射参与检索的几篇",
        size=23, spacing=1.45)

# --------------------------------------------------------------- 功能演示
FEATURES = [
    ("功能一：文献库", "上传即解析，版本自动归并",
     ["拖入 PDF，后台并行解析；断点续传与失败重试",
      "同篇论文的 v1 / v7 判为同一文献，给出保留建议与版本差异",
      "541 页书籍 149.4 秒解析完成，1254 章节，131 万字符",
      "扫描件自动识别并逐页转写，结果参与后续检索"],
     "文献库：上传、解析进度、版本归并"),
    ("功能二：阅读器", "重排、中英对照、引用可回溯",
     ["按解析结构重排：表格是真表格，公式用 KaTeX 排版",
      "中英对照逐句对齐，悬停中文即显示英文原句",
      "点击引用弹卡片，需要时再用按钮跳到参考文献",
      "标题树定位、图表原图切换"],
     "阅读器：重排视图、中英对照、引用卡片"),
    ("功能三：文献问答", "先取证，再作答",
     ["意图路由 → 任务规划 → 工具调用 → 证据绑定 → 流式作答",
      "每条结论带可点击锚点：文献、页码、字符区间",
      "证据不足时明确拒答，不给猜测性结论",
      "思考过程逐条可见，自反馈补检索单独成条"],
     "问答：提问、思考过程、带引用的回答、拒答"),
    ("功能四：图谱综述", "看引用关系，找奠基工作",
     ["页面 PageRank、介数、k-core 与 HITS 联合判定基石",
      "共引与共参考两种相似关系，可切换边视图",
      "矛盾检测：同一指标取值不同判为数值冲突",
      "一键生成领域综述与后续研究机会"],
     "图谱：三视图、基石与衍生、矛盾检测、综述生成"),
    ("功能五：检索推荐", "追踪 arXiv，标注可复现性",
     ["关键词检索与分类追踪，结果一键入库并解析",
      "基于文献库的推荐，优先有公开代码的工作",
      "可复现性检查：仓库是否存在、是否有可运行入口",
      "用户反馈参与排序"],
     "检索：arXiv 检索、一键入库、可复现性标注"),
    ("功能六：写作工作台", "框架、架构图与数据图表",
     ["由文献库生成论文框架，参考文献必须是库内真实存在的",
      "生成 TikZ、Mermaid、Graphviz 架构图",
      "CSV 数据生成学术图表，同时给出图注与关键结论",
      "幻觉引用进入已拦截列表，界面单独展示"],
     "写作：框架、架构图、CSV 转图表、拦截列表"),
    ("特色：证据链与引用卡片", "可信度落在这两处",
     ["取证与生成分离，先生成证据集合再写正文",
      "引用真实性校验：标题模糊匹配与编号精确匹配",
      "空答案即失败：触发自反馈并翻倍预算",
      "引用卡片：点标记看条目，按钮才跳转，页面不跳动"],
     "特色：证据面板、拦截列表、引用卡片"),
]

number = 8
for title, subtitle, points, label in FEATURES:
    slide = page(title, 2, number)
    textbox(slide, 0.62, 2.42, 8.6, 0.45, subtitle, size=17, color=MUTED)
    bars(slide, 0.75, 3.0, 8.2, points, size=20, gap=1.6)
    video_frame(slide, 9.3, 2.45, 10.2, 5.75, label)
    number += 1

# --------------------------------------------------------------- 验证总结
slide = page("实测数据", 3, number)
textbox(slide, 0.75, 2.6, 18.3, 7.6,
        "忠实度　0.8034　　三轮 0.7898 – 0.8168，标准差 0.0110\n"
        "引用可溯源率　1.0000　　104 条支持的断言全部指向字符区间\n"
        "上下文精确率　0.2791　　偏低的来源与改进方向见技术文档 §6.3\n\n"
        "库外对照题　3/3　　全部诚实拒答，0 条编造\n"
        "检索命中率　1.0000　　元数据感知检索后 top-5 由 0.9286 提升\n"
        "结构化解析　149.4 s　　541 页书籍，1254 章节，131 万字符\n\n"
        "测试：容器内 227 通过 / 9 跳过；有真实 PDF 与权重的主机上 236 通过，耗时 52 秒",
        size=23, spacing=1.45)
number += 1

slide = page("总结与展望", 3, number)
textbox(slide, 0.75, 2.6, 18.3, 7.6,
        "已完成\n"
        "　　六个工作台覆盖检索、精读、对比、综述与写作；论文的每个结论都能回原文\n"
        "　　15 篇文献的库上，结构审计 15/15 通过，译文覆盖 99.4% 以上\n\n"
        "创新点\n"
        "　　取证与生成分离的两阶段编排；按断言级的引用对齐修复\n"
        "　　图论联合判定的基石识别与规则化矛盾检测\n\n"
        "后续\n"
        "　　公式改为可检索的符号表达；跨境文献源接入；多用户并发与权限",
        size=23, spacing=1.45)

# ------------------------------------------------------------------ 谢谢
slide = prs.slides.add_slide(prs.slide_layouts[6])
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RED
bg.line.fill.background()
bg.shadow.inherit = False
textbox(slide, 0, 4.2, 20.0, 1.4, "谢谢各位老师", size=48, bold=True,
        color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
textbox(slide, 0, 5.85, 20.0, 0.7, "SciAgent 科研小助手", size=22,
        color=RGBColor(0xFF, 0xE0, 0xE4), align=PP_ALIGN.CENTER)

prs.save(str(OUT))
print("已生成:", OUT, "页数:", len(prs.slides))
