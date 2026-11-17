"""Build the defence deck by editing the XJTU template itself.

Typography follows the request: Chinese in 宋体, Latin and digits in Times New
Roman, black text, bold where it matters.  The measured numbers are laid out as
a three-line table, the innovations get their own page before the summary, and
the reader is shown twice — the original-text view and the bilingual view.
"""

from __future__ import annotations

import copy
import pathlib
import tempfile

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

TEMPLATE = pathlib.Path(r"E:\西交\研究生\面试ppt\党员工作站 - 副本.pptx")
OUT = pathlib.Path(r"E:\西交\研究生\面试ppt\SciAgent-答辩-new.pptx")
ARCH = pathlib.Path(
    r"E:\西交\研究生\比赛\AI大模型创新应用挑战赛\sciagent\docs\figures\fig-arch.png")

RED = RGBColor(0xC0, 0x1B, 0x2E)
INK = RGBColor(0x00, 0x00, 0x00)
MUTED = RGBColor(0x59, 0x59, 0x59)

SONG = "宋体"
ROMAN = "Times New Roman"

SECTIONS = ["项目背景", "系统设计", "功能演示", "验证总结"]
SECTION_EN = ["Background", "Design", "Features", "Validation"]
PILL_X = [0.42, 4.24, 7.68, 10.99]

prs = Presentation(str(TEMPLATE))
slides = list(prs.slides)

cover = slides[0]
ribbon_xml = None
logo_blob = None
for shape in slides[1].shapes:
    if shape.name.startswith("组合 38"):
        ribbon_xml = copy.deepcopy(shape._element)
    if shape.shape_type == 13 and logo_blob is None:
        logo_blob = shape.image.blob

assert ribbon_xml is not None, "模板里没有找到导航条"
for pic in list(ribbon_xml.iter(qn("p:pic"))):
    pic.getparent().remove(pic)

for index in range(len(prs.slides) - 1, 0, -1):
    slide_id = prs.slides._sldIdLst[index]
    prs.part.drop_rel(slide_id.rId)
    del prs.slides._sldIdLst[index]

logo_path = pathlib.Path(tempfile.gettempdir()) / "xjtu-logo.png"
logo_path.write_bytes(logo_blob)


# ------------------------------------------------------------------ typography
def style(run, size: float, *, bold=False, color=INK, latin=ROMAN, ea=SONG) -> None:
    """One run: 宋体 for the Chinese, Times New Roman for the rest."""
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = latin
    rpr = run._r.get_or_add_rPr()
    for tag, face in (("a:ea", ea), ("a:cs", latin)):
        element = rpr.find(qn(tag))
        if element is None:
            element = etree.SubElement(rpr, qn(tag))
        element.set("typeface", face)


def textbox(slide, x, y, w, h, text, *, size=23, bold=False, color=INK,
            align=PP_ALIGN.LEFT, spacing=1.35, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    for index, line in enumerate(text.split("\n")):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        run = para.add_run()
        run.text = line
        style(run, size, bold=bold, color=color)
    return box


def _walk(shapes):
    for shape in shapes:
        yield shape
        if shape.shape_type == 6:
            yield from _walk(shape.shapes)


def _set_two_lines(shape, first: str, second: str) -> None:
    paragraphs = shape.text_frame.paragraphs
    if paragraphs and paragraphs[0].runs:
        paragraphs[0].runs[0].text = first
        for run in paragraphs[0].runs[1:]:
            run.text = ""
    if len(paragraphs) > 1 and paragraphs[1].runs:
        paragraphs[1].runs[0].text = second
        for run in paragraphs[1].runs[1:]:
            run.text = ""


def _set_one_line(shape, text: str) -> None:
    if shape.text_frame.paragraphs and shape.text_frame.paragraphs[0].runs:
        shape.text_frame.paragraphs[0].runs[0].text = text
        for run in shape.text_frame.paragraphs[0].runs[1:]:
            run.text = ""


def add_ribbon(slide, section: int) -> None:
    element = copy.deepcopy(ribbon_xml)
    slide.shapes._spTree.append(element)
    group = slide.shapes[-1]

    tabs = []
    for shape in _walk(group.shapes):
        if not shape.has_text_frame:
            continue
        text = shape.text_frame.text.strip()
        paragraphs = [p for p in shape.text_frame.paragraphs if p.text.strip()]
        if shape.name.startswith("任意多边形") or len(paragraphs) >= 2:
            _set_two_lines(shape, SECTIONS[section], SECTION_EN[section])
            shape.left = Inches(PILL_X[section])
            for para in shape.text_frame.paragraphs:
                para.alignment = PP_ALIGN.CENTER
                for run in para.runs:
                    style(run, 24, bold=True)
            continue
        if text and len(text) <= 5 and shape.left > Inches(4):
            tabs.append(shape)

    tabs.sort(key=lambda shape: shape.left)
    slots = [1.15, 5.00, 8.44, 11.75]
    free = [index for index in range(4) if index != section]
    for shape, slot in zip(tabs[:3], free):
        shape.left = Inches(slots[slot])
        _set_one_line(shape, SECTIONS[slot])
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                style(run, 27, bold=True)

    slide.shapes.add_picture(str(logo_path), Inches(15.41), Inches(0.58),
                             Inches(3.50), Inches(0.94))


def page(title, section):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_ribbon(slide, section)
    textbox(slide, 0.60, 1.66, 15.5, 0.8, title, size=32, bold=True)
    return slide


def bars(slide, x, y, w, items, *, size=23, gap=1.35):
    for index, item in enumerate(items):
        top = y + index * gap
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(top + 0.16),
                                     Inches(0.15), Inches(0.15))
        dot.fill.solid()
        dot.fill.fore_color.rgb = RED
        dot.line.fill.background()
        dot.shadow.inherit = False
        textbox(slide, x + 0.40, top, w - 0.40, gap, item, size=size)


def video_frame(slide, x, y, w, h, label):
    frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    frame.adjustments[0] = 0.03
    frame.fill.solid()
    frame.fill.fore_color.rgb = RGBColor(0xFC, 0xFA, 0xFA)
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
    textbox(slide, x + w / 2 - 0.5, y + h / 2 - 0.85, 1.0, 1.0, "▶", size=28,
            bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
    textbox(slide, x, y + h / 2 + 0.3, w, 0.5, label, size=16, color=MUTED,
            align=PP_ALIGN.CENTER)
    textbox(slide, x, y + h - 0.62, w, 0.4, "插入 → 视频 → 此设备", size=12,
            color=MUTED, align=PP_ALIGN.CENTER)


def rule(slide, x, y, w, thickness=0.02, color=INK):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(thickness))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def three_line_table(slide, x, y, w, rows, col_widths, *, header_size=20,
                     body_size=19, row_height=0.72):
    """A 三线表: a top rule, one under the header, one at the bottom."""
    total = sum(col_widths)
    widths = [w * width / total for width in col_widths]
    height = row_height * len(rows)
    table = slide.shapes.add_table(len(rows), len(col_widths), Inches(x), Inches(y),
                                   Inches(w), Inches(height)).table
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
    for row_index, values in enumerate(rows):
        table.rows[row_index].height = Inches(row_height)
        for col_index, value in enumerate(values):
            cell = table.cell(row_index, col_index)
            cell.margin_left = Inches(0.12)
            cell.margin_right = Inches(0.12)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.background()
            frame = cell.text_frame
            frame.word_wrap = True
            para = frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if col_index == 0 else PP_ALIGN.CENTER
            run = para.add_run()
            run.text = str(value)
            style(run, header_size if row_index == 0 else body_size,
                  bold=(row_index == 0))
    rule(slide, x, y, w, 0.03)
    rule(slide, x, y + row_height, w, 0.015)
    rule(slide, x, y + height, w, 0.03)
    return table


# ------------------------------------------------------------------ 封面
for shape in cover.shapes:
    if shape.has_text_frame and "党员工作站" in shape.text_frame.text:
        para = shape.text_frame.paragraphs[0]
        for run in para.runs[1:]:
            run._r.getparent().remove(run._r)
        para.runs[0].text = "SciAgent 科研小助手"
        style(para.runs[0], 40, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    elif shape.has_text_frame and ("汇报人" in shape.text_frame.text
                                  or "时间" in shape.text_frame.text):
        frame = shape.text_frame
        first = frame.paragraphs[0]
        for extra in list(frame.paragraphs)[1:]:
            extra._p.getparent().remove(extra._p)
        for run in list(first.runs):
            run._r.getparent().remove(run._r)
        first.alignment = PP_ALIGN.CENTER
        run = first.add_run()
        run.text = "答辩人：魏任泽　　指导老师：刘帅"
        style(run, 18, color=RGBColor(0xFF, 0xFF, 0xFF))
        second = frame.add_paragraph()
        second.alignment = PP_ALIGN.CENTER
        run2 = second.add_run()
        run2.text = "全国高校 AI 大模型创新应用挑战赛　　2026 年 11 月"
        style(run2, 18, color=RGBColor(0xFF, 0xFF, 0xFF))
        shape.top = Inches(9.3)
        shape.height = Inches(1.0)

# --------------------------------------------------------------- 项目背景
slide = page("科研文献的五类困难", 0)
bars(slide, 0.70, 2.55, 18.4, [
    "PDF 并非结构化文本：双栏顺序、跨栏图表、公式与脚注，直接抽取即错序",
    "生成内容不可信：杜撰参考文献、混淆作者、把一篇的结论安到另一篇",
    "阅读工具割裂：阅读器不懂语义，翻译丢失指代，管理软件不能提问",
    "方向判断缺依据：读完二十篇，仍说不清哪些空白尚未被填补",
    "写作从零起笔：框架、架构图、数据表，参考文献最易出错",
], size=23, gap=1.32)

slide = page("有据可查的量级", 0)
three_line_table(slide, 1.0, 2.55, 18.0, [
    ["指标", "数值", "来源"],
    ["GPT-4o 带引用生成的引用幻觉率", "78% – 90%", "OpenScholar，Nature 650:857–863"],
    ["PMC 上被判定伪造的引用比例（2023 → 2025）", "1/2828 → 1/458", "Barrot 等统计"],
    ["2026 年前七周的同一比例", "1/277", "同上"],
    ["AI 生成引用完全正确的比例", "约 26.5%", "同上"],
], col_widths=[8.4, 3.4, 6.2], header_size=21, body_size=19, row_height=0.82)

# --------------------------------------------------------------- 系统设计
slide = page("系统定位：结论必须指回原文", 1)
bars(slide, 0.70, 2.55, 18.4, [
    "证据优先：每一项结论都指回原文的字符区间，指不回去就不说",
    "结构优先：标题、表格、公式、参考文献分别建模，不做无结构拼接",
    "可降级：稠密通道失效退化为 BM25；模型超时给出明确错误",
    "可度量：解析准确、回答可信、闭环完整，三项都有实测",
], size=23, gap=1.5)

slide = page("系统架构：五层，每层只依赖下一层", 1)
if ARCH.exists():
    slide.shapes.add_picture(str(ARCH), Inches(0.60), Inches(2.55), width=Inches(11.3))
textbox(slide, 12.3, 2.7, 7.1, 7.0,
        "展现层经 /api 与 SSE 访问编排层\n\n"
        "编排层调用能力层的纯函数\n\n"
        "能力层读写存储，再向下使用模型\n\n"
        "解析器不依赖 HTTP，检索器不依赖模型在线",
        size=18, spacing=1.5)

slide = page("AI 技术方案：选型的理由", 1)
three_line_table(slide, 0.85, 2.5, 18.3, [
    ["环节", "选型", "依据"],
    ["大模型", "Qwen3.8-27B 自托管", "27B 在成本与能力间取平衡；难的部分下沉到确定性代码"],
    ["解析", "PyMuPDF-Layout", "DocLayNet F1 0.8356，Docling 0.8102；参数 1.8M 对 20M"],
    ["检索", "BM25 + bge-m3 + RRF", "精确术语与语义覆盖兼得；RRF 只用排名，无需调参"],
    ["重排", "bge-reranker-v2-m3", "交叉编码器把正确段落由第 2 位提到第 1 位"],
    ["扫描件", "多模态 VLM", "只有它能同时处理双栏顺序与数学公式"],
], col_widths=[2.4, 5.2, 10.7], header_size=20, body_size=18, row_height=0.98)

slide = page("集成方式与优化策略", 1)
textbox(slide, 0.75, 2.5, 18.4, 7.6,
        "集成：取证与生成分离\n"
        "　　阶段 A 只取证，产出带字符区间的证据集合；阶段 B 只依据证据生成\n"
        "　　两阶段以证据为唯一接口，界面可逐条审计\n\n"
        "优化一：编码吞吐　　192 线程反而下降；64 线程时 7.2 → 99.7 篇/秒，约 13.8 倍\n"
        "优化二：答案阶段关闭推理　　跨论文题由 214 秒无答案降到 0.8 秒且正文完整\n"
        "优化三：向量索引按论文文件化，查询期只映射参与检索的几篇",
        size=22, spacing=1.5)

# --------------------------------------------------------------- 功能一
slide = page("功能一：文献库", 2)
textbox(slide, 0.62, 2.35, 8.6, 0.45, "上传即解析，版本自动归并", size=17, color=MUTED)
bars(slide, 0.70, 2.95, 8.2, [
    "拖入 PDF，后台并行解析；断点续传与失败重试",
    "同篇论文的 v1 / v7 判为同一文献，给出保留建议",
    "541 页书籍 149.4 秒解析完成，1254 章节",
    "扫描件自动识别并逐页转写，结果参与检索",
], size=20, gap=1.6)
video_frame(slide, 9.2, 2.45, 10.3, 5.8, "文献库：上传、解析进度、版本归并")

# --------------------------------------------------------------- 功能二（两页）
slide = page("功能二：阅读器 · 原文视图", 2)
textbox(slide, 0.62, 2.35, 8.6, 0.45, "保留版式，但把结构还原出来", size=17, color=MUTED)
bars(slide, 0.70, 2.95, 8.2, [
    "按解析结构重排：表格是真表格，可选中、可复制",
    "公式用 KaTeX 排版，字号跟随正文，编号右对齐",
    "标题树定位、图表原图切换、段落两端对齐",
    "不再是页面截图，长文档也能连续阅读",
], size=20, gap=1.6)
video_frame(slide, 9.2, 2.45, 10.3, 5.8, "阅读器：原文视图、表格与公式")

slide = page("功能二：阅读器 · 中英对照", 2)
textbox(slide, 0.62, 2.35, 8.6, 0.45, "逐句对齐，悬停看原句", size=17, color=MUTED)
bars(slide, 0.70, 2.95, 8.2, [
    "中英逐句对齐，悬停中文即显示对应的英文原句",
    "点击引用弹卡片，需要时再用按钮跳到参考文献",
    "译文按句缓存，打开即读，不必等待",
    "未翻译的句子标为淡色，不静默留白",
], size=20, gap=1.6)
mock = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.2), Inches(2.45),
                              Inches(10.3), Inches(5.8))
mock.fill.solid()
mock.fill.fore_color.rgb = RGBColor(0xFF, 0xFB, 0xF6)
mock.line.color.rgb = RGBColor(0xE2, 0xD8, 0xC8)
mock.shadow.inherit = False
textbox(slide, 9.6, 2.75, 9.5, 0.4, "阅读器 · 中英对照", size=13, color=MUTED)
textbox(slide, 9.6, 3.25, 9.5, 1.4,
        "BERT 在概念上简单，但实验上很有说服力。它在十一项自然语言处理任务上取得了新的最优结果。",
        size=19, spacing=1.5)
textbox(slide, 9.6, 4.55, 9.5, 1.2,
        "BERT is conceptually simple and empirically powerful. It obtains new "
        "state-of-the-art results on eleven natural language processing tasks.",
        size=13, color=MUTED, spacing=1.4)
rule(slide, 9.6, 5.95, 9.5, 0.01, RGBColor(0xD8, 0xD8, 0xD8))
textbox(slide, 9.6, 6.15, 9.5, 1.6,
        "悬停中文，浮层显示英文原句；\n"
        "引用编号可点，弹卡片后再跳到参考文献条目。",
        size=15, color=MUTED, spacing=1.5)

# --------------------------------------------------------------- 功能三到六
FEATURES = [
    ("功能三：文献问答", "先取证，再作答",
     ["意图路由 → 任务规划 → 工具调用 → 证据绑定 → 流式作答",
      "每条结论带可点击锚点：文献、页码、字符区间",
      "证据不足时明确拒答，不给猜测性结论",
      "思考过程逐条可见，自反馈补检索单独成条"],
     "问答：提问、思考过程、带引用的回答、拒答"),
    ("功能四：图谱综述", "看引用关系，找奠基工作",
     ["PageRank、介数、k-core 与 HITS 联合判定基石",
      "共引与共参考两种相似关系，可切换边视图",
      "矛盾检测：同一指标取值不同判为数值冲突",
      "一键生成领域综述与后续研究机会"],
     "图谱：三视图、基石与衍生、矛盾检测、综述"),
    ("功能五：检索推荐", "追踪 arXiv，标注可复现性",
     ["关键词检索与分类追踪，结果一键入库并解析",
      "基于文献库的推荐，优先有公开代码的工作",
      "可复现性检查：仓库是否存在、是否有可运行入口",
      "用户反馈参与排序"],
     "检索：arXiv 检索、一键入库、可复现性标注"),
    ("功能六：写作工作台", "框架、架构图与数据图表",
     ["由文献库生成论文框架，参考文献必须是库内真实存在的",
      "生成 TikZ、Mermaid、Graphviz 架构图",
      "CSV 数据生成学术图表，同时给出图注与结论",
      "幻觉引用进入已拦截列表，界面单独展示"],
     "写作：框架、架构图、CSV 转图表、拦截列表"),
]
for title, subtitle, points, label in FEATURES:
    slide = page(title, 2)
    textbox(slide, 0.62, 2.35, 8.6, 0.45, subtitle, size=17, color=MUTED)
    bars(slide, 0.70, 2.95, 8.2, points, size=20, gap=1.6)
    video_frame(slide, 9.2, 2.45, 10.3, 5.8, label)

slide = page("特色：证据链与引用卡片", 2)
textbox(slide, 0.62, 2.35, 8.6, 0.45, "可信度落在这两处", size=17, color=MUTED)
bars(slide, 0.70, 2.95, 8.2, [
    "取证与生成分离，先生成证据集合再写正文",
    "引用真实性校验：标题模糊匹配与编号精确匹配",
    "空答案即失败：触发自反馈并翻倍预算",
    "引用卡片：点标记看条目，按钮才跳转，页面不跳动",
], size=20, gap=1.6)
video_frame(slide, 9.2, 2.45, 10.3, 5.8, "特色：证据面板、拦截列表、引用卡片")

# --------------------------------------------------------------- 实测数据
slide = page("实测数据", 3)
three_line_table(slide, 0.95, 2.4, 18.1, [
    ["指标", "实测值", "说明"],
    ["忠实度", "0.8034", "三轮 0.7898 – 0.8168，标准差 0.0110"],
    ["引用可溯源率", "1.0000", "104 条支持的断言全部指向字符区间"],
    ["上下文精确率", "0.2791", "偏低的原因与改进方向见技术文档 §6.3"],
    ["库外对照题", "3/3", "全部诚实拒答，0 条编造"],
    ["检索命中率 top-5", "1.0000", "元数据感知检索后由 0.9286 提升"],
    ["结构化解析", "149.4 秒", "541 页书籍，1254 章节，131 万字符"],
], col_widths=[6.4, 3.4, 8.3], header_size=20, body_size=19, row_height=0.8)
textbox(slide, 0.95, 8.55, 18.1, 0.6,
        "测试：容器内 227 通过 / 9 跳过；有真实 PDF 与权重的主机上 236 通过，耗时 52 秒",
        size=15, color=MUTED)

# --------------------------------------------------------------- 创新点
slide = page("创新点", 3)
textbox(slide, 0.75, 2.5, 8.9, 0.5, "技术创新", size=24, bold=True)
bars(slide, 0.75, 3.15, 8.9, [
    "取证与生成分离的两阶段编排，以证据为契约",
    "按断言级的引用对齐修复，引文与指标同源",
    "图论联合判定基石，规则化矛盾检测",
], size=21, gap=1.35)
textbox(slide, 10.4, 2.5, 8.6, 0.5, "应用创新", size=24, bold=True)
bars(slide, 10.4, 3.15, 8.6, [
    "重排阅读：表格可选中、公式可排版",
    "中英逐句对照，悬停即见原句",
    "引用卡片与锚点，读者可当场核对",
], size=21, gap=1.35)
textbox(slide, 0.75, 7.5, 18.3, 1.8,
        "明确不做：多智能体角色扮演层。Agent Laboratory 报告模型评审系统性高估自身产出"
        "（自评 6.1/10，人类 3.8/10），角色越多越难发现问题；本项目宁可少报，不报错。",
        size=18, spacing=1.5)

# --------------------------------------------------------------- 总结与展望
slide = page("总结与展望", 3)
textbox(slide, 0.75, 2.5, 18.3, 7.6,
        "已完成\n"
        "　　六个工作台覆盖检索、精读、对比、综述与写作；论文的每个结论都能回原文\n"
        "　　15 篇文献的库上，结构审计 15/15 通过，译文覆盖 99.4% 以上\n\n"
        "后续\n"
        "　　公式改为可检索的符号表达\n"
        "　　跨境文献源接入，扩大可用语料\n"
        "　　多用户并发与权限，支持团队共用",
        size=22, spacing=1.55)

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
