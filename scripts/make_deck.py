"""Build the defence deck by editing the XJTU template itself.

Typography: Chinese in 宋体, Latin and digits in Times New Roman, black ink,
bold where it matters.  Every content page opens with a short paragraph and only
then lists points; the experiment and the requirement checklist are three-line
tables.
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
FIGURES = pathlib.Path(
    r"E:\西交\研究生\比赛\AI大模型创新应用挑战赛\sciagent\docs\figures")

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
    textbox(slide, 0.60, 1.58, 16.4, 0.85, title, size=34, bold=True)
    return slide


def lead(slide, text, *, y=2.55, size=21, w=18.4, color=INK):
    """The opening paragraph of a page: what this page is about, in prose."""
    return textbox(slide, 0.70, y, w, 1.6, text, size=size, color=color, spacing=1.45)


def bars(slide, x, y, w, items, *, size=23, gap=1.25):
    for index, item in enumerate(items):
        top = y + index * gap
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(top + 0.18),
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
    textbox(slide, x, y + h / 2 + 0.3, w, 0.5, label, size=17, color=MUTED,
            align=PP_ALIGN.CENTER)
    textbox(slide, x, y + h - 0.62, w, 0.4, "插入 → 视频 → 此设备", size=13,
            color=MUTED, align=PP_ALIGN.CENTER)


def rule(slide, x, y, w, thickness=0.02, color=INK):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(thickness))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def three_line_table(slide, x, y, w, rows, col_widths, *, header_size=21,
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

# ------------------------------------------------ 2. 研究背景与五类困难
slide = page("研究背景与五类困难", 0)
lead(slide, "把大模型用在科研上，已经不新鲜了。找文献、筛摘要、改写、查术语，都有现成工具，也确实省时间。"
            "但真正跑完整条流程的研究者会发现，这些工具大多停在「读懂大意」这一步，再往下就走不动了。",
     size=20, y=2.50, w=18.6)
textbox(slide, 0.70, 3.55, 18.6, 5.4,
        "卡住的地方有五个。\n"
        "版面：PDF 是给人排的，双栏、跨栏图表、公式脚注混在一页里，机器直接抽出来是错序的。\n"
        "生成：模型会编参考文献、把不同论文的作者混在一起，甚至把一篇的结论安到另一篇上。\n"
        "阅读：阅读器只管显示，翻译丢了指代，文献管理软件又不能提问。\n"
        "判断：读完二十篇，还是说不清这个方向哪些空白没人填。\n"
        "写作：从零起笔，框架、架构图、参考文献，最后往往倒在参考文献上。",
        size=20, spacing=1.75)
textbox(slide, 0.70, 7.95, 18.6, 0.6,
        "这五件事其实是同一件事：结论指不回原文。",
        size=23, bold=True)
textbox(slide, 0.70, 8.95, 18.6, 1.5,
        "OpenScholar 在 Nature 上统计过一组数：带引用生成时，引用幻觉率在 78% 到 90%；"
        "PubMed Central 上被判定伪造的引用，两年涨了六倍；完全正确的引用只有 26.5%。",
        size=17, spacing=1.4)

# ------------------------------------------------ 3. 系统定位
slide = page("系统定位：先把可信做实", 0)
lead(slide, "要解决上面五个问题，关键不在换一个更强的模型，而在让结论可核查。"
            "可信来自两件事：说法有出处，出处能定位。围绕这一点，系统定了三条原则，"
            "并把它们折算成三个可度量的目标。",
     size=20, y=2.50)
textbox(slide, 0.70, 4.55, 18.4, 0.5, "三条原则", size=23, bold=True)
bars(slide, 0.70, 5.10, 18.4, [
    "证据优先：每一项结论都绑定原文的字符区间，指不回去就不说",
    "结构优先：标题、表格、公式、参考文献分别建模，不做无结构拼接",
    "可降级：稠密通道失效退化为 BM25，模型超时给出明确错误",
], size=21, gap=1.05)
textbox(slide, 0.70, 8.05, 18.4, 0.5, "三个目标", size=23, bold=True)
textbox(slide, 0.70, 8.62, 18.4, 1.8,
        "解析准确：标题层级、作者、表格与参考文献字段正确　·　"
        "回答可信：每条结论可点回原文　·　闭环完整：入库到写作全在系统内",
        size=20, spacing=1.5)

# ------------------------------------------------ 4. 系统架构
slide = page("系统架构：五层，每层只依赖下一层", 1)
arch = FIGURES / "fig-arch.png"
if arch.exists():
    slide.shapes.add_picture(str(arch), Inches(0.45), Inches(2.30), width=Inches(9.4))
textbox(slide, 10.35, 2.30, 9.3, 0.5, "各层职责", size=22, bold=True)
textbox(slide, 10.35, 2.85, 9.3, 5.3,
        "展现层　页面与事件流渲染，只调接口\n\n"
        "编排层　路由、规划、工具循环、作答与校验\n\n"
        "能力层　解析、检索、图谱、写作等纯函数\n\n"
        "存储层　论文、块、引用、会话与审计记录\n\n"
        "模型层　自托管大模型、向量与重排模型",
        size=17, spacing=1.45)
rule(slide, 0.70, 8.72, 18.4, 0.012, RGBColor(0xB8, 0xBE, 0xC8))
textbox(slide, 0.70, 8.86, 18.4, 1.2,
        "三处实测优化：编码吞吐 192 线程反而下降，64 线程时 7.2 → 99.7 篇/秒（13.8 倍）；"
        "答案阶段关闭推理通道，跨论文题 214 秒无答案降到 0.8 秒；向量索引按论文文件化，"
        "查询期只映射参与检索的几篇。",
        size=15.5, spacing=1.35)

# ------------------------------------------------ 5. Agent 角色架构
slide = page("Agent 角色架构：五个角色，一条回路", 1)
agent_fig = FIGURES / "fig-agent.png"
if agent_fig.exists():
    slide.shapes.add_picture(str(agent_fig), Inches(0.45), Inches(2.35), width=Inches(9.3))
textbox(slide, 10.20, 2.35, 9.4, 0.45, "角色的分工", size=21, bold=True)
textbox(slide, 10.20, 2.85, 9.4, 5.2,
        "路由　规则优先判意图，未命中交给模型\n\n"
        "规划　拆成子任务与工具序列\n\n"
        "执行　ReAct 循环取证，带步数上限\n\n"
        "作答　只按证据编号起草\n\n"
        "校验　护栏、引用对齐、自反馈补检索",
        size=16, spacing=1.45)
textbox(slide, 0.70, 8.85, 18.4, 1.0,
        "角色是职责边界，不是各自扮演人格：同一个模型承担五种职责，靠提示与工具权限区分。"
        "多角色互相评审会让模型集体高估自己（自评 6.1 / 人类 3.8），所以只设岗位，不设演员。",
        size=14, spacing=1.4)

# ------------------------------------------------ 6. AI 技术选型
slide = page("AI 技术方案：选型与理由", 1)
lead(slide, "选型遵循一条原则：能确定的事不交给模型。模型只负责语言理解与生成，"
            "解析、检索、图论这些有确定答案的部分，全部下沉到代码里。",
     size=19, y=2.55)
bars(slide, 0.70, 3.95, 18.4, [
    "大模型　Qwen3.8-27B 自托管：27B 在成本与能力之间平衡，结构化输出稳定，数据不出内网",
    "解析　PyMuPDF-Layout：DocLayNet F1 0.8356，对 Docling 的 0.8102，参数 1.8M 且无需 GPU",
    "检索　BM25 + bge-m3 + RRF：稀疏通道精确命中术语，稠密通道覆盖同义改写，RRF 融合只看排名",
    "重排　bge-reranker-v2-m3：交叉编码器逐对打分，把正确段落从第 2 位提到第 1 位",
    "扫描件　多模态 VLM：唯一能同时处理双栏顺序与数学公式的方案",
], size=19, gap=1.30)

# ------------------------------------------------ 7-14. 功能演示
FEATURES = [
    ("功能一：文献库",
     "文献库是整条流水线的入口。上传的 PDF 在这里被解析成结构化数据，"
     "后续的阅读、检索、问答与写作，都建立在它之上。",
     ["版面解析：标题、表格、公式、参考文献分别落库",
      "版本去重：标题与摘要相似度超阈值判为同一篇",
      "扫描件：逐页转写，识别结果参与检索",
      "索引：按论文文件化，检索时只加载相关几篇"],
     "文献库：上传、解析进度、版本归并"),
    ("功能二：阅读器 · 原文视图",
     "阅读器不把 PDF 当图片贴上来，而是按解析出的结构重新排版："
     "长文档可以连续读，表格能选中，公式能排版。",
     ["Block 模型：段落、表格、公式、图各自建模",
      "按版面几何重建阅读顺序，双栏与跨栏都还原",
      "公式用 KaTeX，编号右对齐；表格可选中",
      "图表取自原 PDF 裁剪图，可切回原图"],
     "阅读器：原文视图、表格与公式"),
    ("功能二：阅读器 · 中英对照",
     "中英对照解决的是「读得慢、不敢跳」：译文与英文原句逐句对齐，"
     "悬停即见原句，不必来回翻页。",
     ["句子级对齐：译文与英文原句一一配对",
      "译文按句缓存，命中即出，不重复调用",
      "悬停显示原句，浮层不挡引用点击",
      "引用识别两种写法：数字与作者-年份"],
     "阅读器：中英对照、悬停原句"),
    ("功能三：文献问答",
     "问答是整个系统的主线：先把问题拆开，再去库里取证，最后只依据证据作答，"
     "每句话都能点回原文。",
     ["阶段 A 只取证：ReAct 循环调用检索与读段",
      "证据编号成块，阶段 B 只许引用这些编号",
      "覆盖率不足触发自反馈，补检索后重写",
      "证据不足直接拒答，不给猜测性结论"],
     "问答：提问、思考过程、带引用的回答、拒答"),
    ("功能四：图谱综述",
     "图谱把文献之间的引用关系画出来，用图论指标判断哪几篇是这个方向的奠基工作，"
     "哪几篇是衍生。",
     ["四个指标联合判定：PageRank、介数、k-core、HITS",
      "共引与共参考两种相似边，可切换",
      "矛盾检测规则化：同指标不同值即冲突",
      "综述与后续研究只由库内文献生成"],
     "图谱：三视图、基石与衍生、矛盾检测、综述"),
    ("功能五：检索推荐",
     "检索面向库外的 arXiv：找到、比较、入库，入库后自动进入解析与索引流程，"
     "不需要二次操作。",
     ["arXiv 实时接口与分类追踪",
      "结果一键入库，入库即解析建索引",
      "可复现性检查：仓库是否存在、能否运行",
      "用户反馈参与排序"],
     "检索：arXiv 检索、一键入库、可复现性标注"),
    ("功能六：写作工作台",
     "写作工作台把「读」的结果变成「写」的起点：论文框架、架构图与数据图表，"
     "三件事都在库内文献的基础上完成。",
     ["框架的参考文献以库内文献为白名单",
      "校验不通过的直接拦截，单独列出",
      "架构图输出 Mermaid、TikZ、Graphviz 代码",
      "CSV 出图同时给图注与结论"],
     "写作：框架、架构图、CSV 转图表、拦截列表"),
    ("特色：证据链与引用卡片",
     "可信不是一句口号，它落在三个可检查的地方：证据、引用与拒答。"
     "这一页把这三件事放在一起看。",
     ["取证与生成分离，证据是两者唯一接口",
      "引用对齐到字符区间，摘除无据引用",
      "空答案判失败，翻倍预算重试",
      "引用卡片：点标记看条目，按钮才跳转"],
     "特色：证据面板、拦截列表、引用卡片"),
]

for title, intro, points, label in FEATURES:
    slide = page(title, 2)
    lead(slide, intro, size=19, y=2.50, w=8.8)
    bars(slide, 0.70, 4.50, 8.4, points, size=20, gap=1.30)
    video_frame(slide, 9.30, 2.35, 10.2, 5.75, label)

# ------------------------------------------------ 15. 实验
slide = page("实验：数据、指标与理由", 3)
lead(slide, "在一套冻结的十五篇文献库（461 页）上跑三次，每次独立问答。"
            "指标沿用学术问答评测的通行做法：忠实度与引用可溯源率参照 OpenScholar"
            "（Nature 650:857–863），检索用 top-5 命中率，拒答能力另设库外对照题。",
     size=19, y=2.45, w=18.6)
three_line_table(slide, 0.90, 3.95, 18.2, [
    ["指标", "第一次", "第二次", "第三次", "均值", "极差"],
    ["忠实度", "0.7898", "0.8168", "0.8035", "0.8034", "0.0270"],
    ["引用可溯源率", "1.0000", "1.0000", "1.0000", "1.0000", "0"],
    ["上下文精确率", "0.2874", "0.2632", "0.2867", "0.2791", "0.0242"],
], col_widths=[4.6, 2.8, 2.8, 2.8, 2.8, 2.4], header_size=21, body_size=20, row_height=0.80)
textbox(slide, 0.90, 7.45, 18.2, 0.5, "单次测量", size=21, bold=True)
textbox(slide, 0.90, 7.95, 18.2, 1.1,
        "库外对照题 3/3，题面不在库内，三次全部拒答、零编造；"
        "检索命中率 top-5 1.0000，元数据感知检索后由 0.9286 提升；"
        "541 页书籍结构化解析 149.4 秒，1254 章节、131 万字符。",
        size=17, spacing=1.5)
textbox(slide, 0.90, 9.35, 18.2, 1.3,
        "三次共产生 158 条断言，其中 104 条被判为由所引证据支持，自反馈共触发 28 轮。"
        "上下文精确率偏低，因为它按字符数计算、长段落会稀释，原因与两条改进路径写在文档 6.3 节。",
        size=15.5, color=INK, spacing=1.35)

# ------------------------------------------------ 16. 创新点
slide = page("创新点", 3)
lead(slide, "创新分两层：一是把「可信」做成可检查的机制，二是让阅读这件事真正变得可用。",
     size=21, y=2.50, w=18.4)
textbox(slide, 0.70, 3.55, 8.9, 0.55, "技术创新", size=25, bold=True)
bars(slide, 0.70, 4.20, 8.9, [
    "两阶段取证生成：证据是契约，不是提示词",
    "断言级引用对齐：每条结论统一到字符区间",
    "图论联合判基石：四指标投票与矛盾检测",
], size=23, gap=1.40)
textbox(slide, 10.30, 3.55, 8.6, 0.55, "应用创新", size=25, bold=True)
bars(slide, 10.30, 4.20, 8.6, [
    "重排阅读：表格可选中、公式可排版",
    "中英逐句对照：悬停即见英文原句",
    "引用卡片：先看条目，再决定跳转",
], size=23, gap=1.40)
textbox(slide, 0.70, 8.55, 18.4, 1.4,
        "明确不做：多智能体角色扮演层。多角色互评会让模型系统性高估自身产出，"
        "本项目宁可少报，不报错。",
        size=20, spacing=1.4)

# ------------------------------------------------ 17. 总结与展望
slide = page("总结与展望：赛题要求与完成情况", 3)
lead(slide, "对照赛题要求的六项能力，逐条说明做到什么程度。",
     size=20, y=2.50, w=18.4)
three_line_table(slide, 0.90, 3.55, 18.2, [
    ["赛题要求", "完成情况"],
    ["文献解析与结构化", "15 篇 / 461 页；标题、表格、公式、参考文献全部结构化"],
    ["检索与问答", "两阶段取证作答，引用可溯源率 1.0000"],
    ["阅读与翻译", "重排阅读 + 中英逐句对照，译文覆盖 99.4% 以上"],
    ["知识图谱与综述", "15 节点 / 14 条引用边，基石识别、矛盾检测、综述生成"],
    ["写作辅助", "论文框架、架构图、CSV 图表；幻觉引用拦截并展示"],
    ["部署与开源", "公开仓库、在线演示、27 页技术文档，测试 236 通过"],
], col_widths=[5.6, 12.6], header_size=21, body_size=19, row_height=0.80)
textbox(slide, 0.90, 9.30, 18.2, 1.5,
        "后续三件事：公式改为可检索的符号表达；接入更多文献来源，扩大可用语料；"
        "多用户并发与权限，支持团队共用。",
        size=18, spacing=1.4)

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
