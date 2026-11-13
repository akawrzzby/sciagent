# 相关工作调研与差距清单

> **文档定位**：SciAgent 的竞品与相关工作调研。目的是回答三个问题——**别人怎么做？做得好的地方是什么？我们差在哪？怎么借鉴而不只是抄？**
>
> **调研日期**：2026-11-12
> **调研方法**：公开网络检索 + 论文原文/官方文档/项目主页交叉核对。凡是只拿到二手报道、未核到原文的结论，本文都**显式标注**。
> **对本项目的作用**：本文件是技术文档 §2.3「竞品与现有方案对比」和 §8「创新点说明」的事实依据，也是后续改进项的唯一来源。

---

## 0. 摘要：调研得出的五个结论

1. **我们的两阶段架构方向是对的，但缺一个"回头"。** OpenScholar 和 PaperQA2 都在生成之后加了一个自反馈循环（发现不足 → 补充检索 → 重写）。我们是一次性取证、一次性生成，**在生成后没有再回到检索的机会**。这是与前沿系统最大的结构性差距，而且改动最划算。
2. **我们把"引用含金量"做成了二分类，业界主流是三分类 + 引用原句。** Scite 的 Smart Citations 把每条引用标为 supporting / contrasting / mentioning，并展示引用句原文。这是小改动、大价值。
3. **我们完全没有量化 RAG 效果。** 92 个单元测试证明的是"代码没坏"，不是"答案准"。RAGAS / ARES / ScholarQABench 提供了现成的指标定义，我们完全可以在自己的 13 篇库上跑出数字。
4. **我们的引用图谱只建了直接引用边，而 Connected Papers 用的是共引 + 共参考文献。** 后者呈现的是"研究脉络"，前者只是"引用链"。加共引边不需要新数据，只需要新算法。
5. **学术前沿普遍低估了自己的能力边界，Agent Laboratory 论文自己披露了一个重要事实**：LLM 自动评审给 AI 生成的论文打 6.1/10，人类只给 3.8/10——**自动评估系统性高估**。这条对我们既是警告也是机会：我们的文档如果诚实区分"测试通过"与"效果达标"，可信度反而高于那些只报自动指标的同类作品。

---

## 1. 学术系统：谁在做什么

### 1.1 PaperQA2 — agentic RAG 的参考设计

**出处**：Lála et al., *Language agents achieve superhuman synthesis of scientific knowledge*, arXiv:2409.13740（FutureHouse + University of Rochester）

**核心做法**（与我们的对应关系）：

| 他们的做法 | 我们的做法 | 判断 |
|---|---|---|
| RAG 流水线拆成 Agent 可反复调用的工具，证据不足时换关键词重检 | 取证阶段一次性规划 DAG，工具循环由 Agent 决定 | 接近；我们缺"换关键词重检" |
| map-reduce 先对多源证据做摘要，给模型当 scratchpad | 我们直接把 chunk 喂给生成 | **可借鉴**：多证据汇总步骤 |
| 元数据感知检索（把标题/作者折进 embedding 输入） | 纯正文 chunk 编码 | **可借鉴**：改动极小 |
| 引用精确到**页** | 溯源精确到 `paper_id/page_no/bbox/char_range` | 我们更细 |
| **矛盾检测（ContraCrow）**：每篇论文检出 2.34 ± 1.99 个矛盾，70% 经人类专家验证 | 无 | **我们完全没有** |
| LitQA2 基准驱动开发 | 只有单元测试 | 缺 |

**关键数字**：论文称 PaperQA2 在三类真实文献检索任务上匹敌或超过领域专家；单次查询成本 $1–3。

**可借鉴**：矛盾检测（我们的"多文推理"模块天然可以承载）、元数据感知检索。

### 1.2 OpenScholar — 引用准确性与自反馈循环

**出处**：Asai et al., *Synthesizing scientific literature with retrieval-augmented language models*, **Nature 650, 857–863 (2026)**, DOI [10.1038/s41586-025-10072-4](https://doi.org/10.1038/s41586-025-10072-4)；预印本 arXiv:2411.14199（AI2 + University of Washington）

**核心做法**：

- **数据存储**：4500 万篇开放获取论文，约 2.36 亿 passage embedding（来源 Semantic Scholar）
- **自反馈推理循环**：初稿 → 模型自评（指出覆盖不足 / 结构不佳 / 证据薄弱）→ 触发**补充检索** → 重写并调整引用
- **训练专用 retriever + reranker**（不是通用 bge 系列直接用）
- **最终校验**：需要证据支撑的陈述必须至少有一条检索来源支持

**关键数字**（论文原文摘要）：

| 指标 | 结果 |
|---|---|
| GPT-4o 引用幻觉率 | **78–90%** |
| OpenScholar 引用准确率 | 与人类专家相当 |
| OS-8B 正确性 vs GPT-4o | **+6.1%** |
| OS-8B 正确性 vs PaperQA2 | **+5.5%** |
| OS-GPT-4o 对 GPT-4o 的正确性提升 | **+12%** |
| 专家偏好率（OS-8B / OS-GPT-4o / GPT-4o） | 51% / 70% / 32% |
| ScholarQABench 规模 | 2,967 条专家撰写 query + 208 篇长答案 |

**评估协议值得单独学**：ScholarQABench 把评估拆成两个层次——**引用正确性等自动指标** + **覆盖范围、连贯性、写作质量等人类专家评分**。我们目前只有前者意义上的"测试"，没有后者。

**可借鉴**：自反馈循环（最高优先级）、证据支撑的最终校验（我们已有护栏，可加"陈述→证据"逐条绑定）。

### 1.3 STORM — 多视角提问驱动的长文生成

**出处**：Shao et al., *Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models*, arXiv:2402.14207（Stanford OVAL）

**核心做法**（四阶段）：

1. **视角发现（perspective-driven questioning）**：分析相似主题的维基页面，发现不同视角——资料中提到的五种人格是 Domain Expert / Practitioner / Skeptic / Newcomer / Adjacent Expert
2. **多视角提问**：每个视角独立生成 5–10 个问题
3. **模拟对话**：writer 与 topic expert 对话，答案基于检索到的来源
4. **先大纲后正文**：层次化 outline → 按节展开

**效果**：相比"检索结果直接生成"的基线，文章**结构 +25%、覆盖 +10%**（有经验的维基编辑者评估）。

**局限（论文自述）**：来源偏置会被带进文章；模型会把原本无关的事实强行关联起来。

**可借鉴**：我们的自动综述目前是单视角一次生成。借鉴 STORM 的**视角发现**——而且我们的原料比 STORM 更好：库内每篇论文的 Future Work / Limitations 段天然就是不同视角的素材。

### 1.4 Agent Laboratory — 多智能体科研流水线

**出处**：Schmidgall et al., *Agent Laboratory: Using LLM Agents as Research Assistants*, arXiv:2501.04227（AMD + UPenn, EMNLP 2025）

**核心做法**：角色层级化多智能体——PhD Student（文献综述）→ Postdoc（规划）→ ML/Software Engineer（实验）→ Professor（评审打分）；planning-execution-reflection 循环；mle-solver 迭代修复代码，paper-solver 分节写作；支持 human-in-the-loop 的 "co-pilot" 模式。

**关键数字**：

| 项 | 结果 |
|---|---|
| 成本 | GPT-4o 约 $2.33 / o1-preview 约 $13，较 "The AI Scientist" **降 84%** |
| 单轮研究周期 | 19–103 分钟 |
| MLE-Bench | 4 枚奖牌（2 金 1 银 1 铜），超过 AIDE / OpenHands |
| **LLM 评审 vs 人类评审** | LLM 打 **6.1/10**，人类打 **3.8/10** —— 自动评审系统性高估 |

**对我们的启示**：Agent Laboratory 的"角色层级"和"反思循环"印证了我们自研编排器的方向（不用 LangChain 是对的）；而它的 **optimism gap** 提醒我们：不要把 LLM 自评当效果证据。

### 1.5 GAPMAP — 显式 gap 与隐式 gap 的分离

**出处**：*GAPMAP: Mapping Scientific Knowledge Gaps in Biomedical Literature Using Large Language Models*
> ⚠️ **来源标注**：仅检索到 Lacuna 的二手精读摘要，**未核到原始 arXiv ID / DOI**，引用时需回溯原文。

**核心做法**：把"知识空白"分成两类——

- **Explicit gap**：文本里明说的（"remains unknown"、"further research is needed"）
- **Implicit gap**：需要推理才能得到的（推理链缺一环、泛化可能不成立、冲突结论未调和）

隐式 gap 用 **TABI（Toulmin-Abductive Bucketed Inference）** 结构化输出：**Claims（推断出的空白）/ Grounds（支撑证据）/ Warrants（连接二者的推理）**。配 3-shot prompting 效果最好。

**关键数字**：段落级隐式 gap 准确率最高约 **84.43%**；全文级试点中 83.3% 的参与者认为识别出的 gap **事实成立**，但只有 65% 认为建议方向**可行**——瓶颈在可行性约束（技术、预算），LLM 目前不考虑。

**可借鉴**：我们的 Future Work 挖掘只做显式抽取。TABI 的三段式结构能让输出**可核查**（每个 gap 带证据和推理），这正好呼应赛题"可溯源"的要求。GAPMAP 的"可行性不足"也是个诚实披露的好素材。

### 1.6 自动化文献综述的人机协同框架

**出处**：Mulla et al., *Human-Centred Research Automation: Agentic AI Framework for Literature Discovery, Filtering, and Gap Identification*, HCAIep '26, DOI [10.1145/3777490.3777511](https://doi.org/10.1145/3777490.3777511)

**结果**：基于 Llama 3.1 8B 的筛选准确率 **71.05%**（对照人类主导的 SLR 基线），产出结构化的 gap / 潜在方向 / 候选方案。

**启示**：**小模型可用**。我们用 27B 全流程，不必迷信大模型；71% 这个量级也提醒我们：自动筛选的准确率天花板并不高，产品设计应保留人工复核位。

---

## 2. 解析侧：版面分析与表格/公式结构

| 方案 | 出处 | 关键做法 | 与我们的关系 |
|---|---|---|---|
| **Docling** | arXiv:2408.09869（IBM） | RT-DETR 版面模型 → TableFormer 表格结构 → Texify 公式 → DoclingDocument 树；**HybridChunker** 按标题层级 + token 预算切块 | 版面分类比我们自研的"字号+加粗"启发式更准；表格 TEDS 基准更规范 |
| **MinerU** | arXiv:2409.18839 | 开源精确文档内容抽取 | 对标 |
| **UniMERNet** | arXiv:2404.15254 | 通用数学表达式识别网络 | 公式识别的专用模型 |
| **DocLayout-YOLO** | arXiv:2410.12628 | 版面分析加速 | 对标 |
| **OmniDocBench** | arXiv:2412.07626 | 多样化 PDF 解析评测基准 | **我们缺一个解析侧的公开评测口径** |
| **PyMuPDF-Layout** | pymupdf.io 官方博客（2025-10 数据，2026-03 更新） | **不渲染图片**，直接用 PDF 内部结构特征做版面分类 | ⭐ 见下 |

### 2.1 ⭐ PyMuPDF-Layout：可能直接可用的一次升级

PyMuPDF 官方在 DocLayNet 上给出了自研 layout 模型与 Docling 的对比：

| 模型 | 参数量 | DocLayNet F1 | 是否需要 GPU |
|---|---|---|---|
| Docling（RT-DETR，基线） | 20M | 0.8102 | 需要 |
| PyMuPDF-Layout（PDF 特征） | 1.3M | 0.8270 | **不需要** |
| PyMuPDF-Layout（融合特征） | 1.8M | **0.8356** | **不需要** |

融合特征相比纯 PDF 特征整体 F1 +0.025，其中**页脚检测 +0.134**；模型参数比 Docling 少 11.1 倍。

> **对我们尤其重要**：我们的解析栈本来就在 PyMuPDF 上，且服务器 GPU 已被 vLLM 占满。**PyMuPDF-Layout 是唯一"不占 GPU、且比需要 GPU 的 Docling 更准"的路线**。这条值得实测评估（跑一遍我们 39 个 PDF，比对标题树/表格/图的抽取差异）。

> ⚠️ 边界说明：官方明确 **picture 分类仍是 PDF 特征路线的结构性短板**（与 Docling 差距 −0.385→−0.343）；且 Docling 的分类体系把所有 title 归入 section-header，DocLayNet 的 title 类 F1 因此为 0。

### 2.2 表格结构：我们目前最明确的技术短板

- Docling 表格抽取准确率 97.9%（IBM 公布基准）；Marker 表格抽取落后 4–6 个百分点；Marker 更快（6.1 页/秒）、多栏处理更好（96.1%）
- 评测口径是 **TEDS**（Table Tree Edit Distance Similarity），基准是 PubTabNet / FinTabNet
- 社区实测的失败模式：**合并单元格**、**无框线表格**（网格检测直接失效）、**跨页表格**

**我们的现状**：`PyMuPDF.find_tables()`，串行（实测并行会静默丢表，5→1），全库 53 张表。**这是我们与 Docling/Marker 差距最大的一块，也是最容易被评委追问的一块。**

**可行的中间路线**：保留 `find_tables` 做候选框定位（快、无依赖），再让已有的 VLM 复核/重建表格结构（我们已有 VLM 链路，公式转写就是这么做的）。相当于用视觉模型补上结构模型缺的"合并单元格 / 无框线"能力。

---

## 3. 引用图谱与文献地图

**Connected Papers**（基于 Semantic Scholar 数据）的关键设计：

- **不只用直接引用关系**，而是用 **共引（co-citation）+ 共同参考文献（co-reference）** 建边
- 两种衍生视图：**Prior works**（前身工作簇）与 **Derivative works**（后续工作簇）
- 可视化编码：**节点大小 = 被引次数**，**颜色深浅 = 发表时间**，边 = 引用关系

**同赛道产品**：Litmaps（引文网络）、ResearchRabbit（引文网络 + AI 聚类摘要）、SciSpace。

**我们的现状**：图谱只建**库内论文 ↔ References 条目**的直接引用边；指标为 PageRank + Betweenness + k-core + HITS；实测 11 节点 5 边 2 基石。

**差距与借鉴**：
- 加 **共引边 + 共参考文献边** → 图从"引用链"变成"研究脉络"，基石/衍生分类会立刻更准。这不需要新数据，**只需新算法**。
- 补 **Prior works / Derivative works 双视图** → 前端工作量小，视觉冲击大。
- 节点编码（大小=被引、颜色=年份）我们**已有**（技术文档记载"时间线配色"）。

**Scite 的 Smart Citations**：把每条引用标为 **supporting / contrasting / mentioning** 三类，并展示**引用句原文**。底层是一个深度学习模型（Nicholson et al., 2021）。

**我们的"引用含金量"目前是二分类**（实质性 / 装饰性）。改成三分类 + 展示引用原句，**改动小、可信度提升明显**，且能直接对上赛题 1.4「引用含金量」评分点。

---

## 4. 检索评测：我们最大的空白

### 4.1 指标定义（RAGAS / ARES）

| 框架 | 出处 | 指标 | 与人类标注一致度 |
|---|---|---|---|
| **RAGAS** | Es et al., arXiv:2309.15217 (EACL 2024) | Faithfulness（忠实度）、Answer Relevancy、Context Precision、Context Recall | 0.95 / 0.78 / 0.70（在 50 页维基标注集上） |
| **ARES** | Saad-Falcon et al., arXiv:2311.09476 (NAACL 2024) | Context Relevance、Answer Faithfulness、Answer Relevance；合成数据微调轻量 judge + PPI（预测功效推断） | 用几百个人工标注即可准确评测，且跨域稳健 |

**关键定义（可以直接抄进我们的技术文档）**：

- **Faithfulness** = 答案中被检索上下文支持的陈述数 ÷ 答案陈述总数
- **Answer Relevancy** = 让 LLM 为答案反向生成问题，再与原问题做 embedding 余弦相似度（往返检验）
- **Context Precision** = Top-K 中相关 chunk 数 ÷ K

**评测的关键原则**（AWS Labs 的 RAG 评测章节、Ragas 官方文档均强调）：**检索步和生成步必须分开测**——"模型没见过就答错"必须修检索，"给对了材料还答错"才修生成。

### 4.2 我们的问题

我们现在的 92 个用例测的是：
- 解析层：标题树层级、双栏顺序、表格结构、参考文献字段
- 去重：v1/v7 判为重复
- 检索：BM25/Dense/RRF/rerank 各自通道
- 护栏：证据绑定、幻觉剔除、无据拒答
- 图谱：PageRank 归一、k-core 单调、基石判定

**但没有一个数字回答"答案到底准不准"。** 赛题 1.2.1/1.2.2 明确要求单篇 5 问和多篇 4 问"全部带出处"，这是**可以用指标量化**的，却只写在验收清单的勾选框里。

**建议**：在 13 篇库上实现 3 个指标（忠实度、引用可溯源率、上下文精确率），跑出表格写进技术文档 §6.2。这比再写 50 个单元测试对评委更有说服力。

---

## 5. 商业产品：分工与它们的软肋

| 产品 | 定位 | 核心功能 | 软肋（对我们是机会） |
|---|---|---|---|
| **Elicit** | 早期发现 / 系统综述 | 200M+ 论文索引；Extract data 把方法/样本量/结局抽成表格 | 抽取准确率约 **85–90%**；**约 10% 的字段论文里根本没报，工具会编一个值**；需付费 |
| **Consensus** | 快问快答 | 论文问答 + **Consensus Meter**（Yes / No / Possibly） | 只给摘要不给原文定位；易断章取义 |
| **Scite** | 引用验证 | Smart Citations：supporting / contrasting / mentioning + 引用原句 | 需付费；不校验参考文献列表本身 |
| **ResearchRabbit** | 引文网络 | 图谱 + AI 聚类摘要 | 只读外部索引，不处理你的本地 PDF |
| **Semantic Scholar** | 学术搜索 | TLDR 一句话摘要 + 引文指标 | 摘要不等于证据 |
| **Connected Papers** | 文献地图 | 共引/共参考建边 + Prior/Derivative 双视图 | 依赖外部索引 |

**三条行业共识（可直接写进我们的定位论证）**：

1. **幻觉引用已成为学术事故**：Retraction Watch / STAT 报道的 PMC 审计显示，编造参考文献比例从 2023 年的 1/2828 升至 2025 年的 1/458，2026 年前七周已达 1/277；GPTZero 在 NeurIPS 2025 的 51 篇录用论文中发现 100 条幻觉引用；Enago 的多模型研究显示 AI 生成参考文献仅约 26.5% 完全正确。
   → **这是我们的核心叙事的市场证据**，比我原来写的"科研工作者抱怨答案不敢信"有力得多。
2. **三个工具各管一段，没有一个全包**：发现（Elicit）、证据合成（Consensus）、引用验证（Scite）。
   → 我们的定位可以是"**一条流水线打通六段**"，这正是赛题的六模块结构。
3. **所有产品都仍然需要人工核验**：连 Scite 官方文档都明说它不检查正文引用与参考文献列表是否一致。
   → **诚实披露边界**（我们也在技术文档的"已知限制"里这么做了）反而是加分项。

> ⚠️ 上表中的价格、具体比例来自各家官方文档与第三方评测文章；引用比例数据来自 Retraction Watch/STAT 报道与 GPTZero 分析，**属二手报道转述，引用前建议回溯原始出处**。

---

## 6. 差距清单：别人怎么做 / 我们怎么做 / 差距 / 建议

> 优先级按 **（对赛题得分的贡献 ÷ 改动成本与风险）** 排序。

### P0 — 值得做，改动小、评分点直接受益

| # | 能力 | 别人怎么做 | 我们现状 | 建议 | 成本 |
|---|---|---|---|---|---|
| 1 | **引用情境三分类** | Scite：supporting / contrasting / mentioning + 引用原句 | 二分类（实质性/装饰性） | 改为三分类，前端展示引用句原文与标签 | 小 |
| 2 | **元数据感知检索** | PaperQA2：标题/作者折进 chunk 文本再编码 | 纯正文编码 | chunk 编码时前置 `title / authors / year` | 极小 |
| 3 | **量化评测指标** | RAGAS/ARES：忠实度、上下文精确率、答案相关性 | 只有单元测试 | 实现忠实度 + 引用可溯源率，在 13 篇库上跑数字写进文档 §6.2 | 中 |
| 4 | **共引 / 共参考文献建边** | Connected Papers：co-citation + co-reference | 只有直接引用边 | 图谱加两类衍生边，基石判定重算 | 中 |
| 5 | **Prior / Derivative 双视图** | Connected Papers / ResearchRabbit | 单一图谱 | 图谱工作台加两个派生视图 | 小 |

### P1 — 结构性改进，价值最高

| # | 能力 | 别人怎么做 | 我们现状 | 建议 | 成本 |
|---|---|---|---|---|---|
| 6 | **自反馈推理循环** | OpenScholar：初稿 → 自评 → 补充检索 → 重写 | 一次性取证、一次性生成 | 在护栏后加"证据不足则补检索重生成"的循环（受 `AGENT_MAX_STEPS` 约束） | 中 |
| 7 | **跨论文矛盾检测** | PaperQA2 ContraCrow：每篇 2.34±1.99 个矛盾，70% 专家验证 | 无 | 新增"多文对比"子能力：抽取结论、比对冲突 | 中大 |
| 8 | **多视角综述** | STORM：视角发现 → 分视角提问 → 模拟对话 → 先大纲后写 | 单视角一次生成 | 用库内 Future Work / Limitations 段做视角发现，先出大纲再写 | 中 |
| 9 | **隐式 gap 推理（TABI）** | GAPMAP：Claims / Grounds / Warrants 三段式 | 只抽显式 Future Work | Future Work 输出改成三段式结构 | 中 |

### P2 — 成本高或风险大，需评估后再定

| # | 能力 | 别人怎么做 | 我们现状 | 建议 | 成本 |
|---|---|---|---|---|---|
| 10 | **表格结构模型** | Docling TableFormer，TEDS 97.9% | `find_tables` + 串行 | 先做 VLM 复核路线（保留 find_tables 定位）；或评估接入 Docling | 大 |
| 11 | **版面分类模型** | Docling RT-DETR（F1 0.8102，需 GPU）/ PyMuPDF-Layout（F1 0.8356，**不需 GPU**） | 自研字号+加粗启发式 | **实测 PyMuPDF-Layout** 在我们 39 个 PDF 上的效果（我们本来就在用 PyMuPDF，零新增栈） | 中 |
| 12 | **训练专用 retriever/reranker** | OpenScholar 训了自己的 retriever + reranker | 通用 bge-m3 + bge-reranker-v2-m3 | 我们的语料量（13 篇）远不够训练，先用通用模型 + 元数据增强 | 很高，不建议 |

### 明确**不**建议做的

- **不要引入多智能体"角色扮演"层**：Agent Laboratory 的角色层级（PhD Student / Professor）服务于**代码实验闭环**，我们的场景没有实验执行环节，套用只是装饰。
- **不要追求 4500 万篇的规模**：OpenScholar 的规模优势是数据资产，我们的差异化在**结构化深度 + 完全本地化 + 可溯源**。
- **不要把 LLM 自评当效果证据**：Agent Laboratory 披露的 optimism gap（6.1 vs 3.8）就是反面教材。指标要么用规则算（忠实度、溯源率），要么明确标注是模型评分。

---

## 7. 我们的差异化定位（调研之后重述）

调研把我们的位置讲清楚了：

> **SciAgent = 一条把「发现 → 解析 → 取证问答 → 版式精读 → 引用验证 → 图谱综述 → 写作」串成单条流水线的、可完全本地部署的科研助手；核心差异不在"能不能读"，而在"每一条断言能不能被点回原文的那一行"。**

三个别人没有、我们有（或可以低成本有）的点：

1. **可溯源粒度**：PaperQA2 精确到页，我们精确到 `paper_id / page_no / bbox / char_start / char_end`——断言可以点回原文的**那一行**。OpenScholar 的引用准确率对标人类专家，靠的是"引用真实存在"；我们还能证明"引用说的就是这半句话"。
2. **完全本地化**：Elicit / Consensus / Scite 全部是付费 SaaS，论文和检索历史出本机。我们在 192 核单机上跑通了全部六个模块。
3. **解析深度 + 结构化**：我们不是"文本流 + 摘要"，而是标题树 26 级、双栏还原、表格、图、VLM 转写的公式、结构化参考文献。

---

## 8. 参考文献与来源

**学术系统**

1. Lála et al. *Language agents achieve superhuman synthesis of scientific knowledge.* arXiv:2409.13740 — https://arxiv.org/abs/2409.13740
2. Asai et al. *Synthesizing scientific literature with retrieval-augmented language models.* **Nature 650, 857–863 (2026)**, DOI 10.1038/s41586-025-10072-4 — https://doi.org/10.1038/s41586-025-10072-4 ；预印本 arXiv:2411.14199
3. Shao et al. *Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models.* arXiv:2402.14207（Stanford OVAL）— https://arxiv.org/abs/2402.14207
4. Schmidgall et al. *Agent Laboratory: Using LLM Agents as Research Assistants.* arXiv:2501.04227（EMNLP 2025）— https://arxiv.org/abs/2501.04227
5. Mulla et al. *Human-Centred Research Automation: Agentic AI Framework for Literature Discovery, Filtering, and Gap Identification.* HCAIep '26, DOI 10.1145/3777490.3777511 — https://doi.org/10.1145/3777490.3777511
6. *GAPMAP: Mapping Scientific Knowledge Gaps in Biomedical Literature Using Large Language Models.* —— ⚠️ **仅核到二手精读摘要，原文出处待回溯**

**解析与文档理解**

7. Lysak et al. *Docling Technical Report.* arXiv:2408.09869（IBM Research）— https://arxiv.org/abs/2408.09869
8. *MinerU: An Open-Source Solution for Precise Document Content Extraction.* arXiv:2409.18839
9. *UniMERNet: Universal Mathematical Expression Recognition Network.* arXiv:2404.15254
10. *DocLayout-YOLO: Enhancing Document Layout Analysis.* arXiv:2410.12628
11. *OmniDocBench: Benchmarking Diverse PDF Document Parsing.* arXiv:2412.07626
12. PyMuPDF. *PyMuPDF-Layout Performance on DocLayNet: A Comparative Evaluation.* https://pymupdf.io/blog/pymupdf-layout-performance-on-doclaynet-a-comparative-evaluation
13. *Document Parsing Engines in Production RAG: Comparing Docling, Marker, MinerU, and Unstructured.* https://www.llms.blog/posts/open-source-document-parsing-and-layout-analysis-engines-in-production-comparing-docling-marker-mineru-and-unstructured
14. *Docling vs Unstructured: PDF Accuracy Benchmark*（500 份企业 PDF 实测）— https://www.ertas.ai/blog/pdf-parsing-accuracy-benchmark-docling-unstructured
15. Pfitzmann et al. *DocLayNet: A Large-scale Dataset for Document Layout Analysis.* arXiv:2206.01062
16. Zhong et al. *PubTabNet: Largest Dataset Ever for Table Structure Recognition.*（TEDS 指标来源, 2019）
17. Nassar et al. *TableFormer: Table Structure Understanding with Transformers.* arXiv:2203.01017

**检索与评测**

18. Es et al. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217（EACL 2024）
19. Saad-Falcon et al. *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* arXiv:2311.09476（NAACL 2024）
20. AWS Labs. *Evaluating Retrieval-Augmented Generation (RAG) Systems.* https://awslabs.github.io/generative-ai-atlas/topics/2_0_technical_foundations_and_patterns/2_6_model_evaluation_and_selection_criteria/2_6_4_domain_specific_evaluations/2_6_4_1_evaluating_rag_systems/2_6_4_1_evaluating_rag_systems.html
21. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401（NeurIPS 2020）
22. Cormack et al. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009

**引用图谱与引用情境**

23. Nicholson et al. (2021) —— Scite Smart Citations 的深度学习分类模型
24. Connected Papers —— https://www.connectedpapers.com/（基于 Semantic Scholar 数据）
25. Semantic Scholar —— https://www.semanticscholar.org/

**商业产品与行业数据**

26. Academic AI Trends. *Elicit vs Consensus vs Scite: Which to Pay For (2026).* — https://academicaitrends.com/blog/elicit-vs-consensus-vs-scite-2026
27. ByteLedger. *AI research paper tools in 2026: Elicit, Consensus, ResearchRabbit, scite.* — https://byteledger.vizleo.com/blog/ai-research-paper-tools-2026
28. ProofreaderPro. *Consensus vs Elicit vs Scite: AI Research Assistants Compared*（24 个研究工作流、4 位盲评者）— https://proofreaderpro.ai/blog/consensus-vs-elicit
29. Tesify. *Best AI Tools for Literature Reviews 2026.* — https://tesify.app/best-ai-literature-review-tools-elicit-researchrabbit-2026/
30. Scite 官方文档（Smart Citations / Reference Check）— https://scite.ai

> **引用纪律**：本文中来自官方文档或论文原文的数字均已在上文标注出处；来自第三方评测文章与媒体报道的（价格、抽取准确率、引用幻觉率统计、optimism gap 等）已标注为二手转述。**正式写入技术文档前，凡是要放进"实验结果"栏的数字，必须回溯原始出处核对**。

---

*文档结束 · SciAgent Team · 2026-11-12*
