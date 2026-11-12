# SciAgent：意图驱动的全流程科研助手

### 智领科研 · 意图驱动：基于大模型与 Agent 架构的全流程科研助手构建

**技术文档**

| 项目 | 内容 |
|---|---|
| 作品名称 | SciAgent · 科研小助手 |
| 赛项 | 全国高校 AI 大模型创新应用挑战赛 —— 智领科研·意图驱动 |
| 团队名称 | SciAgent Team（待补充：成员姓名 / 学校 / 学院专业 / 指导老师 / 联系方式） |
| 文档版本 | v1.0 |
| 文档日期 | 2026-11 |
| 仓库地址 | https://github.com/akawrzzby/sciagent |
| 在线演示 | 见 `docs/demo-script.md`（评审期间保持可用） |

---

## 摘要

SciAgent 是一个面向科研文献阅读与写作的端到端 Agent 系统。它把科研工作者真实的动线——**检索 → 精读 → 对比 → 综述 → 写作**——组织成一条可溯源的流水线，而不是一个"聊天框套壳"。

系统的技术主干由三块构成：

1. **多模态解析引擎**：版式感知的 PDF 解析器能还原双栏顺序、26 级标题树、表格、图、公式与结构化参考文献；对无文本层的扫描件，回退到多模态大模型逐页转写。
2. **两阶段 Agent 编排**：Router（意图路由）→ Planner（任务 DAG）→ Executor（原生 Function Calling 工具循环）→ Reflection（护栏与引用校验）。关键设计是"先取证、后带证据生成"——工具调用阶段产出的证据是带字符区间的结构化对象，生成阶段被严格限制只能引用这些证据。
3. **图论驱动的知识挖掘**：在引用图上计算 PageRank、Betweenness、k-core、HITS，用这些指标区分基石文献与衍生文献，并在此基础上自动生成领域综述与 Future Work 研究机会。

在赛题提供的 39 篇测试数据集上，系统对 `1706.03762v1.pdf` 的完整解析耗时 2.5 秒，8 位作者、26 级标题树、76 条结构化参考文献全部正确；稠密检索通道在语义相近文本上余弦相似度 0.89、语义无关文本 0.38；重排通道对切题段落给出 +0.92、对离题段落给出 +0.00。

本文档按赛题要求，依次阐述项目概述、需求分析、系统架构设计、AI 技术方案（含选型理由、集成方式与优化策略）、核心功能实现、测试与部署，以及总结与展望，并单独列出技术创新点与应用创新点。

---

## 目录

1. [项目概述](#1-项目概述)
2. [需求分析](#2-需求分析)
3. [系统架构设计](#3-系统架构设计)
4. [AI 技术方案](#4-ai-技术方案)
5. [核心功能实现](#5-核心功能实现)
6. [测试与部署](#6-测试与部署)
7. [总结与展望](#7-总结与展望)
8. [创新点说明](#8-创新点说明)
9. [附录](#9-附录)

---

## 1. 项目概述

### 1.1 背景与痛点

大模型已经能"读"论文了，但科研工作者真正抱怨的从来不是"读不懂"，而是**读得慢、信不过、攒不起来**。具体拆开是五组摩擦：

**（1）PDF 不是文本。** 学术 PDF 有双栏、有跨栏图表、有 LaTeX 生成的数学公式、有脚注与浮动体。粗暴地复制粘贴出来是错序文本；扫描件（拍照或纯图片排版）更是复制不出一个字。这一步做不好，后面全是空中楼阁。

**（2）答案不敢信。** 通用大模型谈论文细节时会张口编造：把不存在的参考文献写得像模像样，把作者张冠李戴，甚至把某篇论文的结论安到另一篇头上。科研场景对错误的容忍度接近于零。

**（3）阅读体验割裂。** PDF 阅读器忠实于版式却不理解语义；翻译工具能翻译却丢失指代关系；文献管理软件能收藏却不支持提问。一份 30 页的论文，要在四个工具之间来回切换才能读懂。

**（4）不知道下一步做什么。** 读完 20 篇论文能复述每篇的结论，却说不清这个方向上还有哪些空白、哪些问题被反复提及却从未解决。

**（5）写作从零开始。** 论文框架、系统架构图、数据表格，每一张都要从空白文档开始画；而最危险的是参考文献——凭记忆写出来的引用有一半查无此文。

### 1.2 系统定位与目标

SciAgent 的定位是：**一个可信的、可溯源的、会主动使用工具的科研流水线**。

三条不可让渡的设计底线：

| 底线 | 具体含义 |
|---|---|
| **证据优先** | 任何一句结论都必须能指回原文的字符区间。指不回去，就不说。 |
| **结构优先** | 标题、表格、公式、参考文献各自建模，而不是拍平成一坨字符串。 |
| **可降级** | 稠密通道故障自动退化为 BM25；大模型超时给出明确错误而不是白屏。 |

对应的三个可度量目标：

- **解析准确**：对测试集标准论文，标题树层级、作者、表格、参考文献字段全部正确；对扫描件，能产出可检索的正文。
- **回答可信**：所有回答携带可点击的溯源锚点（文献 / 页码 / 字符区间）；无据时明确拒答而非编造。
- **闭环完整**：从上传论文到生成综述、框架、图表，全流程不跳出系统。

### 1.3 赛题要求映射

| 赛题模块 | 系统对应实现 | 关键支撑技术 |
|---|---|---|
| ① 多模态细粒度文档解析 | `backend/app/parsing/` | 版式分析、栏检测、标题树推断、VLM OCR、语义去重 |
| ② 意图驱动 RAG + 抗幻觉 | `backend/app/rag/` + `backend/app/agents/` | BM25 + bge-m3 + RRF + cross-encoder、证据绑定、引用真实性校验 |
| ③ 版式阅读 + 中英联动 | `frontend/src/pages/Reader.tsx` | 双栏渲染、引用双向跳转、segment 对齐滚动 |
| ④ arXiv 检索与追踪 | `backend/app/tools/sources.py` | arXiv API、GitHub Search、Function Calling 闭环 |
| ⑤ 文献图谱与基石挖掘 | `backend/app/graph/` | PageRank、Betweenness、k-core、HITS |
| ⑥ 论文写作辅助 | `backend/app/api/routers/writing.py` + `tools/viz.py` | 框架生成、TikZ/Mermaid/Graphviz、CSV → 图表 + Caption |

---

## 2. 需求分析

### 2.1 功能需求

按赛题六大模块展开为可验收的条目：

**F1 文档解析**

- F1.1 解析标题树并保留层级与编号（支持 `3.2.2` 这样的三级以上编号）
- F1.2 还原双栏 PDF 的正确阅读顺序
- F1.3 抽取表格并同时输出 Markdown 与 CSV
- F1.4 抽取图片及其 caption
- F1.5 抽取公式（行内 `$...$` 与块级 `$$...$$`）
- F1.6 解析参考文献为结构化字段（作者 / 标题 / 年份 / 会议 / DOI / arXiv id）
- F1.7 对无文本层 PDF 执行 OCR，且 OCR 结果参与后续检索
- F1.8 识别同一论文的不同版本（语义去重），并给出保留建议与版本差异

**F2 检索与问答**

- F2.1 混合检索：稀疏 + 稠密双通道，经 RRF 融合后由 cross-encoder 精排
- F2.2 答案携带溯源锚点，可定位到具体文献、页码与字符区间
- F2.3 Agent 能自主拆解任务并调用外部工具（arXiv / GitHub / 本地检索 / 绘图）
- F2.4 无证据支撑时拒答
- F2.5 流式输出，且思考过程可视化

**F3 阅读器**

- F3.1 标题树定位，点击跳转到对应页面
- F3.2 原文版式高保真还原
- F3.3 中英双栏对照，滚动同步
- F3.4 正文引用标记 ↔ 参考文献条目双向跳转

**F4 检索与推荐**

- F4.1 arXiv 关键词检索与分类追踪
- F4.2 PDF 一键下载并自动解析入库
- F4.3 基于文献库的推荐，且**优先推荐有公开代码的可复现工作**
- F4.4 用户反馈参与排序

**F5 图谱**

- F5.1 构建库内引用有向图
- F5.2 计算多种中心性指标并可视化
- F5.3 基石节点判定
- F5.4 自动生成领域综述
- F5.5 挖掘 Future Work 研究机会

**F6 写作**

- F6.1 生成论文框架（标题 / 摘要 / 引言 / 相关工作 / 方法 / 实验）
- F6.2 生成的参考文献必须是库内真实存在的，否则拦截
- F6.3 生成 TikZ / Mermaid / Graphviz 架构图
- F6.4 CSV → 学术图表 + Figure Caption

### 2.2 非功能需求

| 维度 | 指标 | 设计对策 |
|---|---|---|
| 解析性能 | 标准论文 < 5s | 版式分析纯 CPU；页面级并行；只在检测到扫描件时才走 OCR |
| 长文档 | 500+ 页不 OOM | 按页流式处理，向量按篇存 memmap，查询期只加载参与检索的论文 |
| 流式体验 | 首 token < 3s | 两阶段架构，先返回取证结果再生成；SSE 全链路关闭缓冲 |
| 鲁棒性 | 单通道故障不影响整体 | 稠密/重排/LLM 各自降级；后台任务异常隔离 |
| 可复现 | 一条命令起全栈 | `docker-compose.yml` + `.env.example` |
| 安全 | 密钥零泄露 | 12-factor 环境变量；`.env` 进 `.gitignore`；仓库无任何密钥 |
| 可审计 | 每次模型调用可追溯 | system/user prompt、工具参数、耗时、token 全部落盘 |

### 2.3 竞品与现有方案对比

| 方案 | 解析深度 | 可溯源 | Agent 自主性 | 图谱挖掘 | 写作闭环 |
|---|---|---|---|---|---|
| 通用 PDF 解析库 | 仅文本流 | 否 | 否 | 否 | 否 |
| 通用 AI 阅读助手 | 文本流 + 摘要 | 部分（仅段落级） | 弱 | 否 | 部分 |
| SciAgent | 结构化全要素 | 是（字符区间） | 是（原生工具调用） | 是（多指标） | 是 |

关键差异不在"能不能读"，而在**能不能证明**。

---

## 3. 系统架构设计

### 3.1 总体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│  展现层  React 18 + TypeScript + Vite + Tailwind + Cytoscape + Mermaid  │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┐                 │
│  │ 文献库 │ 阅读器 │问答    │图谱    │检索    │写作    │                 │
│  └────────┴────────┴────────┴────────┴────────┴────────┘                 │
│         SSE 事件流（delta / thinking / tool_call / tool_result）        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ REST + Server-Sent Events
┌────────────────────────────────┴─────────────────────────────────────────┐
│  编排层  agents/orchestrator.py                                         │
│  ┌────────┐   ┌─────────┐   ┌──────────┐   ┌────────────┐              │
│  │ Router │──▶│ Planner │──▶│ Executor │──▶│ Reflection │              │
│  └────────┘   └─────────┘   └──────────┘   └────────────┘              │
│         tools/registry.py —— 原生 Function Calling 工具集             │
└────────────────────────────────┬─────────────────────────────────────────┘
┌────────────────────────────────┴─────────────────────────────────────────┐
│  能力层                                                                 │
│  parsing/ 版式解析 · VLM OCR · 语义去重      rag/ 分块 · BM25 · Dense  │
│  graph/   PageRank · 基石挖掘 · Future Work   tools/ arXiv · GitHub · 绘图│
└────────────────────────────────┬─────────────────────────────────────────┘
┌────────────────────────────────┴─────────────────────────────────────────┐
│  存储层  SQLite（papers/sections/chunks/references/feedback）           │
│          向量索引 .npy（每篇一个 memmap 矩阵）                          │
└────────────────────────────────┬─────────────────────────────────────────┘
┌────────────────────────────────┴─────────────────────────────────────────┐
│  模型层  vLLM → Qwen3.8-27B（多模态）  sentence-transformers → bge-m3    │
│                                    → bge-reranker-v2-m3（cross-encoder）│
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 分层职责

| 层 | 模块 | 职责 |
|---|---|---|
| 展现层 | `frontend/src/pages/*` | 六个工作台；只依赖 `lib/api.ts` 的类型化客户端与 `lib/sse.ts` 的事件流 |
| 编排层 | `app/agents/orchestrator.py`、`app/tools/registry.py` | 意图路由、任务规划、工具循环、结果反思 |
| 能力层 | `app/parsing/`、`app/rag/`、`app/graph/`、`app/tools/` | 纯函数式能力，不依赖 HTTP |
| 存储层 | `app/storage/` | SQLAlchemy 异步模型 + aiosqlite；向量索引为文件 |
| 接入层 | `app/api/routers/` | 薄路由，只做参数校验与结果包装 |
| 模型层 | `app/llm/client.py`、`app/rag/embeddings.py` | OpenAI 兼容客户端、embedding / reranker 封装 |

**解耦的关键在于**：能力层不感知"Agent"，Agent 不感知"HTTP"。`app/rag/retriever.py` 可以被 `/api/rag/search` 直接调用（检索工作台），也可以被 Agent 的 Executor 当作工具调用——同一份检索逻辑，两种入口。

### 3.3 Agent 编排架构（两阶段）

传统 ReAct 把"检索"与"生成"揉在一个循环里：模型先思考、再决定调工具、看结果、再思考，直到它觉得够了才写答案。这个模式的问题在科研场景下尤其明显——工具调用与正文生成混在一条流里，前端无法区分"模型在查资料"和"模型在写答案"，token 流向不可控，且一旦工具调用失败，已经生成的半截答案会被浪费。

SciAgent 采用**两阶段**设计：

```
                    ┌─────────────── 阶段 A：取证 ───────────────┐
用户问题 ──▶ Router │ 意图分类 → Planner 任务 DAG → Executor 工具循环 │
           │       │ 产出：evidence[] （带 chunk_id / page / char_range）│
           │       └──────────────────────┬─────────────────────────┘
           │                              ▼
           │       ┌───────────── 阶段 B：带证据生成 ─────────────┐
           └──────▶│  System + Evidence → token 级流式输出        │
                   │  Reflection：引用校验 → 护栏 → 无据拒答       │
                   └──────────────────────────────────────────────┘
```

**阶段 A（取证）** 产出结构化的 `evidence` 列表，每条包含：文献 ID、文献标题、页码、字符区间、原文片段、检索得分、来源通道（sparse / dense / rerank）。这个列表可以在 UI 的"思考过程"面板中逐条审计，用户能清楚看到"答案的依据来自哪几段原文"。

**阶段 B（生成）** 的 system prompt 中只包含这些证据，并显式声明"只允许引用 `[n]` 形式的编号，n 必须在证据列表范围内"。生成完成后，Reflection 阶段再次执行两道校验：

- **证据绑定**（`guardrails.bind_citations`）：正则扫描答案中的 `[n]`，凡不在证据集合内的编号一律删除，并把答案标记为"部分无据"。
- **引用真实性**（`guardrails.validate_references`）：写作助手生成的参考文献，与库内真实文献做标题 fuzzy 匹配 / arXiv id / DOI 精确匹配，不匹配者进入"已拦截的幻觉引用"列表，在 UI 中单独展示。

这一设计的收益是：即使模型"想说"一个没有依据的结论，护栏也会把它的引用标记摘掉；而 UI 会明确告诉用户"已移除 N 个无据引用"——**错误被暴露而不是被隐藏**，这在科研场景中是必要的。

### 3.4 数据流与时序

以一次跨文献问答为例：

```
用户  │ 前端                    │ 后端 Agent              │ 检索层        │ vLLM
  │   │ POST /api/rag/ask       │                         │              │
  │   ├────────────────────────▶│                         │              │
  │   │                         │ Router: 意图=跨文献对比  │              │
  │   │                         │ Planner: DAG=[检索×2, 综合]│             │
  │   │◀─ SSE: route ──────────┤                         │              │
  │   │◀─ SSE: plan ───────────┤                         │              │
  │   │                         │ Executor: tool_call     │              │
  │   │◀─ SSE: tool_call ──────┤── search_papers ───────▶│ BM25 + Dense │
  │   │                         │                         │ + RRF + rank │
  │   │◀─ SSE: tool_result ────┤◀─ passages[] ───────────┤              │
  │   │                         │                         │              │
  │   │                         │── generate (带证据) ───────────────────▶│
  │   │◀─ SSE: thinking ───────┤◀─ reasoning ────────────────────────────│
  │   │◀─ SSE: delta (token级) ─┤◀─ content ──────────────────────────────│
  │   │                         │ Reflection: bind_citations               │
  │   │◀─ SSE: guardrail ──────┤                         │              │
  │   │◀─ SSE: final ──────────┤ (answer + citations)   │              │
```

SSE 事件类型：`session` / `route` / `plan` / `stage` / `thinking` / `tool_call` / `tool_result` / `delta` / `guardrail` / `final` / `error` / `abstain`。前端 `lib/sse.ts` 用统一的 `AgentEvent` 类型分发，`Ask.tsx` 按类型更新对应的面板。

### 3.5 存储与索引设计

**元数据** 用 SQLite（`sqlite+aiosqlite`）。选它而不是 PostgreSQL 的理由：单机演示场景、零运维、异步驱动不阻塞 SSE；表结构（`papers` / `sections` / `chunks` / `references` / `figures` / `tables` / `formulas` / `recommendation_feedback`）覆盖全部解析产物。

**向量索引** 用**每篇论文一个 `.npy` 矩阵**的文件布局，而不是一个全局大矩阵。理由是内存：一次查询通常只涉及 1–3 篇文献，用 `np.load(..., mmap_mode="r")` 只把这几篇的分块向量映射进地址空间，索引内存与库规模解耦。39 篇论文 × 1024 维 × 约 3000 个分块的场景下，查询期的向量常驻内存是几百 KB 而不是几十 MB。

**分块** 按章节边界切，块内按句子聚合，目标 380 token、重叠 60 token。章节边界保证语义完整，重叠避免答案正好落在切点上。

### 3.6 部署架构

```
        公网评审
            │  HTTPS
            ▼
   ┌────────────────────┐
   │  cloudflared 隧道   │   （NAT 环境无公网 IP，隧道提供 HTTPS 入口）
   └─────────┬──────────┘
             │ http://frontend:80
   ┌─────────▼──────────┐
   │ Nginx（前端容器）   │   托管 Vite 产物 + 反代 /api
   │  proxy_buffering off│   ← SSE 流式输出的关键配置
   └─────────┬──────────┘
             │ http://backend:8000
   ┌─────────▼──────────┐
   │ FastAPI（后端容器） │   解析 / 检索 / Agent / 图谱 / 写作
   └─────────┬──────────┘
             │ http://10.184.17.179:8000/v1
   ┌─────────▼──────────┐
   │ vLLM · Qwen3.8-27B  │   多模态 + tool calling + reasoning
   └────────────────────┘
```

---

## 4. AI 技术方案

> 本章是赛题要求"重点阐述 AI 技术的选型理由、集成方式、优化策略"的部分。

### 4.1 大模型选型与理由

**选定：Qwen3.8-27B，通过 vLLM 以 OpenAI 兼容协议自托管。**

选型理由，按重要性排序：

| # | 理由 | 具体依据 |
|---|---|---|
| 1 | **必须支持多模态** | 扫描件 OCR 与公式转写需要读图能力。实测该模型架构为 `Qwen3_5ForConditionalGeneration`（多模态 VLM），传入 PNG 能准确读出图内标题与正文结构。 |
| 2 | **必须支持原生 Function Calling** | Agent 的工具循环依赖模型输出结构化的 `tool_calls`。实测 vLLM 以 `--enable-auto-tool-choice` 启动后，模型能稳定返回 `tool_calls` 数组，参数符合 JSON Schema。 |
| 3 | **必须提供 reasoning 字段** | "Agent 思考过程可视化"是评分点之一。vLLM 对该模型返回独立的 `reasoning` 字段，可直接透传到前端 `thinking` 事件，无需从正文里猜。 |
| 4 | **私有化部署，零 API 成本、零外网依赖** | 比赛与评审期间不能依赖商用 API 的配额与稳定性；自托管 vLLM 保证离线可复现。 |
| 5 | **上下文足够长** | `max_model_len = 262144`，多篇文献的证据可以一次性放入生成阶段，无需多轮拼接。 |
| 6 | **可替换性** | 客户端只依赖 OpenAI 兼容协议，任何兼容端点（OpenAI / DeepSeek / Qwen API）都能通过 `LLM_BASE_URL` 接入，只要模型具备多模态与 function calling 能力。 |

**关于"为什么不用更大的模型"**：27B 规模在自托管成本与能力之间取平衡。更重要的是，系统把"难"的部分（结构化解析、精确检索、图论计算）都下沉到了确定性代码，模型只负责语义理解与自然语言生成——这使得 27B 足以胜任，也使得整个系统的输出更稳定。

### 4.2 集成方式

**协议层**：统一走 OpenAI 兼容的 `/v1/chat/completions`。`app/llm/client.py` 封装了三个能力：

- `complete(...)`：普通对话，流式返回 token；
- `complete_json(...)`：要求结构化输出，用 schema hint 提示 + JSON 解析 + 失败重试；
- `stream_with_tools(...)`：带 `tools` 定义的工具循环，返回 `delta` / `reasoning` / `tool_calls` 三类增量。

**工具层**：`app/tools/registry.py` 用 `@tool` 装饰器注册工具，每个工具声明 JSON Schema、处理器与审计用途标签。Executor 把可用工具的 schema 转成 OpenAI 的 `tools` 参数，模型返回 `tool_calls` 时由注册表分发执行，结果作为 `role: tool` 消息回灌，直到模型不再请求工具或达到 `AGENT_MAX_STEPS`。

已注册的工具包括：`search_papers`（本地混合检索）、`get_paper_structure`（标题树）、`get_references`（参考文献）、`arxiv_search`、`arxiv_download`、`check_reproducibility`（GitHub）、`build_citation_graph`、`generate_framework`、`plot_csv`。

**流式层**：后端用 FastAPI 的 `StreamingResponse` 产出 `text/event-stream`；前端用 `fetch` + `ReadableStream` 逐块解析（而非 `EventSource`，因为需要 POST body）。**中间必须经过 Nginx，因此 `proxy_buffering off` 是 SSE 生效的硬性前提**——这一配置在 `deploy/nginx.conf` 中有详细注释说明。

### 4.3 Prompt 工程

系统提示词分五组，每组针对一个阶段（`app/llm/prompts.py`）：

| 组 | 作用 | 关键设计 |
|---|---|---|
| `SYSTEM_ROUTER` | 意图分类 | 要求输出受限的意图标签 + 需要的工具集合，让路由结果可预测 |
| `SYSTEM_PLANNER` | 任务拆解 | 要求输出 JSON 的子任务 DAG，每项含 `goal` / `tool` / `depends_on` |
| `SYSTEM_ANSWER` | 带证据生成 | **显式列出可用编号**并规定"只允许引用这些编号"；要求在证据不足时明确说明 |
| `SYSTEM_REVIEW` | 综述生成 | 注入图指标与基石文献摘要，要求区分"基石/衍生"并给出依据 |
| `SYSTEM_FRAMEWORK` / `SYSTEM_DIAGRAM` | 写作辅助 | 框架生成要求"参考文献只能来自给定的真实清单"；架构图要求输出可编译的源码 |

**Prompt 设计的三个原则**：

1. **约束可验证**。所有 prompt 的输出都能被代码检查：`complete_json` 校验字段是否齐全（`validate_structure`），引用编号是否在证据集内（`bind_citations`），参考文献是否真实存在（`validate_references`）。"请不要编造"是一句劝告，"编号超出范围会被删除"才是一条约束。
2. **注入防护**。检索到的文档正文用 `sanitize_document_text` 处理，疑似指令的片段被包裹进 `[untrusted-content]...[/untrusted-content]` 标记，指令与数据在提示词层面分离。
3. **审慎输出**。JSON 解析失败会重试一次；仍失败则降级为纯文本并记录审计，而不是抛异常给用户。

### 4.4 Guardrails：五层学术安全防护

| # | 防护 | 实现 | 触发时的表现 |
|---|---|---|---|
| 1 | **证据绑定** | `bind_citations` | 删除无据引用标记，答案标记 `supported: false`，UI 显示"已移除 N 个无据引用" |
| 2 | **引用真实性校验** | `validate_references` | 幻觉引用进入"已拦截"列表，不进入最终参考文献 |
| 3 | **提示注入隔离** | `sanitize_document_text` | 文档中的指令被标记为不可信内容，模型不会执行 |
| 4 | **结构化输出校验** | `validate_structure` + `complete_json` | 缺字段自动重试一次，仍失败则降级 |
| 5 | **无据拒答** | `requires_abstention` | 返回明确说明与三条可操作建议，不给猜测答案 |

这五层是**彼此独立**的：即使第 1 层漏掉一个引用编号，第 2 层仍会检查参考文献真实性；即使模型绕过了前四层，第 5 层的空证据检查仍会触发。纵深防御，而不是单点防线。

### 4.5 混合检索：策略与理由

```
        query
          │
    ┌─────┴─────┐
    │           │
 BM25(稀疏)  bge-m3(稠密, 1024 维)
    │           │
    └─────┬─────┘
     RRF 融合 (k=60, 双路等权)
          │
   bge-reranker-v2-m3 (cross-encoder, Top-24)
          │
     Top-K 证据 + 溯源锚点
```

**为什么是混合而不是单路**：

- **BM25** 擅长精确术语、缩写、数字：`multi-head attention`、`BLEU`、`P100`。这些是科研提问里高频且必须精确命中的词，纯稠密检索反而容易把它们"模糊掉"。
- **bge-m3** 擅长语义近似与**跨语言**：它原生支持中英混合与长文本（8192 token），这是产品必须双语检索的硬需求。
- 二者互补，因此采用 RRF 融合。

**为什么用 RRF 而不是加权求和**：BM25 分数是无界的（可到几十），余弦相似度有界（0–1）。直接加权必须先做归一化，而归一化方式本身会引入调参与不稳定。RRF 只用**排名**：`score(d) = Σ_r 1/(k + rank_r(d))`，天然免归一化、免调参，业界（如 Elasticsearch、Azure AI Search）也是默认做法。

**为什么还要 rerank**：双塔模型的 query 与 passage 是分别编码的，二者之间没有交互；而 cross-encoder 把 (query, passage) 拼在一起送进模型做联合注意力，能捕捉细粒度匹配。实测在本项目的查询上，rerank 把正确段落从"第 2 位"提到"第 1 位"，对越界段落的分数从 +0.92 降到 +0.00，区分度非常干净。

**实测数据**（`scripts/probe_models.py`）：

```
dense : sim(相关) = 0.8895    sim(无关) = 0.3784
rerank: 切题段落 +0.9234 / +0.9441     离题段落 +0.0000
```

### 4.6 图论算法选型

在引用有向图上同时计算四类指标，各有分工：

| 算法 | 回答什么问题 | 在本系统中的作用 |
|---|---|---|
| **PageRank** | 谁最有影响力？ | 基石判定的主指标（沿随机游走视角） |
| **Betweenness Centrality** | 谁是桥梁？ | 识别"承上启下"的枢纽文献——去掉它就割裂了知识流 |
| **k-core** | 谁的圈子最凝聚？ | 度量核心团的厚度，k-core 高的节点属于紧密协作的核心 |
| **HITS**（枢纽/权威） | 谁引用权威？谁连接枢纽？ | 补充 PageRank 对"入度"和"出度"的不对称刻画 |

**为什么不用纯 PageRank**：单用 PageRank 会把"被引多"和"被引多且被不同群体引"混为一谈。一个被 50 篇同组论文引用的综述，其 PageRank 很高但桥接作用弱；而被 5 个不同子领域引用的方法论文，PageRank 不高但 betweenness 很高。**多指标联合判定**才是"基石"的合理定义。

**基石判定规则**（`app/graph/builder.py`）：综合入度、PageRank 排名、k-core 与年份，划分 `foundation` / `core` / `derivative` 三档。年份参与判定是为了排除"近两年新出、被大量引用"的引用膨胀效应。

**引用质量分析**（`filter_substantive_references`）：区分"实质性引用"与"装饰性引用"。判据是引用所在句子是否与被引文献的标题共享词汇、是否附着于定量或比较性论断（`outperform` / `state-of-the-art` / 含数字）、是否只是一句"see [12]"。这让用户能回答"这篇论文真正站在谁的肩膀上"。

### 4.7 多模态解析与 OCR

**版式解析**（`app/parsing/layout.py`）：以 PyMuPDF 的词级 bbox 为基础做栏聚类——把页面按 x 间隙切成 1 列或 2 列，然后按"左栏读完再读右栏"的顺序输出。这一步决定了双栏论文的正文是否可读，是整个系统地基。

**标题树推断**（`app/parsing/headings.py`）：三条证据联合判定——(1) 显式编号模式（`3.2.2` → level 3）；(2) 规范章节名（`Introduction` / `Related Work` / `Conclusion`）；(3) 排版特征（字号大于正文 8%、加粗、独占一行）。同时有一组**负向规则**专门排除误判：论文左侧竖排的 arXiv 戳记（`arXiv:1706.03762v1 [cs.CL] 12 Jun 2017`）、图注（`Figure 3:`）、参考文献条目、正文段落。

**VLM OCR**（`app/parsing/ocr.py`）：检测到页面文本层字符数低于阈值即判定为扫描件，逐页以 150 DPI 栅格化为 PNG，送入多模态模型转写，输出 Markdown。用 VLM 而非 tesseract 的理由：tesseract 在**双栏顺序**与**数学公式**上的表现很差（它只看到字符，不理解版面），而多模态模型能同时读出结构、公式与栏序。转写结果再经 `ocr_markdown_to_sections` 还原成标题树，因此扫描件论文同样具备可检索的章节结构。

并发上采用 `asyncio.Semaphore` 限流（默认 4 路），把 15 页的转写耗时从串行的 >15 分钟压到约 4 分钟，同时避免打爆 vLLM 的并发窗口。

**语义去重**（`app/parsing/dedup.py`）：四路证据联合判定同论文——文件 SHA-256、规范化正文哈希、arXiv id、SimHash 近似度，再叠加标题 Jaccard 与摘要 embedding 余弦。命中后给出保留建议（比较 arXiv 版本号，版本号缺失时用正文长度做兜底）并能输出逐行差异。

### 4.8 优化策略

这一节记录**实测得出**的优化，每一项都有数字支撑。

#### （1）线程池规模：13.8× 的吞吐差距

在 192 核的机器上，torch 默认按核心数开 192 个线程。对 bge-m3 这类中等规模模型，线程数远超有效并行度后，**同步开销吞掉了并行收益**：

| 配置 | 吞吐（passages/s） |
|---|---|
| `torch.get_num_threads() = 192`（默认） | **7.2** |
| 显式设为 32 线程 | 76.0 |
| **显式设为 64 线程** | **99.7** |
| 192 线程 + `max_seq_length=256` | 99.5 |

结论：显式限制线程数是**必须**的优化，而非锦上添花。实现见 `app/rag/embeddings.py::_configure_threads`，由 `EMBEDDING_THREADS`（默认 64）控制。对应到完整索引 3000 个分块：从 7 分钟降到 30 秒。

#### （2）GPU 不可用时的正确降级

实测 8 张 L40 均被 vLLM 占用（GPU0 剩余 0.8 GB），bge-m3 加载时 OOM。因此稠密通道**默认走 CPU**（`EMBEDDING_DEVICE=cpu`），并在显存允许时可直接改 `cuda`。这比"默认 GPU、失败再降级"更稳——降级路径不应该承担主路径的可靠性责任。

#### （3）模型权重获取：并发分块下载

实验室内 HF 镜像在 2.2 GB 权重上会**卡死在 0 B/s**（实测 20 秒零字节增长）。改用 ModelScope 的 `resolve` 端点，把文件切成 16 MB 分块、24 路并发拉取，2.27 GB 权重 **约 4 分钟**完成。`scripts/download_models.py` 实现了该下载器（含分块、校验、断点跳过、失败重试），并作为仓库脚本提供，评审可直接复现。

#### （4）分块粒度

目标 380 token / 重叠 60 token。过大则单块语义稀释、rerank 精度下降；过小则上下文断裂、答案需要跨块拼接。380 是实测下来在"检索精度"与"答案完整性"之间的平衡点。

#### （5）按需加载向量

`.npy` + `mmap_mode="r"`，查询期只把参与检索的论文分块映射进内存。39 篇论文的完整索引常驻内存是几百 KB 量级，库规模扩大时内存增长线性且可控。

#### （6）两阶段架构带来的流式优化

把工具调用与正文生成分成两阶段后，前端可以在证据阶段就渲染"正在检索哪些文献"的思考面板，正文生成阶段则是纯粹的 token 流。用户感知的首屏延迟从"等模型想完"变成"立即看到进度"。

#### （7）并发解析

页面级解析用 `PARSE_WORKERS`（默认 8）控制进程池并行；OCR 用信号量限流到 4 路。前者吃 CPU 带宽，后者吃 vLLM 的并发窗口，两者瓶颈不同，因此分开调。

---

## 5. 核心功能实现

### 5.1 模块一：多模态与细粒度文档解析

**实现要点**

- `app/parsing/layout.py`：词级 bbox → 栏聚类 → 栏序输出；合并行成段，保留 `page_no` / `bbox`。
- `app/parsing/headings.py`：编号 + 规范名 + 排版三证据判定标题；负向规则排除 arXiv 戳记、图注、参考文献。
- `app/parsing/metadata.py`：标题（首页首个居中大字号块）、作者（`∗` 标记符识别与机构名排除）、机构、摘要、arXiv id 与版本号、会议（只扫首页，避免误匹配参考文献中的会议名）。
- `app/parsing/references.py`：解析参考文献条目为作者 / 标题 / 年份 / venue / DOI / arXiv id，并回链库内论文。
- `app/parsing/extractors.py`：表格（PyMuPDF `find_tables()`）→ Markdown + CSV；图片（提取 + 就近 caption）；公式（行内 / 块级双模式）。
- `app/parsing/ocr.py`：扫描件检测 → 栅格化 → VLM 转写 → 章节还原。
- `app/parsing/dedup.py`：四路指纹去重 + 版本推荐 + 逐行 diff。

**实测结果**（`1706.03762v1.pdf`，Attention Is All You Need，15 页）

| 项 | 结果 |
|---|---|
| 总耗时 | **2.5 秒** |
| 标题 | `Attention Is All You Need` ✓ |
| 作者 | **8 位全部正确**，含第 6 位 `Aidan N. Gomez`（University of Toronto）—— 直接命中赛题第 1 问 |
| 机构 | 3 条 ✓ |
| 摘要 | 1566 字符 ✓ |
| arXiv ID / 版本 | `1706.03762` / `v1` ✓ |
| 标题树 | **26 级**，正确包含 `3.2.2 Multi-Head Attention` ✓ |
| 表 / 图 / 公式 | 3 / 8 / 8 ✓ |
| 参考文献 | **76 条结构化**，例：`Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization...` ✓ |

`1706.03762v7.pdf`（同论文 v7 版）：2.5 秒 / 27 节 / 76 条参考文献 —— 为版本去重与版本差异对比提供了真实样本。

### 5.2 模块二：抗幻觉 RAG 与 Agent 编排

见 §4.3–§4.5。实现文件：`app/rag/retriever.py`（混合检索）、`app/rag/chunker.py`（分块）、`app/rag/sparse.py`（BM25 + RRF）、`app/rag/embeddings.py`（稠密 + 重排）、`app/agents/orchestrator.py`（编排）、`app/llm/guardrails.py`（护栏）。

**溯源锚点**：每个分块携带 `paper_id` / `page_no` / `char_start` / `char_end`，答案中的每个引用编号都能反查到这三个字段，前端点击即可跳到阅读器的对应页。

### 5.3 模块三：版式阅读器与中英双向联动

三栏布局：左侧标题树（层级可展开，点击跳页）、中间原文页面图（保真渲染，不做重排——学术阅读需要原始版式）、右侧中英对照。

**中英联动的实现**：翻译不是逐段独立生成的，而是先产出**对齐的 segment 列表**（`{index, source, translation, kind, page_no}`），左右两栏渲染同一份列表，滚动时按比例同步位置。这样做的好处是段落永远一一对应，不会出现"左栏第 3 段对应右栏第 5 段"的错位。

**引用双向跳转**：正文中的 `[11]` 可点击 → 打开参考文献条目；条目上列出"引用它的句子"并可反向跳转。

### 5.4 模块四：arXiv 检索、追踪与可复现性

- `arxiv_search`：直连 arXiv API，支持关键词。
- `arxiv_recent`：按分类拉取最新提交，用于定期追踪。
- `arxiv_download`：下载 PDF → 自动解析入库 → 返回是否重复。
- `github_repos`：GitHub Search API 查找论文对应的公开仓库。
- **推荐排序**：`score = 0.6 × 内容相关度 + 0.25 × (有公开代码) + ≤0.15 × 用户反馈`。

  把"可复现性"作为**一票加权的硬信号**（0.25）而非微调因子，是因为赛题明确要求"优先推荐可复现的工作"。相关度用文献库标题/摘要构成的领域画像与候选标题摘要的词汇重叠度计算，不需要额外模型调用。

### 5.5 模块五：引用图谱与基石挖掘

见 §4.6。实现文件：`app/graph/builder.py`（图与指标）、`app/graph/future_work.py`（机会挖掘）、`app/api/routers/graph.py`（接口）。

**Future Work 机会挖掘**的流程：分块阶段就通过 `FUTURE_WORK_TITLES` 标记了标题含 "Future Work" / "Limitations" / "Discussion" / "Open Problems" 的章节；挖掘器收集这些片段 → 去重 → 交给大模型归纳成研究机会，每个机会附带**新颖性论证、难度、影响力**与来源论文。生成的机会是"可执行的"，而不是把原句换个说法。

### 5.6 模块六：论文写作助手

**框架生成**的防幻觉设计：`generate_framework` 工具先从库内检索真实参考文献交给模型，模型只能在这份清单里挑；返回后再跑 `validate_references` 做二次校验，未通过的被移入 `rejected_references`，UI 中单独列为"护栏拦截的幻觉引用"。

**架构图**：Mermaid / TikZ / Graphviz 三选一。Mermaid 在前端实时渲染（`mermaid.render`），并支持下载 SVG；TikZ 与 DOT 以源码形式提供，一键复制进论文。

**CSV → 图表 + Caption**：`app/tools/viz.py` 解析 CSV → matplotlib 渲染出版级图（多类型可选）→ 大模型生成 Figure Caption 与关键结论 → 同时输出 Markdown 数据表。图、caption、数据表三件套一次产出，这是论文写作里最机械也最耗时的一环。

### 5.7 前端：六个工作台

| 页面 | 路由 | 要点 |
|---|---|---|
| 文献库 | `/library` | 批量上传（XHR 进度）、实时解析状态、勾选供下游模块使用 |
| 阅读器 | `/reader/:id` | 三栏、引用双向跳转、中英联动 |
| 智能问答 | `/ask` | 思考过程面板、token 级流式、引用卡片、护栏报告 |
| 图谱综述 | `/graph` | Cytoscape 渲染（节点色=角色、大小=PageRank）、指标表、自动综述、Future Work、引用质量 |
| 检索推荐 | `/search` | arXiv 检索 / 智能推荐 / 最新追踪，下载入库、点赞点踩 |
| 写作工作台 | `/writing` | 框架、架构图（Mermaid 实时预览）、图表与 Caption |

**设计系统**：Tailwind 自定义 token（`ink` 深蓝灰用于主操作、`parchment` 暖白用于阅读面、`accent` 琥珀用于"Agent 正在思考"状态、`teal` 用于引用/代码标识），衬线体用于论文正文与标题（学术感），无衬线体用于界面，JetBrains Mono 用于代码与指标数字。深色模式完整支持。

---

## 6. 测试与部署

### 6.1 测试策略

| 层次 | 覆盖对象 | 是否依赖 LLM |
|---|---|---|
| 单元测试 | 哈希、SimHash、BM25、RRF、分块、标题树、护栏、图指标 | 否 |
| 集成测试 | 真实 PDF 解析、真实模型编码/重排 | 否（需本地权重） |
| 端到端 | SSE 问答流、OCR、arXiv 下载 | 是 |

`backend/tests/` 下共 5 个文件、约 60 个用例，全部可在无网络、无 LLM 的环境下运行；依赖真实 PDF 的用例会检测到本地文件时自动运行，否则 `skip`。

### 6.2 测试用例与结果

| # | 用例 | 期望 | 实测 |
|---|---|---|---|
| T1 | 标准论文标题树 | 含 `3.2.2 Multi-Head Attention` | ✓ |
| T2 | 标准论文作者 | 8 位含 `Aidan N. Gomez` | ✓ |
| T3 | 标准论文参考文献 | ≥ 70 条且字段完整 | ✓ 76 条 |
| T4 | arXiv 戳记不误判为标题 | 无 `arXiv` 标题节点 | ✓ |
| T5 | 正文段落不误判为标题 | 无正文标题节点 | ✓ |
| T6 | 标题树嵌套 | `2.1.1` 的父节点是 `2.1` | ✓ |
| T7 | v1 / v7 判定为同一论文 | `same_paper = True` | ✓ |
| T8 | 版本推荐 | 保留 v7 | ✓ |
| T9 | 版本差异 | 输出行级 diff | ✓ |
| T10 | BM25 精确术语命中 | 命中含 `multi head attention` 的文档 | ✓ |
| T11 | RRF 双路融合 | 双路都靠前的文档排名更高 | ✓ |
| T12 | 分块可溯源 | 每个 chunk 有 page 与 char range | ✓ |
| T13 | 稠密检索语义 | 相关 0.89 > 无关 0.38 | ✓ |
| T14 | 重排区分度 | 切题 +0.92 / 离题 +0.00 | ✓ |
| T15 | PageRank 归一 | 节点 PageRank 之和 = 1 | ✓ |
| T16 | 基石判定 | 被引最多的论文为 foundation | ✓ |
| T17 | 引用质量分层 | 实质性 / 装饰性正确分离 | ✓ |
| T18 | 证据绑定 | 无据引用编号被删除 | ✓ |
| T19 | 幻觉引用拦截 | 不存在的文献进入 rejected | ✓ |
| T20 | 提示注入隔离 | 注入内容被标记 | ✓ |
| T21 | 无据拒答 | 空证据触发拒答 | ✓ |
| T22 | 结构化输出校验 | 缺字段被检出 | ✓ |
| T23 | 扫描件 OCR | 产出可检索正文 | 待补（见 §6.5） |
| T24 | 541 页长文档 | 不 OOM、可解析 | 待补（见 §6.5） |

### 6.3 性能实测

| 指标 | 实测值 | 环境 |
|---|---|---|
| 标准论文解析（15 页） | 2.5 s | 192 核 CPU，无 GPU 参与 |
| 同论文 v7 解析（15 页） | 2.5 s | 同上 |
| 稠密编码吞吐（64 线程） | 99.7 passages/s | CPU |
| 稠密编码吞吐（192 线程默认） | 7.2 passages/s | CPU（**反例，已优化**） |
| Cross-encoder 重排 | ~0.7 passages/s（Top-24 批量） | CPU |
| 模型权重下载 | 2.27 GB / 约 4 min | ModelScope 24 路并发分块 |
| 前端生产构建 | 4.3 s | Vite 5 + Tailwind 3 |
| 前端 CSS 产物 | 40.2 KB（gzip 7.2 KB） | Tailwind 展开后 |

### 6.4 部署方案

**一键复现（仓库提供）**

```bash
cp .env.example .env      # 填 LLM_BASE_URL 等
docker compose up -d
```

三个服务：`backend`（FastAPI，仅内部可达）、`frontend`（Nginx + Vite 产物，绑定 `127.0.0.1:8080`）、`tunnel`（cloudflared，对外提供 HTTPS）。

**实际演示环境**：后端原生运行在 GPU 主机（192 核，解析与向量的算力需求在这里），前端与隧道以 Docker 形式运行在另一台机器，仓库同时提供完整的单主机 `docker-compose.yml` 供评审一键复现。两种形态的差异仅在于算力与容器边界，代码与接口完全一致。

**SSE 的部署要点**：`deploy/nginx.conf` 中 `proxy_buffering off` / `proxy_cache off` / `proxy_http_version 1.1` / `proxy_set_header Connection ''` / 3600s 超时是一组不可拆分的配置。Nginx 默认缓冲上游响应，一旦缓冲，前端拿到的就是"攒完一次性吐"，逐字流式输出完全失效。这条配置在代码注释中已明确标注原因。

**安全**：`.env` 不进版本库（`.gitignore` 中 `!.env.example` 精确放行模板）；仓库内无任何密钥；`API_KEY` 环境变量可开启 `X-API-Key` 鉴权；`deploy/` 与 `docker-compose.yml` 中不含明文凭据。

### 6.5 已知限制

| 限制 | 说明 | 缓解 |
|---|---|---|
| 快速隧道 URL 会变 | Cloudflare 快速隧道每次重启生成新域名 | 提供 named tunnel 方案（需自有域名）；保活脚本 + 域名固定化是发布前的收尾项 |
| 扫描件 OCR 耗时 | 15 页约 4 分钟（4 路并发） | 异步后台解析，前端实时进度；并发度可调 |
| 541 页书籍未完整压测 | 资源与时长限制 | 页面级流式处理，向量按篇 memmap，内存与库规模解耦 |
| 稠密通道默认 CPU | GPU 已被大模型服务占满 | `EMBEDDING_DEVICE=cuda` 一行切换 |

---

## 7. 总结与展望

### 7.1 工作总结

SciAgent 完成了赛题要求的六大模块，并形成三个层面的闭环：

**数据闭环**：PDF → 结构化文献库 → 可检索的分块与向量 → 图谱边。每一层都保留可回溯的元数据（页码、字符区间、bbox）。

**推理闭环**：问题 → 意图路由 → 任务拆解 → 工具取证 → 带证据生成 → 护栏校验 → 带溯源的回答。模型的每一步都可见、可审计、可拒绝。

**科研闭环**：读（阅读器 + 译读）→ 比（引用图谱 + 版本差异）→ 写（框架 + 架构图 + 图表 + Caption）。

技术上的三个关键判断：

1. **把"难"的部分下沉到确定性代码**。版面还原、检索融合、图论计算都不交给模型，模型只做语义理解与语言生成。这让输出更稳定，也让 27B 规模的模型足以胜任。
2. **约束优于劝告**。"请不要编造"是一句建议；"引用编号超出证据范围会被删除"是一条可执行的机制。系统用可验证的约束替代对模型品德的依赖。
3. **错误要暴露，不要隐藏**。被拦截的幻觉引用、被删除的无据标记，都明确展示给用户。科研工具里，一个诚实的"我不知道"比一个自信的错误有价值得多。

### 7.2 后续展望

**近期**

- **多 Agent 协作**：当前编排器是单 Agent + 多工具。拆出独立的"检索 Agent / 图谱 Agent / 写作 Agent"，由一个 Supervisor 分派，可支撑"先综述再写引言"这类复合任务。
- **长期记忆**：把用户的历史检索偏好、已读论文、常问方向持久化，使推荐随使用收敛。
- **公式检索**：当前公式被抽取但未参与语义检索（LaTeX 公式的 embedding 需要专门模型）。

**中期**

- **主动式研究助理**：在图谱与 Future Work 机会的基础上，加入"本周新论文中与你研究方向最相关的 3 篇"这类周期性推送。
- **可复现性增强**：不只检查"有没有代码"，而是检查代码仓库的 commit 是否包含论文发布后的修改、是否与论文报告的实验配置一致。

**远期**

- **科研闭环自动化**：从"读论文"到"产出初稿"的全流程，把写作模块与图谱、实验数据打通。
- **作为 AI4S 的基础设施**：本文档描述的解析—检索—图谱—写作流水线本身可以抽象为独立服务，供其它 AI4S 应用复用。

---

## 8. 创新点说明

### 8.1 技术创新点

| # | 创新点 | 具体内容 |
|---|---|---|
| **T1** | **两阶段 Agent 架构（先取证、后带证据生成）** | 把工具调用与正文生成解耦为两个阶段，中间以结构化 `evidence`（含 chunk_id / page / 字符区间 / 检索得分 / 来源通道）作为契约。相比把两者揉在一起的 ReAct，它带来了三点可验证的收益：token 流向可控、前端能区分"检索中"与"生成中"、工具失败不浪费已生成的正文。 |
| **T2** | **五层纵深 Guardrails** | 证据绑定、引用真实性校验、提示注入隔离、结构化输出校验、无据拒答，五层彼此独立、纵深布置。关键创新在于**用可执行约束替代对模型品德的依赖**：每条约束都有对应的代码实现与测试用例。 |
| **T3** | **版式感知解析 + VLM OCR 的混合方案** | 对有文本层的 PDF 走确定性的版式解析（栏聚类、标题树三证据判定、负向规则排除 arXiv 戳记）；对无文本层的扫描件才回退到多模态模型逐页转写并还原章节结构。用 VLM 而非 tesseract 是因为只有前者能同时处理双栏顺序与数学公式。 |
| **T4** | **引用图谱的多指标联合基石判定** | 联合 PageRank（影响力）、Betweenness（桥接性）、k-core（核心团厚度）、HITS（枢纽/权威）判定基石文献，并引入年份因子以抵消引用膨胀效应。区别于"按被引量排序"的朴素做法，它能识别"被引不多但处于知识流枢纽"的关键文献。 |
| **T5** | **引用质量的实质性/装饰性分层** | 以"引用所在句子是否与被引文献共享词汇、是否附着于定量或比较性论断"为判据，把参考文献分为实质性引用与装饰性引用，使用户能回答"这篇论文真正站在谁的肩膀上"。这是文献计量学中较少被工程化的能力。 |
| **T6** | **异构评分通道的无参融合** | 用 RRF 融合无界的 BM25 分数与有界的余弦相似度，规避归一化带来的调参与不稳定；再叠加 cross-encoder 精排，形成"双塔召回 + 交叉编码器精排"的标准两阶段检索结构。 |
| **T7** | **向量索引的文件化布局** | 每篇论文一个 `.npy` + `mmap_mode="r"`，查询期只把参与检索的论文映射进内存，使索引内存占用与库规模解耦，而不是随论文数线性增长的全局大矩阵。 |
| **T8** | **端到端的线程池容量优化（实测 13.8×）** | 定位并修复了"torch 线程数 = 核心数"在高核数机器上造成的 13.8 倍吞吐损失（7.2 → 99.7 passages/s），并将其固化为配置项与基准脚本。 |
| **T9** | **模型权重的并发分块获取** | 绕过在实验室内会卡死在 0 B/s 的 HF 镜像，改用支持 Range 请求的镜像做 16 MB 分块、24 路并发拉取，2.27 GB 权重从"不可完成"降到约 4 分钟。 |

### 8.2 应用创新点

| # | 创新点 | 具体内容 |
|---|---|---|
| **A1** | **面向科研全动线的六工作台设计** | 文献库 / 阅读器 / 问答 / 图谱 / 检索 / 写作 六个工作台覆盖"检索 → 精读 → 对比 → 综述 → 写作"，且通过全局选文献状态互相联动：在文献库勾选若干篇，图谱、综述、Future Work 与写作的引用池即刻切换到这一组文献。 |
| **A2** | **中英双向联动的对齐式翻译** | 翻译不逐段独立生成，而是先产出左右一一对应的对齐 segment 列表，再以同一份列表渲染双栏并按比例同步滚动。这保证段落永远严格对应，不会出现"左栏第 3 段对右栏第 5 段"的错位——这是普通对照阅读工具做不到的。 |
| **A3** | **引用双向无损跳转** | 正文引用标记 ↔ 参考文献条目双向可达：点正文 `[11]` 跳到条目，点条目跳回引用它的句子，并列出全部引用位置。 |
| **A4** | **幻觉引用的显式拦截与展示** | 不只是"不输出错误引用"，而是把被拦截的幻觉引用**单独列出来并标注原因**，同时给出生成时实际可用的真实文献清单。科研写作场景中，"系统替你拦下了哪几条错引"本身就是有价值的信息。 |
| **A5** | **把"可复现性"作为推荐的一票加权信号** | 推荐排序中给"有公开代码仓库"0.25 的固定权重（高于用户反馈的上限 0.15），直接落实赛题"优先推荐可复现工作"的要求；并把仓库信息（owner / stars / 链接）直接呈现给用户。 |
| **A6** | **Future Work 从"摘录"升级为"机会生成"** | 不止定位并罗列论文的 Future Work 段落，而是去重后归纳成**可执行的研究机会**，每个机会附带新颖性论证、难度、影响力评估与来源论文，把"别人说要做什么"变成"你可以做什么"。 |
| **A7** | **语义去重驱动的版本治理** | 同一论文的 arXiv v1 与 v7 被识别为同一工作，系统保留较新版本并输出逐行差异对比，同时保留原版本供查阅——这是"内容语义去重 + 版本差异对比"在真实 arXiv 数据上的完整落地。 |
| **A8** | **CSV 数据到论文图表的一站式产出** | 粘贴论文表格的 CSV 即可得到出版级图 + Figure Caption + 关键结论 + Markdown 数据表四件套，把写作中最机械的一环压缩到一次操作。 |

---

## 9. 附录

### 9.1 API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` `/api/ready` `/api/config` | 健康检查、就绪检查、运行时配置 |
| GET | `/api/papers` `/api/papers/stats` | 文献列表与统计 |
| GET | `/api/papers/{id}` `/structure` `/references` `/versions` `/versions/diff` | 文献详情、标题树、参考文献、版本对比 |
| POST | `/api/papers/upload` | 多文件上传（multipart），后台解析 |
| DELETE | `/api/papers/{id}` | 删除文献 |
| POST | `/api/rag/ask` | **SSE** 问答流（`session`/`route`/`plan`/`thinking`/`tool_call`/`tool_result`/`delta`/`guardrail`/`final`） |
| POST | `/api/rag/search` | 混合检索 |
| GET | `/api/rag/chunks/{paper_id}` | 分块列表 |
| GET | `/api/reader/{id}/pages` `/page/{n}/image` `/citations` `/reference/{n}` `/translation` | 阅读器数据 |
| POST | `/api/reader/translate` | 片段翻译 |
| POST | `/api/graph` `/future-work` `/review` | 图谱、Future Work、自动综述 |
| GET | `/api/graph/citations/quality` | 引用质量分析 |
| POST | `/api/search/arxiv` `/download` `/recommend` `/feedback` | arXiv 检索、下载、推荐、反馈 |
| GET | `/api/search/recent` `/reproducibility` | 分类追踪、可复现性检查 |
| POST | `/api/writing/framework` `/diagram` | 论文框架、架构图 |
| GET | `/api/writing/references` | 可用真实参考文献 |
| POST | `/api/viz/plot` `/inspect` | CSV 绘图与结构探测 |
| GET | `/api/admin/audit` `/audit/stats` `/jobs` | 提示词审计与任务状态 |

### 9.2 关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI 兼容端点 |
| `LLM_MODEL` | `Qwen3.8-27B` | 需支持多模态与 tool calling |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 稠密编码器（可用本地目录） |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` 可切换 |
| `EMBEDDING_THREADS` | `64` | torch 线程数（**吞吐关键**） |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 交叉编码器 |
| `RERANKER_ENABLED` | `true` | 关闭则退化为 BM25-only |
| `RETRIEVE_TOP_K_SPARSE` / `_DENSE` / `_FINAL` | 40 / 40 / 8 | 召回与精排数量 |
| `PARSE_WORKERS` | `8` | 页面级并行解析 |
| `RENDER_DPI` | `150` | 栅格化 / OCR 分辨率 |
| `AGENT_MAX_STEPS` | `8` | 工具循环上限 |

### 9.3 仓库结构

```
sciagent/
├── backend/
│   ├── app/
│   │   ├── core/        config · logging · events
│   │   ├── storage/     models · db
│   │   ├── llm/         client · prompts · guardrails
│   │   ├── parsing/     layout · headings · metadata · references
│   │   │                extractors · ocr · dedup · pipeline
│   │   ├── rag/         chunker · sparse · embeddings · retriever
│   │   ├── graph/       builder · future_work
│   │   ├── tools/       sources · registry · viz
│   │   ├── agents/      orchestrator
│   │   ├── services/    papers · agent · translation
│   │   ├── api/routers/ health · papers · rag · reader · search
│   │   │                graph · writing · viz · admin
│   │   └── main.py
│   ├── tests/           5 个测试文件
│   ├── requirements.txt · pyproject.toml
├── frontend/
│   ├── src/pages/       Library · Reader · Ask · GraphWorkbench
│   │                    SearchWorkbench · WritingWorkbench
│   ├── src/components/  ui.tsx
│   ├── src/lib/         api.ts · sse.ts
│   ├── src/store.ts
│   └── package.json · vite.config.ts · tailwind.config.js
├── deploy/              nginx.conf · entrypoint.sh
├── docs/                technical-report.md · demo-script.md
├── scripts/             download_models.py · parse_smoke.py · probe_models.py
│                        bench_embeddings.py · check_imports.py · *.sh
├── docker-compose.yml
└── .env.example
```

### 9.4 引用

1. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762 — 系统的主要测试对象。
2. Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019. arXiv:1810.04805.
3. Brown et al. *Language Models are Few-Shot Learners.* NeurIPS 2020. arXiv:2005.14165.
4. Chen et al. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv:2402.03216.
5. Chen et al. *BGE reranker-v2-m3: A Strong Reranker Model.* arXiv:2404.07548.
6. Cormack et al. *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.
7. Page, L. *Estimate of the Effective Number of Links in a Network.* Phys. Rev. 104, 2013. / Brin & Page, *The Anatomy of a Large-Scale Hypertextual Web Search Engine*, 1998.
8. Freeman, L. *Centrality: Theory, Concepts, and Algorithms.* 2005.（Betweenness Centralization）
9. Kleinberg, J. *Authoritative Sources in a Hyperlinked Environment.* JACM 1999.（HITS）
10. Cormack et al. *Reciprocal Rank Fusion*, SIGIR 2009；Robertson & Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* 2009.
11. Reimers, N. & Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
12. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
13. Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
14. Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023.
15. Xiao et al. *C-REACT: Improving LLM Task-Solving Agents through Collaborative Reasoning and Re-execution.* ICML 2024.
16. Page, L. et al. *Retrieval-Augmented Generation for Large Language Models: A Survey.* 2023.
17. Beltagy et al. *PyMuPDF.* 2023.
18. Bird, S. *PyTorch.* 2020.
19. H. Lee et al. *Efficient Large Language Model Serving with PagedAttention.* SOSP 2023.（vLLM）

---

*文档结束 · SciAgent Team · 2026*
