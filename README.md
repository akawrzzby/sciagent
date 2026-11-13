# SciAgent · 科研小助手

> **智领科研 · 意图驱动** —— 基于大模型与 Agent 架构的全流程科研助手
>
> 面向科研文献阅读与写作的端到端 Agent 系统：多模态解析 → 混合检索溯源 → 自主工具编排 → 图谱挖掘 → 学术写作。

[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/frontend-React%2018-61DAFB?style=flat-square&logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/lang-TypeScript-3178C6?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/lang-Python%203.11-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](#-license)

### 实测数据（全部可复跑，脚本见 `scripts/`）

| 指标 | 实测值 | 出处 |
|---|---|---|
| 回答忠实度 faithfulness | **0.7092**（三轮均值，区间 0.6378 – 0.7521） | `scripts/eval_rag.py` |
| 引用可溯源率 traceability | **1.0000**（三轮全满） | `scripts/eval_rag.py` |
| 库外问题诚实拒答率 | **3 / 3**，编造 0 | `scripts/eval_rag_control.py` |
| 元数据感知检索 hit@5 | **0.9286 → 1.0000**（A/B，各自重建索引） | `scripts/eval_retrieval_ab.sh` |
| 自动化测试 | **178 passed**（无网络、无 LLM） | `backend/tests/` |
| Docker 一键构建 | 后端镜像 **4 min / 1.19 GB**，前端 **102 MB** | `scripts/ci_docker_test.sh` |
| 541 页书籍解析 | **149.4 s** / 1254 章节 / 131 万字符 | §6.5 |
| 稠密编码吞吐 | **99.7 passages/s**（64 线程；192 线程默认仅 7.2） | §6.5 |

> 指标口径、量具的五次修复过程、以及"单篇题接近满分、跨论文对比题只有 0.31–0.56"这一能力边界，见 [`docs/technical-report.md`](./docs/technical-report.md) §6.3。

---

## 📖 目录

- [一、项目介绍](#一项目介绍)
- [二、技术架构](#二技术架构)
- [三、六大核心模块](#三六大核心模块)
- [四、快速开始](#四快速开始)
- [五、在线演示](#五在线演示)
- [六、配置说明](#六配置说明)
- [七、项目结构](#七项目结构)
- [八、测试](#八测试)
- [九、团队信息](#九团队信息)
- [十、License](#十license)

---

## 一、项目介绍

### 1.1 痛点

科研工作者在读文献时反复面对同一组摩擦：

| 痛点 | 现状 | SciAgent 的做法 |
|---|---|---|
| 扫描件与双栏 PDF 读不懂 | 复制粘贴出来是乱序文本 | 版式感知解析 + VLM 逐页转写 |
| 答案不敢信 | 大模型张口就编造参考文献 | 证据绑定 + 引用真实性校验 + 无据拒答 |
| 读过的文献散落各处 | PDF、笔记、表格各在一处 | 统一文献库 + 版式阅读器 + 引用双向跳转 |
| 不知道该做什么 | 读完一堆论文仍不清楚空白 | 引用图谱基石挖掘 + Future Work 机会生成 |
| 写论文从零开始 | 框架、图、表格全靠手工 | 论文框架 / TikZ / CSV → 图表 + Caption 一站式 |

### 1.2 系统定位

SciAgent 不是一个"聊天框套壳"，而是一条**可溯源的科研流水线**：

```
PDF ──▶ 多模态解析 ──▶ 结构化文献库 ──┬──▶ 混合检索 (BM25 + Dense + Rerank)
                                        │         │
                                        │         ▼
                                        │    证据锚点 (chunk_id / page / char range)
                                        │         │
                                        │         ▼
                                        └──▶ Agent 编排 ──▶ 带引用的流式回答
                                                 │
        ┌────────────────────────────────────────┼────────────────────────┐
        ▼                    ▼                   ▼                    ▼
   引用图谱 (PageRank)   arXiv 工具链        论文框架            图表 + Caption
```

### 1.3 设计原则

1. **证据优先**：任何一句结论都必须能指回原文的**字符区间**；指不回去就不说。
2. **结构化解析优先于纯文本**：标题树、表格、公式、参考文献各自建模，而不是拍平成一坨字符串。
3. **可降级**：稠密通道挂了退化为 BM25；LLM 挂了给出明确错误而不是白屏。
4. **全链路可审计**：每一次模型调用的 system / user prompt、工具参数、耗时、token 都落盘。

---

## 二、技术架构

### 2.1 总体分层

```
┌──────────────────────────────────────────────────────────────────────────┐
│  展现层  React 18 + TypeScript + Vite + Tailwind + Cytoscape + Mermaid  │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┐                 │
│  │ 文献库 │ 阅读器 │问答    │图谱    │检索    │写作    │  6 个工作台      │
│  └────────┴────────┴────────┴────────┴────────┴────────┘                 │
│         SSE 事件流（token 级 delta / thinking / tool_call）             │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ REST + Server-Sent Events
┌────────────────────────────────┴─────────────────────────────────────────┐
│  编排层  agents/orchestrator.py                                         │
│  ┌────────┐   ┌─────────┐   ┌──────────┐   ┌────────────┐              │
│  │ Router │──▶│ Planner │──▶│ Executor │──▶│ Reflection │              │
│  │ 意图路由│   │任务 DAG │   │ReAct 循环│   │护栏 & 引用 │              │
│  └────────┘   └─────────┘   └──────────┘   └────────────┘              │
│         tools/registry.py —— 原生 Function Calling 工具集             │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────┴─────────────────────────────────────────┐
│  能力层                                                                 │
│  parsing/ 版式解析 · VLM OCR · 语义去重      rag/ 分块 · BM25 · Dense  │
│  graph/   PageRank · 基石挖掘 · Future Work   tools/ arXiv · GitHub · 绘图│
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────┴─────────────────────────────────────────┐
│  存储层  SQLite (papers / sections / chunks / references / feedback)     │
│          向量索引 .npy (每篇一个 memmap 矩阵，查询期按需加载)            │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────┴─────────────────────────────────────────┐
│  模型层  vLLM → Qwen3.8-27B (多模态)   sentence-transformers → bge-m3     │
│                                    → bge-reranker-v2-m3 (cross-encoder) │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent 编排：两阶段设计

传统 ReAct 把"检索"和"生成"揉在一个循环里，导致模型要先想再搜，token 不可控、流式体验差。SciAgent 采用**两阶段**：

```
                    ┌─────────────── 阶段 A：取证 ───────────────┐
用户问题 ──▶ Router │ 意图分类 → Planner 任务 DAG → Executor 工具循环 │
           │       │ 产出：evidence[] （带 chunk_id/page/char_range）│
           │       └──────────────────────┬─────────────────────────┘
           │                              ▼
           │       ┌───────────── 阶段 B：带证据生成 ─────────────┐
           └──────▶│  System + Evidence → token 级流式输出        │
                   │  Reflection：引用校验 → 护栏 → 无据拒答       │
                   └──────────────────────────────────────────────┘
```

- **阶段 A** 的产出是结构化的 `evidence`，可在 UI 的"思考过程"面板中逐条审计。
- **阶段 B** 只允许引用 `evidence` 中出现的条目；`guardrails.py` 会在输出后再次做一次引用真实性校验，与库内真实文献做 fuzzy 匹配，不匹配即剔除。

### 2.3 混合检索

```
        query
          │
    ┌─────┴─────┐
    │           │
 BM25 (稀疏)  bge-m3 (稠密, 1024-dim)
    │           │
    └─────┬─────┘
     RRF 融合 (k=60)
          │
   bge-reranker-v2-m3 (cross-encoder, Top-24)
          │
     Top-K 证据 + 溯源锚点
```

- **为什么混合**：BM25 擅长精确术语与缩写（"multi-head attention"、"BLEU"），Dense 擅长语义近似与中英跨语言；二者互补。
- **为什么再 rerank**：cross-encoder 直接看 (query, passage) 交互，比双塔模型的余弦相似度准得多，是召回后精排的标准做法。
- **RRF** 而非加权求和：两路分数量纲不同（BM25 是无界和，cosine 是 [0,1]），RRF 只用排名，天然免调参。

### 2.4 关键工程决策

| 决策 | 理由 |
|---|---|
| 自研轻量编排器，不用 LangChain | 需要精确控制"取证/生成"两阶段的 token 流向与 SSE 事件类型；框架反而是黑盒 |
| 扫描件用 VLM 转写而非 tesseract | 实测 Qwen3.8-27B 能准确读出图内文字与双栏顺序；tesseract 在公式与双栏上表现差 |
| 向量按篇存 `.npy` memmap | 一次查询只加载参与检索的论文，内存与库规模解耦 |
| SQLite + aiosqlite | 单文件、零运维；异步驱动不阻塞 SSE |
| 前端同源 `/api`（单进程托管或 Nginx 反代） | API 前缀不可拆分，同源可避免跨域与"前后端两个公网 URL"的不一致 |
| 稠密检索走 CPU + **64 线程** | GPU 已被 vLLM 占满；实测 192 线程仅 7.2 passages/s，**64 线程达 99.7 passages/s**（14×），线程过度调度反而压倒收益 |
| 解析 CPU 密集段下沉线程池 | PyMuPDF 抽取与渲染是同步阻塞调用，放在事件循环里会卡死 SSE 心跳与全部请求 |
| **表格提取强制串行** | 实测 `Page.find_tables()` 存在全局状态竞争：同文档并行只出 **1** 张表，串行出 **5** 张——并行会静默丢表 |
| PyMuPDF Document **每线程独立句柄** | `Document` 不可跨线程共享；用 `doc.name` 取绝对路径后按线程重开，杜绝数据竞争 |

### 2.5 两个真实缺陷的复盘

> 这两处都不是"一开始就想对了"，而是实测撞墙后定位根因再修。放在 README 里，是因为它们最能说明工程能力——**知道边界在哪，比知道怎么写更要紧**。

**① 外键约束与 Unit of Work 的插入顺序**

症状：`/api/papers` 恒为 0，单篇入库报 `sqlite3.IntegrityError: FOREIGN KEY constraint failed`（`INSERT INTO content_blocks`，479 行）。

根因（用 `engine.echo=True` 打出真实 SQL 才定位到）：SQLAlchemy Unit of Work **只按 ORM `relationship()` 推导插入顺序，不认表级 `ForeignKey` 约束**。`ContentBlock` / `PaperTable` / `PaperFigure` / `PaperFormula` 只有 `ForeignKey("papers.id")` 而没有 relationship，被当作互不相关的独立表，其 INSERT 被排到父行之前 → 外键检查失败。

修复（两层）：为四张子表补 `relationship(back_populates=...)`；`_persist` 改传 `paper=paper` 对象并在挂子行前 `await db.flush()` 先落盘父行。

验证：`scripts/diag_fk.py` 五变体全通过（裸 FK / 父行先 flush / 仅带 relationship 的模型 / 赋值 relationship 对象 / 完整修复路径）。

**② 表格提取的线程安全边界**

把页面解析整体并行化后，表格数从 53 掉到 13。`scripts/check_table_threads.py` 复现出决定性证据：同一文档**串行提取得 5 张表，并行只 1 张**——`Page.find_tables()` 存在全局状态竞争。

结论：**表格提取必须串行**，移出并行区；其余抽取与渲染才下沉线程池。同时 PyMuPDF `Document` 改为每线程独立句柄（`doc.name` 取绝对路径后按线程重开）。修复后表格数回到 53。

---

## 三、六大核心模块

### 模块 1 · 多模态与细粒度文档解析 `backend/app/parsing/`

| 能力 | 实现 |
|---|---|
| 版式感知分块 | PyMuPDF 词级 bbox + 栏检测，合并行成段，保留 `page_no` / `bbox` |
| 标题树 | 字体/字号/编号模式（`3.2.2`）联合判定，支持 26 级层级 |
| 双栏还原 | 按栏 x 区间切分，左右栏交替读入，避免串行 |
| 表格 | PyMuPDF `find_tables()` → Markdown + CSV 双输出 |
| 图片 | 提取 `image_path` + `caption`（就近的 `Figure N:` 文本） |
| 公式 | 行内 `$...$` 与块级 `$$...$$` 双模式 + 独立公式块 |
| 参考文献 | 解析作者/标题/年份/venue/DOI/arXiv id，并回链库内论文 |
| 扫描件 OCR | 检测到无文本层 → 逐页 150 DPI 栅格化 → VLM 转写（4 路并发） |
| 语义去重 | 标题 + 摘要 embedding 余弦 > 阈值 → 判为同论文不同版本 |

> **实测**：`1706.03762v1.pdf`（Attention Is All You Need）15 页 **2.5 秒**解析完成，标题、8 位作者、机构、1566 字摘要、26 级标题树（含 `3.2.2 Multi-Head Attention`）、3 表 / 8 图 / 8 公式 / **76 条结构化参考文献** 全部正确。
>
> **长文档与扫描件实测**：
> - 赛题指定的 **541 页书籍**：`UnderstandingDeepLearning_02_09_26_C.pdf` → **149.4 秒**完成，产出 **1254** 个章节节点、1 313 060 字符、**726** 张图、**1472** 个公式，无 OOM、无告警。
> - **纯扫描件** `1706.03762v1_img.pdf`（文本层 0 字符）→ **191.2 秒**，VLM 转写后重建出 **26 级标题树**，`3.2.2 Multi-Head Attention` 层级判定为 level 3 ✓，并解析出 **38 条**结构化参考文献。
> - **全量文献库 13 篇**（含扫描件）→ 433 页 / 2151 片段 / 748 参考文献 / 53 表 / 212 图 / 380 公式；文档级并发入库约 **6 分钟**（串行需 10 分钟以上）。

### 模块 2 · 抗幻觉 RAG 与 Agent 编排 `backend/app/rag/` `backend/app/agents/`

- 混合检索 + RRF 融合 + cross-encoder 精排（见 §2.3）
- 溯源锚点：每个 chunk 带 `paper_id / page_no / char_start / char_end`，UI 可一键跳转原文
- Agent：Router → Planner → Executor → Reflection（见 §2.2）
- Guardrails 五件套：
  1. **证据绑定** —— 只允许引用检索到的 chunk；
  2. **引用真实性校验** —— 与库内真实文献 fuzzy 匹配，剔除幻觉引用；
  3. **提示注入隔离** —— 检索到的正文被包裹在带边界标记的证据块中，指令与数据分离；
  4. **结构化输出校验** —— JSON schema 校验失败自动重试一次，仍失败则降级；
  5. **无据拒答** —— 证据不足时明确回答"文献库中没有依据"，而不是编。

### 模块 3 · 版式阅读器与中英双向联动 `frontend/src/pages/Reader.tsx`

- 三栏：标题树 / 原版页面（保真渲染）/ 中英对照双栏
- **双向引用跳转**：正文 `[11]` → 参考文献条目；条目 → 引用它的句子
- **中英联动滚动**：左右两栏基于同一份对齐 segment 列表，滚动百分比同步
- 逐段翻译缓存，`?target_lang=` 支持中英互译

### 模块 4 · arXiv 检索、追踪与可复现性 `backend/app/tools/sources.py`

- arXiv API 检索 / 按分类追踪最新提交 / PDF 一键下载入库
- **可复现性加权**：GitHub Search API 查找论文对应的公开仓库，有代码的论文在推荐中加权（赛题明确要求优先推荐可复现工作）
- 用户 thumbs up / down 反馈参与排序，形成闭环

### 模块 5 · 引用图谱与基石挖掘 `backend/app/graph/`

- 图论指标：**PageRank**（影响力）、**Betweenness**（桥接作用）、**k-core**（凝聚力）、**HITS**（枢纽/权威）
- **基石节点判定**：综合入度、PageRank、k-core 与年份，区分 `foundation / core / derivative`
- **引用质量分析**：区分"实质性引用"（真正支撑论点，常为方法或对比基准）与"装饰性引用"（顺带提及）
- **自动综述**：基于图指标 + 基石文献摘要生成结构化综述（核心发现 / 基石 / 衍生 / 开放问题）
- **Future Work 机会挖掘**：定位各论文的 Future Work / Limitations 片段 → 去重 → 生成含新颖性论证、难度与影响力的研究机会

### 模块 6 · 论文写作助手 `backend/app/api/routers/writing.py` `backend/app/tools/viz.py`

- **论文框架**：标题 / 摘要 / 引言 / 相关工作 / 方法 / 实验 + 参考文献清单；**参考文献只从库内检索**，模型虚构的会被护栏拦截并单独列在"已拦截的幻觉引用"中
- **架构图**：Mermaid / TikZ / Graphviz 三选一，Mermaid 前端实时渲染，TikZ/DOT 一键复制
- **CSV → 学术图表**：多种图表类型，输出 PNG + 出版级 Figure Caption + 关键结论 + Markdown 数据表

---

## 四、快速开始

### 4.1 前置条件

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | ≥ 3.11 | 后端 |
| Node.js | ≥ 20 | 前端构建 |
| vLLM 服务 | OpenAI 兼容 | 提供 `Qwen3.8-27B`（多模态，含 tool calling） |

> 没有自建 vLLM 也可以：任何 OpenAI 兼容端点（OpenAI / DeepSeek / Qwen API）都能通过 `LLM_BASE_URL` 接入，只要该模型支持多模态与 function calling。

### 4.2 一键 Docker（推荐用于评审复现）

```bash
git clone https://github.com/akawrzzby/sciagent.git
cd sciagent

cp .env.example .env
# 编辑 .env，至少确认 LLM_BASE_URL 指向你的大模型服务

docker compose up -d
docker compose ps          # backend / frontend / tunnel 三个服务
```

访问：`http://localhost:8080`（前端已把 `/api` 反代到后端，**单一入口**）。

**本机实测（Windows + Docker Desktop 29.9.3）**

| 项 | 实测 |
|---|---|
| 后端镜像构建 | apt 28.4 s + pip 190.3 s + 导出 13.6 s ≈ **4 min**（换源前同一构建卡死 15 分钟以上） |
| 镜像体积 | 后端 **1.19 GB**（site-packages 521 MB）/ 前端 **102 MB** |
| 容器内测试 | **171 passed / 7 skipped / 1.49 s**（7 个跳过是镜像内没有真实 PDF 与 bge-m3 权重） |
| 全栈 | backend / frontend 均 healthy；`/`、`/ask`、`/api/health` 全部 200 |
| 缺依赖降级 | `llm.ok=false`、`dense_available=false` 时自动退回 BM25，检索不崩 |

镜像默认走**清华 apt 源与 PyPI 源**（官方源在境内约 355 KB/s，会把构建卡死），可按 `.env` 切回：

```bash
DEBIAN_MIRROR=https://deb.debian.org/debian
DEBIAN_SECURITY_MIRROR=https://deb.debian.org/debian-security
```

**验证镜像内容与测试（推荐在提交前跑一次）**

```bash
bash scripts/ci_docker_test.sh
```

该脚本会构建镜像 → **把镜像内代码与工作树 `diff` 一遍**（不一致直接失败）→ 在镜像内跑完整测试套件。原因见技术文档 §6.6 记录的"镜像 ≠ 源码"事故：两个并发构建竞态会让**先启动、晚结束**的那次构建用旧上下文覆盖 `latest` 标签，而"构建成功"并不代表镜像等于当前源码。

### 4.3 本地开发

**后端**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 拉取稠密检索模型（国内网络建议走 ModelScope 镜像，见 scripts/download_models.py）
python ../scripts/download_models.py BAAI/bge-m3 ~/models/bge-m3 --workers 16
python ../scripts/download_models.py BAAI/bge-reranker-v2-m3 ~/models/bge-reranker-v2-m3 --workers 16

export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=Qwen3.8-27B
export EMBEDDING_MODEL=~/models/bge-m3
export RERANKER_MODEL=~/models/bge-reranker-v2-m3

uvicorn app.main:app --host 0.0.0.0 --port 9000
```

健康检查：`curl http://127.0.0.1:9000/api/health`

**前端**

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173，/api 已代理到 9000
```

**生产构建**

```bash
cd frontend && npm run build   # 产物在 frontend/dist
```

### 4.4 首次使用流程

1. 打开 **文献库** → 上传 PDF（支持多文件批量）→ 后台自动解析
2. 解析完成后进入 **阅读器**：看标题树、原文版式、双语对照
3. 进入 **智能问答**：提问，观察"思考过程"里的工具调用与证据，再看带引用的答案
4. 在文献库勾选多篇 → **图谱综述** 看引用拓扑与基石节点，生成自动综述与 Future Work
5. **检索推荐** 拉新论文入库；**写作工作台** 生成框架、架构图与图表

---

## 五、在线演示

**演示地址：<https://finished-editing-shore-mrna.trycloudflare.com>**

> 完整演示分镜脚本、逐步操作指引与评审账号说明见 **`docs/demo-script.md`**。

技术方案：后端 FastAPI 直接托管前端构建产物（`STATIC_DIR`），`/api` 与 SPA 同源；对外由 **Cloudflare Tunnel** 暴露单一 HTTPS 地址，因此在 NAT 环境（无公网 IP）下也能提供公网可访问的演示地址。

**当前部署形态**：全部收敛到 **单台 179 服务器单进程同源托管**（FastAPI :9000 同时提供 `/api/*` 与前端 SPA），cloudflared quick tunnel 出公网。仓库同时保留完整 Docker Compose 编排，供评审在有 Docker 权限的机器上一键复现。

**一键复现**（仓库已提供完整的容器化编排，评审可在本机完整拉起）：

```bash
cp .env.example .env          # 填入 LLM_BASE_URL / LLM_MODEL 等
docker compose up -d --build
# 前端 http://localhost:8080 ，API http://localhost:8000
```

**单机部署**（无 Docker 环境时）：后端 `bash scripts/run_backend.sh 9000`，前端 `npm ci && npm run build` 后把 `dist/` 交给 `STATIC_DIR`。

**隧道保活**：`scripts/start_tunnel.sh 9000` 会启动 cloudflared 并把公网地址写入 `~/sciagent/tunnel.url`；脚本内置 supervisor 循环，链路断开后自动重连。

> ℹ️ **关于演示地址**：当前采用 Cloudflare **quick tunnel**（随机域名），这是团队 2026-11-12 的**最终决策**——公网演示只需要一个可访问的 HTTPS 入口，quick tunnel 无需账号与自有域名即可在 NAT 环境一键出网。若评审期内需要地址固定，仓库已预留 `deploy/nginx.edge.conf.template` 与 `scripts/deploy_edge.sh`，可平滑切换到 **named tunnel**；若后端进程重启导致隧道重建，请重新执行 `scripts/start_tunnel.sh` 并从 `~/sciagent/tunnel.url` 取新地址。

---

## 六、配置说明

完整的环境变量模板见 [`.env.example`](./.env.example)。**`.env` 不进版本库，密钥一律通过环境变量注入。**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI 兼容端点 |
| `LLM_MODEL` | `Qwen3.8-27B` | 模型名，需支持多模态与 tool calling |
| `LLM_API_KEY` | 空 | 端点需要鉴权时填写 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 稠密编码器，可用本地目录路径 |
| `EMBEDDING_DEVICE` | `cpu` | 改 `cuda` 可用 GPU（需 torch+CUDA） |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 交叉编码器精排 |
| `RERANKER_ENABLED` | `true` | 关掉则退化为 BM25-only |
| `RETRIEVE_TOP_K_SPARSE` / `_DENSE` / `_FINAL` | 40 / 40 / 8 | 召回与精排数量 |
| `PARSE_WORKERS` | `8` | 页面级并行解析数 |
| `RENDER_DPI` | `150` | 栅格化 / OCR 分辨率 |
| `AGENT_MAX_STEPS` | `8` | Agent 工具循环上限 |
| `DATA_DIR` / `INDEX_DIR` | `./data` | 存储与向量索引目录 |
| `API_KEY` | 空 | 设置后需带 `X-API-Key` 头 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 模型镜像 |

---

## 七、项目结构

```
sciagent/
├── backend/
│   ├── app/
│   │   ├── core/        config · logging · SSE 事件总线
│   │   ├── storage/     SQLAlchemy 模型 · 异步会话
│   │   ├── llm/         OpenAI 兼容客户端 · prompts · guardrails
│   │   ├── parsing/     layout · headings · metadata · references
│   │   │                extractors · ocr · dedup · pipeline
│   │   ├── rag/         chunker · sparse(BM25) · embeddings · retriever
│   │   ├── graph/       builder(PageRank等) · future_work
│   │   ├── tools/       sources(arXiv/GitHub) · registry · viz
│   │   ├── agents/      orchestrator
│   │   ├── services/    papers · agent · translation
│   │   ├── api/routers/ health · papers · rag · reader · search
│   │   │                graph · writing · viz · admin
│   │   └── main.py
│   ├── tests/           pytest 用例（解析 / 检索 / 图谱 / 护栏）
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── pages/       Library · Reader · Ask · GraphWorkbench
│   │   │                SearchWorkbench · WritingWorkbench
│   │   ├── components/  ui.tsx（共享原子组件）
│   │   ├── lib/         api.ts（类型化客户端）· sse.ts（事件流）
│   │   └── store.ts     zustand 全局状态
│   ├── package.json
│   └── vite.config.ts
├── deploy/              nginx.conf · nginx.edge.conf.template · entrypoint.sh
├── docs/                technical-report.md · technical-report.pdf · demo-script.md
├── scripts/             部署(run_backend/restart_backend/start_tunnel) · 灌库(seed_*) ·
│                        模型下载(download_models/parallel_download) · 解析(parse_smoke) ·
│                        诊断(diag_fk/check_table_threads/db_inspect) · 文档(md2pdf)
├── docker-compose.yml
└── .env.example
```

---

## 八、测试

```bash
cd backend
pip install pytest pytest-asyncio
pytest -v tests/
```

覆盖范围：

- 解析：标题树层级、双栏顺序、表格结构、参考文献字段完整性、arXiv 戳记误判回归
- 去重：同论文不同版本（v1 vs v7）判为重复
- 检索：BM25 通道、Dense 通道、RRF 融合、rerank 精排
- 护栏：证据绑定、幻觉引用剔除、无据拒答
- 图谱：PageRank 归一性、k-core 单调性、基石节点判定
- 扫描件：OCR Markdown 重建标题树与元数据（`test_ocr_structure.py`）

> **实测：`pytest tests/ -q` → 178 passed in 30.3s**（14 个测试文件，178 个用例）。测试全程离线（conftest 已设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`），不依赖网络。

**在 Docker 镜像里跑同一套测试**（同时校验镜像内容与工作树一致，见 §4.2）：

```bash
bash scripts/ci_docker_test.sh
```

---

## 九、团队信息

> **待补充**（占位符，提交前请替换为真实信息）。

| 项目 | 内容 |
|---|---|
| 团队名称 | SciAgent Team |
| 学校 / 学院 | XX 大学 · XX 学院 |
| 专业 | XX |
| 成员 | 姓名（负责 XXX） |
| 指导老师 | XXX 教授 |
| 联系方式 | 2963542393@qq.com |

> 替换清单：本文本节、`docs/technical-report.md` 首页信息块、Git 提交作者信息。

---

## 十、License

[MIT](./LICENSE) © 2026 SciAgent Team
