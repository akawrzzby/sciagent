# SciAgent · 科研小助手

> **智领科研 · 意图驱动** —— 基于大模型与 Agent 架构的全流程科研助手
>
> 面向科研文献阅读与写作的端到端 Agent 系统：多模态解析 → 混合检索溯源 → 自主工具编排 → 图谱挖掘 → 学术写作。

[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/frontend-React%2018-61DAFB?style=flat-square&logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/lang-TypeScript-3178C6?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/lang-Python%203.11-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](#-license)

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
| 前端同源 `/api` + Nginx 反代 | 避免跨域与"前后端两个公网 URL"的不一致 |

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

> 在线演示地址与评审账号见 **`docs/demo-script.md`**（演示分镜脚本与操作指引同文档）。

技术方案：前端与后端同源部署于 Nginx，`/api` 反向代理到 FastAPI；对外由 **Cloudflare Tunnel** 暴露单一 HTTPS 地址，因此在 NAT 环境（无公网 IP）下也能提供公网可访问的演示地址。

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
├── deploy/              nginx.conf · entrypoint.sh
├── docs/                technical-report.md · demo-script.md
├── scripts/             模型下载 · 解析冒烟 · 启动脚本
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
