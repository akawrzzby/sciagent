/**
 * Module 5 — the citation-graph and review workbench.
 *
 * The graph is rendered with Cytoscape. Node role (foundation / core /
 * derivative) comes from the backend's PageRank + betweenness + k-core
 * analysis, so the visual encoding is not cosmetic: a large amber node is a
 * paper that the graph algorithm considers foundational for this set.
 *
 * The right panel shows the raw graph metrics, which is what a reviewer will
 * ask for ("why is this node a foundation node?").
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import cytoscape, { Core, ElementDefinition, StylesheetStyle } from 'cytoscape'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, CitationGraph, CitationRef, ContradictionResult, GraphEdge, GraphNode } from '../lib/api'
import { EmptyState, Spinner, StatCard, Tabs, Toast } from '../components/ui'
import { useSciStore } from '../store'

type Panel = 'graph' | 'review' | 'future' | 'quality' | 'conflicts'

/** The three edge views of the workbench (plus "all"). */
type GraphView = 'citation' | 'co_citation' | 'co_reference' | 'all'
/** The node view: everything, only the prior work, or only the derivatives. */
type RoleView = 'all' | 'prior' | 'derivative'

const VIEW_LABEL: Record<GraphView, string> = {
  citation: '引用',
  co_citation: '共引',
  co_reference: '共参考',
  all: '全部',
}

const EDGE_KIND_COLOR: Record<string, string> = {
  citation: '#405670',
  co_citation: '#dd6702',
  co_reference: '#2f7f79',
}

interface ReviewResult {
  title: string
  summary: string
  core_findings: string[]
  foundation_papers: { title: string; why_foundational: string }[]
  derivative_papers: { title: string; contribution: string }[]
  open_questions: string[]
}

interface FutureResult {
  statements: { text: string; paper_title: string; page_no: number }[]
  opportunities: {
    idea: string
    evidence: string
    source_papers: string[]
    novelty_argument: string
    difficulty: string
    impact: string
  }[]
}

interface QualityResult {
  kept: CitationRef[]
  dropped: CitationRef[]
  kept_count: number
  dropped_count: number
  threshold?: number
  stance_counts?: Record<string, number>
}

/* ------------------------------------------------------------------ palette */
const ROLE_COLOR: Record<GraphNode['role'], string> = {
  foundation: '#dd6702',
  core: '#405670',
  derivative: '#2f7f79',
}
const ROLE_LABEL: Record<GraphNode['role'], string> = {
  foundation: '基石',
  core: '核心',
  derivative: '衍生',
}

/* The stance of a citation, following the Smart Citations taxonomy. */
const STANCE_COLOR: Record<string, string> = {
  supporting: '#2f7f79',
  contrasting: '#c2410c',
  mentioning: '#a8b8ca',
}
const STANCE_LABEL: Record<string, string> = {
  supporting: '支持',
  contrasting: '对照',
  mentioning: '提及',
}

export default function GraphWorkbench() {
  const navigate = useNavigate()
  const selected = useSciStore((s) => s.selected)
  const [panel, setPanel] = useState<Panel>('graph')
  const [graph, setGraph] = useState<CitationGraph | null>(null)
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [activeNode, setActiveNode] = useState<GraphNode | null>(null)
  const [activeEdge, setActiveEdge] = useState<GraphEdge | null>(null)
  const [review, setReview] = useState<ReviewResult | null>(null)
  const [future, setFuture] = useState<FutureResult | null>(null)
  const [quality, setQuality] = useState<QualityResult | null>(null)
  const [qualityPaper, setQualityPaper] = useState('')
  const [busyPanel, setBusyPanel] = useState<Panel | null>(null)
  const [view, setView] = useState<GraphView>('citation')
  const [roleView, setRoleView] = useState<RoleView>('all')
  const [conflicts, setConflicts] = useState<ContradictionResult | null>(null)

  const container = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)

  /* ------------------------------------------------------------ build graph */
  const build = useCallback(async (nextView: GraphView = view) => {
    setLoading(true)
    setToast(null)
    try {
      const g = await api.graph(selected.map((p) => p.id), nextView)
      setGraph(g)
      setActiveNode(null)
      if (!g.nodes.length) {
        setToast('构建引用图谱至少需要两篇文献。请在文献库中勾选并上传更多文献。')
      }
    } catch (e) {
      setToast(`构建失败：${(e as Error).message}`)
    } finally {
      setLoading(false)
    }
  }, [selected, view])

  useEffect(() => {
    void build()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const allNodes = graph?.nodes ?? []
  const allEdges = graph?.edges ?? []
  /* The Prior / Derivative dual view: the same graph, filtered down to the
     papers that play one of the two roles. */
  const nodes = useMemo(() => {
    if (roleView === 'prior') {
      return allNodes.filter((n) => n.role === 'foundation' || n.role === 'core')
    }
    if (roleView === 'derivative') {
      return allNodes.filter((n) => n.role === 'derivative')
    }
    return allNodes
  }, [allNodes, roleView])
  const visibleIds = useMemo(() => new Set(nodes.map((n) => n.paper_id)), [nodes])
  const edges = useMemo(
    () => allEdges.filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target)),
    [allEdges, visibleIds],
  )
  const foundationCount = allNodes.filter((n) => n.role === 'foundation').length

  /* ------------------------------------------------------------- cytoscape */
  const elements = useMemo<ElementDefinition[]>(() => {
    if (!graph) return []
    const maxPr = Math.max(...nodes.map((n) => n.pagerank), 1e-9)
    const els: ElementDefinition[] = nodes.map((n) => ({
      data: {
        id: n.paper_id,
        label: (n.label || n.title).length > 38 ? `${(n.label || n.title).slice(0, 38)}…` : n.label || n.title,
        title: `${n.label || n.title}\nPageRank ${n.pagerank.toFixed(5)} · Betweenness ${n.betweenness.toFixed(4)}`,
        role: n.role,
        size: 22 + 52 * (n.pagerank / maxPr),
        pagerank: n.pagerank,
      },
    }))
    edges.forEach((e, i) => {
      const kind = e.kind || 'citation'
      const stance = e.stance || 'mentioning'
      const color =
        kind === 'citation'
          ? STANCE_COLOR[stance] ?? '#cfd9e4'
          : EDGE_KIND_COLOR[kind] ?? '#8fa3b8'
      els.push({
        data: {
          id: `e${i}`,
          source: e.source,
          target: e.target,
          relevance: e.relevance,
          context: e.context || '',
          stance,
          stanceLabel: e.stance_label || STANCE_LABEL[stance] || '提及',
          stanceReason: e.stance_reason || '',
          kind,
          kindLabel: e.label || VIEW_LABEL[(kind as GraphView)] || '引用',
          weight: e.weight ?? 1,
          sharedCiting: e.shared_citing_papers || [],
          sharedRefs: e.shared_references || [],
          color,
        },
      })
    })
    return els
  }, [graph, nodes, edges])

  const stylesheet = useMemo<StylesheetStyle[]>(
    () => [
      {
        selector: 'node',
        style: {
          'background-color': 'data(role)',
          label: 'data(label)',
          'font-size': '9px',
          'font-family': 'Inter, "Noto Sans SC", sans-serif',
          color: '#405670',
          'text-valign': 'bottom',
          'text-margin-y': 4,
          'text-wrap': 'wrap',
          'text-max-width': '110px',
          width: 'data(size)',
          height: 'data(size)',
          'border-width': 1.5,
          'border-color': '#ffffff',
          'transition-property': 'opacity, border-width',
        },
      },
      {
        selector: 'node[role = "foundation"]',
        style: { 'background-color': ROLE_COLOR.foundation, color: '#943b0c', 'font-weight': 'bold' },
      },
      {
        selector: 'node[role = "core"]',
        style: { 'background-color': ROLE_COLOR.core, color: '#33465c' },
      },
      {
        selector: 'node[role = "derivative"]',
        style: { 'background-color': ROLE_COLOR.derivative, color: '#1f5250' },
      },
      {
        selector: 'node:selected',
        style: { 'border-width': 3, 'border-color': '#ffa338', 'z-index': 99 },
      },
      {
        selector: 'edge',
        style: {
          width: 'mapData(relevance, 0, 1, 0.6, 2.6)',
          'line-color': 'data(color)',
          'target-arrow-color': 'data(color)',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          opacity: 0.75,
        },
      },
      {
        /* The similarity edges are not citations: they are undirected. */
        selector: 'edge[kind = "co_citation"], edge[kind = "co_reference"]',
        style: {
          'target-arrow-shape': 'none',
          'line-style': 'dashed',
          'line-dash-pattern': [4, 3],
          width: 'mapData(weight, 1, 6, 1.2, 3.4)',
          opacity: 0.85,
        },
      },
      {
        selector: 'edge:selected',
        style: { width: 4, opacity: 1, 'line-color': '#dd6702', 'target-arrow-color': '#dd6702' },
      },
    ],
    [],
  )

  useEffect(() => {
    if (!container.current || !graph) return
    if (cyRef.current) {
      cyRef.current.destroy()
      cyRef.current = null
    }
    const cy = cytoscape({
      container: container.current,
      elements,
      style: stylesheet,
      layout: {
        name: 'cose',
        animate: false,
        fit: true,
        padding: 40,
        nodeRepulsion: () => 9000,
        idealEdgeLength: 110,
        gravity: 0.4,
        numIter: 1200,
      },
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.2,
    })
    cyRef.current = cy

    cy.on('tap', 'node', (evt) => {
      const id = evt.target.id()
      const node = graph.nodes.find((n) => n.paper_id === id)
      setActiveNode(node ?? null)
      setActiveEdge(null)
    })

    // Clicking an edge shows the sentence that contains the citation, so the
    // reviewer can read *why* one paper points at another instead of trusting
    // an edge weight.
    cy.on('tap', 'edge', (evt) => {
      const id = evt.target.id()
      const idx = Number(id.replace(/^e/, ''))
      setActiveEdge(graph.edges[idx] ?? null)
      setActiveNode(null)
    })

    cy.on('tap', (evt) => {
      if (evt.target === cy) setActiveEdge(null)
    })

    return () => {
      cy.destroy()
      cyRef.current = null
    }
  }, [elements, stylesheet, graph])

  /* --------------------------------------------------------------- actions */
  const runReview = async () => {
    setBusyPanel('review')
    setToast(null)
    try {
      const r = (await api.review(selected.map((p) => p.id))) as ReviewResult
      setReview(r)
    } catch (e) {
      setToast(`综述生成失败：${(e as Error).message}`)
    } finally {
      setBusyPanel(null)
    }
  }

  const runFutureWork = async () => {
    setBusyPanel('future')
    setToast(null)
    try {
      const f = await api.futureWork(selected.map((p) => p.id))
      setFuture(f)
    } catch (e) {
      setToast(`Future Work 提取失败：${(e as Error).message}`)
    } finally {
      setBusyPanel(null)
    }
  }

  const runConflicts = async () => {
    setBusyPanel('conflicts')
    setToast(null)
    try {
      const r = await api.contradictions(selected.map((p) => p.id))
      setConflicts(r)
    } catch (e) {
      setToast(`矛盾检测失败：${(e as Error).message}`)
    } finally {
      setBusyPanel(null)
    }
  }

  const runQuality = async () => {
    const paperId = qualityPaper || selected[0]?.id
    if (!paperId) {
      setToast('请先在文献库中勾选一篇文献，或在下方输入 paper_id。')
      return
    }
    setBusyPanel('quality')
    setToast(null)
    try {
      const q = await api.citationQuality(paperId)
      setQuality(q as QualityResult)
    } catch (e) {
      setToast(`引用质量分析失败：${(e as Error).message}`)
    } finally {
      setBusyPanel(null)
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-200/70 px-6 py-3.5 dark:border-ink-800">
        <div>
          <h1 className="font-serif text-xl font-semibold tracking-tight">图谱综述</h1>
          <p className="mt-0.5 font-sans text-xs text-ink-500 dark:text-ink-400">
            按引用关系识别奠基论文，并生成综述与后续研究方向
            {selected.length > 0 && ` · 范围：${selected.length} 篇已选`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Tabs<Panel>
            value={panel}
            onChange={setPanel}
            tabs={[
              { id: 'graph', label: '引用拓扑' },
              { id: 'review', label: '文献综述' },
              { id: 'future', label: 'Future Work' },
              { id: 'quality', label: '引用质量' },
              { id: 'conflicts', label: '矛盾检测' },
            ]}
          />
          <button className="btn-primary" onClick={() => void build()} disabled={loading}>
            {loading ? <Spinner className="h-3.5 w-3.5" /> : '⟳'} 重建图谱
          </button>
        </div>
      </header>

      {toast && (
        <div className="px-6 pt-3">
          <Toast message={toast} kind="error" />
        </div>
      )}

      {/* --------------------------------------------------------- statistics */}
      <div className="grid grid-cols-2 gap-3 px-6 pt-4 md:grid-cols-5">
        <StatCard label="节点" value={nodes.length} hint="已解析文献" />
        <StatCard label="引用边" value={edges.length} hint="库内互引关系" />
        <StatCard label="基石节点" value={foundationCount} hint="图论筛选" tone="accent" />
        <StatCard label="平均出度" value={(nodes.reduce((a, n) => a + n.out_degree, 0) / Math.max(1, nodes.length)).toFixed(2)} />
        <StatCard label="最大 k-core" value={Math.max(0, ...nodes.map((n) => n.k_core))} />
      </div>

      <div className="flex-1 overflow-hidden px-6 py-4">
        {panel === 'graph' && (
          <div className="flex h-full flex-col gap-3">
            {/* ------------------------------------------- view switchers */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-1 rounded-md border border-ink-200 p-0.5 dark:border-ink-700">
                <span className="px-1.5 font-sans text-[10px] text-ink-400">边视图</span>
                {(Object.keys(VIEW_LABEL) as GraphView[]).map((v) => (
                  <button
                    key={v}
                    onClick={() => {
                      setView(v)
                      void build(v)
                    }}
                    className={`rounded px-2 py-0.5 font-sans text-[11px] transition-colors ${
                      view === v
                        ? 'bg-accent-500 text-white'
                        : 'text-ink-500 hover:bg-ink-100 dark:text-ink-300 dark:hover:bg-ink-800'
                    }`}
                  >
                    {VIEW_LABEL[v]}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1 rounded-md border border-ink-200 p-0.5 dark:border-ink-700">
                <span className="px-1.5 font-sans text-[10px] text-ink-400">节点视图</span>
                {(
                  [
                    { id: 'all', label: '全部' },
                    { id: 'prior', label: '前作与基石' },
                    { id: 'derivative', label: '衍生' },
                  ] as { id: RoleView; label: string }[]
                ).map((r) => (
                  <button
                    key={r.id}
                    onClick={() => setRoleView(r.id)}
                    className={`rounded px-2 py-0.5 font-sans text-[11px] transition-colors ${
                      roleView === r.id
                        ? 'bg-accent-500 text-white'
                        : 'text-ink-500 hover:bg-ink-100 dark:text-ink-300 dark:hover:bg-ink-800'
                    }`}
                  >
                    {r.label}
                  </button>
                ))}
              </div>
              <span className="font-sans text-[10.5px] text-ink-400">
                共引 = 被同一篇文献引用；共参考 = 引用同一篇参考文献
              </span>
            </div>
            <div className="flex min-h-0 flex-1 gap-4">
            {/* ------------------------------------------------------ canvas */}
            <div className="panel relative min-w-0 flex-1 overflow-hidden">
              {nodes.length === 0 ? (
                <EmptyState
                  title="尚无引用图谱"
                  hint="上传至少两篇文献后点击「重建图谱」。系统解析每篇文献的参考文献，匹配库内条目，并计算 PageRank / Betweenness / k-core 以识别基石节点。"
                />
              ) : (
                <>
                  <div ref={container} className="h-full w-full" />
                  <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1.5">
                    {(['foundation', 'core', 'derivative'] as const).map((r) => (
                      <div key={r} className="flex items-center gap-1.5">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ background: ROLE_COLOR[r] }}
                        />
                        <span className="font-sans text-[10px] text-ink-500 dark:text-ink-300">
                          {ROLE_LABEL[r]}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="absolute bottom-3 right-3 flex gap-1.5">
                    <button className="btn-secondary !px-2 !py-1 !text-[11px]" onClick={() => cyRef.current?.fit(undefined, 40)}>
                      适配视图
                    </button>
                    <button className="btn-secondary !px-2 !py-1 !text-[11px]" onClick={() => cyRef.current?.layout({ name: 'cose', animate: true, animationDuration: 500, nodeRepulsion: () => 9000, idealEdgeLength: 110, numIter: 1200 }).run()}>
                      重新布局
                    </button>
                  </div>
                </>
              )}
            </div>

            {/* --------------------------------------------------- metrics */}
            <div className="panel flex w-[320px] shrink-0 flex-col overflow-hidden">
              <div className="panel-header">
                <span className="panel-title">节点指标</span>
                <span className="font-sans text-[10px] text-ink-400">点击节点查看详情</span>
              </div>
              <div className="scroll-thin flex-1 overflow-y-auto">
                {activeEdge ? (
                  <div className="space-y-3 p-3">
                    {activeEdge.kind && activeEdge.kind !== 'citation' ? (
                      /* A similarity edge: the two papers are related, not citing. */
                      <>
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-serif text-[13px] font-semibold leading-snug">
                            {activeEdge.kind === 'co_citation' ? '共同被引' : '共同参考'}
                          </span>
                          <span
                            className="chip shrink-0"
                            style={{
                              borderColor: EDGE_KIND_COLOR[activeEdge.kind] ?? '#2f7f79',
                              color: EDGE_KIND_COLOR[activeEdge.kind] ?? '#2f7f79',
                            }}
                          >
                            {activeEdge.label || (activeEdge.kind === 'co_citation' ? '共引' : '共参考')}
                            {' · '}
                            {activeEdge.weight ?? 1}
                          </span>
                        </div>
                        <div className="flex items-start gap-1.5 text-[11px] leading-snug text-ink-500 dark:text-ink-400">
                          <span className="flex-1 break-words">
                            {nodes.find((n) => n.paper_id === activeEdge.source)?.label ?? '论文 A'}
                          </span>
                          <span aria-hidden className="pt-0.5">↔</span>
                          <span className="flex-1 break-words">
                            {nodes.find((n) => n.paper_id === activeEdge.target)?.label ?? '论文 B'}
                          </span>
                        </div>
                        {activeEdge.shared_citing_papers && activeEdge.shared_citing_papers.length > 0 && (
                          <div>
                            <div className="font-sans text-[10px] text-ink-400">
                              同时引用这两篇文献：
                            </div>
                            <ul className="mt-0.5 space-y-0.5">
                              {activeEdge.shared_citing_papers.map((t, i) => (
                                <li key={i} className="font-sans text-[10.5px] leading-snug text-ink-600 dark:text-ink-300">
                                  · {t}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {activeEdge.shared_references && activeEdge.shared_references.length > 0 && (
                          <div>
                            <div className="font-sans text-[10px] text-ink-400">
                              共同引用的参考文献：
                            </div>
                            <ul className="mt-0.5 space-y-0.5">
                              {activeEdge.shared_references.map((t, i) => (
                                <li key={i} className="font-sans text-[10.5px] leading-snug text-ink-600 dark:text-ink-300">
                                  · {t}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        <div className="font-sans text-[10px] text-ink-400">
                          相似边来自共引 / 共参考统计，不表示直接的引用关系。
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-serif text-[13px] font-semibold leading-snug">
                            引用原句
                          </span>
                          <span
                            className="chip shrink-0"
                            style={{
                              borderColor: STANCE_COLOR[activeEdge.stance] ?? '#a8b8ca',
                              color: STANCE_COLOR[activeEdge.stance] ?? '#a8b8ca',
                            }}
                          >
                            {activeEdge.stance_label || STANCE_LABEL[activeEdge.stance] || '提及'}
                          </span>
                        </div>
                        {activeEdge.stance_reason && (
                          <div className="font-sans text-[10px] leading-snug text-ink-400">
                            判定依据：{activeEdge.stance_reason}
                          </div>
                        )}
                        <div className="flex items-start gap-1.5 text-[11px] leading-snug text-ink-500 dark:text-ink-400">
                          <span className="flex-1 break-words">
                            {nodes.find((n) => n.paper_id === activeEdge.source)?.label ?? '引用方'}
                          </span>
                          <span aria-hidden className="pt-0.5">→</span>
                          <span className="flex-1 break-words">
                            {nodes.find((n) => n.paper_id === activeEdge.target)?.label ?? '被引方'}
                          </span>
                        </div>
                        <div
                          className="rounded-md border-l-2 bg-ink-50 px-3 py-2 text-[11.5px] leading-relaxed text-ink-700 dark:bg-ink-800 dark:text-parchment-200"
                          style={{ borderLeftColor: '#dd6702' }}
                        >
                          {activeEdge.context?.trim() ? (
                            <>“{activeEdge.context.trim()}”</>
                          ) : (
                            <span className="text-ink-400">
                              未能在正文中定位到该引用的原句（扫描件可能未被 OCR 识别出引用标记）。
                            </span>
                          )}
                        </div>
                        <div className="font-sans text-[10px] text-ink-400">
                          相关度 {activeEdge.relevance.toFixed(2)}
                        </div>
                      </>
                    )}
                    <button
                      className="btn-secondary w-full !text-[11px]"
                      onClick={() => navigate(`/reader/${activeEdge.source}`)}
                    >
                      在阅读器中打开引用方
                    </button>
                  </div>
                ) : activeNode ? (
                  <div className="space-y-3 p-3">
                    <div>
                      <div className="font-serif text-[13px] font-semibold leading-snug">
                        {activeNode.label || activeNode.title}
                      </div>
                      <div className="mt-1 flex items-center gap-1.5">
                        <span
                          className="chip"
                          style={{
                            borderColor: ROLE_COLOR[activeNode.role],
                            color: ROLE_COLOR[activeNode.role],
                          }}
                        >
                          {ROLE_LABEL[activeNode.role]}
                        </span>
                        {activeNode.year && (
                          <span className="chip border-ink-200 text-ink-500 dark:border-ink-700 dark:text-ink-400">
                            {activeNode.year}
                          </span>
                        )}
                        {activeNode.arxiv_id && (
                          <span className="chip border-ink-200 font-mono text-ink-500 dark:border-ink-700 dark:text-ink-400">
                            {activeNode.arxiv_id}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        ['PageRank', activeNode.pagerank.toFixed(5)],
                        ['Betweenness', activeNode.betweenness.toFixed(4)],
                        ['HITS 权威度', activeNode.authority_score.toFixed(4)],
                        ['HITS 枢纽度', activeNode.hub_score.toFixed(4)],
                        ['k-core', String(activeNode.k_core)],
                        ['入度 / 出度', `${activeNode.in_degree} / ${activeNode.out_degree}`],
                      ].map(([k, v]) => (
                        <div key={k} className="rounded-md bg-ink-50 px-2.5 py-1.5 dark:bg-ink-800">
                          <div className="font-sans text-[9.5px] uppercase tracking-wider text-ink-400">{k}</div>
                          <div className="font-mono text-[12px] font-semibold text-ink-700 dark:text-parchment-200">
                            {v}
                          </div>
                        </div>
                      ))}
                    </div>
                    <button
                      className="btn-secondary w-full !text-[11px]"
                      onClick={() => navigate(`/reader/${activeNode.paper_id}`)}
                    >
                      在阅读器中打开
                    </button>
                  </div>
                ) : (
                  <table className="w-full text-left">
                    <thead className="sticky top-0 bg-ink-50 dark:bg-ink-800">
                      <tr className="font-sans text-[9.5px] uppercase tracking-wider text-ink-400">
                        <th className="px-2.5 py-1.5 font-medium">文献</th>
                        <th className="px-1 py-1.5 font-medium">PR</th>
                        <th className="px-1 py-1.5 font-medium">BC</th>
                        <th className="px-1 py-1.5 font-medium">k</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...nodes]
                        .sort((a, b) => b.pagerank - a.pagerank)
                        .map((n) => (
                          <tr
                            key={n.paper_id}
                            onClick={() => setActiveNode(n)}
                            className="cursor-pointer border-t border-ink-100 text-[11px] hover:bg-ink-50 dark:border-ink-800 dark:hover:bg-ink-800/60"
                          >
                            <td className="px-2.5 py-1.5">
                              <div className="flex items-center gap-1.5">
                                <span
                                  className="h-2 w-2 shrink-0 rounded-full"
                                  style={{ background: ROLE_COLOR[n.role] }}
                                />
                                <span className="line-clamp-2 text-ink-600 dark:text-ink-300">
                                  {n.label || n.title}
                                </span>
                              </div>
                            </td>
                            <td className="px-1 py-1.5 font-mono text-[10px] tabular-nums text-ink-500">
                              {n.pagerank.toFixed(4)}
                            </td>
                            <td className="px-1 py-1.5 font-mono text-[10px] tabular-nums text-ink-500">
                              {n.betweenness.toFixed(3)}
                            </td>
                            <td className="px-1 py-1.5 font-mono text-[10px] tabular-nums text-ink-500">
                              {n.k_core}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
            </div>
          </div>
        )}

        {panel === 'review' && (
          <div className="panel h-full overflow-y-auto p-5">
            {!review ? (
              <div className="flex h-full flex-col items-center justify-center gap-3">
                <EmptyState
                  title="尚未生成综述"
                  hint="系统依据引用图谱的 PageRank / Betweenness / k-core 指标，筛选基石文献与衍生文献，再调用大模型生成结构化的领域综述。"
                />
                <button className="btn-primary" onClick={runReview} disabled={busyPanel === 'review'}>
                  {busyPanel === 'review' ? <Spinner className="h-3.5 w-3.5" /> : null} 生成综述
                </button>
              </div>
            ) : (
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="flex items-center justify-between">
                  <h2 className="font-serif text-lg font-semibold">{review.title}</h2>
                  <button className="btn-secondary !text-[11px]" onClick={runReview} disabled={busyPanel === 'review'}>
                    重新生成
                  </button>
                </div>
                <div className="prose-paper">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{review.summary}</ReactMarkdown>
                </div>
                {review.core_findings.length > 0 && (
                  <section>
                    <h3 className="panel-title mb-1.5">核心发现</h3>
                    <ul className="list-disc space-y-1 pl-5 font-sans text-[13px] text-ink-700 dark:text-parchment-200">
                      {review.core_findings.map((f, i) => (
                        <li key={i}>{f}</li>
                      ))}
                    </ul>
                  </section>
                )}
                <div className="grid gap-4 md:grid-cols-2">
                  <section>
                    <h3 className="panel-title mb-1.5">基石文献</h3>
                    <div className="space-y-2">
                      {review.foundation_papers.map((p, i) => (
                        <div key={i} className="rounded-lg border border-ink-200 px-3 py-2 dark:border-ink-700">
                          <div className="font-sans text-[12px] font-semibold text-ink-800 dark:text-parchment-100">
                            {p.title}
                          </div>
                          <div className="mt-0.5 font-sans text-[11px] leading-snug text-ink-500 dark:text-ink-400">
                            {p.why_foundational}
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                  <section>
                    <h3 className="panel-title mb-1.5">衍生文献</h3>
                    <div className="space-y-2">
                      {review.derivative_papers.map((p, i) => (
                        <div key={i} className="rounded-lg border border-ink-200 px-3 py-2 dark:border-ink-700">
                          <div className="font-sans text-[12px] font-semibold text-ink-800 dark:text-parchment-100">
                            {p.title}
                          </div>
                          <div className="mt-0.5 font-sans text-[11px] leading-snug text-ink-500 dark:text-ink-400">
                            {p.contribution}
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                </div>
                {review.open_questions.length > 0 && (
                  <section>
                    <h3 className="panel-title mb-1.5">开放问题</h3>
                    <ol className="list-decimal space-y-1 pl-5 font-sans text-[13px] text-ink-700 dark:text-parchment-200">
                      {review.open_questions.map((q, i) => (
                        <li key={i}>{q}</li>
                      ))}
                    </ol>
                  </section>
                )}
              </div>
            )}
          </div>
        )}

        {panel === 'future' && (
          <div className="panel h-full overflow-y-auto p-5">
            {!future ? (
              <div className="flex h-full flex-col items-center justify-center gap-3">
                <EmptyState
                  title="尚未提取 Future Work"
                  hint="系统定位每篇文献的 Future Work / Limitations / Discussion 章节片段，去重后交由大模型归纳为研究机会，并标注新颖性论证、难度与影响力。"
                />
                <button className="btn-primary" onClick={runFutureWork} disabled={busyPanel === 'future'}>
                  {busyPanel === 'future' ? <Spinner className="h-3.5 w-3.5" /> : null} 开始提取
                </button>
              </div>
            ) : (
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="flex items-center justify-between">
                  <h2 className="font-serif text-lg font-semibold">研究机会</h2>
                  <button className="btn-secondary !text-[11px]" onClick={runFutureWork} disabled={busyPanel === 'future'}>
                    重新提取
                  </button>
                </div>

                {future.opportunities.length === 0 ? (
                  <EmptyState title="未发现可归纳的 Future Work" hint="请确认文献库中包含带有 Future Work 章节的文献。" />
                ) : (
                  future.opportunities.map((o, i) => (
                    <article key={i} className="rounded-xl border border-ink-200 px-4 py-3.5 dark:border-ink-700">
                      <div className="flex items-start gap-2">
                        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-100 font-sans text-[11px] font-semibold text-accent-700 dark:bg-accent-900/50 dark:text-accent-300">
                          {i + 1}
                        </span>
                        <div className="min-w-0">
                          <div className="font-serif text-[14px] font-semibold leading-snug text-ink-900 dark:text-parchment-50">
                            {o.idea}
                          </div>
                          <div className="mt-1 font-sans text-[12px] leading-relaxed text-ink-600 dark:text-ink-300">
                            {o.evidence}
                          </div>
                          {o.novelty_argument && (
                            <div className="mt-2 rounded-md bg-ink-50 px-2.5 py-1.5 font-sans text-[11px] leading-snug text-ink-600 dark:bg-ink-800 dark:text-ink-300">
                              <span className="font-semibold">新颖性论证：</span>
                              {o.novelty_argument}
                            </div>
                          )}
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {o.difficulty && (
                              <span className="chip border-ink-200 text-ink-500 dark:border-ink-700 dark:text-ink-400">
                                难度 {o.difficulty}
                              </span>
                            )}
                            {o.impact && (
                              <span className="chip border-ink-200 text-ink-500 dark:border-ink-700 dark:text-ink-400">
                                影响力 {o.impact}
                              </span>
                            )}
                            {o.source_papers.map((p) => (
                              <span key={p} className="chip border-accent-200 bg-accent-50 text-accent-700 dark:border-accent-800 dark:bg-accent-900/40 dark:text-accent-300">
                                {p}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </article>
                  ))
                )}

                {future.statements.length > 0 && (
                  <details className="panel overflow-hidden">
                    <summary className="panel-header cursor-pointer select-none">
                      <span className="panel-title">原始 Future Work 片段（{future.statements.length}）</span>
                    </summary>
                    <div className="space-y-2 p-3">
                      {future.statements.map((s, i) => (
                        <div key={i} className="border-l-2 border-accent-300 pl-2.5 dark:border-accent-800">
                          <div className="font-sans text-[11px] leading-relaxed text-ink-600 dark:text-ink-300">
                            {s.text}
                          </div>
                          <div className="mt-0.5 font-sans text-[10px] text-ink-400">
                            — {s.paper_title}{s.page_no ? ` · p.${s.page_no}` : ''}
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        )}

        {panel === 'quality' && (
          <div className="panel h-full overflow-y-auto p-5">
            <div className="mx-auto max-w-3xl space-y-4">
              <div className="flex flex-wrap items-end gap-2">
                <div className="min-w-[220px] flex-1">
                  <label className="mb-1 block font-sans text-[10px] uppercase tracking-wider text-ink-400">
                    分析对象 paper_id
                  </label>
                  <input
                    className="input"
                    value={qualityPaper}
                    onChange={(e) => setQualityPaper(e.target.value)}
                    placeholder={selected[0]?.id ?? '留空则使用文献库勾选的第一篇'}
                  />
                </div>
                <button className="btn-primary" onClick={runQuality} disabled={busyPanel === 'quality'}>
                  {busyPanel === 'quality' ? <Spinner className="h-3.5 w-3.5" /> : null} 分析引用质量
                </button>
              </div>

              {!quality ? (
                <EmptyState
                  title="尚未分析"
                  hint="该功能区分「实质性引用」与「装饰性引用」：前者支撑论点（常为方法或对比基准），后者仅为顺带提及。该指标用于衡量一篇文献对既有工作的依赖程度。"
                />
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    <StatCard label="实质性引用" value={quality.kept_count} tone="accent" />
                    <StatCard label="装饰性引用" value={quality.dropped_count} />
                    <StatCard
                      label="实质性占比"
                      value={`${Math.round(
                        (quality.kept_count / Math.max(1, quality.kept_count + quality.dropped_count)) * 100,
                      )}%`}
                    />
                    <StatCard
                      label="支持 / 对照 / 提及"
                      value={
                        quality.stance_counts
                          ? `${quality.stance_counts.supporting ?? 0} / ${
                              quality.stance_counts.contrasting ?? 0
                            } / ${quality.stance_counts.mentioning ?? 0}`
                          : '—'
                      }
                    />
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <section>
                      <h3 className="panel-title mb-1.5">实质性引用（支撑论点）</h3>
                      <div className="space-y-1.5">
                        {quality.kept.map((r, i) => (
                          <div key={i} className="rounded-md border border-ink-200 px-2.5 py-1.5 dark:border-ink-700">
                            <div className="flex items-start justify-between gap-2">
                              <span className="font-sans text-[11.5px] font-medium text-ink-700 dark:text-parchment-200">
                                {r.title}
                              </span>
                              {r.stance && (
                                <span
                                  className="chip shrink-0"
                                  style={{
                                    borderColor: STANCE_COLOR[r.stance] ?? '#a8b8ca',
                                    color: STANCE_COLOR[r.stance] ?? '#a8b8ca',
                                  }}
                                >
                                  {r.stance_label || STANCE_LABEL[r.stance]}
                                </span>
                              )}
                            </div>
                            <div className="font-sans text-[10px] text-ink-400">
                              {r.authors?.slice(0, 3).join(', ')} {r.year ? `· ${r.year}` : ''}
                            </div>
                            {r.context && (
                              <details className="mt-1">
                                <summary className="cursor-pointer font-sans text-[10px] text-ink-400">
                                  查看引用原句
                                </summary>
                                <div
                                  className="mt-1 border-l-2 bg-ink-50 px-2 py-1 font-sans text-[10.5px] leading-relaxed text-ink-600 dark:bg-ink-800 dark:text-ink-300"
                                  style={{ borderLeftColor: '#dd6702' }}
                                >
                                  “{r.context.trim()}”
                                </div>
                              </details>
                            )}
                          </div>
                        ))}
                      </div>
                    </section>
                    <section>
                      <h3 className="panel-title mb-1.5">装饰性引用（顺带提及）</h3>
                      <div className="space-y-1.5">
                        {quality.dropped.map((r, i) => (
                          <div key={i} className="rounded-md border border-ink-200 px-2.5 py-1.5 opacity-80 dark:border-ink-700">
                            <div className="flex items-start justify-between gap-2">
                              <span className="font-sans text-[11.5px] text-ink-600 dark:text-ink-300">{r.title}</span>
                              {r.stance && (
                                <span
                                  className="chip shrink-0"
                                  style={{
                                    borderColor: STANCE_COLOR[r.stance] ?? '#a8b8ca',
                                    color: STANCE_COLOR[r.stance] ?? '#a8b8ca',
                                  }}
                                >
                                  {r.stance_label || STANCE_LABEL[r.stance]}
                                </span>
                              )}
                            </div>
                            <div className="font-sans text-[10px] text-ink-400">{r.authors?.slice(0, 3).join(', ')}</div>
                            {r.context && (
                              <details className="mt-1">
                                <summary className="cursor-pointer font-sans text-[10px] text-ink-400">
                                  查看引用原句
                                </summary>
                                <div
                                  className="mt-1 border-l-2 bg-ink-50 px-2 py-1 font-sans text-[10.5px] leading-relaxed text-ink-600 dark:bg-ink-800 dark:text-ink-300"
                                  style={{ borderLeftColor: '#dd6702' }}
                                >
                                  “{r.context.trim()}”
                                </div>
                              </details>
                            )}
                          </div>
                        ))}
                      </div>
                    </section>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {panel === 'conflicts' && (
          <div className="panel h-full overflow-y-auto p-5">
            <div className="mx-auto max-w-3xl space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <button className="btn-primary" onClick={runConflicts} disabled={busyPanel === 'conflicts'}>
                  {busyPanel === 'conflicts' ? <Spinner className="h-3.5 w-3.5" /> : null} 扫描矛盾
                </button>
                <span className="font-sans text-[10.5px] text-ink-400">
                  规则化检测：同一指标、同一上下文、不同取值 → 数值冲突；同一主语、相反结论 → 结论冲突
                </span>
              </div>

              {!conflicts ? (
                <EmptyState
                  title="尚未扫描"
                  hint="矛盾检测扫描文献库的全部文本块，找出一处或两处相互冲突的陈述，并同时给出双方原句。"
                />
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    <StatCard label="矛盾总数" value={conflicts.count} tone="accent" />
                    <StatCard label="数值冲突" value={conflicts.numeric_count} />
                    <StatCard label="结论冲突" value={conflicts.claim_count} />
                    <StatCard label="扫描文本块" value={conflicts.scanned_chunks} />
                  </div>

                  {conflicts.count === 0 ? (
                    <EmptyState
                      title="未检出矛盾"
                      hint="当前文献库中没有同一指标的相互冲突取值，也没有相反的结论。"
                    />
                  ) : (
                    <div className="space-y-3">
                      {conflicts.conflicts.map((c, i) => (
                        <div key={i} className="rounded-md border border-ink-200 p-3 dark:border-ink-700">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <span className="font-serif text-[13px] font-semibold">{c.summary}</span>
                            <span className="chip" style={{ borderColor: '#dd6702', color: '#dd6702' }}>
                              {c.kind === 'numeric' ? '数值冲突' : '结论冲突'} · {Math.round(c.confidence * 100)}%
                            </span>
                          </div>
                          <div className="mt-1 font-sans text-[10px] text-ink-400">判定依据：{c.reason}</div>
                          <div className="mt-2 grid gap-2 md:grid-cols-2">
                            {[c.left, c.right].map((side, k) => (
                              <div
                                key={k}
                                className="border-l-2 bg-ink-50 px-2 py-1.5 dark:bg-ink-800"
                                style={{ borderLeftColor: k === 0 ? '#dd6702' : '#2f7f79' }}
                              >
                                <div className="font-sans text-[10px] text-ink-400">
                                  {side.paper_title}
                                  {side.page_no ? ` · p.${side.page_no}` : ''}
                                  {side.value ? ` · ${side.metric} = ${side.value}${side.unit || ''}` : ''}
                                </div>
                                <div className="mt-0.5 font-sans text-[10.5px] leading-relaxed text-ink-600 dark:text-ink-300">
                                  “{side.text}”
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
