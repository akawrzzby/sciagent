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
import { api, CitationGraph, GraphNode } from '../lib/api'
import { EmptyState, Spinner, StatCard, Tabs, Toast } from '../components/ui'
import { useSciStore } from '../store'

type Panel = 'graph' | 'review' | 'future' | 'quality'

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
  kept: { title: string; authors: string[]; year: number | null; relevance?: number }[]
  dropped: { title: string; authors: string[]; year: number | null; reason?: string }[]
  kept_count: number
  dropped_count: number
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

export default function GraphWorkbench() {
  const navigate = useNavigate()
  const selected = useSciStore((s) => s.selected)
  const [panel, setPanel] = useState<Panel>('graph')
  const [graph, setGraph] = useState<CitationGraph | null>(null)
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [activeNode, setActiveNode] = useState<GraphNode | null>(null)
  const [review, setReview] = useState<ReviewResult | null>(null)
  const [future, setFuture] = useState<FutureResult | null>(null)
  const [quality, setQuality] = useState<QualityResult | null>(null)
  const [qualityPaper, setQualityPaper] = useState('')
  const [busyPanel, setBusyPanel] = useState<Panel | null>(null)

  const container = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)

  /* ------------------------------------------------------------ build graph */
  const build = useCallback(async () => {
    setLoading(true)
    setToast(null)
    try {
      const g = await api.graph(selected.map((p) => p.id))
      setGraph(g)
      setActiveNode(null)
      if (!g.nodes.length) {
        setToast('引用图谱至少需要两篇论文，请先在文献库中勾选并上传更多文献。')
      }
    } catch (e) {
      setToast(`构建失败：${(e as Error).message}`)
    } finally {
      setLoading(false)
    }
  }, [selected])

  useEffect(() => {
    void build()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* ------------------------------------------------------------- cytoscape */
  const elements = useMemo<ElementDefinition[]>(() => {
    if (!graph) return []
    const maxPr = Math.max(...graph.nodes.map((n) => n.pagerank), 1e-9)
    const els: ElementDefinition[] = graph.nodes.map((n) => ({
      data: {
        id: n.paper_id,
        label: n.title.length > 38 ? `${n.title.slice(0, 38)}…` : n.title,
        title: `${n.title}\nPageRank ${n.pagerank.toFixed(5)} · Betweenness ${n.betweenness.toFixed(4)}`,
        role: n.role,
        size: 22 + 52 * (n.pagerank / maxPr),
        pagerank: n.pagerank,
      },
    }))
    graph.edges.forEach((e, i) => {
      els.push({
        data: { id: `e${i}`, source: e.source, target: e.target, relevance: e.relevance },
      })
    })
    return els
  }, [graph])

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
          'line-color': '#cfd9e4',
          'target-arrow-color': '#a8b8ca',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          opacity: 0.75,
        },
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
      setToast(`Future Work 挖掘失败：${(e as Error).message}`)
    } finally {
      setBusyPanel(null)
    }
  }

  const runQuality = async () => {
    const paperId = qualityPaper || selected[0]?.id
    if (!paperId) {
      setToast('请先在文献库勾选一篇论文，或在下方输入 paper_id。')
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

  const nodes = graph?.nodes ?? []
  const edges = graph?.edges ?? []
  const foundationCount = nodes.filter((n) => n.role === 'foundation').length

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-200/70 px-6 py-3.5 dark:border-ink-800">
        <div>
          <h1 className="font-serif text-xl font-semibold tracking-tight">图谱综述</h1>
          <p className="mt-0.5 font-sans text-xs text-ink-500 dark:text-ink-400">
            PageRank · Betweenness · k-core 基石挖掘 · 自动综述 · Future Work 探索
            {selected.length > 0 && ` · 范围：${selected.length} 篇已选`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Tabs<Panel>
            value={panel}
            onChange={setPanel}
            tabs={[
              { id: 'graph', label: '引用拓扑' },
              { id: 'review', label: '自动综述' },
              { id: 'future', label: 'Future Work' },
              { id: 'quality', label: '引用质量' },
            ]}
          />
          <button className="btn-primary" onClick={build} disabled={loading}>
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
          <div className="flex h-full gap-4">
            {/* ------------------------------------------------------ canvas */}
            <div className="panel relative min-w-0 flex-1 overflow-hidden">
              {nodes.length === 0 ? (
                <EmptyState
                  title="尚无引用图谱"
                  hint="上传至少两篇论文后点击「重建图谱」。系统会解析每篇论文的参考文献，匹配库内文献，并计算 PageRank / Betweenness / k-core 以识别基石节点。"
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
                {activeNode ? (
                  <div className="space-y-3 p-3">
                    <div>
                      <div className="font-serif text-[13px] font-semibold leading-snug">
                        {activeNode.title}
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
                                  {n.title}
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
        )}

        {panel === 'review' && (
          <div className="panel h-full overflow-y-auto p-5">
            {!review ? (
              <div className="flex h-full flex-col items-center justify-center gap-3">
                <EmptyState
                  title="尚未生成综述"
                  hint="系统会基于引用图谱的 PageRank / Betweenness / k-core 指标，挑选基石文献与衍生文献，并调用大模型生成结构化的领域综述。"
                />
                <button className="btn-primary" onClick={runReview} disabled={busyPanel === 'review'}>
                  {busyPanel === 'review' ? <Spinner className="h-3.5 w-3.5" /> : null} 生成自动综述
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
                  title="尚未挖掘 Future Work"
                  hint="系统会定位每篇论文的 Future Work / Limitations / Discussion 章节片段，去重后交给大模型归纳成可执行的研究机会，并标注新颖性论证、难度与影响力。"
                />
                <button className="btn-primary" onClick={runFutureWork} disabled={busyPanel === 'future'}>
                  {busyPanel === 'future' ? <Spinner className="h-3.5 w-3.5" /> : null} 开始挖掘
                </button>
              </div>
            ) : (
              <div className="mx-auto max-w-3xl space-y-5">
                <div className="flex items-center justify-between">
                  <h2 className="font-serif text-lg font-semibold">研究机会</h2>
                  <button className="btn-secondary !text-[11px]" onClick={runFutureWork} disabled={busyPanel === 'future'}>
                    重新挖掘
                  </button>
                </div>

                {future.opportunities.length === 0 ? (
                  <EmptyState title="未发现可归纳的 Future Work" hint="请确认文献库中包含带有 Future Work 章节的论文。" />
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
                            — {s.paper_title} · p.{s.page_no}
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
                  hint="该功能区分「实质性引用」与「装饰性引用」：前者真正支撑了论点（常为方法或对比基准），后者只是顺带提及。据此可以判断一篇论文真正站在谁的肩膀上。"
                />
              ) : (
                <>
                  <div className="grid grid-cols-3 gap-3">
                    <StatCard label="实质性引用" value={quality.kept_count} tone="accent" />
                    <StatCard label="装饰性引用" value={quality.dropped_count} />
                    <StatCard
                      label="实质性占比"
                      value={`${Math.round(
                        (quality.kept_count / Math.max(1, quality.kept_count + quality.dropped_count)) * 100,
                      )}%`}
                    />
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <section>
                      <h3 className="panel-title mb-1.5">实质性引用（支撑论点）</h3>
                      <div className="space-y-1.5">
                        {quality.kept.map((r, i) => (
                          <div key={i} className="rounded-md border border-ink-200 px-2.5 py-1.5 dark:border-ink-700">
                            <div className="font-sans text-[11.5px] font-medium text-ink-700 dark:text-parchment-200">
                              {r.title}
                            </div>
                            <div className="font-sans text-[10px] text-ink-400">
                              {r.authors?.slice(0, 3).join(', ')} {r.year ? `· ${r.year}` : ''}
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                    <section>
                      <h3 className="panel-title mb-1.5">装饰性引用（顺带提及）</h3>
                      <div className="space-y-1.5">
                        {quality.dropped.map((r, i) => (
                          <div key={i} className="rounded-md border border-ink-200 px-2.5 py-1.5 opacity-70 dark:border-ink-700">
                            <div className="font-sans text-[11.5px] text-ink-600 dark:text-ink-300">{r.title}</div>
                            <div className="font-sans text-[10px] text-ink-400">{r.authors?.slice(0, 3).join(', ')}</div>
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
      </div>
    </div>
  )
}
