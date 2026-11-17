/**
 * Module 6 — the academic writing workbench.
 *
 * Three capabilities, each one a hard requirement of the task statement:
 *   1. 论文框架  — generate a paper skeleton whose references are *verified*
 *                  against the library (hallucinated references are rejected
 *                  and reported back to the user);
 *   2. 架构图    — Mermaid / TikZ / Graphviz code with a live preview and a
 *                  one-click copy into the clipboard;
 *   3. 图表      — paste a CSV, get an academic figure (PNG) *and* a
 *                  publication-ready caption written by the model.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { api, FrameworkResult } from '../lib/api'
import { EmptyState, Spinner, Tabs, Toast } from '../components/ui'

type Panel = 'framework' | 'diagram' | 'plot'

const CHART_TYPES = [
  { id: 'line', label: '折线图' },
  { id: 'bar', label: '柱状图' },
  { id: 'scatter', label: '散点图' },
  { id: 'grouped_bar', label: '分组柱状图' },
  { id: 'area', label: '面积图' },
  { id: 'pie', label: '饼图' },
  { id: 'hist', label: '直方图' },
]

const SAMPLE_CSV = `model,bleu_en_de,bleu_en_fr,params_b
Transformer (base),27.3,38.1,65
Transformer (big),28.4,41.8,213
Transformer (XL),41.8,43.9,1310
`

export default function WritingWorkbench() {
  const [panel, setPanel] = useState<Panel>('framework')
  const [toast, setToast] = useState<{ msg: string; kind: 'info' | 'error' | 'ok' } | null>(null)
  const [busy, setBusy] = useState(false)

  /* ----------------------------------------------------------- framework */
  const [idea, setIdea] = useState('')
  const [framework, setFramework] = useState<FrameworkResult | null>(null)

  /* ------------------------------------------------------------- diagram */
  const [desc, setDesc] = useState('')
  const [diagramType, setDiagramType] = useState('mermaid')
  const [diagram, setDiagram] = useState<{ diagram_type: string; title: string; code: string; description: string } | null>(null)
  const [svg, setSvg] = useState('')
  const svgHost = useRef<HTMLDivElement>(null)

  /* ---------------------------------------------------------------- plot */
  const [csv, setCsv] = useState(SAMPLE_CSV)
  const [chartType, setChartType] = useState('line')
  const [plot, setPlot] = useState<{
    chart_type: string
    image_base64: string
    caption: string
    key_findings: string[]
    table_markdown: string
    columns: string[]
    row_count: number
  } | null>(null)

  /* -------------------------------------------------- mermaid live preview */
  useEffect(() => {
    if (!diagram || diagram.diagram_type !== 'mermaid' || !diagram.code) {
      setSvg('')
      return
    }
    let cancelled = false
    void (async () => {
      try {
        const mermaid = (await import('mermaid')).default
        mermaid.initialize({
          startOnLoad: false,
          theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default',
          securityLevel: 'loose',
          fontFamily: 'Inter, "Noto Sans SC", sans-serif',
        })
        const id = `mmd-${Math.random().toString(36).slice(2, 9)}`
        const { svg: out } = await mermaid.render(id, diagram.code)
        if (!cancelled) setSvg(out)
      } catch (e) {
        if (!cancelled) setToast({ msg: `Mermaid 预览失败：${(e as Error).message}`, kind: 'error' })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [diagram])

  /* -------------------------------------------------------------- actions */
  const runFramework = async () => {
    if (!idea.trim()) return
    setBusy(true)
    setToast(null)
    try {
      const r = await api.framework(idea.trim())
      setFramework(r)
    } catch (e) {
      setToast({ msg: `框架生成失败：${(e as Error).message}`, kind: 'error' })
    } finally {
      setBusy(false)
    }
  }

  const runDiagram = async () => {
    if (!desc.trim()) return
    setBusy(true)
    setToast(null)
    try {
      const r = await api.diagram(desc.trim(), diagramType)
      setDiagram(r)
    } catch (e) {
      setToast({ msg: `架构图生成失败：${(e as Error).message}`, kind: 'error' })
    } finally {
      setBusy(false)
    }
  }

  const runPlot = async () => {
    if (csv.trim().length < 3) return
    setBusy(true)
    setToast(null)
    try {
      const r = await api.plot({ csv_content: csv, chart_type: chartType })
      setPlot(r)
    } catch (e) {
      setToast({ msg: `绘图失败：${(e as Error).message}`, kind: 'error' })
    } finally {
      setBusy(false)
    }
  }

  const copy = async (text: string, what: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setToast({ msg: `${what}已复制到剪贴板。`, kind: 'ok' })
    } catch {
      setToast({ msg: '复制失败，请手动选择文本。', kind: 'error' })
    }
  }

  const downloadSvg = () => {
    if (!svg) return
    const blob = new Blob([svg], { type: 'image/svg+xml' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${(diagram?.title ?? 'diagram').replace(/\s+/g, '_')}.svg`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const frameworkSections = useMemo(() => {
    if (!framework) return []
    return [
      { key: 'intro', title: '引言 Introduction', items: framework.introduction },
      { key: 'related', title: '相关工作 Related Work', items: framework.related_work.map((r) => `${r.topic} — ${r.summary}${r.citations.length ? ` [${r.citations.join(', ')}]` : ''}`) },
      { key: 'method', title: '方法 Method', items: framework.method },
      { key: 'exp', title: '实验 Experiments', items: framework.experiments },
    ].filter((s) => s.items.length > 0)
  }, [framework])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-200/70 px-6 py-3.5 dark:border-ink-800">
        <div>
          <h1 className="font-serif text-xl font-semibold tracking-tight">写作工作台</h1>
          <p className="mt-0.5 font-sans text-xs text-ink-500 dark:text-ink-400">
            论文框架 · 引用真实性校验 · TikZ/Mermaid/Graphviz 架构图 · CSV 转学术图表与 Figure Caption
          </p>
        </div>
        <Tabs<Panel>
          value={panel}
          onChange={setPanel}
          tabs={[
            { id: 'framework', label: '论文框架' },
            { id: 'diagram', label: '架构图' },
            { id: 'plot', label: '图表与 Caption' },
          ]}
        />
      </header>

      {toast && (
        <div className="px-6 pt-3">
          <Toast message={toast.msg} kind={toast.kind} />
        </div>
      )}

      <div className="scroll-thin flex-1 overflow-y-auto px-6 py-4">
        {/* ============================================================ framework */}
        {panel === 'framework' && (
          <div className="mx-auto max-w-4xl space-y-4">
            <div className="panel px-4 py-3">
              <label className="mb-1 block font-sans text-[10px] uppercase tracking-wider text-ink-400">
                研究想法
              </label>
              <textarea
                className="input resize-none"
                rows={3}
                value={idea}
                onChange={(e) => setIdea(e.target.value)}
                placeholder="例如：面向多文献并行场景的大模型科研助手，提供证据溯源与综述生成。"
              />
              <div className="mt-2 flex items-center gap-2">
                <button className="btn-primary" onClick={runFramework} disabled={busy || idea.trim().length < 5}>
                  {busy ? <Spinner className="h-3.5 w-3.5" /> : null} 生成论文框架
                </button>
                <span className="font-sans text-[10.5px] text-ink-400">
                  参考文献仅来自文献库检索结果，模型虚构的引用由护栏拦截并单独列出
                </span>
              </div>
            </div>

            {!framework ? (
              <EmptyState
                title="尚未生成框架"
                hint="输入研究想法后，Agent 检索文献库，筛选与该想法相关的已入库文献，并据此起草标题、摘要与各章节要点。"
              />
            ) : (
              <div className="space-y-4">
                <div className="panel px-4 py-3.5">
                  <h2 className="font-serif text-lg font-semibold leading-snug">{framework.title}</h2>
                  <p className="mt-2 font-sans text-[13px] leading-relaxed text-ink-700 dark:text-parchment-200">
                    {framework.abstract}
                  </p>
                </div>

                {frameworkSections.map((sec) => (
                  <section key={sec.key} className="panel px-4 py-3">
                    <h3 className="panel-title mb-1.5">{sec.title}</h3>
                    <ul className="list-disc space-y-1 pl-5 font-sans text-[12.5px] leading-relaxed text-ink-700 dark:text-parchment-200">
                      {sec.items.map((it, i) => (
                        <li key={i}>{it}</li>
                      ))}
                    </ul>
                  </section>
                ))}

                <section className="panel px-4 py-3">
                  <div className="mb-1.5 flex items-center justify-between">
                    <h3 className="panel-title">参考文献（已校验真实性）</h3>
                    <span className="font-sans text-[10px] text-ink-400">
                      {framework.references.filter((r) => r.verified).length} / {framework.references.length} 条通过校验
                    </span>
                  </div>
                  <div className="space-y-1.5">
                    {framework.references.map((r, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 rounded-md border border-ink-200 px-2.5 py-1.5 dark:border-ink-700"
                      >
                        <span
                          className={`mt-0.5 shrink-0 text-[11px] ${r.verified ? 'text-emerald-600' : 'text-red-500'}`}
                          title={r.verified ? '已在文献库中验证' : '未通过校验'}
                        >
                          {r.verified ? '✓' : '✗'}
                        </span>
                        <div className="min-w-0">
                          <div className="font-sans text-[11.5px] font-medium text-ink-700 dark:text-parchment-200">
                            {r.title}
                          </div>
                          <div className="font-sans text-[10px] text-ink-400">
                            {r.authors} · {r.year} · {r.venue}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {framework.rejected_references.length > 0 && (
                    <details className="mt-2.5 rounded-md bg-ink-50 px-2.5 py-2 dark:bg-ink-800">
                      <summary className="cursor-pointer font-sans text-[11px] text-ink-500">
                        护栏拦截的幻觉引用（{framework.rejected_references.length}）
                      </summary>
                      <div className="mt-1.5 space-y-1">
                        {framework.rejected_references.map((r, i) => (
                          <div key={i} className="font-sans text-[10.5px] text-ink-500">
                            <span className="font-medium">{r.title}</span> — {r.reason}
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </section>
              </div>
            )}
          </div>
        )}

        {/* ============================================================= diagram */}
        {panel === 'diagram' && (
          <div className="mx-auto max-w-4xl space-y-4">
            <div className="panel px-4 py-3">
              <label className="mb-1 block font-sans text-[10px] uppercase tracking-wider text-ink-400">
                图表描述
              </label>
              <textarea
                className="input resize-none"
                rows={3}
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
                placeholder="例如：Agent Router 接收问题后分流至检索、图谱或写作三个子 Agent，检索结果经 RRF 融合后送入生成器。"
              />
              <div className="mt-2 flex items-center gap-2">
                <select
                  className="input w-40"
                  value={diagramType}
                  onChange={(e) => setDiagramType(e.target.value)}
                >
                  <option value="mermaid">Mermaid</option>
                  <option value="tikz">TikZ (LaTeX)</option>
                  <option value="graphviz">Graphviz (DOT)</option>
                </select>
                <button className="btn-primary" onClick={runDiagram} disabled={busy || desc.trim().length < 5}>
                  {busy ? <Spinner className="h-3.5 w-3.5" /> : null} 生成架构图
                </button>
              </div>
            </div>

            {!diagram ? (
              <EmptyState
                title="尚未生成架构图"
                hint="生成 Mermaid / TikZ / Graphviz 代码。Mermaid 提供实时预览，TikZ 与 DOT 源码可复制到论文。"
              />
            ) : (
              <div className="grid gap-4 lg:grid-cols-2">
                <section className="panel px-4 py-3">
                  <div className="mb-1.5 flex items-center justify-between">
                    <h3 className="panel-title">{diagram.title}</h3>
                    <span className="chip border-ink-200 font-mono text-ink-500 dark:border-ink-700 dark:text-ink-400">
                      {diagram.diagram_type}
                    </span>
                  </div>
                  {diagram.diagram_type === 'mermaid' && (
                    <div
                      ref={svgHost}
                      className="mb-2 flex min-h-[220px] items-center justify-center rounded-lg bg-ink-50 p-3 dark:bg-ink-800"
                    >
                      {svg ? (
                        <div className="w-full [&_svg]:h-auto [&_svg]:max-w-full" dangerouslySetInnerHTML={{ __html: svg }} />
                      ) : (
                        <span className="font-sans text-[11px] text-ink-400">渲染中…</span>
                      )}
                    </div>
                  )}
                  <div className="font-sans text-[11.5px] leading-relaxed text-ink-600 dark:text-ink-300">
                    {diagram.description}
                  </div>
                </section>

                <section className="panel flex flex-col px-4 py-3">
                  <div className="mb-1.5 flex items-center justify-between">
                    <h3 className="panel-title">源码</h3>
                    <div className="flex gap-1.5">
                      <button className="btn-secondary !px-2 !py-1 !text-[10.5px]" onClick={() => copy(diagram.code, '代码')}>
                        复制
                      </button>
                      {svg && (
                        <button className="btn-secondary !px-2 !py-1 !text-[10.5px]" onClick={downloadSvg}>
                          下载 SVG
                        </button>
                      )}
                    </div>
                  </div>
                  <pre className="scroll-thin max-h-[420px] overflow-auto rounded-lg bg-ink-900 p-3 font-mono text-[10.5px] leading-relaxed text-parchment-100">
                    {diagram.code}
                  </pre>
                </section>
              </div>
            )}
          </div>
        )}

        {/* =============================================================== plot */}
        {panel === 'plot' && (
          <div className="mx-auto max-w-4xl space-y-4">
            <div className="panel px-4 py-3">
              <div className="mb-1 flex items-center justify-between">
                <label className="font-sans text-[10px] uppercase tracking-wider text-ink-400">
                  CSV 数据（第一行为表头）
                </label>
                <button
                  className="btn-ghost !px-2 !py-0.5 !text-[10.5px]"
                  onClick={() => setCsv(SAMPLE_CSV)}
                >
                  载入示例
                </button>
              </div>
              <textarea
                className="input resize-none font-mono !text-[11.5px]"
                rows={6}
                value={csv}
                onChange={(e) => setCsv(e.target.value)}
              />
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <select className="input w-40" value={chartType} onChange={(e) => setChartType(e.target.value)}>
                  {CHART_TYPES.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.label}
                    </option>
                  ))}
                </select>
                <button className="btn-primary" onClick={runPlot} disabled={busy}>
                  {busy ? <Spinner className="h-3.5 w-3.5" /> : null} 绘制图表
                </button>
              </div>
            </div>

            {!plot ? (
              <EmptyState
                title="尚未绘图"
                hint="粘贴论文表格的 CSV 数据，系统渲染图表并生成 Figure Caption 与关键结论。"
              />
            ) : (
              <div className="space-y-4">
                <section className="panel px-4 py-3">
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="panel-title">图表预览</h3>
                    <button
                      className="btn-secondary !px-2 !py-1 !text-[10.5px]"
                      onClick={() => copy(plot.caption, 'Figure Caption')}
                    >
                      复制 Caption
                    </button>
                  </div>
                  <div className="flex justify-center overflow-x-auto rounded-lg bg-white p-3 dark:bg-ink-950">
                    <img
                      src={`data:image/png;base64,${plot.image_base64}`}
                      alt={plot.caption}
                      className="max-h-[420px] w-auto"
                    />
                  </div>
                  <p className="mt-2.5 font-serif text-[12.5px] italic leading-relaxed text-ink-700 dark:text-parchment-200">
                    {plot.caption}
                  </p>
                  {plot.key_findings.length > 0 && (
                    <ul className="mt-2 list-disc space-y-0.5 pl-5 font-sans text-[12px] text-ink-600 dark:text-ink-300">
                      {plot.key_findings.map((f, i) => (
                        <li key={i}>{f}</li>
                      ))}
                    </ul>
                  )}
                </section>

                <section className="panel px-4 py-3">
                  <h3 className="panel-title mb-1.5">
                    数据表（{plot.row_count} 行 × {plot.columns.length} 列）
                  </h3>
                  <pre className="scroll-thin overflow-auto rounded-lg bg-ink-50 p-3 font-mono text-[10.5px] dark:bg-ink-800">
                    {plot.table_markdown}
                  </pre>
                </section>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
