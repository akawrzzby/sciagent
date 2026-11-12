/**
 * Module 4 — the arXiv search / tracking / recommendation workbench.
 *
 * The three panels cover the full "find → judge → import" loop that a
 * researcher actually performs:
 *   • arXiv 检索     — keyword search against the live arXiv API;
 *   • 智能推荐       — candidates ranked by content relevance, citation-graph
 *                      position, **reproducibility** (public GitHub repo) and
 *                      accumulated user feedback;
 *   • 最新追踪       — the newest submissions of an arXiv category.
 */

import { useEffect, useState } from 'react'
import { api, ArxivPaper, Recommendation } from '../lib/api'
import { EmptyState, Spinner, Tabs, Toast } from '../components/ui'

type Panel = 'arxiv' | 'recommend' | 'recent'

const CATEGORIES = ['cs.CL', 'cs.LG', 'cs.AI', 'cs.CV', 'cs.IR', 'stat.ML', 'cs.SE']

const EXAMPLES = [
  'retrieval augmented generation',
  'large language model agents',
  'chain of thought reasoning',
  'mixture of experts',
]

export default function SearchWorkbench() {
  const [panel, setPanel] = useState<Panel>('arxiv')
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('cs.CL')
  const [results, setResults] = useState<ArxivPaper[]>([])
  const [recs, setRecs] = useState<Recommendation[]>([])
  const [librarySize, setLibrarySize] = useState(0)
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState<{ msg: string; kind: 'info' | 'error' | 'ok' } | null>(null)
  const [downloaded, setDownloaded] = useState<Record<string, string>>({})

  useEffect(() => {
    if (panel === 'recommend') void loadRecommendations()
    if (panel === 'recent') void loadRecent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panel])

  const loadRecommendations = async () => {
    setBusy(true)
    setToast(null)
    try {
      const r = await api.recommend()
      setRecs(r.recommendations)
      setLibrarySize(r.library_size)
    } catch (e) {
      setToast({ msg: `推荐失败：${(e as Error).message}`, kind: 'error' })
    } finally {
      setBusy(false)
    }
  }

  const loadRecent = async () => {
    setBusy(true)
    setToast(null)
    try {
      const r = await api.recent(category, 30)
      setResults(r.papers)
    } catch (e) {
      setToast({ msg: `拉取最新列表失败：${(e as Error).message}`, kind: 'error' })
    } finally {
      setBusy(false)
    }
  }

  const doSearch = async () => {
    const q = query.trim()
    if (!q) return
    setBusy(true)
    setToast(null)
    try {
      const r = await api.arxivSearch(q, 15)
      setResults(r.papers)
      if (!r.papers.length) setToast({ msg: 'arXiv 没有返回匹配结果，换个关键词试试。', kind: 'info' })
    } catch (e) {
      setToast({ msg: `检索失败：${(e as Error).message}`, kind: 'error' })
    } finally {
      setBusy(false)
    }
  }

  const download = async (arxivId: string) => {
    setBusy(true)
    setToast(null)
    try {
      const r = await api.downloadPaper(arxivId)
      setDownloaded((d) => ({ ...d, [arxivId]: r.paper_id }))
      setToast({
        msg: r.duplicate
          ? `该论文已在库中（${r.paper_id}），未重复导入。`
          : `已导入文献库：${r.paper_id}，正在后台解析。`,
        kind: r.duplicate ? 'info' : 'ok',
      })
    } catch (e) {
      setToast({ msg: `下载失败：${(e as Error).message}`, kind: 'error' })
    } finally {
      setBusy(false)
    }
  }

  const rate = async (rec: Recommendation, rating: number) => {
    try {
      await api.feedback(rec.arxiv_id, rating, rec.title)
      setRecs((list) =>
        list.map((r) => (r.arxiv_id === rec.arxiv_id ? { ...r, feedback: r.feedback + rating } : r)),
      )
      setToast({ msg: rating > 0 ? '已记录 👍，将提升该论文的排序。' : '已记录 👎，将降低该论文的排序。', kind: 'ok' })
    } catch (e) {
      setToast({ msg: `反馈失败：${(e as Error).message}`, kind: 'error' })
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-200/70 px-6 py-3.5 dark:border-ink-800">
        <div>
          <h1 className="font-serif text-xl font-semibold tracking-tight">检索推荐</h1>
          <p className="mt-0.5 font-sans text-xs text-ink-500 dark:text-ink-400">
            arXiv 检索 · 可复现性加权推荐 · 分类追踪 · Function Calling 工具闭环
          </p>
        </div>
        <Tabs<Panel>
          value={panel}
          onChange={setPanel}
          tabs={[
            { id: 'arxiv', label: 'arXiv 检索' },
            { id: 'recommend', label: '智能推荐' },
            { id: 'recent', label: '最新追踪' },
          ]}
        />
      </header>

      {toast && (
        <div className="px-6 pt-3">
          <Toast message={toast.msg} kind={toast.kind} />
        </div>
      )}

      {/* ------------------------------------------------------------ controls */}
      <div className="border-b border-ink-200/70 px-6 py-3 dark:border-ink-800">
        {panel === 'arxiv' && (
          <div className="flex items-center gap-2">
            <input
              className="input flex-1"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && doSearch()}
              placeholder="输入检索式，例如：retrieval augmented generation"
            />
            <button className="btn-primary" onClick={doSearch} disabled={busy || !query.trim()}>
              {busy ? <Spinner className="h-3.5 w-3.5" /> : '检索'}
            </button>
          </div>
        )}
        {panel === 'recommend' && (
          <div className="flex items-center justify-between">
            <p className="font-sans text-xs text-ink-500 dark:text-ink-400">
              排序信号：内容相关度 0.6 + 开源代码 0.25 + 用户反馈 ≤0.15
              {librarySize > 0 && ` · 当前文献库 ${librarySize} 篇`}
            </p>
            <button className="btn-secondary !text-[11px]" onClick={loadRecommendations} disabled={busy}>
              {busy ? <Spinner className="h-3.5 w-3.5" /> : '⟳'} 刷新推荐
            </button>
          </div>
        )}
        {panel === 'recent' && (
          <div className="flex items-center gap-2">
            <select
              className="input w-40"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <button className="btn-primary" onClick={loadRecent} disabled={busy}>
              {busy ? <Spinner className="h-3.5 w-3.5" /> : '拉取最新'}
            </button>
            <span className="font-sans text-[11px] text-ink-400">用于定期追踪研究前沿</span>
          </div>
        )}
      </div>

      {/* --------------------------------------------------------------- body */}
      <div className="scroll-thin flex-1 overflow-y-auto px-6 py-4">
        {panel === 'arxiv' && results.length === 0 && !busy && (
          <div className="space-y-3">
            <EmptyState
              title="检索 arXiv"
              hint="直接检索 arXiv 官方 API，结果可一键下载并自动解析入库。"
            />
            <div className="mx-auto flex max-w-2xl flex-wrap justify-center gap-2">
              {EXAMPLES.map((e) => (
                <button
                  key={e}
                  className="chip border-ink-200 bg-white text-ink-600 hover:border-accent-400 dark:border-ink-700 dark:bg-ink-900 dark:text-ink-300"
                  onClick={() => {
                    setQuery(e)
                    setTimeout(doSearch, 0)
                  }}
                >
                  {e}
                </button>
              ))}
            </div>
          </div>
        )}

        {panel === 'arxiv' && results.length > 0 && (
          <div className="mx-auto max-w-4xl space-y-2.5">
            <div className="font-sans text-[11px] text-ink-400">共 {results.length} 条结果</div>
            {results.map((p) => (
              <article key={p.arxiv_id} className="panel px-4 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="font-serif text-[14px] font-semibold leading-snug text-ink-900 dark:text-parchment-50">
                      {p.title}
                    </h3>
                    <div className="mt-1 font-sans text-[11px] text-ink-500 dark:text-ink-400">
                      {p.authors.slice(0, 6).join(', ')}
                      {p.authors.length > 6 && ` 等 ${p.authors.length} 人`}
                      {' · '}
                      <span className="font-mono">{p.version}</span>
                      {' · '}
                      {p.published.slice(0, 10)}
                    </div>
                    <p className="mt-1.5 line-clamp-3 font-sans text-[12px] leading-relaxed text-ink-600 dark:text-ink-300">
                      {p.summary}
                    </p>
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {p.categories.map((c) => (
                        <span key={c} className="chip border-ink-200 text-ink-400 dark:border-ink-700">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-col gap-1.5">
                    <button
                      className="btn-secondary !text-[11px]"
                      onClick={() => download(p.arxiv_id)}
                      disabled={busy}
                    >
                      {downloaded[p.arxiv_id] ? '已导入 ✓' : '下载入库'}
                    </button>
                    <a
                      href={p.abs_url}
                      target="_blank"
                      rel="noreferrer"
                      className="btn-ghost !text-[11px]"
                    >
                      arXiv ↗
                    </a>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}

        {panel === 'recommend' && recs.length === 0 && !busy && (
          <EmptyState
            title="暂无推荐"
            hint="推荐引擎需要文献库作为画像来源：先上传几篇论文，或使用「arXiv 检索」下载几篇后回来刷新。"
          />
        )}

        {panel === 'recommend' && recs.length > 0 && (
          <div className="mx-auto max-w-4xl space-y-2.5">
            {recs.map((r) => (
              <article key={r.arxiv_id} className="panel px-4 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-serif text-[14px] font-semibold leading-snug text-ink-900 dark:text-parchment-50">
                        {r.title}
                      </h3>
                      <span className="chip border-accent-200 bg-accent-50 font-mono text-accent-700 dark:border-accent-800 dark:bg-accent-900/40 dark:text-accent-300">
                        score {r.score.toFixed(3)}
                      </span>
                    </div>
                    <div className="mt-1 font-sans text-[11px] text-ink-500 dark:text-ink-400">
                      {r.authors.slice(0, 5).join(', ')} · {r.published.slice(0, 10)}
                    </div>
                    <p className="mt-1.5 line-clamp-2 font-sans text-[12px] leading-relaxed text-ink-600 dark:text-ink-300">
                      {r.summary}
                    </p>
                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      <span className="chip border-ink-200 text-ink-500 dark:border-ink-700 dark:text-ink-400">
                        相关度 {r.relevance.toFixed(2)}
                      </span>
                      {r.has_code && r.repository && (
                        <a
                          href={r.repository.html_url}
                          target="_blank"
                          rel="noreferrer"
                          className="chip border-teal-500/40 bg-teal-500/10 text-teal-600 hover:underline dark:text-teal-500"
                        >
                          ⌘ {r.repository.full_name} ★{r.repository.stars}
                        </a>
                      )}
                      <span className="chip border-ink-200 text-ink-500 dark:border-ink-700 dark:text-ink-400">
                        {r.reason}
                      </span>
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-col items-center gap-1.5">
                    <div className="flex gap-1">
                      <button
                        className="btn-ghost !px-1.5 !py-0.5 !text-[13px]"
                        title="推荐"
                        onClick={() => rate(r, 1)}
                      >
                        👍
                      </button>
                      <button
                        className="btn-ghost !px-1.5 !py-0.5 !text-[13px]"
                        title="不推荐"
                        onClick={() => rate(r, -1)}
                      >
                        👎
                      </button>
                    </div>
                    <button
                      className="btn-secondary !text-[11px]"
                      onClick={() => download(r.arxiv_id)}
                      disabled={busy}
                    >
                      {downloaded[r.arxiv_id] ? '已导入 ✓' : '下载入库'}
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}

        {panel === 'recent' && results.length === 0 && !busy && (
          <EmptyState title="尚未拉取" hint="选择一个 arXiv 分类，点击「拉取最新」查看最新提交的论文。" />
        )}

        {panel === 'recent' && results.length > 0 && (
          <div className="mx-auto max-w-4xl space-y-1.5">
            <div className="font-sans text-[11px] text-ink-400">
              {category} 最新 {results.length} 篇
            </div>
            {results.map((p) => (
              <div
                key={p.arxiv_id}
                className="panel flex items-start justify-between gap-3 px-3.5 py-2.5"
              >
                <div className="min-w-0">
                  <a
                    href={p.abs_url}
                    target="_blank"
                    rel="noreferrer"
                    className="font-serif text-[13px] font-semibold leading-snug text-ink-900 hover:underline dark:text-parchment-50"
                  >
                    {p.title}
                  </a>
                  <div className="mt-0.5 font-sans text-[10.5px] text-ink-400">
                    {p.authors.slice(0, 4).join(', ')} · {p.published.slice(0, 10)} ·{' '}
                    <span className="font-mono">{p.arxiv_id}</span>
                  </div>
                </div>
                <button
                  className="btn-secondary shrink-0 !text-[11px]"
                  onClick={() => download(p.arxiv_id)}
                  disabled={busy}
                >
                  {downloaded[p.arxiv_id] ? '已导入 ✓' : '入库'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* --------------------------------------------------------------- footer */}
      <footer className="border-t border-ink-200/70 px-6 py-2 dark:border-ink-800">
        <div className="flex items-center gap-2 font-sans text-[10.5px] text-ink-400">
          <span className="kbd">arXiv API</span>
          <span className="kbd">GitHub Search API</span>
          <span className="kbd">Semantic Scholar</span>
          <span className="ml-auto">所有外部调用均由 Agent 通过 Function Calling 调度，可在「智能问答 → 思考过程」中审计</span>
        </div>
      </footer>
    </div>
  )
}
