/**
 * Module 3 — the layout-aware reader.
 *
 * Three panes:
 *   1. the heading tree (levels 1/2/3) for fast navigation;
 *   2. the original document (rendered page images, faithful to the print
 *      layout of the paper);
 *   3. the bilingual column, where the left side is the English source and the
 *      right side the Chinese translation — the two are **synchronously
 *      scrolled** because both render the same aligned segment list.
 *
 * Citation markers ``[11]`` in the text jump to the bibliography entry, and
 * the bibliography entry can jump back to the sentence that cited it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  api,
  Paper,
  PaperStructure,
  ReferenceItem,
} from '../lib/api'
import { EmptyState, Modal, Spinner, Tabs, Toast } from '../components/ui'
import { useSciStore } from '../store'

type ViewMode = 'pages' | 'bilingual'

interface TranslationSegment {
  index: number
  source: string
  translation: string
  kind: string
  page_no: number
}

interface CitationMarker {
  chunk_id: string
  page_no: number
  char_start: number
  char_end: number
  numbers: number[]
  context: string
}

export default function Reader() {
  const { paperId } = useParams<{ paperId: string }>()
  const navigate = useNavigate()
  const papers = useSciStore((s) => s.selected)
  const activePaperId = useSciStore((s) => s.activePaperId)
  const setActivePaper = useSciStore((s) => s.setActivePaper)

  const id = paperId ?? activePaperId ?? papers[0]?.id ?? null

  const [paper, setPaper] = useState<Paper | null>(null)
  const [structure, setStructure] = useState<PaperStructure | null>(null)
  const [markers, setMarkers] = useState<CitationMarker[]>([])
  const [references, setReferences] = useState<ReferenceItem[]>([])
  const [segments, setSegments] = useState<TranslationSegment[]>([])
  const [mode, setMode] = useState<ViewMode>('pages')
  const [loading, setLoading] = useState(false)
  const [translating, setTranslating] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [activeRef, setActiveRef] = useState<number | null>(null)
  const [selection, setSelection] = useState<{ text: string; x: number; y: number } | null>(null)
  const [translated, setTranslated] = useState<string | null>(null)

  const srcPane = useRef<HTMLDivElement>(null)
  const tgtPane = useRef<HTMLDivElement>(null)
  const lockRef = useRef<'src' | 'tgt' | null>(null)
  const lockTimer = useRef<number | null>(null)

  /* --------------------------------------------------------------- load */
  useEffect(() => {
    if (!id) return
    setLoading(true)
    ;(async () => {
      try {
        const [p, s, c] = await Promise.all([
          api.getPaper(id),
          api.structure(id),
          api.citations(id),
        ])
        setPaper(p)
        setStructure(s)
        setMarkers(c.markers)
        setReferences(c.references)
        setActivePaper(id)
      } catch (err) {
        setToast(`加载失败：${(err as Error).message}`)
      } finally {
        setLoading(false)
      }
    })()
  }, [id, setActivePaper])

  /* --------------------------------------------------- bilingual loading */
  const loadTranslation = useCallback(async () => {
    if (!id) return
    setTranslating(true)
    try {
      const res = await api.translation(id, 'zh')
      setSegments(res.segments)
      setMode('bilingual')
    } catch (err) {
      setToast(`翻译失败：${(err as Error).message}`)
    } finally {
      setTranslating(false)
    }
  }, [id])

  /* ------------------------------------------------------- sync scrolling */
  const syncScroll = useCallback((from: 'src' | 'tgt') => {
    if (lockRef.current && lockRef.current !== from) return
    const src = srcPane.current
    const tgt = tgtPane.current
    if (!src || !tgt) return

    lockRef.current = from
    if (lockTimer.current) window.clearTimeout(lockTimer.current)
    lockTimer.current = window.setTimeout(() => {
      lockRef.current = null
    }, 120)

    const fromEl = from === 'src' ? src : tgt
    const toEl = from === 'src' ? tgt : src

    // Find the first segment that is visible in the source pane and align the
    // target pane to the same segment (both panes share the segment list).
    const items = Array.from(fromEl.querySelectorAll<HTMLElement>('[data-seg]'))
    if (!items.length) return
    const fromTop = fromEl.scrollTop
    let active = items[0]
    for (const el of items) {
      if (el.offsetTop - fromEl.offsetTop <= fromTop + 24) active = el
      else break
    }
    const index = active.dataset.seg
    const counterpart = toEl.querySelector<HTMLElement>(`[data-seg="${index}"]`)
    if (!counterpart) return
    const delta = active.offsetTop - counterpart.offsetTop
    toEl.scrollTop = Math.max(0, toEl.scrollTop + delta)
  }, [])

  /* -------------------------------------------------- citation navigation */
  const jumpToReference = async (n: number) => {
    setActiveRef(n)
    const ref = await api.reference(id!, n)
    const marker = markers.find((m) => m.numbers.includes(n))
    if (marker) {
      // Highlight the first citation site in the text.
      const el = document.querySelector<HTMLElement>(`[data-cite="${n}"]`)
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    } else {
      setToast(`参考文献 [${n}] 未在正文中被引用`)
    }
    return ref
  }

  const jumpBack = async (n: number) => {
    const ref = await api.reference(id!, n)
    const site = ref.cited_at?.[0]
    if (site) {
      const el = document.querySelector<HTMLElement>(`[data-cite="${n}"]`)
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }

  /* ---------------------------------------------------------- selection */
  useEffect(() => {
    const handler = () => {
      const sel = window.getSelection()
      if (!sel || sel.isCollapsed || !sel.toString().trim()) {
        setSelection(null)
        return
      }
      const rect = sel.getRangeAt(0).getBoundingClientRect()
      setSelection({ text: sel.toString().trim(), x: rect.left, y: rect.top })
    }
    document.addEventListener('mouseup', handler)
    return () => document.removeEventListener('mouseup', handler)
  }, [])

  const translateSelection = async () => {
    if (!selection) return
    try {
      const res = await api.translateText(selection.text, 'zh')
      setTranslated(res.translation)
    } catch (err) {
      setToast(`翻译失败：${(err as Error).message}`)
    }
  }

  /* ------------------------------------------------------------- render */
  const headingTree = useMemo(() => structure?.sections ?? [], [structure])

  if (!id) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState
          title="尚未选择文献"
          hint="请先在文献库中选择一篇论文，或直接上传新的 PDF。"
          action={
            <button className="btn-primary mt-2" onClick={() => navigate('/library')}>
              去文献库
            </button>
          }
        />
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* ------------------------------------------------------------ head */}
      <header className="flex items-center justify-between gap-4 border-b border-ink-200/70 px-6 py-3 dark:border-ink-800">
        <div className="min-w-0">
          <h1 className="truncate font-serif text-lg font-semibold tracking-tight">
            {paper?.title ?? '加载中…'}
          </h1>
          <div className="mt-0.5 flex items-center gap-2 font-sans text-[11px] text-ink-400">
            <span>{paper?.page_count ?? 0} 页</span>
            {paper?.is_scanned && <span className="text-accent-600">扫描件 · VLM OCR</span>}
            {paper?.arxiv_id && <span>arXiv:{paper.arxiv_id}{paper.version_label}</span>}
            <span className="truncate">{paper?.metadata?.columns as string}</span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Tabs
            tabs={[
              { id: 'pages' as ViewMode, label: '原文版式' },
              { id: 'bilingual' as ViewMode, label: '中英对照' },
            ]}
            value={mode}
            onChange={(m) => (m === 'bilingual' ? void loadTranslation() : setMode(m))}
          />
          {mode === 'bilingual' && translating && <Spinner className="text-ink-400" />}
        </div>
      </header>

      {toast && (
        <div className="px-6 pt-3">
          <Toast message={toast} kind="error" />
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        {/* ------------------------------------------------------- outline */}
        <aside className="scroll-thin w-64 shrink-0 overflow-y-auto border-r border-ink-200/70 px-3 py-4 dark:border-ink-800">
          <div className="mb-2 px-1 font-sans text-[11px] font-semibold uppercase tracking-wider text-ink-400">
            标题树
          </div>
          {headingTree.length === 0 && (
            <div className="px-1 font-sans text-xs text-ink-400">未解析出标题层级</div>
          )}
          <ul className="space-y-0.5">
            {headingTree.map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => {
                    const el = document.querySelector<HTMLElement>(`[data-section="${s.id}"]`)
                    el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                  }}
                  className={[
                    'block w-full rounded-md px-2 py-1 text-left font-sans transition-colors hover:bg-ink-100 dark:hover:bg-ink-800',
                    s.level === 1
                      ? 'text-[12.5px] font-semibold text-ink-800 dark:text-parchment-100'
                      : s.level === 2
                        ? 'pl-4 text-[12px] text-ink-600 dark:text-ink-300'
                        : 'pl-7 text-[11.5px] text-ink-500 dark:text-ink-400',
                  ].join(' ')}
                >
                  {s.number && <span className="mr-1 font-mono text-[10px] text-accent-600">{s.number}</span>}
                  {s.title}
                </button>
              </li>
            ))}
          </ul>

          {references.length > 0 && (
            <>
              <div className="mb-2 mt-5 px-1 font-sans text-[11px] font-semibold uppercase tracking-wider text-ink-400">
                References ({references.length})
              </div>
              <ul className="space-y-0.5">
                {references.map((r) => (
                  <li key={r.id}>
                    <button
                      onClick={() => void jumpToReference(r.index_in_doc)}
                      className="flex w-full gap-1.5 rounded-md px-2 py-1 text-left font-sans text-[11px] text-ink-600 transition-colors hover:bg-ink-100 dark:text-ink-300 dark:hover:bg-ink-800"
                    >
                      <span className="font-mono text-[10px] text-accent-600">[{r.index_in_doc}]</span>
                      <span className="line-clamp-2">{r.title || r.raw_text}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </aside>

        {/* ------------------------------------------------------ document */}
        <section className="min-w-0 flex-1 overflow-hidden">
          {loading ? (
            <div className="flex h-full items-center justify-center">
              <Spinner className="h-5 w-5 text-ink-400" />
            </div>
          ) : mode === 'pages' ? (
            <PagesView paperId={id} structure={structure} />
          ) : (
            <BilingualView
              segments={segments}
              srcPane={srcPane}
              tgtPane={tgtPane}
              onScrollSrc={() => syncScroll('src')}
              onScrollTgt={() => syncScroll('tgt')}
              onCite={jumpToReference}
            />
          )}
        </section>
      </div>

      {/* ------------------------------------------------- reference modal */}
      <Modal
        open={activeRef !== null}
        onClose={() => setActiveRef(null)}
        title={`参考文献 [${activeRef}]`}
      >
        {activeRef !== null && (
          <ReferenceDetail
            paperId={id}
            index={activeRef}
            onJumpBack={() => {
              setActiveRef(null)
              void jumpBack(activeRef)
            }}
          />
        )}
      </Modal>

      {/* -------------------------------------------- selection translation */}
      {selection && (
        <div
          className="fixed z-40 flex items-center gap-1 rounded-lg border border-ink-200 bg-white px-1.5 py-1 shadow-panel-lg dark:border-ink-700 dark:bg-ink-800"
          style={{ left: Math.min(selection.x, window.innerWidth - 120), top: selection.y - 42 }}
        >
          <button className="btn-ghost !px-2 !py-1 text-[11px]" onClick={translateSelection}>
            翻译选区
          </button>
          <button
            className="btn-ghost !px-2 !py-1 text-[11px]"
            onClick={() => {
              setSelection(null)
              setTranslated(null)
            }}
          >
            ✕
          </button>
        </div>
      )}
      {translated && (
        <div className="fixed bottom-5 left-1/2 z-40 w-[min(680px,90vw)] -translate-x-1/2 rounded-xl border border-ink-200 bg-white p-4 shadow-panel-lg dark:border-ink-700 dark:bg-ink-900">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-sans text-[11px] font-semibold uppercase tracking-wider text-ink-400">
              译文
            </span>
            <button className="btn-ghost !px-1 !py-0" onClick={() => setTranslated(null)}>
              ✕
            </button>
          </div>
          <div className="prose-paper max-h-56 overflow-auto text-[15px]">{translated}</div>
        </div>
      )}
    </div>
  )
}

/* ========================================================================== */
/* Pages view — the original PDF rendered page by page                        */
/* ========================================================================== */
function PagesView({ paperId, structure }: { paperId: string; structure: PaperStructure | null }) {
  const [pages, setPages] = useState<{ page_no: number; image_url: string }[]>([])
  const [page, setPage] = useState(1)

  useEffect(() => {
    void api.pages(paperId).then((r) => setPages(r.pages)).catch(() => setPages([]))
  }, [paperId])

  const current = pages.find((p) => p.page_no === page)

  return (
    <div className="flex h-full">
      <div className="scroll-thin flex-1 overflow-auto bg-ink-100/60 p-6 dark:bg-ink-950">
        <div className="mx-auto max-w-4xl space-y-6">
          {structure?.sections.map((s) => (
            <div key={s.id} data-section={s.id} className="panel px-6 py-5">
              <h2
                className={[
                  'font-serif font-semibold text-ink-900 dark:text-parchment-50',
                  s.level === 1 ? 'text-xl' : s.level === 2 ? 'text-lg' : 'text-base',
                ].join(' ')}
              >
                {s.number && <span className="mr-2 font-mono text-accent-600">{s.number}</span>}
                {s.title}
              </h2>
              {s.content && (
                <div className="prose-paper mt-3 max-h-96 overflow-hidden">{s.content}</div>
              )}
            </div>
          ))}

          {structure?.tables.map((t) => (
            <div key={t.id} className="panel px-5 py-4">
              {t.caption && (
                <div className="mb-1 font-sans text-xs text-ink-500">Table · p.{t.page_no}</div>
              )}
              <div className="prose-paper text-[14px]" dangerouslySetInnerHTML={{ __html: mdToHtml(t.markdown) }} />
            </div>
          ))}
        </div>
      </div>

      <aside className="scroll-thin w-56 shrink-0 overflow-y-auto border-l border-ink-200/70 bg-white px-3 py-4 dark:border-ink-800 dark:bg-ink-900">
        <div className="mb-2 font-sans text-[11px] font-semibold uppercase tracking-wider text-ink-400">
          页面
        </div>
        <div className="space-y-1.5">
          {pages.map((p) => (
            <button
              key={p.page_no}
              onClick={() => setPage(p.page_no)}
              className={[
                'block w-full overflow-hidden rounded-md border text-left transition-colors',
                p.page_no === page
                  ? 'border-accent-400 ring-1 ring-accent-400/50'
                  : 'border-ink-200 hover:border-ink-300 dark:border-ink-700',
              ].join(' ')}
            >
              {p.image_url ? (
                <img src={p.image_url} alt={`page ${p.page_no}`} className="w-full" loading="lazy" />
              ) : (
                <div className="flex h-24 items-center justify-center font-sans text-[10px] text-ink-400">
                  p.{p.page_no}
                </div>
              )}
              <div className="px-1.5 py-0.5 font-sans text-[10px] text-ink-400">p.{p.page_no}</div>
            </button>
          ))}
        </div>
        {current && (
          <a
            className="btn-secondary mt-3 w-full !py-1 text-[11px]"
            href={`/api/reader/${paperId}/page/${current.page_no}/image`}
            target="_blank"
            rel="noreferrer"
          >
            打开原图
          </a>
        )}
      </aside>
    </div>
  )
}

/* ========================================================================== */
/* Bilingual view — synchronised scrolling                                     */
/* ========================================================================== */
function BilingualView({
  segments,
  srcPane,
  tgtPane,
  onScrollSrc,
  onScrollTgt,
  onCite,
}: {
  segments: TranslationSegment[]
  srcPane: React.RefObject<HTMLDivElement>
  tgtPane: React.RefObject<HTMLDivElement>
  onScrollSrc: () => void
  onScrollTgt: () => void
  onCite: (n: number) => void
}) {
  if (!segments.length) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState title="正在准备对照翻译…" hint="后端会按段落对齐原文与译文，代码与公式保持英文原文。" />
      </div>
    )
  }

  return (
    <div className="grid h-full grid-cols-2 divide-x divide-ink-200/70 dark:divide-ink-800">
      <div ref={srcPane} onScroll={onScrollSrc} className="scroll-thin overflow-y-auto px-8 py-6">
        <div className="mb-4 font-sans text-[11px] font-semibold uppercase tracking-wider text-ink-400">
          Original (EN)
        </div>
        {segments.map((s) => (
          <article
            key={s.index}
            data-seg={s.index}
            className={[
              'mb-5 border-l-2 border-transparent pl-3 transition-colors',
              s.kind === 'heading' ? 'font-serif text-[15px] font-semibold' : 'prose-paper text-[15px]',
            ].join(' ')}
          >
            <CitationText text={s.source} onCite={onCite} />
          </article>
        ))}
      </div>

      <div ref={tgtPane} onScroll={onScrollTgt} className="scroll-thin overflow-y-auto px-8 py-6">
        <div className="mb-4 font-sans text-[11px] font-semibold uppercase tracking-wider text-ink-400">
          译文 (中文)
        </div>
        {segments.map((s) => (
          <article
            key={s.index}
            data-seg={s.index}
            className={[
              'mb-5 border-l-2 border-transparent pl-3 transition-colors',
              s.kind === 'heading'
                ? 'font-serif text-[15px] font-semibold text-ink-900 dark:text-parchment-50'
                : 'prose-paper text-[15px]',
            ].join(' ')}
          >
            <CitationText text={s.translation} onCite={onCite} />
          </article>
        ))}
      </div>
    </div>
  )
}

/* ========================================================================== */
/* Citation-aware text                                                        */
/* ========================================================================== */
export function CitationText({
  text,
  onCite,
}: {
  text: string
  onCite?: (n: number) => void
}) {
  const parts = text.split(/(\[\d{1,3}(?:[,;\-–]\s*\d{1,3})*\])/g)
  return (
    <>
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d{1,3})/)
        if (!m) return <span key={i}>{part}</span>
        const n = Number(m[1])
        return (
          <button
            key={i}
            data-cite={n}
            onClick={() => onCite?.(n)}
            className="citation-chip"
            title={`跳转到参考文献 [${n}]`}
          >
            {n}
          </button>
        )
      })}
    </>
  )
}

/* ========================================================================== */
function ReferenceDetail({
  paperId,
  index,
  onJumpBack,
}: {
  paperId: string
  index: number
  onJumpBack: () => void
}) {
  const [ref, setRef] = useState<ReferenceItem | null>(null)
  const [citedAt, setCitedAt] = useState<{ context: string; page_no: number }[]>([])

  useEffect(() => {
    void api.reference(paperId, index).then((r) => {
      setRef(r)
      setCitedAt(r.cited_at ?? [])
    })
  }, [paperId, index])

  if (!ref) return <Spinner />

  return (
    <div className="space-y-3">
      <div className="prose-paper text-[14px]">{ref.raw_text}</div>
      <div className="grid grid-cols-2 gap-2 font-sans text-xs text-ink-500">
        {ref.authors?.length > 0 && (
          <div>
            <span className="text-ink-400">作者：</span>
            {ref.authors.join(', ')}
          </div>
        )}
        {ref.year && (
          <div>
            <span className="text-ink-400">年份：</span>
            {ref.year}
          </div>
        )}
        {ref.venue && (
          <div>
            <span className="text-ink-400">发表：</span>
            {ref.venue}
          </div>
        )}
        {ref.arxiv_id && (
          <div>
            <span className="text-ink-400">arXiv：</span>
            {ref.arxiv_id}
          </div>
        )}
      </div>
      {citedAt.length > 0 && (
        <div>
          <div className="mb-1 font-sans text-[11px] font-semibold uppercase tracking-wider text-ink-400">
            正文引用位置
          </div>
          {citedAt.map((c, i) => (
            <div
              key={i}
              className="mb-1 rounded-md bg-ink-50 px-2.5 py-1.5 font-sans text-[11px] leading-relaxed text-ink-600 dark:bg-ink-800 dark:text-ink-300"
            >
              <span className="mr-1 font-mono text-[10px] text-accent-600">p.{c.page_no}</span>
              {c.context}
            </div>
          ))}
          <button className="btn-secondary mt-2 !py-1 text-[11px]" onClick={onJumpBack}>
            ↩ 跳回正文引用位置
          </button>
        </div>
      )}
    </div>
  )
}

/* --------------------------------------------------------------- helpers */
/** A tiny Markdown-table → HTML converter (tables only). */
function mdToHtml(md: string): string {
  const lines = md.split('\n').filter(Boolean)
  if (!lines.length) return ''
  const head = lines[0]
    .replace(/^\||\|$/g, '')
    .split('|')
    .map((c) => c.trim())
  const body = lines.slice(2)
  const headHtml = `<thead><tr>${head.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>`
  const bodyHtml = `<tbody>${body
    .map((row) => {
      const cells = row
        .replace(/^\||\|$/g, '')
        .split('|')
        .map((c) => c.trim())
      return `<tr>${cells.map((c) => `<td>${escapeHtml(c)}</td>`).join('')}</tr>`
    })
    .join('')}</tbody>`
  return `<table>${headHtml}${bodyHtml}</table>`
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
}
