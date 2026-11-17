/**
 * Module 3 — the reading surface.
 *
 * One document model, two modes:
 *
 *   · 英文原文    the parsed document, English only;
 *   · 中英对照    the same document with a Chinese translation layer.  Every
 *                 Chinese sentence reveals the aligned English sentence on
 *                 hover, so the translation never hides the source.
 *
 * The document is assembled from the parsed blocks (text), the figures and
 * the tables, and ordered by their position on the page — a figure appears
 * where it sits in the paper instead of in a list at the end.
 *
 * Citation markers ``[11]`` in the text jump to the bibliography entry, and
 * the bibliography entry can jump back to the sentence that cited it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  api,
  Block,
  FigureItem,
  FormulaItem,
  Paper,
  PaperStructure,
  ReferenceItem,
  Section,
  TableItem,
} from '../lib/api'
import { EmptyState, Modal, Spinner, Tabs, Toast } from '../components/ui'
import { DocumentView } from '../components/document'
import type { PaperDocument } from '../lib/document'
import { useSciStore } from '../store'

type ViewMode = 'en' | 'zh'

interface TranslationSegment {
  index: number
  source: string
  translation: string
  kind: string
  page_no: number
  block_id?: string
  block_order?: number
}

interface CitationMarker {
  chunk_id: string
  page_no: number
  char_start: number
  char_end: number
  numbers: number[]
  context: string
}

interface TranslationState {
  status: 'idle' | 'running' | 'ready' | 'error'
  progress: number
  message: string
  error: string
}

const IDLE_TRANSLATION: TranslationState = {
  status: 'idle',
  progress: 0,
  message: '',
  error: '',
}

/* ========================================================================== */
/* helpers                                                                     */
/* ========================================================================== */

/** Join hard-wrapped PDF lines and repair hyphenated line breaks. */
function normaliseProse(text: string): string {
  if (!text) return ''
  return text
    .replace(/\u00a0/g, ' ')
    .replace(/(?<![-\w])([A-Za-z]{2,})-\n([a-z]{2,})/g, '$1$2')
    .replace(/\s*\n\s*/g, ' ')
    .replace(/[ \t]{2,}/g, ' ')
    .trim()
}

function splitParagraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((p) => normaliseProse(p))
    .filter((p) => p.length > 0)
}

function bboxTop(bbox?: number[]): number {
  return bbox && bbox.length >= 2 ? bbox[1] : 0
}

/**
 * Find the caption of a figure or a table.
 *
 * The parser classifies the caption paragraphs ("Figure 1: …", "Table 2: …")
 * as their own block type, but the link between a caption and the object it
 * describes is a matter of position on the page: a figure caption sits below
 * the figure, a table caption above the table.  The backend's own caption
 * field is used when it is filled in; otherwise the nearest caption block on
 * the same page wins.
 */
function pickCaption(
  media: { page_no: number; bbox?: number[]; caption?: string },
  kind: 'figure' | 'table',
  captions: Block[],
): string {
  const existing = (media.caption ?? '').trim()
  if (existing) return existing

  const top = media.bbox && media.bbox.length >= 2 ? media.bbox[1] : undefined
  const bottom = media.bbox && media.bbox.length >= 4 ? media.bbox[3] : undefined
  const wanted = kind === 'figure' ? /^(figure|fig\.)\s*\d+/i : /^table\s*\d+/i

  let best: { text: string; distance: number } | null = null
  for (const block of captions) {
    if (block.page_no !== media.page_no) continue
    const text = (block.content ?? '').trim()
    if (!text || !wanted.test(text)) continue
    const y0 = block.bbox && block.bbox.length >= 2 ? block.bbox[1] : 0
    const y1 = block.bbox && block.bbox.length >= 4 ? block.bbox[3] : y0

    let distance: number | null = null
    if (kind === 'figure' && bottom !== undefined && y0 >= bottom - 8) {
      distance = y0 - bottom
    } else if (kind === 'table' && top !== undefined && y1 <= top + 8) {
      distance = top - y1
    }
    if (distance === null || distance > 160) continue
    if (!best || distance < best.distance) best = { text, distance }
  }
  return best?.text ?? ''
}

/* ========================================================================== */
/* the document model                                                          */
/* ========================================================================== */

type DocItem =
  | { kind: 'heading'; key: string; section: Section; page_no: number; top: number }
  | {
      kind: 'paragraph'
      key: string
      page_no: number
      top: number
      text: string
      segments: TranslationSegment[]
    }
  | { kind: 'figure'; key: string; page_no: number; top: number; figure: FigureItem }
  | { kind: 'table'; key: string; page_no: number; top: number; table: TableItem }
  | { kind: 'formula'; key: string; page_no: number; top: number; formula: FormulaItem }

interface DocModel {
  sections: { section: Section; items: DocItem[] }[]
  footnotes: { page_no: number; text: string }[]
  preamble: DocItem[]
}

/** The section a page (or a block) belongs to: the last one that starts there. */
function sectionForPage(sections: Section[], page: number): Section | undefined {
  let found: Section | undefined
  for (const section of sections) {
    const start = section.page_start ?? section.page_no ?? 0
    if (start <= page) found = section
    else break
  }
  return found ?? sections[0]
}

function buildDocument(
  structure: PaperStructure | null,
  segments: TranslationSegment[],
): DocModel {
  const sections = [...(structure?.sections ?? [])].sort(
    (a, b) => (a.order_index ?? 0) - (b.order_index ?? 0),
  )
  const blocks = structure?.blocks ?? []
  // The title of the paper is printed on the first page and is parsed as a
  // heading of its own.  The masthead already shows it, so the duplicate is
  // dropped here.
  const paperTitle = normaliseProse(structure?.title ?? '').toLowerCase()

  // One English/Chinese pairing per sentence, grouped by the block it came from.
  const byBlock = new Map<string, TranslationSegment[]>()
  for (const segment of segments) {
    if (!segment.block_id) continue
    const list = byBlock.get(segment.block_id)
    if (list) list.push(segment)
    else byBlock.set(segment.block_id, [segment])
  }

  const footnotes: { page_no: number; text: string }[] = []
  const proseBlocks: Block[] = []

  // The captions of the figures and the tables are shown with the object they
  // belong to.  The caption block must not become a paragraph as well, or the
  // text of the paper would repeat every caption.
  const captionBlocks = blocks.filter((block) => (block.type || '').toLowerCase() === 'caption')
  const media: DocItem[] = [
    ...(structure?.figures ?? []).map((figure) => ({
      kind: 'figure' as const,
      key: figure.id,
      page_no: figure.page_no,
      top: bboxTop(figure.bbox),
      figure: { ...figure, caption: pickCaption(figure, 'figure', captionBlocks) },
    })),
    ...(structure?.tables ?? []).map((table) => ({
      kind: 'table' as const,
      key: table.id,
      page_no: table.page_no,
      top: bboxTop(table.bbox),
      table: { ...table, caption: pickCaption(table, 'table', captionBlocks) },
    })),
    // A display equation is shown as the typeset image of the equation: the
    // extracted text of a formula is flattened and unreadable.
    ...(structure?.formulas ?? []).map((formula) => ({
      kind: 'formula' as const,
      key: formula.id,
      page_no: formula.page_no,
      top: bboxTop(formula.bbox),
      formula,
    })),
  ]
  const usedCaptions = new Set<string>()
  for (const item of media) {
    const caption =
      item.kind === 'figure' ? item.figure.caption : item.kind === 'table' ? item.table.caption : ''
    const text = normaliseProse(caption ?? '').toLowerCase()
    if (text) usedCaptions.add(text)
  }

  // The parser labels every block: ``body`` is running prose, ``text`` is a
  // short fragment (a table cell, a page number) that must not become a
  // paragraph of its own.  Older data has no labels at all, in which case
  // everything that is not excluded is treated as prose.
  const typed = blocks.some((block) => (block.type || '').toLowerCase() === 'body')
  // A table's text is also emitted as a body block, and it arrives flattened
  // ("Average System MNLI-(m/mm) QQP ...") because a PDF has no cell structure
  // in its text layer.  The table itself is shown as the crop of the page, so
  // any block whose centre falls inside a table, figure or formula box is
  // dropped: it is the text of an object that is already on the page.
  const mediaBoxes = media
    .map((item) => item.kind === 'figure' ? item.figure.bbox
      : item.kind === 'table' ? item.table.bbox
      : item.kind === 'formula' ? item.formula.bbox
      : undefined)
    .filter((box): box is number[] => Array.isArray(box) && box.length >= 4)

  const insideMedia = (box?: number[]): boolean => {
    if (!box || box.length < 4) return false
    const cx = (box[0] + box[2]) / 2
    const cy = (box[1] + box[3]) / 2
    return mediaBoxes.some(
      (outer) => cx >= outer[0] - 2 && cx <= outer[2] + 2 && cy >= outer[1] - 2 && cy <= outer[3] + 2,
    )
  }

  for (const block of blocks) {
    const type = (block.type || '').toLowerCase()
    if (type === 'footnote') {
      const text = normaliseProse(block.content)
      if (text) footnotes.push({ page_no: block.page_no, text })
      continue
    }
    // Figure labels and running headers are not part of the prose.
    if (type === 'figure_label' || type === 'header') continue
    if (typed && type !== 'body' && type !== 'caption') continue
    // The title of the paper is not a paragraph of the paper.
    const text = normaliseProse(block.content)
    if (paperTitle && text.toLowerCase() === paperTitle) continue
    // A caption that belongs to a figure or a table is shown with it.
    if (usedCaptions.has(text.toLowerCase())) continue
    // The text of a table or a figure is already shown as the object itself.
    if (type !== 'caption' && insideMedia(block.bbox as number[] | undefined)) continue
    proseBlocks.push(block)
  }

  // Fall back to the section blobs when the backend has no block stream.
  if (!blocks.length) {
    const items: DocItem[] = []
    for (const section of sections) {
      for (const paragraph of splitParagraphs(section.content ?? '')) {
        items.push({
          kind: 'paragraph',
          key: `${section.id}-${items.length}`,
          page_no: section.page_no ?? 0,
          top: 0,
          text: paragraph,
          segments: [],
        })
      }
    }
    const figures = (structure?.figures ?? []).map((figure) => ({
      kind: 'figure' as const,
      key: figure.id,
      page_no: figure.page_no,
      top: bboxTop(figure.bbox),
      figure,
    }))
    const tables = (structure?.tables ?? []).map((table) => ({
      kind: 'table' as const,
      key: table.id,
      page_no: table.page_no,
      top: bboxTop(table.bbox),
      table,
    }))
    const formulas = (structure?.formulas ?? []).map((formula) => ({
      kind: 'formula' as const,
      key: formula.id,
      page_no: formula.page_no,
      top: bboxTop(formula.bbox),
      formula,
    }))
    const first = sections[0]
    // No sections at all: there is no document to lay out.
    if (!first) {
      return { sections: [], footnotes, preamble: [...items, ...figures, ...tables, ...formulas] }
    }
    return {
      sections: [{ section: first, items: [...items, ...figures, ...tables, ...formulas] }],
      footnotes,
      preamble: [],
    }
  }

  // Group the blocks by their section.
  const blocksBySection = new Map<string, Block[]>()
  const orphans: Block[] = []
  for (const block of proseBlocks) {
    const section = block.section_id
      ? sections.find((s) => s.id === block.section_id)
      : sectionForPage(sections, block.page_no)
    if (!section) {
      orphans.push(block)
      continue
    }
    const list = blocksBySection.get(section.id)
    if (list) list.push(block)
    else blocksBySection.set(section.id, [block])
  }

  // Figures and tables belong to the section that is open on their page.
  const mediaBySection = new Map<string, DocItem[]>()
  for (const item of media) {
    const section = sectionForPage(sections, item.page_no)
    if (!section) {
      orphans.push(item as unknown as Block)
      continue
    }
    const list = mediaBySection.get(section.id)
    if (list) list.push(item)
    else mediaBySection.set(section.id, [item])
  }

  const out: DocModel['sections'] = []
  for (const section of sections) {
    // The title of the paper opens a section of its own; the masthead shows
    // it already.
    if (paperTitle && normaliseProse(section.title ?? '').toLowerCase() === paperTitle) {
      continue
    }
    const items: DocItem[] = [
      {
        kind: 'heading',
        key: `h-${section.id}`,
        section,
        page_no: section.page_start ?? section.page_no ?? 0,
        top: bboxTop(section.bbox),
      },
      ...(blocksBySection.get(section.id) ?? []).map((block) => ({
        kind: 'paragraph' as const,
        key: block.id,
        page_no: block.page_no,
        top: bboxTop(block.bbox),
        text: normaliseProse(block.content),
        segments: byBlock.get(block.id) ?? [],
      })),
      ...(mediaBySection.get(section.id) ?? []),
    ]
    items.sort((a, b) => {
      if (a.kind === 'heading') return -1
      if (b.kind === 'heading') return 1
      if (a.page_no !== b.page_no) return a.page_no - b.page_no
      return a.top - b.top
    })
    out.push({ section, items })
  }

  // A section that carries no visible content is dropped: the paper title
  // becomes a heading of its own, and the bibliography entries are rendered
  // in the sidebar instead of the body.
  const filled = out.filter(({ items }) => items.some((item) => item.kind !== 'heading'))

  // Blocks that no heading owns (the text before the first heading) are shown
  // ahead of the sections rather than dropped.
  const preamble: DocItem[] = orphans.map((entry, i) => {
    const anyEntry = entry as unknown as Partial<DocItem> & Block
    if (anyEntry.kind === 'figure' || anyEntry.kind === 'table' || anyEntry.kind === 'formula') {
      return { ...anyEntry, key: `${anyEntry.kind}-${i}` } as DocItem
    }
    return {
      kind: 'paragraph' as const,
      key: entry.id,
      page_no: entry.page_no,
      top: bboxTop(entry.bbox),
      text: normaliseProse(entry.content),
      segments: byBlock.get(entry.id) ?? [],
    }
  })

  return { sections: filled, footnotes, preamble }
}

/* ========================================================================== */
/* the page                                                                    */
/* ========================================================================== */

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
  const [mode, setMode] = useState<ViewMode>('en')
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [activeRef, setActiveRef] = useState<number | null>(null)
  const [translation, setTranslation] = useState<TranslationState>(IDLE_TRANSLATION)
  const [tip, setTip] = useState<{ text: string; top: number; left: number } | null>(null)
  // The paper is rendered from its parsed structure: tables become real
  // tables and formulas are typeset, instead of being shown as pictures of
  // themselves.
  const [paperDoc, setPaperDoc] = useState<PaperDocument | null>(null)
  const [docState, setDocState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')

  const pollTimer = useRef<number | null>(null)

  /* ------------------------------------------------------- document view */
  useEffect(() => {
    if (!id) return
    let cancelled = false
    setDocState('loading')
    ;(async () => {
      try {
        const data = await api.document(id, mode === 'zh' ? 'zh' : 'en')
        if (cancelled) return
        setPaperDoc(data)
        setDocState('ready')
      } catch {
        if (cancelled) return
        // The endpoint can be missing on an older backend; the reader then
        // shows the block rendering of the parsed paper instead of erroring.
        setPaperDoc(null)
        setDocState('error')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [id, mode])

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

  useEffect(() => () => {
    if (pollTimer.current) window.clearInterval(pollTimer.current)
  }, [])

  /* ------------------------------------------------------- translation */
  const loadTranslation = useCallback(async () => {
    if (!id) return
    if (pollTimer.current) window.clearInterval(pollTimer.current)
    setTranslation({ status: 'running', progress: 0, message: '正在准备段落…', error: '' })
    try {
      const started = await api.startTranslation(id, 'zh')
      if (started.status === 'ready') {
        setSegments(started.segments ?? [])
        setTranslation({
          status: 'ready',
          progress: 1,
          message: started.failed ? `译文已就绪，${started.failed} 句未翻译` : '译文已就绪',
          error: '',
        })
        return
      }
      pollTimer.current = window.setInterval(() => {
        void (async () => {
          try {
            const state = await api.translation(id, 'zh')
            if (state.status === 'ready') {
              if (pollTimer.current) window.clearInterval(pollTimer.current)
              setSegments(state.segments ?? [])
              setTranslation({
                status: 'ready',
                progress: 1,
                message: state.failed ? `译文已就绪，${state.failed} 句未翻译` : '译文已就绪',
                error: '',
              })
            } else if (state.status === 'error') {
              if (pollTimer.current) window.clearInterval(pollTimer.current)
              setTranslation({
                status: 'error',
                progress: 0,
                message: state.message ?? '',
                error: state.error ?? '翻译失败',
              })
            } else {
              setTranslation({
                status: 'running',
                progress: state.progress ?? 0,
                message: state.message ?? '翻译中…',
                error: '',
              })
            }
          } catch (err) {
            if (pollTimer.current) window.clearInterval(pollTimer.current)
            setTranslation({
              status: 'error',
              progress: 0,
              message: '',
              error: (err as Error).message,
            })
          }
        })()
      }, 1500)
    } catch (err) {
      setTranslation({ status: 'error', progress: 0, message: '', error: (err as Error).message })
    }
  }, [id])

  const switchMode = useCallback(
    (next: ViewMode) => {
      setMode(next)
      if (next === 'zh' && translation.status !== 'ready') void loadTranslation()
    },
    [loadTranslation, translation.status],
  )

  /* -------------------------------------------------- citation navigation */
  const jumpToReference = async (n: number) => {
    setActiveRef(n)
    const ref = await api.reference(id!, n)
    const marker = markers.find((m) => m.numbers.includes(n))
    if (marker) {
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

  /* ------------------------------------------------------------- render */
  const doc = useMemo(() => buildDocument(structure, segments), [structure, segments])
  // The outline follows the document: a heading that carries no content is
  // not worth a line in the tree.
  const outline = useMemo(
    () => doc.sections.map((s) => s.section).filter((s): s is Section => Boolean(s)),
    [doc],
  )

  if (!id) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState
          title="尚未选择文献"
          hint="请先在文献库中选择一篇文献，或直接上传新的 PDF。"
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
            {paper?.year && <span>{paper.year}</span>}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {mode === 'zh' && translation.status === 'running' && (
            <span className="chip border-accent-300 bg-accent-50 text-accent-700 dark:border-accent-700 dark:bg-accent-900/40 dark:text-accent-300">
              <Spinner className="h-3 w-3" />
              {translation.message || '翻译中…'}
              {translation.progress > 0 && ` ${Math.round(translation.progress * 100)}%`}
            </span>
          )}
          {mode === 'zh' && translation.status === 'error' && (
            <span className="chip border-red-300 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
              翻译失败：{translation.error}
              <button className="ml-1 underline" onClick={() => void loadTranslation()}>
                重试
              </button>
            </span>
          )}
          <Tabs
            tabs={[
              { id: 'en' as ViewMode, label: '英文原文' },
              { id: 'zh' as ViewMode, label: '中英对照' },
            ]}
            value={mode}
            onChange={switchMode}
          />
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
          {outline.length === 0 && (
            <div className="px-1 font-sans text-xs text-ink-400">未解析出标题层级</div>
          )}
          <ul className="space-y-0.5">
            {outline.map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => {
                    // The faithful layout tags each section wrapper; the
                    // re-laid-out view gives the heading itself the id.
                    const el = document.querySelector<HTMLElement>(`[data-section="${s.id}"]`)
                      ?? document.getElementById(s.id)
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
                参考文献 ({references.length})
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
        <section className="scroll-thin min-w-0 flex-1 overflow-y-auto bg-parchment-100 dark:bg-ink-950">
          {loading ? (
            <div className="flex h-full items-center justify-center">
              <Spinner className="h-5 w-5 text-ink-400" />
            </div>
          ) : doc.sections.length === 0 && doc.preamble.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <EmptyState
                title="没有可显示的正文"
                hint="该文献尚未解析出章节结构，可能仍在解析中。"
              />
            </div>
          ) : docState === 'loading' ? (
            <div className="flex h-full items-center justify-center gap-2 text-ink-400">
              <Spinner className="h-5 w-5" />
              <span className="font-sans text-sm">正在排版…</span>
            </div>
          ) : paperDoc ? (
            <div className="doc-scroll mx-auto w-full max-w-[1000px] px-4 py-6 sm:px-8">
              <DocumentView
                doc={paperDoc}
                lang={mode === 'zh' ? 'zh' : 'en'}
                onCite={(marker) => jumpToReference(Number.parseInt(marker, 10))}
              />
            </div>
          ) : (
            <article className="paper-sheet mx-auto max-w-[880px] px-8 py-10 sm:px-12">
              {/* masthead */}
              <header className="paper-masthead">
                <p className="paper-eyebrow">
                  {mode === 'zh' ? '中文对照版' : 'English Reading Edition'}
                </p>
                <h1 className="paper-title">{paper?.title ?? structure?.title ?? ''}</h1>
                {paper?.authors && paper.authors.length > 0 && (
                  <p className="paper-authors">
                    {paper.authors.map((a) => a.name).join('、')}
                  </p>
                )}
                <div className="paper-meta">
                  {paper?.venue && <span>{paper.venue}</span>}
                  {paper?.year && <span>{paper.year}</span>}
                  <span>{paper?.page_count ?? 0} 页</span>
                  {paper?.arxiv_id && <span>arXiv:{paper.arxiv_id}{paper.version_label}</span>}
                </div>
              </header>

              {mode === 'zh' && translation.status === 'running' && (
                <p className="paper-note">
                  译文生成中，当前显示英文原文。译文按句对齐，鼠标悬停可查看英文原句。
                </p>
              )}
              {mode === 'zh' && translation.status === 'ready' && (
                <p className="paper-note">
                  {translation.message.includes('未翻译')
                    ? `${translation.message}，这些句子仍显示英文原文。`
                    : '译文按句对齐，鼠标悬停中文可查看对应的英文原句；引用标记可点击跳转参考文献。'}
                </p>
              )}

              {/* body */}
              {doc.preamble.length > 0 && (
                <section className="paper-section">
                  {doc.preamble.map((item) => (
                    <ItemView
                      key={item.key}
                      item={item}
                      mode={mode}
                      onCite={jumpToReference}
                      onHover={(text, el) => {
                        const rect = el.getBoundingClientRect()
                        setTip({ text, top: rect.top - 10, left: rect.left + rect.width / 2 })
                      }}
                      onLeave={() => setTip(null)}
                    />
                  ))}
                </section>
              )}

              {doc.sections.map(({ section, items }) => (
                <section key={section.id} data-section={section.id} className="paper-section">
                  {items.map((item) => (
                    <ItemView
                      key={item.key}
                      item={item}
                      mode={mode}
                      onCite={jumpToReference}
                      onHover={(text, el) => {
                        const rect = el.getBoundingClientRect()
                        setTip({ text, top: rect.top - 10, left: rect.left + rect.width / 2 })
                      }}
                      onLeave={() => setTip(null)}
                    />
                  ))}
                </section>
              ))}

              {doc.footnotes.length > 0 && (
                <section className="paper-section">
                  <h2 className="paper-h2">注释</h2>
                  <ol className="paper-footnotes">
                    {doc.footnotes.map((note, i) => (
                      <li key={i}>
                        <span className="mr-1 font-mono text-[10px] text-ink-400">p.{note.page_no}</span>
                        {note.text}
                      </li>
                    ))}
                  </ol>
                </section>
              )}
            </article>
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

      {/* ------------------------------------------------- hover tooltip */}
      {tip && (
        <div className="paper-tip" style={{ top: tip.top, left: tip.left }} role="tooltip">
          <span className="paper-tip-label">English source</span>
          {tip.text}
        </div>
      )}
    </div>
  )
}

/* ========================================================================== */
/* one item of the document                                                    */
/* ========================================================================== */

function ItemView({
  item,
  mode,
  onCite,
  onHover,
  onLeave,
}: {
  item: DocItem
  mode: ViewMode
  onCite: (n: number) => void
  onHover: (text: string, el: HTMLElement) => void
  onLeave: () => void
}) {
  if (item.kind === 'heading') {
    const { section } = item
    return (
      <h2
        className={[
          'paper-h2',
          section.level === 1 ? 'paper-h2-top' : '',
          section.level >= 3 ? 'paper-h3' : '',
        ].join(' ')}
      >
        {section.number && <span className="paper-number">{section.number}</span>}
        {section.title}
      </h2>
    )
  }

  if (item.kind === 'figure') {
    const { figure } = item
    return (
      <figure className="paper-figure">
        {figure.image_url ? (
          <img src={figure.image_url} alt={figure.caption || `figure on page ${figure.page_no}`} loading="lazy" />
        ) : (
          <div className="paper-figure-missing">
            图片未生成（可在页面视图中查看第 {figure.page_no} 页）
          </div>
        )}
        {figure.caption && <figcaption>{figure.caption}</figcaption>}
      </figure>
    )
  }

  if (item.kind === 'formula') {
    const { formula } = item
    return (
      <figure className="paper-formula">
        {formula.image_url ? (
          <img
            src={formula.image_url}
            alt={formula.latex || `equation on page ${formula.page_no}`}
            loading="lazy"
            className="paper-formula-image"
          />
        ) : (
          <div className="paper-formula-fallback">{formula.raw_text}</div>
        )}
      </figure>
    )
  }

  if (item.kind === 'table') {
    const { table } = item
    return (
      <figure className="paper-table">
        {table.caption && <figcaption className="paper-table-caption">{table.caption}</figcaption>}
        {table.image_url ? (
          // The table is shown as it is printed on the page: the cell borders
          // and the alignment of the numbers are part of the result.
          <img
            src={table.image_url}
            alt={table.caption || `table on page ${table.page_no}`}
            loading="lazy"
            className="paper-table-image"
          />
        ) : (
          <div
            className="prose-paper text-[14px]"
            dangerouslySetInnerHTML={{ __html: mdToHtml(table.markdown) }}
          />
        )}
      </figure>
    )
  }

  // paragraph
  const useTranslation = mode === 'zh' && item.segments.length > 0
  if (!useTranslation) {
    return (
      <>
        {splitParagraphs(item.text).map((paragraph, i) => (
          <p key={i} className="paper-paragraph">
            <CitationText text={paragraph} onCite={onCite} />
          </p>
        ))}
      </>
    )
  }

  return (
    <p className="paper-paragraph">
      {item.segments.map((segment) => (
        <span
          key={segment.index}
          className="paper-sentence"
          tabIndex={0}
          onMouseEnter={(e) => onHover(segment.source, e.currentTarget)}
          onFocus={(e) => onHover(segment.source, e.currentTarget)}
          onMouseLeave={onLeave}
          onBlur={onLeave}
        >
          {segment.translation || segment.source}
        </span>
      ))}
    </p>
  )
}

/* ========================================================================== */
/* citation-aware text                                                         */
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
            跳回正文引用位置
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
