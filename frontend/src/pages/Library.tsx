/**
 * Module 1 — the literature library.
 *
 * Shows the dashboard statistics, the drag & drop upload area and the parsed
 * paper list.  When the backend detects a duplicate it pushes an event and the
 * UI shows a modal that explains *why* the two documents are considered the
 * same and which version should be kept.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, LibraryStats, Paper } from '../lib/api'
import { EmptyState, Modal, Spinner, StatCard, Toast, formatAuthors } from '../components/ui'
import { useSciStore } from '../store'

interface DuplicateInfo {
  paper_id: string
  duplicate_of: string
  recommendation: {
    recommendation: string
    keep: string
    reason: string
    uploaded_version: string
    library_version: string
    similarity: number
    reasons: string[]
  }
  reasons: string[]
  diff?: { op: string; old: string; new: string }[]
}

export default function Library() {
  const navigate = useNavigate()
  const [papers, setPapers] = useState<Paper[]>([])
  const [stats, setStats] = useState<LibraryStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [query, setQuery] = useState('')
  const [toast, setToast] = useState<{ msg: string; kind: 'info' | 'ok' | 'error' } | null>(null)
  const [duplicate, setDuplicate] = useState<DuplicateInfo | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const selected = useSciStore((s) => s.selected)
  const togglePaper = useSciStore((s) => s.togglePaper)
  const setActivePaper = useSciStore((s) => s.setActivePaper)

  const refresh = useCallback(async () => {
    try {
      const [list, st] = await Promise.all([api.listPapers(query), api.stats()])
      setPapers(list.papers)
      setStats(st)
    } catch (err) {
      setToast({ msg: `加载失败：${(err as Error).message}`, kind: 'error' })
    } finally {
      setLoading(false)
    }
  }, [query])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // Poll while papers are being parsed (the backend parses in the background).
  useEffect(() => {
    const pending = papers.some((p) => p.status !== 'ready')
    if (!pending) return
    const timer = setInterval(() => void refresh(), 4000)
    return () => clearInterval(timer)
  }, [papers, refresh])

  const handleFiles = async (files: File[]) => {
    const pdfs = files.filter((f) => f.name.toLowerCase().endsWith('.pdf'))
    if (!pdfs.length) {
      setToast({ msg: '请选择 PDF 文件', kind: 'error' })
      return
    }
    setUploading(true)
    setProgress(0)
    try {
      await api.upload(pdfs, setProgress)
      setToast({ msg: `已接收 ${pdfs.length} 个文件，正在后台解析…`, kind: 'ok' })
      await refresh()
    } catch (err) {
      setToast({ msg: `上传失败：${(err as Error).message}`, kind: 'error' })
    } finally {
      setUploading(false)
      setProgress(0)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* ------------------------------------------------------------ header */}
      <header className="flex items-center justify-between gap-4 border-b border-ink-200/70 px-6 py-4 dark:border-ink-800">
        <div>
          <h1 className="font-serif text-xl font-semibold tracking-tight">文献库</h1>
          <p className="mt-0.5 font-sans text-xs text-ink-500 dark:text-ink-400">
            PDF 上传 · 版本归并 · 版式解析 · 索引构建
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            className="input !w-64"
            placeholder="搜索标题 / arXiv ID…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button className="btn-secondary" onClick={() => void refresh()}>
            刷新
          </button>
        </div>
      </header>

      <div className="scroll-thin flex-1 overflow-auto px-6 py-5">
        {toast && (
          <div className="mb-4">
            <Toast message={toast.msg} kind={toast.kind} />
          </div>
        )}

        {/* ---------------------------------------------------------- stats */}
        {stats && (
          <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <StatCard label="文献" value={stats.papers} />
            <StatCard label="总页数" value={stats.pages} />
            <StatCard label="检索片段" value={stats.chunks} />
            <StatCard label="扫描件" value={stats.scanned_papers} tone="accent" />
            <StatCard label="参考文献" value={stats.references} />
            <StatCard label="图表 / 公式" value={`${stats.tables} / ${stats.formulas}`} />
          </div>
        )}

        {/* --------------------------------------------------------- upload */}
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            void handleFiles(Array.from(e.dataTransfer.files))
          }}
          onClick={() => fileInput.current?.click()}
          className={[
            'mb-5 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-9 transition-colors',
            dragOver
              ? 'border-accent-400 bg-accent-50/70 dark:bg-accent-900/20'
              : 'border-ink-200 bg-white/60 hover:border-ink-300 dark:border-ink-800 dark:bg-ink-900/40 dark:hover:border-ink-700',
          ].join(' ')}
        >
          <input
            ref={fileInput}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            className="hidden"
            onChange={(e) => void handleFiles(Array.from(e.target.files ?? []))}
          />
          {uploading ? (
            <>
              <Spinner className="h-5 w-5 text-accent-500" />
              <div className="font-sans text-sm text-ink-600 dark:text-parchment-300">
                上传中 {Math.round(progress * 100)}%
              </div>
              <div className="h-1 w-56 overflow-hidden rounded-full bg-ink-200 dark:bg-ink-800">
                <div
                  className="h-full rounded-full bg-accent-500 transition-all"
                  style={{ width: `${progress * 100}%` }}
                />
              </div>
            </>
          ) : (
            <>
              <div className="font-sans text-sm font-medium text-ink-700 dark:text-parchment-200">
                拖拽 PDF 到此处，或点击选择文件
              </div>
              <div className="font-sans text-[11px] text-ink-400">
                支持批量上传 · 扫描件经多模态 OCR 转写 · arXiv 同版本重复时提示合并
              </div>
            </>
          )}
        </div>

        {/* ---------------------------------------------------- paper list */}
        {loading ? (
          <div className="flex justify-center py-16">
            <Spinner className="h-5 w-5 text-ink-400" />
          </div>
        ) : papers.length === 0 ? (
          <EmptyState
            title="文献库为空"
            hint="上传第一篇文献后，系统将抽取标题结构、识别表格与公式、解析参考文献，并建立向量索引。"
          />
        ) : (
          <div className="space-y-2.5">
            {papers.map((p) => {
              const isSelected = selected.some((s) => s.id === p.id)
              return (
                <div
                  key={p.id}
                  className={[
                    'panel group flex items-start gap-3 px-4 py-3 transition-shadow hover:shadow-panel-lg',
                    isSelected ? 'ring-1 ring-accent-400/60' : '',
                  ].join(' ')}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => togglePaper({ id: p.id, title: p.title })}
                    className="mt-1 h-3.5 w-3.5 accent-[#f98412]"
                    title="勾选后用于图谱与综述"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start gap-2">
                      <button
                        className="truncate text-left font-serif text-[15px] font-semibold leading-snug text-ink-900 hover:text-accent-600 dark:text-parchment-50 dark:hover:text-accent-400"
                        onClick={() => {
                          setActivePaper(p.id)
                          navigate(`/reader/${p.id}`)
                        }}
                      >
                        {p.title || p.filename || '(未命名文献)'}
                      </button>
                      {p.is_scanned && (
                        <span className="chip border-accent-300 bg-accent-50 text-accent-700 dark:border-accent-700 dark:bg-accent-900/40 dark:text-accent-300">
                          扫描件 · VLM OCR
                        </span>
                      )}
                      {p.version_label && (
                        <span className="chip border-ink-200 bg-ink-50 text-ink-500 dark:border-ink-700 dark:bg-ink-800 dark:text-ink-300">
                          {p.version_label}
                        </span>
                      )}
                    </div>
                    <div className="mt-1 truncate font-sans text-xs text-ink-500 dark:text-ink-400">
                      {formatAuthors(p.authors)}
                      {p.year ? ` · ${p.year}` : ''}
                      {p.arxiv_id ? ` · arXiv:${p.arxiv_id}` : ''}
                    </div>
                    <div className="mt-1.5 flex items-center gap-3 font-sans text-[11px] text-ink-400">
                      <span>{p.page_count} 页</span>
                      {p.ocr_pages ? <span>OCR {p.ocr_pages} 页</span> : null}
                      <span>
                        {String(p.metadata?.columns ?? '')}
                      </span>
                      <span className={p.status === 'ready' ? 'text-emerald-600' : 'text-amber-600'}>
                        {p.status === 'ready' ? '已就绪' : '解析中…'}
                      </span>
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <button
                      className="btn-ghost !px-2 !py-1 text-[11px]"
                      onClick={() => {
                        setActivePaper(p.id)
                        navigate(`/reader/${p.id}`)
                      }}
                    >
                      阅读
                    </button>
                    <button
                      className="btn-ghost !px-2 !py-1 text-[11px]"
                      onClick={() => {
                        setActivePaper(p.id)
                        navigate('/ask')
                      }}
                    >
                      提问
                    </button>
                    <button
                      className="btn-ghost !px-2 !py-1 text-[11px] text-red-500"
                      onClick={async () => {
                        await api.deletePaper(p.id)
                        await refresh()
                      }}
                    >
                      删除
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* --------------------------------------------------- duplicate modal */}
      <Modal
        open={Boolean(duplicate)}
        onClose={() => setDuplicate(null)}
        title="检测到重复文献"
        wide
      >
        {duplicate && (
          <div className="space-y-4">
            <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm dark:border-amber-700 dark:bg-amber-950/40">
              <div className="font-medium text-amber-900 dark:text-amber-200">
                {duplicate.recommendation?.reason ?? '库中已存在高度重合的文献'}
              </div>
              <div className="mt-1 text-xs text-amber-800/80 dark:text-amber-300/80">
                建议保留：
                <strong>
                  {duplicate.recommendation?.keep === 'uploaded' ? '本次上传的版本' : '库中已有版本'}
                </strong>
                （相似度 {(duplicate.recommendation?.similarity ?? 0).toFixed(4)}）
              </div>
              <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs text-amber-800/80 dark:text-amber-300/80">
                {(duplicate.reasons ?? []).map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>

            {duplicate.diff && duplicate.diff.length > 0 && (
              <div>
                <div className="mb-1 font-sans text-xs font-semibold text-ink-600 dark:text-parchment-300">
                版本差异（前 12 处）
                </div>
                <div className="scroll-thin max-h-64 space-y-1 overflow-auto rounded-lg border border-ink-200 p-2 dark:border-ink-700">
                  {duplicate.diff
                    .filter((d) => d.op !== 'equal')
                    .slice(0, 12)
                    .map((d, i) => (
                      <div key={i} className="font-mono text-[11px] leading-relaxed">
                        <span className="mr-2 rounded bg-red-100 px-1 text-red-700 dark:bg-red-950 dark:text-red-300">
                          −
                        </span>
                        <span className="text-ink-500 line-through dark:text-ink-400">
                          {(d.old || '').slice(0, 200)}
                        </span>
                        <br />
                        <span className="mr-2 rounded bg-emerald-100 px-1 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                          +
                        </span>
                        <span className="text-ink-700 dark:text-parchment-200">
                          {(d.new || '').slice(0, 200)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            <div className="flex justify-end gap-2">
              <button
                className="btn-secondary"
                onClick={async () => {
                  if (!duplicate) return
                  await api.upload([], () => {})
                  setDuplicate(null)
                }}
              >
                保留库中版本
              </button>
              <button
                className="btn-primary"
                onClick={() => {
                  setDuplicate(null)
                  void refresh()
                }}
              >
                用推荐版本覆盖
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}
