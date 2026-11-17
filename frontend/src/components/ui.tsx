/** Small shared UI primitives used across the workbenches. */

import { ReactNode, useEffect, useState } from 'react'

/* ------------------------------------------------------------------ spinner */
export function Spinner({ className = 'h-4 w-4' }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

/* -------------------------------------------------------------------- toast */
export function Toast({ message, kind = 'info' }: { message: string; kind?: 'info' | 'error' | 'ok' }) {
  const tone =
    kind === 'error'
      ? 'border-red-300 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/50 dark:text-red-200'
      : kind === 'ok'
        ? 'border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200'
        : 'border-ink-200 bg-white text-ink-700 dark:border-ink-700 dark:bg-ink-800 dark:text-parchment-200'
  return (
    <div className={`animate-slide-up rounded-lg border px-3 py-2 font-sans text-xs shadow-panel ${tone}`}>
      {message}
    </div>
  )
}

/* ------------------------------------------------------------------- empty */
export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string
  hint?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-16 text-center">
      <div className="flex h-11 w-11 items-center justify-center rounded-full bg-ink-100 text-ink-400 dark:bg-ink-800">
        <svg viewBox="0 0 20 20" fill="none" className="h-5 w-5" stroke="currentColor" strokeWidth="1.4">
          <path d="M10 3v14M3 10h14" />
        </svg>
      </div>
      <div className="font-sans text-sm font-medium text-ink-600 dark:text-parchment-300">{title}</div>
      {hint && (
        <div className="max-w-md font-sans text-xs leading-relaxed text-ink-400 dark:text-ink-400">
          {hint}
        </div>
      )}
      {action}
    </div>
  )
}

/* ------------------------------------------------------------------- stats */
export function StatCard({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string
  value: ReactNode
  hint?: string
  tone?: 'default' | 'accent'
}) {
  return (
    <div className="panel px-4 py-3">
      <div className="font-sans text-[11px] uppercase tracking-wider text-ink-400">{label}</div>
      <div
        className={`mt-0.5 font-serif text-2xl font-semibold tabular-nums ${
          tone === 'accent' ? 'text-accent-600 dark:text-accent-400' : 'text-ink-900 dark:text-parchment-50'
        }`}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 font-sans text-[11px] text-ink-400">{hint}</div>}
    </div>
  )
}

/* -------------------------------------------------------------------- tabs */
export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
}: {
  tabs: { id: T; label: string }[]
  value: T
  onChange: (id: T) => void
}) {
  return (
    <div className="flex gap-1 rounded-lg bg-ink-100 p-1 dark:bg-ink-800">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={[
            'rounded-md px-3 py-1 font-sans text-xs font-medium transition-colors',
            value === t.id
              ? 'bg-white text-ink-900 shadow-sm dark:bg-ink-700 dark:text-parchment-50'
              : 'text-ink-500 hover:text-ink-800 dark:text-ink-400 dark:hover:text-parchment-200',
          ].join(' ')}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ tooltip */
export function InfoHint({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  return (
    <span
      className="relative inline-flex cursor-help items-center"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-ink-300 text-[9px] font-semibold text-ink-400 dark:border-ink-600">
        i
      </span>
      {open && (
        <span className="absolute left-1/2 top-5 z-50 w-56 -translate-x-1/2 rounded-md bg-ink-900 px-2 py-1.5 text-left font-sans text-[11px] leading-snug text-parchment-100 shadow-panel-lg dark:bg-ink-700">
          {text}
        </span>
      )}
    </span>
  )
}

/* ------------------------------------------------------------------- modal */
export function Modal({
  open,
  onClose,
  title,
  children,
  wide,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  wide?: boolean
}) {
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/40 p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className={`panel animate-slide-up max-h-[85vh] w-full overflow-auto ${
          wide ? 'max-w-4xl' : 'max-w-lg'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-header sticky top-0 z-10 bg-white/95 backdrop-blur dark:bg-ink-900/95">
          <span className="panel-title">{title}</span>
          <button className="btn-ghost !px-2" onClick={onClose} aria-label="关闭">
            ✕
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ format */
export function formatBytes(n: number): string {
  if (!n) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export function formatAuthors(authors: { name: string }[] | string[], max = 4): string {
  const names = authors.map((a) => (typeof a === 'string' ? a : a.name)).filter(Boolean)
  if (!names.length) return '未知作者'
  if (names.length <= max) return names.join(', ')
  return `${names.slice(0, max).join(', ')} +${names.length - max}`
}
