/**
 * Application shell: the sidebar navigation, the top bar and the router.
 */

import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { useEffect, useState } from 'react'
import Library from './pages/Library'
import Reader from './pages/Reader'
import Ask from './pages/Ask'
import GraphWorkbench from './pages/GraphWorkbench'
import SearchWorkbench from './pages/SearchWorkbench'
import WritingWorkbench from './pages/WritingWorkbench'
import { useSciStore } from './store'
import { api } from './lib/api'

interface NavItem {
  to: string
  label: string
  hint: string
  icon: JSX.Element
}

const NAV: NavItem[] = [
  {
    to: '/library',
    label: '文献库',
    hint: '上传 · 解析 · 去重',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4" stroke="currentColor" strokeWidth="1.6">
        <path d="M3 5.5A1.5 1.5 0 0 1 4.5 4h3.2l1.3 1.6h6.5A1.5 1.5 0 0 1 17 7.1v7.4a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 3 14.5z" />
      </svg>
    ),
  },
  {
    to: '/reader',
    label: '阅读器',
    hint: '版式 · 溯源 · 中英对照',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4" stroke="currentColor" strokeWidth="1.6">
        <path d="M3.5 4.5h5a2 2 0 0 1 2 2v9a1.6 1.6 0 0 0-1.4-1H3.5zM16.5 4.5h-5a2 2 0 0 0-2 2v9a1.6 1.6 0 0 1 1.4-1h5.6z" />
      </svg>
    ),
  },
  {
    to: '/ask',
    label: '智能问答',
    hint: 'Agent · RAG · 溯源',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4" stroke="currentColor" strokeWidth="1.6">
        <path d="M10 2.6 11.6 7l4.4 1.6-4.4 1.6L10 14.6 8.4 10.2 4 8.6 8.4 7z" />
      </svg>
    ),
  },
  {
    to: '/graph',
    label: '图谱综述',
    hint: '引用拓扑 · 基石挖掘',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4" stroke="currentColor" strokeWidth="1.6">
        <circle cx="5" cy="6" r="2" />
        <circle cx="15" cy="6" r="2" />
        <circle cx="10" cy="14.5" r="2" />
        <path d="M6.6 7.4 8.8 12.9M13.4 7.4 11.2 12.9M7 6h6" />
      </svg>
    ),
  },
  {
    to: '/search',
    label: '检索推荐',
    hint: 'arXiv · 追踪 · 可复现',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4" stroke="currentColor" strokeWidth="1.6">
        <circle cx="8.8" cy="8.8" r="5.3" />
        <path d="m12.8 12.8 4 4" />
      </svg>
    ),
  },
  {
    to: '/writing',
    label: '写作工作台',
    hint: '框架 · 绘图 · 图表',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4" stroke="currentColor" strokeWidth="1.6">
        <path d="M4 16.2 4.8 12 13 3.8l3.2 3.2L8 15.2z" />
        <path d="M11.6 5.2 14.8 8.4" />
      </svg>
    ),
  },
]

export default function App() {
  const theme = useSciStore((s) => s.theme)
  const toggleTheme = useSciStore((s) => s.toggleTheme)
  const [llmOk, setLlmOk] = useState<boolean | null>(null)
  const [model, setModel] = useState('')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  useEffect(() => {
    api
      .health()
      .then((h) => {
        const llm = h.llm as { ok?: boolean } | undefined
        setLlmOk(Boolean(llm?.ok))
        setModel(String((h.llm as { models?: string[] })?.models?.[0] ?? ''))
      })
      .catch(() => setLlmOk(false))
  }, [])

  return (
    <div className="flex h-full">
      {/* ------------------------------------------------------------ sidebar */}
      <aside className="flex w-[228px] shrink-0 flex-col border-r border-ink-200/70 bg-white/70 dark:border-ink-800 dark:bg-ink-900/60">
        <div className="flex items-center gap-2.5 px-5 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-900 font-serif text-sm font-semibold text-accent-400 dark:bg-accent-500 dark:text-ink-950">
            S
          </div>
          <div className="leading-tight">
            <div className="font-sans text-sm font-semibold tracking-tight">SciAgent</div>
            <div className="font-sans text-[10px] uppercase tracking-[0.14em] text-ink-400">
              科研小助手
            </div>
          </div>
        </div>

        <nav className="mt-1 flex-1 space-y-0.5 px-3">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                [
                  'group flex items-start gap-2.5 rounded-lg px-2.5 py-2 transition-colors',
                  isActive
                    ? 'bg-ink-900 text-parchment-50 dark:bg-ink-800 dark:text-parchment-50'
                    : 'text-ink-600 hover:bg-ink-100 dark:text-ink-300 dark:hover:bg-ink-800/70',
                ].join(' ')
              }
            >
              <span className="mt-0.5 shrink-0 opacity-80">{item.icon}</span>
              <span className="leading-tight">
                <span className="block font-sans text-[13px] font-medium">{item.label}</span>
                <span className="block font-sans text-[10px] opacity-60">{item.hint}</span>
              </span>
            </NavLink>
          ))}
        </nav>

        <div className="space-y-2 border-t border-ink-200/70 px-4 py-3 dark:border-ink-800">
          <div className="flex items-center gap-1.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                llmOk === null
                  ? 'bg-ink-300'
                  : llmOk
                    ? 'bg-emerald-500'
                    : 'bg-red-500 animate-pulse-soft'
              }`}
            />
            <span className="font-mono text-[10px] text-ink-500 dark:text-ink-400">
              {llmOk === null ? 'connecting…' : llmOk ? (model || 'LLM ready') : 'LLM offline'}
            </span>
          </div>
          <button
            onClick={toggleTheme}
            className="btn-ghost w-full justify-start !px-2 !py-1 text-[11px]"
          >
            {theme === 'light' ? '◐ 深色模式' : '◑ 浅色模式'}
          </button>
        </div>
      </aside>

      {/* ------------------------------------------------------------- content */}
      <main className="min-w-0 flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Navigate to="/library" replace />} />
          <Route path="/library" element={<Library />} />
          <Route path="/reader" element={<Reader />} />
          <Route path="/reader/:paperId" element={<Reader />} />
          <Route path="/ask" element={<Ask />} />
          <Route path="/graph" element={<GraphWorkbench />} />
          <Route path="/search" element={<SearchWorkbench />} />
          <Route path="/writing" element={<WritingWorkbench />} />
          <Route path="*" element={<Navigate to="/library" replace />} />
        </Routes>
      </main>
    </div>
  )
}
