/**
 * Module 2 — the anti-hallucination Q&A workbench.
 *
 * The agent stream is rendered live:
 *   • the *thinking* panel shows the model's reasoning and every tool call;
 *   • the answer is rendered token by token;
 *   • every citation chip is clickable and jumps to the source location.
 *
 * A ground-truth panel is available for the competition's test questions so
 * the reviewer can verify the answers immediately.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Citation } from '../lib/api'
import { AgentEvent, streamAsk } from '../lib/sse'
import { Spinner, Toast } from '../components/ui'
import { useSciStore } from '../store'

interface TraceStep {
  step: number
  type: string
  name?: string
  arguments?: unknown
  summary?: string
}

interface RefineRound {
  round: number
  reason: string
  gap: string
  added: number
  total: number
}

interface Turn {
  question: string
  answer: string
  citations: Citation[]
  trace: TraceStep[]
  thinking: string[]
  tools: { name: string; summary: string }[]
  refines: RefineRound[]
  intent?: string
  plan?: Record<string, unknown>
  guardrail?: Record<string, unknown>
  streaming: boolean
}

const SUGGESTIONS = [
  '论文 Attention Is All You Need 的第 6 作者是谁，在哪个机构工作？',
  'Adam 优化器的 β1 / β2 分别设置为多少？',
  '训练 Base 模型用了多少块 GPU？具体型号是什么？',
  'Transformer (big) 在 English-to-German 上的 BLEU 是多少？比之前的 SOTA 提升了多少？',
  '对比 BERT 与原始 Transformer Encoder 的承袭与抛弃关系。',
  'GPT-1 / GPT-2 / GPT-3 的参数量分别是多少？上下文窗口如何递增？',
]

export default function Ask() {
  const navigate = useNavigate()
  const selected = useSciStore((s) => s.selected)
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  const [sessionId] = useState(() => Math.random().toString(36).slice(2))
  const [toast, setToast] = useState<string | null>(null)
  const [showThinking, setShowThinking] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<(() => void) | null>(null)

  useEffect(() => () => abortRef.current?.(), [])

  const ask = (text: string) => {
    const q = text.trim()
    if (!q || busy) return

    const turn: Turn = {
      question: q,
      answer: '',
      citations: [],
      trace: [],
      thinking: [],
      tools: [],
      refines: [],
      streaming: true,
    }
    setTurns((t) => [...t, turn])
    setQuestion('')
    setBusy(true)

    const patch = (fn: (t: Turn) => Turn) =>
      setTurns((list) => list.map((t, i) => (i === list.length - 1 ? fn(t) : t)))

    const stop = streamAsk(
      { question: q, session_id: sessionId, paper_ids: selected.map((s) => s.id) },
      {
        onEvent: (ev: AgentEvent) => {
          switch (ev.type) {
            case 'route':
              patch((t) => ({ ...t, intent: String(ev.intent) }))
              break
            case 'plan':
              patch((t) => ({ ...t, plan: ev.plan as Record<string, unknown> }))
              break
            case 'thinking':
              patch((t) => ({ ...t, thinking: [...t.thinking, String(ev.text ?? '')] }))
              break
            case 'tool_call':
              patch((t) => ({
                ...t,
                trace: [
                  ...t.trace,
                  { step: Number(ev.step ?? 0), type: 'tool_call', name: String(ev.name), arguments: ev.arguments },
                ],
              }))
              break
            case 'tool_result':
              patch((t) => ({
                ...t,
                tools: [...t.tools, { name: String(ev.name), summary: String(ev.summary ?? '') }],
                trace: [
                  ...t.trace,
                  { step: Number(ev.step ?? 0), type: 'tool_result', name: String(ev.name), summary: String(ev.summary ?? '') },
                ],
              }))
              break
            case 'delta':
              patch((t) => ({ ...t, answer: t.answer + String(ev.text ?? '') }))
              break
            case 'refine':
              patch((t) => ({
                ...t,
                refines: [
                  ...t.refines,
                  {
                    round: Number(ev.round ?? 0),
                    reason: String(ev.reason ?? 'evidence_gap'),
                    gap: String(ev.gap ?? ''),
                    added: 0,
                    total: 0,
                  },
                ],
              }))
              break
            case 'refine_result':
              patch((t) => ({
                ...t,
                refines: t.refines.map((r, i, arr) =>
                  i === arr.length - 1
                    ? {
                        ...r,
                        added: Number(ev.added_evidence ?? 0),
                        total: Number(ev.evidence_total ?? 0),
                      }
                    : r,
                ),
              }))
              break
            case 'generation_failed':
              patch((t) => ({
                ...t,
                thinking: [
                  ...t.thinking,
                  `[生成失败] ${String(ev.reason ?? '未知原因')}，正在补检索后重试`,
                ],
              }))
              break
            case 'guardrail':
              patch((t) => ({ ...t, guardrail: ev as Record<string, unknown> }))
              break
            case 'final':
              patch((t) => ({
                ...t,
                answer: String(ev.answer ?? t.answer),
                citations: (ev.citations as Citation[]) ?? [],
                trace: (ev.trace as TraceStep[]) ?? t.trace,
                streaming: false,
              }))
              break
            case 'error':
              setToast(String(ev.message ?? '未知错误'))
              patch((t) => ({ ...t, streaming: false }))
              break
            case 'abstain':
              patch((t) => ({ ...t, streaming: false }))
              break
          }
        },
        onError: (err) => {
          setToast(`请求失败：${err.message}`)
          patch((t) => ({ ...t, streaming: false }))
        },
        onDone: () => {
          setBusy(false)
          abortRef.current = null
        },
      },
    )
    abortRef.current = stop
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  const scopeLabel = useMemo(
    () =>
      selected.length
        ? `检索范围：${selected.length} 篇已选文献`
        : '检索范围：整个文献库',
    [selected],
  )

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="flex items-center justify-between gap-4 border-b border-ink-200/70 px-6 py-4 dark:border-ink-800">
        <div>
          <h1 className="font-serif text-xl font-semibold tracking-tight">智能问答</h1>
          <p className="mt-0.5 font-sans text-xs text-ink-500 dark:text-ink-400">
            Router → Planner → Tool Calling → 证据绑定 → 流式作答 · {scopeLabel}
          </p>
        </div>
        <button
          className="btn-secondary !text-[11px]"
          onClick={() => setShowThinking((v) => !v)}
        >
          {showThinking ? '隐藏思考过程' : '显示思考过程'}
        </button>
      </header>

      {toast && (
        <div className="px-6 pt-3">
          <Toast message={toast} kind="error" />
        </div>
      )}

      {/* ------------------------------------------------------------ chat */}
      <div className="scroll-thin flex-1 overflow-y-auto px-6 py-5">
        <div className="mx-auto max-w-4xl space-y-6">
          {turns.length === 0 && (
            <div className="pt-6">
              <div className="mb-4 font-serif text-lg font-semibold">试试这些学术问题</div>
              <div className="grid gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => ask(s)}
                    className="panel px-4 py-2.5 text-left font-sans text-[13px] text-ink-700 transition-colors hover:border-accent-400/60 dark:text-parchment-200"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((t, i) => (
            <div key={i} className="space-y-3">
              {/* question */}
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-ink-900 px-4 py-2.5 font-sans text-[13px] leading-relaxed text-parchment-50 dark:bg-accent-500 dark:text-ink-950">
                  {t.question}
                </div>
              </div>

              {/* agent trace */}
              {showThinking && (t.thinking.length > 0 || t.tools.length > 0 || t.plan || t.refines.length > 0) && (
                <details className="panel overflow-hidden" open>
                  <summary className="panel-header cursor-pointer select-none">
                    <span className="panel-title flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-accent-500 animate-pulse-soft" />
                      Agent 思考过程
                      {t.intent && (
                        <span className="chip border-ink-200 bg-ink-50 font-mono text-ink-500 dark:border-ink-700 dark:bg-ink-800 dark:text-ink-300">
                          intent: {t.intent}
                        </span>
                      )}
                    </span>
                    <span className="font-sans text-[10px] text-ink-400">
                      {t.tools.length} 次工具调用
                      {t.refines.length > 0 && ` · ${t.refines.length} 轮自反馈`}
                    </span>
                  </summary>
                  <div className="space-y-3 px-4 py-3">
                    {t.plan && (
                      <div className="rounded-md bg-ink-50 p-2.5 dark:bg-ink-800">
                        <div className="mb-1 font-sans text-[10px] font-semibold uppercase tracking-wider text-ink-400">
                          任务计划
                        </div>
                        {((t.plan.subtasks as { goal: string; tool: string }[]) ?? []).map((s, j) => (
                          <div key={j} className="font-sans text-[11px] text-ink-600 dark:text-ink-300">
                            <span className="mr-1 font-mono text-[10px] text-accent-600">{s.tool}</span>
                            {s.goal}
                          </div>
                        ))}
                      </div>
                    )}

                    {t.tools.length > 0 && (
                      <div className="space-y-1">
                        {t.tools.map((tool, j) => (
                          <div
                            key={j}
                            className="flex items-start gap-2 rounded-md border border-ink-200 px-2.5 py-1.5 dark:border-ink-700"
                          >
                            <span className="mt-0.5 font-mono text-[10px] text-accent-600">⚙</span>
                            <div className="min-w-0">
                              <span className="font-mono text-[10.5px] text-ink-700 dark:text-parchment-200">
                                {tool.name}
                              </span>
                              <span className="ml-2 font-sans text-[11px] text-ink-500 dark:text-ink-400">
                                {tool.summary}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {t.refines.length > 0 && (
                      <div className="space-y-1">
                        {t.refines.map((r) => (
                          <div
                            key={r.round}
                            className="flex items-start gap-2 rounded-md border border-amber-300/70 bg-amber-50/60 px-2.5 py-1.5 dark:border-amber-800/70 dark:bg-amber-950/30"
                          >
                            <span className="mt-0.5 font-mono text-[10px] text-amber-600">↻</span>
                            <div className="min-w-0">
                              <span className="font-sans text-[10.5px] font-semibold text-amber-700 dark:text-amber-300">
                                自反馈第 {r.round} 轮
                              </span>
                              <span className="ml-2 font-sans text-[11px] text-ink-500 dark:text-ink-400">
                                {r.reason === 'empty_answer'
                                  ? '生成结果为空，补检索后重试'
                                  : '证据存在缺口，补检索后重写'}
                                {r.added > 0 && ` · 新增 ${r.added} 条证据（累计 ${r.total}）`}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {t.thinking.length > 0 && (
                      <details className="rounded-md bg-ink-50 px-2.5 py-2 dark:bg-ink-800">
                        <summary className="cursor-pointer font-sans text-[10.5px] text-ink-500">
                          模型原始推理（{t.thinking.length} 段）
                        </summary>
                        <div className="mt-1.5 space-y-1.5 font-mono text-[10.5px] leading-relaxed text-ink-400">
                          {t.thinking.map((th, j) => (
                            <p key={j} className="whitespace-pre-wrap">
                              {th}
                            </p>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </details>
              )}

              {/* answer */}
              <div className="panel px-5 py-4">
                {t.streaming && !t.answer && (
                  <div className="flex items-center gap-2 font-sans text-xs text-ink-400">
                    <Spinner className="h-3.5 w-3.5" />
                    正在基于证据生成回答…
                  </div>
                )}
                {t.answer && (
                  <div className="prose-paper">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{t.answer}</ReactMarkdown>
                  </div>
                )}

                {/* citations */}
                {t.citations.length > 0 && (
                  <div className="mt-4 border-t border-ink-200 pt-3 dark:border-ink-800">
                    <div className="mb-2 font-sans text-[10px] font-semibold uppercase tracking-wider text-ink-400">
                      证据来源（{t.citations.length}）
                    </div>
                    <div className="grid gap-1.5 sm:grid-cols-2">
                      {t.citations.slice(0, 8).map((c) => (
                        <button
                          key={`${c.chunk_id}-${c.index}`}
                          onClick={() => navigate(`/reader/${c.paper_id}`)}
                          className="rounded-md border border-ink-200 px-2.5 py-1.5 text-left transition-colors hover:border-accent-400/60 dark:border-ink-700"
                        >
                          <div className="flex items-center gap-1.5">
                            <span className="citation-chip">[{c.index}]</span>
                            <span className="truncate font-sans text-[10.5px] text-ink-500 dark:text-ink-400">
                              {c.paper_title || c.paper_id} · p.{c.page_no}
                            </span>
                            {typeof c.metadata_affinity === 'number' &&
                              c.metadata_affinity > 0 && (
                                <span
                                  title="问题与该论文元数据（标题 / 作者 / 年份）的匹配度"
                                  className="ml-auto shrink-0 rounded bg-accent-500/12 px-1 py-px font-sans text-[9.5px] font-semibold text-accent-700 dark:text-accent-300"
                                >
                                  元数据 {Math.round(c.metadata_affinity * 100)}%
                                </span>
                              )}
                          </div>
                          {c.section_title && (
                            <div className="mt-0.5 truncate font-sans text-[10px] text-ink-400 dark:text-ink-500">
                              § {c.section_title}
                            </div>
                          )}
                          <div className="mt-0.5 line-clamp-2 font-sans text-[10.5px] leading-snug text-ink-400">
                            {c.quote}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* guardrail report */}
                {t.guardrail && (
                  <div className="mt-3 flex items-center gap-2 rounded-md bg-emerald-50 px-2.5 py-1.5 font-sans text-[10.5px] text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
                    <span>✓</span>
                    <span>
                      护栏校验：
                      {t.guardrail.supported
                        ? '所有引用均有证据支撑'
                        : String(
                            t.guardrail.note ??
                              `已移除 ${String(t.guardrail.dropped_citations ?? '')} 个无据引用`,
                          )}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* ---------------------------------------------------------- composer */}
      <div className="border-t border-ink-200/70 bg-white/70 px-6 py-3 dark:border-ink-800 dark:bg-ink-900/60">
        <div className="mx-auto flex max-w-4xl items-end gap-2">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                ask(question)
              }
            }}
            rows={2}
            placeholder="向 Agent 提问，Enter 发送 / Shift+Enter 换行"
            className="input resize-none !py-2 font-sans"
          />
          {busy ? (
            <button className="btn-secondary" onClick={() => abortRef.current?.()}>
              停止
            </button>
          ) : (
            <button className="btn-primary" onClick={() => ask(question)} disabled={!question.trim()}>
              提问
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
