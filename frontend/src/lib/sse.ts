/**
 * Server-Sent-Events client for the agent stream.
 *
 * The backend emits a typed event stream:
 *
 *   route | plan | stage | thinking | tool_call | tool_result | delta
 *   guardrail | final | error | abstain
 */

export type AgentEventType =
  | 'session'
  | 'route'
  | 'plan'
  | 'stage'
  | 'thinking'
  | 'tool_call'
  | 'tool_result'
  | 'agent_stop'
  | 'delta'
  | 'guardrail'
  | 'abstain'
  | 'final'
  | 'error'

export interface AgentEvent {
  type: AgentEventType
  [key: string]: unknown
}

export interface AskPayload {
  question: string
  session_id?: string
  paper_ids?: string[]
}

export interface AgentStreamHandlers {
  onEvent?: (event: AgentEvent) => void
  onError?: (error: Error) => void
  onDone?: () => void
}

/**
 * Open a streaming agent conversation.
 *
 * Returns an abort function so the caller can cancel a running turn.
 */
export function streamAsk(
  payload: AskPayload,
  handlers: AgentStreamHandlers = {},
): () => void {
  const controller = new AbortController()
  const base = import.meta.env.VITE_API_BASE ?? '/api'

  ;(async () => {
    try {
      const res = await fetch(`${base}/rag/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, stream: true }),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) {
        throw new Error(`agent stream failed: ${res.status} ${res.statusText}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE frames are separated by a blank line.
        const frames = buffer.split('\n\n')
        buffer = frames.pop() ?? ''

        for (const frame of frames) {
          for (const line of frame.split('\n')) {
            if (!line.startsWith('data:')) continue
            const body = line.slice(5).trim()
            if (!body || body === '[DONE]') continue
            try {
              const event = JSON.parse(body) as AgentEvent
              handlers.onEvent?.(event)
            } catch {
              /* ignore malformed frame */
            }
          }
        }
      }
      handlers.onDone?.()
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      handlers.onError?.(err as Error)
    }
  })()

  return () => controller.abort()
}
