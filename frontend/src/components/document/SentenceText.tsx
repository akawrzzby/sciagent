/**
 * The text layer of the re-laid-out paper.
 *
 * One `SentenceText` renders a whole text block.  The sentence pairs of the
 * block are laid out according to `lang`:
 *
 * - `en`   — the English source only.
 * - `zh`   — the Chinese translation; hovering or keyboard-focusing a sentence
 *            reveals the English source in a floating tooltip (never `title`,
 *            which cannot be styled, timed or dismissed with Escape).
 * - `both` — Chinese first, the English source right behind it in a weaker
 *            colour, sentence by sentence.
 *
 * Inline citation markers (`[12]`, `[3,4]`) always stay clickable: the tooltip
 * is `pointer-events: none` and sits above the line, so it can never swallow a
 * citation click.
 */
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import type { DocSentence, ReadLang } from '../../lib/document'

/** `[12]`, `[3,4]`, `[7, 8]`, `[1-3]`, `[1–3]` — a bracket with only citations. */
const CITATION = /\[(\d{1,3}(?:\s*[,;，、\-–]\s*\d{1,3})*)\]/g

export interface SentenceTextProps {
  /** Sentence pairs of the block; the preferred source for `zh`/`both`. */
  sentences?: DocSentence[]
  /** Whole-block English text, used when `sentences` is missing or empty. */
  text?: string
  /** Whole-block translation, used when `sentences` is missing or empty. */
  target?: string
  lang: ReadLang
  /** Called with the bare marker of a citation: `'12'` for `[12]`, `'3,4'` for `[3,4]`. */
  onCite?: (marker: string) => void
  /**
   * Show the soft "未翻译" hint when the block has no translation at all.
   * Headings and list items switch it off — a badge per line drowns the text.
   */
  showHint?: boolean
  className?: string
}

interface Piece {
  source: string
  target: string
}

interface TipState {
  text: string
  rect: DOMRect
  /** Which sentence owns the tooltip, so only it gets `aria-describedby`. */
  key: string
}

interface Placement {
  left: number
  top: number
  below: boolean
}

/** Split a text run into plain strings and clickable citation markers. */
function withCitations(value: string, onCite?: (marker: string) => void): ReactNode[] {
  if (!value) return []
  const nodes: ReactNode[] = []
  let cursor = 0
  let key = 0
  CITATION.lastIndex = 0
  let match = CITATION.exec(value)
  while (match !== null) {
    if (match.index > cursor) nodes.push(value.slice(cursor, match.index))
    const raw = match[0]
    const marker = match[1].replace(/\s+/g, '')
    const chip = raw.slice(1, -1)
    nodes.push(
      onCite ? (
        <button
          key={`cite-${key++}`}
          type="button"
          className="doc-cite"
          data-cite={marker}
          aria-label={`参考文献 ${marker}`}
          onClick={(event) => {
            event.stopPropagation()
            onCite(marker)
          }}
        >
          {chip}
        </button>
      ) : (
        <span key={`cite-${key++}`} className="doc-cite doc-cite-static" data-cite={marker}>
          {chip}
        </span>
      ),
    )
    cursor = match.index + raw.length
    match = CITATION.exec(value)
  }
  if (cursor < value.length) nodes.push(value.slice(cursor))
  return nodes
}

/**
 * A caption, figure or table, follows the same language policy as the body:
 * the translation leads and the English original stays as a secondary line.
 */
export interface CaptionTextProps {
  caption?: string
  captionTarget?: string
  lang: ReadLang
}

export function CaptionText({ caption, captionTarget, lang }: CaptionTextProps): ReactNode {
  const source = (caption ?? '').trim()
  const target = (captionTarget ?? '').trim()
  if (lang !== 'en' && target) {
    return (
      <>
        <span className="doc-caption-zh">{target}</span>
        {source && <span className="doc-caption-en">{source}</span>}
      </>
    )
  }
  if (source) return <span className="doc-caption-en doc-caption-en-primary">{source}</span>
  if (target) return <span className="doc-caption-zh">{target}</span>
  return null
}

export function SentenceText({
  sentences,
  text,
  target,
  lang,
  onCite,
  showHint = true,
  className,
}: SentenceTextProps) {
  const pieces = useMemo<Piece[]>(() => {
    const pairs = (sentences ?? [])
      .map((sentence) => ({ source: sentence.source ?? '', target: sentence.target ?? '' }))
      .filter((piece) => piece.source.trim() !== '' || piece.target.trim() !== '')
    if (pairs.length > 0) return pairs
    if ((text ?? '').trim() !== '' || (target ?? '').trim() !== '') {
      return [{ source: text ?? '', target: target ?? '' }]
    }
    return []
  }, [sentences, text, target])

  const [tip, setTip] = useState<TipState | null>(null)
  const [place, setPlace] = useState<Placement | null>(null)
  const tipRef = useRef<HTMLDivElement | null>(null)
  const anchorRef = useRef<HTMLElement | null>(null)
  const tipId = useId()

  const show = useCallback((source: string, anchor: HTMLElement, key: string) => {
    if (!source.trim()) return
    anchorRef.current = anchor
    setPlace(null)
    setTip({ text: source, rect: anchor.getBoundingClientRect(), key })
  }, [])

  const hide = useCallback(() => {
    anchorRef.current = null
    setPlace(null)
    setTip(null)
  }, [])

  // The tooltip belongs to the Chinese layer only; a mode switch drops it.
  useEffect(() => {
    if (lang !== 'zh') hide()
  }, [lang, hide])

  const tipVisible = tip !== null

  // Escape dismisses the tooltip; scrolling and resizing keep it glued to the
  // sentence it describes instead of leaving it detached mid-screen.
  useEffect(() => {
    if (!tipVisible) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        anchorRef.current = null
        setPlace(null)
        setTip(null)
      }
    }
    const follow = () => {
      const anchor = anchorRef.current
      if (!anchor) return
      setTip((prev) => (prev ? { ...prev, rect: anchor.getBoundingClientRect() } : prev))
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('scroll', follow, true)
    window.addEventListener('resize', follow)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('scroll', follow, true)
      window.removeEventListener('resize', follow)
    }
  }, [tipVisible])

  // Measure first, position second: the tooltip is centred on its sentence but
  // clamped into the viewport, and flips under the line when it would clip.
  useLayoutEffect(() => {
    const element = tipRef.current
    if (!tip || !element) {
      setPlace(null)
      return
    }
    const width = element.offsetWidth
    const height = element.offsetHeight
    const viewportWidth = window.innerWidth
    const viewportHeight = window.innerHeight
    const gap = 10
    const margin = 8
    const half = Math.min(width / 2 + margin, viewportWidth / 2)
    const center = tip.rect.left + tip.rect.width / 2
    const left = Math.min(Math.max(center, half), Math.max(viewportWidth - half, half))
    const fitsAbove = tip.rect.top - gap - height >= margin
    const top = fitsAbove
      ? tip.rect.top - gap
      : Math.min(tip.rect.bottom + gap, Math.max(viewportHeight - height - margin, margin))
    setPlace({ left, top, below: !fitsAbove })
  }, [tip])

  const tooltip =
    tip && typeof document !== 'undefined'
      ? createPortal(
          <div
            ref={tipRef}
            id={tipId}
            role="tooltip"
            className="doc-tip"
            style={{
              left: place ? `${place.left}px` : '0px',
              top: place ? `${place.top}px` : '0px',
              transform: place?.below ? 'translate(-50%, 0)' : 'translate(-50%, -100%)',
              visibility: place ? 'visible' : 'hidden',
            }}
          >
            <span className="doc-tip-label">English source</span>
            {tip.text}
          </div>,
          document.body,
        )
      : null

  const wrapperClass = ['doc-text', className].filter(Boolean).join(' ')

  if (pieces.length === 0) return null

  // The whole block in English: the block text when there is one, otherwise the
  // sentence sources joined back together.
  const english = (text ?? '').trim() || pieces.map((piece) => piece.source).join(' ')

  if (lang === 'en') {
    return <span className={wrapperClass}>{withCitations(english, onCite)}</span>
  }

  // Nothing in this block has a translation: show the original in one piece and
  // say so once, instead of stamping a badge onto every sentence.
  if (!pieces.some((piece) => piece.target.trim() !== '')) {
    return (
      <span className={wrapperClass}>
        {withCitations(english, onCite)}
        {showHint && <span className="doc-untranslated-badge">未翻译</span>}
      </span>
    )
  }

  return (
    <>
      <span className={wrapperClass}>
        {pieces.map((piece, index) => {
          const key = `sent-${index}`
          const translated = piece.target.trim() !== ''

          if (!translated) {
            // A hole in an otherwise translated block: the English sentence
            // stays in place, quietly dimmed, so the paragraph still reads.
            return (
              <span key={key} className="doc-sent-untranslated">
                {withCitations(piece.source, onCite)}
              </span>
            )
          }

          if (lang === 'both') {
            return (
              <span key={key} className="doc-pair">
                <span className="doc-sent-zh">{withCitations(piece.target, onCite)}</span>
                <span className="doc-sent-en">{withCitations(piece.source, onCite)}</span>
              </span>
            )
          }

          return (
            <span
              key={key}
              className="doc-sent-zh doc-sent-interactive"
              data-source={piece.source}
              tabIndex={0}
              aria-describedby={tip && tip.key === key ? tipId : undefined}
              onMouseEnter={(event) => show(piece.source, event.currentTarget, key)}
              onFocus={(event) => show(piece.source, event.currentTarget, key)}
              onMouseLeave={hide}
              onBlur={(event) => {
                const next = event.relatedTarget
                // Tabbing into a citation inside the same sentence keeps the tip.
                if (next instanceof Node && event.currentTarget.contains(next)) return
                hide()
              }}
            >
              {withCitations(piece.target, onCite)}
            </span>
          )
        })}
      </span>
      {tooltip}
    </>
  )
}
