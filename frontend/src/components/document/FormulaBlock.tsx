/**
 * A display equation of the re-laid-out paper.
 *
 * The LaTeX is rendered with KaTeX, centred, with the equation number pinned
 * to the right.  Anything KaTeX cannot typeset (unknown macros, a truncated
 * extraction) falls back to the page crop, and then to the raw source — a bad
 * formula must never take the reading view down with it.
 */
import { useMemo } from 'react'
import katex from 'katex'
import type { KatexOptions } from 'katex'
import 'katex/dist/katex.min.css'
import type { DocBlock } from '../../lib/document'

export interface FormulaBlockProps {
  block: DocBlock
}

const KATEX_OPTIONS: KatexOptions = {
  displayMode: true,
  throwOnError: true,
  strict: false,
  trust: false,
  output: 'html',
}

export function FormulaBlock({ block }: FormulaBlockProps) {
  const rawLatex = (block.latex ?? '').trim()
  const number = (block.number ?? '').trim()
  const image = (block.image_url ?? '').trim()

  // An extracted equation often still carries its own `\tag{3}`; KaTeX would
  // print that number at the right margin, so it is removed and the number chip
  // below takes over — otherwise the equation shows its number twice.
  const latex = number ? rawLatex.replace(/\\tag\*?\s*\{[^{}]*\}/g, '').trim() : rawLatex

  const html = useMemo(() => {
    if (!latex) return null
    try {
      return katex.renderToString(latex, KATEX_OPTIONS)
    } catch (error) {
      console.warn('[FormulaBlock] KaTeX could not render this equation', error)
      return null
    }
  }, [latex])

  return (
    <figure className="doc-formula" aria-label={number ? `公式 ${number}` : '公式'}>
      <div className="doc-formula-row">
        <div className="doc-formula-body">
          {html ? (
            <div className="doc-formula-rendered" dangerouslySetInnerHTML={{ __html: html }} />
          ) : image ? (
            <img
              className="doc-formula-image"
              src={image}
              alt={number ? `公式 ${number}` : '公式'}
              loading="lazy"
            />
          ) : (
            <code className="doc-formula-raw">{latex || rawLatex || '公式内容缺失'}</code>
          )}
        </div>
        {number && <span className="doc-formula-number">({number})</span>}
      </div>
    </figure>
  )
}
