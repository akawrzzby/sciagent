/**
 * The re-laid-out paper, rendered as a readable document.
 *
 * `DocumentView` owns the reading order: one block after another, each block
 * dispatched to the component that knows how to present it.  Every block keeps
 * a link back to the page it came from, so the reader never loses the thread
 * between the reflowed text and the original PDF.
 */
import '../../styles/document.css'
import type { ReactNode } from 'react'
import type { DocBlock, PaperDocument, ReadLang } from '../../lib/document'
import { FormulaBlock } from './FormulaBlock'
import { FigureBlock } from './FigureBlock'
import { SentenceText } from './SentenceText'
import { TableBlock } from './TableBlock'

export interface DocumentViewProps {
  doc: PaperDocument
  /** Which layer of the translation to show. */
  lang: ReadLang
  /** A citation marker was clicked: `'12'` for `[12]`, `'3,4'` for `[3,4]`. */
  onCite?: (marker: string) => void
  /** The page chip of a block was clicked: jump to that page of the PDF. */
  onPageClick?: (page: number) => void
}

interface BlockContentProps {
  block: DocBlock
  lang: ReadLang
  onCite?: (marker: string) => void
}

/** The body of one block, without the page chip that the view adds. */
function BlockContent({ block, lang, onCite }: BlockContentProps): ReactNode {
  if (block.kind === 'heading') {
    // The backend numbers heading levels for this view: 2 is a top level
    // section (1 is the paper title), so a block level maps straight onto an
    // <h2>/<h3>/<h4> and onto the three visual sizes.
    const depth = Math.max(1, Math.min((block.level ?? 2) - 1, 3))
    const Tag: 'h2' | 'h3' | 'h4' = depth === 1 ? 'h2' : depth === 2 ? 'h3' : 'h4'
    return (
      <Tag className={`doc-heading doc-heading-${depth}`}>
        <SentenceText
          text={block.text}
          target={block.target}
          lang={lang}
          onCite={onCite}
          showHint={false}
        />
      </Tag>
    )
  }

  if (block.kind === 'paragraph') {
    return (
      <p className="doc-paragraph">
        <SentenceText
          sentences={block.sentences}
          text={block.text}
          target={block.target}
          lang={lang}
          onCite={onCite}
        />
      </p>
    )
  }

  if (block.kind === 'list') {
    const items = (block.items ?? []).map((item) => item.trim()).filter(Boolean)
    const sentences = block.sentences ?? []
    // The backend aligns one sentence pair per list item when it can; when the
    // counts disagree we do not guess, the item simply shows its own language.
    const aligned = sentences.length === items.length
    const translated = items.some((_, index) => aligned && sentences[index].target.trim() !== '')
    if (items.length === 0) {
      return (
        <p className="doc-paragraph">
          <SentenceText sentences={sentences} lang={lang} onCite={onCite} />
        </p>
      )
    }
    // A fully untranslated list reads better as one note than one badge per row.
    const itemLang: ReadLang = lang !== 'en' && !translated ? 'en' : lang
    return (
      <>
        <ul className="doc-list">
          {items.map((item, index) => (
            <li key={index} className="doc-list-item">
              <SentenceText
                text={item}
                target={aligned ? sentences[index].target : undefined}
                lang={itemLang}
                onCite={onCite}
                showHint={false}
              />
            </li>
          ))}
        </ul>
        {lang !== 'en' && !translated && (
          <p className="doc-hint">该列表暂无译文，显示英文原文。</p>
        )}
      </>
    )
  }

  if (block.kind === 'table') return <TableBlock block={block} lang={lang} />
  if (block.kind === 'figure') return <FigureBlock block={block} lang={lang} />
  if (block.kind === 'formula') return <FormulaBlock block={block} />

  if (block.kind === 'footnote' || block.kind === 'reference') {
    const isReference = block.kind === 'reference'
    return (
      <p
        className={isReference ? 'doc-reference' : 'doc-footnote'}
        // The anchor a citation marker jumps to: `data-cite` is the number the
        // body prints, `id` is what the reader can scroll to directly.
        id={isReference ? block.id : undefined}
        data-cite={isReference ? block.number : undefined}
      >
        {isReference && block.number ? (
          <span className="doc-reference-number">[{block.number}]</span>
        ) : null}
        <SentenceText
          sentences={block.sentences}
          text={block.text}
          target={block.target}
          lang={lang}
          onCite={onCite}
        />
      </p>
    )
  }

  return null
}

export function DocumentView({ doc, lang, onCite, onPageClick }: DocumentViewProps) {
  const blocks = doc.blocks ?? []

  const sourceTitle = (doc.title ?? '').trim()
  const targetTitle = (doc.title_target ?? '').trim()
  const title = lang === 'en' ? sourceTitle : targetTitle || sourceTitle
  const titleSource =
    lang !== 'en' && targetTitle !== '' && sourceTitle !== '' && sourceTitle !== targetTitle
      ? sourceTitle
      : ''

  const authors = (doc.authors ?? []).map((author) => author.trim()).filter(Boolean)

  return (
    <article className="doc-view">
      <header className="doc-masthead">
        <h1 className="doc-title">
          {title}
          {titleSource && <span className="doc-title-source">{titleSource}</span>}
        </h1>
        {authors.length > 0 && <p className="doc-authors">{authors.join(' · ')}</p>}
      </header>

      <div className="doc-blocks">
        {blocks.map((block, index) => {
          const key = block.id || `block-${index}`
          const page = block.page_no ?? 0
          return (
            <div
              key={key}
              id={block.id || undefined}
              className="doc-block"
              data-kind={block.kind}
              data-page={page}
            >
              <div className="doc-block-body">
                <BlockContent block={block} lang={lang} onCite={onCite} />
                {block.note && <p className="doc-note">{block.note}</p>}
              </div>
              {page > 0 &&
                (onPageClick ? (
                  <button
                    type="button"
                    className="doc-page"
                    aria-label={`跳转到原文第 ${page} 页`}
                    onClick={() => onPageClick(page)}
                  >
                    p.{page}
                  </button>
                ) : (
                  <span className="doc-page doc-page-static" aria-hidden="true">
                    p.{page}
                  </span>
                ))}
            </div>
          )
        })}
      </div>

      {blocks.length === 0 && <p className="doc-empty">该文献尚未解析出可重排的正文。</p>}
    </article>
  )
}
