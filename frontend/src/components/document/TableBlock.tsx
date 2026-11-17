/**
 * A table of the re-laid-out paper.
 *
 * The cells are rendered as a real `<table>` (so the reading view keeps the
 * row/column structure, `rowspan`/`colspan` and header semantics), inside a
 * horizontally scrollable box so a wide table can never push the page layout
 * around.  The page crop is one click away when the printed image carries
 * information the cells cannot (rules, alignment, in-cell drawings).
 */
import { useMemo, useState, type ReactNode } from 'react'
import type { DocBlock, DocCell, ReadLang } from '../../lib/document'
import { CaptionText } from './SentenceText'

export interface TableBlockProps {
  block: DocBlock
  lang: ReadLang
}

interface CellProps {
  cell: DocCell
  header: boolean
  lang: ReadLang
}

function CellView({ cell, header, lang }: CellProps) {
  const source = (cell.text ?? '').trim()
  const target = (cell.target ?? '').trim()
  const rowSpan = Math.max(1, cell.rowspan || 1)
  const colSpan = Math.max(1, cell.colspan || 1)

  let content: ReactNode
  if (lang === 'en') {
    content = source || null
  } else if (target) {
    content =
      lang === 'both' && source ? (
        <>
          {target}
          <span className="doc-cell-en">{source}</span>
        </>
      ) : (
        target
      )
  } else {
    // Numeric and symbolic cells are rarely translated: keep them readable
    // instead of greying a whole results table out.
    content = source || null
  }

  if (header) {
    return (
      <th scope="col" rowSpan={rowSpan} colSpan={colSpan}>
        {content}
      </th>
    )
  }
  return (
    <td rowSpan={rowSpan} colSpan={colSpan}>
      {content}
    </td>
  )
}

export function TableBlock({ block, lang }: TableBlockProps) {
  const [showImage, setShowImage] = useState(false)

  // Every row is kept, including the ones whose cells were all absorbed by a
  // rowspan above: the browser needs those rows to exist, otherwise the
  // spanning cell has nothing to span into and the grid collapses.
  const rows = useMemo(() => block.rows ?? [], [block.rows])
  const headerRows = Math.max(0, Math.min(block.header_rows ?? 0, rows.length))
  const caption = (block.caption ?? '').trim()
  const captionTarget = (block.caption_target ?? '').trim()
  const image = (block.image_url ?? '').trim()
  const viewingImage = showImage && image !== ''
  const label = captionTarget || caption || `第 ${block.page_no} 页表格`

  return (
    <figure className="doc-table">
      {(caption || captionTarget || image) && (
        <div className="doc-table-head">
          <span className="doc-caption doc-table-caption">
            <CaptionText caption={caption} captionTarget={captionTarget} lang={lang} />
          </span>
          {image && (
            <button
              type="button"
              className="doc-toggle"
              aria-pressed={viewingImage}
              onClick={() => setShowImage((value) => !value)}
            >
              {viewingImage ? '查看表格' : '查看原图'}
            </button>
          )}
        </div>
      )}

      {viewingImage ? (
        <img className="doc-table-image" src={image} alt={label} loading="lazy" />
      ) : rows.length > 0 ? (
        <div
          className="doc-table-scroll"
          role="group"
          aria-label="表格，可横向滚动"
          tabIndex={0}
        >
          <table className="doc-table-grid">
            <caption className="doc-sr-only">{label}</caption>
            <tbody>
              {rows.map((row, rowIndex) => {
                const headerRow = rowIndex < headerRows
                if (row.length === 0) {
                  // fully covered by the rowspans above: keep the row slot
                  return <tr key={rowIndex} className="doc-table-row-spacer" />
                }
                return (
                  <tr key={rowIndex} className={headerRow ? 'doc-table-row-header' : undefined}>
                    {row.map((cell, cellIndex) => (
                      <CellView
                        key={`${rowIndex}-${cellIndex}`}
                        cell={cell}
                        header={headerRow || cell.header === true}
                        lang={lang}
                      />
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="doc-empty">表格内容缺失（可在原文第 {block.page_no} 页查看）</p>
      )}
    </figure>
  )
}
