/**
 * The document model returned by `GET /api/reader/{paper_id}/document`.
 *
 * Mirrors `backend/app/render/model.py`; the backend owns the shape, this file
 * only describes it to TypeScript.
 */

export interface DocSentence {
  id: string
  source: string
  target: string
  block_id: string
}

export interface DocCell {
  text: string
  target: string
  rowspan: number
  colspan: number
  header: boolean
}

export type DocBlockKind =
  | 'heading'
  | 'paragraph'
  | 'list'
  | 'table'
  | 'figure'
  | 'formula'
  | 'footnote'
  | 'reference'

export interface DocBlock {
  kind: DocBlockKind
  page_no: number
  id: string
  bbox: number[]
  /** heading */
  level?: number
  /** heading, paragraph, footnote, reference */
  text?: string
  target?: string
  sentences?: DocSentence[]
  /** list */
  items?: string[]
  /** table */
  caption?: string
  caption_target?: string
  header_rows?: number
  rows?: DocCell[][]
  /** figure, formula, table */
  image_url?: string
  latex?: string
  number?: string
  note?: string
}

export interface PaperDocument {
  paper_id: string
  title: string
  title_target: string
  authors: string[]
  abstract: string
  abstract_target: string
  lang: string
  counts: Record<string, number>
  blocks: DocBlock[]
}

/** How much of the translation the reader shows. */
export type ReadLang = 'en' | 'zh' | 'both'
