/**
 * The re-laid-out document reading view.
 *
 * `DocumentView` is the entry point: hand it a `PaperDocument` and the reading
 * language, and it renders title, authors and every block in reading order.
 * The block components are exported on their own as well, for callers that want
 * to compose a view themselves (a single table in a sidebar, say).
 *
 * `DocumentView` imports `styles/document.css`; the block components depend on
 * those class names, so mount them inside a `doc-` styled page.
 */
export { DocumentView } from './DocumentView'
export type { DocumentViewProps } from './DocumentView'

export { SentenceText, CaptionText } from './SentenceText'
export type { SentenceTextProps, CaptionTextProps } from './SentenceText'

export { TableBlock } from './TableBlock'
export type { TableBlockProps } from './TableBlock'

export { FigureBlock } from './FigureBlock'
export type { FigureBlockProps } from './FigureBlock'

export { FormulaBlock } from './FormulaBlock'
export type { FormulaBlockProps } from './FormulaBlock'
