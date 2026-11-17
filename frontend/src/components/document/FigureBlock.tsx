/**
 * A figure of the re-laid-out paper: the crop of the page, plus its caption.
 *
 * The caption follows the reading language: in `zh`/`both` the translation
 * leads and the English original stays as a weaker secondary line.
 */
import type { DocBlock, ReadLang } from '../../lib/document'
import { CaptionText } from './SentenceText'

export interface FigureBlockProps {
  block: DocBlock
  lang: ReadLang
}

export function FigureBlock({ block, lang }: FigureBlockProps) {
  const caption = (block.caption ?? '').trim()
  const captionTarget = (block.caption_target ?? '').trim()
  const image = (block.image_url ?? '').trim()
  const alt = captionTarget || caption || `第 ${block.page_no} 页插图`

  return (
    <figure className="doc-figure">
      {image ? (
        <img className="doc-figure-image" src={image} alt={alt} loading="lazy" />
      ) : (
        <div className="doc-figure-missing">
          图片未生成（可在原文第 {block.page_no} 页查看）
        </div>
      )}
      {(caption || captionTarget) && (
        <figcaption className="doc-caption">
          <CaptionText caption={caption} captionTarget={captionTarget} lang={lang} />
        </figcaption>
      )}
    </figure>
  )
}
