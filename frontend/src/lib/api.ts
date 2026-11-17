/**
 * Typed API client for the SciAgent backend.
 *
 * Every call goes through the same-origin `/api` prefix so the production
 * deployment (nginx) and the development server (vite proxy) behave
 * identically.
 */

import type { PaperDocument } from './document'

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

/* ------------------------------------------------------------------ types */
export interface Paper {
  id: string
  title: string
  authors: { name: string; affiliations?: number[] }[]
  affiliations: string[]
  year: number | null
  arxiv_id: string
  version_label: string
  page_count: number
  is_scanned: boolean
  ocr_pages: number
  status: string
  filename?: string
  abstract?: string
  venue?: string
  doi?: string
  created_at: string
  metadata: Record<string, unknown>
}

export interface Section {
  id: string
  title: string
  number: string
  level: number
  page_no: number
  page_start?: number
  page_end?: number
  bbox: number[]
  content: string
  order_index: number
}

export interface ReferenceItem {
  id: string
  index_in_doc: number
  raw_text: string
  authors: string[]
  title: string
  year: number | null
  venue: string
  doi: string
  arxiv_id: string
  linked_paper_id: string | null
}

export interface FigureItem {
  id: string
  page_no: number
  caption: string
  image_path: string
  image_url?: string
  bbox: number[]
  description: string
}

export interface TableItem {
  id: string
  page_no: number
  caption: string
  markdown: string
  csv: string
  image_url?: string
  bbox: number[]
}

export interface FormulaItem {
  id: string
  page_no: number
  latex: string
  raw_text: string
  image_url?: string
  bbox: number[]
}

/** One block of the parsed document, in reading order. */
export interface Block {
  id: string
  page_no: number
  /** body | caption | footnote | figure_label | header | text */
  type: string
  bbox: number[]
  content: string
  section_id?: string | null
  order_index?: number
}

export interface PaperStructure {
  paper_id: string
  title: string
  page_count?: number
  sections: Section[]
  blocks?: Block[]
  tables: TableItem[]
  figures: FigureItem[]
  formulas: FormulaItem[]
  references: ReferenceItem[]
}

export interface Citation {
  index: number
  paper_id: string
  paper_title: string
  page_no: number
  section_title: string
  chunk_id: string
  char_start: number
  char_end: number
  quote: string
  score: number
  source?: string
  /** How strongly the query matched the paper's metadata (0..1). */
  metadata_affinity?: number
}

export interface Passage extends Citation {
  ordinal: number
  kind: string
  content: string
  sparse_rank: number
  dense_rank: number
  rerank_score: number
}

export interface GraphNode {
  paper_id: string
  title: string
  label: string
  year: number | null
  arxiv_id: string
  version_label: string
  is_scanned: boolean
  in_degree: number
  out_degree: number
  pagerank: number
  betweenness: number
  k_core: number
  hub_score: number
  authority_score: number
  foundation_score: number
  role: 'foundation' | 'core' | 'derivative'
  cited_by: string[]
  cites: string[]
}

export interface GraphEdge {
  source: string
  target: string
  relevance: number
  context: string
  stance: 'supporting' | 'contrasting' | 'mentioning'
  stance_label: string
  stance_reason: string
  /** citation | co_citation | co_reference */
  kind?: string
  /** How many shared items the two papers have (similarity edges). */
  weight?: number
  label?: string
  shared_citing_papers?: string[]
  shared_references?: string[]
}

export interface ConflictSide {
  metric: string
  value: number
  unit: string
  text: string
  context: string
  paper_id: string
  paper_title: string
  page_no: number
  chunk_id: string
}

export interface Conflict {
  kind: 'numeric' | 'claim'
  metric: string
  summary: string
  left: ConflictSide
  right: ConflictSide
  delta: number | null
  confidence: number
  reason: string
}

export interface ContradictionResult {
  conflicts: Conflict[]
  count: number
  numeric_count: number
  claim_count: number
  papers_involved: string[]
  scanned_chunks: number
  markdown?: string
  note?: string
}

export interface CitationRef {
  title: string
  authors: string[]
  year: number | null
  relevance?: number
  context?: string
  stance?: 'supporting' | 'contrasting' | 'mentioning'
  stance_label?: string
  stance_reason?: string
  citation_count?: number
}

export interface CitationGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  stats: Record<string, number | string | boolean>
}

/** One OpenAI-compatible endpoint the app can talk to. */
export interface ModelProfile {
  id: string
  label: string
  base_url: string
  model: string
  has_key: boolean
  key_hint: string
}

export interface LibraryStats {
  papers: number
  pages: number
  chunks: number
  scanned_papers: number
  references: number
  tables: number
  figures: number
  formulas: number
}

/* -------------------------------------------------------------- utilities */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(init?.headers ?? {}),
    },
  })
  if (!res.ok) {
    let detail: unknown
    let message = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      detail = body?.detail ?? body
      if (typeof detail === 'string') message = detail
      else if (detail && typeof detail === 'object') {
        message = (detail as Record<string, string>).error ?? message
      }
    } catch {
      /* keep the status text */
    }
    throw new ApiError(message, res.status, detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

/* ------------------------------------------------------------------ API */
export const api = {
  /* ---- health ---- */
  health: () => request<Record<string, unknown>>('/health'),
  config: () => request<Record<string, unknown>>('/config'),

  /* ---- model endpoints ---- */
  models: () => request<{ active: string | null; profiles: ModelProfile[] }>('/models'),
  addModel: (body: { label: string; base_url: string; model: string; api_key: string }) =>
    request<ModelProfile>('/models', { method: 'POST', body: JSON.stringify(body) }),
  activateModel: (id: string) =>
    request<{ active: string }>(`/models/${id}/activate`, { method: 'POST' }),
  deleteModel: (id: string) => request<{ deleted: string }>(`/models/${id}`, { method: 'DELETE' }),
  probeModel: (body: { base_url: string; model?: string; api_key?: string }) =>
    request<{ ok: boolean; models?: string[]; error?: string }>('/models/probe', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /* ---- library ---- */
  listPapers: (q = '') => request<{ papers: Paper[]; count: number }>(`/papers?q=${encodeURIComponent(q)}`),
  stats: () => request<LibraryStats>('/papers/stats'),
  getPaper: (id: string) => request<Paper>(`/papers/${id}`),
  structure: (id: string) => request<PaperStructure>(`/papers/${id}/structure`),
  references: (id: string) => request<{ references: ReferenceItem[] }>(`/papers/${id}/references`),
  deletePaper: (id: string) => request<{ deleted: string }>(`/papers/${id}`, { method: 'DELETE' }),

  upload: async (files: File[], onProgress?: (pct: number) => void) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    // XHR gives us upload progress events (fetch does not, reliably).
    return new Promise<{ accepted: { paper_id: string; filename: string }[] }>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/papers/upload`)
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          reject(new ApiError(`upload failed (${xhr.status})`, xhr.status))
        }
      }
      xhr.onerror = () => reject(new ApiError('network error during upload', 0))
      xhr.send(form)
    })
  },

  /* ---- retrieval ---- */
  search: (query: string, paperIds: string[] = [], topK = 8) =>
    request<{ passages: Passage[]; strategy: string; diagnostics: Record<string, unknown> }>(
      '/rag/search',
      { method: 'POST', body: JSON.stringify({ query, paper_ids: paperIds, top_k: topK }) },
    ),

  /* ---- graph ---- */
  graph: (paperIds: string[] = [], view: string = 'citation') =>
    request<CitationGraph>('/graph', { method: 'POST', body: JSON.stringify({ paper_ids: paperIds, view }) }),
  contradictions: (paperIds: string[] = []) =>
    request<ContradictionResult>('/graph/contradictions', {
      method: 'POST',
      body: JSON.stringify({ paper_ids: paperIds }),
    }),
  futureWork: (paperIds: string[] = []) =>
    request<{
      statements: { text: string; paper_title: string; page_no: number }[]
      opportunities: {
        idea: string
        evidence: string
        source_papers: string[]
        novelty_argument: string
        difficulty: string
        impact: string
      }[]
    }>('/graph/future-work', { method: 'POST', body: JSON.stringify({ paper_ids: paperIds }) }),
  review: (paperIds: string[] = []) =>
    request<{
      title: string
      summary: string
      core_findings: string[]
      foundation_papers: { title: string; why_foundational: string }[]
      derivative_papers: { title: string; contribution: string }[]
      open_questions: string[]
      graph: CitationGraph
    }>('/graph/review', { method: 'POST', body: JSON.stringify({ paper_ids: paperIds }) }),
  citationQuality: (paperId: string) =>
    request<{
      kept: CitationRef[]
      dropped: CitationRef[]
      kept_count: number
      dropped_count: number
      threshold: number
      stance_counts: Record<string, number>
    }>(`/graph/citations/quality?paper_id=${paperId}`),

  /* ---- search / recommend ---- */
  arxivSearch: (query: string, maxResults = 10) =>
    request<{ query: string; count: number; papers: ArxivPaper[] }>('/search/arxiv', {
      method: 'POST',
      body: JSON.stringify({ query, max_results: maxResults }),
    }),
  downloadPaper: (arxivId: string) =>
    request<{ paper_id: string; status: string; duplicate: boolean; message: string }>(
      '/search/download',
      { method: 'POST', body: JSON.stringify({ arxiv_id: arxivId }) },
    ),
  recommend: () =>
    request<{ recommendations: Recommendation[]; library_size: number }>('/search/recommend', {
      method: 'POST',
      body: JSON.stringify({}),
    }),
  feedback: (arxivId: string, rating: number, title = '') =>
    request<{ ok: boolean }>('/search/feedback', {
      method: 'POST',
      body: JSON.stringify({ arxiv_id: arxivId, rating, title }),
    }),
  recent: (category = 'cs.CL', maxResults = 30) =>
    request<{ category: string; count: number; papers: ArxivPaper[] }>(
      `/search/recent?category=${category}&max_results=${maxResults}`,
    ),

  /* ---- reader ---- */
  pages: (paperId: string) =>
    request<{ paper_id: string; page_count: number; pages: { page_no: number; image_url: string }[] }>(
      `/reader/${paperId}/pages`,
    ),
  citations: (paperId: string) =>
    request<{
      paper_id: string
      references: ReferenceItem[]
      markers: {
        chunk_id: string
        page_no: number
        char_start: number
        char_end: number
        numbers: number[]
        context: string
      }[]
      count: number
    }>(`/reader/${paperId}/citations`),
  reference: (paperId: string, index: number) =>
    request<ReferenceItem & { cited_at: { chunk_id: string; page_no: number; context: string }[] }>(
      `/reader/${paperId}/reference/${index}`,
    ),
  translateText: (text: string, targetLang = 'zh') =>
    request<{ translation: string; target_lang: string }>('/reader/translate', {
      method: 'POST',
      body: JSON.stringify({ text, target_lang: targetLang }),
    }),
  translation: (paperId: string, targetLang = 'zh') =>
    request<{
      status: 'idle' | 'running' | 'ready' | 'error'
      job_id?: string
      progress?: number
      message?: string
      error?: string
      markdown: string
      segments: {
        index: number
        source: string
        translation: string
        kind: string
        page_no: number
        block_id?: string
        block_order?: number
      }[]
      cached: boolean
      failed?: number
    }>(`/reader/${paperId}/translation?target_lang=${targetLang}`),
  /**
   * The re-laid-out document: paragraphs, tables, figures and formulas in
   * reading order, with the translation already attached to each sentence.
   */
  document: (paperId: string, lang = 'zh') =>
    request<PaperDocument>(`/reader/${paperId}/document?lang=${lang}`),
  startTranslation: (paperId: string, targetLang = 'zh') =>
    request<{
      status: 'started' | 'running' | 'ready' | 'error'
      job_id?: string
      progress?: number
      message?: string
      error?: string
      markdown?: string
      segments?: {
        index: number
        source: string
        translation: string
        kind: string
        page_no: number
        block_id?: string
        block_order?: number
      }[]
      cached: boolean
      failed?: number
    }>(`/reader/${paperId}/translation?target_lang=${targetLang}`, { method: 'POST' }),

  /* ---- writing ---- */
  framework: (idea: string) =>
    request<FrameworkResult>('/writing/framework', { method: 'POST', body: JSON.stringify({ idea }) }),
  diagram: (description: string, diagramType = 'mermaid') =>
    request<{ diagram_type: string; title: string; code: string; description: string }>(
      '/writing/diagram',
      { method: 'POST', body: JSON.stringify({ description, diagram_type: diagramType }) },
    ),
  availableReferences: () =>
    request<{ references: { title: string; authors: string[]; year: number | null; venue: string }[] }>(
      '/writing/references',
    ),

  /* ---- visualisation ---- */
  plot: (payload: {
    csv_content: string
    chart_type: string
    title?: string
    x_column?: string
    y_columns?: string[]
  }) =>
    request<{
      chart_type: string
      image_base64: string
      caption: string
      key_findings: string[]
      table_markdown: string
      columns: string[]
      row_count: number
    }>('/viz/plot', { method: 'POST', body: JSON.stringify(payload) }),

  /* ---- audit ---- */
  audit: (limit = 30, purpose = '') =>
    request<{
      records: {
        id: string
        purpose: string
        system_prompt: string
        user_prompt: string
        tools: string[]
        response: string
        reasoning: string
        prompt_tokens: number
        completion_tokens: number
        latency_ms: number
        created_at: string
      }[]
      count: number
    }>(`/admin/audit?limit=${limit}${purpose ? `&purpose=${purpose}` : ''}`),
}

/* --------------------------------------------------------------- extra types */
export interface ArxivPaper {
  arxiv_id: string
  version: string
  title: string
  authors: string[]
  summary: string
  published: string
  updated: string
  categories: string[]
  pdf_url: string
  abs_url: string
}

export interface Recommendation {
  arxiv_id: string
  title: string
  authors: string[]
  summary: string
  published: string
  abs_url: string
  relevance: number
  has_code: boolean
  repository: { full_name: string; html_url: string; stars: number; description: string } | null
  feedback: number
  score: number
  reason: string
}

export interface FrameworkResult {
  title: string
  abstract: string
  introduction: string[]
  related_work: { topic: string; summary: string; citations: string[] }[]
  method: string[]
  experiments: string[]
  references: { title: string; authors: string; year: number; venue: string; verified: boolean }[]
  rejected_references: { title: string; reason: string }[]
}
