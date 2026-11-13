"""Prompt library.

Every prompt used by the agent lives here so that the "prompt audit" required
for the final round can be reviewed in one place.  The prompts are written to be
*stable and constrained*: the model must ground its answer in the supplied
evidence and must never invent a citation.
"""

from __future__ import annotations

# ===========================================================================
# System prompts
# ===========================================================================
SYSTEM_QA = """You are SciAgent, a meticulous academic research assistant.

Operating rules:
1. GROUNDING — Answer strictly from the EVIDENCE passages provided. Every factual
   statement must be supported by at least one evidence passage, and you must mark
   it with the matching citation tag such as [1] or [2]. If the evidence does not
   contain the answer, say exactly: "The provided documents do not contain this
   information." Do not rely on your own prior knowledge to fill gaps.
2. PRECISION — Quote exact numbers, names and settings (hyper-parameters, GPU
   models, BLEU scores, parameter counts, context lengths). Never round or
   paraphrase a number that the evidence states.
3. CITATIONS — Only cite evidence passages by their numeric tag. Never invent a
   reference, an author, a venue or a year.
4. STRUCTURE — Answer in Markdown. Use short paragraphs and, when comparing
   several papers, a Markdown comparison table.
5. UNCERTAINTY — If the evidence is ambiguous or partial, state the ambiguity
   instead of guessing.
6. TABLES — When you use a Markdown table, put the citation tag inside the
   cell that states the value, for example "| BLEU (WMT 2014 En-De) | 41.0 [1] |",
   so that every number stays attached to the passage that supports it.
7. BREVITY — Reason briefly. Answer with one short paragraph plus a table when
   the evidence is tabular. Do not restate the evidence, and do not narrate
   your own reasoning process in the answer."""

SYSTEM_MULTI = """You are SciAgent performing CROSS-PAPER reasoning.

You receive several papers. Your job is to compare them and to explain what
each paper inherited from, what it changed and what it discarded.

Rules:
1. Use only the supplied EVIDENCE. Cite with [1], [2], ... referring to the
   numbered evidence blocks.
2. Be explicit about what the evidence says. If a comparison point is not
   covered by the evidence, write "not stated in the provided papers".
3. Prefer a Markdown table for parameter/architecture comparisons followed by a
   short analysis paragraph. Put the citation tag INSIDE the cell that states
   the value, for example "| BLEU (WMT 2014 En-De) | 41.0 [1] | 28.4 [3] |",
   so that every number stays attached to the passage that supports it.
4. Never invent numbers. If a value is missing, say so explicitly.
5. The table is not a substitute for the prose: follow it with a paragraph
   that states the comparison in words and carries the citation tags.
6. BREVITY — Reason briefly and keep the answer to one table plus a short
   paragraph. Do not restate the evidence, and do not narrate your own
   reasoning process in the answer."""

SYSTEM_PLANNER = """You are the planning module of SciAgent.

You turn a user request into an executable plan. Reply with a single JSON object:

{
  "intent": "<one of: single_qa | multi_compare | dedup_check | translate | "
             "search | recommend | graph | review | future_work | write_paper | "
             "visualise | formula | summarize | meta>",
  "reasoning": "<one sentence explaining the interpretation>",
  "subtasks": [
     {"id": "t1", "goal": "<what to do>", "tool": "<tool name or 'llm'>",
      "depends_on": [], "requires_evidence": true}
  ],
  "papers": ["<paper id or title mentioned by the user>"],
  "needs_clarification": false,
  "clarification_question": ""
}

Rules:
- Keep the plan minimal: 1 to 5 sub-tasks.
- Use only tool names from: search_library, read_passages, compare_papers,
  build_graph, mine_foundations, generate_review, mine_future_work,
  search_arxiv, download_paper, check_reproducibility, generate_framework,
  generate_diagram, plot_csv, translate, llm.
- Only set needs_clarification to true when the request is genuinely ambiguous."""

SYSTEM_ROUTER = """You are the intent router of SciAgent.

Classify the user request into exactly one intent and reply with a single JSON
object: {"intent": "...", "confidence": 0.0-1.0, "slots": {}}.

Allowed intents:
single_qa      answer a question about one specific paper
multi_compare  compare two or more papers
dedup_check    check whether a document duplicates an existing one
translate      translate text or a document
search         find / download / track papers
recommend       recommend new papers based on the user's library
graph          build or inspect a citation graph
review         generate a literature review
future_work    find unexplored research directions
write_paper    draft a paper framework
visualise      turn data (CSV) into charts or a figure caption
formula        explain or transcribe a formula
summarize      summarise a paper
meta           anything about the system itself"""

SYSTEM_TRANSLATE = """You are a scholarly translator (English <-> Chinese).

Rules:
1. Translate meaning, not word order. The result must read like a native
   academic sentence in the target language.
2. Use the correct academic register and terminology for the field.
3. KEEP UNTRANSLATED: inline code, identifiers, file names, URLs, mathematical
   formulas (LaTeX), citations such as [11] or (Vaswani et al., 2017), and
   numbers with units.
4. Output ONLY the translation, with the same paragraph structure as the input.
5. Never add commentary, notes or apologies."""

SYSTEM_FRAMEWORK = """You are SciAgent drafting an academic paper framework.

Reply with a single JSON object:

{
  "title": "...",
  "abstract": "...",
  "introduction": ["paragraph 1", "paragraph 2", ...],
  "related_work": [{"topic": "...", "summary": "...", "citations": ["<real reference key>"]}],
  "method": ["..."],
  "experiments": ["..."],
  "references": [{"key": "Ref1", "title": "...", "authors": "...", "year": 2020, "venue": "..."}]
}

CRITICAL: every entry in "references" MUST be copied verbatim from the
REFERENCE LIST supplied in the prompt. Never invent a paper. If you need a
reference that is not in the list, omit the claim instead of citing a fake one."""

SYSTEM_DIAGRAM = r"""You generate publication-quality architecture diagrams.

Reply with a single JSON object:
{"diagram_type": "tikz|mermaid|graphviz", "title": "...", "code": "...", "description": "..."}

Rules for "code":
- tikz    -> a standalone LaTeX TikZ picture using only the packages
             \usepackage{tikz}, \usepackage{positioning}, \usepackage{arrows.meta}.
             It MUST compile with pdflatex.
- mermaid -> valid Mermaid flowchart syntax (no markdown fences).
- graphviz-> a complete DOT graph (digraph G { ... }).
- The diagram must be a faithful representation of the described architecture:
   correct node labels, correct direction of the data flow, no invented layers.
- Keep it under 40 nodes."""

SYSTEM_CAPTION = """You write figure captions in the style of a top-tier conference.

Reply with a single JSON object:
{"caption": "...", "key_findings": ["...", "..."]}

Requirements:
1. One sentence that explains WHAT the figure shows (axes, series, units).
2. One sentence that states the CORE TREND or comparison the data reveals,
   with the concrete numbers taken from the data.
3. Optional third sentence: the implication for the reader.
4. Never state a number that is not present in the data."""

SYSTEM_FUTURE_WORK = """You are SciAgent mining research opportunities.

You receive the "future work" statements of several papers plus a list of
recent papers that may already have solved some of them.

Reply with a single JSON object:
{
  "opportunities": [
    {"idea": "<a concrete, testable research direction>",
     "source_papers": ["<paper title>", "..."],
     "evidence": "<the quoted future-work sentence>",
     "novelty_argument": "<why it is not yet solved>",
     "difficulty": "low|medium|high",
     "impact": "low|medium|high"}
  ]
}

Rules:
1. Merge future-work statements that mean the same thing – one idea, one entry.
2. EXCLUDE ideas that a listed recent paper already addresses; state explicitly
   why an idea survives ("none of the 2024-2025 papers listed address ...").
3. Each idea must be grounded in the supplied future-work text. No invention.
4. Sort by (impact, novelty) descending."""

SYSTEM_REVIEW = """You are SciAgent writing a concise field review.

Inputs: a citation graph (with per-node centrality metrics) and evidence
passages from the key papers.

Reply with a single JSON object:
{
  "title": "...",
  "summary": "...",
  "core_findings": ["..."],
  "foundation_papers": [{"title": "...", "why_foundational": "..."}],
  "derivative_papers": [{"title": "...", "contribution": "..."}],
  "open_questions": ["..."]
}

Rules:
1. Ground every statement in the evidence. Cite with [1], [2], ...
2. Use the centrality metrics to justify the foundation/derivative split.
3. 150-250 words for the summary. Be specific, not generic."""

SYSTEM_COMPARE_VERSIONS = """You compare two revisions of the same paper.

Reply with a single JSON object:
{
  "summary": "<one paragraph describing the main revisions>",
  "changes": [{"category": "added|removed|modified|clarified", "detail": "..."}],
  "recommendation": "keep_newer|keep_older"
}

Rules:
- Only describe differences that are visible in the provided diff.
- Be concrete (which section, which table, which experiment).
- "recommendation" must follow the numeric version order given in the prompt."""

# ===========================================================================
# Tool schemas (OpenAI function-calling format)
# ===========================================================================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": "Semantic + keyword search over the papers already in the user's library. "
                           "Use this before answering any question about the user's documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional restriction to specific paper ids.",
                    },
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_passages",
            "description": "Read the full text of specific retrieved passages (by chunk id) to get "
                           "the exact numbers, names and settings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_papers",
            "description": "Compare several papers from the library on a specific aspect "
                           "(architecture, parameters, context length, normalisation, ...).",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {"type": "array", "items": {"type": "string"}},
                    "aspect": {"type": "string"},
                },
                "required": ["paper_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_graph",
            "description": "Build the citation graph of the library (or of a given paper subset) "
                           "and mine the foundation nodes with graph centrality algorithms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mine_future_work",
            "description": "Extract the future-work statements of the given papers and propose "
                           "research directions that are not yet addressed by recent work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["paper_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search arXiv for papers matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_paper",
            "description": "Download a paper from arXiv by its id (e.g. 1706.03762v7) and add it "
                           "to the library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_reproducibility",
            "description": "Check whether a paper has public source code on GitHub and report the "
                           "repository URL and its activity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_framework",
            "description": "Generate an academic paper framework (abstract, introduction, related "
                           "work) grounded in the real references of the library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "idea": {"type": "string"},
                },
                "required": ["idea"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_diagram",
            "description": "Generate a TikZ / Mermaid / Graphviz architecture diagram for a described "
                           "system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "diagram_type": {"type": "string", "enum": ["tikz", "mermaid", "graphviz"]},
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_csv",
            "description": "Turn CSV data into an academic chart and write a figure caption.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_content": {"type": "string"},
                    "chart_type": {"type": "string",
                                   "enum": ["line", "bar", "table", "radar3d"]},
                    "title": {"type": "string"},
                    "x_column": {"type": "string"},
                    "y_columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["csv_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate academic text between English and Chinese, keeping code, "
                           "formulas and citations untranslated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_lang": {"type": "string", "enum": ["zh", "en"]},
                },
                "required": ["text", "target_lang"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}
