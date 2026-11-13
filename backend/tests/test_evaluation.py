"""The three rule based RAG metrics.

They are computed without an LLM judge on purpose -- see the module
docstring of ``app.rag/evaluation.py``.
"""

from __future__ import annotations

from app.rag import evaluation


CITATIONS = [
    {"index": 1, "chunk_id": "c1",
     "quote": "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German."},
    {"index": 2, "chunk_id": "c2",
     "quote": "We use 8 NVIDIA P100 GPUs for training."},
]


def test_a_fully_supported_answer_scores_one() -> None:
    answer = ("The model achieves 28.4 BLEU on the WMT 2014 English-to-German task [1]. "
              "Training used eight NVIDIA P100 GPUs [2].")
    report = evaluation.evaluate_answer(answer, CITATIONS)
    assert report.faithfulness == 1.0
    assert report.citation_traceability == 1.0
    assert report.context_precision == 1.0
    assert report.unsupported == []


def test_a_dangling_citation_is_not_traceable() -> None:
    answer = "The model achieves 28.4 BLEU [1] and 41.0 BLEU [9]."
    report = evaluation.evaluate_answer(answer, CITATIONS)
    assert report.total_citations == 2
    assert report.resolvable_citations == 1
    assert report.citation_traceability == 0.5


def test_a_claim_without_a_citation_is_not_supported() -> None:
    answer = ("The model was trained on eight NVIDIA P100 GPUs [2]. "
              "It also won a Turing Award.")
    report = evaluation.evaluate_answer(answer, CITATIONS)
    assert len(report.claims) == 2
    assert report.unsupported
    assert "Turing Award" in report.unsupported[0].text
    assert report.faithfulness == 0.5


def test_a_claim_the_evidence_does_not_mention_is_not_supported() -> None:
    answer = "The model was trained for 3.5 days on eight GPUs [2]."
    report = evaluation.evaluate_answer(answer, CITATIONS)
    assert report.faithfulness == 0.0
    assert report.unsupported[0].critical < 1.0
    assert "关键事实缺失" in report.unsupported[0].reason


def test_context_precision_separates_retrieval_from_generation() -> None:
    """Only one of the two retrieved passages is used -> precision 0.5."""
    answer = "The model achieves 28.4 BLEU on the WMT 2014 English-to-German task [1]."
    report = evaluation.evaluate_answer(answer, CITATIONS)
    assert report.context_precision == 0.5
    assert report.used_chunk_ids == ["c1"]
    assert report.retrieved_chunk_ids == ["c1", "c2"]


def test_an_answer_without_citations_has_no_traceability() -> None:
    report = evaluation.evaluate_answer("The model achieves 28.4 BLEU.", CITATIONS)
    assert report.total_citations == 0
    assert report.citation_traceability == 0.0


def test_headings_and_table_rows_are_not_claims() -> None:
    answer = ("## Results\n"
              "| model | BLEU |\n"
              "|---|---|\n"
              "| Transformer | 28.4 |\n"
              "\n"
              "The model achieves 28.4 BLEU on WMT 2014 English-to-German [1].")
    claims = evaluation.extract_claims(answer)
    assert len(claims) == 1
    assert claims[0].citations == [1]


def test_chinese_answers_are_split_and_covered_too() -> None:
    citations = [{"index": 1, "chunk_id": "c1",
                  "quote": "模型在 WMT 2014 英德翻译任务上取得 28.4 BLEU 分数。"}]
    answer = "该模型在 WMT 2014 英德翻译任务上取得 28.4 BLEU 分数 [1]。"
    report = evaluation.evaluate_answer(answer, citations)
    assert report.faithfulness == 1.0
    assert report.citation_traceability == 1.0


# --- tables -----------------------------------------------------------------
#
# A cross-paper answer is a table, and the value in a cell is the claim.  The
# table used to be dropped from the analysis altogether, which left the
# comparison questions with no claim at all.

TABLE_ANSWER = (
    "| Metric | Paper A | Paper B |\n"
    "|---|---|---|\n"
    "| BLEU (WMT 2014 En-De) | 41.0 [1] | 28.4 [2] |\n"
)


def test_a_table_cell_becomes_a_claim() -> None:
    claims = evaluation.extract_claims(TABLE_ANSWER)
    assert len(claims) == 2
    assert claims[0].citations == [1]
    assert "BLEU" in claims[0].text
    assert "41.0" in claims[0].text


def test_a_table_cell_that_contradicts_the_passage_is_not_supported() -> None:
    citations = [
        {"index": 1, "chunk_id": "c1", "quote": "BLEU score of 41.8 on WMT 2014."},
        {"index": 2, "chunk_id": "c2", "quote": "BLEU score of 28.4 on WMT 2014."},
    ]
    report = evaluation.evaluate_answer(TABLE_ANSWER, citations)
    assert len(report.claims) == 2
    assert report.faithfulness == 0.5
    assert report.unsupported[0].critical < 1.0


# --- critical facts ---------------------------------------------------------

def test_a_wrong_model_name_is_not_supported() -> None:
    """Eight A100 GPUs are not eight P100 GPUs, even if the wording fits."""
    citations = [{"index": 2, "chunk_id": "c2",
                  "quote": "We use 8 NVIDIA P100 GPUs for training."}]
    answer = "The model was trained on eight NVIDIA A100 GPUs [2]."
    report = evaluation.evaluate_answer(answer, citations)
    assert report.faithfulness == 0.0
    assert report.unsupported[0].critical < 1.0


def test_generic_framing_words_do_not_count_against_a_claim() -> None:
    """A passage that lists the authors supports "the 6th author is ..."."""
    citations = [{"index": 1, "chunk_id": "c1",
                  "quote": "Illia Polosukhin*\nGoogle Research"}]
    answer = "The 6th author is Illia Polosukhin [1]."
    report = evaluation.evaluate_answer(answer, citations)
    assert report.faithfulness == 1.0


def test_the_quote_points_at_the_line_that_backs_the_claim() -> None:
    passage = ("arXiv:1706.03762v7  [cs.CL]\n"
               "Ashish Vaswani*\n"
               "Google Brain\n"
               "avaswani@google.com\n"
               "Noam Shazeer*\n"
               "Google Brain\n"
               "noam@google.com\n"
               "Illia Polosukhin*\n"
               "illia.polosukhin@gmail.com\n"
               "Abstract\n"
               "The dominant sequence transduction models are based on recurrent "
               "neural networks.")
    window = evaluation.best_supporting_window(passage, ["Illia Polosukhin"])
    assert "Illia Polosukhin" in window
    assert "Abstract" not in window
    assert len(window) < len(passage)


def test_a_passage_that_states_nothing_of_the_claim_falls_back_to_its_head() -> None:
    passage = "arXiv:1706.03762v7\nAbstract\nThe dominant sequence transduction models."
    window = evaluation.best_supporting_window(passage, ["Vaswani was an author"])
    assert window.startswith("arXiv:1706.03762v7")


# --- comparative claims -----------------------------------------------------

def test_a_comparative_claim_is_judged_by_its_cited_passages_together() -> None:
    """A cross-paper comparison draws one fact from each paper.

    No single passage covers the whole sentence, so a per-passage judgement
    would call it unsupported even though the cited evidence does support it.
    """
    citations = [
        {"index": 1, "chunk_id": "c1",
         "quote": "BERT uses a bidirectional encoder."},
        {"index": 2, "chunk_id": "c2",
         "quote": "The original Transformer includes an autoregressive decoder."},
    ]
    answer = ("BERT uses a bidirectional encoder [1] whereas the original "
              "Transformer includes an autoregressive decoder [2].")
    report = evaluation.evaluate_answer(answer, citations)
    assert report.unsupported == []
    assert report.faithfulness == 1.0


def test_a_claim_scattered_over_many_passages_needs_one_real_anchor() -> None:
    """Citing a pile of passages must not buy a free pass.

    The words of the claim are spread thinly over four passages; no single one
    of them is about what the claim says, so the claim is not supported.
    """
    citations = [
        {"index": 1, "chunk_id": "c1", "quote": "attention mechanism"},
        {"index": 2, "chunk_id": "c2", "quote": "queries keys"},
        {"index": 3, "chunk_id": "c3", "quote": "values scaled"},
        {"index": 4, "chunk_id": "c4", "quote": "dot product"},
    ]
    answer = ("The attention mechanism queries keys values scaled dot product "
              "[1][2][3][4].")
    report = evaluation.evaluate_answer(answer, citations)
    assert report.faithfulness == 0.0
    assert "没有单段证据支撑" in report.unsupported[0].reason


def test_a_single_passage_is_judged_exactly_as_before() -> None:
    """The union does not change the judgement when one passage is cited."""
    citations = [{"index": 2, "chunk_id": "c2",
                  "quote": "We use 8 NVIDIA P100 GPUs for training."}]
    answer = "The model was trained on eight NVIDIA A100 GPUs [2]."
    report = evaluation.evaluate_answer(answer, citations)
    assert report.faithfulness == 0.0
    assert report.unsupported[0].critical < 1.0
