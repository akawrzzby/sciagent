"""Front-matter extraction: the author layouts and the metadata traps.

The cases are taken from the papers in the organiser's test set -- the dense
40+ author Codex paper, the two-column QLORA paper and the standard arXiv
layout -- so that a regression in any of them is caught before it reaches
the library.

Every test here corresponds to a defect that was actually observed in the
library, not to a hypothetical one.
"""

from __future__ import annotations

import pytest

from app.parsing import metadata
from app.parsing.layout import Block


def _person_names(raw: str) -> list[str]:
    return [n for n, _m in metadata._split_author_list(raw)
            if metadata._looks_like_person_name(n)]


# ------------------------------------------------------------- author layouts
def test_comma_separated_layout_is_untouched() -> None:
    raw = "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit"
    assert _person_names(raw) == ["Ashish Vaswani", "Noam Shazeer",
                                  "Niki Parmar", "Jakob Uszkoreit"]


def test_dense_author_band_is_split_on_the_affiliation_index() -> None:
    """The Codex paper prints five authors per line, separated by indices."""
    raw = ("Mark Chen * 1 Jerry Tworek * 1 Heewoo Jun * 1 Qiming Yuan * 1 "
           "Henrique Ponde de Oliveira Pinto * 1")
    assert _person_names(raw) == ["Mark Chen", "Jerry Tworek", "Heewoo Jun",
                                  "Qiming Yuan", "Henrique Ponde de Oliveira Pinto"]


def test_dense_author_band_with_a_marker_in_front_of_the_index() -> None:
    raw = "Jared Kaplan * 2 Harri Edwards 1 Yuri Burda 1 Nicholas Joseph 2"
    assert _person_names(raw) == ["Jared Kaplan", "Harri Edwards", "Yuri Burda",
                                  "Nicholas Joseph"]


def test_dense_author_band_without_indices() -> None:
    raw = "Mark Chen * Jerry Tworek * Heewoo Jun * Qiming Yuan *"
    assert _person_names(raw) == ["Mark Chen", "Jerry Tworek", "Heewoo Jun",
                                  "Qiming Yuan"]


def test_a_block_that_stacks_one_author_per_line_keeps_them_all() -> None:
    """QLORA puts three authors in one block, one per line."""
    block = Block(text="Tim Dettmers∗\nArtidoro Pagnoni∗\nAri Holtzman",
                  bbox=(0.0, 100.0, 300.0, 160.0), spans=[], page_no=1, column=0)
    meta = metadata.extract_metadata([block], body_size=10.0, filename="q.pdf",
                                     raw_text=block.text, page_width=612.0)
    assert [a.name for a in meta.authors] == ["Tim Dettmers", "Artidoro Pagnoni",
                                               "Ari Holtzman"]


def test_a_name_with_a_particle_is_still_one_person() -> None:
    assert metadata._looks_like_person_name("Henrique Ponde de Oliveira Pinto")
    assert metadata._looks_like_person_name("Felipe Petroski Such")
    assert metadata._looks_like_person_name("Andrew N. Carr")


def test_affiliations_and_places_are_never_authors() -> None:
    assert metadata._looks_like_person_name("OpenAI, San Francisco") is False
    assert metadata._looks_like_person_name("Mountain View") is False
    assert metadata._looks_like_person_name("Google Research") is False
    # A surname that merely looks like a place has to survive.
    assert metadata._looks_like_person_name("Jin Park") is True


# -------------------------------------------- the one-author-per-line template
def test_a_trailing_affiliation_index_is_split_off_the_name() -> None:
    """The TMLR template prints the institution index after the name."""
    assert metadata._split_affiliation_index("Jason Wei 1") == ("Jason Wei", "1")
    assert metadata._split_affiliation_index("Colin Raﬀel 3") == ("Colin Raffel", "3")


def test_a_trailing_number_is_kept_when_it_is_not_an_index() -> None:
    """``GPT 3`` is a model name, not an author with affiliation 3."""
    assert metadata._split_affiliation_index("GPT 3") == ("GPT 3", "")
    assert metadata._split_affiliation_index("Wei 3") == ("Wei 3", "")


def test_the_ligature_in_a_name_is_expanded() -> None:
    """LaTeX renders "Raffel" with an ff ligature."""
    assert metadata._clean("Colin Raﬀel") == "Colin Raffel"


def test_a_block_that_stacks_one_author_per_line_with_indices() -> None:
    """The TMLR template: one author per line, an index and an e-mail."""
    block = Block(
        text=("Jason Wei 1\njasonwei@google.com\n"
              "Yi Tay 1\nyitay@google.com\n"
              "Rishi Bommasani 2\nnlprishi@stanford.edu"),
        bbox=(0.0, 100.0, 300.0, 200.0), spans=[], page_no=1, column=0)
    meta = metadata.extract_metadata([block], body_size=10.0, filename="e.pdf",
                                     raw_text=block.text, page_width=612.0)
    assert [a.name for a in meta.authors] == ["Jason Wei", "Yi Tay",
                                              "Rishi Bommasani"]
    # The e-mail printed under an author belongs to that author.
    assert [a.email for a in meta.authors] == [
        "jasonwei@google.com", "yitay@google.com", "nlprishi@stanford.edu"]


def test_an_author_is_linked_to_the_institution_it_is_indexed_with() -> None:
    block = Block(
        text=("Jason Wei 1\njasonwei@google.com\n"
              "Rishi Bommasani 2\nnlprishi@stanford.edu\n"
              "1Google Research\n2Stanford University"),
        bbox=(0.0, 100.0, 300.0, 200.0), spans=[], page_no=1, column=0)
    meta = metadata.extract_metadata([block], body_size=10.0, filename="e.pdf",
                                     raw_text=block.text, page_width=612.0)
    assert meta.authors[0].affiliation_indices == [1]
    assert meta.authors[1].affiliation_indices == [2]
    assert meta.affiliations == ["1Google Research", "2Stanford University"]


def test_an_indexed_affiliation_without_a_university_word_is_kept() -> None:
    assert metadata._is_affiliation_line("3UNC Chapel Hill") is True
    assert metadata._is_affiliation_line("4DeepMind") is True
    assert metadata._is_affiliation_line("12 Introduction") is False


# ------------------------------------------------------------------ the year
@pytest.mark.parametrize("arxiv_id,expected", [
    ("1706.03762", 2017),
    ("1910.02054", 2019),
    ("1910.10683", 2019),
    ("2005.14165", 2020),
    ("2305.14314", 2023),
    ("2307.09288", 2023),
])
def test_year_is_derived_from_the_arxiv_identifier(arxiv_id: str,
                                                   expected: int) -> None:
    """The identifier encodes YYMM; scanning the page for a number does not."""
    assert metadata._year_from_arxiv_id(arxiv_id) == expected


def test_old_style_arxiv_identifiers_have_no_derivable_year() -> None:
    assert metadata._year_from_arxiv_id("cs/0701001") is None
    assert metadata._year_from_arxiv_id("") is None


# ---------------------------------------------------------------- the abstract
def test_abstract_keeps_its_own_body() -> None:
    raw = ("We show that the Transformer generalizes well to other tasks by "
           "applying it successfully to English constituency parsing.")
    assert metadata._trim_abstract(raw) == raw


def test_abstract_is_cut_at_the_equal_contribution_footnote() -> None:
    raw = ("We present Codex. ∗Equal contribution. Listing order is random. "
           "Jakob proposed replacing RNNs with self-attention. "
           "31st Conference on Neural Information Processing Systems.")
    assert metadata._trim_abstract(raw) == "We present Codex."


def test_abstract_is_cut_at_the_keyword_list() -> None:
    raw = ("We release our data set, pre-trained models, and code.1 "
           "Keywords: transfer learning, deep learning 1. Introduction")
    trimmed = metadata._trim_abstract(raw)
    assert trimmed.startswith("We release our data set")
    assert "Keywords" not in trimmed
    assert "Introduction" not in trimmed


def test_abstract_is_cut_at_the_arxiv_stamp() -> None:
    raw = ("We are releasing our models and inference code. "
           "arXiv:2212.04356v1 [eess.AS] 6 Dec 2022")
    trimmed = metadata._trim_abstract(raw)
    assert trimmed.startswith("We are releasing our models")
    assert "arXiv" not in trimmed
