from promo_finder.pairwise_comparison.schemas import PairOutcome
from promo_finder.cold_viewer.schemas import JudgedCandidate
from promo_finder.cold_viewer.service import JudgeContext
from promo_finder.config import build_model_settings
from promo_finder.llm.cache.repository import LlmCacheRepository
from promo_finder.llm.schemas import LlmCompletion, LlmUsage
from promo_finder.llm.service import LlmService
from promo_finder.pairwise_comparison.service import compare_in_pairs, fit_bradley_terry, select_pairs
from tests.cold_viewer.test_service import BIBLE, _build_promo


def _outcome(first: str, second: str, first_score: float) -> PairOutcome:
    return PairOutcome(first_promo_id=first, second_promo_id=second, first_score=first_score, reason_first_shown_first="", reason_second_shown_first="")


def test_neighbour_window_limits_the_pairs():
    assert select_pairs(4, 3) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert len(select_pairs(15, 3)) == 39
    assert select_pairs(1, 3) == []


def test_bradley_terry_orders_a_consistent_chain():
    strengths = fit_bradley_terry(["a", "b", "c"], [_outcome("a", "b", 1.0), _outcome("b", "c", 1.0), _outcome("a", "c", 1.0)])
    assert strengths["a"] > strengths["b"] > strengths["c"]


def test_ties_from_order_disagreement_keep_promos_level():
    strengths = fit_bradley_terry(["a", "b"], [_outcome("a", "b", 0.5)])
    assert abs(strengths["a"] - strengths["b"]) < 1e-9


def test_a_promo_that_wins_everything_keeps_a_finite_strength():
    strengths = fit_bradley_terry(["a", "b"], [_outcome("a", "b", 1.0)])
    assert 1.0 < strengths["a"] < float("inf")


def test_pairwise_counts_a_win_only_when_it_survives_swapping_the_order(tmp_path):

    class _AlwaysFirstClient:
        """A judge with pure position bias: always picks whatever is shown as A."""

        def complete_json(self, request):
            return LlmCompletion(text='{"winner": "A", "reason": "first shown"}', usage=LlmUsage(), served_by_model="test")

    show, promo = _build_promo(tmp_path)
    other = promo.model_copy(update={"candidate_id": "other"})
    context = JudgeContext(
        show=show, line_map={}, features_by_block_id={}, bible=BIBLE,
        llm_service=LlmService(_AlwaysFirstClient(), LlmCacheRepository(tmp_path / "cache")), model_settings=build_model_settings(),
    )
    result = compare_in_pairs([JudgedCandidate(candidate=promo), JudgedCandidate(candidate=other)], context)
    assert [outcome.first_score for outcome in result.outcomes] == [0.5]
    assert result.strength_by_promo_id == {promo.candidate_id: 0.5, "other": 0.5}
