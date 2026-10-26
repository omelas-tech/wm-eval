import math

import pytest

from wm_eval.stats import mcnemar_exact, normalized, wilson


def test_wilson_matches_reference_values():
    # Reference: Newcombe (1998), Table I, 81/263 -> 0.2553 to 0.3662.
    lo, hi = wilson(81, 263)
    assert lo == pytest.approx(0.2553, abs=5e-4)
    assert hi == pytest.approx(0.3662, abs=5e-4)


def test_wilson_edges_stay_in_unit_interval():
    assert wilson(0, 50)[0] == 0.0
    assert wilson(50, 50)[1] == 1.0
    lo, hi = wilson(43, 50)  # a typical "86%" headline
    assert hi - lo > 0.18  # the interval a 50-episode number really carries


@pytest.mark.parametrize("k, n", [(-1, 10), (11, 10), (0, 0)])
def test_wilson_rejects_bad_counts(k, n):
    with pytest.raises(ValueError):
        wilson(k, n)


def test_normalized_score():
    assert normalized(0.26, 0.26) == 0.0
    assert normalized(1.0, 0.26) == 1.0
    assert normalized(0.63, 0.26) == pytest.approx(0.5)
    assert normalized(0.5, 0.25, 0.75) == pytest.approx(0.5)
    assert normalized(0.9, 1.0) is None


def test_mcnemar_exact():
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    # 10 discordant pairs, all favouring a: p = 2 * 0.5**10
    a = [True] * 10 + [False] * 5
    b = [False] * 10 + [False] * 5
    res = mcnemar_exact(a, b)
    assert (res["only_a"], res["only_b"]) == (10, 0)
    assert res["p_value"] == pytest.approx(2 * 0.5**10)
    # symmetric split carries no evidence
    assert mcnemar_exact([True, False] * 4, [False, True] * 4)["p_value"] == pytest.approx(
        min(1.0, 2 * sum(math.comb(8, i) for i in range(5)) / 2**8)
    )


def test_mcnemar_requires_pairs():
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])
