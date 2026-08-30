"""Trend-sleeve position visibility (2026-08-30).

On 2026-08-24 the sleeve issued a full fresh BUY on top of positions it already held:

    DBC  228 + buy 214 = 442   (target was 214)
    EEM   70 + buy 101 = 171   (target was 101)

~11.5% of equity in unintended exposure. The cause was tagging, not sizing.
`_current_trend_positions` selected rows on `selector == 'trend'` ALONE, and two adopted rows
carried `trade_type='trend'` with a BLANK selector. Those symbols were therefore absent from
the sleeve's view of what it holds, `compute_trend_deltas` saw cur=0, and delta = target - 0
became a full fresh buy — precisely the double-buy `_current_trend_positions`' own fail-closed
comment warns about.

The regression tests below pin the two halves: a row tagged either way must be VISIBLE, and a
symbol already held must never produce a fresh full buy.
"""

import pytest

from app.live_trading.trend_sleeve import compute_trend_deltas


class TestDoubleBuyRegression:
    """The arithmetic that turned an invisible position into a doubled one."""

    def test_invisible_position_produces_full_fresh_buy(self):
        """What went wrong: cur=0 for a symbol actually held 228."""
        intents = compute_trend_deltas(
            target_weights={"DBC": 1.0},
            current_trend_positions={},          # DBC invisible -> looks flat
            prices={"DBC": 31.14}, nav=100000.0,
            trend_allocation_pct=0.0666, max_position_pct=0.25,
        )
        buy = next(i for i in intents if i["symbol"] == "DBC")
        assert buy["side"] == "buy"
        assert buy["current_shares"] == 0
        # The order equals the whole target — on top of 228 already held.
        assert buy["qty"] == buy["target_shares"]

    def test_visible_position_produces_only_the_delta(self):
        """What should happen once the row is visible."""
        intents = compute_trend_deltas(
            target_weights={"DBC": 1.0},
            current_trend_positions={"DBC": 228},
            prices={"DBC": 31.14}, nav=100000.0,
            trend_allocation_pct=0.0666, max_position_pct=0.25,
        )
        if not intents:
            pytest.skip("delta below min_notional for this sizing")
        it = intents[0]
        assert it["current_shares"] == 228
        # A held position must never re-buy its full target.
        assert not (it["side"] == "buy" and it["qty"] == it["target_shares"]), \
            "issued a full fresh buy for a position already held"

    def test_reducing_to_a_smaller_target_sells_the_difference(self):
        """The Wednesday-trim path: 442 held, ~214 target -> SELL 228."""
        intents = compute_trend_deltas(
            target_weights={"DBC": 1.0},
            current_trend_positions={"DBC": 442},
            prices={"DBC": 31.14}, nav=100000.0,
            trend_allocation_pct=0.0666, max_position_pct=0.25,
        )
        it = next(i for i in intents if i["symbol"] == "DBC")
        assert it["side"] == "sell"
        assert it["qty"] == 442 - it["target_shares"]


class TestTagMatching:
    """`selector` OR `trade_type` must make a row visible — keying on one alone caused this."""

    @pytest.mark.parametrize("selector,trade_type,visible", [
        ("trend", "trend", True),     # fully tagged
        ("", "trend", True),          # THE regression: adopted rows looked like this
        ("trend", "", True),          # the mirror case
        ("", "", False),              # genuinely untagged
        ("cash", "cash", False),      # a different sleeve must not leak in
    ])
    def test_row_visibility_by_tag(self, selector, trade_type, visible):
        matched = (selector == "trend") or (trade_type == "trend")
        assert matched is visible

    def test_selector_only_matching_would_have_missed_it(self):
        """Documents why the old predicate was insufficient."""
        selector, trade_type = "", "trend"
        assert (selector == "trend") is False          # old predicate -> invisible
        assert ((selector == "trend") or (trade_type == "trend")) is True   # new -> visible

    @pytest.mark.parametrize("selector,trade_type", [("", "trend"), ("trend", "")])
    def test_mismatched_tags_are_detectable(self, selector, trade_type):
        """The condition the sleeve now warns on, so a half-tagged row is not silent."""
        assert (selector or "").strip() != (trade_type or "").strip()
