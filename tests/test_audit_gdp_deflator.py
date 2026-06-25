"""Audit cleanup: the GDP PRICE INDEX / DEFLATOR must not be misclassified as growth.

It is an INFLATION measure (lower-is-risk-on), but "GDP Price Index" substring-matched the "gdp"
growth keyword and was labelled GDP (higher_better) — so a HOT deflator showed risk-ON (wrong
direction). It is now classed neutral OTHER_HIGH (no wrong risk-on/off claim).
"""
from __future__ import annotations

from app.news.sources.finnhub_source import _classify_event
from app.news.macro_polarity import classify_outcome


def test_gdp_deflator_namings_are_not_growth():
    for name in ("GDP Price Index", "GDP Price Index QoQ", "GDP Deflator",
                 "Gross Domestic Product Price Index",
                 "GDP Implicit Price Deflator QoQ Final"):
        assert _classify_event(name) == "OTHER_HIGH", name


def test_real_gdp_growth_still_classifies_as_gdp():
    for name in ("GDP Growth Rate QoQ", "GDP QoQ Adv", "Gross Domestic Product Annualized",
                 "Real GDP"):
        assert _classify_event(name) == "GDP", name


def test_deflator_polarity_is_neutral_not_risk_on():
    # the whole point: a HOT deflator (above estimate) must NOT read risk_on. OTHER_HIGH is neutral
    # polarity, so classify_outcome makes no risk-on/off claim (vs GDP higher_better which would).
    et = _classify_event("GDP Price Index QoQ")              # -> OTHER_HIGH
    out = classify_outcome(et, actual=3.1, estimate=2.5)     # hotter-than-expected
    assert out["polarity"] == "neutral"
    assert out["market_outcome"] != "risk_on"               # would have been risk_on under GDP


def test_other_high_impact_events_unaffected():
    assert _classify_event("Consumer Price Index (CPI) YoY") == "CPI"
    assert _classify_event("Nonfarm Payrolls") == "NFP"
    assert _classify_event("Initial Jobless Claims") == "JOBLESS_CLAIMS"
    assert _classify_event("Something Irrelevant") is None
