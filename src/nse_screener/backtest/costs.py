"""What a delivery trade on NSE actually costs.

Costs are not a detail here. The Stage 7 study measured an average forward
return of 0.4-0.8% over ten sessions; a round trip below costs roughly 0.25-0.5%
of turnover. The gap between "this strategy is flat" and "this strategy loses
money steadily" is entirely inside this file, so it is modelled properly rather
than as a single hand-waved percentage.

Rates are the statutory ones for **delivery** equity trades, which is what a
daily-bar swing strategy does. Intraday and F&O are charged differently and
these numbers do not apply to them.

Defaults assume a discount broker (zero brokerage on delivery). Set
`brokerage_per_order` if yours charges.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Costs:
    """Statutory and broker charges, as fractions of turnover unless noted."""

    brokerage_per_order: float = 0.0
    """Rupees per order. Zero at most discount brokers for delivery; Rs 20 flat
    at some. Not a percentage."""

    stt_buy: float = 0.001
    stt_sell: float = 0.001
    """Securities Transaction Tax, 0.1% on both legs for delivery."""

    exchange_txn: float = 0.0000297
    """NSE transaction charge, 0.00297% per leg."""

    sebi_charges: float = 0.000001
    """SEBI turnover fee, Rs 10 per crore."""

    stamp_duty_buy: float = 0.00015
    """0.015% on the buy leg only."""

    gst: float = 0.18
    """18%, charged on brokerage + exchange + SEBI charges -- not on STT or
    stamp duty."""

    slippage: float = 0.0005
    """Per leg. The gap between the price you saw and the price you got. A
    modelling assumption, not a statutory rate -- and on a mid-cap at the open
    it is easily larger than this."""

    def entry_cost(self, value: float | pd.Series) -> float | pd.Series:
        """Total charges on a buy of `value` rupees, excluding slippage."""
        statutory = value * (self.stt_buy + self.stamp_duty_buy)
        taxable = self.brokerage_per_order + value * (
            self.exchange_txn + self.sebi_charges
        )
        return statutory + taxable * (1 + self.gst)

    def exit_cost(self, value: float | pd.Series) -> float | pd.Series:
        """Total charges on a sell of `value` rupees, excluding slippage."""
        statutory = value * self.stt_sell
        taxable = self.brokerage_per_order + value * (
            self.exchange_txn + self.sebi_charges
        )
        return statutory + taxable * (1 + self.gst)

    def round_trip_pct(self, value: float = 100_000) -> float:
        """Total cost of a round trip as a percentage of turnover.

        Handy for a sanity check: on a Rs 1 lakh position the default settings
        give about 0.32%, which is inside the 0.25-0.5% the plan assumes.
        """
        both = self.entry_cost(value) + self.exit_cost(value)
        slip = 2 * value * self.slippage
        return (both + slip) / value * 100

    def buy_price(self, price: float | pd.Series) -> float | pd.Series:
        """You pay slightly more than you saw."""
        return price * (1 + self.slippage)

    def sell_price(self, price: float | pd.Series) -> float | pd.Series:
        """And receive slightly less."""
        return price * (1 - self.slippage)
