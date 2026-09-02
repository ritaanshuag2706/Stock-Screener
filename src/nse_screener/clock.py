"""Market time.

A trading date is always an IST date. `date.today()` reads the machine's local
timezone, so on any clock outside IST it is a different calendar day from the
exchange for part of every day. Anything that asks "what is today?" in this
project must go through here.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def ist_today() -> date:
    """Today's date in Indian market time."""
    return datetime.now(IST).date()


def ist_now() -> datetime:
    """Current IST timestamp, timezone-aware."""
    return datetime.now(IST)
