"""trefyranio — a FiveThirtyEight-style forecasting model for the Swedish Riksdag.

trefyran.io reads as "tre fyra n-io" = 3-4-9 = the 349 seats of the Riksdag.
"""

from trefyranio.allocator import (
    NationalResult,
    allocate_national,
    allocate_seats,
    first_divisor_for_year,
    qualified_parties,
)

__all__ = [
    "NationalResult",
    "allocate_national",
    "allocate_seats",
    "first_divisor_for_year",
    "qualified_parties",
]
