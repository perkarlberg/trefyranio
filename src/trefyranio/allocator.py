"""Swedish Riksdag seat allocation.

Implements the deterministic seat-allocation core of the Swedish electoral
system, which is the heart of the trefyranio simulator: every Monte Carlo draw
of party vote shares is turned into a seat distribution by this module.

The system (since the 2018 election):

* 349 seats total = 310 fixed constituency seats + 39 leveling/adjustment seats
  (utjamningsmandat). The leveling seats make the final allocation nationally
  proportional among parties that clear the threshold, so the *national* result
  can be reproduced by running the allocation method on national vote totals.
* Threshold: a party needs >= 4% of valid votes nationally, OR >= 12% in a
  single constituency, to take part in seat allocation.
* Method: modified Sainte-Lague (jamkade uddatalsmetoden) with a first divisor
  of 1.2 (lowered from 1.4 starting the 2018 election), then 3, 5, 7, 9, ...

References:
  https://sv.wikipedia.org/wiki/J%C3%A4mkade_uddatalsmetoden
  https://en.wikipedia.org/wiki/Elections_in_Sweden
"""

from __future__ import annotations

from dataclasses import dataclass

TOTAL_SEATS = 349
FIRST_DIVISOR = 1.2  # since the 2018 election
FIRST_DIVISOR_PRE_2018 = 1.4  # 1970-2014
FIRST_DIVISOR_CHANGE_YEAR = 2018
NATIONAL_THRESHOLD = 0.04  # 4% of valid votes nationally
CONSTITUENCY_THRESHOLD = 0.12  # 12% in a single constituency


def first_divisor_for_year(year: int) -> float:
    """The modified Sainte-Lague first divisor in force for an election year:
    1.4 through 2014, lowered to 1.2 from the 2018 election onward."""
    return FIRST_DIVISOR if year >= FIRST_DIVISOR_CHANGE_YEAR else FIRST_DIVISOR_PRE_2018


def _divisor(seats_held: int, first_divisor: float = FIRST_DIVISOR) -> float:
    """Modified Sainte-Lague divisor for a party that already holds
    ``seats_held`` seats and is competing for the next one."""
    if seats_held == 0:
        return first_divisor
    return 2 * seats_held + 1


def qualified_parties(
    national_votes: dict[str, int],
    constituency_shares: dict[str, dict[str, float]] | None = None,
) -> set[str]:
    """Return the set of parties eligible for seats.

    A party qualifies if it reaches the national 4% threshold, or (if
    per-constituency shares are supplied) 12% in any single constituency.
    """
    total = sum(national_votes.values())
    if total == 0:
        return set()
    qualified = {
        party
        for party, votes in national_votes.items()
        if votes / total >= NATIONAL_THRESHOLD
    }
    if constituency_shares:
        for party in national_votes:
            if any(
                shares.get(party, 0.0) >= CONSTITUENCY_THRESHOLD
                for shares in constituency_shares.values()
            ):
                qualified.add(party)
    return qualified


def allocate_seats(
    votes: dict[str, int],
    n_seats: int = TOTAL_SEATS,
    eligible: set[str] | None = None,
    first_divisor: float = FIRST_DIVISOR,
) -> dict[str, int]:
    """Allocate ``n_seats`` among parties by modified Sainte-Lague.

    ``votes`` maps party -> vote count. If ``eligible`` is given, only those
    parties compete (use :func:`qualified_parties` to apply the threshold);
    otherwise every party with votes competes. ``first_divisor`` is the divisor
    for a party's first seat (1.2 since 2018, 1.4 before — see
    :func:`first_divisor_for_year`).

    Seats are assigned one at a time to the party with the highest current
    quotient ``votes / divisor(seats_held)``.
    """
    if eligible is None:
        eligible = set(votes)
    contenders = {p: votes[p] for p in eligible if votes.get(p, 0) > 0}
    seats = {p: 0 for p in contenders}

    for _ in range(n_seats):
        # Pick the party with the highest quotient. Ties broken by vote count,
        # then party name, for determinism.
        winner = max(
            contenders,
            key=lambda p: (
                contenders[p] / _divisor(seats[p], first_divisor),
                contenders[p],
                p,
            ),
        )
        seats[winner] += 1
    return seats


@dataclass
class NationalResult:
    """National seat allocation plus the threshold decision per party."""

    seats: dict[str, int]
    qualified: set[str]
    vote_share: dict[str, float]


def allocate_national(
    national_votes: dict[str, int],
    constituency_shares: dict[str, dict[str, float]] | None = None,
    n_seats: int = TOTAL_SEATS,
    first_divisor: float = FIRST_DIVISOR,
    ignore_parties: frozenset[str] = frozenset(),
) -> NationalResult:
    """End-to-end national allocation: apply the threshold, then distribute
    all ``n_seats`` proportionally among qualifying parties.

    This reproduces the *final* national seat totals because Sweden's leveling
    seats render the outcome nationally proportional above the threshold.
    ``first_divisor`` defaults to the current 1.2; pass
    ``first_divisor_for_year(year)`` to reproduce pre-2018 elections.

    ``ignore_parties`` (e.g. the "other" aggregate bucket) still count toward
    the valid-vote total that the 4% threshold is measured against, but never
    receive seats — a lumped "Övriga" total can exceed 4% without any single
    party qualifying, so it must be excluded from allocation.

    Per-constituency allocation of the 310 fixed seats is a separate step
    (added once valkrets-level data is wired in); pure national-proportional
    reproduces most but not all elections exactly.
    """
    total = sum(national_votes.values())
    eligible = qualified_parties(national_votes, constituency_shares) - ignore_parties
    seats = allocate_seats(
        national_votes, n_seats=n_seats, eligible=eligible, first_divisor=first_divisor
    )
    # Parties that didn't qualify still appear with 0 seats for completeness.
    for party in national_votes:
        seats.setdefault(party, 0)
    shares = {p: (v / total if total else 0.0) for p, v in national_votes.items()}
    return NationalResult(seats=seats, qualified=eligible, vote_share=shares)
