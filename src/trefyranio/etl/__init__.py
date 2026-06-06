"""ETL for the trefyranio data spine: polls, results, and demographics.

Principle: **ETL preserves, the model transforms.** Ingesters here normalize
*structure* (tidy long format, canonical party/pollster names, parsed dates)
but never alter the *numbers* a pollster reported. Decisions like dropping the
undecided bucket or renormalizing onto the decided-voter simplex belong to the
modeling layer (Phase 3), so the raw shares stay auditable.
"""
