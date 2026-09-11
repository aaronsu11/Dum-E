"""Numerical producer API backed by the canonical dependency-layer instrument.

Validation and arithmetic are shared with parity_gate; this module never owns a
second acceptance implementation. Fabricated arrays require explicit test evidence.
"""

from policy_guard.parity_gate import (  # noqa: F401
    array_comparison, chunk_index_slopes, compare_tiers, joint_deviations,
    metrics, signed_bias, validate_coverage,
)
