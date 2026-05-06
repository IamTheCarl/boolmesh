# Geo Boolean Op Non-Determinism Issue

## Problem (Fixed)

The `geo::boolean_op()` function was producing non-deterministic results across processes, causing test failures in `test_extrude_twisted` and `test_revolve_half`.

## Root Cause

The `i_overlay` integer-space overlay algorithm (used by `geo::boolean_op`) had a `multithreading` feature enabled by default. This feature used `par_sort_unstable_by` from `rayon`, which is non-deterministic because:

1. **Unstable sort**: Elements that compare equal can end up in any order
2. **Parallel execution**: Thread scheduling varies across processes, leading to different merge orders

## The Fix

Two complementary fixes were applied:

### 1. Disable multithreading in geo (Cargo.toml)

```toml
# Before
geo = { version = "0.32", features = ["serde"] }

# After
geo = { version = "0.32", default-features = false, features = ["serde", "earcutr", "spade"] }
```

This disables `i_overlay/allow_multithreading`, forcing single-threaded deterministic sorting.

### 2. Canonicalize ring direction (tests/test_examples.rs & src/bin/generate_fixtures.rs)

Added `force_ccw_ring()` and `force_cw_ring()` functions that:
- Rotate each ring to start at the lexicographically smallest coordinate
- Force exterior rings to CCW direction
- Force interior rings to CW direction

This ensures consistent vertex ordering regardless of internal crate ordering.

## Result

All 36 tests now pass consistently across processes.
