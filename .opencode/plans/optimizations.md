# boolmesh Speed Optimization Plan

## Overview
8 optimizations identified across the codebase, ranked by estimated performance impact.
All are safe, correctness-preserving changes.

---

## Optimization 1: Enable BVH tree traversal in `compute_query_flat_tree`
**File:** `src/triangulation/flat_tree.rs:35-50`
**Impact:** Very high — O(n²) → O(n log n) for ear clipping cost checks
**Status:** The BVH tree is built by `compute_flat_tree` but `compute_query_flat_tree` ignores it.

### Current code (line 35-50):
```rust
pub fn compute_query_flat_tree<F, T>(pts: &[Pt<T>], rect: &Rect<T>, mut func: F)
where
    T: BoolReal,
    F: FnMut(&Pt<T>),
{
    for p in pts.iter() {        // O(n) linear scan — tree is unused!
        if rect.contains(&p.pos) {
            func(p);
        }
    }
    //query_two_d_tree(pts, rect.clone(), func);  // ← already written, just uncomment
}
```

### Change:
Uncomment the `query_two_d_tree` call. The function is fully implemented at `flat_tree.rs:53-125` and uses the interleaved BVH layout to skip subtrees that don't overlap the query rectangle.

**Risk:** Low. The tree is already built; this just activates the existing traversal.

---

## Optimization 2: `HashMap::entry()` in `Manifold::new` vertex welding
**File:** `src/manifold/mod.rs:61-72`
**Impact:** High — 2x fewer hash lookups during mesh construction
**Current pattern:** Two lookups per vertex (get then insert).

### Change:
Replace the `get` + `insert` pattern with `hash.entry(k)`:
```rust
match hash.entry(k) {
    std::collections::hash_map::Entry::Occupied(e) => rmap[i] = *e.get(),
    std::collections::hash_map::Entry::Vacant(e) => {
        let n = weld.len();
        weld.push(v);
        rmap[i] = n;
        e.insert(n);
    }
}
```

**Risk:** Low. Purely mechanical refactor, same semantics.

---

## Optimization 3: Hash map for edge grouping in `edge_topology`
**File:** `src/manifold/hmesh.rs:41-99`
**Impact:** High — O(E log E) → O(E) average for edge grouping
**Current:** Sorts all edge tuples with `[v1, v2, tri_id, edge_idx]` to find shared edges.

### Change:
Replace the sort-based approach with `FxHashMap<(u32, u32), Vec<(usize, usize)>>`:
```rust
let mut edge_map = FxHashMap::default();
for (tri_id, edge_idx) in ... {
    let key = ((v1 as u32) << 32 | v2 as u32);  // canonical v1<v2
    edge_map.entry(key).or_default().push((tri_id, edge_idx));
}
```

**Risk:** Medium. The sort approach has predictable behavior for duplicate geometry
(edges with 4+ entries). The hash map must handle the same case. Need to verify the
duplicate geometry handling path at `hmesh.rs:84-97` works equivalently.

---

## Optimization 4: Remove unnecessary `sort` in `compute_coplanar_idx`
**File:** `src/manifold/mod.rs:347-406`
**Impact:** Medium — saves O(T log T) per manifold (T = triangle count)
**Current:** Sorts triangles by area before flood-filling coplanar groups.

### Rationale:
The area-based priority determines which triangle is the "root" of a coplanar group
but doesn't affect which triangles belong to each group. The flood-fill is correct
regardless of traversal order. The sort only adds ~5% overhead for the benefit of
determining root order (which has no downstream effect).

### Change:
Remove the `priority.sort_by(...)` line and iterate `(area, t)` tuples in any order.
The flood fill at `interior.pop()` will still find all coplanar triangles.

**Risk:** Low. The coplanar grouping result is semantically identical. Only the
arbitrary "root" ID per group changes, which has no downstream effect since coplanar
triangles share the same normal lookup.

---

## Optimization 5: Pre-allocate HashMaps in `boolean45`
**File:** `src/boolean45/mod.rs:512-534`
**Impact:** Medium — reduces allocation churn in tight loops
**Current:** `FxHashMap::default()` starts with minimal capacity, causing reallocations.

### Change:
```rust
let mut pt_p = FxHashMap::with_capacity_and_hasher(p1q2.len() * 2, Default::default());
let mut pt_new = FxHashMap::with_capacity_and_hasher(p1q2.len() * 2, Default::default());
```

**Risk:** Low. Purely capacity hint, no behavioral change.

---

## Optimization 6: Pass `Tref` by value in `is_coplanar`
**File:** `src/simplification/collapse.rs:205`
**Impact:** Low — micro-optimization, eliminates reference indirection
**Current:** `fn is_coplanar(t0: &Tref, t1: &Tref)` takes references.

### Change:
```rust
#[inline]
fn is_coplanar(t0: Tref, t1: Tref) -> bool {
    t0.mid == t1.mid && t0.pid == t1.pid
}
```
Update call sites in `collapse.rs:26,30` to pass by value.

**Risk:** Very low. Tref is Copy (3 u32/u32/i32 fields = 12 bytes).

---

## Optimization 7: Fix `size_output` to avoid `Vec::concat`
**File:** `src/boolean45/mod.rs:98`
**Impact:** Low — eliminates unnecessary allocation
**Current:**
```rust
let side_pq = [&side_p[..], &side_q[..]].concat();
```
This allocates a new Vec by iterating both slices.

### Change:
```rust
let mut side_pq = Vec::with_capacity(side_p.len() + side_q.len());
side_pq.extend_from_slice(&side_p);
side_pq.extend_from_slice(&side_q);
```

**Risk:** Low. Same semantics, just avoids intermediate allocation.

---

## Optimization 8: Single-pass `collapse_short_edges`
**File:** `src/simplification/collapse.rs:177-202`
**Impact:** Low-medium — avoids rescanning unchanged halfedges
**Current:** Full scan of all halfedges each iteration.

### Change:
Track only affected halfedges in a work queue instead of rescanning. When a collapse
changes the topology, only re-check neighboring halfedges rather than the entire array.

**Risk:** Medium. The loop-based approach is simpler and already bounded. A work-queue
approach requires tracking adjacency correctly.

---

## Implementation Order

1. **Opt 4** — Remove sort in `compute_coplanar_idx` (simplest, safest)
2. **Opt 2** — `HashMap::entry()` in `Manifold::new` (simple, well-tested pattern)
3. **Opt 7** — Fix `size_output` Vec::concat (trivial)
4. **Opt 6** — Pass Tref by value (trivial)
5. **Opt 5** — Pre-allocate HashMaps (trivial)
6. **Opt 1** — Enable BVH tree traversal (biggest win, but needs testing)
7. **Opt 3** — Hash map in `edge_topology` (needs careful testing for edge cases)
8. **Opt 8** — Single-pass collapse (deferred, incremental improvement)

## Testing Strategy
- `cargo test` — run all existing tests first
- After each optimization: `cargo test` to verify correctness
- After all optimizations: `cargo test --all-features`
- Manual perf test: menger_sponge example to check timing improvement

## Expected Total Speedup
- **Opt 1 alone:** ~2-10x faster triangulation (dominant bottleneck)
- **All optimizations combined:** ~1.5-3x end-to-end improvement
- The BVH activation (Opt 1) is the single biggest win by far.
