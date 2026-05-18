# boolmesh — Agent Instructions

## Commands
```
cargo build                             # dev build
cargo test                              # run tests (lib only) — 8 tests pass
cargo test --all-features               # full suite — integration tests in test_examples.rs fail (21 pass, 15 fail)
cargo run --bin generate_fixtures --features serde  # regenerate test fixtures
cargo build --example <name> --features bevy   # build a bevy example
cargo run --release --example <name> --features bevy  # run a bevy example
```

Note: The `--all-features` integration tests fail due to non-deterministic ordering in `geo::unary_union` results. This is a known issue — the fixture comparisons fail because polygon vertex order varies between runs.

## Architecture (one-pass boolean)
```
compute_boolean()
  → boolean03()         — winding number test (kernels 01/02/03)
  → boolean45()         — construct intermediate halfedge mesh (Boolean45)
  → triangulate()       — ear-clipping or direct triangulation
  → simplify_topology() — dedup → collapse short/collinear → swap degenerates
  → cleanup_unused_verts_impl()
  → Manifold::new_from_raw()
```

### Key modules
| Module | Purpose |
|---|---|
| `src/common.rs` | `BoolReal` trait, `VertexId`, `HalfEdgeId`, `HalfEdge`, `Tref`, CCW helpers |
| `src/manifold/mod.rs` | `Manifold` struct, `Triangle`, `Manifold::cleanup()`, `project_xy()`, `slice()` |
| `src/manifold/hmesh.rs` | `Hmesh` builder (internal, `pub(in crate::manifold)`) |
| `src/boolean03/` | Kernel 03: intersection detection, winding numbers |
| `src/boolean45/mod.rs` | `Boolean45` intermediate struct (pub fields), `boolean45()` |
| `src/triangulation/` | Ear clipping, flat-tree BVH, halfedge assembly |
| `src/simplification/` | Topology cleanup: dedup, collapse, edge swap |
| `src/compose/` | Primitive generators: cube, sphere, cylinder, cone, torus, extrusion |

## Manifold — fields and accessors

All fields are `pub(crate)`. External access only via methods:

| Method | Returns |
|---|---|
| `manifold.positions()` | `&[Vector3<T>]` |
| `manifold.halfedges()` | `&[HalfEdge]` |
| `manifold.triangles()` | `impl Iterator<Item = Triangle<T>>` — best for iterating triangles |
| `manifold.vertex_count()` / `face_count()` / `halfedge_count()` | `usize` |
| `manifold.pos_at(i)` | `Vector3<T>` |
| `manifold.face_vertex_ids(i)` | `[usize; 3]` |
| `manifold.face_positions(i)` | `[Vector3<T>; 3]` |
| `manifold.halfedge_at(i)` | `&HalfEdge` |
| `manifold.cleanup(&mut self)` | removes unused verts, updates counts |
| `manifold.project_xy()` | `Result<MultiPolygon<T>, ProjectionError>` |
| `manifold.slice(height)` | `Result<MultiPolygon<T>, SliceError>` |

### Manifold fields (pub(crate))
`positions`, `halfedges`, `vertex_count`, `face_count`, `halfedge_count`, `spatial_tol`, `snap_tol`, `bounding_box`, `face_normals`, `vert_normals`, `original_idx`, `collider`, `coplanar`

## Boolean45 — internal struct
`Boolean45` (in `src/boolean45/mod.rs`) has **pub** fields (`ps`, `ns`, `hs`, `rs`, `hid_per_f`, `nv_from_p`, `nv_from_q`). It is internal — don't change its fields to pub(crate) without checking all call sites first.

## Deprecated — use these instead
| Deprecated | Use |
|---|---|
| `cleanup_unused_verts()` | `Manifold::cleanup()` |
| `compute_projection()` | `Manifold::project_xy()` |
| `compute_slice()` | `Manifold::slice(height)` |

`Manifold::new_impl()` does not exist. `Manifold::new()` internally calls `Manifold::new_from_raw()` after dedup vertices and removes collapsed triangles.

## VertexId / HalfEdgeId — critical pattern
`HalfEdge.tail`, `.head` are `VertexId`, `.pair` is `HalfEdgeId`. **Not raw `usize`**.

- **Read raw**: `he.tail.0`, `he.head.0`, `he.pair.0` (private u32 field, accessible in same crate)
- **Construct**: `HalfEdge::new(tail, head, pair)` takes `usize` and wraps internally
- **Invalid sentinel**: `VertexId::invalid()`, `HalfEdgeId::invalid()`
- **Removed sentinel**: `HalfEdgeId::invalid_removed()` (used during simplification)
- **Conversions**: `From<u32>`/`From<usize>`/`Into<u32>`/`Into<usize>` on both types. `From<usize>` panics on truncation.
- **Serde**: `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` — always enable `serde` feature

### HalfEdgeId layout helpers
`HalfEdgeId` packs `face_id * 3 + edge_index` into a single u32.
- `hid.face_id()`, `hid.edge_index()`, `hid.next_in_face()`, `hid.prev_in_face()`

## Iterating triangles (examples)
Use `manifold.triangles()` — it returns `Triangle<T>` with `positions: [Vector3<T>; 3]` and `normal: Vector3<T>`.

```rust
for tri in manifold.triangles() {
    let [p0, p1, p2] = tri.positions;
    let n = tri.normal;
    // use p0, p1, p2, n
}
```

Do **not** manually index `positions` and `face_normals` by face_id.

## BoolReal — precision constants
`BoolReal` gates f32 vs f64. `K_PRECISION` differs per impl:
| T | K_PRECISION | Use case |
|---|---|---|
| f64 | 1e-12 | Default, production geometry |
| f32 | 1e-4 | Lower-precision / performance-sensitive |

The tolerance used in `Manifold::new_from_raw` is `T::K_PRECISION * bounding_box.scale()`. Changing the precision without adjusting tolerance expectations will cause false intersections or missed collisions.

## Constraints
- Input meshes **must be manifold** (no boundaries, no overlapping geometry)
- `Hmesh` uses raw `usize` — conversion at `Manifold::new_from_raw` boundary
- `HalfEdgeId::INVALID_REMOVED` is `u32::MAX - 1`

## Hmesh edge topology — duplicate geometry gotcha
When composing multiple meshes at the same position, `Manifold::new()` welds vertices but triangles are **not** deduplicated. `edge_topology()` sees edges with 4+ entries in `ett` (not the expected 2). The algorithm:
- Counts edge groups by iterating sorted `ett` and detecting consecutive entries with identical `[v1, v2]` pairs
- Skip **all** consecutive duplicates for a shared edge, not just one
- Write `f2e` for **every** entry in a shared-edge group
- `nh = nf * 3` (not `ne * 2`) because duplicate geometry produces >2 halfedges per logical edge

## Feature flags
| Feature | Effect |
|---|---|
| `rayon` | Multi-threading support |
| `serde` | Serialize/deserialize support |
| `bevy` | Bevy rendering examples (requires bevy examples) |
| `verbose` | Verbose output |

## Gotchas
- `compose::compose()` accesses Manifold fields directly via `pub(crate)` — this is fine, but don't make those fields private without updating it
- `Boolean45` fields (`ps`, `hs`, etc.) are still pub and used by `triangulate()` and `simplify_topology()` — don't rename them without checking all callers
- `Manifold::project_xy()` and `Manifold::slice()` require the type to implement `BoolOpsNum` from geo — the generic bound is on the method, not on `Manifold<T>` itself
- `geo-bevy` has a known recursion overflow in some Rust versions — if `cargo build --features bevy` fails with recursion errors, that's a dependency issue, not a code issue
