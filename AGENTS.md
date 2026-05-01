# boolmesh — Agent Instructions

## Commands
```
cargo build                             # dev build
cargo test                              # run tests (lib only)
cargo test --all-features               # full test suite (includes serde feature)
cargo run --release --example menger_sponge --features=bevy,rayon,f32   # perf example
cargo run --release --example <name> --features=bevy,rayon,f32          # bevy examples
```

## Architecture (one-pass boolean)
```
compute_boolean()
  → boolean03()   — winding number test (kernels 01/02/03)
  → boolean45()   — construct intermediate halfedge mesh
  → triangulate() — ear-clipping or direct triangulation
  → simplify_topology() — dedup → collapse short/collinear → swap degenerates
  → cleanup_unused_verts()
  → Manifold::new_impl()
```

### Key modules
| Module | Purpose |
|---|---|
| `src/common.rs` | Core types: `BoolReal`, `VertexId`, `HalfEdgeId`, `HalfEdge`, `Tref`, CCW helpers |
| `src/manifold/` | `Manifold` struct, `Hmesh` builder, collider, bounds |
| `src/boolean03/` | Kernel 03: intersection detection, winding numbers |
| `src/boolean45/` | Kernel 45: intermediate halfedge mesh construction |
| `src/triangulation/` | Ear clipping, flat-tree BVH, halfedge assembly |
| `src/simplification/` | Topology cleanup: dedup, collapse, edge swap |
| `src/compose/` | Primitive generators (cube, sphere, cylinder, etc.) |

## VertexId / HalfEdgeId — critical pattern
`HalfEdge.tail`, `.head` are `VertexId`, `.pair` is `HalfEdgeId`. **Not raw `usize`**.

- **Read raw**: `he.tail.0`, `he.head.0`, `he.pair.0` (private u32 field, accessible in same crate)
- **Construct**: `HalfEdge::new(tail, head, pair)` takes `usize` and wraps internally
- **Invalid sentinel**: `VertexId::invalid()`, `HalfEdgeId::invalid()`
- **Removed sentinel**: `HalfEdgeId::invalid_removed()` (used during simplification)
- **Methods**: `VertexId::is_invalid()`, `HalfEdgeId::is_invalid()`, `is_removed()`
- **Conversions**: `From<u32>`/`From<usize>`/`Into<u32>`/`Into<usize>` on both types. `From<usize>` panics on truncation.
- **Serde**: `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` — always enable `serde` feature

### HalfEdgeId layout helpers
`HalfEdgeId` packs `face_id * 3 + edge_index` into a single u32. Use:
- `hid.face_id()` → `usize` (which face this halfedge belongs to)
- `hid.edge_index()` → `usize` (0, 1, or 2 within the face)
- `hid.next_in_face()` → `usize` (next halfedge in same face)
- `hid.prev_in_face()` → `usize` (previous halfedge in same face)

### Manifold ergonomic accessors
- `manifold.pos_at(vertex_id)` — get position of a vertex
- `manifold.face_vertex_ids(face_id)` — get 3 vertex IDs of a face
- `manifold.face_positions(face_id)` — get 3 positions of a face
- `manifold.halfedge_at(edge_id)` — get a halfedge by ID

## Simplification helpers (private, in `src/simplification/mod.rs`)
- `head_of(hs, i)` → `usize`
- `tail_of(hs, i)` → `usize`
- `pair_of(hs, i)` → `usize`
- `pair_up(hs, i, j)` — mutual twin assignment
- `hids_of(i)` → `(usize, usize, usize)` — three halfedges of a triangle

## Constraints
- Input meshes **must be manifold** (no boundaries, no overlapping geometry)
- `BoolReal` trait gates `f32` vs `f64` precision — use `f32` feature for single-precision
- `Hmesh` (internal, `pub(in crate::manifold)`) uses raw `usize` — conversion at `Manifold::new_impl` boundary
- `HalfEdgeId::INVALID_REMOVED` is `u32::MAX - 1`

## Hmesh edge topology — duplicate geometry gotcha
When composing multiple meshes at the same position, `Manifold::new()` welds vertices but triangles are not deduplicated. This means `edge_topology()` sees edges with 4+ entries in `ett` (not the expected 2). The algorithm:
- Counts edge groups by iterating sorted `ett` and detecting consecutive entries with identical `[v1, v2]` pairs
- In the while loop, skip **all** consecutive duplicates for a shared edge, not just one
- Write `f2e` for **every** entry in a shared-edge group (not just first/last)
- `nh = nf * 3` (not `ne * 2`) because duplicate geometry can produce more than 2 halfedges per logical edge

## Feature flags
| Feature | Effect |
|---|---|
| `rayon` | Multi-threading support |
| `serde` | Serialize/deserialize support |
| `bevy` | Bevy rendering examples (requires bevy examples) |
| `f32` | Use `f32` as `BoolReal` (must be passed via `--features=f32`) |
