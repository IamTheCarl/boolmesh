# boolmesh — Agent Instructions

## Commands
```
cargo build                             # dev build
cargo test                              # run tests (lib only)
cargo test --all-features               # full test suite (includes serde feature)
cargo run --release --example menger_sponge --features=rayon   # perf example
cargo run --release --example <name> --features=bevy,rayon,f32  # bevy examples
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
| `src/common.rs` | Core types: `BoolReal`, `EdgeId`, `HalfEdge`, `Tref`, CCW helpers |
| `src/manifold/` | `Manifold` struct, `Hmesh` builder, collider, bounds |
| `src/boolean03/` | Kernel 03: intersection detection, winding numbers |
| `src/boolean45/` | Kernel 45: intermediate halfedge mesh construction |
| `src/triangulation/` | Ear clipping, flat-tree BVH, halfedge assembly |
| `src/simplification/` | Topology cleanup: dedup, collapse, edge swap |
| `src/compose/` | Primitive generators (cube, sphere, cylinder, etc.) |

## EdgeId — critical pattern
`HalfEdge.tail`, `.head`, `.pair` are `EdgeId(pub usize)`, **not** `usize`.

- **Read**: `he.tail.0`, `he.head.0`, `he.pair.0`
- **Write**: `EdgeId(value)` or `EdgeId::INVALID`
- **Construct**: `HalfEdge::new(tail, head, pair)` takes `usize` and wraps internally
- **Invalid sentinel**: `EdgeId::INVALID` == `EdgeId(usize::MAX)`
- `half()` methods return `Option<usize>` (None if INVALID)
- `From<EdgeId> for usize` and `From<usize> for EdgeId` are implemented
- **Serde**: `EdgeId` has `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` — always include `serde` feature when serializing

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
- `REMOVE_FLAG` in `tri_halfs.rs` is `usize::MAX - 1`
