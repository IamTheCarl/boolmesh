# Plan: Unit Tests Derived from Examples with Serde Fixtures

## Goal
Create integration tests that derive from the project's examples. Tests create the same manifolds/projections/slices and compare against known-good results saved/loaded via serde.

## Design Decisions
- **Fixture format**: `bincode` — binary serde format that natively supports NaN, preserves exact float bits
- **Fixture generation**: Standalone binary (`cargo run --bin generate_fixtures`) that outputs `.bincode` files to `tests/fixtures/`
- **Test format**: Integration tests (`tests/test_examples.rs`) that load fixtures, create manifolds the same way the examples do, and compare
- **Scope**: Tests cover all non-bevy functionality from examples (primitives, boolean ops, extrusions, revolutions, projections, slices)
- **Precision**: Tests use f64. Fixture comparison is exact byte-for-byte via bincode.

## Files to Create/Modify

### 1. `src/manifold/mod.rs` — Add serde derives
Add `#[derive(serde::Serialize, serde::Deserialize)]` (cfg-gated on `serde` feature) to the `Manifold<T>` struct. All its fields already have the necessary serde support.

Also add `Manifold::approx_eq()` method for geometric comparison with epsilon tolerance.

### 2. `tests/fixtures/` — New directory
Generated bincode fixture files, one per test case. Each file contains a serialized `Manifold<f64>` or `geo::MultiPolygon<f64>`.

### 3. `src/bin/generate_fixtures.rs` — Fixture generator binary
Standalone binary that:
- Creates `tests/fixtures/` directory
- Generates fixtures for every test case
- Serializes via `bincode::serialize()` → writes `.bincode` files
- Prints summary of generated fixtures

### 4. `tests/test_examples.rs` — Integration test module
Loads fixtures from `tests/fixtures/`, creates manifolds the same way examples do, and compares.

### 5. `Cargo.toml` — Updates
- Add `bincode` as regular dependency (needed by both binary and tests)
- Add `[[bin]] generate_fixtures` target
- Ensure `serde` feature includes bincode

## Test Cases (derived from examples)

### Primitives (from `primitives_demo.rs`)
| Name | Operation | Fixture Type |
|------|-----------|-------------|
| `cube` | `generate_cube()` | Manifold |
| `cylinder` | `generate_cylinder(1., 1., 30, 10)` | Manifold |
| `uv_sphere` | `generate_uv_sphere(30, 30)` | Manifold |
| `icosphere` | `generate_icosphere(1)` | Manifold |
| `torus` | `generate_torus(1., 0.1, 30, 30)` | Manifold |
| `cone` | `generate_cone(apex, center, 1., 30)` | Manifold |

### Transformations (from `projection.rs`, `slicing.rs`)
| Name | Operation | Fixture Type |
|------|-----------|-------------|
| `cube_translated` | `cube.translate(2., 0., 0.)` | Manifold |
| `cube_rotated` | `cube.rotate(PI/4., 0., 0.)` | Manifold |
| `cube_scaled` | `cube.scale(2., 1., 1.)` | Manifold |

### Boolean Operations (from `primitives_demo.rs`, `multiple_models.rs`)
| Name | Operation | Fixture Type |
|------|-----------|-------------|
| `cube_add_cube` | `compute_boolean(cube, cube.translate(2.,0.,0.), Add)` | Manifold |
| `cube_sub_cube` | `compute_boolean(cube, cube.translate(0.1,0.,0.), Subtract)` | Manifold |
| `cube_intersect_cube` | `compute_boolean(cube, cube.translate(0.1,0.,0.), Intersect)` | Manifold |
| `cyl_sphere_add` | `compute_boolean(cylinder, uv_sphere.translate(0.,0.5,0.), Add)` | Manifold |
| `cyl_sphere_sub_torus` | `result.subtract(torus)` | Manifold |

### Extrusions (from `extrusion.rs`)
| Name | Operation | Fixture Type |
|------|-----------|-------------|
| `extrude_square` | `square.extrude(1., 1, 0., (1.,1.))` | Manifold |
| `extrude_hollow_square` | `(square - hole).extrude(1., 1, 0., (1.,1.))` | Manifold |
| `extrude_twisted` | `(square - hole).extrude(1., 20, PI, (1.,1.))` | Manifold |
| `extrude_tapered` | `(square - hole).extrude(1., 50, PI, (0.,0.))` | Manifold |

### Revolutions (from `revolution.rs`)
| Name | Operation | Fixture Type |
|------|-----------|-------------|
| `revolve_square` | `square.revolve(15, PI*2.)` | Manifold |
| `revolve_half` | `square.revolve(5, PI*0.5)` | Manifold |

### Projections (from `projection.rs`)
| Name | Operation | Fixture Type |
|------|-----------|-------------|
| `cube_project` | `cube.project_xy()` | MultiPolygon |
| `cylinder_project` | `cylinder.project_xy()` | MultiPolygon |
| `uv_sphere_project` | `uv_sphere.project_xy()` | MultiPolygon |
| `icosphere_project` | `icosphere.project_xy()` | MultiPolygon |
| `torus_project` | `torus.project_xy()` | MultiPolygon |
| `extrude_square_project` | `extrude_square.project_xy()` | MultiPolygon |
| `extrude_hollow_project` | `extrude_hollow.project_xy()` | MultiPolygon |
| `revolve_square_project` | `revolve_square.project_xy()` | MultiPolygon |

### Slices (from `slicing.rs`)
| Name | Operation | Fixture Type |
|------|-----------|-------------|
| `cube_slice_0` | `cube.slice(0.0)` | MultiPolygon |
| `cube_slice_025` | `cube.slice(0.25)` | MultiPolygon |
| `cube_slice_minus025` | `cube.slice(-0.25)` | MultiPolygon |
| `cube_slice_out_of_bounds` | `cube.slice(2.0)` → Err | (no fixture) |

### Composition (from `compose::compose()`)
| Name | Operation | Fixture Type |
|------|-----------|-------------|
| `compose_two_cubes` | `compose(&[cube, cube.translate(2.,0.,0.)])` | Manifold |

## Test Implementation

Each test follows this pattern:

```rust
#[test]
fn test_<name>() {
    // 1. Load fixture (exact binary comparison via bincode)
    let bytes = include_bytes!(concat!("fixtures/", stringify!(<name>), ".bincode"));
    let expected: Manifold<f64> = bincode::deserialize(bytes).unwrap();

    // 2. Create manifold the same way the example does
    let actual = generate_cube::<f64>().unwrap();

    // 3. Compare via approx_eq (for geometric comparison with epsilon)
    assert!(actual.approx_eq(&expected, 1e-10), "mismatch for <name>");
}
```

For projections/slices (MultiPolygon, can't use approx_eq):
```rust
#[test]
fn test_projection_cube() {
    let bytes = include_bytes!("fixtures/cube_project.bincode");
    let expected: geo::MultiPolygon<f64> = bincode::deserialize(bytes).unwrap();
    let actual = helpers::cube().project_xy().unwrap();
    
    assert!(helpers::polygons_approx_eq(&actual, &expected, 1e-10));
}
```

For boolean ops where fixture comparison fails due to float bit differences, use `approx_eq`:
```rust
#[test]
fn test_boolean_subtract() {
    let bytes = include_bytes!("fixtures/cube_sub_cube.bincode");
    let expected: Manifold<f64> = bincode::deserialize(bytes).unwrap();
    let a = helpers::cube();
    let b = a.translate(0.1, 0., 0.).unwrap();
    let actual = compute_boolean(&a, &b, OpType::Subtract).unwrap();
    
    assert!(actual.approx_eq(&expected, 1e-10));
}
```

## Fixture Generation Flow

1. User runs: `cargo run --bin generate_fixtures`
2. Binary creates `tests/fixtures/` if needed
3. For each test case: generates manifold → serializes via bincode → writes `.bincode` file
4. Prints summary

## Key Differences from JSON Approach

| Aspect | JSON | Bincode |
|--------|------|---------|
| NaN support | `null` (can't deserialize) | Exact bytes preserved |
| Float precision | Text formatting may alter bits | Byte-exact |
| Readability | Human-readable | Binary |
| Compare method | String comparison | `approx_eq` for Manifold, `polygons_approx_eq` for MultiPolygon |

## `Manifold::approx_eq` Method

Compares geometry-relevant fields with epsilon tolerance:
- Counts: `vertex_count`, `face_count`, `halfedge_count`
- Lengths: `positions`, `halfedges`, `face_normals`, `vert_normals`
- Values: positions, halfedge topology (tail/head/pair), normals

Skips: `bounding_box`, `spatial_tol`, `snap_tol`, `collider`, `coplanar`, `original_idx`

```rust
pub fn approx_eq(&self, other: &Self, eps: T) -> bool {
    // Compare counts, lengths, positions, halfedge topology, normals
    // with epsilon tolerance on all float fields
}
```
