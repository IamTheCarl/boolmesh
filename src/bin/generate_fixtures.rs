//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

//! Fixture generator — produces bincode files in `tests/fixtures/` from example logic.
//!
//! Bincode is used because it preserves NaN values exactly (JSON would turn them into `null`).
//!
//! Run with: `cargo run --bin generate_fixtures`

use std::fs;
use std::path::Path;

use boolmesh::prelude::*;
use geo::{BooleanOps, Coord, LineString, MultiPolygon, Polygon, Rect};
use nalgebra::Vector3;
use std::f64::consts::PI;

/// Canonicalize a MultiPolygon by removing duplicate closing points and starting
/// each ring at the lexicographically smallest coordinate. Exterior rings are
/// forced CCW, interior rings CW, to guarantee consistent vertex ordering.
fn canonicalize_polygon(poly: &MultiPolygon<f64>) -> MultiPolygon<f64> {
    let polygons: Vec<_> = poly
        .0
        .iter()
        .map(|p| {
            let ext = force_ccw_ring(p.exterior());
            let ints: Vec<_> = p.interiors().iter().map(force_cw_ring).collect();
            Polygon::new(ext, ints)
        })
        .collect();
    MultiPolygon(polygons)
}

fn canonicalize_ring(ring: &LineString<f64>) -> LineString<f64> {
    let coords: Vec<_> = ring.coords().map(|c| Coord { x: c.x, y: c.y }).collect();
    let mut unique = coords;
    if unique.len() >= 2
        && unique[0].x == unique[unique.len() - 1].x
        && unique[0].y == unique[unique.len() - 1].y
    {
        unique.pop();
    }
    if unique.len() >= 2 {
        let min_idx = (1..unique.len())
            .min_by(|&i, &j| {
                (unique[i].x, unique[i].y)
                    .partial_cmp(&(unique[j].x, unique[j].y))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(1);
        if min_idx > 0 {
            unique.rotate_left(min_idx);
        }
    }
    unique.push(unique[0]);
    LineString::from(unique)
}

/// Force ring direction to counter-clockwise (CCW).
/// Exterior rings should be CCW, interiors CW — we force all exterior rings CCW
/// to guarantee consistent vertex ordering regardless of geo crate non-determinism.
fn force_ccw_ring(ring: &LineString<f64>) -> LineString<f64> {
    let mut r = canonicalize_ring(ring);
    // Calculate signed area (shoelace formula) - positive = CCW, negative = CW
    let coords: Vec<_> = r.coords().map(|c| Coord { x: c.x, y: c.y }).collect();
    let area: f64 = coords
        .iter()
        .zip(coords.iter().skip(1).chain(coords.iter().take(1)))
        .fold(0.0f64, |acc, (a, b)| acc + a.x * b.y - a.y * b.x);
    if area < 0.0 {
        // Reverse the ring to make it CCW
        let mut reversed = coords;
        reversed.reverse();
        // Add closing point
        reversed.push(reversed[0]);
        r = LineString::from(reversed);
    }
    r
}

/// Force ring direction to clockwise (CW).
fn force_cw_ring(ring: &LineString<f64>) -> LineString<f64> {
    let mut r = canonicalize_ring(ring);
    // Calculate signed area (shoelace formula) - positive = CCW, negative = CW
    let coords: Vec<_> = r.coords().map(|c| Coord { x: c.x, y: c.y }).collect();
    let area: f64 = coords
        .iter()
        .zip(coords.iter().skip(1).chain(coords.iter().take(1)))
        .fold(0.0f64, |acc, (a, b)| acc + a.x * b.y - a.y * b.x);
    if area > 0.0 {
        // Reverse the ring to make it CW
        let mut reversed = coords;
        reversed.reverse();
        // Add closing point
        reversed.push(reversed[0]);
        r = LineString::from(reversed);
    }
    r
}

fn main() {
    let fixture_dir = Path::new("tests/fixtures");
    fs::create_dir_all(fixture_dir).expect("Failed to create fixtures directory");

    println!("Generating fixtures in {}", fixture_dir.display());

    // Primitives
    generate_primitive_fixtures(fixture_dir);

    // Transformations
    generate_transform_fixtures(fixture_dir);

    // Boolean operations
    generate_boolean_fixtures(fixture_dir);

    // Extrusions
    generate_extrusion_fixtures(fixture_dir);

    // Revolutions
    generate_revolution_fixtures(fixture_dir);

    // Projections & slices
    generate_projection_slice_fixtures(fixture_dir);

    // Composition
    generate_composition_fixtures(fixture_dir);

    println!(
        "Done. {} fixture files generated.",
        fs::read_dir(fixture_dir).unwrap().count()
    );
}

fn write_fixture<T: serde::Serialize>(path: &Path, label: &str, value: &T) {
    let bytes = bincode::serialize(value).expect("Failed to serialize fixture");
    fs::write(path, &bytes).expect("Failed to write fixture file");
    println!(
        "  {} → {}",
        label,
        path.file_name().unwrap().to_string_lossy()
    );
}

// --- Primitives ---

fn generate_primitive_fixtures(dir: &Path) {
    let cube = generate_cube::<f64>().unwrap();
    write_fixture(&dir.join("cube.bincode"), "cube", &cube);

    let cyl = generate_cylinder::<f64>(1., 1., 30, 10).unwrap();
    write_fixture(&dir.join("cylinder.bincode"), "cylinder", &cyl);

    let sphere: Manifold = generate_uv_sphere::<f64>(30, 30).unwrap();
    write_fixture(&dir.join("uv_sphere.bincode"), "uv_sphere", &sphere);

    let ico: Manifold = generate_icosphere::<f64>(1).unwrap();
    write_fixture(&dir.join("icosphere.bincode"), "icosphere", &ico);

    let torus = generate_torus::<f64>(1., 0.1, 30, 30).unwrap();
    write_fixture(&dir.join("torus.bincode"), "torus", &torus);

    let cone =
        generate_cone::<f64>(Vector3::new(0., 0., 0.), Vector3::new(0., 0., 1.), 1., 30).unwrap();
    write_fixture(&dir.join("cone.bincode"), "cone", &cone);
}

// --- Transformations ---

fn generate_transform_fixtures(dir: &Path) {
    let cube = generate_cube::<f64>().unwrap();

    let translated = cube.translate(2., 0., 0.).unwrap();
    write_fixture(
        &dir.join("cube_translated.bincode"),
        "cube_translated",
        &translated,
    );

    let rotated = cube.rotate(PI / 4., 0., 0.).unwrap();
    write_fixture(&dir.join("cube_rotated.bincode"), "cube_rotated", &rotated);

    let scaled = cube.scale(2., 1., 1.).unwrap();
    write_fixture(&dir.join("cube_scaled.bincode"), "cube_scaled", &scaled);
}

// --- Boolean Operations ---

fn generate_boolean_fixtures(dir: &Path) {
    let cube = generate_cube::<f64>().unwrap();
    let cube2 = cube.translate(2., 0., 0.).unwrap();
    let add = compute_boolean(&cube, &cube2, OpType::Add).unwrap();
    write_fixture(&dir.join("cube_add_cube.bincode"), "cube_add_cube", &add);

    let overlap = cube.translate(0.1, 0., 0.).unwrap();
    let sub = compute_boolean(&cube, &overlap, OpType::Subtract).unwrap();
    write_fixture(&dir.join("cube_sub_cube.bincode"), "cube_sub_cube", &sub);

    let inter = compute_boolean(&cube, &overlap, OpType::Intersect).unwrap();
    write_fixture(
        &dir.join("cube_intersect_cube.bincode"),
        "cube_intersect_cube",
        &inter,
    );

    let cyl = generate_cylinder::<f64>(1., 1., 30, 10).unwrap();
    let sph = generate_uv_sphere::<f64>(30, 30).unwrap();
    let sph_t = sph.translate(0., 0.5, 0.).unwrap();
    let cyl_sph = compute_boolean(&cyl, &sph_t, OpType::Add).unwrap();
    write_fixture(
        &dir.join("cyl_sphere_add.bincode"),
        "cyl_sphere_add",
        &cyl_sph,
    );

    let torus = generate_torus::<f64>(1., 0.1, 30, 30).unwrap();
    let result = compute_boolean(&cyl_sph, &torus, OpType::Subtract).unwrap();
    write_fixture(
        &dir.join("cyl_sphere_sub_torus.bincode"),
        "cyl_sphere_sub_torus",
        &result,
    );
}

// --- Extrusions ---

fn generate_extrusion_fixtures(dir: &Path) {
    fn make_hollow_square() -> MultiPolygon<f64> {
        let square = Rect::new(Coord { x: -0.5, y: -0.5 }, Coord { x: 0.5, y: 0.5 }).to_polygon();
        let hole = Rect::new(Coord { x: -0.25, y: -0.25 }, Coord { x: 0.25, y: 0.25 }).to_polygon();
        let result = square.boolean_op(&hole, geo::OpType::Difference);
        let canon = canonicalize_polygon(&result);
        println!(
            "make_hollow_square exterior: {:?}",
            canon.0[0].exterior().coords().collect::<Vec<_>>()
        );
        if !canon.0[0].interiors().is_empty() {
            println!(
                "make_hollow_square interior: {:?}",
                canon.0[0].interiors()[0].coords().collect::<Vec<_>>()
            );
        }
        canon
    }

    let square = Rect::new(Coord { x: -0.5, y: -0.5 }, Coord { x: 0.5, y: 0.5 }).to_polygon();
    let extrude_square = square.extrude(1.0, 1, 0.0, Vector2::new(1.0, 1.0)).unwrap();
    write_fixture(
        &dir.join("extrude_square.bincode"),
        "extrude_square",
        &extrude_square,
    );

    let hollow = make_hollow_square();
    let extrude_hollow = hollow.extrude(1.0, 1, 0.0, Vector2::new(1.0, 1.0)).unwrap();
    write_fixture(
        &dir.join("extrude_hollow_square.bincode"),
        "extrude_hollow_square",
        &extrude_hollow,
    );

    let extrude_twisted = hollow.extrude(1.0, 20, PI, Vector2::new(1.0, 1.0)).unwrap();
    write_fixture(
        &dir.join("extrude_twisted.bincode"),
        "extrude_twisted",
        &extrude_twisted,
    );

    let extrude_tapered = hollow.extrude(1.0, 50, PI, Vector2::new(0.0, 0.0)).unwrap();
    write_fixture(
        &dir.join("extrude_tapered.bincode"),
        "extrude_tapered",
        &extrude_tapered,
    );
}

// --- Revolutions ---

fn generate_revolution_fixtures(dir: &Path) {
    let square = Rect::new(Coord { x: 0.0, y: -0.5 }, Coord { x: 1.0, y: 0.5 }).to_polygon();
    let bite = Rect::new(Coord { x: 0.25, y: -0.25 }, Coord { x: 0.75, y: 0.5 }).to_polygon();
    let square_with_bite_result = square.boolean_op(&bite, geo::OpType::Difference);
    let square_with_bite = canonicalize_polygon(&square_with_bite_result);

    let revolve = square_with_bite.revolve(15, PI * 2.0).unwrap();
    write_fixture(
        &dir.join("revolve_square.bincode"),
        "revolve_square",
        &revolve,
    );

    let revolve_half = square_with_bite.revolve(5, PI * 0.5).unwrap();
    write_fixture(
        &dir.join("revolve_half.bincode"),
        "revolve_half",
        &revolve_half,
    );
}

// --- Projections & Slices ---

fn generate_projection_slice_fixtures(dir: &Path) {
    let cube = generate_cube::<f64>().unwrap();

    let proj = cube.project_xy().unwrap();
    write_fixture(&dir.join("cube_project.bincode"), "cube_project", &proj);

    let slice_0 = cube.slice(0.0).unwrap();
    write_fixture(&dir.join("cube_slice_0.bincode"), "cube_slice_0", &slice_0);

    let slice_025 = cube.slice(0.25).unwrap();
    write_fixture(
        &dir.join("cube_slice_025.bincode"),
        "cube_slice_025",
        &slice_025,
    );

    let slice_m025 = cube.slice(-0.25).unwrap();
    write_fixture(
        &dir.join("cube_slice_minus025.bincode"),
        "cube_slice_minus025",
        &slice_m025,
    );

    let cyl = generate_cylinder::<f64>(1., 1., 30, 10).unwrap();
    let cyl_proj = cyl.project_xy().unwrap();
    write_fixture(
        &dir.join("cylinder_project.bincode"),
        "cylinder_project",
        &cyl_proj,
    );

    let sphere = generate_uv_sphere::<f64>(30, 30).unwrap();
    let sphere_proj = sphere.project_xy().unwrap();
    write_fixture(
        &dir.join("uv_sphere_project.bincode"),
        "uv_sphere_project",
        &sphere_proj,
    );

    let ico = generate_icosphere::<f64>(1).unwrap();
    let ico_proj = ico.project_xy().unwrap();
    write_fixture(
        &dir.join("icosphere_project.bincode"),
        "icosphere_project",
        &ico_proj,
    );

    let torus = generate_torus::<f64>(1., 0.1, 30, 30).unwrap();
    let torus_proj = torus.project_xy().unwrap();
    write_fixture(
        &dir.join("torus_project.bincode"),
        "torus_project",
        &torus_proj,
    );

    let cone =
        generate_cone::<f64>(Vector3::new(0., 0., 0.), Vector3::new(0., 0., 1.), 1., 30).unwrap();
    match cone.project_xy() {
        Ok(proj) => write_fixture(&dir.join("cone_project.bincode"), "cone_project", &proj),
        Err(_) => println!("  cone_project → SKIPPED (no polygons from XY projection)"),
    }

    let square = Rect::new(Coord { x: -0.5, y: -0.5 }, Coord { x: 0.5, y: 0.5 }).to_polygon();
    let extrude_square = square.extrude(1.0, 1, 0.0, Vector2::new(1.0, 1.0)).unwrap();
    let proj = extrude_square.project_xy().unwrap();
    write_fixture(
        &dir.join("extrude_square_project.bincode"),
        "extrude_square_project",
        &proj,
    );

    let hollow = {
        let square = Rect::new(Coord { x: -0.5, y: -0.5 }, Coord { x: 0.5, y: 0.5 }).to_polygon();
        let hole = Rect::new(Coord { x: -0.25, y: -0.25 }, Coord { x: 0.25, y: 0.25 }).to_polygon();
        let result = square.boolean_op(&hole, geo::OpType::Difference);
        canonicalize_polygon(&result)
    };
    let extrude_hollow = hollow.extrude(1.0, 1, 0.0, Vector2::new(1.0, 1.0)).unwrap();
    let proj = extrude_hollow.project_xy().unwrap();
    write_fixture(
        &dir.join("extrude_hollow_project.bincode"),
        "extrude_hollow_project",
        &proj,
    );

    let square_with_bite = {
        let square = Rect::new(Coord { x: 0.0, y: -0.5 }, Coord { x: 1.0, y: 0.5 }).to_polygon();
        let bite = Rect::new(Coord { x: 0.25, y: -0.25 }, Coord { x: 0.75, y: 0.5 }).to_polygon();
        let result = square.boolean_op(&bite, geo::OpType::Difference);
        canonicalize_polygon(&result)
    };
    let revolve = square_with_bite.revolve(15, PI * 2.0).unwrap();
    let proj = revolve.project_xy().unwrap();
    write_fixture(
        &dir.join("revolve_square_project.bincode"),
        "revolve_square_project",
        &proj,
    );
}

// --- Composition ---

fn generate_composition_fixtures(dir: &Path) {
    let cube = generate_cube::<f64>().unwrap();
    let cube2 = cube.translate(2., 0., 0.).unwrap();
    let composed = compose(&vec![cube, cube2]).unwrap();
    write_fixture(
        &dir.join("compose_two_cubes.bincode"),
        "compose_two_cubes",
        &composed,
    );
}
