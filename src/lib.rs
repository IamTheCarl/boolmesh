//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_abs_to_unsigned)]
#![allow(unused_braces)]

mod boolean03;
mod boolean45;
mod common;
mod compose;
mod manifold;
mod simplification;
mod tests;
mod triangulation;

use geo::bool_ops::BoolOpsNum;
use geo::MultiPolygon;
use nalgebra::Vector3;
use thiserror::Error;

use crate::boolean03::boolean03;
use crate::boolean45::boolean45;
use crate::common::*;
use crate::manifold::*;
use crate::simplification::simplify_topology;
use crate::manifold::cleanup_unused_verts_impl;
use crate::triangulation::triangulate;
use crate::triangulation::TriangulationError;

pub use crate::common::BoolReal;
pub use crate::common::{VertexId, HalfEdgeId};

pub mod prelude {
    pub use crate::common::OpType;
    pub use crate::common::{VertexId, HalfEdgeId};
    pub use crate::compose::{
        compose, fractal, generate_cone, generate_cube, generate_cylinder,
        generate_icosphere, generate_torus, generate_uv_sphere, ExtrudePoly, ExtrusionError,
    };
    pub use crate::compute_boolean;
    pub use crate::manifold::{Manifold, Triangle};

    pub use nalgebra::{self, Vector3, Vector2};
}

pub fn compute_boolean<T: BoolReal>(mp: &Manifold<T>, mq: &Manifold<T>, op: OpType) -> Result<Manifold<T>, BooleanError> {
    let eps = mp.spatial_tol.max(mq.spatial_tol);
    let tol = mp.snap_tol.max(mq.snap_tol);

    let b03 = boolean03(mp, mq, &op);
    let mut b45 = boolean45(mp, mq, &b03, &op);
    let mut trg = triangulate(mp, mq, &b45, eps)?;

    simplify_topology(
        &mut trg.hs,
        &mut b45.ps,
        &mut trg.ns,
        &mut trg.rs,
        b45.nv_from_p,
        b45.nv_from_q,
        eps,
    );

    cleanup_unused_verts_impl(&mut b45.ps, &mut trg.hs);

    let manifold = Manifold::new_from_raw(
        b45.ps,
        trg.hs
            .chunks(3)
            .map(|h| Vector3::new(usize::from(h[0].tail), usize::from(h[1].tail), usize::from(h[2].tail)))
            .collect(),
        Some(eps),
        Some(tol),
    )?;

    Ok(manifold)
}

#[derive(Debug, Error)]
pub enum BooleanError {
    #[error("{0}")]
    Trangulate(#[from] TriangulationError),

    #[error("{0}")]
    Manifold(#[from] ManifoldError),
}

//pub fn compute_boolean_from_raw_data(
//    pos0: &[Real],
//    idx0: &[usize],
//    pos1: &[Real],
//    idx1: &[usize],
//    op_type: usize
//) -> Result<Manifold, String>{
//    let mp = Manifold::new(&pos0, &idx0)?;
//    let mq = Manifold::new(&pos1, &idx1)?;
//    let op = match op_type {
//        0 => OpType::Add,
//        1 => OpType::Subtract,
//        2 => OpType::Intersect,
//        _ => return Err("Invalid op_type".into())
//    };
//    compute_boolean(&mp, &mq, op)
//}

#[derive(Debug, Error)]
pub enum ProjectionError {
    #[error("No polygons were produced by the operation")]
    NoPolygons,
}

/// Projects the manifold onto the XY plane. Rotate the manifold to project onto custom planes.
/// * manifold - Input manifold to project
#[deprecated(since = "0.1.10", note = "Use `Manifold::project_xy()` instead")]
pub fn compute_projection<T: BoolReal + BoolOpsNum>(manifold: &Manifold<T>) -> Result<MultiPolygon<T>, ProjectionError> {
    manifold.project_xy()
}

#[deprecated(since = "0.1.10", note = "Use `Manifold::slice()` instead")]
pub fn compute_slice<T: BoolReal + BoolOpsNum>(manifold: &Manifold<T>, height: T) -> Result<MultiPolygon<T>, SliceError> {
    manifold.slice(height)
}

#[derive(Debug, Error)]
pub enum SliceError {
    #[error("No polygons were produced by the operation")]
    NoPolygons,
}
