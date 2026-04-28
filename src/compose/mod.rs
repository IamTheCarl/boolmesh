//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.
#![allow(clippy::unnecessary_cast)]

pub mod cone;
pub mod cube;
pub mod cylinder;
pub mod extrusion;
pub mod sphere;
pub mod torus;

pub use cone::*;
pub use cube::*;
pub use cylinder::*;
pub use extrusion::*;
pub use sphere::*;
pub use torus::*;

use crate::manifold::ManifoldError;
use crate::{BoolReal, Manifold};

pub fn compose<T: BoolReal>(ms: &Vec<Manifold<T>>) -> Result<Manifold<T>, ManifoldError> {
    let mut ps = vec![];
    let mut ts = vec![];
    let mut offset = 0;
    for m in ms {
        for h in m.hs.iter() {
            ts.push(h.tail.0 + offset);
        }
        for p in m.ps.iter() {
            ps.push(p.x);
            ps.push(p.y);
            ps.push(p.z);
        }
        offset += m.nv;
    }
    Manifold::new(&ps, &ts)
}

pub fn fractal<T: BoolReal>(
    hole: &Manifold<T>,
    holes: &mut Vec<Manifold<T>>,
    x: T,
    y: T,
    w: T,
    depth: usize,
    depth_max: usize,
) -> Result<(), ManifoldError> {
    let w = w / T::cast_from_usize(3);
    let mut m = hole.clone();
    m = m.scale(w, w, T::one())?;
    m = m.translate(x, y, T::zero())?;
    holes.push(m);

    if depth == depth_max {
        return Ok(());
    }

    for xy in [
        (x - w, y - w),
        (x - w, y),
        (x - w, y + w),
        (x, y + w),
        (x + w, y + w),
        (x + w, y),
        (x + w, y - w),
        (x, y - w),
    ] {
        fractal(hole, holes, xy.0, xy.1, w, depth + 1, depth_max)?;
    }

    Ok(())
}
