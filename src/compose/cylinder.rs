//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use nalgebra::Vector3;
use thiserror::Error;

use crate::{Manifold, common::BoolReal, manifold::ManifoldError};

pub fn generate_cylinder<T: BoolReal>(
    r: T,      // radius
    h: T,      // height
    d0: usize, // sectors
    d1: usize, // stacks
) -> Result<Manifold<T>, CylinderError> {
    if d0 < 3 || d1 < 1 {
        return Err(CylinderError::InvalidSectorCount);
    }
    let mut ps = vec![];
    let mut ts = vec![];

    ps.push(Vector3::new(
        T::zero(),
        h * T::cast_from_f64(0.5),
        T::zero(),
    ));
    ps.push(Vector3::new(
        T::zero(),
        -h * T::cast_from_f64(0.5),
        T::zero(),
    ));

    for i in 0..=d1 {
        let y = h * T::cast_from_f64(0.5) - (T::cast_from_usize(i) / T::cast_from_usize(d1)) * h;
        for j in 0..d0 {
            let (s, c) =
                (T::cast_from_usize(2) * T::PI * (T::cast_from_usize(j) / T::cast_from_usize(d0)))
                    .sin_cos();
            ps.push(Vector3::new(r * c, y, r * s));
        }
    }

    for j in 0..d0 {
        let k = (j + 1) % d0;
        let v0 = 2 + j;
        let v1 = 2 + k;
        let v2 = 2 + d1 * d0 + j;
        let v3 = 2 + d1 * d0 + k;
        ts.push(Vector3::new(0, v1, v0));
        ts.push(Vector3::new(1, v2, v3));
    }

    for i in 0..d1 {
        let r0 = 2 + i * d0;
        let r1 = 2 + (i + 1) * d0;
        for j in 0..d0 {
            let k = (j + 1) % d0;
            ts.push(Vector3::new(r0 + j, r0 + k, r1 + j));
            ts.push(Vector3::new(r0 + k, r1 + k, r1 + j));
        }
    }

    let manifold = Manifold::new_from_raw(ps, ts, None, None)?;

    Ok(manifold)
}

#[derive(Debug, Error)]
pub enum CylinderError {
    #[error("sectors must be >= 3 and stacks must be >= 1")]
    InvalidSectorCount,

    #[error("{0}")]
    Manifold(#[from] ManifoldError),
}
