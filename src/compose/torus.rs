//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use nalgebra::Vector3;

use crate::{Manifold, common::BoolReal, manifold::ManifoldError};

pub fn generate_torus<T: BoolReal>(
    r0: T,     // major radius
    r1: T,     // minor radius
    d0: usize, // rings
    d1: usize, // sectors
) -> Result<Manifold<T>, ManifoldError> {
    let mut ps = Vec::with_capacity(d0 * d1);
    let mut ts = Vec::with_capacity(d0 * d1 * 6);

    for i in 0..d0 {
        let i = T::cast_from_usize(i);
        let u = i * T::cast_from_usize(2) * T::PI / T::cast_from_usize(d0);
        let (su, cu) = u.sin_cos();

        for j in 0..d1 {
            let j = T::cast_from_usize(j);
            let v = j * T::cast_from_usize(2) * T::PI / T::cast_from_usize(d1);
            let (sv, cv) = v.sin_cos();
            let x = (r0 + r1 * cv) * cu;
            let y = r1 * sv;
            let z = (r0 + r1 * cv) * su;
            ps.push(Vector3::new(x, y, z));
        }
    }

    for i in 0..d0 {
        let ni = (i + 1) % d0;
        for j in 0..d1 {
            let nj = (j + 1) % d1;
            let v0 = i * d1 + j;
            let v1 = i * d1 + nj;
            let v2 = ni * d1 + j;
            let v3 = ni * d1 + nj;
            ts.push(Vector3::new(v0, v1, v2));
            ts.push(Vector3::new(v1, v3, v2));
        }
    }

    Manifold::new_from_raw(ps, ts, None, None)
}
