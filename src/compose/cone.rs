//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use nalgebra::Vector3;

use crate::common::BoolReal;
use crate::manifold::ManifoldError;
use crate::{compute_orthogonal, Manifold};

pub fn generate_cone<T: BoolReal>(
    apex: Vector3<T>,
    center: Vector3<T>,
    radius: T,
    divide: usize,
) -> Result<Manifold<T>, ManifoldError> {
    let d = T::PI * T::cast_from_usize(2) / T::cast_from_usize(divide);
    let n = (center - apex).normalize();
    let b1 = compute_orthogonal(n);
    let b2 = n.cross(&b1).normalize();
    let mut ps = vec![];
    let mut ts = vec![];

    let ia = divide;
    let ib = divide + 1;
    for i in 0..divide {
        let r = d * T::cast_from_usize(divide);
        ps.push(center + b1 * r.cos() * radius + b2 * r.sin() * radius);
        ts.push(Vector3::new(i, ia, (i + 1) % divide));
        ts.push(Vector3::new(ib, i, (i + 1) % divide));
    }

    ps.push(apex);
    ps.push(center);
    Manifold::new_from_raw(ps, ts, None, None)
}
