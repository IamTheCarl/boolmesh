//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use fxhash::FxHashMap;
use nalgebra::Vector3;
use thiserror::Error;

use crate::common::BoolReal;
use crate::{manifold::ManifoldError, Manifold};

pub fn generate_uv_sphere<T: BoolReal>(
    d0: usize, // sectors
    d1: usize, // stacks
) -> Result<Manifold<T>, UVSphereError> {
    if d0 < 3 || d1 < 2 {
        return Err(UVSphereError::InvalidSectorCount);
    }

    let mut ps = vec![];
    let mut ts = vec![];

    ps.push(Vector3::y());
    ps.push(-Vector3::y());

    for i in 1..d1 {
        let i = T::cast_from_usize(i);
        let (sp, cp) = (T::PI * (i / T::cast_from_usize(d1))).sin_cos();
        for j in 0..d0 {
            let j = T::cast_from_usize(j);

            let (st, ct) = (T::cast_from_usize(2) * T::PI * (j / T::cast_from_usize(d0))).sin_cos();
            ps.push(Vector3::new(sp * ct, cp, sp * st));
        }
    }

    for s in 0..d1 {
        for j in 0..d0 {
            let k = (j + 1) % d0;
            if s == 0 {
                ts.push(Vector3::new(0, 2 + k, 2 + j));
            } else if s == d1 - 1 {
                let r0 = 2 + (s - 1) * d0;
                ts.push(Vector3::new(r0 + j, r0 + k, 1));
            } else {
                let r0 = 2 + (s - 1) * d0;
                let r1 = 2 + s * d0;
                ts.push(Vector3::new(r0 + j, r0 + k, r1 + j));
                ts.push(Vector3::new(r0 + k, r1 + k, r1 + j));
            }
        }
    }

    let manifold = Manifold::new_from_raw(ps, ts, None, None)?;

    Ok(manifold)
}

pub fn generate_icosphere<T: BoolReal>(subdivisions: u32) -> Result<Manifold<T>, ManifoldError> {
    let phi = (T::one() + T::cast_from_f64(5.0).sqrt()) / T::cast_from_usize(2);

    let mut ps = vec![
        Vector3::new(-T::one(), phi, T::zero()).normalize(),
        Vector3::new(T::one(), phi, T::zero()).normalize(),
        Vector3::new(-T::one(), -phi, T::zero()).normalize(),
        Vector3::new(T::one(), -phi, T::zero()).normalize(),
        Vector3::new(T::zero(), -T::one(), phi).normalize(),
        Vector3::new(T::zero(), T::one(), phi).normalize(),
        Vector3::new(T::zero(), -T::one(), -phi).normalize(),
        Vector3::new(T::zero(), T::one(), -phi).normalize(),
        Vector3::new(phi, T::zero(), -T::one()).normalize(),
        Vector3::new(phi, T::zero(), T::one()).normalize(),
        Vector3::new(-phi, T::zero(), -T::one()).normalize(),
        Vector3::new(-phi, T::zero(), T::one()).normalize(),
    ];

    let mut ts = vec![
        Vector3::new(0, 11, 5),
        Vector3::new(0, 5, 1),
        Vector3::new(0, 1, 7),
        Vector3::new(0, 7, 10),
        Vector3::new(0, 10, 11),
        Vector3::new(1, 5, 9),
        Vector3::new(5, 11, 4),
        Vector3::new(11, 10, 2),
        Vector3::new(10, 7, 6),
        Vector3::new(7, 1, 8),
        Vector3::new(3, 9, 4),
        Vector3::new(3, 4, 2),
        Vector3::new(3, 2, 6),
        Vector3::new(3, 6, 8),
        Vector3::new(3, 8, 9),
        Vector3::new(4, 9, 5),
        Vector3::new(2, 4, 11),
        Vector3::new(6, 2, 10),
        Vector3::new(8, 6, 7),
        Vector3::new(9, 8, 1),
    ];

    let mut cache = FxHashMap::default();

    let get_midpoint = |vid1: usize,
                        vid2: usize,
                        verts: &mut Vec<Vector3<T>>,
                        cache: &mut FxHashMap<(usize, usize), usize>| {
        let e = if vid1 < vid2 {
            (vid1, vid2)
        } else {
            (vid2, vid1)
        };
        if let Some(&i) = cache.get(&e) {
            return i;
        }

        let v1 = verts[vid1];
        let v2 = verts[vid2];
        verts.push((v1 + v2).normalize());
        let i_ = verts.len() - 1;
        cache.insert(e, i_);
        i_
    };

    for _ in 0..subdivisions {
        let mut ts_ = Vec::with_capacity(ts.len() * 4);
        for t in ts {
            let a = get_midpoint(t[0], t[1], &mut ps, &mut cache);
            let b = get_midpoint(t[1], t[2], &mut ps, &mut cache);
            let c = get_midpoint(t[2], t[0], &mut ps, &mut cache);

            ts_.push(Vector3::new(t[0], a, c));
            ts_.push(Vector3::new(t[1], b, a));
            ts_.push(Vector3::new(t[2], c, b));
            ts_.push(Vector3::new(a, b, c));
        }
        ts = ts_;
    }

    Manifold::new_from_raw(ps, ts, None, None)
}

#[derive(Debug, Error)]
pub enum UVSphereError {
    #[error("sectors must be >= 3 and stacks must be >= 2")]
    InvalidSectorCount,

    #[error("{0}")]
    Manifold(#[from] ManifoldError),
}
