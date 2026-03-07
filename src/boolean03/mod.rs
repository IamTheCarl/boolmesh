//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

pub mod kernel01;
pub mod kernel02;
pub mod kernel03;
pub mod kernel11;
pub mod kernel12;
use nalgebra::Vector3;

use crate::boolean03::kernel03::winding03;
use crate::boolean03::kernel12::intersect12;
use crate::common::{BoolReal, OpType};
use crate::manifold::Manifold;

pub struct Boolean03<T> {
    pub p1q2: Vec<[usize; 2]>,
    pub p2q1: Vec<[usize; 2]>,
    pub x12: Vec<i32>,
    pub x21: Vec<i32>,
    pub w03: Vec<i32>,
    pub w30: Vec<i32>,
    pub v12: Vec<Vector3<T>>,
    pub v21: Vec<Vector3<T>>,
}

pub fn boolean03<T: BoolReal>(mp: &Manifold<T>, mq: &Manifold<T>, op: &OpType) -> Boolean03<T> {
    let e = if op == &OpType::Add {
        T::one()
    } else {
        -T::one()
    };
    let mut p1q2 = vec![];
    let mut p2q1 = vec![];
    let x12;
    let v12;
    let x21;
    let v21;
    let w03;
    let w30;

    #[cfg(feature = "rayon")]
    {
        (((x12, v12), w03), ((x21, v21), w30)) = rayon::join(
            || {
                rayon::join(
                    || intersect12(mp, mq, &mut p1q2, e, true),
                    || winding03(mp, mq, e, true),
                )
            },
            || {
                rayon::join(
                    || intersect12(mp, mq, &mut p2q1, e, false),
                    || winding03(mp, mq, e, false),
                )
            },
        );
    }

    #[cfg(not(feature = "rayon"))]
    {
        ((x12, v12), w03) = (
            intersect12(mp, mq, &mut p1q2, e, true),
            winding03(mp, mq, e, true),
        );
        ((x21, v21), w30) = (
            intersect12(mp, mq, &mut p2q1, e, false),
            winding03(mp, mq, e, false),
        );
    }

    Boolean03 {
        p1q2,
        p2q1,
        x12,
        x21,
        w03,
        w30,
        v12,
        v21,
    }
}
