//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use nalgebra::Vector2;

use super::kernel02::Kernel02;
use crate::Manifold;
use crate::bounds::{BPos, Query};
use crate::common::BoolReal;

pub fn winding03<T: BoolReal>(
    mp: &Manifold<T>,
    mq: &Manifold<T>,
    expand: T,
    fwd: bool,
) -> Vec<i32> {
    let ma = if fwd { mp } else { mq };
    let mb = if fwd { mq } else { mp };

    let mut w03 = vec![0; ma.vertex_count];
    let k02 = Kernel02 {
        ps_p: &ma.positions,
        ps_q: &mb.positions,
        hs_q: &mb.halfedges,
        ns: &mp.vert_normals,
        expand,
        fwd,
    };

    mb.collider.collision(
        ma.positions.iter().enumerate().map(|(i, p)| {
            Query::Pt(BPos {
                id: Some(i),
                pos: Vector2::new(p.x, p.y),
            })
        }),
        &mut |a, b| {
            if let Some((s, _)) = k02.op(a, b) {
                w03[a] += s * if fwd { 1 } else { -1 };
            }
        },
    );

    w03
}
