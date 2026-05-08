//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

pub mod ear_clip;
pub mod flat_tree;
pub mod tri_halfs;

use crate::boolean45::Boolean45;
use crate::common::BoolReal;
use crate::triangulation::ear_clip::EarClip;
#[cfg(feature = "rayon")]
use crate::triangulation::tri_halfs::tri_halfs_multi;
#[cfg(not(feature = "rayon"))]
use crate::triangulation::tri_halfs::tri_halfs_single;
use crate::{compute_aa_proj, get_aa_proj_matrix, is_ccw_3d, HalfEdge, Manifold, Tref};
use nalgebra::{Vector2, Vector3};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::{BTreeMap, VecDeque};
use thiserror::Error;

pub struct Triangulation<T> {
    pub hs: Vec<HalfEdge>,
    pub rs: Vec<Tref>,
    pub ns: Vec<Vector3<T>>,
}

pub fn triangulate<T: BoolReal>(
    mp: &Manifold<T>,
    mq: &Manifold<T>,
    b45: &Boolean45<T>,
    eps: T,
) -> Result<Triangulation<T>, TriangulationError> {
    #[cfg(feature = "rayon")]
    {
        let indexed: Vec<_> = (0..b45.hid_per_f.len() - 1)
            .into_par_iter()
            .map(|fid| {
                let hid = b45.hid_per_f[fid] as usize;
                let ts_ = process_face(&b45, fid, eps);
                let rs_ = vec![b45.rs[hid].clone(); ts_.len()];
                let ns_ = vec![b45.ns[fid].clone(); ts_.len()];
                (fid, ts_, rs_, ns_)
            })
            .collect();
        
        let mut ts = Vec::with_capacity(indexed.len() * 2);
        let mut rs = vec![];
        let mut ns = vec![];
        for (_, ts_, rs_, ns_) in indexed {
            ts.extend(ts_);
            rs.extend(rs_);
            ns.extend(ns_);
        }
        update_reference(mp, mq, &mut rs);
        Ok(Triangulation {
            hs: tri_halfs_multi(&mut ts),
            ns,
            rs,
        })
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut ts = vec![];
        let mut ns = vec![];
        let mut rs = vec![];

        for fid in 0..b45.hid_per_f.len() - 1 {
            let hid = b45.hid_per_f[fid] as usize;
            let t = process_face(b45, fid, eps);
            let r = b45.rs[hid];
            let n = b45.ns[fid];
            rs.extend(vec![r; t.len()]);
            ns.extend(vec![n; t.len()]);
            ts.extend(t);
        }
        update_reference(mp, mq, &mut rs);
        Ok(Triangulation {
            hs: tri_halfs_single(&ts),
            ns,
            rs,
        })
    }
}

fn process_face<T: BoolReal>(b45: &Boolean45<T>, fid: usize, eps: T) -> Vec<Vector3<usize>> {
    let e0 = b45.hid_per_f[fid] as usize;
    let e1 = b45.hid_per_f[fid + 1] as usize;
    match e1 - e0 {
        3 => single_triangulate(b45, e0),
        4 => square_triangulate(b45, fid, eps),
        _ => general_triangulate(b45, fid, eps),
    }
}

pub fn assemble_halfs(hs: &[HalfEdge], hid_f: &[i32], fid: usize) -> Vec<Vec<usize>> {
    let bgn = hid_f[fid] as usize;
    let end = hid_f[fid + 1] as usize;
    let num = end - bgn;
    let mut v2h = BTreeMap::new();

    for (i, half) in hs.iter().enumerate().skip(bgn).take(num) {
        let id = usize::from(half.tail);
        v2h.entry(id).or_insert_with(VecDeque::new).push_front(i);
    }

    let mut loops: Vec<Vec<usize>> = vec![];
    let mut hid0 = 0;
    let mut hid1 = 0;
    loop {
        if hid1 == hid0 {
            if v2h.is_empty() {
                break;
            }
            hid0 = v2h.first_entry().unwrap().get().back().copied().unwrap();
            hid1 = hid0;
            loops.push(Vec::new());
        }
        loops.last_mut().unwrap().push(hid1);

        let edge_id = usize::from(hs[hid1].head);
        let queue = v2h.get_mut(&edge_id).unwrap();
        hid1 = queue.pop_back().unwrap();
        if queue.is_empty() {
            v2h.remove(&edge_id);
        }
    }
    loops
}

fn single_triangulate<T: BoolReal>(b45: &Boolean45<T>, hid: usize) -> Vec<Vector3<usize>> {
    let mut idcs = [hid, hid + 1, hid + 2];
    let mut tails = vec![];
    let mut heads = vec![];
    for id in idcs.iter() {
        tails.push(usize::from(b45.hs[*id].tail));
        heads.push(usize::from(b45.hs[*id].head));
    }
    if heads[0] == tails[2] {
        idcs.swap(1, 2);
    }

    vec![Vector3::new(
        usize::from(b45.hs[idcs[0]].tail),
        usize::from(b45.hs[idcs[1]].tail),
        usize::from(b45.hs[idcs[2]].tail),
    )]
}

fn square_triangulate<T: BoolReal>(b45: &Boolean45<T>, fid: usize, eps: T) -> Vec<Vector3<usize>> {
    let ccw = |tri: Vector3<usize>| {
        is_ccw_3d(
            &b45.ps[usize::from(b45.hs[tri[0]].tail)],
            &b45.ps[usize::from(b45.hs[tri[1]].tail)],
            &b45.ps[usize::from(b45.hs[tri[2]].tail)],
            &b45.ns[fid],
            eps,
        ) >= 0
    };

    let q = &assemble_halfs(&b45.hs, &b45.hid_per_f, fid)[0];
    let tris = [
        vec![
            Vector3::new(q[0], q[1], q[2]),
            Vector3::new(q[0], q[2], q[3]),
        ],
        vec![
            Vector3::new(q[1], q[2], q[3]),
            Vector3::new(q[0], q[1], q[3]),
        ],
    ];
    let mut choice: usize = 0;

    if !(ccw(tris[0][0]) && ccw(tris[0][1])) {
        choice = 1;
    } else if ccw(tris[1][0]) && ccw(tris[1][1]) {
        let diag0 = b45.ps[usize::from(b45.hs[q[0]].tail)] - b45.ps[usize::from(b45.hs[q[2]].tail)];
        let diag1 = b45.ps[usize::from(b45.hs[q[1]].tail)] - b45.ps[usize::from(b45.hs[q[3]].tail)];
        if diag0.norm() > diag1.norm() {
            choice = 1;
        }
    }

    tris[choice]
        .iter()
        .map(|t| Vector3::new(usize::from(b45.hs[t.x].tail), usize::from(b45.hs[t.y].tail), usize::from(b45.hs[t.z].tail)))
        .collect()
}

fn general_triangulate<T: BoolReal>(b45: &Boolean45<T>, fid: usize, eps: T) -> Vec<Vector3<usize>> {
    let proj = get_aa_proj_matrix(&b45.ns[fid]);
    let loops = assemble_halfs(&b45.hs, &b45.hid_per_f, fid);
    let polys = loops
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&e| {
                    let i = usize::from(b45.hs[e].tail);
                    let p = compute_aa_proj(&proj, &b45.ps[i]);
                    Pt { pos: p, idx: e }
                })
                .collect()
        })
        .collect::<Vec<Vec<_>>>();

    EarClip::new(&polys, eps)
        .triangulate()
        .iter()
        .map(|t| Vector3::new(usize::from(b45.hs[t.x].tail), usize::from(b45.hs[t.y].tail), usize::from(b45.hs[t.z].tail)))
        .collect()
}

#[derive(Debug, Clone)]
pub struct Pt<T: BoolReal> {
    pub pos: Vector2<T>,
    pub idx: usize,
}

fn update_reference<T: BoolReal>(mp: &Manifold<T>, mq: &Manifold<T>, rs: &mut [Tref]) {
    for r in rs.iter_mut() {
        let fid = r.fid;
        let pq = r.mid == 0;
        r.pid = if pq {
            mp.coplanar[fid]
        } else {
            mq.coplanar[fid]
        };
    }
}

#[derive(Debug, Error)]
pub enum TriangulationError {}
