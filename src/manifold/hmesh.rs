//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.
#![allow(clippy::needless_range_loop)]

use nalgebra::{Vector2, Vector3};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use thiserror::Error;

use crate::common::{BoolReal, VectorExt as _};
use fxhash::FxHashMap;

/// Hmesh preserves the order of pos and idx in any cases.
/// Edges are ordered so as the edge is forward (tail idx < head idx)
#[derive(Debug, Clone)]
pub(in crate::manifold) struct Hmesh<T> {
    pub nv: usize,
    pub nf: usize,
    pub nh: usize,
    pub twin: Vec<usize>,
    pub head: Vec<usize>,
    pub tail: Vec<usize>,
    pub half: Vec<usize>,
    pub vns: Vec<Vector3<T>>,
    pub fns: Vec<Vector3<T>>,
}

fn edge_topology<T: BoolReal>(
    pos: &[Vector3<T>],
    idx: &[Vector3<usize>],
    e2v: &mut Vec<Vector2<usize>>,
    e2f: &mut Vec<Vector2<usize>>,
    f2e: &mut Vec<Vector3<usize>>,
) -> Result<(), HmeshError> {
    if pos.is_empty() {
        return Err(HmeshError::EmptyPositionMatrix);
    }
    if idx.is_empty() {
        return Err(HmeshError::EmptyIndexMatrix);
    }

    let mut edge_map: FxHashMap<(usize, usize), Vec<(usize, usize)>> = FxHashMap::default();

    for (tri_id, idx_) in idx.iter().enumerate() {
        for j in 0..3 {
            let mut v1 = idx_[j];
            let mut v2 = idx_[(j + 1) % 3];
            if v1 > v2 {
                std::mem::swap(&mut v1, &mut v2);
            }
            edge_map.entry((v1, v2)).or_default().push((tri_id, j));
        }
    }

    let mut edge_keys: Vec<_> = edge_map.keys().copied().collect();
    edge_keys.sort();

    let ne = edge_keys.len();
    e2v.resize(ne, Vector2::from_element(usize::MAX));
    e2f.resize(ne, Vector2::from_element(usize::MAX));
    f2e.resize(idx.len(), Vector3::from_element(usize::MAX));

    for (ne_idx, &(v1, v2)) in edge_keys.iter().enumerate() {
        let mut entries = edge_map.remove(&(v1, v2)).unwrap();
        entries.sort();

        if entries.len() == 1 {
            // Border edge
            e2v[ne_idx] = Vector2::new(v1, v2);
            e2f[ne_idx] = Vector2::new(entries[0].0, usize::MAX);
            f2e[entries[0].0][entries[0].1] = ne_idx;
        } else {
            // Shared edge — write f2e for every entry, use first pair as twin
            e2v[ne_idx] = Vector2::new(v1, v2);
            for &(tri_id, edge_idx) in &entries {
                f2e[tri_id][edge_idx] = ne_idx;
            }
            e2f[ne_idx] = Vector2::new(entries[0].0, entries[entries.len() - 1].0);
        }
    }

    for i in 0..e2f.len() {
        let fid = e2f[i][0];
        let mut flip = true;
        for j in 0..3 {
            if idx[fid][j] == e2v[i][0] && idx[fid][(j + 1) % 3] == e2v[i][1] {
                flip = false;
            }
        }

        if flip {
            let tmp = e2f[i][0];
            e2f[i][0] = e2f[i][1];
            e2f[i][1] = tmp;
        }
    }
    Ok(())
}

impl<T: BoolReal> Hmesh<T> {
    pub fn new(pos: &[Vector3<T>], idx: &[Vector3<usize>]) -> Result<Self, HmeshError> {
        let mut e2v = Default::default();
        let mut e2f = Default::default();
        let mut f2e = Default::default();
        edge_topology(pos, idx, &mut e2v, &mut e2f, &mut f2e)?;

        let nv = pos.len();
        let nf = idx.len();
        let ne = e2v.len();
        let nh = nf * 3;
        let np = 3;
        let mut v2h = vec![usize::MAX; nv];
        let mut e2h = vec![usize::MAX; ne];
        let mut f2h = vec![usize::MAX; nf];
        let mut next = vec![usize::MAX; nh];
        let mut prev = vec![usize::MAX; nh];
        let mut twin = vec![usize::MAX; nh];
        let mut head = vec![usize::MAX; nh];
        let mut tail = vec![usize::MAX; nh];
        let mut edge = vec![usize::MAX; nh];
        let mut face = vec![usize::MAX; nh];

        for it in 0..nf {
            for ip in 0..np {
                let ih_bgn = it * np;
                let iv = idx[it][ip];
                let ie = f2e[it][ip];
                let ih = ih_bgn + ip;
                next[ih] = ih_bgn + (ip + 1) % np;
                prev[ih] = ih_bgn + (ip + np - 1) % np;
                head[ih] = idx[it][(ip + 1) % np];
                tail[ih] = iv;
                edge[ih] = ie;
                face[ih] = it;
                if f2h[it] == usize::MAX {
                    f2h[it] = ih;
                }
                if v2h[iv] == usize::MAX {
                    v2h[iv] = ih;
                }
                if e2h[ie] == usize::MAX {
                    e2h[ie] = ih;
                } else {
                    twin[ih] = e2h[ie];
                    twin[e2h[ie]] = ih;
                }
            }
        }

        if twin.iter().any(|v| v == &usize::MAX) {
            return Err(HmeshError::ContainsBoundaryEdges);
        }

        let mut half = vec![];
        for i in 0..nh {
            half.push(i);
        }
        let mut vns = vec![Vector3::zeros(); nv];
        let mut fns = vec![Vector3::zeros(); nf];

        #[cfg(feature = "rayon")]
        fns.par_iter_mut().enumerate().for_each(|(i, n)| {
            let ih = f2h[i];
            let p2 = pos[head[ih]];
            let p1 = pos[tail[ih]];
            let p0 = pos[tail[prev[ih]]];
            let x = p2 - p1;
            let t = (p1 - p0) * -T::one();
            *n = x.cross(&t).normalize_or_zero();
        });

        #[cfg(not(feature = "rayon"))]
        for i in 0..nf {
            let ih = f2h[i];
            let p2 = pos[head[ih]];
            let p1 = pos[tail[ih]];
            let p0 = pos[tail[prev[ih]]];
            let x = p2 - p1;
            let t = (p1 - p0) * -T::one();
            fns[i] = x.cross(&t).normalize_or_zero();
        }

        for i in 0..nf {
            for j in 0..3 {
                let i_curr = idx[i][j];
                let v_prev = pos[idx[i][(j + 2) % 3]];
                let v_curr = pos[i_curr];
                let v_next = pos[idx[i][(j + 1) % 3]];
                let e_curr = (v_next - v_curr).normalize();
                let e_prev = (v_curr - v_prev).normalize();
                if e_curr.is_nan() || e_prev.is_nan() {
                    continue;
                }
                let dot = -e_prev.dot(&e_curr);
                let phi = if dot >= T::one() {
                    T::zero()
                } else if dot <= -T::one() {
                    T::PI
                } else {
                    dot.acos()
                };
                vns[i_curr] += fns[i] * phi;
            }
        }

        #[cfg(feature = "rayon")]
        vns.par_iter_mut().for_each(|n| *n = n.normalize_or_zero());

        #[cfg(not(feature = "rayon"))]
        for n in &mut vns {
            *n = n.normalize_or_zero();
        }

        Ok(Hmesh {
            nv,
            nf,
            nh,
            twin,
            head,
            tail,
            half,
            vns,
            fns,
        })
    }
}

#[derive(Debug, Error)]
pub enum HmeshError {
    #[error("empty pos matrix")]
    EmptyPositionMatrix,

    #[error("empty idx matrix")]
    EmptyIndexMatrix,

    #[error("Input mesh must not contain boundary edges.")]
    ContainsBoundaryEdges,
}
