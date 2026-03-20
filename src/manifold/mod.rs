//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

pub mod bounds;
pub mod collider;
pub mod hmesh;

use super::hmesh::Hmesh;
use crate::collider::{morton_code, MortonCollider, K_NO_CODE};
use crate::common::BoolReal;
use crate::manifold::hmesh::HmeshError;
use crate::{next_of, Half};
use bounds::BBox;
use fxhash::FxBuildHasher;
use nalgebra::{Matrix4, Point3, Rotation3, Vector3};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use thiserror::Error;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct Manifold<T: BoolReal = f64> {
    pub ps: Vec<Vector3<T>>,           // positions
    pub hs: Vec<Half>,                 // halfedges
    pub nv: usize,                     // number of vertices
    pub nf: usize,                     // number of faces
    pub nh: usize,                     // number of halfedges
    pub eps: T,                        // epsilon
    pub tol: T,                        // tolerance
    pub bounding_box: BBox<T>,         //
    pub face_normals: Vec<Vector3<T>>, //
    pub vert_normals: Vec<Vector3<T>>, //
    pub original_idx: Vec<usize>,      //
    pub collider: MortonCollider<T>,   //
    pub coplanar: Vec<i32>,            // indices of coplanar faces
}

impl<T: BoolReal> Manifold<T> {
    pub fn new(pos: &[T], idx: &[usize]) -> Result<Self, ManifoldError> {
        if !pos.len().is_multiple_of(3) {
            return Err(ManifoldError::PositionArrayNotMultipleOf3);
        }
        if !idx.len().is_multiple_of(3) {
            return Err(ManifoldError::IndexArrayNotMultipleOf3);
        }

        // dedup vertices
        let mut hash = HashMap::with_capacity_and_hasher(pos.len() / 3, FxBuildHasher::default());
        let mut weld = Vec::with_capacity(pos.len() / 3);
        let mut rmap = vec![0; pos.len()];

        for (i, p) in pos.chunks(3).enumerate() {
            let v = Vector3::new(p[0], p[1], p[2]);
            let k = (v.x.to_bits(), v.y.to_bits(), v.z.to_bits());
            if let Some(&w) = hash.get(&k) {
                rmap[i] = w;
            } else {
                let n = weld.len();
                weld.push(v);
                hash.insert(k, n);
                rmap[i] = n;
            }
        }

        // remove collapsed triangles
        let idx = idx
            .chunks(3)
            .map(|i| Vector3::new(rmap[i[0]], rmap[i[1]], rmap[i[2]]))
            .filter(|&is| is.x != is.y && is.y != is.z && is.z != is.x)
            .collect::<Vec<_>>();

        Self::new_impl(weld, idx, None, None)
    }

    pub fn new_impl(
        ps: Vec<Vector3<T>>,
        idx: Vec<Vector3<usize>>,
        eps: Option<T>,
        tol: Option<T>,
    ) -> Result<Self, ManifoldError> {
        let bb = BBox::new(None, &ps);
        let (mut f_bb, mut f_mt) = compute_face_morton(&ps, &idx, &bb);
        let hm = sort_faces(&ps, &idx, &mut f_bb, &mut f_mt)?;
        let hs = hm
            .half
            .iter()
            .map(|&i| Half::new(hm.tail[i], hm.head[i], hm.twin[i]))
            .collect::<Vec<_>>();

        let mut e = T::K_PRECISION * bb.scale();
        e = if e.is_finite() { e } else { -T::one() };
        let eps = if let Some(e_) = eps { e_ } else { e };
        let tol = if let Some(t_) = tol { t_ } else { e };
        let collider = MortonCollider::new(&f_bb, &f_mt);
        let coplanar = compute_coplanar_idx(&ps, &hm.fns, &hs, eps);

        let mfd = Manifold {
            nv: hm.nv,
            nf: hm.nf,
            nh: hm.nh,
            ps,
            hs,
            bounding_box: bb,
            vert_normals: hm.vns,
            face_normals: hm.fns,
            original_idx: vec![],
            eps,
            tol,
            collider,
            coplanar,
        };

        if !mfd.is_manifold() {
            return Err(ManifoldError::InputNotManifold);
        }
        Ok(mfd)
    }

    pub fn get_indices(&self) -> Vec<Vector3<usize>> {
        self.hs
            .chunks(3)
            .map(|cs| Vector3::new(cs[0].tail, cs[1].tail, cs[2].tail))
            .collect()
    }

    pub fn set_epsilon(&mut self, min_epsilon: T, use_single: bool) {
        let scl = self.bounding_box.scale();
        let mut e = min_epsilon.max(T::K_PRECISION * scl);
        e = if e.is_finite() { e } else { -T::one() };
        let t = if use_single {
            e.max(T::EPSILON * scl)
        } else {
            e
        };
        self.eps = e;
        self.tol = self.tol.max(t);
    }

    pub fn is_manifold(&self) -> bool {
        self.hs.iter().enumerate().all(|(i, h)| {
            if h.tail().is_none() || h.head().is_none() {
                return true;
            }
            match h.pair() {
                None => false,
                Some(pair) => {
                    let mut good = true;
                    good &= self.hs[pair].pair() == Some(i);
                    good &= h.tail != h.head;
                    good &= h.tail == self.hs[pair].head;
                    good &= h.head == self.hs[pair].tail;
                    good
                }
            }
        })
    }

    pub fn transform(&self, transformation: Matrix4<T>) -> Result<Manifold<T>, ManifoldError> {
        let p = self
            .ps
            .iter()
            .map(|p| {
                transformation
                    .transform_point(&Point3 { coords: *p })
                    .coords
            })
            .collect();
        Manifold::new_impl(p, self.get_indices(), None, None)
    }

    pub fn translate(&self, x: T, y: T, z: T) -> Result<Manifold<T>, ManifoldError> {
        let t = Vector3::new(x, y, z);
        let p = self.ps.iter().map(|p| *p + t).collect();
        Manifold::new_impl(p, self.get_indices(), None, None)
    }

    pub fn rotate(&self, x: T, y: T, z: T) -> Result<Manifold<T>, ManifoldError> {
        let r = Rotation3::from_euler_angles(x, y, z);
        let p = self.ps.iter().map(|p| r.transform_vector(p)).collect();
        Manifold::new_impl(p, self.get_indices(), None, None)
    }

    pub fn scale(&self, x: T, y: T, z: T) -> Result<Manifold<T>, ManifoldError> {
        let p = self
            .ps
            .iter()
            .map(|p| Vector3::new(p.x * x, p.y * y, p.z * z))
            .collect();
        Manifold::new_impl(p, self.get_indices(), None, None)
    }
}

#[derive(Debug, Error)]
pub enum ManifoldError {
    #[error("The input mesh is not manifold")]
    InputNotManifold,

    #[error("pos must be a multiple of 3")]
    PositionArrayNotMultipleOf3,

    #[error("idx must be a multiple of 3")]
    IndexArrayNotMultipleOf3,

    #[error("Failed to construct Hmesh: {0:?}")]
    Hmesh(#[from] HmeshError),
}

fn compute_face_morton<T: BoolReal>(
    pos: &[Vector3<T>],
    idx: &[Vector3<usize>],
    bb: &BBox<T>,
) -> (Vec<BBox<T>>, Vec<u32>) {
    let n = idx.len();
    let mut bbs = vec![BBox::default(); n];
    let mut mts = vec![0; n];

    #[cfg(feature = "rayon")]
    {
        bbs.par_iter_mut()
            .zip(mts.par_iter_mut())
            .zip(idx.par_iter())
            .for_each(|((bb_, mt_), f)| {
                let p0 = pos[f.x];
                let p1 = pos[f.y];
                let p2 = pos[f.z];
                bb_.union(&p0);
                bb_.union(&p1);
                bb_.union(&p2);
                *mt_ = morton_code(&((p0 + p1 + p2) / T::cast_from_usize(3)), bb);
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for (i, f) in idx.iter().enumerate() {
            let p0 = pos[f.x];
            let p1 = pos[f.y];
            let p2 = pos[f.z];
            bbs[i].union(&p0);
            bbs[i].union(&p1);
            bbs[i].union(&p2);
            mts[i] = morton_code(&((p0 + p1 + p2) / T::cast_from_f64(3.0)), bb);
        }
    }

    (bbs, mts)
}

fn sort_faces<T: BoolReal>(
    pos: &[Vector3<T>],
    idx: &[Vector3<usize>],
    face_bboxes: &mut Vec<BBox<T>>,
    face_morton: &mut Vec<u32>,
) -> Result<Hmesh<T>, ManifoldError> {
    let mut map = (0..face_morton.len()).collect::<Vec<_>>();
    map.sort_by_key(|&i| face_morton[i]);
    *face_bboxes = map
        .iter()
        .map(|&i| face_bboxes[i].clone())
        .collect::<Vec<_>>();
    *face_morton = map.iter().map(|&i| face_morton[i]).collect::<Vec<_>>();

    let hmesh = Hmesh::new(pos, &map.iter().map(|&i| idx[i]).collect::<Vec<_>>())?;
    Ok(hmesh)
}

fn compute_coplanar_idx<T: BoolReal>(
    ps: &[Vector3<T>],
    ns: &[Vector3<T>],
    hs: &[Half],
    tol: T,
) -> Vec<i32> {
    let nt = hs.len() / 3;
    let mut priority = vec![];
    let mut res = vec![-1; nt];

    for t in 0..nt {
        let i = t * 3;
        let area = if hs[i].tail().is_none() {
            T::zero()
        } else {
            let p0 = ps[hs[i].tail];
            let p1 = ps[hs[i].head];
            let p2 = ps[hs[i + 1].head];
            (p1 - p0).cross(&(p2 - p0)).norm_squared()
        };
        priority.push((area, t));
    }

    priority.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut interior = vec![];
    for (_, t) in priority.iter() {
        if res[*t] != -1 {
            continue;
        }
        res[*t] = *t as i32;

        let i = t * 3;
        let p = ps[hs[i].tail];
        let n = ns[*t];

        interior.clear();
        interior.extend_from_slice(&[i, i + 1, i + 2]);

        while let Some(hi) = interior.pop() {
            let h1 = next_of(hs[hi].pair);
            let t1 = h1 / 3;

            if res[t1] != -1 {
                continue;
            }

            if (ps[hs[h1].head] - p).dot(&n).abs() < tol {
                res[t1] = *t as i32;
                if interior.last().copied() == Some(hs[h1].pair) {
                    interior.pop();
                } else {
                    interior.push(h1);
                }
                interior.push(next_of(h1));
            }
        }
    }
    res
}

pub fn cleanup_unused_verts<T: BoolReal>(ps: &mut Vec<Vector3<T>>, hs: &mut Vec<Half>) {
    let bb = BBox::new(None, ps);
    let mt = ps.iter().map(|p| morton_code(p, &bb)).collect::<Vec<_>>();

    let mut new2old = (0..ps.len()).collect::<Vec<_>>();
    let mut old2new = vec![0; ps.len()];
    new2old.sort_by_key(|&i| mt[i]);
    for (new, &old) in new2old.iter().enumerate() {
        old2new[old] = new;
    }

    // reindex verts
    for h in hs.iter_mut() {
        if h.pair().is_none() {
            continue;
        }
        h.tail = old2new[h.tail];
        h.head = old2new[h.head];
    }

    // truncate pos container
    let nv = new2old
        .iter()
        .position(|&v| mt[v] == K_NO_CODE)
        .unwrap_or(new2old.len());

    new2old.truncate(nv);

    *ps = new2old.iter().map(|&i| ps[i]).collect();
    *hs = hs.iter().filter(|h| h.pair().is_some()).cloned().collect();
}
