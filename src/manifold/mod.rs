//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

pub mod bounds;
pub mod collider;
pub mod hmesh;

use super::hmesh::Hmesh;
use crate::collider::{morton_code, MortonCollider, K_NO_CODE};
use crate::manifold::bounds::Query;
use crate::common::BoolReal;
use crate::manifold::hmesh::HmeshError;
use crate::{next_of, HalfEdge, HalfEdgeId};
use bounds::BBox;
use fxhash::FxBuildHasher;
use nalgebra::{Matrix4, Point3, Rotation3, Vector3};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use thiserror::Error;

#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Debug)]
pub struct Manifold<T: BoolReal = f64> {
    pub(crate) positions: Vec<Vector3<T>>,
    pub(crate) halfedges: Vec<HalfEdge>,
    pub(crate) vertex_count: usize,
    pub(crate) face_count: usize,
    pub(crate) halfedge_count: usize,
    pub(crate) spatial_tol: T,
    pub(crate) snap_tol: T,
    pub(crate) bounding_box: BBox<T>,
    pub(crate) face_normals: Vec<Vector3<T>>,
    pub(crate) vert_normals: Vec<Vector3<T>>,
    pub(crate) original_idx: Vec<usize>,
    pub(crate) collider: MortonCollider<T>,
    pub(crate) coplanar: Vec<i32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Triangle<T> {
    pub positions: [Vector3<T>; 3],
    pub normal: Vector3<T>,
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
            match hash.entry(k) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    rmap[i] = *e.get();
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    let n = weld.len();
                    weld.push(v);
                    e.insert(n);
                    rmap[i] = n;
                }
            }
        }

        // remove collapsed triangles
        let idx = idx
            .chunks(3)
            .map(|i| Vector3::new(rmap[i[0]], rmap[i[1]], rmap[i[2]]))
            .filter(|&is| is.x != is.y && is.y != is.z && is.z != is.x)
            .collect::<Vec<_>>();

        Self::new_from_raw(weld, idx, None, None)
    }

   pub(crate) fn new_from_raw(
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
            .map(|&i| HalfEdge::new(hm.tail[i], hm.head[i], hm.twin[i]))
            .collect::<Vec<_>>();

        let mut e = T::K_PRECISION * bb.scale();
        e = if e.is_finite() { e } else { -T::one() };
        let eps = if let Some(e_) = eps { e_ } else { e };
        let tol = if let Some(t_) = tol { t_ } else { e };
        let collider = MortonCollider::new(&f_bb, &f_mt);
        let coplanar = compute_coplanar_idx(&ps, &hm.fns, &hs, eps);

        let mfd = Manifold {
            vertex_count: hm.nv,
            face_count: hm.nf,
            halfedge_count: hm.nh,
            positions: ps,
            halfedges: hs,
            bounding_box: bb,
            vert_normals: hm.vns,
            face_normals: hm.fns,
            original_idx: vec![],
            spatial_tol: eps,
            snap_tol: tol,
            collider,
            coplanar,
        };

        if !mfd.is_manifold() {
            return Err(ManifoldError::InputNotManifold);
        }
        Ok(mfd)
    }

    pub fn get_indices(&self) -> Vec<Vector3<usize>> {
        self.halfedges
            .chunks(3)
            .map(|cs| Vector3::new(usize::from(cs[0].tail), usize::from(cs[1].tail), usize::from(cs[2].tail)))
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
        self.spatial_tol = e;
        self.snap_tol = self.snap_tol.max(t);
    }

    pub fn is_manifold(&self) -> bool {
        self.halfedges.iter().enumerate().all(|(i, h)| {
            if h.tail().is_none() || h.head().is_none() {
                return true;
            }
match h.pair() {
                None => false,
                Some(pair) => {
                    let mut good = true;
                   good &= self.halfedges[pair as usize].pair() == Some(i as u32);
                    good &= u32::from(h.tail) == self.halfedges[pair as usize].head().unwrap();
                    good &= u32::from(h.head) == self.halfedges[pair as usize].tail().unwrap();
                    good
                }
            }
        })
    }

    pub fn transform(&self, transformation: Matrix4<T>) -> Result<Manifold<T>, ManifoldError> {
        let p = self
            .positions
            .iter()
            .map(|p| {
                transformation
                    .transform_point(&Point3 { coords: *p })
                    .coords
            })
            .collect();
        Manifold::new_from_raw(p, self.get_indices(), None, None)
    }

    pub fn translate(&self, x: T, y: T, z: T) -> Result<Manifold<T>, ManifoldError> {
        let t = Vector3::new(x, y, z);
        let p = self.positions.iter().map(|p| *p + t).collect();
        Manifold::new_from_raw(p, self.get_indices(), None, None)
    }

    pub fn rotate(&self, x: T, y: T, z: T) -> Result<Manifold<T>, ManifoldError> {
        let r = Rotation3::from_euler_angles(x, y, z);
        let p = self.positions.iter().map(|p| r.transform_vector(p)).collect();
        Manifold::new_from_raw(p, self.get_indices(), None, None)
    }

    pub fn scale(&self, x: T, y: T, z: T) -> Result<Manifold<T>, ManifoldError> {
        let p = self
            .positions
            .iter()
            .map(|p| Vector3::new(p.x * x, p.y * y, p.z * z))
            .collect();
        Manifold::new_from_raw(p, self.get_indices(), None, None)
    }

    #[inline]
    pub fn pos_at(&self, vid: usize) -> Vector3<T> { self.positions[vid] }

    #[inline]
    pub fn face_vertex_ids(&self, fid: usize) -> [usize; 3] {
        [
            usize::from(self.halfedges[3 * fid].tail),
            usize::from(self.halfedges[3 * fid + 1].tail),
            usize::from(self.halfedges[3 * fid + 2].tail),
        ]
    }

    #[inline]
    pub fn face_positions(&self, fid: usize) -> [Vector3<T>; 3] {
        [
            self.positions[usize::from(self.halfedges[3 * fid].tail)],
            self.positions[usize::from(self.halfedges[3 * fid + 1].tail)],
            self.positions[usize::from(self.halfedges[3 * fid + 2].tail)],
        ]
    }

    #[inline]
    pub fn halfedge_at(&self, hid: usize) -> &HalfEdge { &self.halfedges[hid] }

    pub fn triangles(&self) -> impl Iterator<Item = Triangle<T>> + '_ {
        self.halfedges
            .chunks(3)
            .zip(self.face_normals.iter().cloned())
            .map(|(h, n)| {
                Triangle {
                    positions: [
                        self.positions[usize::from(h[0].tail)],
                        self.positions[usize::from(h[1].tail)],
                        self.positions[usize::from(h[2].tail)],
                    ],
                    normal: n,
                }
            })
    }

    #[inline]
    pub fn vertex_count(&self) -> usize { self.vertex_count }

    #[inline]
    pub fn face_count(&self) -> usize { self.face_count }

    #[inline]
    pub fn halfedge_count(&self) -> usize { self.halfedge_count }

    #[inline]
    pub fn positions(&self) -> &[Vector3<T>] { &self.positions }

    #[inline]
    pub fn halfedges(&self) -> &[HalfEdge] { &self.halfedges }

    /// Compare two manifolds for geometric equality within an epsilon tolerance.
    ///
    /// Compares geometry semantically — positions, faces (by vertex set), and normals
    /// are matched regardless of array indexing order.
    /// Skips derived fields (bounding_box, tolerances, collider, coplanar, original_idx).
    pub fn approx_eq(&self, other: &Self, eps: T) -> bool {
        if self.vertex_count != other.vertex_count
            || self.face_count != other.face_count
            || self.halfedge_count != other.halfedge_count
        {
            return false;
        }

        if self.positions.len() != other.positions.len()
            || self.face_normals.len() != other.face_normals.len()
            || self.vert_normals.len() != other.vert_normals.len()
        {
            return false;
        }

        let n = self.positions.len();
        let nf = self.face_normals.len();

        // Sort position indices by coordinate tuple to build canonical mapping.
        let mut self_idx: Vec<usize> = (0..n).collect();
        self_idx.sort_by(|&a, &b| {
            let pa = &self.positions[a];
            let pb = &self.positions[b];
            pa.x.partial_cmp(&pb.x)
                .unwrap_or(Ordering::Equal)
                .then_with(|| pa.y.partial_cmp(&pb.y).unwrap_or(Ordering::Equal))
                .then_with(|| pa.z.partial_cmp(&pb.z).unwrap_or(Ordering::Equal))
        });

        let mut other_idx: Vec<usize> = (0..n).collect();
        other_idx.sort_by(|&a, &b| {
            let pa = &other.positions[a];
            let pb = &other.positions[b];
            pa.x.partial_cmp(&pb.x)
                .unwrap_or(Ordering::Equal)
                .then_with(|| pa.y.partial_cmp(&pb.y).unwrap_or(Ordering::Equal))
                .then_with(|| pa.z.partial_cmp(&pb.z).unwrap_or(Ordering::Equal))
        });

        // Verify positions match after sorting and build mapping.
        let mut map = vec![0usize; n];
        for i in 0..n {
            let a = &self.positions[self_idx[i]];
            let b = &other.positions[other_idx[i]];
            if (a.x - b.x).abs() > eps
                || (a.y - b.y).abs() > eps
                || (a.z - b.z).abs() > eps
            {
                return false;
            }
            map[self_idx[i]] = other_idx[i];
        }

        // Collect faces as sorted vertex triplets with their normals for canonical comparison.
        let mut self_faces: Vec<([usize; 3], Vector3<T>)> = Vec::with_capacity(nf);
        let mut other_faces: Vec<([usize; 3], Vector3<T>)> = Vec::with_capacity(nf);

        for fid in 0..nf {
            let mut si = [
                map[usize::from(self.halfedges[3 * fid].tail)],
                map[usize::from(self.halfedges[3 * fid + 1].tail)],
                map[usize::from(self.halfedges[3 * fid + 2].tail)],
            ];
            si.sort();
            self_faces.push((si, self.face_normals[fid]));

            let mut oi = [
                map[usize::from(other.halfedges[3 * fid].tail)],
                map[usize::from(other.halfedges[3 * fid + 1].tail)],
                map[usize::from(other.halfedges[3 * fid + 2].tail)],
            ];
            oi.sort();
            other_faces.push((oi, other.face_normals[fid]));
        }

        self_faces.sort_by(|a, b| a.0.cmp(&b.0));
        other_faces.sort_by(|a, b| a.0.cmp(&b.0));

        for i in 0..nf {
            if self_faces[i].0 != other_faces[i].0 {
                return false;
            }
        }

        // Compare face normals at the canonical face positions.
        for i in 0..nf {
            let n1 = &self_faces[i].1;
            let n2 = &other_faces[i].1;
            if (n1.x - n2.x).abs() > eps
                || (n1.y - n2.y).abs() > eps
                || (n1.z - n2.z).abs() > eps
            {
                return false;
            }
        }

        // Compare vertex normals using the position-based mapping.
        for i in 0..n {
            let a = &self.vert_normals[self_idx[i]];
            let b = &other.vert_normals[other_idx[i]];
            if (a.x - b.x).abs() > eps
                || (a.y - b.y).abs() > eps
                || (a.z - b.z).abs() > eps
            {
                return false;
            }
        }

        true
    }

    pub fn cleanup(&mut self) {
        cleanup_unused_verts_impl(&mut self.positions, &mut self.halfedges);
        self.vertex_count = self.positions.len();
        self.face_count = self.halfedges.len() / 3;
        self.halfedge_count = self.halfedges.len();
    }

    pub fn project_xy(&self) -> Result<geo::MultiPolygon<T>, crate::ProjectionError>
    where T: geo::bool_ops::BoolOpsNum {
        compute_projection_impl(self)
    }

    pub fn slice(&self, height: T) -> Result<geo::MultiPolygon<T>, crate::SliceError>
    where T: geo::bool_ops::BoolOpsNum {
        compute_slice_impl(self, height)
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
    map.sort_by(|&a, &b| face_morton[a].cmp(&face_morton[b]).then_with(|| a.cmp(&b)));
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
    hs: &[HalfEdge],
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
            let p0 = ps[usize::from(hs[i].tail)];
            let p1 = ps[usize::from(hs[i].head)];
            let p2 = ps[usize::from(hs[i + 1].head)];
            (p1 - p0).cross(&(p2 - p0)).norm_squared()
        };
        priority.push((area, t));
    }

    let mut interior = vec![];
    for (_area, t) in priority.iter() {
        if res[*t] != -1 {
            continue;
        }
        res[*t] = *t as i32;

        let i = t * 3;
        let p = ps[usize::from(hs[i].tail)];
        let n = ns[*t];

        interior.clear();
        interior.extend_from_slice(&[i, i + 1, i + 2]);

        while let Some(hi) = interior.pop() {
            let h1 = HalfEdgeId::from(usize::from(hs[hi].pair)).next_in_face();
            let t1 = HalfEdgeId::from(h1).face_id();

            if res[t1] != -1 {
                continue;
            }

            if (ps[usize::from(hs[h1].head)] - p).dot(&n).abs() < tol {
                res[t1] = *t as i32;
                if interior.last().copied() == Some(usize::from(hs[h1].pair)) {
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

pub(crate) fn cleanup_unused_verts_impl<T: BoolReal>(ps: &mut Vec<Vector3<T>>, hs: &mut Vec<HalfEdge>) {
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
        h.tail = crate::VertexId::from(old2new[h.tail().unwrap() as usize]);
        h.head = crate::VertexId::from(old2new[h.head().unwrap() as usize]);
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

#[deprecated(since = "0.1.10", note = "Use `Manifold::cleanup()` instead")]
pub fn cleanup_unused_verts<T: BoolReal>(ps: &mut Vec<Vector3<T>>, hs: &mut Vec<HalfEdge>) {
    cleanup_unused_verts_impl(ps, hs);
}

pub fn compute_projection_impl<T: BoolReal + geo::bool_ops::BoolOpsNum>(manifold: &Manifold<T>) -> Result<geo::MultiPolygon<T>, crate::ProjectionError> {
    let mut edge_ids: std::collections::BTreeMap<usize, std::collections::VecDeque<usize>> = std::collections::BTreeMap::new();

    for (edge_id, edge) in manifold.halfedges.iter().enumerate() {
        if edge_id > usize::from(edge.pair_id()) {
            continue;
        }

        let pair_he = usize::from(edge.pair_id());
        let face_a = edge_id / 3;
        let face_b = HalfEdgeId::from(pair_he).face_id();
        let na_z = manifold.face_normals[face_a].z;
        let nb_z = manifold.face_normals[face_b].z;

        let na_up = na_z.to_f64().is_finite() && na_z > T::zero();
        let nb_up = nb_z.to_f64().is_finite() && nb_z > T::zero();

        if na_up != nb_up {
            if nb_up {
                edge_ids.entry(usize::from(manifold.halfedges[pair_he].tail)).or_default().push_front(pair_he);
            } else {
                edge_ids.entry(usize::from(edge.tail)).or_default().push_front(edge_id);
            }
        }
    }

    let mut polygons: Vec<_> = Vec::new();

    loop {
        let first_edge_id = match edge_ids.first_key_value() {
            Some((&first_key, queue)) => {
                let value = queue.back().copied();
                let q = edge_ids.get_mut(&first_key).unwrap();
                q.pop_back();
                if q.is_empty() {
                    edge_ids.remove(&first_key);
                }
                value.unwrap()
            }
            None => break,
        };
        let mut current_edge_id = first_edge_id;
        let first_vertex = usize::from(manifold.halfedges[first_edge_id].tail);
        let first_point = manifold.positions[usize::from(manifold.halfedges[first_edge_id].head)];
        let mut line_string = vec![geo::Coord { x: first_point.x, y: first_point.y }];

        loop {
            if usize::from(manifold.halfedges[current_edge_id].head) == first_vertex {
                break;
            }

            let next_tail = usize::from(manifold.halfedges[current_edge_id].head);
            if let Some(queue) = edge_ids.get_mut(&next_tail) {
                if let Some(next_edge_id) = queue.pop_back() {
                    if queue.is_empty() {
                        edge_ids.remove(&next_tail);
                    }
                    current_edge_id = next_edge_id;
                } else {
                    break;
                }
            } else {
                break;
            }

            let point = manifold.positions[usize::from(manifold.halfedges[current_edge_id].head)];
            line_string.push(geo::Coord { x: point.x, y: point.y });
        }

        let mut line_string = geo::LineString(line_string);
        line_string.close();
        polygons.push(geo::Polygon::new(line_string, vec![]));
    }

    if polygons.is_empty() {
        Err(crate::ProjectionError::NoPolygons)
    } else {
        let polygon = geo::unary_union(&polygons);
        Ok(polygon)
    }
}

pub fn compute_slice_impl<T: BoolReal + geo::bool_ops::BoolOpsNum>(manifold: &Manifold<T>, height: T) -> Result<geo::MultiPolygon<T>, crate::SliceError> {
    let mut bounding_box = manifold.bounding_box.clone();
    bounding_box.min.z = height;
    bounding_box.max.z = height;
    bounding_box.id = Some(0);

    let mut triangle_ids = std::collections::BTreeSet::new();

    manifold
        .collider
        .collision([Query::Bb(bounding_box)], &mut |_query_id, triangle_id| {
            let z_points = [0, 1, 2]
                .into_iter()
                .map(|j| manifold.positions[usize::from(manifold.halfedges[3 * triangle_id + j].tail)].z);

            let min = z_points
                .clone()
                .min_by(|a: &T, b: &T| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
            let max = z_points.max_by(|a: &T, b: &T| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));

            if let (Some(min), Some(max)) = (min, max) && min <= height && max > height {
                triangle_ids.insert(triangle_id);
            }
        });

    fn next3(j: usize) -> usize {
        (j + 1) % 3
    }

    let mut polygons = Vec::new();

    while !triangle_ids.is_empty() {
        let start_triangle_id = *triangle_ids.first().ok_or(crate::SliceError::NoPolygons)?;

        let mut vertex_index = 0;
        for j in [0, 1, 2] {
            if manifold.positions[usize::from(manifold.halfedges[3 * start_triangle_id + j].tail)].z > height &&
                manifold.positions[usize::from(manifold.halfedges[3 * start_triangle_id + next3(j)].tail)].z <= height {
                vertex_index = next3(j);
                break;
            }
        }

        let mut line_string = Vec::new();
        let mut current_triangle_id = start_triangle_id;
        loop {
            triangle_ids.remove(&current_triangle_id);

            if manifold.positions[usize::from(manifold.halfedges[3 * current_triangle_id + vertex_index].head)].z <= height {
                vertex_index = next3(vertex_index);
            }

            let up = &manifold.halfedges[3 * current_triangle_id + vertex_index];
            let below = manifold.positions[usize::from(up.tail)];
            let above = manifold.positions[usize::from(up.head)];
            let a = (height - below.z) / (above.z - below.z);
            let point = below.lerp(&above, a);
            line_string.push(geo::Coord { x: point.x, y: point.y });

            let pair = usize::from(up.pair);
            current_triangle_id = HalfEdgeId::from(pair).face_id();
            vertex_index = next3(HalfEdgeId::from(pair).edge_index());

            if current_triangle_id == start_triangle_id {
                break;
            }
        }

        let mut line_string = geo::LineString(line_string);
        line_string.close();
        polygons.push(geo::Polygon::new(line_string, vec![]));
    }

    let polygon = geo::unary_union(&polygons);

    if polygons.is_empty() {
        Err(crate::SliceError::NoPolygons)
    } else {
        Ok(polygon)
    }
}
