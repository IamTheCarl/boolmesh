//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use indexmap::IndexMap;
use nalgebra::Vector3;

use crate::OpType;
use crate::boolean03::Boolean03;
use crate::bounds::BBox;
use crate::common::BoolReal;
use crate::{HalfEdge, HalfEdgeId, Manifold, Tref, face_of};
use std::mem;

fn duplicate_verts<T: BoolReal>(
    inc: &[i32],
    vt_r: &[i32],
    ps_p: &[Vector3<T>],
    ps_r: &mut [Vector3<T>],
    vid: usize,
) {
    let n = inc[vid].abs() as usize;
    for i in 0..n {
        ps_r[vt_r[vid] as usize + i] = ps_p[vid];
    }
}

fn inclusive_scan(input: &[i32], output: &mut [i32], offset: i32) {
    if input.is_empty() || output.is_empty() {
        return;
    }
    let mut sum = offset;
    for (i, &v) in input.iter().enumerate() {
        sum += v;
        if i < output.len() {
            output[i] = sum;
        }
    }
}

fn exclusive_scan(input: &[i32], output: &mut [i32], offset: i32) {
    if input.is_empty() || output.is_empty() {
        return;
    }
    let mut sum = offset;
    output[0] = sum;
    for i in 1..input.len() {
        sum += input[i - 1];
        if i < output.len() {
            output[i] = sum;
        }
    }
}

fn size_output<T: BoolReal>(
    mp: &Manifold<T>,
    mq: &Manifold<T>,
    i03: &[i32],
    i30: &[i32],
    i12: &[i32],
    i21: &[i32],
    p1q2: &[[usize; 2]],
    p2q1: &[[usize; 2]],
    fns: &mut Vec<Vector3<T>>,
    inv: bool, // whether to invert mesh of q
) -> (Vec<i32>, Vec<i32>) {
    let mut side_p = vec![0; mp.face_count];
    let mut side_q = vec![0; mq.face_count];

    // equivalent to CountVerts
    for (i, h) in mp.halfedges.iter().enumerate() {
        side_p[face_of(i)] += i03[usize::from(h.tail)].abs();
    }
    for (i, h) in mq.halfedges.iter().enumerate() {
        side_q[face_of(i)] += i30[usize::from(h.tail)].abs();
    }

    // equivalent to CountNewVerts
    for i in 0..i12.len() {
        let hid0 = p1q2[i][0];
        let hid1 = mp.halfedges[hid0].pair;
        let inc = i12[i].abs();
        side_p[face_of(hid0)] += inc;
        side_p[face_of(usize::from(hid1))] += inc;
        side_q[p1q2[i][1]] += inc;
    }

    for i in 0..i21.len() {
        let hid0 = p2q1[i][1];
        let hid1 = mq.halfedges[hid0].pair;
        let inc = i21[i].abs();
        side_q[face_of(hid0)] += inc;
        side_q[face_of(usize::from(hid1))] += inc;
        side_p[p2q1[i][0]] += inc;
    }

    // a map from face_p and face_q to face_r
    let mut face_pq2r = vec![0; mp.face_count + mq.face_count + 1];
    let mut side_pq = Vec::with_capacity(side_p.len() + side_q.len());
    side_pq.extend_from_slice(&side_p);
    side_pq.extend_from_slice(&side_q);
    let keep_fs = side_pq
        .iter()
        .map(|&x| if x > 0 { 1 } else { 0 })
        .collect::<Vec<_>>();

    inclusive_scan(&keep_fs, &mut face_pq2r[1..], 0);
    let nf_r = *face_pq2r.last().unwrap() as usize;
    face_pq2r.truncate(mp.face_count + mq.face_count);
    fns.resize(nf_r, Vector3::zeros());

    let mut fid_r = 0;
    for (i, n) in mp.face_normals.iter().enumerate() {
        if side_p[i] > 0 {
            fns[fid_r] = *n;
            fid_r += 1;
        }
    }
    for (i, n) in mq.face_normals.iter().enumerate() {
        if side_q[i] > 0 {
            fns[fid_r] = *n * if inv { -T::one() } else { T::one() };
            fid_r += 1;
        }
    }

    let truncated = side_pq
        .iter()
        .filter(|s| **s > 0)
        .copied()
        .collect::<Vec<_>>();
    let mut ih_per_f = vec![0; truncated.len()];

    inclusive_scan(&truncated, &mut ih_per_f, 0);
    ih_per_f.insert(0, 0);

    (ih_per_f, face_pq2r)
}

// Sort of intermediate data store for halfedge creation
#[derive(Clone, Debug)]
struct EdgePt<T> {
    val: T,        // dot value of edge
    vid: usize,    // vertex id
    cid: usize,    // collision id
    is_tail: bool, //
}

fn add_new_edge_verts<T: BoolReal>(
    p1q2: &[[usize; 2]],
    i12: &[i32],
    v12_r: &[i32],
    hs_p: &[HalfEdge],
    pt_old: &mut IndexMap<usize, Vec<EdgePt<T>>>,
    pt_new: &mut IndexMap<(usize, usize), Vec<EdgePt<T>>>,
    fwd: bool,
    oft: usize,
) {
    for i in 0..p1q2.len() {
        let hid_p = p1q2[i][if fwd { 0 } else { 1 }];
        let fid_q = p1q2[i][if fwd { 1 } else { 0 }];
        let vid_r = v12_r[i] as usize;
        let inc = i12[i];
        let hid0 = hid_p;
        let hid1 = hs_p[hid_p].pair;
        let key_l = if fwd {
            (face_of(hid0), fid_q)
        } else {
            (fid_q, face_of(hid0))
        };
        let key_r = if fwd {
            (face_of(usize::from(hid1)), fid_q)
        } else {
            (fid_q, face_of(usize::from(hid1)))
        };
        let dir = inc < 0;
        pt_old.entry(hid_p).or_default();
        pt_new.entry(key_l).or_default();
        pt_new.entry(key_r).or_default();
        let dir0 = dir ^ !fwd;
        let dir1 = dir ^ fwd;
        let inc_ = inc.abs() as usize;
        for j in 0..inc_ {
            pt_old.get_mut(&hid_p).unwrap().push(EdgePt {
                val: T::zero(),
                vid: vid_r + j,
                cid: i + oft,
                is_tail: dir,
            });
        }
        for j in 0..inc_ {
            pt_new.get_mut(&key_r).unwrap().push(EdgePt {
                val: T::zero(),
                vid: vid_r + j,
                cid: i + oft,
                is_tail: dir0,
            });
        }
        for j in 0..inc_ {
            pt_new.get_mut(&key_l).unwrap().push(EdgePt {
                val: T::zero(),
                vid: vid_r + j,
                cid: i + oft,
                is_tail: dir1,
            });
        }
    }
}

// Creating a partial halfedges from a list of positions.
// It's very confusing, but it's not aiming to pair twins (pair is -1).
// It's more likely to say pairing sta-end vertex and make a halfedge
fn pair_up<T: BoolReal>(pts: &mut [EdgePt<T>]) -> Vec<HalfEdge> {
    assert_eq!(pts.len() % 2, 0);
    let nh = pts.len() / 2;
    let mid_idx = {
        let mut sta_idx = 0;
        let mut end_idx = pts.len();

        while sta_idx < end_idx {
            if pts[sta_idx].is_tail {
                sta_idx += 1;
            } else {
                end_idx -= 1;
                pts.swap(sta_idx, end_idx);
            }
        }
        sta_idx
    };

    let cmp = |a: &EdgePt<T>, b: &EdgePt<T>| {
        a.val
            .partial_cmp(&b.val)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cid.cmp(&b.cid))
    };
    pts[..mid_idx].sort_by(cmp);
    pts[mid_idx..].sort_by(cmp);

    let mut edges = Vec::with_capacity(nh);
    for i in 0..nh {
        edges.push(HalfEdge::new_without_pair(pts[i].vid, pts[i + nh].vid));
    }
    edges
}

fn append_partial_edges<T: BoolReal>(
    i03: &[i32],                                //
    hs_p: &[HalfEdge],                          // halfedges in mfd_p
    ps_p: &[Vector3<T>],                        //
    ps_r: &[Vector3<T>],                        // the vert pos of mfd_r, already fulfilled so far
    vid_p2r: &[i32],                            // map from vid in mfd_p to vid in mfd_r
    fid_p2r: &[i32],                            // map from fid in mfd_p to fid in mfd_r
    fwd: bool,                                  //
    hs_r: &mut [HalfEdge],                      // halfedge data of mfd_r, empty yet
    rs_r: &mut [Tref],                          // map from halfedge in mfd_r to triangle info
    pt_p: &mut IndexMap<usize, Vec<EdgePt<T>>>, //
    face_ptr_r: &mut [i32],                     //
    whole_flag: &mut [bool], // a flag to find out a halfedge from mfd_p is entirely usable in mfd_r
) {
    for (hid_p, pt) in pt_p {
        let hpos_p = pt;
        let h = &hs_p[*hid_p];
        whole_flag[*hid_p] = false;
        whole_flag[usize::from(h.pair)] = false;

        // assigning 0-1 value to hpos_p
        let dif = ps_p[usize::from(h.head)] - ps_p[usize::from(h.tail)];
        for p in hpos_p.iter_mut() {
            p.val = dif.dot(&ps_r[p.vid]);
        }

        let i_tail = i03[usize::from(h.tail)]; // mostly 0 or 1
        let i_head = i03[usize::from(h.head)]; // mostly 0 or 1
        let p_tail = ps_r[vid_p2r[usize::from(h.tail)] as usize];
        let p_head = ps_r[vid_p2r[usize::from(h.head)] as usize];

        for i in 0..i_tail.abs() as usize {
            hpos_p.push(EdgePt {
                val: p_tail.dot(&dif),
                vid: vid_p2r[usize::from(h.tail)] as usize + i,
                cid: usize::MAX,
                is_tail: i_tail > 0,
            });
        }
        for i in 0..i_head.abs() as usize {
            hpos_p.push(EdgePt {
                val: p_head.dot(&dif),
                vid: vid_p2r[usize::from(h.head)] as usize + i,
                cid: usize::MAX,
                is_tail: i_head < 0,
            });
        }

        let mut half_seq = pair_up(hpos_p);
        let fp_l = face_of(*hid_p);
        let fp_r = HalfEdgeId::from(usize::from(h.pair)).face_id();
        let fid_l = fid_p2r[fp_l] as usize;
        let fid_r = fid_p2r[fp_r] as usize;

        // Negative inclusion means the halfedges are reversed, which means our
        // reference is now to the head instead of the tail, which is one
        // position advanced CCW. This is only valid if this is a retained vert;
        // it will be ignored later if the vert is new.

        let fw_tri = Tref {
            mid: if fwd { 0 } else { 1 },
            fid: fp_l,
            ..Default::default()
        };
        let bk_tri = Tref {
            mid: if fwd { 0 } else { 1 },
            fid: fp_r,
            ..Default::default()
        };

        for h in half_seq.iter_mut() {
            let fw_edge = face_ptr_r[fid_l] as usize;
            let bk_edge = face_ptr_r[fid_r] as usize;
            face_ptr_r[fid_l] += 1;
            face_ptr_r[fid_r] += 1;
            hs_r[fw_edge] = HalfEdge::new(usize::from(h.tail), usize::from(h.head), bk_edge);
            hs_r[bk_edge] = HalfEdge::new(usize::from(h.head), usize::from(h.tail), fw_edge);
            rs_r[fw_edge] = fw_tri;
            rs_r[bk_edge] = bk_tri;
        }
    }
}

fn append_new_edges<T: BoolReal>(
    ps_r: &[Vector3<T>],    // the vert pos of mfd_r, already fulfilled so far
    fid_pq2r: &[i32],       //
    nf_p: usize,            // num of faces in mfd_p
    face_ptr_r: &mut [i32], //
    pt_new: &mut IndexMap<(usize, usize), Vec<EdgePt<T>>>, //
    hs_r: &mut [HalfEdge],  // the halfedge data of mfd_r, empty yet
    rs_r: &mut [Tref],      //
) {
    for ((fid_p, fid_q), pt_init) in pt_new.iter_mut() {
        let pt = pt_init;
        let mut bb = BBox::default();
        for p in pt.iter() {
            bb.union(&ps_r[p.vid]);
        }

        let d = bb.longest_dim();
        for p in pt.iter_mut() {
            p.val = ps_r[p.vid][d];
        }

        let mut half_seq = pair_up(pt);
        let fid_l = fid_pq2r[*fid_p] as usize;
        let fid_r = fid_pq2r[*fid_q + nf_p] as usize;
        let fw_ref = Tref {
            mid: 0,
            fid: *fid_p,
            ..Default::default()
        };
        let bk_ref = Tref {
            mid: 1,
            fid: *fid_q,
            ..Default::default()
        };

        for h in half_seq.iter_mut() {
            let fw_edge = face_ptr_r[fid_l] as usize;
            let bk_edge = face_ptr_r[fid_r] as usize;
            face_ptr_r[fid_l] += 1;
            face_ptr_r[fid_r] += 1;
            hs_r[fw_edge] = HalfEdge::new(usize::from(h.tail), usize::from(h.head), bk_edge);
            hs_r[bk_edge] = HalfEdge::new(usize::from(h.head), usize::from(h.tail), fw_edge);
            rs_r[fw_edge] = fw_ref;
            rs_r[bk_edge] = bk_ref;
        }
    }
}

fn append_whole_edges(
    i03: &[i32],
    half_p: &[HalfEdge],
    fid_p2r: &[i32],
    vid_p2r: &[i32],
    whole_flag: &[bool],
    fwd: bool,
    face_ptr_r: &mut [i32],
    hs_r: &mut [HalfEdge],
    rs_r: &mut [Tref],
) {
    for (i, hp) in half_p.iter().enumerate() {
        if !whole_flag[i] || !hp.is_forward() {
            continue;
        }

        let mut h = hp.clone();
        let inc = i03[usize::from(h.tail)];
        if inc == 0 {
            continue;
        }
        if inc < 0 {
            mem::swap(&mut h.tail, &mut h.head);
        }

        h.tail = crate::VertexId::from(vid_p2r[usize::from(h.tail)] as usize);
        h.head = crate::VertexId::from(vid_p2r[usize::from(h.head)] as usize);

        let fp_l = face_of(i);
        let fp_r = HalfEdgeId::from(usize::from(hp.pair)).face_id();
        let fid_l = fid_p2r[fp_l] as usize;
        let fid_r = fid_p2r[fp_r] as usize;
        let fw_ref = Tref {
            mid: if fwd { 0 } else { 1 },
            fid: fp_l,
            ..Default::default()
        };
        let bk_ref = Tref {
            mid: if fwd { 0 } else { 1 },
            fid: fp_r,
            ..Default::default()
        };

        for _ in 0..inc.abs() as usize {
            let fw_edge = face_ptr_r[fid_l] as usize;
            let bk_edge = face_ptr_r[fid_r] as usize;
            face_ptr_r[fid_l] += 1;
            face_ptr_r[fid_r] += 1;
            hs_r[fw_edge] = HalfEdge::new(usize::from(h.tail), usize::from(h.head), bk_edge);
            hs_r[bk_edge] = HalfEdge::new(usize::from(h.head), usize::from(h.tail), fw_edge);
            rs_r[fw_edge] = fw_ref;
            rs_r[bk_edge] = bk_ref;
            h.tail = crate::VertexId::from(usize::from(h.tail) + 1);
            h.head = crate::VertexId::from(usize::from(h.head) + 1);
        }
    }
}

pub struct Boolean45<T> {
    pub ps: Vec<Vector3<T>>,
    pub ns: Vec<Vector3<T>>,
    pub hs: Vec<HalfEdge>,
    pub rs: Vec<Tref>,
    pub hid_per_f: Vec<i32>,
    pub nv_from_p: usize,
    pub nv_from_q: usize,
}

pub fn boolean45<T: BoolReal>(
    mp: &Manifold<T>,
    mq: &Manifold<T>,
    b03: &Boolean03<T>,
    op: &OpType,
) -> Boolean45<T> {
    let c1 = if op == &OpType::Intersect { 0 } else { 1 };
    let c2 = if op == &OpType::Add { 1 } else { 0 };
    let c3 = if op == &OpType::Intersect { 1 } else { -1 };
    let i12: Vec<i32> = b03.x12.iter().map(|v| c3 * v).collect();
    let i21: Vec<i32> = b03.x21.iter().map(|v| c3 * v).collect();
    let i03: Vec<i32> = b03.w03.iter().map(|v| c1 + c3 * v).collect();
    let i30: Vec<i32> = b03.w30.iter().map(|v| c2 + c3 * v).collect();
    let mut nv = 0;
    let mut vid_p2r = vec![0; mp.vertex_count];
    let mut vid_q2r = vec![0; mq.vertex_count];
    let mut vid_12r = vec![0; b03.v12.len()];
    let mut vid_21r = vec![0; b03.v21.len()];

    exclusive_scan(
        &i03.iter().map(|i| i.abs()).collect::<Vec<_>>(),
        &mut vid_p2r,
        nv,
    );
    nv = (*vid_p2r.last().unwrap()).abs() + i03.last().unwrap().abs();
    let nv_rp = nv;

    exclusive_scan(
        &i30.iter().map(|i| i.abs()).collect::<Vec<_>>(),
        &mut vid_q2r,
        nv,
    );
    nv = (*vid_q2r.last().unwrap()).abs() + i30.last().unwrap().abs();
    let nv_rq = nv - nv_rp;

    if !b03.v12.is_empty() {
        exclusive_scan(
            &i12.iter().map(|i| i.abs()).collect::<Vec<_>>(),
            &mut vid_12r,
            nv,
        );
        nv = (*vid_12r.last().unwrap()).abs() + i12.last().unwrap().abs();
    }
    let nv_12 = nv - nv_rp - nv_rq;

    if !b03.v21.is_empty() {
        exclusive_scan(
            &i21.iter().map(|i| i.abs()).collect::<Vec<_>>(),
            &mut vid_21r,
            nv,
        );
        nv = (*vid_21r.last().unwrap()).abs() + i21.last().unwrap().abs();
    }
    let nv_21 = nv - nv_rp - nv_rq - nv_12;

    let mut ps_r = vec![Vector3::zeros(); nv as usize];

    for i in 0..mp.vertex_count {
        duplicate_verts(&i03, &vid_p2r, &mp.positions, &mut ps_r, i);
    }
    for i in 0..mq.vertex_count {
        duplicate_verts(&i30, &vid_q2r, &mq.positions, &mut ps_r, i);
    }
    for i in 0..nv_12 {
        duplicate_verts(&i12, &vid_12r, &b03.v12, &mut ps_r, i as usize);
    }
    for i in 0..nv_21 {
        duplicate_verts(&i21, &vid_21r, &b03.v21, &mut ps_r, i as usize);
    }

    let mut pt_p = IndexMap::with_capacity_and_hasher(b03.p1q2.len() * 2, Default::default());
    let mut pt_q = IndexMap::with_capacity_and_hasher(b03.p2q1.len() * 2, Default::default());
    let mut pt_new = IndexMap::with_capacity_and_hasher(b03.p1q2.len() * 2, Default::default());
    add_new_edge_verts(
        &b03.p1q2,
        &i12,
        &vid_12r,
        &mp.halfedges,
        &mut pt_p,
        &mut pt_new,
        true,
        0,
    );
    add_new_edge_verts(
        &b03.p2q1,
        &i21,
        &vid_21r,
        &mq.halfedges,
        &mut pt_q,
        &mut pt_new,
        false,
        b03.p1q2.len(),
    );

    let mut ns_r = vec![];
    let inv = op == &OpType::Subtract;
    let (hid_per_f, fid_pq2r) = size_output(
        mp, mq, &i03, &i30, &i12, &i21, &b03.p1q2, &b03.p2q1, &mut ns_r, inv,
    );

    let nh = *hid_per_f.last().unwrap() as usize;
    let mut face_ptr_r = hid_per_f.clone();
    let mut whole_flag_p = vec![true; mp.halfedge_count];
    let mut whole_flag_q = vec![true; mq.halfedge_count];
    let mut rs_r = vec![Tref::default(); nh];
    let mut hs_r = vec![HalfEdge::default(); nh];
    let fid_p2r = &fid_pq2r[0..mp.face_count];
    let fid_q2r = &fid_pq2r[mp.face_count..];

    append_partial_edges(
        &i03,
        &mp.halfedges,
        &mp.positions,
        &ps_r,
        &vid_p2r,
        fid_p2r,
        true,
        &mut hs_r,
        &mut rs_r,
        &mut pt_p,
        &mut face_ptr_r,
        &mut whole_flag_p,
    );
    append_partial_edges(
        &i30,
        &mq.halfedges,
        &mq.positions,
        &ps_r,
        &vid_q2r,
        fid_q2r,
        false,
        &mut hs_r,
        &mut rs_r,
        &mut pt_q,
        &mut face_ptr_r,
        &mut whole_flag_q,
    );

    append_new_edges(
        &ps_r,
        &fid_pq2r,
        mp.face_count,
        &mut face_ptr_r,
        &mut pt_new,
        &mut hs_r,
        &mut rs_r,
    );

    append_whole_edges(
        &i03,
        &mp.halfedges,
        fid_p2r,
        &vid_p2r,
        &whole_flag_p,
        true,
        &mut face_ptr_r,
        &mut hs_r,
        &mut rs_r,
    );
    append_whole_edges(
        &i30,
        &mq.halfedges,
        fid_q2r,
        &vid_q2r,
        &whole_flag_q,
        false,
        &mut face_ptr_r,
        &mut hs_r,
        &mut rs_r,
    );

    Boolean45 {
        ps: ps_r,
        ns: ns_r,
        hs: hs_r,
        rs: rs_r,
        hid_per_f,
        nv_from_p: nv_rp as usize,
        nv_from_q: nv_rq as usize,
    }
}
