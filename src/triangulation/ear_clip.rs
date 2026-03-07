//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use nalgebra::{Vector2, Vector3};

use super::flat_tree::{compute_flat_tree, compute_query_flat_tree, Rect};
use crate::common::BoolReal;
use crate::triangulation::Pt;
use crate::{det2x2, is_ccw_2d, safe_normalize};
use std::cell::RefCell;
use std::cmp::{Ordering, PartialEq};
use std::collections::BTreeSet;
use std::rc::{Rc, Weak};

#[derive(Clone, Debug)]
pub struct Ecvt<T> {
    pub idx: usize,                          // vert idx
    pub pos: Vector2<T>,                     // vert pos
    pub dir: Vector2<T>,                     // right dir
    pub ear: Option<Weak<RefCell<Ecvt<T>>>>, // itr to self, just needed for quick removal from the ear queue
    pub vl: Option<Weak<RefCell<Ecvt<T>>>>,
    pub vr: Option<Weak<RefCell<Ecvt<T>>>>,
    pub cost: T,
}

impl<T: BoolReal> Ecvt<T> {
    pub fn ptr_l(&self) -> EvPtr<T> {
        self.vl.as_ref().unwrap().upgrade().unwrap()
    }
    pub fn ptr_r(&self) -> EvPtr<T> {
        self.vr.as_ref().unwrap().upgrade().unwrap()
    }
    pub fn idx_l(&self) -> usize {
        self.ptr_l().borrow().idx
    }
    pub fn idx_r(&self) -> usize {
        self.ptr_r().borrow().idx
    }
    pub fn pos_l(&self) -> Vector2<T> {
        self.ptr_l().borrow().pos
    }
    pub fn pos_r(&self) -> Vector2<T> {
        self.ptr_r().borrow().pos
    }
    pub fn dir_l(&self) -> Vector2<T> {
        self.ptr_l().borrow().dir
    }
    pub fn dir_r(&self) -> Vector2<T> {
        self.ptr_r().borrow().dir
    }
    pub fn ptr_l_of_r(&self) -> EvPtr<T> {
        self.ptr_r().borrow().ptr_l()
    }
    pub fn ptr_r_of_l(&self) -> EvPtr<T> {
        self.ptr_l().borrow().ptr_r()
    }

    pub fn new(idx: usize, pos: Vector2<T>) -> Self {
        Self {
            idx,
            pos,
            dir: Vector2::zeros(),
            ear: None,
            vl: None,
            vr: None,
            cost: T::zero(),
        }
    }

    // Shorter than half of epsilon, to be conservative so that it doesn't
    // cause CW triangles that exceed epsilon due to rounding error.
    pub fn is_short(&self, eps: T) -> bool {
        (self.pos_r() - self.pos).norm_squared() * T::cast_from_usize(4) < eps.powi(2)
    }

    // Returns true if Vert is on inside the edge that goes from tail to tail->right.
    // This will walk the edges if necessary until a clear answer is found (beyond epsilon).
    // If toLeft is true, this Vert will walk its edges to the left. This should be chosen
    // so that the edges walk in the same general direction - tail always walks to the right.
    pub fn inside_edge(&self, pair: &EvPtr<T>, eps: T, to_left: bool) -> bool {
        let mut nl = self.ptr_r_of_l(); // next left
        let mut nr = pair.borrow().ptr_r(); // next right
        let mut ct = Rc::clone(pair); // center
        let mut lt = Rc::clone(pair); // last

        while !Rc::ptr_eq(&nl, &nr)
            && !Rc::ptr_eq(pair, &nr)
            && !Rc::ptr_eq(&nl, &(if to_left { self.ptr_r() } else { self.ptr_l() }))
        {
            let l2 = (nl.borrow().pos - ct.borrow().pos).norm_squared();
            let r2 = (nr.borrow().pos - ct.borrow().pos).norm_squared();

            if l2 <= eps.powi(2) {
                nl = if to_left {
                    nl.borrow().ptr_l()
                } else {
                    nl.borrow().ptr_r()
                };
                continue;
            }

            if r2 <= eps.powi(2) {
                nr = { nr.borrow().ptr_r() };
                continue;
            }

            let e = nr.borrow().pos - nl.borrow().pos;
            if e.norm_squared() <= eps.powi(2) {
                lt = Rc::clone(&ct);
                ct = Rc::clone(&nl);
                nl = if to_left {
                    nl.borrow().ptr_l()
                } else {
                    nl.borrow().ptr_r()
                };
                if Rc::ptr_eq(&nl, &nr) {
                    break;
                }
                nr = { nr.borrow().ptr_r() };
                continue;
            }

            let mut convex = is_ccw_2d(&nl.borrow().pos, &ct.borrow().pos, &nr.borrow().pos, eps);
            if !Rc::ptr_eq(&ct, &lt) {
                convex += is_ccw_2d(&lt.borrow().pos, &ct.borrow().pos, &nl.borrow().pos, eps)
                    + is_ccw_2d(&nr.borrow().pos, &ct.borrow().pos, &lt.borrow().pos, eps);
            }
            if convex != 0 {
                return convex > 0;
            }

            if l2 < r2 {
                ct = Rc::clone(&nl);
                nl = if to_left {
                    nl.borrow().ptr_l()
                } else {
                    nl.borrow().ptr_r()
                };
            } else {
                ct = Rc::clone(&nr);
                nr = { nr.borrow().ptr_r() };
            }
            lt = Rc::clone(&ct);
        }

        // The whole polygon is degenerate - consider this to be convex.
        true
    }

    // Returns true for convex or collinear ears.
    pub fn is_convex(&self, eps: T) -> bool {
        is_ccw_2d(&self.pos_l(), &self.pos, &self.pos_r(), eps) >= 0
    }

    // Subtly different from !IsConvex because IsConvex will return true for collinear
    // non-folded verts, while IsReflex will always check until actual certainty is determined.
    pub fn is_reflex(&self, eps: T) -> bool {
        !self
            .ptr_l()
            .borrow()
            .inside_edge(&self.ptr_r_of_l(), eps, true)
    }

    // Returns the x-value on this edge corresponding to the start.y value,
    // returning NAN if the edge does not cross the value from below to above,
    // right of start - all within an epsilon tolerance. If onTop != 0,
    // this restricts which end is allowed to terminate within the epsilon band.
    pub fn interpolate_y2x(&self, bgn: &Vector2<T>, on_top: i32, eps: T) -> Option<T> {
        if (self.pos.y - bgn.y).abs() <= eps {
            if self.pos_r().y <= bgn.y + eps || on_top == 1 {
                return None;
            }
            return Some(self.pos.x);
        }
        if self.pos.y < bgn.y - eps {
            if self.pos_r().y > bgn.y + eps {
                let aspect = (self.pos_r().x - self.pos.x) / (self.pos_r().y - self.pos.y);
                return Some(self.pos.x + (bgn.y - self.pos.y) * aspect);
            }

            if self.pos_r().y < bgn.y - eps || on_top == -1 {
                return None;
            }
            return Some(self.pos_r().x);
        }
        None
    }

    // This finds the cost of this vert relative to one of the two closed sides of the ear.
    // Points are valid even when they touch, so long as their edge goes to the outside.
    // No need to check the other side, since all verts are processed in the EarCost loop.
    pub fn signed_dist(&self, pair: &Ecvt<T>, unit: Vector2<T>, eps: T) -> T {
        let d = det2x2(&unit, &(pair.pos - self.pos));
        if d.abs() < eps {
            let dr = det2x2(&unit, &(pair.pos_r() - self.pos));
            let dl = det2x2(&unit, &(pair.pos_l() - self.pos));
            if dr.abs() > eps {
                return dr;
            }
            if dl.abs() > eps {
                return dl;
            }
        }
        d
    }

    // Find the cost of Vert v within this ear, where openSide is the unit
    // vector from Verts right to left - passed in for reuse.
    pub fn cost(&self, pair: &Ecvt<T>, open_side: &Vector2<T>, eps: T) -> T {
        let c0 = self.signed_dist(pair, self.dir, eps);
        let c1 = self.signed_dist(pair, self.dir_l(), eps);
        let co = det2x2(open_side, &(pair.pos - self.pos_r()));
        c0.min(c1).min(co)
    }

    // For verts outside the ear, apply a cost based on the Delaunay condition
    // to aid in prioritization and produce cleaner triangulations. This doesn't
    // affect robustness but may be adjusted to improve output.
    pub fn delaunay_cost(&self, diff: &Vector2<T>, scl: T, eps: T) -> T {
        -eps - scl * diff.norm_squared()
    }

    // This is the expensive part of the algorithm, checking this ear against
    // every Vert to ensure none are inside. The Collider brings the total
    // triangulator cost down from O(n^2) to O(nlogn) for most large polygons.
    // Think of a cost as vaguely a distance metric - 0 is right on the edge of being invalid.
    // Cost > epsilon is definitely invalid. cost < -epsilon is definitely valid, so all improvement
    // costs are designed to always give values < -epsilon so they will never affect validity.
    // The first totalCost is designed to give priority to sharper angles.
    // Any cost < (-1 - epsilon) has satisfied the Delaunay condition.
    pub fn ear_cost(&self, eps: T, collider: &IdxCollider<T>) -> T {
        let dif = self.pos_l() - self.pos_r();
        let len = dif.norm();
        let scl = if len > eps {
            T::cast_from_usize(4) / len.powi(2)
        } else {
            T::MAX
        };
        let center = (self.pos_l() + self.pos_r()) * T::cast_from_f64(0.5);
        let radius = len * T::cast_from_f64(0.5);
        let open_side = dif.normalize();

        let mut total = self.dir_l().dot(&self.dir) - T::one() - eps;
        if is_ccw_2d(&self.pos, &self.pos_l(), &self.pos_r(), eps) == 0 {
            return total;
        }

        let mut bb = Rect::new(
            &Vector2::new(center.x - radius, center.y - radius),
            &Vector2::new(center.x + radius, center.y + radius),
        );
        bb.union(self.pos);
        bb.min -= Vector2::new(eps, eps);
        bb.max += Vector2::new(eps, eps);

        compute_query_flat_tree(&collider.pts, &bb, |v| {
            let test = Rc::clone(&collider.rfs[v.idx]);
            if !clipped(&test)
                && test.borrow().idx != self.idx
                && test.borrow().idx != self.idx_l()
                && test.borrow().idx != self.idx_r()
            {
                let mut cost = self.cost(&test.borrow(), &open_side, eps);
                if cost < -eps {
                    cost = self.delaunay_cost(&(test.borrow().pos - center), scl, eps);
                }
                if cost > total {
                    total = cost;
                }
            }
        });
        total
    }
}

type EvPtr<T> = Rc<RefCell<Ecvt<T>>>;

// When an ear vert is clipped, its neighbors get linked, so they get unlinked
// from it, but it is still linked to them.
fn clipped<T: BoolReal>(v: &EvPtr<T>) -> bool {
    !Rc::ptr_eq(&v.borrow().ptr_l_of_r(), v)
}
fn folded<T: BoolReal>(v: &EvPtr<T>) -> bool {
    Rc::ptr_eq(&v.borrow().ptr_l(), &v.borrow().ptr_r())
}

impl<T: PartialEq> PartialEq for Ecvt<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl<T: PartialOrd> PartialOrd for Ecvt<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

#[derive(Clone)]
struct EvPtrMinCost<T>(EvPtr<T>);
#[derive(Clone)]
struct EvPtrMaxPosX<T>(EvPtr<T>, Rect<T>);

impl<T: BoolReal> Eq for EvPtrMinCost<T> {}
impl<T: BoolReal> PartialEq for EvPtrMinCost<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().cost == other.0.borrow().cost && Rc::ptr_eq(&self.0, &other.0)
    }
}
impl<T: BoolReal> PartialOrd for EvPtrMinCost<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: BoolReal> Ord for EvPtrMinCost<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .borrow()
            .cost
            .partial_cmp(&other.0.borrow().cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                let ptr1 = Rc::as_ptr(&self.0) as usize;
                let ptr2 = Rc::as_ptr(&other.0) as usize;
                ptr1.cmp(&ptr2)
            })
    }
}

impl<T: BoolReal> Eq for EvPtrMaxPosX<T> {}
impl<T: BoolReal> PartialEq for EvPtrMaxPosX<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().pos.x == other.0.borrow().pos.x && Rc::ptr_eq(&self.0, &other.0)
    }
}
impl<T: BoolReal> PartialOrd for EvPtrMaxPosX<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: BoolReal> Ord for EvPtrMaxPosX<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .borrow()
            .pos
            .x
            .partial_cmp(&self.0.borrow().pos.x)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                let ptr1 = Rc::as_ptr(&self.0) as usize;
                let ptr2 = Rc::as_ptr(&other.0) as usize;
                ptr1.cmp(&ptr2)
            })
    }
}

// Apply `func` to each unclipped vertex in a polygonal circular list starting at `v`.
fn do_loop<F, T>(v: &mut EvPtr<T>, mut func: F) -> Option<EvPtr<T>>
where
    T: BoolReal,
    F: FnMut(&mut EvPtr<T>),
{
    let mut w = Rc::clone(v);
    loop {
        if clipped(&w) {
            // Update first to an unclipped vert so we will return to it instead of infinite-loop
            *v = w.borrow().ptr_l_of_r();
            if !clipped(v) {
                w = Rc::clone(v);
                if folded(&w) {
                    return None;
                }
                func(&mut w);
            }
        } else {
            if folded(&w) {
                return None;
            }
            func(&mut w);
        }

        w = { w.borrow().ptr_r() };
        if Rc::ptr_eq(&w, v) {
            return Some(w);
        }
    }
}

/*
 * Ear-clipping triangulator based on David Eberly's approach from Geometric
 * Tools, but adjusted to handle epsilon-valid polygons, and including a
 * fallback that ensures manifold triangulation even for overlapping polygons.
 * This is reduced from an O(n^2) algorithm by means of our BVH Collider.
 *
 * The main adjustments for robustness involve clipping the sharpest ears first
 * (a known technique to get higher triangle quality), and doing an exhaustive
 * search to determine ear convexity exactly if the first geometric result is
 * within epsilon.
 */

pub struct IdxCollider<T: BoolReal> {
    pub pts: Vec<Pt<T>>,
    pub rfs: Vec<EvPtr<T>>,
}

pub struct EarClip<T> {
    polygon: Vec<EvPtr<T>>,
    simples: Vec<EvPtr<T>>, // contour + recursive ccw loops
    contour: Vec<EvPtr<T>>,
    queue: BTreeSet<EvPtrMinCost<T>>,
    holes: BTreeSet<EvPtrMaxPosX<T>>,
    tris: Vec<Vector3<usize>>,
    bbox: Rect<T>,
    eps: T,
}

impl<T: BoolReal> EarClip<T> {
    pub fn new(polys: &[Vec<Pt<T>>], eps: T) -> Self {
        let mut clip = Self {
            polygon: vec![],
            simples: vec![],
            contour: vec![],
            queue: BTreeSet::new(),
            holes: BTreeSet::new(),
            tris: vec![],
            bbox: Rect::default(),
            eps,
        };

        let mut inits = clip.initialize(polys);

        let keys = clip.polygon.to_vec();
        for v in keys {
            clip.clip_degenerate(&v);
        }
        for v in inits.iter_mut() {
            clip.find_start(v);
        }

        clip
    }

    pub fn triangulate(&mut self) -> Vec<Vector3<usize>> {
        let vs = self.holes.iter().cloned().collect::<Vec<_>>();
        for v in vs {
            self.cut_key_hole(&v);
        }
        let vs = self.simples.iter().map(Rc::clone).collect::<Vec<_>>();
        for mut v in vs {
            self.triangulate_poly(&mut v);
        }
        std::mem::take(&mut self.tris)
    }

    // This function and JoinPolygons are the only functions that affect
    // the circular list data structure. This helps ensure it remains circular.
    fn link(vl: &EvPtr<T>, vr: &EvPtr<T>) {
        let mut bl = vl.borrow_mut();
        let mut br = vr.borrow_mut();
        bl.vr = Some(Rc::downgrade(vr));
        br.vl = Some(Rc::downgrade(vl));
        bl.dir = safe_normalize(br.pos - bl.pos);
    }

    pub fn clip_ear(&mut self, ear: &EvPtr<T>) {
        Self::link(&ear.borrow().ptr_l(), &ear.borrow().ptr_r());
        let i = ear.borrow().idx;
        let il = ear.borrow().idx_l();
        let ir = ear.borrow().idx_r();
        if il != i && ir != i && il != ir {
            self.tris.push(Vector3::new(il, i, ir));
        }
    }

    fn clip_degenerate(&mut self, ear: &EvPtr<T>) {
        if clipped(ear) || folded(ear) {
            return;
        }
        let eps = self.eps;
        let eb = ear.borrow();
        let p = ear.borrow().pos;
        let pl = ear.borrow().pos_l();
        let pr = ear.borrow().pos_r();
        if eb.is_short(eps)
            || (is_ccw_2d(&pl, &p, &pr, eps) == 0 && (pl - p).dot(&(pr - p)) > T::zero())
        {
            self.clip_ear(ear);
            self.clip_degenerate(&eb.ptr_l());
            self.clip_degenerate(&eb.ptr_r());
        }
    }

    fn initialize(&mut self, polys: &[Vec<Pt<T>>]) -> Vec<EvPtr<T>> {
        let mut bgns = vec![];
        for poly in polys.iter() {
            let v = poly.first().unwrap();
            self.polygon
                .push(Rc::new(RefCell::new(Ecvt::new(v.idx, v.pos))));

            let first = Rc::clone(self.polygon.last().unwrap());
            let mut last = Rc::clone(&first);
            self.bbox.union(first.borrow().pos);
            bgns.push(Rc::clone(&first));

            for v in poly.iter().skip(1) {
                self.bbox.union(v.pos);
                self.polygon
                    .push(Rc::new(RefCell::new(Ecvt::new(v.idx, v.pos))));
                let next = Rc::clone(self.polygon.last().unwrap());
                Self::link(&last, &next);
                last = Rc::clone(&next);
            }
            Self::link(&last, &first);
        }

        if self.eps < T::zero() {
            self.eps = self.bbox.scale() * T::K_PRECISION;
        }

        // Slightly more than enough, since each hole can cause two extra triangles.
        self.tris.reserve(self.polygon.len() + 2 * bgns.len());

        bgns
    }

    fn find_start(&mut self, first: &mut EvPtr<T>) {
        let origin = first.borrow().pos;
        let mut bgn = Rc::clone(first);
        let mut max = T::MIN;
        let mut bbox: Rect<T> = Rect::default();
        let mut area = T::zero();
        let mut comp = T::zero(); // For Kahan's summation

        let add_point = |v: &mut EvPtr<T>| {
            bbox.union(v.borrow().pos);
            let tmp0 = det2x2(&(v.borrow().pos - origin), &(v.borrow().pos_r() - origin));
            let tmp1 = area + tmp0;
            comp += (area - tmp1) + tmp0;
            area = tmp1;
            if v.borrow().pos.x > max {
                max = v.borrow().pos.x;
                bgn = Rc::clone(v);
            }
        };

        if do_loop(first, add_point).is_none() {
            return;
        }
        area += comp;
        let size = bbox.size();
        let min_area = self.eps * size.x.max(size.y);

        if max.is_finite() && area < -min_area {
            self.holes.insert(EvPtrMaxPosX(Rc::clone(&bgn), bbox));
        } else {
            self.simples.push(Rc::clone(&bgn));
            if area > min_area {
                self.contour.push(Rc::clone(&bgn));
            }
        }
    }

    // Create a collider of all vertices in this polygon, each expanded by epsilon_.
    // Each ear uses this BVH to quickly find a subset of vertices to check for cost.
    fn vert_collider(start: &mut EvPtr<T>) -> IdxCollider<T> {
        let mut pts = vec![];
        let mut rfs = vec![];
        do_loop(start, |v| {
            pts.push(Pt {
                pos: v.borrow().pos,
                idx: rfs.len(),
            });
            rfs.push(Rc::clone(v));
        });

        compute_flat_tree(&mut pts);
        IdxCollider { pts, rfs }
    }

    // All holes must be key-holed (attached to an outer polygon) before ear clipping can commerce.
    // Instead of relying on sorting, which may be incorrect due to epsilon,
    // we check for polygon edges both ahead and behind to ensure all valid options are found.
    fn cut_key_hole(&mut self, bgn: &EvPtrMaxPosX<T>) {
        let p_bgn = bgn.0.borrow().pos;
        let eps = self.eps;
        let top = if p_bgn.y >= bgn.1.max.y - eps {
            1
        } else if p_bgn.y <= bgn.1.min.y + eps {
            -1
        } else {
            0
        };

        let mut con: Option<EvPtr<T>> = None;

        for first in self.contour.iter_mut() {
            do_loop(first, |v| {
                let vb = v.borrow();
                if let Some(x) = vb.interpolate_y2x(&p_bgn, top, eps) {
                    let flag = match &con {
                        None => true,
                        Some(c) => {
                            let cb = c.borrow();
                            let f1 =
                                is_ccw_2d(&Vector2::new(x, p_bgn.y), &cb.pos, &cb.pos_r(), eps)
                                    == 1;
                            let f2 = if cb.pos.y < vb.pos.y {
                                vb.inside_edge(c, eps, false)
                            } else {
                                !cb.inside_edge(v, eps, false)
                            };
                            f1 || f2
                        }
                    };
                    if bgn.0.borrow().inside_edge(v, eps, true) && flag {
                        con = Some(Rc::clone(v));
                    }
                }
            });
        }

        match con {
            None => {
                self.simples.push(Rc::clone(&bgn.0));
            }
            Some(c) => {
                let p = self.find_closer_bridge(&bgn.0, &c);
                self.join_polygons(&bgn.0, &p);
            }
        }
    }

    fn find_closer_bridge(&mut self, bgn: &EvPtr<T>, end: &EvPtr<T>) -> EvPtr<T> {
        let eb = end.borrow();
        let p_end = end.borrow().pos;
        let p_bgn = bgn.borrow().pos;
        let mut con = if p_end.x < p_bgn.x {
            eb.ptr_r()
        } else if eb.pos_r().x < p_bgn.x {
            Rc::clone(end)
        } else if eb.pos_r().y - p_bgn.y > p_bgn.y - p_end.y {
            Rc::clone(end)
        } else {
            eb.ptr_r()
        };

        if (con.borrow().pos.y - p_bgn.y).abs() <= self.eps {
            return Rc::clone(&con);
        }

        let above = if con.borrow().pos.y > p_bgn.y {
            T::one()
        } else {
            -T::one()
        };

        for first in self.contour.iter_mut() {
            do_loop(first, |v| {
                let vb = v.borrow();
                let vp = v.borrow().pos;
                let inside = above.as_i32() * is_ccw_2d(&p_bgn, &vp, &con.borrow().pos, self.eps);
                let f1 = vp.x > p_bgn.x - self.eps;
                let f2 = vp.y * above > p_bgn.y * above - self.eps;
                let f3 = inside == 0
                    && vp.x < con.borrow().pos.x
                    && vp.y * above < con.borrow().pos.y * above;
                let f4 = vb.inside_edge(end, self.eps, true);
                let f5 = vb.is_reflex(self.eps);
                if f1 && f2 && (inside > 0 || f3) && f4 && f5 {
                    con = Rc::clone(v);
                };
            });
        }
        Rc::clone(&con)
    }

    // Creates a keyhole between the start vert of a hole and the connector vert of an outer polygon.
    // To do this, both verts are duplicated and reattached. This process may create degenerate ears,
    // so these are clipped if necessary to keep from confusing sub_sequent key-holing operations.
    fn join_polygons(&mut self, sta: &EvPtr<T>, con: &EvPtr<T>) {
        let sta1 = Rc::new(RefCell::new(sta.borrow().clone()));
        let con1 = Rc::new(RefCell::new(con.borrow().clone()));
        self.polygon.push(Rc::clone(&sta1));
        self.polygon.push(Rc::clone(&con1));
        sta.borrow().ptr_r().borrow_mut().vl = Some(Rc::downgrade(&sta1));
        con.borrow().ptr_l().borrow_mut().vr = Some(Rc::downgrade(&con1));
        Self::link(sta, con);
        Self::link(&con1, &sta1);
        self.clip_degenerate(sta);
        self.clip_degenerate(&sta1);
        self.clip_degenerate(con);
        self.clip_degenerate(&con1);
    }

    // Recalculate the cost of the Vert v ear,
    // updating it in the queue by removing and reinserting it.
    fn process_ear(&mut self, v: &mut EvPtr<T>, collider: &IdxCollider<T>) {
        let taken = {
            let mut b = v.borrow_mut();
            b.ear.take()
        };
        if let Some(e) = taken {
            self.queue.remove(&EvPtrMinCost(e.upgrade().unwrap()));
        }

        if v.borrow().is_short(self.eps) {
            v.borrow_mut().cost = T::K_BEST;
            let ptr = EvPtrMinCost(Rc::clone(v));
            v.borrow_mut().ear = Some(Rc::downgrade(&ptr.0));
            self.queue.insert(ptr);
            return;
        }
        if v.borrow().is_convex(T::cast_from_usize(2) * self.eps) {
            v.borrow_mut().cost = { v.borrow().ear_cost(self.eps, collider) };
            let ptr = EvPtrMinCost(Rc::clone(v));
            v.borrow_mut().ear = Some(Rc::downgrade(&ptr.0));
            self.queue.insert(ptr);
            return;
        }

        v.borrow_mut().cost = T::one(); // not used, but marks reflex verts for debug
    }

    pub fn triangulate_poly(&mut self, first: &mut EvPtr<T>) {
        let c = Self::vert_collider(first);
        if c.rfs.is_empty() {
            return;
        }

        let mut nt = -2;
        self.queue.clear();

        if let Some(mut v) = do_loop(first, |v| {
            self.process_ear(v, &c);
            nt += 1;
        }) {
            while nt > 0 {
                if let Some(q) = self.queue.pop_first() {
                    v = Rc::clone(&q.0);
                }
                self.clip_ear(&v);
                nt -= 1;
                self.process_ear(&mut v.borrow().ptr_l(), &c);
                self.process_ear(&mut v.borrow().ptr_r(), &c);
                let ptr_r = v.borrow().ptr_r();
                v = Rc::clone(&ptr_r);
            }
        }
    }
}
