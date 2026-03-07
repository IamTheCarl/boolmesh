//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use nalgebra::Vector2;

use crate::{common::BoolReal, triangulation::Pt};

pub fn compute_flat_tree<T: BoolReal>(pts: &mut [Pt<T>]) {
    if pts.len() <= 8 {
        return;
    }
    compute_flat_tree_impl(pts, true);
}

fn compute_flat_tree_impl<T: BoolReal>(pts: &mut [Pt<T>], sort_x: bool) {
    let eq = std::cmp::Ordering::Equal;
    if sort_x {
        pts.sort_by(|a, b| a.pos.x.partial_cmp(&b.pos.x).unwrap_or(eq));
    } else {
        pts.sort_by(|a, b| a.pos.y.partial_cmp(&b.pos.y).unwrap_or(eq));
    }

    if pts.len() < 2 {
        return;
    }

    let (l, mr) = pts.split_at_mut(pts.len() / 2);
    if !mr.is_empty() {
        let (_, r) = mr.split_first_mut().unwrap();
        compute_flat_tree_impl(l, !sort_x);
        compute_flat_tree_impl(r, !sort_x);
    }
}

pub fn compute_query_flat_tree<F, T>(pts: &[Pt<T>], rect: &Rect<T>, mut func: F)
where
    T: BoolReal,
    F: FnMut(&Pt<T>),
{
    for p in pts.iter() {
        if rect.contains(&p.pos) {
            func(p);
        }
    }

    //if pts.len() <= 8 {
    //    for p in pts.iter() { if rect.contains(&p.pos) { func(p);} }
    //} else {
    //    query_two_d_tree(pts, rect.clone(), func);
    //}
}

pub fn query_two_d_tree<F, T>(pts: &[Pt<T>], r: Rect<T>, mut f: F)
where
    T: BoolReal,
    F: FnMut(&Pt<T>),
{
    let mut cur: Rect<T> = Rect::default();
    let mut lev: i32 = 0;
    let mut bgn: usize = 0;
    let mut len: usize = pts.len();

    cur.min = Vector2::from_element(T::MIN);
    cur.max = Vector2::from_element(T::MAX);

    // Stack holds deferred right subtrees: (rect, start, len, level)
    let mut stack: Vec<(Rect<T>, usize, usize, i32)> = Vec::with_capacity(64);

    loop {
        if len <= 2 {
            for i in 0..len {
                let p = &pts[bgn + i];
                if r.contains(&p.pos) {
                    f(p);
                }
            }
            if let Some((r, b, ln, lv)) = stack.pop() {
                cur = r;
                bgn = b;
                len = ln;
                lev = lv;
                continue;
            } else {
                break;
            }
        }

        let mid_oft = len / 2;
        let mid_idx = bgn + mid_oft;
        let mid = &pts[mid_idx];

        let mut rect_l = cur.clone();
        let mut rect_r = cur.clone();
        if lev % 2 == 0 {
            rect_l.max.x = mid.pos.x;
            rect_r.min.x = mid.pos.x;
        } else {
            rect_l.max.y = mid.pos.y;
            rect_r.min.y = mid.pos.y;
        }

        if r.contains(&mid.pos) {
            f(mid);
        }

        let overlaps_l = rect_l.overlap(&r);
        let overlaps_r = rect_r.overlap(&r);

        if overlaps_l {
            if overlaps_r {
                let r_bgn = mid_idx + 1;
                let r_len = len - (mid_oft + 1);
                stack.push((rect_r, r_bgn, r_len, lev + 1));
            }
            cur = rect_l;
            len = mid_oft;
            lev += 1;
        } else {
            cur = rect_r;
            bgn = mid_idx + 1;
            len -= mid_oft + 1;
            lev += 1;
        }
    }
}

#[derive(Clone)]
pub struct Rect<T> {
    pub min: Vector2<T>,
    pub max: Vector2<T>,
}

impl<T: BoolReal> Rect<T> {
    pub fn default() -> Self {
        Self {
            min: Vector2::from_element(T::MAX),
            max: Vector2::from_element(T::MIN),
        }
    }

    pub fn new(a: &Vector2<T>, b: &Vector2<T>) -> Self {
        Self {
            min: Vector2::new(a.x.min(b.x), a.y.min(b.y)),
            max: Vector2::new(a.x.max(b.x), a.y.max(b.y)),
        }
    }

    pub fn contains(&self, p: &Vector2<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn size(&self) -> Vector2<T> {
        self.max - self.min
    }

    pub fn scale(&self) -> T {
        let a_min = self.min.x.abs().max(self.min.y.abs());
        let a_max = self.max.x.abs().max(self.max.y.abs());
        a_min.max(a_max)
    }

    pub fn overlap(&self, r: &Rect<T>) -> bool {
        self.max.x >= r.min.x
            && self.max.y >= r.min.y
            && self.min.x <= r.max.x
            && self.min.y <= r.max.y
    }

    pub fn union(&mut self, p: Vector2<T>) {
        self.min = Vector2::new(self.min.x.min(p.x), self.min.y.min(p.y));
        self.max = Vector2::new(self.max.x.max(p.x), self.max.y.max(p.y));
    }
}
