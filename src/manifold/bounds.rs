//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use nalgebra::{Vector2, Vector3};

use crate::common::{BoolReal, VectorExt as _};

#[derive(Clone, Debug)]
pub enum Query<T: BoolReal> {
    Bb(BBox<T>),
    Pt(BPos<T>),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct BBox<T: BoolReal> {
    pub id: Option<usize>,
    pub min: Vector3<T>,
    pub max: Vector3<T>,
}

#[derive(Clone, Debug)]
pub struct BPos<T> {
    pub id: Option<usize>,
    pub pos: Vector2<T>,
}

impl<T: BoolReal> BBox<T> {
    pub fn default() -> Self {
        BBox {
            id: None,
            min: Vector3::from_element(T::MAX),
            max: Vector3::from_element(T::MIN),
        }
    }

    pub fn new(id: Option<usize>, pts: &[Vector3<T>]) -> Self {
        let mut b = BBox {
            id,
            min: Vector3::from_element(T::MAX),
            max: Vector3::from_element(T::MIN),
        };
        for pt in pts {
            b.union(pt);
        }
        b
    }

    pub fn size(&self) -> Vector3<T> {
        self.max - self.min
    }

    pub fn scale(&self) -> T {
        let s = self.size();
        s.x.abs().max(s.y.abs()).max(s.z.abs())
    }

    pub fn overlaps(&self, q: &Query<T>) -> bool {
        match q {
            Query::Bb(b) => {
                self.min.iter().zip(b.max.iter()).all(|(s, b)| s <= b)
                    && self.max.iter().zip(b.min.iter()).all(|(s, b)| s >= b)
            }
            Query::Pt(p) => {
                // only evaluates xy axis
                self.min.x <= p.pos.x
                    && self.min.y <= p.pos.y
                    && self.max.x >= p.pos.x
                    && self.max.y >= p.pos.y
            }
        }
    }

    pub fn union(&mut self, p: &Vector3<T>) {
        if p.x.is_nan() {
            return;
        }
        self.min = self.min.min_components(p);
        self.max = self.max.max_components(p);
    }

    pub fn longest_dim(&self) -> usize {
        let s = self.size();
        if s.x > s.y && s.x > s.z {
            0
        } else if s.y > s.z {
            1
        } else {
            2
        }
    }
}

pub fn union_bbs<T: BoolReal>(b0: &BBox<T>, b1: &BBox<T>) -> BBox<T> {
    let min = b0.min.min_components(&b1.min);
    let max = b0.max.max_components(&b1.max);
    BBox { id: None, min, max }
}
