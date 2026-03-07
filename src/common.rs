//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

use nalgebra::{ArrayStorage, Matrix, RealField, Vector2, Vector3};
use num_traits::NumCast;

// TODO rename this once you have removed all old references to the "Real" type.
pub trait BoolReal:
    std::fmt::Debug + RealField + geo::CoordNum + Copy + PartialOrd + PartialEq
{
    const K_PRECISION: Self;
    const K_BEST: Self;
    const MAX: Self;
    const MIN: Self;
    const NAN: Self;
    const PI: Self;
    const EPSILON: Self;
    const HALF: Self;

    type BitsType: std::cmp::Eq + std::hash::Hash;
    type IndexType: std::ops::Add + NumCast;

    fn is_nan(&self) -> bool;
    fn is_infinite(&self) -> bool;
    fn to_bits(&self) -> Self::BitsType;

    fn cast_from_usize(value: usize) -> Self;
    fn cast_from_f64(value: f64) -> Self;
    fn as_i32(&self) -> i32;
    fn as_u32(&self) -> u32;
}

impl BoolReal for f64 {
    const K_PRECISION: f64 = 1e-12;
    const K_BEST: f64 = f64::MIN;
    const MAX: f64 = f64::MAX;
    const MIN: f64 = f64::MIN;
    const NAN: f64 = f64::NAN;
    const PI: f64 = std::f64::consts::PI;
    const EPSILON: f64 = f64::EPSILON;
    const HALF: f64 = 0.5;

    type BitsType = u64;
    type IndexType = u32;

    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
    }
    fn is_infinite(&self) -> bool {
        f64::is_infinite(*self)
    }
    fn to_bits(&self) -> Self::BitsType {
        f64::to_bits(*self)
    }

    fn cast_from_usize(value: usize) -> Self {
        value as Self
    }
    fn cast_from_f64(value: f64) -> Self {
        value as Self
    }
    fn as_i32(&self) -> i32 {
        *self as i32
    }
    fn as_u32(&self) -> u32 {
        *self as u32
    }
}

impl BoolReal for f32 {
    const K_PRECISION: f32 = 1e-4;
    const K_BEST: f32 = f32::MIN;
    const MAX: f32 = f32::MAX;
    const MIN: f32 = f32::MIN;
    const NAN: f32 = f32::NAN;
    const PI: f32 = std::f32::consts::PI;
    const EPSILON: f32 = f32::EPSILON;
    const HALF: f32 = 0.5;

    type BitsType = u32;
    type IndexType = u16;

    fn is_nan(&self) -> bool {
        f32::is_nan(*self)
    }
    fn is_infinite(&self) -> bool {
        f32::is_infinite(*self)
    }
    fn to_bits(&self) -> Self::BitsType {
        f32::to_bits(*self)
    }

    fn cast_from_usize(value: usize) -> Self {
        value as Self
    }
    fn cast_from_f64(value: f64) -> Self {
        value as Self
    }
    fn as_i32(&self) -> i32 {
        *self as i32
    }
    fn as_u32(&self) -> u32 {
        *self as u32
    }
}

pub trait VectorExt {
    fn min_components(&self, other: &Self) -> Self;
    fn max_components(&self, other: &Self) -> Self;
    fn is_nan(&self) -> bool;
    fn normalize_or_zero(&self) -> Self;
}

impl<T, const W: usize> VectorExt
    for Matrix<T, nalgebra::Const<W>, nalgebra::U1, ArrayStorage<T, W, 1>>
where
    T: BoolReal,
{
    fn min_components(&self, other: &Self) -> Self {
        Self::from_iterator(self.iter().zip(other.iter()).map(|(s, o)| s.min(*o)))
    }

    fn max_components(&self, other: &Self) -> Self {
        Self::from_iterator(self.iter().zip(other.iter()).map(|(s, o)| s.max(*o)))
    }

    fn is_nan(&self) -> bool {
        self.iter().any(|c| c.is_nan())
    }

    fn normalize_or_zero(&self) -> Self {
        self.try_normalize(T::EPSILON).unwrap_or(Self::zeros())
    }
}

#[derive(PartialEq)]
pub enum OpType {
    Add,
    Subtract,
    Intersect,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct Half {
    pub tail: usize,
    pub head: usize,
    pub pair: usize,
}

impl Default for Half {
    fn default() -> Self {
        Self {
            tail: usize::MAX,
            head: usize::MAX,
            pair: usize::MAX,
        }
    }
}

impl Half {
    pub fn new(tail: usize, head: usize, pair: usize) -> Self {
        Self { tail, head, pair }
    }
    pub fn new_without_pair(tail: usize, head: usize) -> Self {
        Self {
            tail,
            head,
            pair: usize::MAX,
        }
    }
    pub fn is_forward(&self) -> bool {
        self.tail < self.head
    }
    pub fn tail(&self) -> Option<usize> {
        if self.tail == usize::MAX {
            None
        } else {
            Some(self.tail)
        }
    }
    pub fn head(&self) -> Option<usize> {
        if self.head == usize::MAX {
            None
        } else {
            Some(self.head)
        }
    }
    pub fn pair(&self) -> Option<usize> {
        if self.pair == usize::MAX {
            None
        } else {
            Some(self.pair)
        }
    }
}

pub fn face_of(hid: usize) -> usize {
    hid / 3
}
pub fn next_of(hid: usize) -> usize {
    let mut i = hid + 1;
    if i.is_multiple_of(3) {
        i -= 3;
    }
    i
}

#[derive(Clone, Debug, Copy)]
pub struct Tref {
    pub mid: usize, // mesh id
    pub fid: usize, // face id
    pub pid: i32,   // planer id
}

impl Default for Tref {
    fn default() -> Self {
        Self {
            mid: usize::MAX,
            fid: usize::MAX,
            pid: -1,
        }
    }
}

pub fn det2x2<T: BoolReal>(a: &Vector2<T>, b: &Vector2<T>) -> T {
    a.x * b.y - a.y * b.x
}

pub fn get_aa_proj_matrix<T: BoolReal>(n: &Vector3<T>) -> (Vector3<T>, Vector3<T>) {
    let a = n.abs();
    let m: T;
    let r1: Vector3<T>;
    let r2: Vector3<T>;

    if a.z > a.x && a.z > a.y {
        r1 = Vector3::new(T::one(), T::zero(), T::zero());
        r2 = Vector3::new(T::zero(), T::one(), T::zero());
        m = n.z;
    }
    // preserve x, y
    else if a.y > a.x {
        r1 = Vector3::new(T::zero(), T::zero(), T::one());
        r2 = Vector3::new(T::one(), T::zero(), T::zero());
        m = n.y;
    }
    // preserve z, x
    else {
        r1 = Vector3::new(T::zero(), T::one(), T::zero());
        r2 = Vector3::new(T::zero(), T::zero(), T::one());
        m = n.x;
    } // preserve y, z

    if m < T::zero() {
        (-r1, r2)
    } else {
        (r1, r2)
    }
}

pub fn compute_aa_proj<T: BoolReal>(p: &(Vector3<T>, Vector3<T>), v: &Vector3<T>) -> Vector2<T> {
    Vector2::new(p.0.dot(v), p.1.dot(v))
}

pub fn is_ccw_2d<T: BoolReal>(p0: &Vector2<T>, p1: &Vector2<T>, p2: &Vector2<T>, t: T) -> i32 {
    let v1 = p1 - p0;
    let v2 = p2 - p0;
    let area = v1.x * v2.y - v1.y * v2.x;
    let base = v1.norm_squared().max(v2.norm_squared());
    if area.powi(2) * T::cast_from_usize(4) <= base * t.powi(2) {
        return 0;
    }
    if area > T::zero() {
        1
    } else {
        -1
    }
}

pub fn is_ccw_3d<T: BoolReal>(
    p0: &Vector3<T>,
    p1: &Vector3<T>,
    p2: &Vector3<T>,
    n: &Vector3<T>,
    t: T,
) -> i32 {
    let p = get_aa_proj_matrix(n);
    is_ccw_2d(
        &compute_aa_proj(&p, p0),
        &compute_aa_proj(&p, p1),
        &compute_aa_proj(&p, p2),
        t,
    )
}

pub fn safe_normalize<T: BoolReal>(v: Vector2<T>) -> Vector2<T> {
    let n = v.normalize();
    if n.x.is_finite() && !n.x.is_nan() && n.y.is_finite() && !n.y.is_nan() {
        n
    } else {
        Vector2::new(T::zero(), T::zero())
    }
}

pub fn compute_orthogonal<T: BoolReal>(n: Vector3<T>) -> Vector3<T> {
    let b = if n.x.abs() < T::cast_from_f64(0.9) {
        Vector3::new(T::one(), T::zero(), T::one())
    } else {
        Vector3::new(T::zero(), T::one(), T::zero())
    };
    n.cross(&b).normalize()
}

/*
enum CsgNodeType { Union, Intersection, Difference, Leaf }
trait CsgNode { fn ToLeafNode () { } }
struct CsgOpNode { }
struct CsgLeafNode { }
impl CsgNode for CsgOpNode { fn ToLeafNode () { } }
impl CsgNode for CsgLeafNode { }
*/
