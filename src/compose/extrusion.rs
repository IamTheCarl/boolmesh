use geo::{BoundingRect, Coord, LineString, MultiPolygon, Polygon};
use nalgebra::{Matrix4, Vector2, Vector3};
use thiserror::Error;

use crate::{
    common::BoolReal,
    manifold::ManifoldError,
    prelude::Manifold,
    triangulation::{ear_clip::EarClip, Pt},
};

trait IterStrings<T: BoolReal> {
    fn strings(&self) -> impl Iterator<Item = &LineString<T>>;
    fn coords(&self) -> impl Iterator<Item = &Coord<T>>;
    fn num_coords(&self) -> usize;
}

impl<T: BoolReal> IterStrings<T> for Polygon<T> {
    fn strings(&self) -> impl Iterator<Item = &LineString<T>> {
        [self.exterior()].into_iter().chain(self.interiors())
    }

    fn coords(&self) -> impl Iterator<Item = &Coord<T>> {
        self.strings().flat_map(|string| string.coords())
    }

    fn num_coords(&self) -> usize {
        // TODO instead of counting each coor individually, sum up the length of all the strings.
        self.coords().count()
    }
}

/// Controls how faces are created on an extruded/revolved manifold.
enum FaceMode {
    /// Close the face
    Close,

    /// No face, loop the structure back to its first layer
    Loop,
}

fn raw_extrude_impl<'s, T, P>(
    polygon_iter_builder: impl Fn() -> P,
    divisions: usize,
    face_mode: FaceMode,
    transform: impl Fn(T) -> Matrix4<T>,
) -> Result<Manifold<T>, ManifoldError>
where
    T: BoolReal,
    P: IntoIterator<Item = &'s Polygon<T>>,
{
    // let n = Vector3::new(T::zero(), T::zero(), T::one());
    // let proj = crate::common::get_aa_proj_matrix(&n);

    // let pts: Vec<_> = polygon_iter_builder()
    //     .into_iter()
    //     .flat_map(|polygon| polygon.coords())
    //     .map(|p| Vector3::new(p.x, p.y, T::zero()))
    //     .collect();
    // let poly = pts
    //     .iter()
    //     .enumerate()
    //     .map(|(i, p)| Pt {
    //         pos: crate::common::compute_aa_proj(&proj, p),
    //         idx: i,
    //     })
    //     .collect::<Vec<_>>();
    // let idcs = EarClip::new(&[poly], T::K_PRECISION).triangulate();

    // let mut oft_ps = vec![];
    // let mut oft_ts = vec![];
    // let n = pts.len();
    // for p in pts.iter() {
    //     oft_ps.push(*p);
    // }
    // let transform = transform(T::one());
    // for p in pts.iter() {
    //     oft_ps.push(
    //         transform
    //             .transform_point(&nalgebra::OPoint { coords: *p })
    //             .coords,
    //     );
    // }
    // for i in idcs.iter() {
    //     oft_ts.push(Vector3::new(i.z, i.y, i.x));
    // }
    // for i in idcs.iter() {
    //     oft_ts.push(Vector3::new(i.x + n, i.y + n, i.z + n));
    // }
    // for i in 0..n {
    //     let j = (i + 1) % n;
    //     oft_ts.push(Vector3::new(i, j, i + n));
    //     oft_ts.push(Vector3::new(i + n, j, j + n));
    // }
    // Manifold::new_impl(oft_ps, oft_ts, None, None)

    fn points<T: BoolReal>(polygon: &Polygon<T>) -> impl Iterator<Item = Vector3<T>> {
        polygon
            .coords()
            .map(|coord| Vector3::new(coord.x, coord.y, T::zero()))
    }

    let face_indicies = if matches!(face_mode, FaceMode::Close) {
        let mut i = 0;

        // TODO I dislike collecting the polygons into a throw-away vec like this, but I want to
        // avoid changing the core library for now.
        let polygons: Vec<Vec<Pt<T>>> = polygon_iter_builder()
            .into_iter()
            .map(|polygon| {
                polygon
                    .coords()
                    .map(|c| {
                        let pt = Pt {
                            pos: Vector2::new(c.x, c.y),
                            idx: i,
                        };
                        i += 1;
                        pt
                    })
                    .collect::<Vec<Pt<T>>>()
            })
            .collect();

        Some(EarClip::new(&polygons, T::K_PRECISION).triangulate())
    } else {
        // If we don't need to close the faces, then we don't need to calculate the face indicies.
        None
    };

    let mut oft_ps = vec![];
    let mut oft_ts = vec![];
    let points_per_division: usize = polygon_iter_builder()
        .into_iter()
        .map(|polygon| polygon.num_coords())
        .sum();

    // Insert bottom verticies.
    for p in polygon_iter_builder().into_iter().flat_map(points) {
        oft_ps.push(p);
    }

    if let Some(face_indicies) = face_indicies.as_ref() {
        // Insert bottom vertex references.
        for i in face_indicies.iter() {
            oft_ts.push(Vector3::new(i.z, i.y, i.x));
        }
    }

    // Incert divisions. Note that the top of the shape counts as a division.
    for layer in 0..divisions {
        let alpha = T::cast_from_usize(layer + 1) / T::cast_from_usize(divisions);
        let transform = transform(alpha);

        // Insert the next division's verticies
        let mut polygon_point_offset = 0;
        for polygon in polygon_iter_builder() {
            let base_offset = layer * points_per_division + polygon_point_offset;
            let points_in_polygon = polygon.num_coords();
            for (vertex_index, position) in points(polygon).enumerate() {
                // Conversion is necessary for 32bit support.
                #[allow(clippy::useless_conversion)]
                oft_ps.push(transform.transform_point(&position.into()).coords.into());

                // Corners of a quardrangle making up a a side of the extruded shape.
                // k--l
                // |  |
                // i--j
                let i = base_offset + vertex_index;
                let j = base_offset + (vertex_index + 1) % points_in_polygon;
                let k = i + points_per_division;
                let l = j + points_per_division;

                oft_ts.push(Vector3::new(i, j, k));
                oft_ts.push(Vector3::new(k, j, l));
            }

            polygon_point_offset += points_in_polygon;
        }
    }

    if let Some(face_indicies) = face_indicies.as_ref() {
        // Insert top vertex references.
        // We do not need to insert their verticies because they were provided by the final layer of
        // the divisions loop.
        for i in face_indicies.iter() {
            oft_ts.push(Vector3::new(
                i.x + points_per_division * divisions,
                i.y + points_per_division * divisions,
                i.z + points_per_division * divisions,
            ));
        }
    } else {
        // Loop the final layer back to the first layer.
        let mut polygon_point_offset = 0;
        for polygon in polygon_iter_builder() {
            let ending_offset = points_per_division * divisions + polygon_point_offset;
            let starting_offset = polygon_point_offset;
            let points_in_polygon = polygon.num_coords();
            for (vertex_index, _position) in points(polygon).enumerate() {
                // Corners of a quardrangle making up a a side of the extruded shape.
                // k--l
                // |  |
                // i--j
                let k = vertex_index + starting_offset;
                let l = (vertex_index + 1) % points_in_polygon + starting_offset;
                let i = vertex_index + ending_offset;
                let j = (vertex_index + 1) % points_in_polygon + ending_offset;

                oft_ts.push(Vector3::new(i, j, k));
                oft_ts.push(Vector3::new(k, j, l));
            }

            polygon_point_offset += points_in_polygon;
        }
    }

    Manifold::new_impl(oft_ps, oft_ts, None, None)
}

#[derive(Debug, Error)]
pub enum ExtrusionError {
    #[error("Extrusion height must be greater than zero")]
    InvalidHeight,

    #[error("Height of extrusion top must be greater than or equel to zero")]
    InvalidScale,

    #[error("Error buiding manifold: {0}")]
    Manifold(#[from] ManifoldError),
}

fn extrude_impl<'s, P, T>(
    polygon_iter_builder: impl Fn() -> P,
    height: T,
    divisions: usize,
    twist_radians: T,
    scale_top: Vector2<T>,
) -> Result<Manifold<T>, ExtrusionError>
where
    T: BoolReal,
    P: IntoIterator<Item = &'s Polygon<T>>,
{
    if height <= T::zero() {
        return Err(ExtrusionError::InvalidHeight);
    }

    if scale_top.x < T::zero() || scale_top.y < T::zero() {
        return Err(ExtrusionError::InvalidScale);
    }

    let manifold = raw_extrude_impl(polygon_iter_builder, divisions, FaceMode::Close, |alpha| {
        let scale = Matrix4::new_nonuniform_scaling(
            &Vector3::from_element(T::one())
                .lerp(&Vector3::new(scale_top.x, scale_top.y, T::one()), alpha),
        );
        let translation =
            Matrix4::new_translation(&Vector3::new(T::zero(), T::zero(), alpha * height));
        let rotation = Matrix4::from_euler_angles(T::zero(), T::zero(), alpha * twist_radians);

        scale * rotation * translation
    })?;

    Ok(manifold)
}

#[derive(Debug, Error)]
pub enum RevolveError {
    #[error("Revolution angle must be greater than zero")]
    InvalidAngle,

    #[error("Geometry must not be present on the left side of the Y axis")]
    LeftOfYAxis,

    #[error("Error buiding manifold: {0}")]
    Manifold(#[from] ManifoldError),
}

fn revolve_impl<'s, P, T>(
    polygon_iter_builder: impl Fn() -> P,
    divisions: usize,
    angle_radians: T,
) -> Result<Manifold<T>, RevolveError>
where
    T: BoolReal,
    P: IntoIterator<Item = &'s Polygon<T>>,
{
    if angle_radians <= T::zero() {
        return Err(RevolveError::InvalidAngle);
    }

    // Checks if any geometry has a point left of the Y axis
    if polygon_iter_builder().into_iter().any(|polygon| {
        polygon
            .bounding_rect()
            .is_some_and(|rect| rect.min().x < T::zero())
    }) {
        return Err(RevolveError::LeftOfYAxis);
    }

    let max_angle = T::PI * T::cast_from_usize(2);

    // Cap the angle at 2Pi.
    let angle_radians = max_angle.min(angle_radians);

    let face_mode = if angle_radians < max_angle {
        FaceMode::Close
    } else {
        FaceMode::Loop
    };

    let manifold = raw_extrude_impl(polygon_iter_builder, divisions, face_mode, |alpha| {
        Matrix4::from_euler_angles(T::zero(), -angle_radians * alpha, T::zero())
    })?;

    Ok(manifold)
}

pub trait ExtrudePoly<T: BoolReal> {
    fn extrude(
        &self,
        height: T,
        divisions: usize,
        twist_radians: T,
        scale_top: Vector2<T>,
    ) -> Result<Manifold<T>, ExtrusionError>;

    fn revolve(&self, divisions: usize, angle_radians: T) -> Result<Manifold<T>, RevolveError>;
}

impl<T: BoolReal> ExtrudePoly<T> for Polygon<T> {
    fn extrude(
        &self,
        height: T,
        divisions: usize,
        twist_radians: T,
        scale_top: Vector2<T>,
    ) -> Result<Manifold<T>, ExtrusionError> {
        extrude_impl(|| [self], height, divisions, twist_radians, scale_top)
    }

    fn revolve(&self, divisions: usize, angle_radians: T) -> Result<Manifold<T>, RevolveError> {
        revolve_impl(|| [self], divisions, angle_radians)
    }
}

impl<T: BoolReal> ExtrudePoly<T> for MultiPolygon<T> {
    fn extrude(
        &self,
        height: T,
        divisions: usize,
        twist_radians: T,
        scale_top: Vector2<T>,
    ) -> Result<Manifold<T>, ExtrusionError> {
        extrude_impl(|| self.iter(), height, divisions, twist_radians, scale_top)
    }

    fn revolve(&self, divisions: usize, angle_radians: T) -> Result<Manifold<T>, RevolveError> {
        revolve_impl(|| self.iter(), divisions, angle_radians)
    }
}
