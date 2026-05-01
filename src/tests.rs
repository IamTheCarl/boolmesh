//--- Copyright (C) 2025 Saki Komikado <komietty@gmail.com>,
//--- This Source Code Form is subject to the terms of the Mozilla Public License v.2.0.

#[cfg(test)]
mod test_intersection {
    use nalgebra::Vector3;

    use crate::boolean03::kernel03::winding03;
    use crate::boolean03::kernel12::intersect12;
    use crate::boolean03::Boolean03;
    use crate::boolean45::boolean45;
    use crate::OpType;

    type Manifold = crate::Manifold<f64>;

    pub fn gen_tet_a() -> Manifold {
        Manifold::new(
            &vec![
                -0.866025, -1., 0.5, 0., -1., -1., 0.866025, -1., 0.5, 0., 1., 0.,
            ],
            &vec![0, 3, 1, 1, 2, 0, 1, 3, 2, 2, 3, 0],
        )
        .unwrap()
    }

    pub fn gen_tet_b() -> Manifold {
        Manifold::new(
            &vec![
                -1., -0.866025, 0.5, -1., 0., -1., -1., 0.866025, 0.5, 1., 0., 0.,
            ],
            &vec![1, 3, 0, 1, 0, 2, 2, 3, 1, 0, 3, 2],
        )
        .unwrap()
    }

    pub fn gen_tet_c() -> Manifold {
        Manifold::new(
            &vec![
                -2., -0.866025, 0.5, -2., -0., -1., -2., 0.866025, 0.5, 0., 0., 0.,
            ],
            &vec![1, 3, 0, 1, 0, 2, 2, 3, 1, 0, 3, 2],
        )
        .unwrap()
    }

    #[test]
    fn test_tet_sub_inclusion_case() {
        let expand = -1.;
        let mfd_p = gen_tet_a();
        let mfd_q = gen_tet_c();
        let mut p1q2 = vec![];
        let mut p2q1 = vec![];
        let (x12, v12) = intersect12(&mfd_p, &mfd_q, &mut p1q2, expand, true);
        let (x21, v21) = intersect12(&mfd_p, &mfd_q, &mut p2q1, expand, false);
        let w03 = winding03(&mfd_p, &mfd_q, expand, true);
        let w30 = winding03(&mfd_p, &mfd_q, expand, false);

        assert_eq!(w03, vec![0, 0, 0, 0]);
        assert_eq!(w30, vec![0, 0, 0, 1]);
        assert_eq!(x12.len(), 0);
        assert_eq!(v12.len(), 0);
        assert_eq!(x21, vec![-1, -1, -1]);
        let v21_ = vec![
            Vector3::new(-0.224009, 0., -0.112005),
            Vector3::new(-0.294367, 0.127465, 0.0735918),
            Vector3::new(-0.395087, -0.171077, 0.0987716),
        ];
        for i in 0..3 {
            assert!((v21[i] - v21_[i]).norm() < 1e-6);
        }
        let op = OpType::Subtract;
        let b03 = Boolean03 {
            p1q2,
            p2q1,
            x12,
            x21,
            w03,
            w30,
            v12,
            v21,
        };
        let b45 = boolean45(&mfd_p, &mfd_q, &b03, &op);
    }

    #[test]
    fn test_tet_sub_penetration_case() {
        let expand = -1.;
        let mfd_p = gen_tet_a();
        let mfd_q = gen_tet_b();
        let mut p1q2 = vec![];
        let mut p2q1 = vec![];
        let (x12, v12) = intersect12(&mfd_p, &mfd_q, &mut p1q2, -1., true);
        let (x21, v21) = intersect12(&mfd_p, &mfd_q, &mut p2q1, -1., false);
        let w03 = winding03(&mfd_p, &mfd_q, expand, true);
        let w30 = winding03(&mfd_p, &mfd_q, expand, false);

        let v12_ = vec![
            Vector3::new(-0.763707, -0.763707, 0.440927),
            Vector3::new(-0.242656, 0.439609, 0.140098),
            Vector3::new(0., 0., -0.5),
            Vector3::new(0., 0., -0.5),
        ];
        let v21_ = vec![
            Vector3::new(0.302169, 0.302169, 0.174458),
            Vector3::new(0.439609, -0.242656, 0.140098),
            Vector3::new(0.302169, 0.302169, 0.174458),
            Vector3::new(-0.763707, -0.763707, 0.440927),
        ];

        assert_eq!(w03, vec![0, 0, 0, 0]);
        assert_eq!(w30, vec![0, 0, 0, 0]);
        assert_eq!(x12, vec![-1, 1, -1, 1]);
        assert_eq!(x21, vec![1, 1, -1, -1]);
        for i in 0..3 {
            assert!((v12[i] - v12_[i]).norm() < 1e-6);
            assert!((v21[i] - v21_[i]).norm() < 1e-6);
        }

        let op = OpType::Subtract;
        let b03 = Boolean03 {
            p1q2,
            p2q1,
            x12,
            x21,
            w03,
            w30,
            v12,
            v21,
        };
        let b45 = boolean45(&mfd_p, &mfd_q, &b03, &op);
    }
}

#[cfg(test)]
mod test_triangulation {
    use nalgebra::{Vector2, Vector3};

    use crate::triangulation::ear_clip::EarClip;
    use crate::triangulation::Pt;

    #[test]
    fn test_ear_clip() {
        let polys = vec![
            vec![
                Pt {
                    idx: 2120,
                    pos: Vector2::new(0.048238, 0.680959),
                },
                Pt {
                    idx: 2124,
                    pos: Vector2::new(-0.0145625, -0.676874),
                },
                Pt {
                    idx: 2123,
                    pos: Vector2::new(0.0245192, -0.68213),
                },
                Pt {
                    idx: 2119,
                    pos: Vector2::new(0.0482562, -0.681659),
                },
            ],
            vec![
                Pt {
                    idx: 2122,
                    pos: Vector2::new(-0.068635, -0.673357),
                },
                Pt {
                    idx: 2125,
                    pos: Vector2::new(-0.0487738, -0.690778),
                },
                Pt {
                    idx: 2121,
                    pos: Vector2::new(-0.02279, -0.676339),
                },
            ],
        ];
        let res0 = vec![
            Vector3::new(2123, 2119, 2120),
            Vector3::new(2123, 2120, 2124),
            Vector3::new(2125, 2121, 2122),
        ];

        let res1 = EarClip::new(&polys, 1e-12).triangulate();
        for i in 0..3 {
            assert_eq!(res0[i], res1[i]);
        }
    }
}

#[cfg(test)]
mod test_simplification {

    use crate::simplification::collapse::collapse_collinear_edges;

    type Vector3 = nalgebra::Vector3<f64>;

    #[test]
    fn test_collapse() {
        use crate::common::{HalfEdge, Tref};

        let mut hs = vec![
            HalfEdge::new(0, 4, 5),
            HalfEdge::new(4, 1, 42),
            HalfEdge::new(1, 0, 9),
            HalfEdge::new(0, 2, 26),
            HalfEdge::new(2, 4, 50),
            HalfEdge::new(4, 0, 0),
            HalfEdge::new(0, 11, 11),
            HalfEdge::new(11, 9, 57),
            HalfEdge::new(9, 0, 21),
            HalfEdge::new(0, 1, 2),
            HalfEdge::new(1, 11, 14),
            HalfEdge::new(11, 0, 6),
            HalfEdge::new(1, 10, 17),
            HalfEdge::new(10, 11, 68),
            HalfEdge::new(11, 1, 10),
            HalfEdge::new(1, 3, 35),
            HalfEdge::new(3, 10, 20),
            HalfEdge::new(10, 1, 12),
            HalfEdge::new(3, 13, 31),
            HalfEdge::new(13, 10, 66),
            HalfEdge::new(10, 3, 16),
            HalfEdge::new(0, 9, 8),
            HalfEdge::new(9, 12, 59),
            HalfEdge::new(12, 0, 24),
            HalfEdge::new(0, 12, 23),
            HalfEdge::new(12, 2, 27),
            HalfEdge::new(2, 0, 3),
            HalfEdge::new(2, 12, 25),
            HalfEdge::new(12, 13, 64),
            HalfEdge::new(13, 2, 30),
            HalfEdge::new(2, 13, 29),
            HalfEdge::new(13, 3, 18),
            HalfEdge::new(3, 2, 36),
            HalfEdge::new(1, 7, 47),
            HalfEdge::new(7, 3, 37),
            HalfEdge::new(3, 1, 15),
            HalfEdge::new(2, 3, 32),
            HalfEdge::new(3, 7, 34),
            HalfEdge::new(7, 2, 51),
            HalfEdge::new(4, 6, 49),
            HalfEdge::new(6, 5, 54),
            HalfEdge::new(5, 4, 43),
            HalfEdge::new(1, 4, 1),
            HalfEdge::new(4, 5, 41),
            HalfEdge::new(5, 1, 45),
            HalfEdge::new(1, 5, 44),
            HalfEdge::new(5, 7, 56),
            HalfEdge::new(7, 1, 33),
            HalfEdge::new(2, 6, 53),
            HalfEdge::new(6, 4, 39),
            HalfEdge::new(4, 2, 4),
            HalfEdge::new(2, 7, 38),
            HalfEdge::new(7, 6, 55),
            HalfEdge::new(6, 2, 48),
            HalfEdge::new(5, 6, 40),
            HalfEdge::new(6, 7, 52),
            HalfEdge::new(7, 5, 46),
            HalfEdge::new(9, 11, 7),
            HalfEdge::new(11, 12, 61),
            HalfEdge::new(12, 9, 22),
            HalfEdge::new(8, 12, 65),
            HalfEdge::new(12, 11, 58),
            HalfEdge::new(11, 8, 69),
            HalfEdge::new(8, 13, 71),
            HalfEdge::new(13, 12, 28),
            HalfEdge::new(12, 8, 60),
            HalfEdge::new(10, 13, 19),
            HalfEdge::new(13, 11, 70),
            HalfEdge::new(11, 10, 13),
            HalfEdge::new(8, 11, 62),
            HalfEdge::new(11, 13, 67),
            HalfEdge::new(13, 8, 63),
        ];

        let mut ps = vec![
            Vector3::new(-0.8, -1., -1.),
            Vector3::new(-0.8, -1., 1.),
            Vector3::new(-0.8, 1., -1.),
            Vector3::new(-0.8, 1., 1.),
            Vector3::new(1.2, -1., -1.),
            Vector3::new(1.2, -1., 1.),
            Vector3::new(1.2, 1., -1.),
            Vector3::new(1.2, 1., 1.),
            Vector3::new(0., 0., 0.),
            Vector3::new(-0.8, -0.14641, -0.14641),
            Vector3::new(-0.8, 0.2, 0.2),
            Vector3::new(-0.8, -0.34641, 0.2),
            Vector3::new(-0.8, 0., -0.4),
            Vector3::new(-0.8, 0.34641, 0.2),
        ];

        let mut ns = vec![
            Vector3::new(-0., -1., -0.),
            Vector3::new(-0., -0., -1.),
            Vector3::new(-1., -0., 0.),
            Vector3::new(-1., -0., 0.),
            Vector3::new(-1., -0., 0.),
            Vector3::new(-1., -0., 0.),
            Vector3::new(-1., -0., -0.),
            Vector3::new(-1., -0., -0.),
            Vector3::new(-1., -0., -0.),
            Vector3::new(-1., -0., -0.),
            Vector3::new(-1., -0., -0.),
            Vector3::new(0., 0., 1.),
            Vector3::new(0., 1., 0.),
            Vector3::new(1., 0., 0.),
            Vector3::new(0., -1., 0.),
            Vector3::new(-0., 0., 1.),
            Vector3::new(0., 0., -1.),
            Vector3::new(-0., 1., 0.),
            Vector3::new(1., 0., -0.),
            Vector3::new(-0.242536, 0.840168, 0.485071),
            Vector3::new(-0.242536, 0.840168, 0.485071),
            Vector3::new(-0.242536, -0.840168, 0.485071),
            Vector3::new(-0.242536, -3.22877e-20, -0.970143),
            Vector3::new(-0.242536, -3.22877e-20, -0.970143),
        ];

        let mut refs = vec![
            Tref {
                mid: 1,
                fid: 0,
                pid: 3,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 5,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 1,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 0,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 4,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 3,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 1,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 5,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 0,
            },
            Tref {
                mid: 1,
                fid: 0,
                pid: 4,
            },
            Tref {
                mid: 2,
                fid: 0,
                pid: 0,
            },
            Tref {
                mid: 2,
                fid: 0,
                pid: 0,
            },
            Tref {
                mid: 2,
                fid: 0,
                pid: 3,
            },
            Tref {
                mid: 2,
                fid: 0,
                pid: 2,
            },
            Tref {
                mid: 2,
                fid: 0,
                pid: 2,
            },
        ];

        collapse_collinear_edges(&mut hs, &mut ps, &mut ns, &mut refs, 9, 1e-6);

        let hs_out = vec![
            (0, 4, 5),
            (4, 1, 42),
            (1, 0, 9),
            (0, 2, 26),
            (2, 4, 50),
            (4, 0, 0),
            (0, 11, 11),
            (11, 12, 61),
            (12, 0, 24),
            (0, 1, 2),
            (1, 11, 17),
            (11, 0, 6),
            (-1, -1, -1),
            (-1, -1, -1),
            (-1, -1, -1),
            (1, 3, 35),
            (3, 11, 20),
            (11, 1, 10),
            (3, 13, 31),
            (13, 11, 70),
            (11, 3, 16),
            (-1, -1, -1),
            (-1, -1, -1),
            (-1, -1, -1),
            (0, 12, 8),
            (12, 2, 27),
            (2, 0, 3),
            (2, 12, 25),
            (12, 13, 64),
            (13, 2, 30),
            (2, 13, 29),
            (13, 3, 18),
            (3, 2, 36),
            (1, 7, 47),
            (7, 3, 37),
            (3, 1, 15),
            (2, 3, 32),
            (3, 7, 34),
            (7, 2, 51),
            (4, 6, 49),
            (6, 5, 54),
            (5, 4, 43),
            (1, 4, 1),
            (4, 5, 41),
            (5, 1, 45),
            (1, 5, 44),
            (5, 7, 56),
            (7, 1, 33),
            (2, 6, 53),
            (6, 4, 39),
            (4, 2, 4),
            (2, 7, 38),
            (7, 6, 55),
            (6, 2, 48),
            (5, 6, 40),
            (6, 7, 52),
            (7, 5, 46),
            (-1, -1, -1),
            (-1, -1, -1),
            (-1, -1, -1),
            (8, 12, 65),
            (12, 11, 7),
            (11, 8, 69),
            (8, 13, 71),
            (13, 12, 28),
            (12, 8, 60),
            (-1, -1, -1),
            (-1, -1, -1),
            (-1, -1, -1),
            (8, 11, 62),
            (11, 13, 19),
            (13, 8, 63),
        ];

        let ps_out = vec![
            Vector3::new(-0.8, -1., -1.),
            Vector3::new(-0.8, -1., 1.),
            Vector3::new(-0.8, 1., -1.),
            Vector3::new(-0.8, 1., 1.),
            Vector3::new(1.2, -1., -1.),
            Vector3::new(1.2, -1., 1.),
            Vector3::new(1.2, 1., -1.),
            Vector3::new(1.2, 1., 1.),
            Vector3::new(0., 0., 0.),
            Vector3::from_element(f64::NAN),
            Vector3::from_element(f64::NAN),
            Vector3::new(-0.8, -0.34641, 0.2),
            Vector3::new(-0.8, 0., -0.4),
            Vector3::new(-0.8, 0.34641, 0.2),
        ];

        for i in 0..hs.len() {
            let h = &hs[i];
            let a = hs[i].tail;
            let b = hs[i].head;
            let c = hs[i].pair;
            let (d, e, f) = hs_out[i];
            if h.tail().is_none() {
                assert_eq!(d, -1);
            } else {
                assert_eq!(a.to_u32() as i32, d);
            }
            if h.head().is_none() {
                assert_eq!(e, -1);
            } else {
                assert_eq!(b.to_u32() as i32, e);
            }
            if h.pair().is_none() {
                assert_eq!(f, -1);
            } else {
                assert_eq!(c.to_u32() as i32, f);
            }
        }

        for i in 0..ps.len() {
            let p0 = &ps[i];
            let p1 = &ps_out[i];
            if !p0.x.is_nan() && !p0.y.is_nan() && p0.z.is_nan() {
                assert!((p0 - p1).norm() < 1e-6);
            }
        }
    }
}

#[cfg(test)]
mod test_mesh_cleanup {
    use crate::prelude::Manifold;

    #[test]
    fn test_dedup_verts() {
        let pos = vec![
            -0.866025, -1., 0.5, // duplicated
            0., -1., -1., 0.866025, -1., 0.5, // duplicated 2
            -0.866025, -1., 0.5, // duplicated
            0., 1., 0., 0.866025, -1., 0.5, // duplicated 2
        ];
        let idx = vec![
            0, 4, 1, 0, 3, 1, // collapsed
            0, 3, 2, // collapsed
            0, 3, 4, // collapsed
            1, 2, 0, 1, 4, 2, 2, 4, 0, 2, 5, 0, // collapsed
            5, 2, 1, // collapsed
        ];
        let mfd = Manifold::new(&pos, &idx).unwrap();

        assert_eq!(mfd.nv, 4);
        assert_eq!(mfd.nf, 4);
    }
}

#[cfg(test)]
mod test_projection {
    use crate::compute_projection;
    use crate::prelude::{compose, generate_cube, Manifold};

    #[test]
    fn test_projection_cube() {
        let cube = generate_cube::<f64>().unwrap();
        let result = compute_projection(&cube);
        assert!(result.is_ok());
        let polygon = result.unwrap();
        assert!(!polygon.0.is_empty());
    }

    #[test]
    fn test_projection_composed() {
        let a = generate_cube::<f64>().unwrap();
        let b = generate_cube::<f64>().unwrap();
        let b = b.translate(2.0, 0.0, 0.0).unwrap();
        let meshes: Vec<Manifold<f64>> = vec![a, b];
        let composed = compose(&meshes).unwrap();
        let result = compute_projection(&composed);
        assert!(result.is_ok());
        let polygon = result.unwrap();
        assert!(!polygon.0.is_empty());
    }
}
