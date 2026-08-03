//! Tracker-seedability gates.
//!
//! Correlation-filter trackers such as CSRT hard-fail (an assertion, not a
//! graceful error, in OpenCV's C++) if asked to seed a box that is too
//! small, too extreme an aspect ratio, or clipped to a sliver by the frame
//! edge. This module makes "a box the tracker can actually seed" a
//! constructible type ([`SeedBox`]) rather than a runtime hope, so a
//! nomination that would crash a real backend is rejected here, before it
//! ever reaches [`crate::tracker::Tracker::reinit`].
//!
//! Two independent checks, both load-bearing:
//! 1. Requested extent/aspect — a pure function of the box as asked for,
//!    nothing to do with the frame.
//! 2. In-frame extent after clipping — proportional to the requested size,
//!    not a flat constant (a correlation tracker scales its template from
//!    the box as REQUESTED, then crops to the frame — a naive flat-2px rule
//!    under-rejects boxes that are mostly off-frame).
//!
//! Ported (with the surrounding IPC/bus context stripped) from the original
//! `autonomous-videography` monorepo's `av-track::seed` — see README
//! "Provenance". The specific thresholds below were derived empirically
//! against OpenCV's `TrackerCSRT`; a different backend may need different
//! constants, which is exactly why this is a gate the algorithm core owns
//! and calls before handing off to whatever `Tracker` is plugged in.

use crate::types::BBox;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDims {
    pub width: i64,
    pub height: i64,
}

/// Smallest REQUESTED extent (px, per axis) that has a safe answer at all.
const MIN_SEEDABLE_REQUEST_PX: i64 = 3;

/// Largest requested aspect ratio (long side / short side) a correlation
/// tracker will reliably seed.
const MAX_SEEDABLE_ASPECT: i64 = 20;

/// Minimum in-frame extent (px) needed to seed a `requested`-px extent.
/// `3 + ceil(requested / 7)`, clamped to `requested` itself and to at least
/// 2 — both clamps are load-bearing: the bare fit is unsatisfiable below
/// ~4 px.
fn min_in_frame_extent(requested: i64) -> i64 {
    let fit = 3 + (requested + 6) / 7;
    2.max(requested.min(fit))
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum SeedError {
    #[error("requested extent {req_w}x{req_h} px below the {min} px seedability minimum (box={box_:?})")]
    ExtentTooSmall { box_: BBox, req_w: i64, req_h: i64, min: i64 },
    #[error("requested aspect above the {max_aspect}:1 seedability maximum (box={box_:?})")]
    AspectTooExtreme { box_: BBox, max_aspect: i64 },
    #[error(
        "in-frame extent {in_w}x{in_h} px below the {need_w}x{need_h} px minimum for a {req_w}x{req_h} px box (box={box_:?}, frame={frame_w}x{frame_h})"
    )]
    InFrameExtentTooSmall {
        box_: BBox,
        in_w: i64,
        in_h: i64,
        need_w: i64,
        need_h: i64,
        req_w: i64,
        req_h: i64,
        frame_w: i64,
        frame_h: i64,
    },
}

/// A box a real correlation tracker can actually seed a template from — the
/// only type [`crate::tracker::Tracker::reinit`] accepts (made a
/// compile-time precondition rather than a runtime check someone has to
/// remember to call).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeedBox(BBox);

impl SeedBox {
    /// Three independent checks, in order: requested extent, requested
    /// aspect, then in-frame extent after clipping to `frame`.
    pub fn new(b: BBox, frame: FrameDims) -> Result<Self, SeedError> {
        let (req_w, req_h) = (b.width(), b.height());
        if req_w < MIN_SEEDABLE_REQUEST_PX || req_h < MIN_SEEDABLE_REQUEST_PX {
            return Err(SeedError::ExtentTooSmall {
                box_: b,
                req_w,
                req_h,
                min: MIN_SEEDABLE_REQUEST_PX,
            });
        }
        if req_w.max(req_h) > MAX_SEEDABLE_ASPECT * req_w.min(req_h) {
            return Err(SeedError::AspectTooExtreme {
                box_: b,
                max_aspect: MAX_SEEDABLE_ASPECT,
            });
        }
        let in_w = b.x2.min(frame.width) - b.x1.max(0);
        let in_h = b.y2.min(frame.height) - b.y1.max(0);
        let (need_w, need_h) = (min_in_frame_extent(req_w), min_in_frame_extent(req_h));
        if in_w < need_w || in_h < need_h {
            return Err(SeedError::InFrameExtentTooSmall {
                box_: b,
                in_w,
                in_h,
                need_w,
                need_h,
                req_w,
                req_h,
                frame_w: frame.width,
                frame_h: frame.height,
            });
        }
        Ok(SeedBox(b))
    }

    pub fn get(&self) -> BBox {
        self.0
    }
}

/// Why a raw operator/external "force lock to this box" command is unusable,
/// or `None` if it is fine. Distinct from [`SeedBox::new`]: this is an
/// UNTRUSTED-INPUT validity check (strict in-frame bounds — a click outside
/// the frame is a command error worth rejecting outright), meant to be
/// applied before [`SeedBox::new`]'s tracker-CAPABILITY check runs on the
/// same box.
pub fn force_lock_box_error(b: BBox, frame: FrameDims) -> Option<String> {
    if b.x2 <= b.x1 || b.y2 <= b.y1 {
        return Some(format!("degenerate box (width={}, height={})", b.x2 - b.x1, b.y2 - b.y1));
    }
    if b.x1 < 0 || b.y1 < 0 || b.x2 > frame.width || b.y2 > frame.height {
        return Some(format!("box outside {}x{} frame bounds", frame.width, frame.height));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: FrameDims = FrameDims { width: 200, height: 200 };

    #[test]
    fn a_1px_extent_never_seeds() {
        let err = SeedBox::new(BBox::new(10, 10, 11, 10), FRAME).unwrap_err();
        assert!(matches!(err, SeedError::ExtentTooSmall { .. }));
    }

    #[test]
    fn a_zero_in_frame_extent_is_rejected() {
        let err = SeedBox::new(BBox::new(-50, 10, -40, 20), FRAME).unwrap_err();
        assert!(matches!(err, SeedError::InFrameExtentTooSmall { .. }) || matches!(err, SeedError::ExtentTooSmall { .. }));
    }

    #[test]
    fn a_20x20_box_clipped_to_4_in_frame_px_hard_fails() {
        let b = BBox::new(-16, 10, 4, 30); // requested 20x20, in-frame 4x20
        let err = SeedBox::new(b, FRAME).unwrap_err();
        assert!(matches!(err, SeedError::InFrameExtentTooSmall { .. }));
    }

    #[test]
    fn a_healthy_interior_box_seeds() {
        assert!(SeedBox::new(BBox::new(50, 50, 100, 100), FRAME).is_ok());
    }

    #[test]
    fn an_overhanging_but_adequately_sized_box_seeds() {
        assert!(SeedBox::new(BBox::new(180, 50, 220, 90), FRAME).is_ok());
    }

    #[test]
    fn extreme_aspect_ratio_is_rejected_even_fully_in_frame() {
        let err = SeedBox::new(BBox::new(50, 50, 53, 190), FRAME).unwrap_err(); // 3x140
        assert!(matches!(err, SeedError::AspectTooExtreme { .. }));
    }

    #[test]
    fn a_3px_box_seeds_via_the_min_clamp() {
        assert!(SeedBox::new(BBox::new(50, 50, 53, 72), FRAME).is_ok());
    }

    #[test]
    fn force_lock_rejects_degenerate_box() {
        assert!(force_lock_box_error(BBox::new(10, 10, 10, 20), FRAME).is_some());
    }

    #[test]
    fn force_lock_rejects_out_of_frame_box() {
        assert!(force_lock_box_error(BBox::new(10, 10, 250, 50), FRAME).is_some());
    }

    #[test]
    fn force_lock_accepts_a_valid_in_frame_box() {
        assert!(force_lock_box_error(BBox::new(10, 10, 50, 50), FRAME).is_none());
    }
}
