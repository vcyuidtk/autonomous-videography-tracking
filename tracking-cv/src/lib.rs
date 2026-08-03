//! `tracking-cv` — the only crate in this repo that links `opencv-rust`.
//! [`CsrtTracker`] implements `tracking_core::Tracker` against
//! `opencv::tracking::TrackerCSRT` (contrib). This is the production-grade
//! visual tracker backend; `tracking_core::SimpleTracker` is the
//! dependency-free stand-in used for the core crate's own tests and the
//! eval harness.
//!
//! Ported (bus/IPC context stripped) from the original
//! `autonomous-videography` monorepo's `av-cv::CsrtTracker` — see README
//! "Provenance".

use opencv::core::{Mat, Ptr, Rect};
use opencv::prelude::*;
use opencv::tracking::{TrackerCSRT, TrackerCSRT_Params};
use tracking_core::{BBox, FrameView, SeedBox, Tracker, TrackerError};

/// Single-target visual tracker wrapping `opencv::tracking::TrackerCSRT`.
#[derive(Default)]
pub struct CsrtTracker {
    impl_: Option<Ptr<TrackerCSRT>>,
}

impl CsrtTracker {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Build an owned `cv::Mat` from a [`FrameView`] — OpenCV retains a template
/// reference from `init()` for the tracker's lifetime, so this must be a
/// private copy: a shared-memory view would otherwise be pinned open for as
/// long as CSRT holds it, well past whatever buffer `frame` borrows from.
fn to_owned_mat(frame: FrameView<'_>) -> Result<Mat, TrackerError> {
    // SAFETY: `frame.pixels` is exactly `height * width * channels` bytes,
    // row-major, no padding (the `FrameView` contract) — matches what
    // `Mat::new_rows_cols_with_bytes` requires for a `CV_8UC{channels}`
    // buffer, and the constructor copies immediately rather than retaining
    // a pointer into it.
    let borrowed = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            frame.height as i32,
            frame.width as i32,
            opencv::core::CV_8UC(frame.channels as i32),
            frame.pixels.as_ptr() as *mut std::ffi::c_void,
            opencv::core::Mat_AUTO_STEP,
        )
        .map_err(|e| TrackerError::Init(format!("Mat::new_rows_cols_with_data_unsafe: {e}")))?
    };
    borrowed.try_clone().map_err(|e| TrackerError::Init(format!("Mat::try_clone: {e}")))
}

impl Tracker for CsrtTracker {
    fn is_active(&self) -> bool {
        self.impl_.is_some()
    }

    fn reinit(&mut self, frame: FrameView<'_>, seed: SeedBox) -> Result<(), TrackerError> {
        let b = seed.get();
        let rect = Rect::new(b.x1 as i32, b.y1 as i32, b.width() as i32, b.height() as i32);
        let mat = to_owned_mat(frame)?;
        let params = TrackerCSRT_Params::default().map_err(|e| TrackerError::Init(format!("TrackerCSRT_Params::default: {e}")))?;
        let mut t = TrackerCSRT::create(&params).map_err(|e| TrackerError::Init(format!("TrackerCSRT::create: {e}")))?;
        t.init(&mat, rect).map_err(|e| TrackerError::Init(e.to_string()))?;
        self.impl_ = Some(t);
        Ok(())
    }

    fn update(&mut self, frame: FrameView<'_>) -> Result<Option<BBox>, TrackerError> {
        let Some(t) = self.impl_.as_mut() else {
            return Ok(None);
        };
        let mat = to_owned_mat(frame)?;
        let mut rect = Rect::default();
        let ok = t.update(&mat, &mut rect).map_err(|e| TrackerError::Update(e.to_string()))?;
        if !ok {
            return Ok(None);
        }
        Ok(Some(BBox::new(
            rect.x as i64,
            rect.y as i64,
            (rect.x + rect.width) as i64,
            (rect.y + rect.height) as i64,
        )))
    }

    fn reset(&mut self) {
        self.impl_ = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracking_core::seed::FrameDims;

    /// A synthetic RGB8 frame with a distinguishable bright square at a
    /// known position — enough texture for CSRT's correlation filter to
    /// lock onto, unlike a flat frame.
    fn synthetic_frame(width: u32, height: u32) -> Vec<u8> {
        let mut buf = vec![40u8; (width * height * 3) as usize];
        for y in 40..80.min(height) {
            for x in 40..80.min(width) {
                let i = ((y * width + x) * 3) as usize;
                buf[i] = 220;
                buf[i + 1] = 220;
                buf[i + 2] = 220;
            }
        }
        buf
    }

    #[test]
    fn is_active_false_until_reinit() {
        let t = CsrtTracker::new();
        assert!(!t.is_active());
    }

    #[test]
    fn reinit_then_update_tracks_a_real_frame() {
        let (w, h) = (200, 200);
        let pixels = synthetic_frame(w, h);
        let frame = FrameView {
            pixels: &pixels,
            width: w,
            height: h,
            channels: 3,
        };
        let seed = SeedBox::new(
            BBox::new(35, 35, 85, 85),
            FrameDims {
                width: w as i64,
                height: h as i64,
            },
        )
        .unwrap();

        let mut t = CsrtTracker::new();
        t.reinit(frame, seed).expect("real OpenCV CSRT init on a real frame must succeed");
        assert!(t.is_active());

        let result = t.update(frame).expect("update must not error on the same frame");
        assert!(
            result.is_some(),
            "CSRT should still report the target on the very next (identical) frame"
        );
    }

    #[test]
    fn update_before_reinit_returns_none_not_an_error() {
        let (w, h) = (64, 64);
        let pixels = synthetic_frame(w, h);
        let frame = FrameView {
            pixels: &pixels,
            width: w,
            height: h,
            channels: 3,
        };
        let mut t = CsrtTracker::new();
        assert_eq!(t.update(frame).unwrap(), None);
    }

    #[test]
    fn reset_deactivates_the_tracker() {
        let (w, h) = (200, 200);
        let pixels = synthetic_frame(w, h);
        let frame = FrameView {
            pixels: &pixels,
            width: w,
            height: h,
            channels: 3,
        };
        let seed = SeedBox::new(
            BBox::new(35, 35, 85, 85),
            FrameDims {
                width: w as i64,
                height: h as i64,
            },
        )
        .unwrap();
        let mut t = CsrtTracker::new();
        t.reinit(frame, seed).unwrap();
        assert!(t.is_active());
        t.reset();
        assert!(!t.is_active());
    }
}
