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

use std::sync::Once;

use opencv::boxed_ref::BoxedRef;
use opencv::core::{Mat, Ptr, Rect, Vec3b, Vec4b};
use opencv::prelude::*;
use opencv::tracking::{TrackerCSRT, TrackerCSRT_Params};
use tracking_core::{BBox, FrameView, SeedBox, Tracker, TrackerError};

/// Ceiling on OpenCV's internal thread pool, applied process-wide before the
/// first tracker is built. OpenCV defaults `cv::parallel_for_` to one worker
/// per core, and CSRT on a single small ROI scales badly across them: on a
/// 320x240 clip, uncapped costs 2.5x the core occupancy (390% vs 155%) to buy
/// 34% per-frame latency (11.2ms vs 16.9ms). Those cores are not free — the
/// rest of the stack shares the SBC — and 16.9ms still clears a 30fps budget.
const OPENCV_THREAD_CAP: i32 = 2;

static THREAD_CAP: Once = Once::new();

/// Idempotent, process-wide: `cv::setNumThreads` is global state, not
/// per-tracker. A failure here is not a tracking failure — the tracker runs
/// correctly, just uncapped — so it is deliberately not propagated.
fn cap_opencv_threads() {
    THREAD_CAP.call_once(|| {
        let _ = opencv::core::set_num_threads(OPENCV_THREAD_CAP);
    });
}

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

/// Wrap a [`FrameView`]'s pixels in a `cv::Mat` **without copying** — the
/// returned handle borrows the caller's buffer, and the borrow is checked:
/// it cannot outlive `frame`.
///
/// Only 1/3/4-channel 8-bit frames are representable this way; anything else
/// is rejected rather than silently reinterpreted.
fn borrowed_mat(frame: FrameView<'_>) -> Result<BoxedRef<'_, Mat>, String> {
    let (rows, cols) = (frame.height as i32, frame.width as i32);
    match frame.channels {
        1 => Mat::new_rows_cols_with_bytes::<u8>(rows, cols, frame.pixels),
        3 => Mat::new_rows_cols_with_bytes::<Vec3b>(rows, cols, frame.pixels),
        4 => Mat::new_rows_cols_with_bytes::<Vec4b>(rows, cols, frame.pixels),
        n => return Err(format!("unsupported channel count: {n}")),
    }
    .map_err(|e| format!("Mat::new_rows_cols_with_bytes: {e}"))
}

/// Copy a frame into a `Mat` this crate owns. Required on the reinit path
/// only: `TrackerCSRT::init` retains its template frame for the tracker's
/// lifetime, so a borrowed view would pin the caller's buffer open for as
/// long as CSRT holds it. `update` retains nothing past the call, so it
/// borrows via [`borrowed_mat`] instead of paying a per-frame copy.
fn to_owned_mat(frame: FrameView<'_>) -> Result<Mat, String> {
    borrowed_mat(frame)?.try_clone().map_err(|e| format!("Mat::try_clone: {e}"))
}

impl Tracker for CsrtTracker {
    fn is_active(&self) -> bool {
        self.impl_.is_some()
    }

    fn reinit(&mut self, frame: FrameView<'_>, seed: SeedBox) -> Result<(), TrackerError> {
        cap_opencv_threads();
        let b = seed.get();
        let rect = Rect::new(b.x1 as i32, b.y1 as i32, b.width() as i32, b.height() as i32);
        let mat = to_owned_mat(frame).map_err(TrackerError::Init)?;
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
        let mat = borrowed_mat(frame).map_err(TrackerError::Update)?;
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

    /// The per-frame path must alias the caller's buffer, the reinit path
    /// must not — that split is the whole point of having two constructors.
    #[test]
    fn borrowed_mat_aliases_the_caller_buffer_and_to_owned_mat_does_not() {
        let (w, h) = (64, 64);
        let pixels = synthetic_frame(w, h);
        let frame = FrameView {
            pixels: &pixels,
            width: w,
            height: h,
            channels: 3,
        };

        let borrowed = borrowed_mat(frame).unwrap();
        assert_eq!(
            borrowed.data(),
            pixels.as_ptr(),
            "update's Mat must point straight at the caller's pixels — a copy here is a per-frame regression"
        );

        let owned = to_owned_mat(frame).unwrap();
        assert_ne!(owned.data(), pixels.as_ptr(), "reinit's Mat must be a private copy");
    }

    #[test]
    fn borrowed_mat_accepts_supported_channel_counts_only() {
        let (w, h) = (32u32, 32u32);
        for (channels, expect_ok, desc) in [
            (1u32, true, "8-bit grayscale"),
            (3, true, "RGB8/BGR8"),
            (4, true, "RGBA8"),
            (2, false, "2 channels is not a frame format this backend accepts"),
            (0, false, "zero channels is degenerate"),
        ] {
            let pixels = vec![0u8; (w * h * channels.max(1)) as usize];
            let frame = FrameView {
                pixels: &pixels,
                width: w,
                height: h,
                channels,
            };
            assert_eq!(borrowed_mat(frame).is_ok(), expect_ok, "{desc}");
        }
    }

    /// CSRT keeps its template frame for the tracker's lifetime, so `reinit`
    /// must copy: the buffer it was seeded from is gone by the time `update`
    /// runs here.
    #[test]
    fn reinit_survives_its_seed_frame_being_freed() {
        let (w, h) = (200, 200);
        let seed = SeedBox::new(
            BBox::new(35, 35, 85, 85),
            FrameDims {
                width: w as i64,
                height: h as i64,
            },
        )
        .unwrap();

        let mut t = CsrtTracker::new();
        {
            let seed_pixels = synthetic_frame(w, h);
            t.reinit(
                FrameView {
                    pixels: &seed_pixels,
                    width: w,
                    height: h,
                    channels: 3,
                },
                seed,
            )
            .unwrap();
        }

        let later_pixels = synthetic_frame(w, h);
        let result = t
            .update(FrameView {
                pixels: &later_pixels,
                width: w,
                height: h,
                channels: 3,
            })
            .expect("update must not error after the seed frame's buffer is gone");
        assert!(result.is_some(), "the target is still present, so CSRT should still report it");
    }

    #[test]
    fn reinit_caps_the_opencv_thread_pool() {
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
        assert!(
            opencv::core::get_num_threads().unwrap() <= OPENCV_THREAD_CAP,
            "cv::parallel_for_ must not be left free to fan out across every core"
        );
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
