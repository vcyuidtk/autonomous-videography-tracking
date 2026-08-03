//! [`SimpleTracker`]: a dependency-free reference [`Tracker`] implementation
//! — grayscale template matching (sum-of-absolute-differences) over a
//! bounded search window around the last known position.
//!
//! **This is a demo/test/eval-harness tracker, not a production one.** It
//! exists so `tracking-core` is fully runnable — tests, docs, `cargo run
//! --bin eval` — with zero system dependencies. A real deployment should use
//! a proper correlation-filter or learned tracker (the sibling `tracking-cv`
//! crate wraps OpenCV CSRT for that) via the same [`Tracker`] trait; nothing
//! else in this crate cares which one is plugged in.

use crate::seed::SeedBox;
use crate::tracker::{FrameView, Tracker, TrackerError};
use crate::types::BBox;

/// How many pixels beyond the template's own footprint to search, per axis,
/// per `update()` call. Bounds the O(search_area * template_area) cost and
/// caps how fast a target can move (in px/frame) before this tracker loses
/// it — deliberately generous for synthetic eval sequences.
const SEARCH_MARGIN_PX: i64 = 24;

/// SAD (sum of absolute differences), normalised by pixel count, above which
/// a search-window match is rejected as "not the target" rather than
/// accepted as a (bad) match. Grayscale 0-255 scale.
const MAX_MEAN_SAD: f64 = 40.0;

struct Template {
    width: i64,
    height: i64,
    /// Grayscale (luma) samples, row-major, `width * height` long.
    gray: Vec<u8>,
}

fn to_gray(pixels: &[u8], width: u32, height: u32, channels: u32, roi: BBox) -> Vec<u8> {
    let (w, h) = (width as i64, height as i64);
    let mut out = Vec::with_capacity((roi.width() * roi.height()) as usize);
    for y in roi.y1..roi.y2 {
        for x in roi.x1..roi.x2 {
            if x < 0 || y < 0 || x >= w || y >= h {
                out.push(0);
                continue;
            }
            let i = ((y * w + x) * channels as i64) as usize;
            if channels >= 3 {
                let (r, g, b) = (pixels[i] as u32, pixels[i + 1] as u32, pixels[i + 2] as u32);
                out.push(((r * 299 + g * 587 + b * 114) / 1000) as u8);
            } else {
                out.push(pixels[i]);
            }
        }
    }
    out
}

/// A grayscale-SAD template-matching tracker. See module docs — not a
/// production-grade tracker, a dependency-free reference/eval one.
#[derive(Default)]
pub struct SimpleTracker {
    template: Option<Template>,
    current_box: Option<BBox>,
}

impl SimpleTracker {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Tracker for SimpleTracker {
    fn is_active(&self) -> bool {
        self.template.is_some()
    }

    fn reinit(&mut self, frame: FrameView<'_>, seed: SeedBox) -> Result<(), TrackerError> {
        let b = seed.get();
        let gray = to_gray(frame.pixels, frame.width, frame.height, frame.channels, b);
        self.template = Some(Template {
            width: b.width(),
            height: b.height(),
            gray,
        });
        self.current_box = Some(b);
        Ok(())
    }

    fn update(&mut self, frame: FrameView<'_>) -> Result<Option<BBox>, TrackerError> {
        let (Some(t), Some(cur)) = (&self.template, self.current_box) else {
            return Ok(None);
        };
        let (fw, fh) = (frame.width as i64, frame.height as i64);
        let mut best: Option<(i64, i64, f64)> = None;
        for dy in -SEARCH_MARGIN_PX..=SEARCH_MARGIN_PX {
            for dx in -SEARCH_MARGIN_PX..=SEARCH_MARGIN_PX {
                let x1 = cur.x1 + dx;
                let y1 = cur.y1 + dy;
                let candidate = BBox::new(x1, y1, x1 + t.width, y1 + t.height);
                if candidate.x1 < 0 || candidate.y1 < 0 || candidate.x2 > fw || candidate.y2 > fh {
                    continue;
                }
                let gray = to_gray(frame.pixels, frame.width, frame.height, frame.channels, candidate);
                let sad: u64 = gray.iter().zip(&t.gray).map(|(a, b)| (*a as i64 - *b as i64).unsigned_abs()).sum();
                let mean_sad = sad as f64 / gray.len().max(1) as f64;
                if best.map(|(_, _, s)| mean_sad < s).unwrap_or(true) {
                    best = Some((x1, y1, mean_sad));
                }
            }
        }
        match best {
            Some((x1, y1, score)) if score <= MAX_MEAN_SAD => {
                let new_box = BBox::new(x1, y1, x1 + t.width, y1 + t.height);
                self.current_box = Some(new_box);
                Ok(Some(new_box))
            }
            _ => Ok(None),
        }
    }

    fn reset(&mut self) {
        self.template = None;
        self.current_box = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::FrameDims;

    /// A flat background with a distinct bright square, movable by `offset`.
    fn synthetic_frame(width: u32, height: u32, offset: i64) -> Vec<u8> {
        let mut buf = vec![30u8; (width * height * 3) as usize];
        let (x0, y0) = (40 + offset, 40);
        for y in y0..(y0 + 30).min(height as i64) {
            for x in x0..(x0 + 30).min(width as i64) {
                if x < 0 || y < 0 {
                    continue;
                }
                let i = ((y * width as i64 + x) * 3) as usize;
                buf[i] = 230;
                buf[i + 1] = 230;
                buf[i + 2] = 230;
            }
        }
        buf
    }

    #[test]
    fn is_active_false_until_reinit() {
        assert!(!SimpleTracker::new().is_active());
    }

    #[test]
    fn tracks_a_target_that_moves_a_few_pixels_per_frame() {
        let (w, h) = (200u32, 200u32);
        let mut t = SimpleTracker::new();
        let f0 = synthetic_frame(w, h, 0);
        let seed = SeedBox::new(
            BBox::new(40, 40, 70, 70),
            FrameDims {
                width: w as i64,
                height: h as i64,
            },
        )
        .unwrap();
        t.reinit(
            FrameView {
                pixels: &f0,
                width: w,
                height: h,
                channels: 3,
            },
            seed,
        )
        .unwrap();

        for step in 1..=5 {
            let f = synthetic_frame(w, h, step * 3);
            let got = t
                .update(FrameView {
                    pixels: &f,
                    width: w,
                    height: h,
                    channels: 3,
                })
                .unwrap();
            assert!(got.is_some(), "tracker lost the target at step {step}");
        }
    }

    #[test]
    fn update_before_reinit_returns_none_not_an_error() {
        let (w, h) = (64u32, 64u32);
        let pixels = synthetic_frame(w, h, 0);
        let mut t = SimpleTracker::new();
        assert_eq!(
            t.update(FrameView {
                pixels: &pixels,
                width: w,
                height: h,
                channels: 3
            })
            .unwrap(),
            None
        );
    }

    #[test]
    fn reset_deactivates_the_tracker() {
        let (w, h) = (200u32, 200u32);
        let f0 = synthetic_frame(w, h, 0);
        let seed = SeedBox::new(
            BBox::new(40, 40, 70, 70),
            FrameDims {
                width: w as i64,
                height: h as i64,
            },
        )
        .unwrap();
        let mut t = SimpleTracker::new();
        t.reinit(
            FrameView {
                pixels: &f0,
                width: w,
                height: h,
                channels: 3,
            },
            seed,
        )
        .unwrap();
        assert!(t.is_active());
        t.reset();
        assert!(!t.is_active());
    }
}
