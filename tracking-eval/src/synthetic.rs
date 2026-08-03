//! Synthetic frame-sequence generation with known ground truth. No real
//! video/images involved — each frame is a flat background with one or two
//! flat-colour axis-aligned squares rendered directly into an RGB8 buffer.
//! That's enough texture for both [`tracking_core::SimpleTracker`]'s SAD
//! template matcher and a real correlation filter to lock onto, while
//! keeping generation trivial and fully deterministic.

use crate::rng::Rng;
use tracking_core::{BBox, Detection, FrameView};

pub const WIDTH: u32 = 320;
pub const HEIGHT: u32 = 240;
const TARGET_SIZE: i64 = 28;
const DISTRACTOR_SIZE: i64 = 28;

/// One synthetic object's ground-truth box at a given tick.
#[derive(Debug, Clone, Copy)]
pub struct GroundTruth {
    pub bbox: BBox,
    /// `false` during a simulated occlusion window — the object still
    /// exists but no detector would report it and its pixels are not drawn.
    pub visible: bool,
}

/// A generated sequence: ground truth for the target (and, in the
/// distractor scenario, a second decoy object) plus per-tick rendered
/// frames and detector output.
pub struct Sequence {
    pub name: &'static str,
    pub frames: Vec<Vec<u8>>,
    pub target_gt: Vec<GroundTruth>,
    pub distractor_gt: Vec<Option<GroundTruth>>,
    /// This tick's detector output — `None` when the detector didn't run
    /// this tick (cadence gap); `Some(vec![])` when it ran but produced
    /// nothing (false negative / occluded). See
    /// `tracking_core::TrackingSession::step`'s docs for why that
    /// distinction matters to two-batch confirmation.
    pub detections: Vec<Option<Vec<Detection>>>,
}

impl Sequence {
    pub fn frame_view(&self, tick: usize) -> FrameView<'_> {
        FrameView {
            pixels: &self.frames[tick],
            width: WIDTH,
            height: HEIGHT,
            channels: 3,
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

fn blank_frame() -> Vec<u8> {
    vec![25u8; (WIDTH * HEIGHT * 3) as usize]
}

fn draw_square(buf: &mut [u8], b: BBox, color: (u8, u8, u8)) {
    let (w, h) = (WIDTH as i64, HEIGHT as i64);
    for y in b.y1.max(0)..b.y2.min(h) {
        for x in b.x1.max(0)..b.x2.min(w) {
            let i = ((y * w + x) * 3) as usize;
            buf[i] = color.0;
            buf[i + 1] = color.1;
            buf[i + 2] = color.2;
        }
    }
}

fn clamp_box(cx: i64, cy: i64, size: i64) -> BBox {
    let half = size / 2;
    let x1 = cx - half;
    let y1 = cy - half;
    BBox::new(x1, y1, x1 + size, y1 + size)
}

/// Straight-line motion, no occlusion, dense (every-tick) detections with
/// small jitter — the easy baseline case.
pub fn straight_line(num_ticks: usize, seed: u64) -> Sequence {
    build(
        "straight_line",
        num_ticks,
        seed,
        |t| {
            let cx = 40 + (t as i64 * 2) % (WIDTH as i64 - 80) + 40;
            let cy = HEIGHT as i64 / 2;
            (clamp_box(cx, cy, TARGET_SIZE), true)
        },
        DetectorParams {
            cadence: 1,
            jitter_px: 2,
            false_negative_prob: 0.0,
        },
        None,
    )
}

/// Same trajectory, but the target is occluded (undetectable, undrawn) for
/// a stretch in the middle, and the detector only runs every 3rd tick with
/// some jitter and dropout — exercises loss/age-out/re-acquisition.
pub fn occlusion_and_sparse_detections(num_ticks: usize, seed: u64) -> Sequence {
    let occlude_start = num_ticks / 3;
    let occlude_end = occlude_start + num_ticks / 6;
    build(
        "occlusion_and_sparse_detections",
        num_ticks,
        seed,
        move |t| {
            let cx = 40 + (t as i64 * 2) % (WIDTH as i64 - 80) + 40;
            let cy = HEIGHT as i64 / 2;
            let visible = !(occlude_start..occlude_end).contains(&t);
            (clamp_box(cx, cy, TARGET_SIZE), visible)
        },
        DetectorParams {
            cadence: 3,
            jitter_px: 3,
            false_negative_prob: 0.1,
        },
        None,
    )
}

/// The target moves alongside a similarly-sized, similarly-confident
/// distractor that crosses close to it mid-sequence — the class-agnostic
/// acquisition/refresh policy's stress case (does the lock ever jump to the
/// wrong object?).
pub fn distractor_crossing(num_ticks: usize, seed: u64) -> Sequence {
    build(
        "distractor_crossing",
        num_ticks,
        seed,
        |t| {
            let cx = 40 + (t as i64 * 2) % (WIDTH as i64 - 80) + 40;
            let cy = HEIGHT as i64 / 3;
            (clamp_box(cx, cy, TARGET_SIZE), true)
        },
        DetectorParams {
            cadence: 1,
            jitter_px: 2,
            false_negative_prob: 0.0,
        },
        Some(Box::new(|t: usize| {
            // Crosses the target's row around the midpoint, moving the
            // opposite direction, close enough to be a real confusion risk.
            let cx = (WIDTH as i64 - 40) - (t as i64 * 2) % (WIDTH as i64 - 80);
            let cy = HEIGHT as i64 / 3 + 6;
            clamp_box(cx, cy, DISTRACTOR_SIZE)
        })),
    )
}

struct DetectorParams {
    /// Detector runs once every `cadence` ticks.
    cadence: usize,
    jitter_px: i64,
    false_negative_prob: f64,
}

fn build(
    name: &'static str,
    num_ticks: usize,
    seed: u64,
    target_traj: impl Fn(usize) -> (BBox, bool),
    det: DetectorParams,
    distractor_traj: Option<Box<dyn Fn(usize) -> BBox>>,
) -> Sequence {
    let mut rng = Rng::new(seed);
    let mut frames = Vec::with_capacity(num_ticks);
    let mut target_gt = Vec::with_capacity(num_ticks);
    let mut distractor_gt = Vec::with_capacity(num_ticks);
    let mut detections = Vec::with_capacity(num_ticks);
    let mut next_det_id = 0u64;

    for t in 0..num_ticks {
        let (tb, visible) = target_traj(t);
        let mut frame = blank_frame();
        if visible {
            draw_square(&mut frame, tb, (230, 230, 230));
        }
        let dgt = distractor_traj.as_ref().map(|f| {
            let db = f(t);
            draw_square(&mut frame, db, (215, 215, 215));
            GroundTruth { bbox: db, visible: true }
        });

        let detector_ran = t % det.cadence == 0;
        let dets = if detector_ran {
            let mut batch = Vec::new();
            if visible && !rng.chance(det.false_negative_prob) {
                let mut jitter = |c: i64| c + rng.range_i64(-det.jitter_px, det.jitter_px);
                let jb = BBox::new(jitter(tb.x1), jitter(tb.y1), jitter(tb.x2), jitter(tb.y2));
                batch.push(Detection {
                    bbox: jb,
                    confidence: tracking_core::Confidence::clamp(0.85),
                    id: next_det_id,
                });
                next_det_id += 1;
            }
            if let Some(GroundTruth { bbox: db, .. }) = dgt {
                if !rng.chance(det.false_negative_prob) {
                    batch.push(Detection {
                        bbox: db,
                        confidence: tracking_core::Confidence::clamp(0.85),
                        id: next_det_id,
                    });
                    next_det_id += 1;
                }
            }
            Some(batch)
        } else {
            None
        };

        frames.push(frame);
        target_gt.push(GroundTruth { bbox: tb, visible });
        distractor_gt.push(dgt);
        detections.push(dets);
    }

    Sequence {
        name,
        frames,
        target_gt,
        distractor_gt,
        detections,
    }
}
