//! [`TrackingSession`]: the crate's headline entry point. Push a frame (plus
//! optional fresh detection boxes) in, get this tick's [`TrackState`] out —
//! one synchronous call, no bus/ring/async runtime involved. See the README
//! "Interface design" section for why this shape was chosen over the
//! alternatives (an actor/channel API, a batch/offline-only API, a
//! callback-registration API).

use crate::stage::{Config, Stage};
use crate::tracker::{FrameView, Tracker};
use crate::types::{BBox, Confidence, Detection, TrackState};

/// A single-target tracking session over one video stream. Owns the
/// [`Tracker`] backend and all lock/loss state; `step()` is the only call
/// most consumers need.
pub struct TrackingSession<T: Tracker> {
    stage: Stage<T>,
}

impl<T: Tracker> TrackingSession<T> {
    /// `frame_w`/`frame_h` are the stream's fixed frame dimensions (used by
    /// the seedability gate and the central-acquisition policy — a session
    /// is scoped to one resolution for its lifetime).
    pub fn new(tracker: T, frame_w: i64, frame_h: i64, config: Config) -> Self {
        TrackingSession {
            stage: Stage::new(tracker, frame_w, frame_h, config),
        }
    }

    /// Advance the session by one frame. `detections` is this tick's fresh
    /// detector output:
    ///
    /// * `None` — the detector did not produce a batch this tick (it runs
    ///   slower than the video, or this tick is between its outputs). The
    ///   visual tracker still steps forward on its own; acquisition state
    ///   (e.g. a pending two-batch confirmation candidate) is left
    ///   untouched, since no detector observation was actually made.
    /// * `Some(&[])` — the detector DID run this tick and reported zero
    ///   qualifying boxes. This is a real (negative) observation and folds
    ///   into acquisition/refresh exactly like a non-empty batch would (a
    ///   pending confirmation candidate can be dropped by it).
    ///
    /// Passing `Some(&[])` on every tick when your detector actually runs
    /// slower than your video would silently defeat two-batch confirmation
    /// — every gap tick would look like "detector ran and saw nothing,"
    /// wiping the previous tick's candidate before it can be confirmed.
    /// Use `None` for a skipped tick, `Some(_)` only when the detector
    /// genuinely produced (possibly empty) output.
    ///
    /// `now_s` is a monotonically increasing seconds clock the caller
    /// controls (wall clock, sim clock, whatever) — only used for staleness
    /// age-out; the confirmation gate has no timing dependency of its own.
    pub fn step(&mut self, frame: FrameView<'_>, detections: Option<&[Detection]>, now_s: f64) -> TrackState {
        let pending = detections.and_then(|d| self.stage.on_detection_batch(d));
        self.stage.tick(frame, now_s, pending, false)
    }

    /// Force a lock onto `bbox` this tick, bypassing acquisition/
    /// confirmation entirely — an operator "track this" command. Always
    /// reinitialises the backend, even over a healthy existing lock.
    /// `matched_confidence` is the caller's best-effort confidence for the
    /// forced box if it has one (e.g. from a detection that happened to
    /// overlap the requested box); `None` if the box came from nowhere
    /// tracked (a raw click/coordinate).
    pub fn force_lock(&mut self, frame: FrameView<'_>, bbox: BBox, now_s: f64, matched_confidence: Option<Confidence>) -> TrackState {
        let pending = self.stage.on_force_lock(bbox, matched_confidence);
        self.stage.tick(frame, now_s, Some(pending), true)
    }

    /// Escape hatch to the lower-level [`Stage`] API (e.g. to route
    /// detections through `on_detection_batch` separately from `tick`, for
    /// a pipeline that batches detections before the frame is ready).
    pub fn stage_mut(&mut self) -> &mut Stage<T> {
        &mut self.stage
    }
}
