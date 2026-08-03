//! The tracking state machine: acquisition -> lock -> refresh -> loss ->
//! age-out. Factored so the whole thing is unit-testable against a fake
//! [`Tracker`] with no real CV backend. [`crate::session::TrackingSession`]
//! is a thinner, single-call wrapper over this for the common case; use
//! `Stage` directly if you need the two-phase `on_detection_batch`/`tick`
//! split (e.g. to route detections from multiple sources before the tick).
//!
//! Ported (bus/frame-id/camera-id/label plumbing stripped — that's host
//! wiring, not algorithm) from the original `autonomous-videography`
//! monorepo's `tracking::stage::Stage` — see README "Provenance".

use crate::acquisition::{confirm_central, match_detection};
use crate::seed::{FrameDims, SeedBox};
use crate::tracker::{FrameView, Tracker};
use crate::types::{BBox, Confidence, Detection, HeldTrack, TrackState};

/// A candidate lock nomination pending the tracker-capability gate
/// ([`SeedBox`]) and this tick's frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PendingLock {
    pub bbox: BBox,
    pub confidence: Confidence,
    pub source_detection_id: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateTag {
    Idle,
    Locked,
    Lost,
}

/// Tunable policy knobs — confidence floor, IoU-association threshold,
/// two-batch confirmation gate distance, and staleness age-out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub min_conf: Confidence,
    pub iou_threshold: f64,
    pub gate_px: f64,
    pub max_age_s: f64,
}

impl Default for Config {
    /// Reasonable starting points for a ~640x480, ~10-30fps stream; tune
    /// per deployment. Not claimed optimal for any particular camera/lens.
    fn default() -> Self {
        Config {
            min_conf: Confidence::new(0.3).unwrap(),
            iou_threshold: 0.3,
            gate_px: 50.0,
            max_age_s: 2.0,
        }
    }
}

/// Owns one lock's worth of mutable state across ticks. `T: Tracker` so
/// tests can inject a fake and exercise the acquisition/refresh/age-out
/// logic without a real CV backend.
pub struct Stage<T: Tracker> {
    tracker: T,
    frame_w: i64,
    frame_h: i64,
    config: Config,

    acquire_candidate: Option<Detection>,
    current_box: Option<BBox>,
    current_confidence: Confidence,
    last_refresh_s: f64,

    // Survives a LOST tick — cleared only when the state actually reaches
    // Idle, so a Lost tick always carries a real last-known box.
    held: Option<HeldTrack>,
    prev_state: StateTag,
}

impl<T: Tracker> Stage<T> {
    pub fn new(tracker: T, frame_w: i64, frame_h: i64, config: Config) -> Self {
        Stage {
            tracker,
            frame_w,
            frame_h,
            config,
            acquire_candidate: None,
            current_box: None,
            current_confidence: Confidence::new(0.0).unwrap(),
            last_refresh_s: 0.0,
            held: None,
            prev_state: StateTag::Idle,
        }
    }

    pub fn tracker(&self) -> &T {
        &self.tracker
    }

    /// Step 1: fold in one detector batch. With nothing locked: class-
    /// agnostic two-batch confirmation. With a lock: IoU association against
    /// `current_box`.
    pub fn on_detection_batch(&mut self, detections: &[Detection]) -> Option<PendingLock> {
        let chosen = if !self.tracker.is_active() {
            let (confirmed, next) = confirm_central(
                self.acquire_candidate,
                detections,
                self.frame_w,
                self.frame_h,
                self.config.min_conf,
                self.config.gate_px,
            );
            self.acquire_candidate = next;
            confirmed
        } else {
            self.current_box
                .and_then(|cur| match_detection(cur, detections, self.config.min_conf, self.config.iou_threshold))
        };
        chosen.map(|d| PendingLock {
            bbox: d.bbox,
            confidence: d.confidence,
            source_detection_id: Some(d.id),
        })
    }

    /// Step 1b: an external "force lock" command — beats any step-1
    /// nomination and always forces a from-scratch reinit. `matched` is the
    /// caller's best-effort label/confidence for the forced box, if it knows
    /// one (e.g. best-IoU-overlap against the freshest detection batch).
    pub fn on_force_lock(&mut self, bbox: BBox, matched_confidence: Option<Confidence>) -> PendingLock {
        // A force-lock beats whatever step 1 nominated — drop any
        // in-progress acquisition confirmation so it can't leak into an
        // unrelated later lock.
        self.acquire_candidate = None;
        PendingLock {
            bbox,
            confidence: matched_confidence.unwrap_or(Confidence::new(0.0).unwrap()),
            source_detection_id: None,
        }
    }

    /// Steps 1c/2/3/4: gate `pending` through [`SeedBox`], run reinit/update
    /// against `frame`, age out if stale, and assemble this tick's
    /// [`TrackState`]. `force_reinit` is `true` only when `pending` came
    /// from [`Self::on_force_lock`].
    pub fn tick(&mut self, frame: FrameView<'_>, now_s: f64, pending: Option<PendingLock>, force_reinit: bool) -> TrackState {
        // 1c: tracker-capability gate — a nomination the backend cannot
        // physically seed is dropped here, not discovered as a crash inside
        // reinit.
        let (pending, force_reinit) = match pending {
            Some(p) => match SeedBox::new(
                p.bbox,
                FrameDims {
                    width: self.frame_w,
                    height: self.frame_h,
                },
            ) {
                Ok(seed) => (Some((p, seed)), force_reinit),
                Err(_) => (None, false),
            },
            None => (None, force_reinit),
        };

        let mut box_out: Option<BBox> = None;
        // Detection id that produced/refreshed `box_out`, if any — `None`
        // for a pure tracker-only step (no detection this tick).
        let mut box_source_id: Option<u64> = None;

        if let Some((p, seed)) = pending {
            let needs_reinit = force_reinit || !self.tracker.is_active() || self.prev_state == StateTag::Lost;
            let mut raw_box = None;
            if !needs_reinit {
                if let Ok(Some(b)) = self.tracker.update(frame) {
                    raw_box = Some(b);
                }
            }
            if needs_reinit || raw_box.is_none() {
                match self.tracker.reinit(frame, seed) {
                    Ok(()) => {
                        self.current_box = Some(p.bbox);
                        box_out = Some(p.bbox);
                        box_source_id = p.source_detection_id;
                    }
                    Err(_) => {
                        self.tracker.reset();
                        self.reset_lock_state();
                        box_out = None;
                    }
                }
            } else {
                self.current_box = raw_box;
                box_out = raw_box;
                box_source_id = p.source_detection_id;
            }
            if box_out.is_some() {
                self.acquire_candidate = None;
                self.current_confidence = p.confidence;
                self.last_refresh_s = now_s;
            }
        } else if self.tracker.is_active() {
            match self.tracker.update(frame) {
                Ok(Some(b)) => {
                    self.current_box = Some(b);
                    box_out = Some(b);
                }
                Ok(None) => { /* current_box kept — association anchor for a re-lock */ }
                Err(_) => {
                    self.tracker.reset();
                    self.reset_lock_state();
                }
            }
        }

        // 3. Age out if stale.
        if self.tracker.is_active() && crate::is_stale(self.last_refresh_s, now_s, self.config.max_age_s) {
            self.tracker.reset();
            self.reset_lock_state();
            box_out = None;
        }

        // 4. Assemble this tick's state. LOST carries the HELD box.
        if let Some(b) = box_out {
            self.held = Some(HeldTrack {
                bbox: b,
                confidence: self.current_confidence,
                source_detection_id: box_source_id,
            });
            self.prev_state = StateTag::Locked;
            TrackState::Locked(self.held.unwrap())
        } else if self.tracker.is_active() || self.prev_state == StateTag::Locked {
            self.prev_state = StateTag::Lost;
            match self.held {
                Some(h) => TrackState::Lost(h),
                // Should not happen (a LOST tick always follows a prior
                // LOCKED tick that set `held`), but never fabricate a box.
                None => TrackState::Idle,
            }
        } else {
            self.prev_state = StateTag::Idle;
            self.held = None;
            TrackState::Idle
        }
    }

    fn reset_lock_state(&mut self) {
        self.current_box = None;
        self.current_confidence = Confidence::new(0.0).unwrap();
        self.last_refresh_s = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::TrackerError;

    #[derive(Default)]
    struct FakeTracker {
        active: bool,
        next_update: Option<BBox>,
        init_calls: u32,
        update_calls: u32,
    }

    impl Tracker for FakeTracker {
        fn is_active(&self) -> bool {
            self.active
        }
        fn reinit(&mut self, _frame: FrameView<'_>, seed: SeedBox) -> Result<(), TrackerError> {
            self.init_calls += 1;
            self.active = true;
            self.next_update = Some(seed.get());
            Ok(())
        }
        fn update(&mut self, _frame: FrameView<'_>) -> Result<Option<BBox>, TrackerError> {
            self.update_calls += 1;
            Ok(self.next_update)
        }
        fn reset(&mut self) {
            self.active = false;
            self.next_update = None;
        }
    }

    fn frame() -> FrameView<'static> {
        static PIXELS: [u8; 12] = [0; 12];
        FrameView {
            pixels: &PIXELS,
            width: 2,
            height: 2,
            channels: 3,
        }
    }

    fn stage() -> Stage<FakeTracker> {
        Stage::new(
            FakeTracker::default(),
            640,
            480,
            Config {
                min_conf: Confidence::new(0.1).unwrap(),
                iou_threshold: 0.3,
                gate_px: 50.0,
                max_age_s: 2.0,
            },
        )
    }

    fn det(x1: i64, y1: i64, x2: i64, y2: i64, conf: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, y1, x2, y2),
            confidence: Confidence::new(conf).unwrap(),
            id: 1,
        }
    }

    #[test]
    fn idle_with_no_input_stays_idle() {
        let mut s = stage();
        let state = s.tick(frame(), 0.0, None, false);
        assert_eq!(state, TrackState::Idle);
    }

    #[test]
    fn a_single_batch_does_not_lock() {
        let mut s = stage();
        let pending = s.on_detection_batch(&[det(300, 220, 340, 260, 0.5)]);
        let state = s.tick(frame(), 0.0, pending, false);
        assert_eq!(state, TrackState::Idle);
    }

    #[test]
    fn two_agreeing_batches_lock() {
        let mut s = stage();
        let p1 = s.on_detection_batch(&[det(300, 220, 340, 260, 0.5)]);
        let _ = s.tick(frame(), 0.0, p1, false);

        let p2 = s.on_detection_batch(&[det(302, 222, 342, 262, 0.5)]);
        assert!(p2.is_some(), "second agreeing batch must confirm");
        let state = s.tick(frame(), 0.1, p2, false);
        assert!(matches!(state, TrackState::Locked(_)));
    }

    #[test]
    fn healthy_refresh_does_not_reinit() {
        let mut s = stage();
        let p1 = s.on_detection_batch(&[det(300, 220, 340, 260, 0.5)]);
        let _ = s.tick(frame(), 0.0, p1, false);
        let p2 = s.on_detection_batch(&[det(302, 222, 342, 262, 0.5)]);
        let _ = s.tick(frame(), 0.1, p2, false);
        assert_eq!(s.tracker().init_calls, 1, "the fresh lock is the only reinit so far");

        let p3 = s.on_detection_batch(&[det(303, 223, 343, 263, 0.5)]);
        assert!(p3.is_some());
        s.tracker.next_update = Some(BBox::new(304, 224, 344, 264));
        let state = s.tick(frame(), 0.2, p3, false);
        assert!(matches!(state, TrackState::Locked(_)));
        assert_eq!(s.tracker().init_calls, 1, "a healthy refresh must NOT call reinit again");
    }

    #[test]
    fn age_out_produces_lost_with_the_held_box() {
        let mut s = stage();
        let p1 = s.on_detection_batch(&[det(300, 220, 340, 260, 0.5)]);
        let _ = s.tick(frame(), 0.0, p1, false);
        let p2 = s.on_detection_batch(&[det(302, 222, 342, 262, 0.5)]);
        let locked = s.tick(frame(), 0.1, p2, false);
        let TrackState::Locked(held) = locked else {
            panic!("expected Locked")
        };

        let state = s.tick(frame(), 100.0, None, false);
        match state {
            TrackState::Lost(h) => assert_eq!(h.bbox, held.bbox),
            other => panic!("expected Lost carrying the held box, got {other:?}"),
        }
    }

    #[test]
    fn seed_incapable_nomination_is_dropped_not_reinit() {
        let mut s = stage();
        let pending = Some(PendingLock {
            bbox: BBox::new(10, 10, 11, 10),
            confidence: Confidence::new(0.9).unwrap(),
            source_detection_id: Some(1),
        });
        let state = s.tick(frame(), 0.0, pending, false);
        assert_eq!(state, TrackState::Idle);
        assert_eq!(s.tracker().init_calls, 0, "an unseedable nomination must never reach reinit");
    }

    #[test]
    fn force_lock_always_reinits_even_over_a_healthy_lock() {
        let mut s = stage();
        let p1 = s.on_detection_batch(&[det(300, 220, 340, 260, 0.5)]);
        let _ = s.tick(frame(), 0.0, p1, false);
        let p2 = s.on_detection_batch(&[det(302, 222, 342, 262, 0.5)]);
        let _ = s.tick(frame(), 0.1, p2, false);
        assert_eq!(s.tracker().init_calls, 1);

        let force = s.on_force_lock(BBox::new(10, 10, 60, 60), Some(Confidence::new(0.7).unwrap()));
        let state = s.tick(frame(), 0.2, Some(force), true);
        assert_eq!(s.tracker().init_calls, 2, "force-lock must reinit even over a healthy lock");
        match state {
            TrackState::Locked(h) => assert_eq!(h.bbox, BBox::new(10, 10, 60, 60)),
            other => panic!("expected Locked on the force-lock box, got {other:?}"),
        }
    }
}
