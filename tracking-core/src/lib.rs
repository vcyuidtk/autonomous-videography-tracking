//! `tracking-core` — the single-target visual-tracking algorithm: track
//! init/update, detection association, and the lock/loss/age-out state
//! machine. No I/O, no vendor CV SDK, no bus/ring/async runtime.
//!
//! See the repo README for the interface-design rationale and how this
//! crate is meant to be consumed. Quick start: [`session::TrackingSession`].

pub mod acquisition;
pub mod seed;
pub mod session;
pub mod simple_tracker;
pub mod stage;
pub mod tracker;
pub mod types;

pub use acquisition::{confirm_central, iou, match_detection, select_central};
pub use seed::{force_lock_box_error, FrameDims, SeedBox, SeedError};
pub use session::TrackingSession;
pub use simple_tracker::SimpleTracker;
pub use stage::{Config, PendingLock, Stage};
pub use tracker::{FrameView, Tracker, TrackerError};
pub use types::{BBox, Confidence, Detection, HeldTrack, TrackState};

/// Staleness check: a locked-but-stale track (no matching detection has
/// refreshed the template within `max_age_s`) must have its lock dropped.
pub fn is_stale(last_refresh_s: f64, now_s: f64, max_age_s: f64) -> bool {
    now_s - last_refresh_s > max_age_s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_stale_true_past_max_age() {
        assert!(is_stale(0.0, 10.0, 5.0));
    }

    #[test]
    fn is_stale_false_within_max_age() {
        assert!(!is_stale(0.0, 4.0, 5.0));
    }
}
