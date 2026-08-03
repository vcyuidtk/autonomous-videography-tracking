pub mod metrics;
pub mod rng;
pub mod synthetic;

use tracking_core::{Config, SimpleTracker, TrackState, TrackingSession};

/// Run one synthetic sequence through a fresh `TrackingSession<SimpleTracker>`
/// and return this run's per-tick states, for scoring by [`metrics::score`].
pub fn run(seq: &synthetic::Sequence, config: Config) -> Vec<TrackState> {
    let mut session = TrackingSession::new(SimpleTracker::new(), synthetic::WIDTH as i64, synthetic::HEIGHT as i64, config);
    let mut out = Vec::with_capacity(seq.len());
    for t in 0..seq.len() {
        let state = session.step(seq.frame_view(t), seq.detections[t].as_deref(), t as f64 * (1.0 / 15.0));
        out.push(state);
    }
    out
}
