//! Domain types for the tracking algorithm core. Deliberately independent
//! of any host application's wire schema (bus messages, frame IDs, camera
//! IDs) — see the crate README's "Interface design" section. A consumer
//! embedding this crate maps its own detector/bus types to these at the
//! boundary; that mapping is the consumer's job, not this crate's.

/// An axis-aligned pixel box `(x1, y1, x2, y2)`, `x1 < x2`, `y1 < y2` for any
/// meaningful box (not enforced by the type itself — degenerate boxes are
/// rejected where it matters, e.g. [`crate::seed::SeedBox`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BBox {
    pub x1: i64,
    pub y1: i64,
    pub x2: i64,
    pub y2: i64,
}

impl BBox {
    pub fn new(x1: i64, y1: i64, x2: i64, y2: i64) -> Self {
        BBox { x1, y1, x2, y2 }
    }

    pub fn width(&self) -> i64 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> i64 {
        self.y2 - self.y1
    }

    pub fn centre(&self) -> (f64, f64) {
        ((self.x1 + self.x2) as f64 / 2.0, (self.y1 + self.y2) as f64 / 2.0)
    }

    pub fn area(&self) -> i64 {
        self.width().max(0) * self.height().max(0)
    }
}

/// A confidence score, clamped to `[0.0, 1.0]` at construction — mirrors the
/// host stack's `Confidence` newtype so a detector confidence can't silently
/// be an out-of-range float deep inside the acquisition gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confidence(f32);

impl Confidence {
    pub fn new(v: f32) -> Option<Self> {
        if (0.0..=1.0).contains(&v) {
            Some(Confidence(v))
        } else {
            None
        }
    }

    /// Clamps rather than rejecting — convenient at the eval-harness/demo
    /// boundary where synthetic scores are already known to be in range.
    pub fn clamp(v: f32) -> Self {
        Confidence(v.clamp(0.0, 1.0))
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

/// One detector observation this tick — class-agnostic by construction
/// (HWC-43 upstream: this algorithm core never reads a label; see
/// `acquisition.rs`'s module docs for why that's load-bearing, not
/// incidental). `id` is an opaque handle the caller can use to map a
/// [`crate::session::StepOutcome`]'s association back to its own richer
/// detection record (label, embedding, whatever) — this crate never looks
/// at it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    pub confidence: Confidence,
    pub id: u64,
}

/// A locked track's box plus the detection (if any) it was last refreshed
/// against — carried through a `Lost` tick so a consumer never sees a
/// blanked box mid-loss (see `stage.rs` module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeldTrack {
    pub bbox: BBox,
    pub confidence: Confidence,
    /// [`Detection::id`] of the detection that produced/refreshed this box,
    /// if the box came from a detection rather than a pure tracker step or
    /// an operator force-lock.
    pub source_detection_id: Option<u64>,
}

/// This tick's track state, returned by [`crate::session::TrackingSession::step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackState {
    /// No target locked and nothing pending confirmation.
    Idle,
    /// Actively locked; `bbox` is this tick's fresh estimate.
    Locked(HeldTrack),
    /// Was locked, no fresh box this tick (tracker reported loss and no
    /// qualifying detection refreshed it) — `bbox` is the last-known box,
    /// not a fabricated current one. A `Lost` track ages out to `Idle`
    /// after `max_age_s` with no refresh (see [`crate::Config::max_age_s`]).
    Lost(HeldTrack),
}
