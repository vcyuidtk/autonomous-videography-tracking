//! [`Tracker`]: the visual-tracker seam. This crate defines the trait and a
//! reference/demo implementation ([`crate::simple_tracker::SimpleTracker`])
//! but links no vendor CV SDK itself — a real correlation-filter tracker
//! (e.g. OpenCV CSRT) lives in the sibling `tracking-cv` crate so consumers
//! who don't need it don't pay for it. Ported from the original
//! `autonomous-videography` monorepo's `av-track::Tracker` (git history in
//! this repo starts from that extraction — see README "Provenance").

use crate::types::BBox;

/// A read-only view of one decoded frame, RGB8, row-major, no padding.
/// Borrowed, not owned: implementations must not retain a pointer past the
/// call that handed it a `FrameView` (an OpenCV binding, for instance, must
/// copy before returning).
#[derive(Debug, Clone, Copy)]
pub struct FrameView<'a> {
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    /// Bytes per pixel (3 for RGB8/BGR8).
    pub channels: u32,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum TrackerError {
    #[error("tracker init/reinit failed: {0}")]
    Init(String),
    #[error("tracker update failed: {0}")]
    Update(String),
}

/// Single-target per-frame visual tracker. The state machine in
/// [`crate::stage::Stage`] treats any implementation identically — real CV
/// backend or a test fake.
pub trait Tracker: Send {
    /// `true` once `reinit()` has locked a target, until `reset()`.
    fn is_active(&self) -> bool;

    /// Lock the tracker onto a new target (or refresh the current one) — the
    /// expensive re-initialisation step. Only [`crate::seed::SeedBox`]-
    /// validated boxes are accepted, at the type level.
    fn reinit(&mut self, frame: FrameView<'_>, seed: crate::seed::SeedBox) -> Result<(), TrackerError>;

    /// Step the tracker forward one frame — the cheap incremental step.
    /// `Ok(None)` means the tracker reports the target lost this frame;
    /// `Err` is a tracker-internal failure. Callers must treat both the same
    /// way — transition to `Lost` with the held box, never panic (this is
    /// what [`crate::stage::Stage::tick`] does).
    fn update(&mut self, frame: FrameView<'_>) -> Result<Option<BBox>, TrackerError>;

    /// Drop the current target; `is_active()` returns `false` until the next
    /// `reinit()`.
    fn reset(&mut self);
}
