//! Scoring: objective metrics computed from a run's per-tick [`TrackState`]
//! output against a [`Sequence`]'s known ground truth. See README "Eval
//! harness" for what each metric means and why it was picked.

use crate::synthetic::Sequence;
use tracking_core::{iou, TrackState};

/// The IoU a `Locked`/`Lost` report must clear against ground truth to count
/// as a tracking success on a visible-target tick. 0.5 is the standard
/// single-object-tracking convention (VOT/OTB benchmarks use the same bar).
pub const SUCCESS_IOU: f64 = 0.5;

#[derive(Debug, Clone)]
pub struct Report {
    pub scenario: &'static str,
    pub num_ticks: usize,
    /// Ticks where the target was visible in ground truth.
    pub visible_ticks: usize,
    /// Of the visible ticks, how many the tracker reported a box with
    /// IoU >= [`SUCCESS_IOU`] against ground truth (Locked or Lost both
    /// count — Lost still carries the last real box).
    pub success_ticks: usize,
    /// Mean IoU against ground truth over ticks the tracker reported
    /// `Locked` while the target was visible. `None` if it was never
    /// `Locked` on a visible tick.
    pub mean_iou_locked: Option<f64>,
    /// Ticks from sequence start to the first `Locked` state, or `None` if
    /// it never locked.
    pub time_to_lock: Option<usize>,
    /// Ticks from a visible target's re-appearance after occlusion to the
    /// next `Locked` state — `None` if the sequence has no occlusion or the
    /// tracker never re-locked.
    pub reacquisition_ticks: Option<usize>,
    /// Count of ticks where the reported box switched from matching the
    /// real target to matching the distractor (or vice versa) — a wrong-
    /// object lock. Only meaningful for sequences with a distractor.
    pub id_switches: u32,
}

impl Report {
    pub fn success_rate(&self) -> f64 {
        if self.visible_ticks == 0 {
            return 1.0;
        }
        self.success_ticks as f64 / self.visible_ticks as f64
    }
}

pub fn score(seq: &Sequence, states: &[TrackState]) -> Report {
    assert_eq!(seq.len(), states.len());

    let mut visible_ticks = 0usize;
    let mut success_ticks = 0usize;
    let mut iou_sum = 0.0f64;
    let mut iou_n = 0usize;
    let mut time_to_lock = None;
    let mut id_switches = 0u32;
    let mut prev_on_distractor = false;

    // Occlusion window (if any): the first visible=false run's start/end.
    let mut occlusion_end: Option<usize> = None;
    {
        let mut in_gap = false;
        for (t, gt) in seq.target_gt.iter().enumerate() {
            if !gt.visible {
                in_gap = true;
            } else if in_gap {
                occlusion_end = Some(t);
                break;
            }
        }
    }
    let mut reacquisition_ticks = None;

    for (t, state) in states.iter().enumerate() {
        if time_to_lock.is_none() && matches!(state, TrackState::Locked(_)) {
            time_to_lock = Some(t);
        }

        let gt = seq.target_gt[t];
        let reported = match state {
            TrackState::Locked(h) | TrackState::Lost(h) => Some(h.bbox),
            TrackState::Idle => None,
        };

        if gt.visible {
            visible_ticks += 1;
            if let Some(b) = reported {
                if iou(b, gt.bbox) >= SUCCESS_IOU {
                    success_ticks += 1;
                }
            }
        }

        if let (Some(b), TrackState::Locked(_)) = (reported, state) {
            if gt.visible {
                iou_sum += iou(b, gt.bbox);
                iou_n += 1;
            }
        }

        if let (Some(occ_end), Some(b)) = (occlusion_end, reported) {
            if t >= occ_end && reacquisition_ticks.is_none() && matches!(state, TrackState::Locked(_)) {
                reacquisition_ticks = Some(t - occ_end);
            }
            let _ = b;
        }

        if let (Some(b), Some(dgt)) = (reported, seq.distractor_gt.get(t).copied().flatten()) {
            let on_distractor = iou(b, dgt.bbox) > iou(b, gt.bbox) && iou(b, dgt.bbox) > 0.1;
            if on_distractor && !prev_on_distractor {
                id_switches += 1;
            }
            prev_on_distractor = on_distractor;
        }
    }

    Report {
        scenario: seq.name,
        num_ticks: seq.len(),
        visible_ticks,
        success_ticks,
        mean_iou_locked: if iou_n > 0 { Some(iou_sum / iou_n as f64) } else { None },
        time_to_lock,
        reacquisition_ticks,
        id_switches,
    }
}
