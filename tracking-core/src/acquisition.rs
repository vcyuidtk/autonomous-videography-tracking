//! Acquisition (central policy, two-batch confirmation) and refresh
//! (IoU association) policy.
//!
//! **Class-agnostic acquisition and two-batch confirmation are a pair.**
//! [`Detection`] carries no label — deliberately, so the acquisition/refresh
//! policy in this module cannot read one even by accident. A single
//! detector batch is treated as noisy and never locks a track on its own;
//! only two consecutive batches that spatially agree confirm a lock. This
//! guards against a one-frame false-positive detection stealing the lock.
//! Ported (bus/label context stripped) from the original
//! `autonomous-videography` monorepo's `av-track::acquisition` — see
//! README "Provenance".

use crate::types::{BBox, Confidence, Detection};

/// Intersection-over-union of two boxes.
pub fn iou(a: BBox, b: BBox) -> f64 {
    let (ix1, iy1) = (a.x1.max(b.x1), a.y1.max(b.y1));
    let (ix2, iy2) = (a.x2.min(b.x2), a.y2.min(b.y2));
    let inter = (ix2 - ix1).max(0) * (iy2 - iy1).max(0);
    if inter == 0 {
        return 0.0;
    }
    let union = a.area() + b.area() - inter;
    if union > 0 {
        inter as f64 / union as f64
    } else {
        0.0
    }
}

fn sq_dist_to(b: BBox, cx: f64, cy: f64) -> f64 {
    let (bx, by) = b.centre();
    (bx - cx).powi(2) + (by - cy).powi(2)
}

/// Central acquisition policy: the most-central qualifying detection (box
/// centre closest to the frame centre, among those with
/// `confidence >= min_conf`), or `None` when nothing qualifies.
pub fn select_central(detections: &[Detection], frame_w: i64, frame_h: i64, min_conf: Confidence) -> Option<Detection> {
    let (cx, cy) = (frame_w as f64 / 2.0, frame_h as f64 / 2.0);
    detections
        .iter()
        .filter(|d| d.confidence.value() >= min_conf.value())
        .min_by(|a, b| sq_dist_to(a.bbox, cx, cy).total_cmp(&sq_dist_to(b.bbox, cx, cy)))
        .copied()
}

/// Two-batch acquisition confirmation (the class-agnostic noise gate).
/// Returns `(confirmed, next_candidate)`:
///
/// * If `prev_candidate` is set and this batch has a qualifying detection
///   whose centre is within `gate_px` of it, that detection (the nearest
///   such) is CONFIRMED — lock onto it, `next_candidate` is `None`.
/// * Otherwise nothing is confirmed and `next_candidate` is this batch's
///   central nomination (possibly `None`), to carry into the next batch.
///
/// The only path to a confirmed result is two consecutive calls agreeing
/// spatially — there is no path that locks on one batch alone.
pub fn confirm_central(
    prev_candidate: Option<Detection>,
    detections: &[Detection],
    frame_w: i64,
    frame_h: i64,
    min_conf: Confidence,
    gate_px: f64,
) -> (Option<Detection>, Option<Detection>) {
    if let Some(prev) = prev_candidate {
        let (px, py) = prev.bbox.centre();
        let dist_to_prev = |d: &Detection| {
            let (cx, cy) = d.bbox.centre();
            ((cx - px).powi(2) + (cy - py).powi(2)).sqrt()
        };
        let near = detections
            .iter()
            .filter(|d| d.confidence.value() >= min_conf.value() && dist_to_prev(d) <= gate_px)
            .min_by(|a, b| dist_to_prev(a).total_cmp(&dist_to_prev(b)))
            .copied();
        if let Some(confirmed) = near {
            return (Some(confirmed), None);
        }
    }
    (None, select_central(detections, frame_w, frame_h, min_conf))
}

/// IoU association: the qualifying detection with the highest IoU against
/// `current_box`, provided that IoU is at or above `iou_threshold`; `None`
/// when nothing matches (keep tracking on the visual tracker alone).
pub fn match_detection(current_box: BBox, detections: &[Detection], min_conf: Confidence, iou_threshold: f64) -> Option<Detection> {
    detections
        .iter()
        .filter(|d| d.confidence.value() >= min_conf.value() && iou(current_box, d.bbox) >= iou_threshold)
        .max_by(|a, b| iou(current_box, a.bbox).total_cmp(&iou(current_box, b.bbox)))
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: i64, y1: i64, x2: i64, y2: i64, conf: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, y1, x2, y2),
            confidence: Confidence::new(conf).unwrap(),
            id: 0,
        }
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let b = BBox::new(0, 0, 10, 10);
        assert!((iou(b, b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        assert_eq!(iou(BBox::new(0, 0, 10, 10), BBox::new(100, 100, 110, 110)), 0.0);
    }

    #[test]
    fn select_central_picks_the_most_central_qualifying_detection() {
        let dets = vec![det(0, 0, 20, 20, 0.5), det(90, 90, 110, 110, 0.5)];
        let picked = select_central(&dets, 100, 100, Confidence::new(0.3).unwrap()).unwrap();
        assert_eq!(picked.bbox, BBox::new(0, 0, 20, 20));
    }

    #[test]
    fn select_central_excludes_below_confidence_floor() {
        let dets = vec![det(90, 90, 110, 110, 0.1)];
        assert!(select_central(&dets, 100, 100, Confidence::new(0.3).unwrap()).is_none());
    }

    #[test]
    fn confirm_central_never_confirms_on_a_single_batch() {
        let dets = vec![det(90, 90, 110, 110, 0.5)];
        let (confirmed, candidate) = confirm_central(None, &dets, 100, 100, Confidence::new(0.3).unwrap(), 20.0);
        assert!(confirmed.is_none());
        assert!(candidate.is_some());
    }

    #[test]
    fn confirm_central_confirms_when_two_batches_agree() {
        let min_conf = Confidence::new(0.3).unwrap();
        let batch1 = vec![det(90, 90, 110, 110, 0.5)];
        let (confirmed1, candidate) = confirm_central(None, &batch1, 100, 100, min_conf, 20.0);
        assert!(confirmed1.is_none());
        let batch2 = vec![det(92, 92, 112, 112, 0.5)];
        let (confirmed2, _) = confirm_central(candidate, &batch2, 100, 100, min_conf, 20.0);
        assert!(confirmed2.is_some());
    }

    #[test]
    fn confirm_central_does_not_confirm_when_batches_disagree() {
        let min_conf = Confidence::new(0.3).unwrap();
        let batch1 = vec![det(0, 0, 20, 20, 0.5)];
        let (_, candidate) = confirm_central(None, &batch1, 100, 100, min_conf, 5.0);
        let batch2 = vec![det(80, 80, 100, 100, 0.5)];
        let (confirmed2, next) = confirm_central(candidate, &batch2, 100, 100, min_conf, 5.0);
        assert!(confirmed2.is_none());
        assert!(next.is_some());
    }

    #[test]
    fn match_detection_picks_highest_iou_above_threshold() {
        let current = BBox::new(0, 0, 10, 10);
        let dets = vec![det(0, 0, 9, 9, 0.9), det(1, 1, 11, 11, 0.9)];
        let matched = match_detection(current, &dets, Confidence::new(0.3).unwrap(), 0.3).unwrap();
        assert!(iou(current, matched.bbox) >= 0.3);
    }

    #[test]
    fn match_detection_returns_none_below_iou_threshold() {
        let current = BBox::new(0, 0, 10, 10);
        let dets = vec![det(200, 200, 210, 210, 0.9)];
        assert!(match_detection(current, &dets, Confidence::new(0.3).unwrap(), 0.3).is_none());
    }
}
