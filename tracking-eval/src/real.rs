//! Real-video/real-image-sequence evaluation: load an actual frame sequence
//! plus ground-truth track annotations for a sample clip, run it through
//! `TrackingSession::step` for real, and score it with the exact same
//! [`crate::metrics::score`] used for synthetic scenarios.
//!
//! This directly answers `autonomous-videography-tracking` GH issue #1: the
//! synthetic harness in [`crate::synthetic`] never touches a decoded frame,
//! which misses the repo's founding brief ("test algorithms against sample
//! videos and evaluate them objectively"). This module is that path.
//!
//! Two input shapes are supported:
//!
//! - an **image-sequence directory**: any number of PNG/JPEG frames, sorted
//!   lexicographically by filename, all the same dimensions. This is the
//!   baseline, dependency-light real-frame source (see [`load_image_sequence`]).
//! - a **real video file**: decoded to an image sequence with the system
//!   `ffmpeg` binary (see [`extract_video_frames`]), then loaded the same
//!   way. `tracking-eval` does not link a video-decode library itself (that
//!   would cost every synthetic-only consumer a system dependency — see the
//!   crate README's "no real video, no OpenCV" framing for `eval`'s
//!   zero-dependency design); shelling out to `ffmpeg`, which this
//!   environment already has for other purposes, is the pragmatic way to
//!   accept an actual video file without pulling a decoder into the crate
//!   graph.
//!
//! ## Ground truth format
//!
//! A plain CSV, one row per annotated box: `frame,track_id,x1,y1,x2,y2`.
//! Blank lines and lines starting with `#` are ignored; a header row (first
//! field not parseable as an integer) is tolerated and skipped.
//!
//! - `track_id 0` is **the target** — the box that is both fed to
//!   `TrackingSession::step` as ground-truth-perfect "detections" (see below)
//!   and scored against.
//! - `track_id 1` is an optional **distractor** — plumbed into
//!   [`crate::synthetic::Sequence::distractor_gt`] purely so
//!   [`crate::metrics::score`]'s ID-switch metric has something to compare
//!   against, exactly like the synthetic `distractor_crossing` scenario.
//!   Frames with no distractor row simply have none that tick.
//! - Any other `track_id` is rejected with an error — this crate's
//!   `TrackingSession` is single-target (see the README's "Interface
//!   design"), so a real multi-track annotation file is only usable here
//!   for its target + one distractor track.
//!
//! A frame with no `track_id 0` row is treated as "target not visible this
//! tick" (occluded/out of frame) — same meaning as `synthetic::GroundTruth`'s
//! `visible: false`.
//!
//! ## Detections: ground truth as a stand-in "perfect detector"
//!
//! `TrackingSession::step` takes tracking and detection as separate concerns
//! — it needs per-tick `Detection`s, not just ground truth. This module does
//! not run any real object detector; it reuses the target's ground-truth box
//! as the detector output, confidence 1.0, on every tick where the target is
//! visible (`Some(vec![])` on ticks it isn't — a real, if trivially perfect,
//! negative observation; see `TrackingSession::step`'s docs on why that's not
//! the same as `None`). This is a deliberate, legitimate simplification for
//! **tracking-only** evaluation (isolating the acquisition/association/
//! lock-loss state machine from detector quality) — it is not an end-to-end
//! detector+tracker evaluation. Wiring in a real detector's (e.g.
//! `autonomous-videography-perception`'s) output instead of ground truth
//! would need only a different `Vec<Option<Vec<Detection>>>` fed into
//! [`RealSequence::into_tracking_sequence`]'s caller — the frame-loading and
//! scoring code is agnostic to where detections came from.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracking_core::{BBox, Confidence, Detection};

use crate::synthetic::{GroundTruth, Sequence};

#[derive(Debug, thiserror::Error)]
pub enum RealSeqError {
    #[error("frames directory {0:?} has no image files")]
    EmptyFrameDir(PathBuf),
    #[error("failed to read {path:?}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("failed to decode image {path:?}: {source}")]
    Decode { path: PathBuf, source: image::ImageError },
    #[error("frame {path:?} is {actual_w}x{actual_h}, expected {expected_w}x{expected_h} (first frame's size) — a real sequence must be constant-resolution")]
    SizeMismatch {
        path: PathBuf,
        actual_w: u32,
        actual_h: u32,
        expected_w: u32,
        expected_h: u32,
    },
    #[error("ground truth line {line_no} ({raw:?}): {reason}")]
    BadGroundTruthLine { line_no: usize, raw: String, reason: String },
    #[error("ground truth has track_id {0}, but only 0 (target) and 1 (distractor) are supported — TrackingSession is single-target, see real.rs module docs")]
    UnsupportedTrackId(u64),
    #[error("ground truth references frame {frame}, but the sequence only has {num_frames} frames")]
    FrameOutOfRange { frame: usize, num_frames: usize },
    #[error("`ffmpeg` failed to decode {video:?}: {stderr}")]
    FfmpegFailed { video: PathBuf, stderr: String },
    #[error("failed to launch `ffmpeg` (is it installed and on PATH?): {0}")]
    FfmpegLaunch(std::io::Error),
}

/// Decode `video_path` to a numbered PNG sequence in `out_dir` via the
/// system `ffmpeg` binary. `out_dir` is created if it doesn't exist.
/// Frames come out as `frame_000001.png`, `frame_000002.png`, ... — in
/// source order, which is what [`load_image_sequence`] expects.
pub fn extract_video_frames(video_path: &Path, out_dir: &Path) -> Result<(), RealSeqError> {
    fs::create_dir_all(out_dir).map_err(|e| RealSeqError::Io {
        path: out_dir.to_path_buf(),
        source: e,
    })?;
    let pattern = out_dir.join("frame_%06d.png");
    let output = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(video_path)
        .args(["-vsync", "0"])
        .arg(&pattern)
        .output()
        .map_err(RealSeqError::FfmpegLaunch)?;
    if !output.status.success() {
        return Err(RealSeqError::FfmpegFailed {
            video: video_path.to_path_buf(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}

/// Load every image file in `dir`, sorted by filename, decoded to tightly
/// packed RGB8. All frames must share the first frame's dimensions.
pub fn load_image_sequence(dir: &Path) -> Result<(Vec<Vec<u8>>, u32, u32), RealSeqError> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| RealSeqError::Io {
            path: dir.to_path_buf(),
            source: e,
        })?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "bmp"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(RealSeqError::EmptyFrameDir(dir.to_path_buf()));
    }

    let mut frames = Vec::with_capacity(paths.len());
    let mut dims: Option<(u32, u32)> = None;
    for path in &paths {
        let img = image::open(path).map_err(|e| RealSeqError::Decode {
            path: path.clone(),
            source: e,
        })?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        match dims {
            None => dims = Some((w, h)),
            Some((ew, eh)) if (ew, eh) != (w, h) => {
                return Err(RealSeqError::SizeMismatch {
                    path: path.clone(),
                    actual_w: w,
                    actual_h: h,
                    expected_w: ew,
                    expected_h: eh,
                })
            }
            _ => {}
        }
        frames.push(rgb.into_raw());
    }
    let (w, h) = dims.expect("checked non-empty above");
    Ok((frames, w, h))
}

/// One parsed ground-truth row.
#[derive(Debug)]
struct GtRow {
    frame: usize,
    track_id: u64,
    bbox: BBox,
}

fn parse_ground_truth(text: &str) -> Result<Vec<GtRow>, RealSeqError> {
    const NUM_FIELDS: usize = 6; // frame,track_id,x1,y1,x2,y2

    let mut rows = Vec::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();

        // Tolerate a header row (first field not an integer) rather than
        // erroring on it outright.
        if fields[0].parse::<usize>().is_err() {
            continue;
        }

        if fields.len() != NUM_FIELDS {
            return Err(RealSeqError::BadGroundTruthLine {
                line_no,
                raw: line.to_string(),
                reason: format!(
                    "expected {NUM_FIELDS} comma-separated fields (frame,track_id,x1,y1,x2,y2), got {}",
                    fields.len()
                ),
            });
        }

        let parse_i64 = |i: usize, name: &str| -> Result<i64, RealSeqError> {
            fields[i].parse::<i64>().map_err(|e| RealSeqError::BadGroundTruthLine {
                line_no,
                raw: line.to_string(),
                reason: format!("field {i} ({name}) not an integer: {e}"),
            })
        };

        let frame = parse_i64(0, "frame")? as usize;
        let track_id = parse_i64(1, "track_id")? as u64;
        let x1 = parse_i64(2, "x1")?;
        let y1 = parse_i64(3, "y1")?;
        let x2 = parse_i64(4, "x2")?;
        let y2 = parse_i64(5, "y2")?;

        rows.push(GtRow {
            frame,
            track_id,
            bbox: BBox::new(x1, y1, x2, y2),
        });
    }
    Ok(rows)
}

/// Build a [`Sequence`] from a loaded real frame sequence plus parsed ground
/// truth. `name` is a label for the report only.
pub fn build_sequence(name: &'static str, frames: Vec<Vec<u8>>, width: u32, height: u32, gt_csv: &str) -> Result<Sequence, RealSeqError> {
    let num_frames = frames.len();
    let rows = parse_ground_truth(gt_csv)?;

    let mut target: Vec<Option<BBox>> = vec![None; num_frames];
    let mut distractor: Vec<Option<BBox>> = vec![None; num_frames];
    for row in rows {
        if row.frame >= num_frames {
            return Err(RealSeqError::FrameOutOfRange {
                frame: row.frame,
                num_frames,
            });
        }
        match row.track_id {
            0 => target[row.frame] = Some(row.bbox),
            1 => distractor[row.frame] = Some(row.bbox),
            other => return Err(RealSeqError::UnsupportedTrackId(other)),
        }
    }

    let mut target_gt = Vec::with_capacity(num_frames);
    let mut distractor_gt = Vec::with_capacity(num_frames);
    let mut detections = Vec::with_capacity(num_frames);
    let mut next_det_id = 0u64;

    for t in 0..num_frames {
        match target[t] {
            Some(bbox) => {
                target_gt.push(GroundTruth { bbox, visible: true });
                detections.push(Some(vec![Detection {
                    bbox,
                    confidence: Confidence::clamp(1.0),
                    id: next_det_id,
                }]));
                next_det_id += 1;
            }
            None => {
                target_gt.push(GroundTruth {
                    bbox: BBox::new(0, 0, 0, 0),
                    visible: false,
                });
                detections.push(Some(Vec::new()));
            }
        }
        distractor_gt.push(distractor[t].map(|bbox| GroundTruth { bbox, visible: true }));
    }

    Ok(Sequence {
        name,
        width,
        height,
        frames,
        target_gt,
        distractor_gt,
        detections,
    })
}

/// Convenience: load an image-sequence directory plus a ground-truth CSV
/// file straight into a [`Sequence`].
pub fn load_sequence(name: &'static str, frames_dir: &Path, gt_path: &Path) -> Result<Sequence, RealSeqError> {
    let (frames, width, height) = load_image_sequence(frames_dir)?;
    let gt_csv = fs::read_to_string(gt_path).map_err(|e| RealSeqError::Io {
        path: gt_path.to_path_buf(),
        source: e,
    })?;
    build_sequence(name, frames, width, height, &gt_csv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_temp_dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("tracking-eval-real-test-{}-{n}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_png(path: &Path, w: u32, h: u32, rgb: [u8; 3]) {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb(rgb));
        img.save(path).unwrap();
    }

    #[test]
    fn parse_ground_truth_basic() {
        let csv = "frame,track_id,x1,y1,x2,y2\n0,0,10,10,20,20\n1,0,12,10,22,20\n1,1,100,100,110,110\n";
        let rows = parse_ground_truth(csv).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].frame, 0);
        assert_eq!(rows[0].track_id, 0);
        assert_eq!(rows[0].bbox, BBox::new(10, 10, 20, 20));
        assert_eq!(rows[2].track_id, 1);
    }

    #[test]
    fn parse_ground_truth_ignores_blank_and_comment_lines() {
        let csv = "# comment\n\n0,0,1,2,3,4\n";
        let rows = parse_ground_truth(csv).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn parse_ground_truth_rejects_malformed_row() {
        let csv = "0,0,1,2,3\n"; // missing a field
        let err = parse_ground_truth(csv).unwrap_err();
        assert!(matches!(err, RealSeqError::BadGroundTruthLine { .. }));
    }

    #[test]
    fn build_sequence_marks_missing_frames_not_visible() {
        // 3 tiny 2x2 frames, target present on frames 0 and 2 only.
        let frames = vec![vec![0u8; 2 * 2 * 3]; 3];
        let gt = "frame,track_id,x1,y1,x2,y2\n0,0,0,0,1,1\n2,0,0,0,1,1\n";
        let seq = build_sequence("t", frames, 2, 2, gt).unwrap();
        assert_eq!(seq.target_gt.len(), 3);
        assert!(seq.target_gt[0].visible);
        assert!(!seq.target_gt[1].visible);
        assert!(seq.target_gt[2].visible);
        // Detections mirror ground truth as a perfect-detector stand-in.
        assert_eq!(seq.detections[0].as_ref().unwrap().len(), 1);
        assert_eq!(seq.detections[1].as_ref().unwrap().len(), 0);
    }

    #[test]
    fn build_sequence_rejects_unsupported_track_id() {
        let frames = vec![vec![0u8; 2 * 2 * 3]];
        let gt = "0,2,0,0,1,1\n"; // track_id 2 is not target(0) or distractor(1)
        let err = build_sequence("t", frames, 2, 2, gt).unwrap_err();
        assert!(matches!(err, RealSeqError::UnsupportedTrackId(2)));
    }

    #[test]
    fn build_sequence_rejects_frame_out_of_range() {
        let frames = vec![vec![0u8; 2 * 2 * 3]]; // 1 frame
        let gt = "5,0,0,0,1,1\n";
        let err = build_sequence("t", frames, 2, 2, gt).unwrap_err();
        assert!(matches!(err, RealSeqError::FrameOutOfRange { frame: 5, num_frames: 1 }));
    }

    #[test]
    fn load_image_sequence_round_trips_real_png_files() {
        // Real PNG encode + real PNG decode via the `image` crate — not the
        // app's synthetic renderer — exercising the actual decode path real
        // frames go through.
        let dir = unique_temp_dir();
        write_png(&dir.join("frame_000000.png"), 4, 3, [10, 20, 30]);
        write_png(&dir.join("frame_000001.png"), 4, 3, [40, 50, 60]);
        // A non-image file must be ignored, not crash the loader.
        fs::write(dir.join("ground_truth.csv"), "0,0,0,0,1,1\n").unwrap();

        let (frames, w, h) = load_image_sequence(&dir).unwrap();
        assert_eq!((w, h), (4, 3));
        assert_eq!(frames.len(), 2);
        assert_eq!(&frames[0][0..3], &[10, 20, 30]);
        assert_eq!(&frames[1][0..3], &[40, 50, 60]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_image_sequence_rejects_mixed_resolutions() {
        let dir = unique_temp_dir();
        write_png(&dir.join("frame_000000.png"), 4, 3, [1, 2, 3]);
        write_png(&dir.join("frame_000001.png"), 5, 3, [1, 2, 3]);

        let err = load_image_sequence(&dir).unwrap_err();
        assert!(matches!(err, RealSeqError::SizeMismatch { .. }));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_sequence_end_to_end_real_frames_through_tracking_session() {
        // Full path: real PNG files on disk -> load_image_sequence ->
        // build_sequence -> tracking_eval::run -> metrics::score, with a
        // target that moves in a straight line, dense perfect-detector
        // input every tick — should lock on and stay locked.
        use tracking_core::Config;

        let dir = unique_temp_dir();
        const N: usize = 20;
        let mut gt_lines = vec!["frame,track_id,x1,y1,x2,y2".to_string()];
        for t in 0..N {
            let path = dir.join(format!("frame_{t:06}.png"));
            // Textured background so SimpleTracker's SAD matcher has
            // something to lock onto, matching synthetic.rs's rationale.
            let mut img = image::RgbImage::new(64, 48);
            for (x, y, px) in img.enumerate_pixels_mut() {
                *px = image::Rgb([(x * 3) as u8, (y * 5) as u8, ((x + y) * 2) as u8]);
            }
            let x1 = 5 + t as i64;
            let (y1, x2, y2) = (10i64, x1 + 8, 18i64);
            for yy in y1..y2 {
                for xx in x1..x2 {
                    img.put_pixel(xx as u32, yy as u32, image::Rgb([250, 250, 250]));
                }
            }
            img.save(&path).unwrap();
            gt_lines.push(format!("{t},0,{x1},{y1},{x2},{y2}"));
        }
        fs::write(dir.join("ground_truth.csv"), gt_lines.join("\n")).unwrap();

        let seq = load_sequence("e2e", &dir, &dir.join("ground_truth.csv")).unwrap();
        assert_eq!(seq.frames.len(), N);

        let states = crate::run(&seq, Config::default());
        let report = crate::metrics::score(&seq, &states);
        assert!(
            report.success_rate() > 0.8,
            "expected a straight-line real-frame target to be tracked reliably, got {report:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }
}
