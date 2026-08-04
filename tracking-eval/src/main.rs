//! `eval` — the tracking-core evaluation harness.
//!
//! Two modes:
//!
//! - **default** (no args): runs the fixed set of synthetic scenarios
//!   (straight-line motion, occlusion + sparse detections, a crossing
//!   distractor) through `TrackingSession<SimpleTracker>` and prints
//!   objective scores against each scenario's known ground truth. Exits
//!   non-zero if any scenario's success rate falls below the threshold baked
//!   in below — a deliberately loose regression gate, not a tuned target
//!   (see README "Eval harness" for why).
//! - **`real`**: runs a real image-sequence (optionally decoded from a real
//!   video file) against real ground-truth track annotations, through the
//!   same `TrackingSession<SimpleTracker>` and the same scoring — see
//!   `tracking_eval::real`'s module docs. Usage:
//!
//!   ```text
//!   cargo run --bin eval -- real --frames <dir> [--gt <file>] [--video <file>]
//!   ```
//!
//!   `--frames` is an image-sequence directory (read directly, or as the
//!   extraction target for `--video` if given). `--gt` defaults to
//!   `<frames>/ground_truth.csv` if omitted.

use std::path::PathBuf;
use std::process::ExitCode;

use tracking_core::Config;
use tracking_eval::{metrics, real, run, synthetic};

/// Below this success rate on any scenario, `eval` exits non-zero. Loose on
/// purpose: this harness's job is to catch a real regression (a change that
/// breaks acquisition or association), not to chase a benchmark number for
/// `SimpleTracker`, which is a reference implementation, not the production
/// tracker.
const MIN_ACCEPTABLE_SUCCESS_RATE: f64 = 0.5;

fn print_header() {
    println!(
        "{:<32} {:>7} {:>9} {:>10} {:>8} {:>8} {:>8}",
        "scenario", "ticks", "success%", "mean_iou", "ttl", "reacq", "id_sw"
    );
}

fn print_report(report: &metrics::Report, pass: bool) {
    println!(
        "{:<32} {:>7} {:>8.1}% {:>10} {:>8} {:>8} {:>8}{}",
        report.scenario,
        report.num_ticks,
        report.success_rate() * 100.0,
        report.mean_iou_locked.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into()),
        report.time_to_lock.map(|v| v.to_string()).unwrap_or_else(|| "never".into()),
        report.reacquisition_ticks.map(|v| v.to_string()).unwrap_or_else(|| "n/a".into()),
        report.id_switches,
        if pass { "" } else { "  <-- BELOW THRESHOLD" },
    );
}

fn run_synthetic() -> bool {
    let num_ticks = 90;
    let config = Config::default();

    let scenarios = [
        synthetic::straight_line(num_ticks, 1),
        synthetic::occlusion_and_sparse_detections(num_ticks, 2),
        synthetic::distractor_crossing(num_ticks, 3),
    ];

    let mut all_pass = true;
    print_header();
    for seq in &scenarios {
        let states = run(seq, config);
        let report = metrics::score(seq, &states);
        let pass = report.success_rate() >= MIN_ACCEPTABLE_SUCCESS_RATE;
        all_pass &= pass;
        print_report(&report, pass);
    }

    println!();
    println!(
        "success threshold: {:.0}% (IoU >= {:.1} counts as success on a visible-target tick)",
        MIN_ACCEPTABLE_SUCCESS_RATE * 100.0,
        metrics::SUCCESS_IOU
    );

    all_pass
}

struct RealArgs {
    frames: PathBuf,
    gt: Option<PathBuf>,
    video: Option<PathBuf>,
}

fn parse_real_args(args: &[String]) -> Result<RealArgs, String> {
    let mut frames = None;
    let mut gt = None;
    let mut video = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" => {
                frames = Some(PathBuf::from(args.get(i + 1).ok_or("--frames needs a value")?));
                i += 2;
            }
            "--gt" => {
                gt = Some(PathBuf::from(args.get(i + 1).ok_or("--gt needs a value")?));
                i += 2;
            }
            "--video" => {
                video = Some(PathBuf::from(args.get(i + 1).ok_or("--video needs a value")?));
                i += 2;
            }
            other => return Err(format!("unrecognised argument: {other}")),
        }
    }
    Ok(RealArgs {
        frames: frames.ok_or("real mode needs --frames <dir>")?,
        gt,
        video,
    })
}

fn run_real(args: &RealArgs) -> Result<bool, String> {
    if let Some(video) = &args.video {
        println!("extracting frames from {video:?} into {:?} via ffmpeg...", args.frames);
        real::extract_video_frames(video, &args.frames).map_err(|e| e.to_string())?;
    }
    let gt_path = args.gt.clone().unwrap_or_else(|| args.frames.join("ground_truth.csv"));
    let seq = real::load_sequence("real_video", &args.frames, &gt_path).map_err(|e| e.to_string())?;

    println!(
        "loaded {} frames ({}x{}) from {:?}, ground truth {:?}",
        seq.frames.len(),
        seq.width,
        seq.height,
        args.frames,
        gt_path
    );

    let config = Config::default();
    let states = run(&seq, config);
    let report = metrics::score(&seq, &states);
    let pass = report.success_rate() >= MIN_ACCEPTABLE_SUCCESS_RATE;

    print_header();
    print_report(&report, pass);
    println!();
    println!(
        "success threshold: {:.0}% (IoU >= {:.1} counts as success on a visible-target tick)",
        MIN_ACCEPTABLE_SUCCESS_RATE * 100.0,
        metrics::SUCCESS_IOU
    );

    Ok(pass)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let all_pass = if args.first().map(String::as_str) == Some("real") {
        match parse_real_args(&args[1..]) {
            Ok(real_args) => match run_real(&real_args) {
                Ok(pass) => pass,
                Err(e) => {
                    eprintln!("eval real: {e}");
                    return ExitCode::FAILURE;
                }
            },
            Err(e) => {
                eprintln!("eval real: {e}");
                eprintln!("usage: eval real --frames <dir> [--gt <file>] [--video <file>]");
                return ExitCode::FAILURE;
            }
        }
    } else if args.is_empty() {
        run_synthetic()
    } else {
        eprintln!("usage: eval | eval real --frames <dir> [--gt <file>] [--video <file>]");
        return ExitCode::FAILURE;
    };

    if !all_pass {
        eprintln!("eval: one or more scenarios fell below the acceptance threshold");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
