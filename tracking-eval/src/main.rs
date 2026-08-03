//! `eval` — the tracking-core evaluation harness. Runs a fixed set of
//! synthetic scenarios (straight-line motion, occlusion + sparse detections,
//! a crossing distractor) through `TrackingSession<SimpleTracker>` and
//! prints objective scores against each scenario's known ground truth.
//!
//! Usage: `cargo run --bin eval` (or `make eval`). Exits non-zero if any
//! scenario's success rate falls below the threshold baked in below — a
//! deliberately loose regression gate, not a tuned target (see README "Eval
//! harness" for why).

use tracking_core::Config;
use tracking_eval::{metrics, run, synthetic};

/// Below this success rate on any scenario, `eval` exits non-zero. Loose on
/// purpose: this harness's job is to catch a real regression (a change that
/// breaks acquisition or association), not to chase a benchmark number for
/// `SimpleTracker`, which is a reference implementation, not the production
/// tracker.
const MIN_ACCEPTABLE_SUCCESS_RATE: f64 = 0.5;

fn main() {
    let num_ticks = 90;
    let config = Config::default();

    let scenarios = [
        synthetic::straight_line(num_ticks, 1),
        synthetic::occlusion_and_sparse_detections(num_ticks, 2),
        synthetic::distractor_crossing(num_ticks, 3),
    ];

    let mut all_pass = true;
    println!(
        "{:<32} {:>7} {:>9} {:>10} {:>8} {:>8} {:>8}",
        "scenario", "ticks", "success%", "mean_iou", "ttl", "reacq", "id_sw"
    );
    for seq in &scenarios {
        let states = run(seq, config);
        let report = metrics::score(seq, &states);
        let pass = report.success_rate() >= MIN_ACCEPTABLE_SUCCESS_RATE;
        all_pass &= pass;
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

    println!();
    println!(
        "success threshold: {:.0}% (IoU >= {:.1} counts as success on a visible-target tick)",
        MIN_ACCEPTABLE_SUCCESS_RATE * 100.0,
        metrics::SUCCESS_IOU
    );

    if !all_pass {
        eprintln!("eval: one or more scenarios fell below the acceptance threshold");
        std::process::exit(1);
    }
}
