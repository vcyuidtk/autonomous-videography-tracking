# autonomous-videography-tracking

Single-target visual-tracking algorithm — extracted from the
`autonomous-videography` Rust piloting stack so it can be developed, tested,
and reused independently of that stack's bus/ring/orchestrator plumbing.

This is **the algorithm**, not a video pipeline: detection association,
tracker init/update, and the lock → refresh → loss → age-out state machine.
It does not decode video, does not talk to a message bus, and does not know
what a drone is.

## Workspace layout

| Crate | What it is | System deps |
|---|---|---|
| [`tracking-core`](tracking-core) | The algorithm: acquisition, seedability gating, association, the `Stage` state machine, and `TrackingSession` (the public entry point). Includes `SimpleTracker`, a dependency-free reference tracker backend. | none |
| [`tracking-cv`](tracking-cv) | A production-grade `Tracker` backend wrapping OpenCV's `TrackerCSRT` (contrib). | libclang, libopencv (+contrib) |
| [`tracking-eval`](tracking-eval) | Synthetic-data evaluation harness — generates known-ground-truth sequences and scores a tracking run against them. Ships the `eval` binary. | none |

`tracking-core` has no dependency on `tracking-cv` or vice versa (`tracking-cv`
depends on `tracking-core`, never the reverse) — a consumer who doesn't need
real CSRT (tests, another tracker backend, a different CV library entirely)
never pulls in OpenCV or libclang.

## Interface design

The headline type is [`TrackingSession`](tracking-core/src/session.rs):

```rust
let mut session = TrackingSession::new(SimpleTracker::new(), frame_w, frame_h, Config::default());

// Every video frame:
let state = session.step(frame_view, detections, now_s);
match state {
    TrackState::Idle => { /* nothing locked */ }
    TrackState::Locked(held) => { /* held.bbox is this tick's box */ }
    TrackState::Lost(held) => { /* held.bbox is the last-known box, not fabricated */ }
}
```

**Why a synchronous push-frame/pull-state call, not an actor/channel API or
an async stream:** the algorithm itself has no concurrency or I/O in it —
every real dependency (video decode, detector inference, IPC) belongs to the
caller. A synchronous function call is the smallest interface that doesn't
presuppose a threading model, so it works equally well called directly from
a blocking loop, from inside an async task via `spawn_blocking`, or (as here)
from a batch evaluation harness with no runtime at all. The cost is that the
caller owns frame pacing and scheduling — reasonable, since a video pipeline
mostly always already needs to own that.

**Why `detections: Option<&[Detection]>` and not just `&[Detection]`:** the
detector very often runs slower than the video (e.g. video at 30fps,
detector inference at 10fps). `None` means "the detector produced nothing
this tick — this is a video-only tick, don't touch the acquisition
confirmation candidate." `Some(&[])` means "the detector ran and genuinely
saw zero qualifying boxes" — a real negative observation the acquisition
policy is entitled to act on. Collapsing these two into one `&[]` case would
silently break the two-batch confirmation gate (see `tracking-core`'s
`session.rs` docs) on every tick between detector outputs, because "an empty
batch just arrived" would wipe the pending confirmation candidate before the
next real detection had a chance to confirm it. This is not a hypothetical:
the `occlusion_and_sparse_detections` eval scenario (sparse, cadence-3
detections) caught exactly this during this crate's development, when
`step()`'s first draft took a plain `&[Detection]` — success rate on that
scenario was 0% until the signature changed to `Option<&[Detection]>`.

**Why single-target, not a multi-track API:** this mirrors the algorithm
that was actually running in production — one locked target at a time, with
class-agnostic two-batch confirmation deciding *which* detection to lock
onto. A multi-target tracker is a materially different algorithm (needs
identity/re-association across tracks, not just against one held box); the
honest scope call was to extract what exists rather than design a new
system. Running several independent `TrackingSession`s side by side is the
straightforward way to track several targets with this crate, if the
targets don't need to reason about each other.

**Why `Stage` is also exported, separately from `TrackingSession`:** a
caller whose detections and frames arrive on different schedules (route
several detection batches into `on_detection_batch` before the frame for
this tick is ready) needs the two-phase API `TrackingSession::step` collapses
for the common case. Use `Stage` directly for that; `TrackingSession` is a
thin wrapper over it, not a separate implementation.

## Provenance

Ported from the `autonomous-videography` monorepo's `av-track`, `av-cv`, and
`tracking::stage::Stage` (Rust port, W6, GH `autonomous-videography` issue
#186 and follow-ups). The port kept the algorithm and its test coverage
byte-for-byte equivalent in behaviour, stripping only:

- bus/ring/IPC types (`av-interfaces`'s wire schema) — replaced with this
  crate's own minimal `BBox`/`Detection`/`TrackState` types, see
  `tracking-core/src/types.rs`'s module docs for why
- `FrameId`/`CameraId`/`Label` bookkeeping — that's the host pipeline's
  concern, not the algorithm's
- the `tracking` binary's bus wiring (`main.rs`, `config.rs`, `pending.rs`)
  — stays in `autonomous-videography` as glue code, not algorithm

The commit history in this repository starts from that extraction and says
so in each commit that ports a file.

## Building `tracking-cv`

`tracking-cv` links `opencv-rust` against a real OpenCV build with the
`tracking` (contrib) and `video` features, same pin as the original
`av-cv` crate. You need:

- `libclang` (for `bindgen`) — Ubuntu ships it as `libclang-<N>.so`, not
  `libclang.so`; if `cargo build -p tracking-cv` fails with "couldn't find
  any valid shared libraries matching \['libclang.so', ...\]", point
  `LIBCLANG_PATH` at a directory containing a `libclang.so` symlink to the
  real versioned file.
- `libopencv-dev` with the `tracking` contrib module built in — a plain
  `apt install libopencv-dev` on some Ubuntu releases does NOT include
  contrib; if `TrackerCSRT`/`opencv::tracking` fails to resolve, you need an
  OpenCV build with `OPENCV_EXTRA_MODULES_PATH` pointed at `opencv_contrib`.

`tracking-core` and `tracking-eval` need neither — that split is the point.

## Tests

```
cargo test --workspace          # everything (needs tracking-cv's system deps)
cargo test -p tracking-core     # algorithm only, zero system deps
```

36 unit tests total as of this writing: acquisition/seedability/state-machine
logic in `tracking-core` (32, including the ported `SimpleTracker`
reference-backend tests), real-CSRT integration tests in `tracking-cv` (4).

## Eval harness

```
cargo run --bin eval     # or: make eval
```

Runs three synthetic scenarios through `TrackingSession<SimpleTracker>` and
prints objective scores against known ground truth — no real video, no
OpenCV. `SimpleTracker` (a grayscale SAD template matcher, see
`tracking-core/src/simple_tracker.rs`) stands in for a real backend so the
harness has zero system dependencies and stays fast; it exercises the
*algorithm* (acquisition/association/state machine), not any particular
tracker implementation's pixel-level fidelity.

**Scenarios:**

- `straight_line` — a target moving steadily across the frame, detections
  every tick with small jitter. The easy baseline.
- `occlusion_and_sparse_detections` — the same trajectory, but the target is
  occluded (undetectable) for a stretch mid-sequence, and the detector only
  runs every 3rd tick with 10% false-negative noise. Exercises loss,
  age-out, and re-acquisition.
- `distractor_crossing` — a second, similar-looking object crosses close to
  the real target mid-sequence. Exercises whether the class-agnostic
  acquisition/refresh policy ever locks onto the wrong object.

**Metrics** (see `tracking-eval/src/metrics.rs`):

- **Success rate** — fraction of ground-truth-visible ticks where the
  reported box has IoU ≥ 0.5 against ground truth (the standard VOT/OTB
  single-object-tracking success bar). This is the headline number; `eval`
  exits non-zero if any scenario falls below 50%.
- **Mean IoU (locked)** — average IoU against ground truth over ticks
  reported `Locked`, a finer-grained accuracy signal than the pass/fail rate.
- **Time-to-lock** — ticks from sequence start to the first `Locked` state.
- **Reacquisition ticks** — ticks from a target's re-appearance after
  occlusion to the next `Locked` state (only meaningful for the occlusion
  scenario).
- **ID switches** — count of ticks where the reported box's IoU against the
  distractor's ground truth exceeds its IoU against the real target's — a
  wrong-object lock, counted once per contiguous switched run (not per
  tick).

These were picked because they map directly onto the three failure modes the
state machine exists to prevent (never locking, losing and not recovering,
locking onto the wrong thing) rather than chasing a single scalar score.

**Representative run** (90 ticks/scenario, fixed seeds — deterministic,
your numbers will match exactly):

```
scenario                           ticks  success%   mean_iou      ttl    reacq    id_sw
straight_line                         90     98.9%      0.868        1      n/a        0
occlusion_and_sparse_detections       90     72.0%      0.819        3       18        0
distractor_crossing                   90     93.3%      0.759        1      n/a        1

success threshold: 50% (IoU >= 0.5 counts as success on a visible-target tick)
```

The `distractor_crossing` scenario's 1 ID switch is a real, expected finding,
not a bug: the class-agnostic central-acquisition policy has no notion of
object identity, so a distractor that becomes more central than the real
target for a tick or two *can* steal association — this is a known,
documented trade-off in the original design (HWC-43 in the source
provenance), not something this extraction introduced or hid.

## Consuming this crate from a host pipeline

1. Depend on `tracking-core` (path or git dependency); add `tracking-cv` too
   if you want the real CSRT backend rather than `SimpleTracker`.
2. Implement (or reuse `tracking-cv::CsrtTracker`) the `Tracker` trait for
   whatever CV backend you're using.
3. At your pipeline's detector-output boundary, map your own detection type
   to `tracking_core::Detection` (an opaque `id: u64` you control lets you
   map a `TrackState`'s `source_detection_id` back to your richer record —
   label, embedding, whatever this crate doesn't need to know about).
4. Own a `TrackingSession` per tracked stream; call `step()` once per video
   frame with that tick's fresh detections (or `None` if your detector
   didn't produce output this tick — see "Interface design" above for why
   that distinction matters).
5. Map `TrackState` back to your own wire/bus types at the boundary, same as
   step 3 in reverse.

`autonomous-videography`'s `tracking` binary is the reference consumer —
see its crate for the bus/ring wiring this crate deliberately does not
include.
