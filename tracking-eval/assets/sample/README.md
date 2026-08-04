# Sample clip for `eval real`

`sample.mp4` (60 frames, 320x240, 15fps, H.264) + `ground_truth.csv` (one
`track_id 0` box per frame) is a real MP4 file, decoded through a real
`ffmpeg`-based decode path and a real PNG decode (`image` crate) by
`tracking_eval::real` — see that module's docs for the full pipeline.

**Honesty note:** the clip's *content* is procedurally generated (a
textured background + a solid box moving in a straight line at a known,
exactly-computed trajectory), not camera footage of a real scene. No
real-world sample video ships in this repo. What this asset genuinely
exercises end-to-end is real video **decode** (container demux, H.264
decode, PNG re-encode/decode, filesystem I/O) feeding real frame buffers
into `TrackingSession::step` — the actual gap issue #1 identified
(`tracking-eval` never touched a decoded frame). It does not exercise
tracker robustness against real-world noise, compression artefacts from a
genuine capture, lighting, motion blur, etc., the way real footage would.

Regenerated with:

```sh
python3 generate_sample.py   # writes frames/*.png + frames/ground_truth.csv
ffmpeg -y -framerate 15 -i frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p sample.mp4
cp frames/ground_truth.csv ground_truth.csv
```

(`generate_sample.py` is not checked in — it's a ~20-line PIL script that
draws a moving box on a synthetic textured background and writes ground
truth to match exactly; regenerate similarly if you need a different
trajectory.)
