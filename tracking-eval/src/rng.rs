//! A tiny deterministic PRNG (xorshift64*) — the eval harness needs
//! reproducible "noisy" synthetic data (jittered detection boxes, dropped
//! frames) without pulling in the `rand` crate for one call site.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform float in `[0.0, 1.0)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[lo, hi]`, inclusive.
    pub fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.f64() * (hi - lo + 1) as f64) as i64
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }
}
