use std::thread::sleep;
use zzz::*;

fn main() {
    for _ in (0..100_000_000_000u64)
        .into_iter()
        .with_pb(ProgressBar::spinner().force_spinner())
    {
        // do.. absolutely nothing but displaying the PB.
    }
}
