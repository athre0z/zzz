use rayon::prelude::*;
use zzz::*;

fn main() {
    let pb = ProgressBar::smart();
    pb.set_desc_sync(Some("Parallel hot loop:"));

    (0..100_000_000_000u64)
        .into_par_iter()
        .for_each(|_| pb.tick_sync());
}
