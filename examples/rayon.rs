use rayon::prelude::*;
use zzz::*;

fn main() {
    let mut pb = ProgressBar::smart();
    pb.set_message(Some("Parallel hot loop:"));

    (0..100_000_000_000u64).into_par_iter().for_each(|_| {
        pb.add_sync(1);
    });
}
