use zzz::*;

fn main() {
    let mut pb = ProgressBar::smart();
    pb.set_message(Some("Hot loop:"));

    for _ in (0..100_000_000_000u64).into_iter().with_pb(pb) {
        // do.. absolutely nothing but displaying the PB.
    }
}
