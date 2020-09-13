use std::thread::sleep;
use std::time::Duration;
use zzz::*;

fn main() {
    for _ in (0..1000)
        .into_iter()
        .with_pb(ProgressBar::spinner().force_spinner())
    {
        sleep(Duration::from_millis(33));
    }
}
