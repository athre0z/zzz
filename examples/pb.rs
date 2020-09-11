use std::thread::sleep;
use std::time::Duration;
use zzz::*;

fn main() {
    for _ in (0..1000).into_iter().pb() {
        sleep(Duration::from_millis(33));
    }
}
