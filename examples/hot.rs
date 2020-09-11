use std::thread::sleep;
use std::time::Duration;
use zzz::*;

fn main() {
    for _ in (0..100_000_000_000u64).into_iter().pb() {
        // do.. absolutely nothing but displaying the PB.
    }
}
