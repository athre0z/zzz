use futures::{stream, StreamExt};
use zzz::*;
use std::thread::sleep;
use std::time::Duration;

#[tokio::main]
async fn main() {
    for _ in (0..1000).into_iter().pb() {
        sleep(Duration::from_millis(10));
    }

    println!();

    let mut stream = stream::iter((0..1000).into_iter()).pb();
    while let Some(_) = stream.next().await {
        tokio::time::delay_for(Duration::from_millis(10)).await;
    }
}
