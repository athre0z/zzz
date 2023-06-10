💤
===

[![Crates.io][crates-badge]][crates-url]
[![docs.rs][docs-badge]][docs-url]
[![MIT licensed][mit-badge]][mit-url]
[![Apache 2.0 licensed][apache-badge]][apache-url]

The progress bar with sane defaults that doesn't slow down your loops. Inspired by [tqdm].

[crates-badge]: https://img.shields.io/crates/v/zzz.svg
[crates-url]: https://crates.io/crates/zzz
[docs-badge]: https://docs.rs/zzz/badge.svg
[docs-url]: https://docs.rs/zzz/
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: LICENSE-MIT
[apache-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[apache-url]: LICENSE-APACHE
[tqdm]: https://github.com/tqdm/tqdm.git
[screenshot]: https://i.imgur.com/9oMncYv.png

![Screenshot][screenshot]

```toml
[dependencies]
zzz = "0.3"
```

## Features

- Seamless integration with iterators and streams
  - If possible, `zzz` infers the target size from `size_hint()`
- Automagically determines and updates a good printing frequency
- Very low overhead: doesn't slow down your loop, pretty much no matter how simple the loop body. On Skylake, the average overhead per iteration for a
    - `!Sync`/`add` based progress bar is 3 CPU cycles
    - `Sync`/`add_sync` based progress bar is ~40 CPU cycles (depends on how many threads are updating the shared state)
  
## Cargo Features
- `streams`: Enables support for `.progress()` on async streams (`futures::streams::Stream`)

## Usage examples

**Adding a progress bar to an iterator**

```rust
use zzz::ProgressBarIterExt as _;

for _ in (0..1000).into_iter().progress() {
    //                         ^^^^^^^^
}
```

If `size_hint()` for the iterator defines an upper bound, it is automatically taken as the target. Otherwise, a progress indicator ("spinner") is displayed. 

**Manually creating and advancing a progress bar**
```rust
use zzz::ProgressBar;

let mut pb = ProgressBar::with_target(1234);
for _ in 0..1234 {
    pb.add(1);
}
```

**Manually creating a spinner** (for unknown target progress indicator)

```rust
use zzz::ProgressBar;

let mut pb = ProgressBar::spinner();
for _ in 0..5678 {
    pb.add(1);
}
```
