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

```toml
[dependencies]
zzz = { version = "0.1" }
```

## Features

- Practical defaults
- Seamless integration with iterators and streams
  - If possible, `zzz` infers the target size from `size_hint()`
- Automagically determines and updates a good printing frequency
- Very low overhead: doesn't slow down your loop, pretty much no matter how simple the loop body. On average a ...
    - `!Sync` progress bar update is 4 instructions (on average, 3 CPU cycles on Skylake)
    - ` Sync` progress bar update is also 4 instructions (on average, 38 CPU cycles on Skylake)
  
## Cargo Features
- `streams`: Enables support for `.progress()` on async streams (`futures::streams::Stream`)

## Usage examples

Adding a progress bar to an iterator:
```rust
use zzz::ProgressBarIterExt as _;

//                             vvvvvvvv
for _ in (0..1000).into_iter().progress() {
    // ...
}
```

Manually creating and advancing a progress bar:
```rust
use zzz::ProgressBar;

let mut pb = ProgressBar::with_target(1234);
for i in 0..1234 {
    // ...
    pb.add(1);
}
```