💤
===

[![Crates.io][crates-badge]][crates-url]
[![docs.rs][docs-badge]][docs-url]
[![MIT licensed][mit-badge]][mit-url]
[![Apache 2.0 licensed][apache-badge]][apache-url]

[crates-badge]: https://img.shields.io/crates/v/zzz.svg
[crates-url]: https://crates.io/crates/zzz
[docs-badge]: https://docs.rs/zzz/badge.svg
[docs-url]: https://docs.rs/zzz/
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: LICENSE-MIT
[apache-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[apache-url]: LICENSE-APACHE

Yet another progress bar library.

```toml
[dependencies]
zzz = { version = "*" }
```

## Core values

- Convenient application to iterators and streams
- Doesn't slow down your loop, pretty much no matter how simple the subject
  of the progress bar is, on average a ...
    - `!Sync` progress bar update is 4 instructions (3 CPU cycles on Skylake)
    - ` Sync` progress bar update is also 4 instructions (38 CPU cycles on Skylake[^1])
- Automagically determines and updates a good printing frequency
- Looks pretty decent out of the box

[^1]: For multi-threaded loops, cycle counts may vary 

## Usage examples

Looping over an iterator with a progress bar:
```rust
fn main() {
    //                            vvvvv
    for _ in (0..1000).into_iter().pb() {
        sleep(Duration::from_millis(10));
    }
}
```