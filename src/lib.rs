#![feature(rustc_attrs)]

use futures_core::core_reexport::pin::Pin;
use futures_core::task::{Context, Poll};
use futures_core::Stream;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
// use std::fmt;

/// Requirements:
/// - Seamless integration with iterators and streams
/// - Unless explicitly set, try to infer size from iter/stream
/// - Automatically throttle printing to a sane amount
/// - The PB should render at a somewhat consistent freq
/// - It must be expected that the iterator's yield speed varies greatly
///   over the execution period. Often, the first few iterations will take
///   quite a bit longer due to things such as file system cache warmup
/// - The PB shouldn't be a bottleneck for even the most mundane and tight
///   loops running at millions of iterations per second. Optimally, the
///   amortized case should not involve more than a few adds and one branch.
/// - Try to implement the throttling without doing a syscall or RDTSC
/// - It must be expected that, for non-iterator PBs, the value is bumped
///   by more than one, so using modulo on the value for determining when to
///   print is not an option
///   - Also, modulo with a dynamic RHS is a pretty heavy operation
/// - `!#[no_std]` would be nice to have

const ATOMIC_ORD: Ordering = Ordering::Relaxed;

// ========================================================================== //
// [General configuration]                                                    //
// ========================================================================== //

pub struct ProgressBarConfig {
    width: u32,
    desc: Option<String>,
    theme: &'static dyn ProgressBarTheme,
    updates_per_sec: f32,
}

static DEFAULT_CFG: ProgressBarConfig = ProgressBarConfig {
    width: 60,
    desc: None,
    theme: &DefaultProgressBarTheme,
    updates_per_sec: 5.0,
};

// ========================================================================== //
// [Customizable printing]                                                    //
// ========================================================================== //

pub trait ProgressBarTheme: Sync {
    fn render(&self, pb: &ProgressBar);
}

#[derive(Debug, Default)]
struct DefaultProgressBarTheme;

fn bar(progress: f32, length: u32) -> u32 {
    let rescaled = (progress * length as f32 * 8.0) as u32;
    let (i, r) = (rescaled / 8, rescaled % 8);

    for _ in 0..i {
        print!("█");
    }

    let chr = '▏' as u32 - r;
    let chr = unsafe { std::char::from_u32_unchecked(chr) };
    print!("{}", chr);

    i + 1
}

fn human_time(duration: Duration) -> String {
    let total = duration.as_secs();

    let h = total / 3600;
    let m = total % 3600 / 60;
    let s = total % 60;

    format!("{:02}:{:02}:{:02}", h, m, s)
}

impl ProgressBarTheme for DefaultProgressBarTheme {
    fn render(&self, pb: &ProgressBar) {
        let progress = pb.progress().unwrap();

        print!("\r");

        if let Some(desc) = pb.cfg.desc.as_deref() {
            print!("{}: ", desc);
        }

        print!("{:>6.2}% ", progress * 100.0);
        let bar_len = bar(progress, pb.cfg.width);

        for _ in 0..(pb.cfg.width as i64 - bar_len as i64).max(0) {
            print!(" ");
        }

        print!(" {}/{}", pb.value(), pb.target.unwrap());

        if let Some(eta) = pb.eta() {
            print!(" ETA: {}", human_time(eta));
        } else {
            print!(" {}", human_time(pb.elapsed()));
        }

        std::io::stdout().flush();
    }
}

// ========================================================================== //
// [Main progress bar struct]                                                 //
// ========================================================================== //

pub struct ProgressBar {
    /// Configuration to use.
    cfg: &'static ProgressBarConfig,
    /// The expected, possibly approximate target of the progress bar.
    target: Option<usize>,
    /// Whether the target was specified explicitly.
    explicit_target: bool,
    /// Creation time of the progress bar.
    start: Instant,
    /// Progress value displayed to the user.
    value: AtomicUsize,
    /// Number of progress bar updates so far.
    update_ctr: AtomicUsize,
    /// Next print at `update_ctr == next_print`.
    next_print: AtomicUsize,
}

/// Constructors.
impl ProgressBar {
    pub fn new(target: Option<usize>) -> Self {
        Self {
            cfg: &DEFAULT_CFG,
            value: 0.into(),
            update_ctr: 0.into(),
            target,
            next_print: 1.into(),
            explicit_target: target.is_some(),
            start: Instant::now(),
        }
    }

    pub fn spinner() -> Self {
        Self::new(None)
    }

    pub fn with_target(target: usize) -> Self {
        Self::new(Some(target))
    }
}

/// Builder-style methods.
impl ProgressBar {
    pub fn config(self, cfg: &'static ProgressBarConfig) -> Self {
        Self { cfg, ..self }
    }
}

/// Value manipulation and access.
impl ProgressBar {
    pub fn set(&self, n: usize) {
        self.update_ctr.fetch_add(1, ATOMIC_ORD);
        self.value.store(n, ATOMIC_ORD);
    }

    pub fn add(&self, n: usize) -> usize {
        self.value.fetch_add(n, ATOMIC_ORD);
        self.update_ctr.fetch_add(1, ATOMIC_ORD)
    }

    pub fn inc(&self) -> usize {
        self.add(1)
    }

    pub fn update_ctr(&self) -> usize {
        self.update_ctr.load(ATOMIC_ORD)
    }

    pub fn value(&self) -> usize {
        self.value.load(ATOMIC_ORD)
    }

    pub fn progress(&self) -> Option<f32> {
        let target = self.target?;
        Some(self.value() as f32 / target as f32)
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn eta(&self) -> Option<Duration> {
        // wen eta?!
        let left = 1. / self.progress()?;
        Some(self.elapsed().mul_f32(left))
    }
}

/// Internals.
impl ProgressBar {
    fn next_print(&self) -> usize {
        self.next_print.load(ATOMIC_ORD)
    }

    /// Calculate next print.
    fn update_next_print(&self) {
        // Give the loop some time to warm up.
        if self.update_ctr() < 10 {
            self.next_print.fetch_add(1, ATOMIC_ORD);
            return;
        }

        let elapsed_sec = self.elapsed().as_secs_f32();
        let iters_per_sec = self.value() as f32 / elapsed_sec;
        let freq = (iters_per_sec / self.cfg.updates_per_sec) as usize;
        let freq = freq.max(1);

        self.next_print.fetch_add(freq as usize, ATOMIC_ORD);
    }

    #[rustfmt::skip]
    fn process_size_hint(&mut self, hint: (usize, Option<usize>)) {
        if self.explicit_target {
            return;
        }

        // Prefer hi over lo, treat lo = 0 as unknown.
        self.target = match hint {
            (_ , Some(hi)) => Some(hi),
            (0 , None    ) => None,
            (lo, None    ) => Some(lo),
        };
    }

    #[cold]
    fn heavy_tick(&self) {
        self.cfg.theme.render(self);
        self.update_next_print();
    }

    fn tick(&self) {
        if self.inc() == self.next_print() {
            self.heavy_tick();
        }
    }
}

// ========================================================================== //
// [Iterator integration]                                                     //
// ========================================================================== //

pub struct ProgressBarIter<I: Iterator> {
    bar: ProgressBar,
    inner: I,
}

impl<I: Iterator> Iterator for ProgressBarIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|x| {
            self.bar.tick();
            x
        })
    }
}

pub trait ProgressBarIterExt: Iterator + Sized {
    fn pb(self) -> ProgressBarIter<Self> {
        let mut bar = ProgressBar::spinner();
        bar.process_size_hint(self.size_hint());
        ProgressBarIter { bar, inner: self }
    }

    fn with_pb(self, bar: ProgressBar) -> ProgressBarIter<Self> {
        ProgressBarIter { bar, inner: self }
    }
}

impl<I: Iterator + Sized> ProgressBarIterExt for I {}

// ========================================================================== //
// [Stream integration]                                                       //
// ========================================================================== //

pub struct ProgressBarStream<S: Stream + Unpin> {
    bar: ProgressBar,
    inner: S,
}

impl<S: Stream + Unpin> Stream for ProgressBarStream<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = Pin::into_inner(self);
        let inner = Pin::new(&mut this.inner);

        match inner.poll_next(cx) {
            x @ Poll::Ready(Some(_)) => {
                this.bar.tick();
                x
            }
            x => x,
        }
    }
}

pub trait ProgressBarStreamExt: Stream + Unpin + Sized {
    fn pb(self) -> ProgressBarStream<Self> {
        let mut bar = ProgressBar::spinner();
        bar.process_size_hint(self.size_hint());
        ProgressBarStream { bar, inner: self }
    }
}

impl<S: Stream + Unpin + Sized> ProgressBarStreamExt for S {}
