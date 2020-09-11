use futures_core::core_reexport::pin::Pin;
use futures_core::task::{Context, Poll};
use futures_core::Stream;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
// use std::fmt;

/// Requirements
/// ------------
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
    updates_per_sec: 60.0,
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
    print!("{}", "█".repeat(i as usize));
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

fn human_amount(x: f32) -> String {
    let (n, unit) = if x > 1e9 {
        (x / 1e9, "b")
    } else if x > 1e6 {
        (x / 1e6, "m")
    } else if x > 1e3 {
        (x / 1e3, "k")
    } else {
        (x, "")
    };

    format!("{:.02}{}", n, unit)
}

fn spinner(x: f32, width: u32) -> String {
    fn easing_quad(mut x: f32) -> f32 {
        x *= 2.0;

        if x > 1.0 {
            -0.5 * ((x - 1.0) * (x - 3.0) - 1.0)
        } else {
            0.5 * x * x
        }
    }

    // fn easing_cubic(mut x: f32) -> f32 {
    //     x *= 2.0;
    //
    //     if x < 1.0 {
    //         0.5 * x.powi(3)
    //     } else {
    //         x -= 2.;
    //         0.5 * (x.powi(3) + 2.)
    //     }
    // }

    let x = ((-x + 0.5).abs() - 0.5) * -2.;
    let x = easing_quad(x).max(0.).min(1.);
    let x = (width as f32 * x).round() as i64;

    let lpad = x as usize;
    let rpad = (width as i64 - x) as usize;

    format!("|{}◯{}|", " ".repeat(lpad), " ".repeat(rpad))
}

impl ProgressBarTheme for DefaultProgressBarTheme {
    fn render(&self, pb: &ProgressBar) {
        print!("\r");

        // If a description is set, print it now.
        if let Some(desc) = pb.cfg.desc.as_deref() {
            print!("{}: ", desc);
        }

        // Draw a progress bar for known-length bars.
        if let Some(progress) = pb.progress() {
            print!("{:>6.2}% |", progress * 100.0);

            let bar_len = bar(progress, pb.cfg.width);
            for _ in 0..(pb.cfg.width as i64 - bar_len as i64).max(0) {
                print!(" ");
            }

            print!("|");
        }
        // And a spinner for unknown-length bars.
        else {
            let duration = Duration::from_secs(2);
            let pos = pb.timer_progress(duration);

            // Make the spinner turn around in the end.
            print!("{}", spinner(pos, pb.cfg.width));
        }

        // Print "done/total" part
        print!(
            " {}/{}",
            pb.value(),
            pb.target
                .map(|x| x.to_string())
                .unwrap_or_else(|| "?".to_owned())
        );

        if let Some(eta) = pb.eta() {
            print!(" [{}]", human_time(eta));
        } else {
            print!(" {}", human_time(pb.elapsed()));
        }

        // Print iteration rate.
        let iters_per_sec = pb.iters_per_sec();
        if iters_per_sec >= 1.0 {
            print!(" ({} it/s)", human_amount(iters_per_sec));
        } else {
            print!(" ({:.0} s/it)", 1.0 / iters_per_sec);
        }

        std::io::stdout().flush().unwrap();
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

impl Drop for ProgressBar {
    fn drop(&mut self) {
        println!();
    }
}

/// Constructors.
impl ProgressBar {
    pub fn new(target: Option<usize>) -> Self {
        Self {
            cfg: &DEFAULT_CFG,
            value: 1.into(),
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
    pub fn config(mut self, cfg: &'static ProgressBarConfig) -> Self {
        self.cfg = cfg;
        self
    }
}

/// Value manipulation and access.
impl ProgressBar {
    #[inline]
    pub fn sync_set(&self, n: usize) {
        self.update_ctr.fetch_add(1, ATOMIC_ORD);
        self.value.store(n, ATOMIC_ORD);
    }

    #[inline]
    pub fn set(&mut self, n: usize) {
        *self.update_ctr.get_mut() += 1;
        *self.value.get_mut() = n;
    }

    #[inline]
    pub fn sync_add(&self, n: usize) -> usize {
        self.value.fetch_add(n, ATOMIC_ORD);
        self.update_ctr.fetch_add(1, ATOMIC_ORD)
    }

    #[inline]
    pub fn add(&mut self, n: usize) -> usize {
        *self.value.get_mut() += n;
        let prev = *self.update_ctr.get_mut();
        *self.update_ctr.get_mut() += 1;
        prev
    }

    #[inline]
    pub fn update_ctr(&self) -> usize {
        self.update_ctr.load(ATOMIC_ORD)
    }

    #[inline]
    pub fn value(&self) -> usize {
        self.value.load(ATOMIC_ORD)
    }

    #[inline]
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

    pub fn iters_per_sec(&self) -> f32 {
        let elapsed_sec = self.elapsed().as_secs_f32();
        self.value() as f32 / elapsed_sec
    }

    /// Calculates the progress of a rolling timer.
    ///
    /// Returned values are always between 0 and 1. Timers are calculated
    /// from the start of the progress bar.
    pub fn timer_progress(&self, timer: Duration) -> f32 {
        let elapsed_sec = self.elapsed().as_secs_f32();
        let timer_sec = timer.as_secs_f32();

        (elapsed_sec % timer_sec) / timer_sec
    }

    #[inline]
    pub fn sync_tick(&self) {
        if self.sync_add(1) == self.next_print() {
            self.heavy_tick();
        }
    }

    #[inline]
    pub fn tick(&mut self) {
        if self.add(1) == self.next_print() {
            self.heavy_tick();
        }
    }
}

/// Internals.
impl ProgressBar {
    #[inline]
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

        let freq = (self.iters_per_sec() / self.cfg.updates_per_sec) as usize;
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
        let next = self.inner.next()?;
        self.bar.tick();
        Some(next)
    }
}

impl<I: Iterator> ProgressBarIter<I> {
    pub fn into_inner(self) -> I {
        self.inner
    }
}

pub trait ProgressBarIterExt: Iterator + Sized {
    fn pb(self) -> ProgressBarIter<Self> {
        let mut bar = ProgressBar::spinner();
        // bar.process_size_hint(self.size_hint());
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

impl<S: Stream + Unpin> ProgressBarStream<S> {
    pub fn into_inner(self) -> S {
        self.inner
    }
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
