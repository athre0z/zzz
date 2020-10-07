use std::{
    fmt::Write as _,
    io::{stdout, Write as _},
    sync::atomic::{AtomicUsize, Ordering},
    sync::RwLock,
    time::{Duration, Instant},
};

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
    width: Option<u32>,
    /// Minimum width to bother with drawing the bar for.
    min_bar_width: u32,
    theme: &'static dyn ProgressBarTheme,
    max_fps: f32,
}

static DEFAULT_CFG: ProgressBarConfig = ProgressBarConfig {
    width: None,
    min_bar_width: 5,
    theme: &DefaultProgressBarTheme,
    max_fps: 60.0,
};

// ========================================================================== //
// [Utils]                                                                    //
// ========================================================================== //

/// Pads and aligns a value to the length of a cache line.
///
/// Adapted from crossbeam:
/// https://docs.rs/crossbeam/0.7.3/crossbeam/utils/struct.CachePadded.html
#[cfg_attr(target_arch = "x86_64", repr(align(128)))]
#[cfg_attr(not(target_arch = "x86_64"), repr(align(64)))]
pub struct CachePadded<T>(T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

// ========================================================================== //
// [Customizable printing]                                                    //
// ========================================================================== //

pub trait ProgressBarTheme: Sync {
    fn render(&self, pb: &ProgressBar);
}

#[derive(Debug, Default)]
struct DefaultProgressBarTheme;

/// Creates a unicode progress bar.
fn bar(progress: f32, length: u32) -> String {
    if length == 0 {
        return String::new();
    }

    let inner_len = length.saturating_sub(2);
    let rescaled = (progress * (inner_len - 1) as f32 * 8.0).round() as u32;
    let (i, r) = (rescaled / 8, rescaled % 8);
    let main = "█".repeat(i as usize);
    let tail = '▏' as u32 - r;
    let tail = unsafe { std::char::from_u32_unchecked(tail) };
    let pad_len = inner_len - i - 1 /* tail */;
    let pad = " ".repeat(pad_len as usize);

    let bar = format!("|{}{}{}|", main, tail, pad);
    debug_assert_eq!(bar.chars().count() as u32, length);
    bar
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
        (x / 1e9, "B")
    } else if x > 1e6 {
        (x / 1e6, "M")
    } else if x > 1e3 {
        (x / 1e3, "K")
    } else {
        (x, "")
    };

    format!("{:.02}{}", n, unit)
}

fn spinner(x: f32, width: u32) -> String {
    // Subtract two pipes + spinner char
    let inner_width = width.saturating_sub(3);

    // fn easing_inout_quad(mut x: f32) -> f32 {
    //     x *= 2.0;
    //
    //     if x > 1.0 {
    //         -0.5 * ((x - 1.0) * (x - 3.0) - 1.0)
    //     } else {
    //         0.5 * x * x
    //     }
    // }

    fn easing_inout_cubic(mut x: f32) -> f32 {
        x *= 2.0;

        if x < 1.0 {
            0.5 * x.powi(3)
        } else {
            x -= 2.;
            0.5 * (x.powi(3) + 2.)
        }
    }

    // fn easing_out_quad(x: f32) -> f32 {
    //     -x * (x - 2.)
    // }

    // Make the spinner turn around after half the period.
    let x = ((-x + 0.5).abs() - 0.5) * -2.;

    // Apply easing function.
    let x = easing_inout_cubic(x).max(0.).min(1.);
    // Transform 0..1 scale to int width.
    let x = ((inner_width as f32) * x).round() as u32;

    let lpad = x as usize;
    let rpad = inner_width.saturating_sub(x) as usize;

    let ball_offs = x / 8 % 8; // slow anim down
    let ball = unsafe { std::char::from_u32_unchecked('🌑' as u32 + ball_offs) };

    let spinner = format!("[{}{}{}]", " ".repeat(lpad), ball, " ".repeat(rpad));
    debug_assert_eq!(spinner.chars().count() as u32, width);
    spinner
}

/*
barr1 = UInt32[0x00, 0x40, 0x04, 0x02, 0x01]
barr2 = UInt32[0x00, 0x80, 0x20, 0x10, 0x08]
function braille(a::Float64, b::Float64)
    bchar(a::UInt32) = '⠀' + a
    idx(x) = min(x * 4 + 1, 5) |> round |> UInt32

    x = barr1[1:idx(a)] |> sum
    x |= barr2[1:idx(b)] |> sum

    x |> UInt32 |> bchar
end
*/

impl ProgressBarTheme for DefaultProgressBarTheme {
    fn render(&self, pb: &ProgressBar) {
        let mut o = stdout();

        // Draw left side.
        let left = {
            let mut buf = String::new();

            // If a description is set, print it.
            if let Some(desc) = pb.message() {
                write!(buf, "{} ", desc).unwrap();
            }

            if let Some(progress) = pb.progress() {
                write!(buf, "{:>6.2}% ", progress * 100.0).unwrap();
            }

            buf
        };

        // Draw right side.
        let right = {
            let mut buf = String::new();

            // Print "done/total" part
            write!(
                buf,
                " {}/{}",
                human_amount(pb.value() as f32),
                pb.target
                    .map(|x| human_amount(x as f32))
                    .unwrap_or_else(|| "?".to_owned())
            )
            .unwrap();

            // Print ETA / time elapsed.
            if let Some(eta) = pb.eta() {
                write!(buf, " [{}]", human_time(eta)).unwrap();
            } else {
                write!(buf, " [{}]", human_time(pb.elapsed())).unwrap();
            }

            // Print iteration rate.
            let iters_per_sec = pb.iters_per_sec();
            if iters_per_sec >= 1.0 {
                write!(buf, " ({} it/s)", human_amount(iters_per_sec)).unwrap();
            } else {
                write!(buf, " ({:.0} s/it)", 1.0 / iters_per_sec).unwrap();
            }

            buf
        };

        let max_width = pb
            .cfg
            .width
            .or_else(|| term_size::dimensions().map(|x| x.0 as u32))
            .unwrap_or(80);

        let bar_width = max_width
            .saturating_sub(left.len() as u32)
            .saturating_sub(right.len() as u32);

        write!(o, "{}", left).unwrap();

        if bar_width > pb.cfg.min_bar_width {
            // Draw a progress bar for known-length bars.
            if let Some(progress) = pb.progress() {
                write!(o, "{}", bar(progress, bar_width)).unwrap();
            }
            // And a spinner for unknown-length bars.
            else {
                let duration = Duration::from_secs(3);
                let pos = pb.timer_progress(duration);
                // Sub 1 from width because many terms render emojis with double width.
                write!(o, "{}", spinner(pos, bar_width - 1)).unwrap();
            }
        }

        write!(o, "{}\r", right).unwrap();

        o.flush().unwrap();
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
    /// Description of the progress bar, e.g. "Downloading image".
    message: RwLock<Option<String>>,
    /// Progress value displayed to the user.
    value: CachePadded<AtomicUsize>,
    /// Number of progress bar updates so far.
    update_ctr: CachePadded<AtomicUsize>,
    /// Next print at `update_ctr == next_print`.
    next_print: CachePadded<AtomicUsize>,
}

impl Drop for ProgressBar {
    fn drop(&mut self) {
        println!();
    }
}

/// Constructors.
impl ProgressBar {
    fn new(target: Option<usize>, explicit_target: bool) -> Self {
        Self {
            cfg: &DEFAULT_CFG,
            target,
            explicit_target,
            start: Instant::now(),
            value: CachePadded(0.into()),
            update_ctr: CachePadded(0.into()),
            next_print: CachePadded(1.into()),
            message: RwLock::new(None),
        }
    }

    pub fn smart() -> Self {
        Self::new(None, false)
    }

    pub fn spinner() -> Self {
        Self::new(None, true)
    }

    pub fn with_target(target: usize) -> Self {
        Self::new(Some(target), true)
    }
}

/// Builder-style methods.
impl ProgressBar {
    /// Replace the config of the progress bar.
    pub fn config(mut self, cfg: &'static ProgressBarConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Force display as a spinner even if size hints are present.
    pub fn force_spinner(mut self) -> Self {
        self.explicit_target = true;
        self.target = None;
        self
    }
}

/// Value manipulation and access.
impl ProgressBar {
    /// Set the progress bar value to a new, absolute value.
    ///
    /// See `set_sync` for a thread-safe version.
    #[inline]
    pub fn set(&mut self, n: usize) {
        *self.update_ctr.get_mut() += 1;
        *self.value.get_mut() = n;
    }

    /// Synchronized version fo `set`.
    #[inline]
    pub fn set_sync(&self, n: usize) {
        self.update_ctr.fetch_add(1, ATOMIC_ORD);
        self.value.store(n, ATOMIC_ORD);
    }

    /// Add `n` to the value of the progress bar.
    ///
    /// See `add_sync` for a thread-safe version.
    #[inline]
    pub fn add(&mut self, n: usize) -> usize {
        *self.value.get_mut() += n;
        let prev = *self.update_ctr.get_mut();
        *self.update_ctr.get_mut() += 1;
        prev
    }

    /// Synchronized version fo `add`.
    #[inline]
    pub fn add_sync(&self, n: usize) -> usize {
        self.value.fetch_add(n, ATOMIC_ORD);
        self.update_ctr.fetch_add(1, ATOMIC_ORD)
    }

    /// How often has the value been changed since creation?
    #[inline]
    pub fn update_ctr(&self) -> usize {
        self.update_ctr.load(ATOMIC_ORD)
    }

    /// Get the current value of the progress bar.
    #[inline]
    pub fn value(&self) -> usize {
        self.value.load(ATOMIC_ORD)
    }

    /// Get the current task description text.
    pub fn message(&self) -> Option<String> {
        self.message.read().unwrap().clone()
    }

    /// Set the current task description text.
    pub fn set_message(&mut self, text: Option<impl Into<String>>) {
        *self.message.get_mut().unwrap() = text.map(Into::into);
    }

    /// Synchronized version fo `set_message`.
    pub fn set_message_sync(&self, text: Option<impl Into<String>>) {
        let mut message_lock = self.message.write().unwrap();
        *message_lock = text.map(Into::into);
    }

    /// Calculate the current progress, `0.0 .. 1.0`.
    #[inline]
    pub fn progress(&self) -> Option<f32> {
        let target = self.target?;
        Some(self.value() as f32 / target as f32)
    }

    /// Calculate the elapsed time since creation of the progress bar.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Estimate the duration until completion.
    pub fn eta(&self) -> Option<Duration> {
        // wen eta?!
        let left = 1. / self.progress()?;
        Some(self.elapsed().mul_f32(left))
    }

    /// Calculate the mean iterations per second since creation of the progress bar.
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
    pub fn tick_sync(&self) {
        if self.add_sync(1) == self.next_print() {
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

    /// Calculate next print
    fn update_next_print(&self) {
        // Give the loop some time to warm up.
        if self.update_ctr() < 10 {
            self.next_print.fetch_add(1, ATOMIC_ORD);
            return;
        }

        let freq = (self.iters_per_sec() / self.cfg.max_fps) as usize;
        let freq = freq.max(1);

        self.next_print.fetch_add(freq as usize, ATOMIC_ORD);
    }

    #[rustfmt::skip]
    fn process_size_hint(&mut self, hint: (usize, Option<usize>)) {
        // If an explicit target is set, disregard size hints.
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
        let mut bar = ProgressBar::smart();
        bar.process_size_hint(self.size_hint());
        ProgressBarIter { bar, inner: self }
    }

    fn with_pb(self, mut bar: ProgressBar) -> ProgressBarIter<Self> {
        bar.process_size_hint(self.size_hint());
        ProgressBarIter { bar, inner: self }
    }
}

impl<I: Iterator + Sized> ProgressBarIterExt for I {}

// ========================================================================== //
// [ParIter integration]                                                      //
// ========================================================================== //

// #[cfg(feature = "rayon")]
// pub mod par_iter {
//     use rayon::iter::ParallelIterator;
//
//     pub struct ProgressBarParIter<I: Par> {
//         bar: ProgressBar,
//         inner: I,
//     }
// }
//
// #[cfg(feature = "rayon")]
// pub use par_iter::*;

// ========================================================================== //
// [Stream integration]                                                       //
// ========================================================================== //

#[cfg(feature = "streams")]
pub mod streams {
    use super::*;
    use futures_core::{
        core_reexport::pin::Pin,
        task::{Context, Poll},
        Stream,
    };

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
            let mut bar = ProgressBar::smart();
            bar.process_size_hint(self.size_hint());
            ProgressBarStream { bar, inner: self }
        }

        fn with_pb(self, mut bar: ProgressBar) -> ProgressBarStream<Self> {
            bar.process_size_hint(self.size_hint());
            ProgressBarStream { bar, inner: self }
        }
    }

    impl<S: Stream + Unpin + Sized> ProgressBarStreamExt for S {}
}

#[cfg(feature = "streams")]
pub use streams::*;
