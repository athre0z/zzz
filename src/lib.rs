#![doc = include_str!("../README.md")]

use std::{
    fmt::{self, Write as _},
    io::{self, stderr, Write as _},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
    sync::RwLock,
    time::{Duration, Instant},
};

// ============================================================================================== //
// [Prelude module]                                                                               //
// ============================================================================================== //

/// Mass-import for the main progress bar type as well as the convenience extension traits.
pub mod prelude {
    #[cfg(feature = "streams")]
    pub use crate::ProgressBarStreamExt;
    pub use crate::{ProgressBar, ProgressBarIterExt};
}

// ============================================================================================== //
// [General configuration]                                                                        //
// ============================================================================================== //

#[doc(hidden)]
#[deprecated(note = "renamed to just `Config`")]
pub type ProgressBarConfig = Config;

/// Configuration for a progress bar.
///
/// This is a separate struct from the actual progress bar in order to allow a
/// configuration to be reused in different progress bar instances.
#[derive(Clone)]
pub struct Config {
    /// Width of the progress bar.
    pub width: Option<u32>,
    /// Minimum width to bother with drawing the bar for.
    pub min_bar_width: u32,
    /// Theme to use when drawing.
    pub theme: &'static dyn Theme,
    /// Maximum redraw rate rate (draws per second).
    pub max_fps: f32,
    /// Called to determine whether the progress bar should be drawn or not.
    ///
    /// The default value always returns `true`.
    pub should_draw: &'static (dyn Fn() -> bool + Sync),
}

static DEFAULT_CFG: Config = Config::const_default();

impl Config {
    /// `const` variant of [`Config::default`].
    pub const fn const_default() -> Self {
        Config {
            width: None,
            min_bar_width: 5,
            theme: &DefaultTheme,
            max_fps: 60.0,
            should_draw: &|| true,
        }
    }
}

impl Default for Config {
    #[inline]
    fn default() -> Self {
        Config::const_default()
    }
}

/// Selects the currently active global configuration.
///
/// This stores a `*const ProgressBarConfig`. We use `AtomicUsize` instead of
/// the seemingly more idiomatic `AtomicPtr` here because the latter requires a
/// **mutable** pointer, which would in turn force us to take the config as
/// mutable reference to not run into UB. There is no const variant of `AtomicPtr`.
/// Using `AtomicUsize` seemed like the lesser evil here.
static GLOBAL_CFG: AtomicUsize = AtomicUsize::new(0);

/// Gets the currently active global configuration.
pub fn global_config() -> &'static Config {
    match GLOBAL_CFG.load(Relaxed) {
        0 => &DEFAULT_CFG,
        ptr => unsafe { &*(ptr as *const Config) }
    }
}

/// Set a new global default configuration.
///
/// This configuration is used when no explicit per instance configuration
/// is specified via [`ProgressBar::config`].
pub fn set_global_config(new_cfg: &'static Config) {
    GLOBAL_CFG.store(new_cfg as *const _ as _, Relaxed);
}

// ============================================================================================== //
// [Utils]                                                                                        //
// ============================================================================================== //

/// Pads and aligns a value to the length of a cache line.
///
/// Adapted from crossbeam:
/// https://docs.rs/crossbeam/0.7.3/crossbeam/utils/struct.CachePadded.html
#[cfg_attr(target_arch = "x86_64", repr(align(128)))]
#[cfg_attr(not(target_arch = "x86_64"), repr(align(64)))]
struct CachePadded<T>(T);

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

// ============================================================================================== //
// [Error type]                                                                                   //
// ============================================================================================== //

/// Errors that can ocurr while drawing the progress bar.
#[derive(Debug)]
pub enum RenderError {
    Io(io::Error),
    Fmt(fmt::Error),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::Fmt(e) => e.fmt(f),
            RenderError::Io(e) => e.fmt(f),
        }
    }
}

// TODO: this should probably forward everything
impl std::error::Error for RenderError {}

impl From<io::Error> for RenderError {
    fn from(e: io::Error) -> Self {
        RenderError::Io(e)
    }
}

impl From<fmt::Error> for RenderError {
    fn from(e: fmt::Error) -> Self {
        RenderError::Fmt(e)
    }
}

// ============================================================================================== //
// [Customizable printing]                                                                        //
// ============================================================================================== //

/// Trait defining how the progress bar is rendered.
pub trait Theme: Sync {
    fn render(&self, pb: &ProgressBar) -> Result<(), RenderError>;
}

#[derive(Debug, Default)]
struct DefaultTheme;

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

/// Determines the dimensions of stderr.
#[cfg(feature = "auto-width")]
fn stderr_dimensions() -> (usize, usize) {
    // term_size doesn't support stderr on Windows, so just use stdout and 
    // hope for the best. We should probably replace term_size anyway in the
    // long run since it's unmaintained, but this works for the moment.
    #[cfg(target_os = "windows")]
    return term_size::dimensions_stdout().unwrap_or((80, 30));

    #[cfg(not(target_os = "windows"))]
    return term_size::dimensions_stderr().unwrap_or((80, 30));
}

/// Determines the dimensions of stderr.
#[cfg(not(feature = "auto-width"))]
fn stderr_dimensions() -> (usize, usize) {
    (80, 30)
}

impl Theme for DefaultTheme {
    fn render(&self, pb: &ProgressBar) -> Result<(), RenderError> {
        let mut o = stderr();
        let cfg = pb.active_config();

        // Draw left side.
        let left = {
            let mut buf = String::new();

            // If a description is set, print it.
            if let Some(desc) = pb.message() {
                write!(buf, "{} ", desc)?;
            }

            if let Some(progress) = pb.progress() {
                write!(buf, "{:>6.2}% ", progress * 100.0)?;
            }

            buf
        };

        // Draw right side.
        let right = {
            let mut buf = String::new();

            // Print "done/total" part
            buf.write_char(' ')?;
            pb.unit.write_total(&mut buf, pb.value())?;
            buf.write_char('/')?;
            match pb.target {
                Some(target) => pb.unit.write_total(&mut buf, target)?,
                None => buf.write_char('?')?,
            }

            // Print ETA / time elapsed.
            if let Some(eta) = pb.eta() {
                write!(buf, " [{}]", human_time(eta))?;
            } else {
                write!(buf, " [{}]", human_time(pb.elapsed()))?;
            }

            // Print iteration rate.
            buf.write_str(" (")?;
            pb.unit.write_rate(&mut buf, pb.iters_per_sec())?;
            buf.write_char(')')?;

            buf
        };

        let max_width = cfg
            .width
            .unwrap_or_else(|| stderr_dimensions().0 as u32);

        let bar_width = max_width
            .saturating_sub(left.len() as u32)
            .saturating_sub(right.len() as u32);

        write!(o, "{}", left)?;

        if bar_width > cfg.min_bar_width {
            // Draw a progress bar for known-length bars.
            if let Some(progress) = pb.progress() {
                write!(o, "{}", bar(progress, bar_width))?;
            }
            // And a spinner for unknown-length bars.
            else {
                let duration = Duration::from_secs(3);
                let pos = pb.timer_progress(duration);
                // Sub 1 from width because many terms render emojis with double width.
                write!(o, "{}", spinner(pos, bar_width - 1))?;
            }
        }

        write!(o, "{}\r", right)?;

        o.flush().map_err(Into::into)
    }
}

// ============================================================================================== //
// [Units]                                                                                        //
// ============================================================================================== //

/// Determines the unit used for printing iteration speed.
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Unit {
    Iterations,
    Bytes,
}

fn human_iter_unit(x: usize) -> (&'static str, f32) {
    if x > 10usize.pow(9) {
        ("B", 1e9)
    } else if x > 10usize.pow(6) {
        ("M", 1e6)
    } else if x > 10usize.pow(3) {
        ("K", 1e3)
    } else {
        ("", 1e0)
    }
}

fn bytes_unit(x: usize) -> (&'static str, f32) {
    if x > 1024usize.pow(4) {
        ("TiB", 1024_f32.powi(4))
    } else if x > 1024usize.pow(3) {
        ("GiB", 1024_f32.powi(3))
    } else if x > 1024usize.pow(2) {
        ("MiB", 1024_f32.powi(2))
    } else if x > 1024usize.pow(1) {
        ("KiB", 1024_f32.powi(1))
    } else {
        ("b", 1024_f32.powi(0))
    }
}

impl Unit {
    /// Formats an absolute amount, e.g. "1200 iterations".
    fn write_total<W: fmt::Write>(self, mut out: W, amount: usize) -> fmt::Result {
        match self {
            Unit::Iterations => {
                let (unit, div) = human_iter_unit(amount);
                write!(out, "{:.2}{}", (amount as f32) / div, unit)
            }
            Unit::Bytes => {
                let (unit, div) = bytes_unit(amount);
                write!(out, "{:.2}{}", (amount as f32) / div, unit)
            }
        }
    }

    /// Formats a rate of change, e.g. "1200 it/sec".
    fn write_rate<W: fmt::Write>(self, mut out: W, rate: f32) -> fmt::Result {
        match self {
            Unit::Iterations => {
                if rate >= 1.0 {
                    let (unit, div) = human_iter_unit(rate as usize);
                    write!(out, "{:.2}{} it/s", rate / div, unit)
                } else {
                    write!(out, "{:.0} s/it", 1.0 / rate)
                }
            }
            Unit::Bytes => {
                let (unit, div) = bytes_unit(rate as usize);
                write!(out, "{:.2}{}/s", rate / div, unit)
            }
        }
    }
}

// ============================================================================================== //
// [Main progress bar struct]                                                                     //
// ============================================================================================== //

/// Progress bar to be rendered on the terminal.
///
/// # Example
///
/// ```rust
/// use zzz::prelude::*;
///
/// let mut bar = ProgressBar::with_target(123);
/// for _ in 0..123 {
///     bar.add(1);
/// }
/// ```
pub struct ProgressBar {
    /// Configuration to use.
    cfg: Option<&'static Config>,
    /// The expected, possibly approximate target of the progress bar.
    target: Option<usize>,
    /// Whether the target was specified explicitly.
    explicit_target: bool,
    /// Whether the target was specified explicitly.
    pub(crate) unit: Unit,
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
        self.redraw();
        eprintln!();
    }
}

/// Constructors.
impl ProgressBar {
    fn new(target: Option<usize>, explicit_target: bool) -> Self {
        Self {
            cfg: None,
            target,
            explicit_target,
            start: Instant::now(),
            unit: Unit::Iterations,
            value: CachePadded(0.into()),
            update_ctr: CachePadded(0.into()),
            next_print: CachePadded(1.into()),
            message: RwLock::new(None),
        }
    }

    /// Creates a smart progress bar, attempting to infer the target from size hints.
    pub fn smart() -> Self {
        Self::new(None, false)
    }

    /// Creates a spinner, a progress bar with indeterminate target value.
    pub fn spinner() -> Self {
        Self::new(None, true)
    }

    /// Creates a progress bar with an explicit target value.
    pub fn with_target(target: usize) -> Self {
        Self::new(Some(target), true)
    }
}

/// Builder-style methods.
impl ProgressBar {
    /// Replace the config of the progress bar.
    ///
    /// Takes precedence over a global config set via [`set_global_config`].
    pub fn config(mut self, cfg: &'static Config) -> Self {
        self.cfg = Some(cfg);
        self
    }

    /// Force display as a spinner even if size hints are present.
    pub fn force_spinner(mut self) -> Self {
        self.explicit_target = true;
        self.target = None;
        self
    }

    /// Set the unit to be used when formatting values.
    pub fn unit(mut self, unit: Unit) -> Self {
        self.unit = unit;
        self
    }
}

/// Value manipulation and access.
impl ProgressBar {
    /// Returns the currently active configuration.
    #[inline]
    pub fn active_config(&self) -> &'static Config {
        self.cfg.unwrap_or_else(global_config)
    }

    #[rustfmt::skip]
    pub fn process_size_hint(&mut self, hint: (usize, Option<usize>)) {
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

    /// Set the progress bar value to a new, absolute value.
    ///
    /// This doesn't automatically redraw the progress-bar.
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
        self.update_ctr.fetch_add(1, Relaxed);
        self.value.store(n, Relaxed);
    }

    /// Add `n` to the value of the progress bar.
    ///
    /// See `add_sync` for a thread-safe version.
    #[inline]
    pub fn add(&mut self, n: usize) -> usize {
        *self.value.get_mut() += n;
        let prev = *self.update_ctr.get_mut();
        *self.update_ctr.get_mut() += 1;
        self.maybe_redraw(prev);
        prev
    }

    /// Synchronized version fo `add`.
    #[inline]
    pub fn add_sync(&self, n: usize) -> usize {
        self.value.fetch_add(n, Relaxed);
        let prev = self.update_ctr.fetch_add(1, Relaxed);
        self.maybe_redraw(prev);
        prev
    }

    /// How often has the value been changed since creation?
    #[inline]
    pub fn update_ctr(&self) -> usize {
        self.update_ctr.load(Relaxed)
    }

    /// Get the current value of the progress bar.
    #[inline]
    pub fn value(&self) -> usize {
        self.value.load(Relaxed)
    }

    /// Get the current task description text.
    pub fn message(&self) -> Option<String> {
        self.message.read().unwrap().clone()
    }

    /// Set the current task description text.
    pub fn set_message(&mut self, text: Option<impl Into<String>>) {
        *self.message.get_mut().unwrap() = text.map(Into::into);
    }

    /// Synchronized version for `set_message`.
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
        let elapsed = self.elapsed();
        let estimated_total = elapsed.mul_f32(left);
        Some(estimated_total.saturating_sub(elapsed))
    }

    /// Calculate the mean iterations per second since creation of the progress bar.
    pub fn iters_per_sec(&self) -> f32 {
        let elapsed_sec = self.elapsed().as_secs_f32();
        self.value() as f32 / elapsed_sec
    }

    /// Calculate the mean progress bar updates per second since creation of the progress bar.
    pub fn updates_per_sec(&self) -> f32 {
        let elapsed_sec = self.elapsed().as_secs_f32();
        self.update_ctr() as f32 / elapsed_sec
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

    /// Forces a redraw of the progress bar.
    pub fn redraw(&self) {
        self.active_config().theme.render(self).unwrap();
        self.update_next_print();
    }
}

/// Internals.
impl ProgressBar {
    #[inline]
    fn next_print(&self) -> usize {
        self.next_print.load(Relaxed)
    }

    /// Calculate next print
    fn update_next_print(&self) {
        // Give the loop some time to warm up.
        if self.update_ctr() < 10 {
            self.next_print.fetch_add(1, Relaxed);
            return;
        }

        let freq = (self.updates_per_sec() / self.active_config().max_fps) as usize;
        let freq = freq.max(1);

        self.next_print.fetch_add(freq as usize, Relaxed);
    }

    #[inline]
    fn maybe_redraw(&self, prev: usize) {
        #[cold]
        fn cold_redraw(this: &ProgressBar) {
            if (this.active_config().should_draw)() {
                this.redraw();
            }
        }

        if prev == self.next_print() {
            cold_redraw(self);
        }
    }
}

// ============================================================================================== //
// [Iterator integration]                                                                         //
// ============================================================================================== //

/// Iterator / stream wrapper that automatically updates a progress bar during iteration.
pub struct ProgressBarIter<Inner> {
    bar: ProgressBar,
    inner: Inner,
}

impl<Inner> ProgressBarIter<Inner> {
    pub fn into_inner(self) -> Inner {
        self.inner
    }
}

impl<Inner: Iterator> Iterator for ProgressBarIter<Inner> {
    type Item = Inner::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.inner.next()?;
        self.bar.add(1);
        Some(next)
    }
}

/// Extension trait implemented for all iterators, adding methods for
/// conveniently adding a progress bar to an existing iterator.
///
/// # Example
///
/// ```rust
/// # fn main() {
/// use zzz::prelude::*;
///
/// for _ in (0..123).progress() {
///     // ...
/// }
/// # }
/// ```
pub trait ProgressBarIterExt: Iterator + Sized {
    fn progress(self) -> ProgressBarIter<Self> {
        let mut bar = ProgressBar::smart();
        bar.process_size_hint(self.size_hint());
        ProgressBarIter { bar, inner: self }
    }

    fn with_progress(self, mut bar: ProgressBar) -> ProgressBarIter<Self> {
        bar.process_size_hint(self.size_hint());
        ProgressBarIter { bar, inner: self }
    }
}

impl<Inner: Iterator + Sized> ProgressBarIterExt for Inner {}

// ============================================================================================== //
// [Stream integration]                                                                           //
// ============================================================================================== //

#[cfg(feature = "streams")]
pub mod streams {
    use super::*;
    use core::pin::Pin;
    use futures_core::{
        task::{Context, Poll},
        Stream,
    };

    impl<Inner: Stream> Stream for ProgressBarIter<Inner> {
        type Item = Inner::Item;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            // SAFETY: This is no different than what pin_project would do, except without
            //         requiring the dependency on the lib.
            let (inner, bar) = unsafe {
                let this = self.get_unchecked_mut();
                (Pin::new_unchecked(&mut this.inner), &mut this.bar)
            };

            match inner.poll_next(cx) {
                x @ Poll::Ready(Some(_)) => {
                    bar.add(1);
                    x
                }
                x => x,
            }
        }
    }

    /// Extension trait implemented for all streams, adding methods for conveniently adding a
    /// progress bar to an existing iterator.
    pub trait ProgressBarStreamExt: Stream + Sized {
        fn progress(self) -> ProgressBarIter<Self> {
            let mut bar = ProgressBar::smart();
            bar.process_size_hint(self.size_hint());
            ProgressBarIter { bar, inner: self }
        }

        fn with_progress(self, mut bar: ProgressBar) -> ProgressBarIter<Self> {
            bar.process_size_hint(self.size_hint());
            ProgressBarIter { bar, inner: self }
        }
    }

    impl<Inner: Stream + Sized> ProgressBarStreamExt for Inner {}
}

#[cfg(feature = "streams")]
pub use streams::*;

// ============================================================================================== //
// [Tests]                                                                                        //
// ============================================================================================== //

#[cfg(doctest)]
mod tests {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }

    // Ensure the examples in README.md work.
    external_doc_test!(include_str!("../README.md"));
}

// ============================================================================================== //
