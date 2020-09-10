use futures_core::core_reexport::pin::Pin;
use futures_core::task::{Context, Poll};
use futures_core::Stream;
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};
// use std::fmt;

// ========================================================================== //
// [General configuration]                                                    //
// ========================================================================== //

pub struct ProgressBarConfig {
    width: u32,
    desc: Cow<'static, str>,
    theme: &'static dyn ProgressBarTheme,
}

static DEFAULT_CFG: ProgressBarConfig = ProgressBarConfig {
    width: 60,
    desc: Cow::Borrowed("Progress"),
    theme: &DefaultProgressBarTheme,
};

// ========================================================================== //
// [Customizable printing]                                                    //
// ========================================================================== //

pub trait ProgressBarTheme: Sync {
    fn render(&self, pb: &ProgressBar);
}

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

impl ProgressBarTheme for DefaultProgressBarTheme {
    fn render(&self, pb: &ProgressBar) {
        let progress = pb.progress();
        print!("{:>6.2}% ", progress * 100.0);
        let bar_len = bar(progress, pb.cfg.width);
        for _ in 0..(pb.cfg.width as i64 - bar_len as i64).max(0) {
            print!(" ");
        }
        print!(" {}/{}\r", pb.value(), pb.target.unwrap());
    }
}

// ========================================================================== //
// [Main progress bar struct]                                                 //
// ========================================================================== //

pub struct ProgressBar {
    cfg: &'static ProgressBarConfig,
    value: AtomicUsize,
    target: Option<usize>,
}

/// Constructors.
impl ProgressBar {
    pub fn new(target: Option<usize>) -> Self {
        Self {
            cfg: &DEFAULT_CFG,
            value: AtomicUsize::new(0),
            target,
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

/// Accessors.
impl ProgressBar {
    pub fn add(&self, n: usize) -> usize {
        self.value.fetch_add(n, Ordering::Relaxed)
    }

    pub fn inc(&self) -> usize {
        self.add(1)
    }

    pub fn value(&self) -> usize {
        self.value.load(Ordering::Relaxed)
    }

    pub fn progress(&self) -> f32 {
        if let Some(target) = self.target {
            self.value() as f32 / target as f32
        } else {
            0.5
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
            self.bar.inc();
            self.bar.cfg.theme.render(&self.bar);
            x
        })
    }
}

pub trait ProgressBarIterExt: Iterator + Sized {
    fn pb(self) -> ProgressBarIter<Self> {
        let (_, hi) = self.size_hint();
        let bar = ProgressBar::new(hi);
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
                this.bar.inc();
                this.bar.cfg.theme.render(&this.bar);
                x
            }
            x => x,
        }
    }
}

pub trait ProgressBarStreamExt: Stream + Unpin + Sized {
    fn pb(self) -> ProgressBarStream<Self> {
        let (_, hi) = self.size_hint();
        let bar = ProgressBar::new(hi);
        ProgressBarStream { bar, inner: self }
    }
}

impl<S: Stream + Unpin + Sized> ProgressBarStreamExt for S {}
