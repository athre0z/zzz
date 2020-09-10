use futures_core::core_reexport::pin::Pin;
use futures_core::task::{Context, Poll};
use futures_core::Stream;
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fmt, mem};

pub struct ProgressBarConfig {
    width: u32,
    desc: Cow<'static, str>,
    theme: *const dyn ProgressBarTheme,
}

unsafe impl Sync for ProgressBarConfig {}

static DEFAULT_CFG: ProgressBarConfig = ProgressBarConfig {
    width: 60,
    desc: Cow::Borrowed("Progress"),
    theme: &DefaultProgressBarTheme,
};

pub trait ProgressBarTheme: Sync {
    fn render(&self, pb: &ProgressBar);
}

struct DefaultProgressBarTheme;

fn bar(progress: f32, length: u32) {
    let rescaled = (progress * length as f32 * 8.0) as usize;
    let (i, r) = (rescaled / 8, rescaled % 8);

    for _ in 0..i {
        print!("█");
    }

    let chr = '▏' as u32 - r as u32;
    let chr = unsafe { std::char::from_u32_unchecked(chr) };
    print!("{}", chr);
}

impl ProgressBarTheme for DefaultProgressBarTheme {
    fn render(&self, pb: &ProgressBar) {
        let progress = pb.progress();
        let cfg = unsafe { &*pb.cfg };
        print!("{:>6.2}% ", progress * 100.0);
        bar(progress, cfg.width);
        print!(" {}/{}\r", pb.value(), pb.target.unwrap());
    }
}

pub struct ProgressBar {
    cfg: *const ProgressBarConfig,
    value: AtomicUsize,
    target: Option<usize>,
}

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
            unsafe { (*(*self.bar.cfg).theme).render(&self.bar) };
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
                unsafe { (*(*this.bar.cfg).theme).render(&this.bar) };
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
