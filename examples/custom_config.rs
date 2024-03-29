use std::thread::sleep;
use std::time::Duration;
use zzz::*;

static MY_CONFIG: Config = Config {
    max_fps: 10.0,
    ..Config::const_default()
};

static MY_CONFIG_2: Config = Config {
    min_bar_width: 80,
    ..Config::const_default()
};

fn main() {
    // Replace global default config.
    set_global_config(&MY_CONFIG);
    for _ in (0..1000).into_iter().progress() {
        sleep(Duration::from_millis(5));
    }

    // Replace config only for this instance.
    let pb = ProgressBar::smart().config(&MY_CONFIG_2);
    for _ in (0..1000).into_iter().with_progress(pb) {
        sleep(Duration::from_millis(5));
    }
}
