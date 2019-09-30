#![feature(never_type)]

mod serial;

use serial::{SerialRead, SerialWrite};

fn main() {
    let mut memory = Vec::new();
    memory.resize(2048, 0);
    wam::run_wam(
        SerialRead::new(b"S\nP000000020000000100000002\n"),
        SerialWrite::new(),
        memory.as_mut_slice(),
    );
}
