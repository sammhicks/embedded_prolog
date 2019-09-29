#![feature(never_type)]

mod serial;

use serial::{SerialRead, SerialWrite};

static mut MEMORY: [u32; 2048] = [0; 2048];

struct Hal {}

impl wam::Hal for Hal {
    type SerialRead = SerialRead;
    type SerialWrite = SerialWrite;

    fn get_serial_reader(&self) -> SerialRead {
        SerialRead::new(b"SP000000020000000100000002")
    }

    fn get_serial_writer(&self) -> SerialWrite {
        SerialWrite::new()
    }

    fn get_ram(&self) -> &'static mut [u32] {
        unsafe { &mut MEMORY }
    }
}

fn main() {
    wam::run_wam(Hal {});
}
