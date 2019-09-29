use embedded_hal::serial::{Read, Write};

pub trait Hal {
    type SerialRead: Read<u8, Error = !>;
    type SerialWrite: Write<u8, Error = !>;

    fn get_serial_reader(&self) -> Self::SerialRead;
    fn get_serial_writer(&self) -> Self::SerialWrite;

    fn get_ram(&self) -> &'static mut [u32];
}
