use core::convert::TryFrom;
use core::fmt::Write;

use embedded_hal::serial;
use nb::block;
use num_enum::{IntoPrimitive, TryFromPrimitive, TryFromPrimitiveError};

trait Never {}

impl Never for ! {}
impl Never for core::fmt::Error {}

fn ignore_error<R: core::default::Default, E: Never>(r: core::result::Result<R, E>) -> R {
    r.unwrap_or_default()
}

pub struct SerialWriter<'a, SW: serial::Write<u8, Error = !>> {
    w: &'a mut SW,
}

impl<'a, SW: serial::Write<u8, Error = !>> SerialWriter<'a, SW> {
    pub fn new(w: &'a mut SW) -> Self {
        SerialWriter { w }
    }

    pub fn send_char(w: &'a mut SW, c: char) {
        ignore_error(SerialWriter::new(w).write_char(c));
    }
}

impl<'a, SW: serial::Write<u8, Error = !>> core::fmt::Write for SerialWriter<'a, SW> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            ignore_error(block!(self.w.write(b)));
        }

        Ok(())
    }
}

pub struct MessageWriter<'a, SW: serial::Write<u8, Error = !>> {
    header: char,
    writer: &'a mut SW,
}

impl<'a, SW: serial::Write<u8, Error = !>> MessageWriter<'a, SW> {
    fn new(header: char, writer: &'a mut SW) -> Self {
        MessageWriter { header, writer }
    }
}

impl<'a, SW: serial::Write<u8, Error = !>> core::fmt::Write for MessageWriter<'a, SW> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let mut writer = SerialWriter::new(self.writer);

        ignore_error(writer.write_char(self.header));

        let len = s.bytes().len() as u32;
        ignore_error(write!(&mut writer, "{:08x}", len));

        for b in s.bytes() {
            ignore_error(write!(&mut writer, "{:x}", b));
        }

        Ok(())
    }
}

pub fn message<SW: serial::Write<u8, Error = !>>(
    writer: &mut SW,
    header: char,
    args: core::fmt::Arguments,
) {
    ignore_error(MessageWriter::new(header, writer).write_fmt(args));
}

#[macro_export]
macro_rules! error {
    ($writer:expr, $($arg:tt)*) => (
        io::message($writer, 'E', format_args!($($arg)*))
    )
}

#[macro_export]
macro_rules! info {
    ($writer:expr, $($arg:tt)*) => (
        io::message($writer, 'I', format_args!($($arg)*))
    )
}

#[derive(Debug)]
pub enum ReadError {
    NotUTF8(core::str::Utf8Error),
    NotHex(core::num::ParseIntError),
}

pub fn read_u8<'a, SR: serial::Read<u8, Error = !>>(reader: &'a mut SR) -> Result<u8, ReadError> {
    let mut buffer = [0; 2];

    for i in buffer.iter_mut() {
        *i = ignore_error(block!(reader.read()));
    }

    let s = core::str::from_utf8(&buffer).map_err(ReadError::NotUTF8)?;

    u8::from_str_radix(s, 16).map_err(ReadError::NotHex)
}

pub fn read_u32<'a, SR: serial::Read<u8, Error = !>>(reader: &'a mut SR) -> Result<u32, ReadError> {
    let mut buffer = [0; 8];

    for i in buffer.iter_mut() {
        *i = ignore_error(block!(reader.read()));
    }

    let s = core::str::from_utf8(&buffer).map_err(ReadError::NotUTF8)?;

    u32::from_str_radix(s, 16).map_err(ReadError::NotHex)
}

#[derive(IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Header {
    ReportStatus = ('S' as u8),
    SubmitProgram = ('P' as u8),
    SubmitQuery = ('Q' as u8),
}

pub fn read_header<R: serial::Read<u8, Error = !>>(
    reader: &mut R,
) -> Result<Header, TryFromPrimitiveError<Header>> {
    Header::try_from(ignore_error(block!(reader.read())))
}
