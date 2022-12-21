use super::{SerialConnection, SerialWrite, WriteError};

struct SerialFormatTarget<W>(W);

impl<'a, 'b, W: SerialWrite<u8>> core::fmt::Write
    for SerialFormatTarget<&'b mut ErrorWriter<'a, W>>
{
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            self.0.writer.write_byte(b).map_err(|err| {
                self.0.state = Err(err);

                core::fmt::Error
            })?;
        }

        Ok(())
    }
}

struct ErrorWriter<'a, W: SerialWrite<u8>> {
    writer: &'a mut SerialConnection<W>,
    state: Result<(), super::WriteError<W>>,
}

impl<'a, W: SerialWrite<u8>> ErrorWriter<'a, W> {
    fn new(writer: &'a mut SerialConnection<W>) -> Self {
        let state = writer.write_char('E');

        Self { writer, state }
    }
}

impl<'a, W: SerialWrite<u8>> core::fmt::Write for ErrorWriter<'a, W> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let mut writer = SerialFormatTarget(self);

        s.bytes().try_for_each(|b| write!(&mut writer, "{:02X}", b))
    }
}

pub(crate) fn write_error<W: SerialWrite<u8>>(
    writer: &mut SerialConnection<W>,
    args: core::fmt::Arguments<'_>,
) -> Result<(), WriteError<W>> {
    let mut writer = ErrorWriter::new(writer);

    core::fmt::write(&mut writer, args).map_err(|core::fmt::Error| match writer.state {
        Ok(()) => WriteError::Format,
        Err(err) => err,
    })?;

    writer.writer.write_char('S')
}

#[macro_export]
macro_rules! error {
    ($writer:expr, $($arg:tt)*) => {
        {
            $crate::log_error!($($arg)*);
            $crate::errors::write_error($writer, format_args!($($arg)*))
        }
    }
}
