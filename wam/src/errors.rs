use super::{IoError, SerialConnection, SerialWrite};

struct SerialFormatTarget<W>(W);

impl<'a, W: SerialWrite<u8>> core::fmt::Write for SerialFormatTarget<&'a mut SerialConnection<W>> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            self.0.write_byte(b).map_err(|_err| core::fmt::Error)?;
        }

        Ok(())
    }
}

pub struct ErrorWriter<'a, W: SerialWrite<u8>>(Option<&'a mut SerialConnection<W>>);

impl<'a, W: SerialWrite<u8>> ErrorWriter<'a, W> {
    pub fn new(writer: &'a mut SerialConnection<W>) -> Self {
        match writer.write_char('E') {
            Ok(()) => Self(Some(writer)),
            Err(IoError) => Self(None),
        }
    }
}

impl<'a, W: SerialWrite<u8>> core::fmt::Write for ErrorWriter<'a, W> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        if let Some(writer) = self.0.take() {
            let mut writer = SerialFormatTarget(writer);
            for b in s.bytes() {
                write!(&mut writer, "{:02X}", b)?;
            }
            self.0 = Some(writer.0);
            Ok(())
        } else {
            Err(core::fmt::Error)
        }
    }
}

impl<'a, W: SerialWrite<u8>> core::ops::Drop for ErrorWriter<'a, W> {
    fn drop(&mut self) {
        if let Some(writer) = self.0.take() {
            writer.write_char('S').and_then(|()| writer.flush()).ok();
        }
    }
}

#[macro_export]
macro_rules! error {
    ($writer:expr, $($arg:tt)*) => {
        {
            $crate::log_error!($($arg)*);
            core::fmt::write(&mut $crate::errors::ErrorWriter::new($writer), format_args!($($arg)*))
            .map_err(|_| $crate::IoError)
        }
    }
}
