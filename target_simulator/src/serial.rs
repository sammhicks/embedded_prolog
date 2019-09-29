use embedded_hal::serial::{Read, Write};

pub struct SerialRead {
    src: &'static [u8],
}

impl SerialRead {
    pub fn new(src: &'static [u8]) -> Self {
        SerialRead { src: src }
    }
}

impl Read<u8> for SerialRead {
    type Error = !;

    fn read(&mut self) -> nb::Result<u8, !> {
        match self.src.split_first() {
            Some((&head, tail)) => {
                self.src = tail;
                Ok(head)
            }
            None => Err(nb::Error::WouldBlock),
        }
    }
}

pub struct SerialWrite {}

impl SerialWrite {
    pub fn new() -> Self {
        SerialWrite {}
    }
}

impl Write<u8> for SerialWrite {
    type Error = !;

    fn write(&mut self, word: u8) -> nb::Result<(), !> {
        println!("{}", word as char);
        Ok(())
    }

    fn flush(&mut self) -> nb::Result<(), !> {
        Ok(())
    }
}
