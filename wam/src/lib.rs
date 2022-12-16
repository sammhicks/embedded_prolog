#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

pub use embedded_hal::serial::{Read as SerialRead, Write as SerialWrite};
pub use nb;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use sha2::Digest;

mod device;
mod device_with_program;
mod device_with_query;
mod errors;
mod hex;
mod logging;
mod machine;
mod serializable;

pub use device::Device;
pub use machine::{system_call_handler, system_calls, SystemCalls};

#[derive(Debug)]
pub enum Never {}

const SUCCESS: char = 'S';

#[derive(Debug)]
pub struct IoError;

impl core::fmt::Display for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("WAM IO Error")
    }
}

#[derive(Debug)]
pub enum ProcessInputError {
    Unexpected(u8),
    IoError,
}

impl From<IoError> for ProcessInputError {
    fn from(_: IoError) -> Self {
        Self::IoError
    }
}

pub struct SerialConnection<T>(pub T);

impl<R: SerialRead<u8>> SerialConnection<R> {
    fn read_ascii_char(&mut self) -> Result<u8, IoError> {
        loop {
            match self.0.read() {
                Ok(b) => {
                    if !b.is_ascii_whitespace() {
                        return Ok(b);
                    }
                }
                Err(nb::Error::Other(_)) => return Err(IoError),
                Err(nb::Error::WouldBlock) => (),
            }
        }
    }

    fn read_hex_u4(&mut self) -> Result<u8, ProcessInputError> {
        let b = self.read_ascii_char()?;
        let c = b as char;
        match c.to_digit(16) {
            Some(digit) => Ok(digit as u8),
            None => Err(ProcessInputError::Unexpected(b)),
        }
    }

    fn read_be_u8_hex(&mut self) -> Result<u8, ProcessInputError> {
        let a = self.read_hex_u4()?;
        let b = self.read_hex_u4()?;
        Ok((a << 4) + b)
    }

    fn read_be_serializable_hex<S: serializable::Serializable>(
        &mut self,
    ) -> Result<S, ProcessInputError> {
        let mut buffer = S::Bytes::default();
        for b in buffer.as_mut() {
            *b = self.read_be_u8_hex()?;
        }
        Ok(S::from_be_bytes(buffer))
    }
}

impl<W: SerialWrite<u8>> SerialConnection<W> {
    fn flush(&mut self) -> Result<(), IoError> {
        nb::block!(self.0.flush()).map_err(|_| IoError)
    }

    fn write_byte(&mut self, b: u8) -> Result<(), IoError> {
        nb::block!(self.0.write(b)).map_err(|_| IoError)
    }

    fn write_char(&mut self, c: char) -> Result<(), IoError> {
        let mut buffer = [0; 4];
        for b in c.encode_utf8(&mut buffer).bytes() {
            self.write_byte(b)?;
        }

        Ok(())
    }

    fn write_single_char(&mut self, c: char) -> Result<(), IoError> {
        self.write_char(c)?;
        self.flush()?;

        Ok(())
    }

    fn write_be_u4_hex(&mut self, b: u8) -> Result<(), IoError> {
        self.write_char(char::from_digit(b.into(), 16).unwrap())
    }

    fn write_be_u8_hex(&mut self, b: u8) -> Result<(), IoError> {
        self.write_be_u4_hex((b >> 4) & 0xF)?;
        self.write_be_u4_hex(b & 0xF)?;
        Ok(())
    }

    fn write_be_serializable_hex<S: serializable::Serializable>(
        &mut self,
        value: S,
    ) -> Result<(), IoError> {
        for &b in value.into_be_bytes().as_ref() {
            self.write_be_u8_hex(b)?;
        }
        Ok(())
    }

    fn write_value(
        &mut self,
        address: machine::Address,
        value: machine::ReferenceOrValue,
        data: impl Iterator<Item = u8>,
        subterms: impl Iterator<Item = Option<machine::Address>>,
    ) -> Result<(), IoError> {
        match value {
            machine::ReferenceOrValue::Reference(reference) => {
                self.write_char('R')?;
                self.write_be_serializable_hex(Some(reference))?;
            }
            machine::ReferenceOrValue::Value(machine::Value::Structure(f, n)) => {
                self.write_char('S')?;
                self.write_be_serializable_hex(f)?;
                self.write_be_serializable_hex(n)?;
            }
            machine::ReferenceOrValue::Value(machine::Value::List) => {
                self.write_char('L')?;
            }
            machine::ReferenceOrValue::Value(machine::Value::Constant(c)) => {
                self.write_char('C')?;
                self.write_be_serializable_hex(c)?;
            }
            machine::ReferenceOrValue::Value(machine::Value::Integer { sign, bytes_count }) => {
                self.write_char('I')?;
                self.write_char(match sign {
                    machine::IntegerSign::Positive => '+',
                    machine::IntegerSign::Negative => '-',
                })?;
                self.write_be_serializable_hex(bytes_count)?;
            }
        }

        for byte in data {
            self.write_be_serializable_hex(byte)?;
        }

        for subterm in subterms {
            self.write_be_serializable_hex(subterm)?;
        }

        self.write_be_serializable_hex(Some(address))?;

        self.flush()
    }

    fn write_system_calls<Calls: SystemCalls>(
        &mut self,
        system_calls: &Calls,
    ) -> Result<(), IoError> {
        self.write_be_serializable_hex(system_calls.count())?;

        system_calls.for_each_call(|name, arity| {
            self.write_be_serializable_hex(name.len() as u8)?;
            for b in name.bytes() {
                self.write_be_serializable_hex(b)?;
            }
            self.write_be_serializable_hex(arity)?;

            Ok(())
        })?;

        self.flush()
    }
}

struct LoadedCode<'code, 'rest> {
    code_section: machine::Instructions<'code>,
    rest_of_memory: &'rest mut [u32],
}

fn load_code<
    'code,
    'rest,
    'memory: 'code + 'rest,
    'memory_ref: 'code + 'rest,
    S: SerialRead<u8> + SerialWrite<u8>,
    SC: core::borrow::BorrowMut<SerialConnection<S>>,
>(
    memory: &'memory mut [u32],
    serial_connection: &mut SC,
) -> Result<Option<LoadedCode<'code, 'rest>>, ProcessInputError> {
    let mut serial_connection = serial_connection.borrow_mut();

    let code_length = serial_connection.read_be_serializable_hex::<u32>()? as usize;
    let memory_capacity = memory.len();

    log_debug!("Code length:     {}", code_length);
    log_debug!("Memory capacity: {}", memory_capacity);

    if code_length >= memory_capacity {
        error!(
            &mut serial_connection,
            "Code length ({}) is more than memory capacity ({})", code_length, memory_capacity
        )?;
        return Ok(None);
    }

    let (code_section, rest_of_memory) = memory.split_at_mut(code_length);

    let mut hasher = sha2::Sha256::new();

    for memory_word in code_section.iter_mut() {
        let mut bytes = [0; 4];
        for b in bytes.as_mut() {
            *b = serial_connection.read_be_u8_hex()?;
        }
        hasher.update(bytes);
        let word = u32::from_be_bytes(bytes);
        log_trace!("Word: {:08X}", word);
        *memory_word = word;
    }

    let mut received_hash = [0; 32];

    for b in &mut received_hash {
        *b = serial_connection.read_be_u8_hex()?;
    }

    let calculated_hash = hasher.finalize();

    log_debug!("Received Hash:   {:X}", hex::Hex(&received_hash[..]));
    log_debug!("Calculated Hash: {:X}", calculated_hash);

    if received_hash
        .iter()
        .zip(calculated_hash.iter())
        .any(|(&a, &b)| a != b)
    {
        error!(&mut serial_connection, "Hashes don't match")?;
        return Ok(None);
    }

    Ok(Some(LoadedCode {
        code_section: machine::Instructions::new(code_section),
        rest_of_memory,
    }))
}

#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum CommandHeader {
    ReportStatus = b'S',
    SubmitProgram = b'P',
    SubmitQuery = b'Q',
    LookupMemory = b'M',
    NextSolution = b'C',
}

impl CommandHeader {
    fn parse(value: u8) -> Result<Self, ProcessInputError> {
        Self::try_from(value).map_err(|_err| ProcessInputError::Unexpected(value))
    }
}
