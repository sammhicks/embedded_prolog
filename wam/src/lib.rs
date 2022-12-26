#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

use core::fmt;

use machine::{Address, ReferenceOrValue};
use sha2::Digest;

#[cfg(feature = "logging")]
pub use log;

mod device;
mod device_with_program;
mod device_with_query;
mod logging;
mod machine;

pub use device::Device;
pub use machine::{system_call_handler, system_calls, SystemCalls};

#[derive(Debug)]
pub enum Never {}

struct Hex<const N: usize>([u8; N]);

impl<const N: usize> fmt::Display for Hex<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02X}"))
    }
}

pub trait SerialReadWrite {
    type Error;

    fn read_one(&mut self) -> Result<u8, Self::Error> {
        let mut buffer = 0;
        self.read_exact(core::slice::from_mut(&mut buffer))?;
        Ok(buffer)
    }

    fn read_until_zero<T, F: FnOnce(&mut [u8]) -> T>(&mut self, f: F) -> Result<T, Self::Error>;
    fn read_exact(&mut self, buffer: &mut [u8]) -> Result<(), Self::Error>;

    fn write_all(&mut self, buffer: &[u8]) -> Result<(), Self::Error>;
    fn flush(&mut self) -> Result<(), Self::Error>;
}

pub enum IoError<S: SerialReadWrite> {
    Serial(S::Error),
    CborEncoding(minicbor::encode::Error<S::Error>),
    CborDecoding(minicbor::decode::Error),
    CobsDecoding,
    Format,
}

impl<S: SerialReadWrite> fmt::Debug for IoError<S>
where
    S::Error: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Serial(inner) => f.debug_tuple("Serial").field(inner).finish(),
            IoError::CborEncoding(inner) => f.debug_tuple("Encoding").field(inner).finish(),
            IoError::CborDecoding(inner) => f.debug_tuple("Decoding").field(inner).finish(),
            IoError::CobsDecoding => write!(f, "CobsDecoding"),
            IoError::Format => write!(f, "Format"),
        }
    }
}

impl<S: SerialReadWrite> fmt::Display for IoError<S>
where
    S::Error: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Serial(inner) => inner.fmt(f),
            IoError::CborEncoding(inner) => inner.fmt(f),
            IoError::CborDecoding(inner) => inner.fmt(f),
            IoError::CobsDecoding => "Failed to decode COBS".fmt(f),
            IoError::Format => "Formatting Error".fmt(f),
        }
    }
}

const fn zero_array<const N: usize>() -> [u8; N] {
    [0; N]
}

pub struct SerialConnection<S: SerialReadWrite> {
    connection: S,
    write_buffer: [u8; 512],
    write_buffer_usage: usize,
}

impl<S: SerialReadWrite> SerialConnection<S> {
    pub fn new(connection: S) -> Self {
        Self {
            connection,
            write_buffer: zero_array(),
            write_buffer_usage: 0,
        }
    }

    fn read_exact<'b>(&mut self, buffer: &'b mut [u8]) -> Result<&'b mut [u8], IoError<S>> {
        self.connection
            .read_exact(buffer)
            .map_err(IoError::Serial)?;

        Ok(buffer)
    }

    fn decode<T>(&mut self) -> Result<T, IoError<S>>
    where
        for<'b> T: minicbor::Decode<'b, ()>,
    {
        self.connection
            .read_until_zero(|bytes| {
                let length = cobs::decode_in_place(bytes).map_err(|()| IoError::CobsDecoding)?;
                minicbor::decode(&bytes[..length]).map_err(IoError::CborDecoding)
            })
            .map_err(IoError::Serial)?
    }

    fn encode<T: minicbor::Encode<()>>(&mut self, value: T) -> Result<(), IoError<S>> {
        self.ensure_not_full().map_err(IoError::Serial)?;
        self.write_buffer_usage += 1;

        let mut writer = CborWriter {
            connection: self,
            state: cobs::EncoderState::default(),
            total_bytes_sent: 0,
        };

        minicbor::encode(value, &mut writer).map_err(IoError::CborEncoding)?;

        writer.finalize().map_err(IoError::Serial)?;

        self.do_flush().map_err(IoError::Serial)
    }

    fn do_flush(&mut self) -> Result<(), S::Error> {
        self.connection
            .write_all(&self.write_buffer[..self.write_buffer_usage])?;
        self.write_buffer_usage = 0;
        self.connection.flush()
    }

    fn ensure_not_full(&mut self) -> Result<(), S::Error> {
        if self.write_buffer_usage == self.write_buffer.len() {
            self.do_flush()
        } else {
            Ok(())
        }
    }
}

struct CborWriter<'a, S: SerialReadWrite> {
    connection: &'a mut SerialConnection<S>,
    state: cobs::EncoderState,
    total_bytes_sent: usize,
}

impl<'a, S: SerialReadWrite> CborWriter<'a, S> {
    fn finalize(mut self) -> Result<(), S::Error> {
        self.ensure_not_full()?;

        let (index, byte) = self.state.clone().finalize();
        self.set_byte(index, byte);
        self.push_byte(0);

        Ok(())
    }

    fn ensure_not_full(&mut self) -> Result<(), S::Error> {
        let current_usage = self.connection.write_buffer_usage;
        self.connection.ensure_not_full()?;
        self.total_bytes_sent += current_usage - self.connection.write_buffer_usage;

        Ok(())
    }

    fn push_byte(&mut self, byte: u8) {
        self.connection.write_buffer[self.connection.write_buffer_usage] = byte;
        self.connection.write_buffer_usage += 1;
    }

    fn set_byte(&mut self, index: usize, byte: u8) {
        self.connection.write_buffer[index - self.total_bytes_sent] = byte;
    }

    fn handle_cobs(&mut self, result: cobs::PushResult) -> Result<(), S::Error> {
        self.ensure_not_full()?;

        match result {
            cobs::PushResult::AddSingle(byte) => self.push_byte(byte),
            cobs::PushResult::ModifyFromStartAndSkip((index, byte)) => {
                self.set_byte(index, byte);
                self.connection.write_buffer_usage += 1;
            }
            cobs::PushResult::ModifyFromStartAndPushAndSkip((index, byte, new_byte)) => {
                self.set_byte(index, byte);
                self.push_byte(new_byte);
                self.ensure_not_full()?;
                self.connection.write_buffer_usage += 1;
            }
        }

        Ok(())
    }
}

impl<'a, S: SerialReadWrite> minicbor::encode::Write for CborWriter<'a, S> {
    type Error = S::Error;

    fn write_all(&mut self, buffer: &[u8]) -> Result<(), Self::Error> {
        for &byte in buffer {
            let result = self.state.push(byte);
            self.handle_cobs(result)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
enum LoadCodeError {
    NoSpaceForCode {
        code_length: usize,
        memory_capacity: usize,
    },
    HashMismatch {
        hash: Hash,
        calculated_hash: Hash,
    },
}

impl fmt::Display for LoadCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadCodeError::NoSpaceForCode { code_length, memory_capacity } => write!(f, "No space for code: code_length = {code_length}, memory_capacity = {memory_capacity}"),
            LoadCodeError::HashMismatch { hash, calculated_hash } => write!(f, "Hashes do not match: received = {hash:X}, calculated = {calculated_hash:X}"),
        }
    }
}

struct LoadedCode<'code, 'rest> {
    code_section: machine::Instructions<'code>,
    rest_of_memory: &'rest mut [u32],
}

fn code_section_buffer(code_section: &mut [[u8; 4]]) -> &mut [u8] {
    let core::ops::Range { start, end } = code_section.as_mut_ptr_range();
    let start = start as *mut u8;
    let end = end as *mut u8;

    // Safety: this is a valid slice, and u8 has an alignment of 1
    unsafe {
        core::slice::from_raw_parts_mut(
            start,
            usize::try_from(end.offset_from(start)).unwrap_unchecked(),
        )
    }
}

fn load_code<
    'code,
    'rest,
    'memory: 'code + 'rest,
    'memory_ref: 'code + 'rest,
    S: SerialReadWrite,
>(
    memory: &'memory mut [u32],
    serial_connection: &mut SerialConnection<S>,
    CodeSubmission { code_length, hash }: CodeSubmission,
) -> Result<Result<LoadedCode<'code, 'rest>, LoadCodeError>, IoError<S>> {
    let memory_capacity = memory.len();

    log_debug!("Code length:     {}", code_length);
    log_debug!("Memory capacity: {}", memory_capacity);

    if code_length >= memory_capacity {
        return Ok(Err(LoadCodeError::NoSpaceForCode {
            code_length,
            memory_capacity,
        }));
    }

    let (code_section, rest_of_memory) = memory.split_at_mut(code_length);

    let code_section = unsafe { core::mem::transmute::<_, &mut [[u8; 4]]>(code_section) };

    let calculated_hash =
        sha2::Sha256::digest(serial_connection.read_exact(code_section_buffer(code_section))?);

    for &code in code_section.iter() {
        log_trace!("Word: {}", Hex(code));
    }

    log_debug!("Received Hash:   {:X}", hash);
    log_debug!("Calculated Hash: {:X}", calculated_hash);

    if hash != calculated_hash {
        return Ok(Err(LoadCodeError::HashMismatch {
            hash,
            calculated_hash,
        }));
    }

    Ok(Ok(LoadedCode {
        code_section: machine::Instructions::new(code_section),
        rest_of_memory,
    }))
}

pub type Hash = sha2::digest::Output<sha2::Sha256>;

fn decode_hash<'b, Ctx>(
    d: &mut minicbor::Decoder<'b>,
    _ctx: &mut Ctx,
) -> Result<Hash, minicbor::decode::Error> {
    d.bytes().and_then(|bytes| {
        <[u8; 32]>::try_from(bytes)
            .map(Hash::from)
            .map_err(|_| minicbor::decode::Error::message("Bad Hash Length"))
    })
}

#[derive(Debug, minicbor::Decode)]
pub struct CodeSubmission {
    #[n(0)]
    code_length: usize,
    #[n(1)]
    #[cbor(decode_with = "decode_hash")]
    hash: Hash,
}

#[derive(Debug, minicbor::Decode)]
pub enum Command {
    #[n(0)]
    ReportStatus,
    #[n(1)]
    GetSystemCalls,
    #[n(2)]
    SubmitProgram {
        #[n(0)]
        code_submission: CodeSubmission,
    },
    #[n(3)]
    SubmitQuery {
        #[n(0)]
        code_submission: CodeSubmission,
    },
    #[n(4)]
    LookupMemory {
        #[n(0)]
        address: machine::Address,
    },
    #[n(5)]
    NextSolution,
}

struct ErrorWriter<'a, W: minicbor::encode::Write> {
    encoder: &'a mut minicbor::Encoder<W>,
    state: Result<(), minicbor::encode::Error<W::Error>>,
}

impl<'a, W: minicbor::encode::Write> core::fmt::Write for ErrorWriter<'a, W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.state.is_ok() {
            self.encoder.str(s).map_err(|err| {
                self.state = Err(err);

                core::fmt::Error
            })?;

            Ok(())
        } else {
            Err(core::fmt::Error)
        }
    }
}

pub struct ErrorResponse<E: fmt::Display>(E);

impl<E: fmt::Display, C> minicbor::Encode<C> for ErrorResponse<E> {
    fn encode<W: minicbor::encode::Write>(
        &self,
        e: &mut minicbor::Encoder<W>,
        _ctx: &mut C,
    ) -> Result<(), minicbor::encode::Error<W::Error>> {
        use core::fmt::Write;

        log_error!("{}", self.0);

        e.array(2)?.u32(0)?.array(1)?.begin_str()?;

        let mut writer = ErrorWriter {
            encoder: e,
            state: Ok(()),
        };

        write!(&mut writer, "{}", self.0)
            .map_err(|core::fmt::Error| minicbor::encode::Error::message("Formatting Error"))?;
        writer.state?;

        e.end()?.ok()
    }
}

// #[derive(Debug, minicbor::Encode)]
// pub enum CommandResponse<'a> {
//     #[n(1)]
//     Success,
//     #[n(2)]
//     WaitingForProgram,
//     #[n(3)]
//     WaitingForQuery,
//     #[n(4)]
//     NoSolution,
//     #[n(5)]
//     Solution(#[n(0)] &'a machine::Solution<'a>),
// }

#[derive(Debug, minicbor::Encode)]
pub enum ReportStatusResponse {
    #[n(1)]
    WaitingForProgram,
    #[n(2)]
    WaitingForQuery,
}

impl fmt::Display for ReportStatusResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReportStatusResponse::WaitingForQuery => "Waiting for Query".fmt(f),
            ReportStatusResponse::WaitingForProgram => "Waiting for Program".fmt(f),
        }
    }
}

#[derive(Debug, minicbor::Encode)]
pub enum GetSystemCallsResponse<Calls> {
    #[n(3)]
    SystemCalls(#[n(0)] Calls),
}

#[derive(Debug, minicbor::Encode)]
pub enum SubmitProgramResponse {
    #[n(4)]
    Success,
}

#[derive(Debug, minicbor::Encode)]
pub enum SubmitQueryResponse<'a> {
    #[n(5)]
    NoSolution,
    #[n(6)]
    Solution(#[n(0)] &'a machine::Solution<'a>),
}

#[derive(minicbor::Encode)]
enum LookupMemoryResponse<'a> {
    #[n(7)]
    MemoryValue {
        #[n(0)]
        address: Address,
        #[n(1)]
        value: ReferenceOrValue<'a>,
    },
}
